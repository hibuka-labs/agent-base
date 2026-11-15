use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use futures_core::Stream;
use futures_util::StreamExt;
use serde_json::Value;
use tracing::Span;

use crate::engine::runtime::event_bus::EventBus;
use crate::llm::{ReasoningConfig, StreamChunk, UsageInfo};
use crate::types::{AgentResult, ChatMessage, RuntimeEvent, SessionId};

/// Bound on the wait for the LLM to *start* responding (time to response
/// headers). Generous enough for slow providers cold-starting a long prompt
/// (typical TTFT is well under 10 s), tight enough to surface a dead
/// network within a minute instead of hanging forever.
const TTFB_TIMEOUT: Duration = Duration::from_secs(60);

/// Build a `ChatRequest` from individual parameters.
///
/// Every picture is shrunk to the wire size and judged by
/// [`crate::image_ops::apply_image_budget`] on the way out: over either
/// budget line the WHOLE request is refused with a size inventory — the one
/// place a picture may leave the wire, and never silently. The pass runs on
/// this wire copy only; the session's history keeps the originals (each turn
/// re-shrinks its own copy).
fn build_chat_request(
    messages: &[ChatMessage],
    tool_definitions: &[Value],
    reasoning: Option<&ReasoningConfig>,
    response_format: Option<&crate::types::ResponseFormat>,
    thinking_disabled: bool,
    model_override: Option<&str>,
) -> AgentResult<llm_trait::ChatRequest> {
    let mut messages = messages.to_vec();
    crate::image_ops::apply_image_budget(&mut messages)
        .map_err(|e| crate::types::AgentError::llm(e.to_string()))?;
    let mut request = llm_trait::ChatRequest::new(messages);
    if !tool_definitions.is_empty() {
        request = request.with_tools(tool_definitions.to_vec());
    }
    // Only include reasoning if not disabled
    // This is a workaround for LLM providers that ignore budget_tokens limit
    if let Some(r) = reasoning {
        if !thinking_disabled {
            request = request.with_reasoning(r.clone());
        } else {
            tracing::info!(
                "thinking disabled for rest of run due to too many reasoning-only responses"
            );
        }
    }
    request.response_format = response_format.cloned();

    // A conversation appends: it never rewrites its own prefix. Say so, so a
    // provider that caches prefixes can stop re-billing the settled part of
    // the prompt at the input rate. Marking through the newest message is the
    // standard shape — the breakpoint trails the conversation and each call
    // pays only for what is new. Providers without caching ignore the claim.
    let cache_through = request.messages.len().saturating_sub(1);
    request = request.with_cache_through(cache_through);

    // Tool-call batching is a permission, and the engine can honour it: the
    // orchestrator already runs an approved batch concurrently. Claiming it
    // costs nothing when the model emits one call per turn, and the alternative
    // is a provider default we did not choose. It is not a promise the model
    // will batch — phimint session 20261115_2fbff82e issued 1.41 calls per turn
    // with nothing forbidding it.
    //
    // Note what this cannot do: on an Anthropic-shaped request there is nowhere
    // to put the claim, because that protocol has no such field and the model
    // batches on its own. This is the OpenAI-side lever only — see
    // `notes/tool-face-design.md` §2.4 for the probe that established that.
    request = request.with_parallel_tool_calls(true);

    // Apply model override if specified
    if let Some(model) = model_override {
        request = request.with_model(model);
    }

    Ok(request)
}

/// Convert a `ChatStream` (new) to the old stream type consumed by `process_stream`.
fn chat_stream_to_old(
    chat_stream: llm_trait::ChatStream,
) -> Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>> {
    let inner = chat_stream.into_inner();
    Box::pin(inner.map(|item| item.map_err(Into::into)))
}

/// Run-liveness accounting for [`LlmEngine::begin_run`]: counts one run
/// against the engine and lands staged provider swaps when the last run
/// ends. Held for the duration of one `run` / `run_turn`.
pub(crate) struct RunGuard<'a> {
    engine: &'a LlmEngine,
}

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.engine.active_runs.fetch_sub(1, AtomicOrdering::AcqRel);
        self.engine.flush_pending_if_idle();
    }
}

/// A live view of the engine's provider slot ([`LlmEngine::provider_handle`]).
///
/// Resolves the current provider at USE time, so a holder survives
/// [`LlmEngine::set_provider`] / [`LlmEngine::set_provider_when_idle`] swaps
/// (a `/model` switch). A provider cloned out of the engine at build time is
/// a boot-time snapshot: the guard's completion judge kept judging on the old
/// model after a swap and treated that model's own identity as the session's
/// ground truth (session 20261107_27dc727a).
#[derive(Clone)]
pub struct ProviderHandle(Arc<RwLock<Arc<dyn llm_trait::LlmProvider>>>);

impl ProviderHandle {
    /// A standalone handle over `initial` — for holders that get their handle
    /// directly instead of from an engine (and for tests).
    pub fn new(initial: Arc<dyn llm_trait::LlmProvider>) -> Self {
        Self(Arc::new(RwLock::new(initial)))
    }

    /// The provider currently in the slot.
    pub fn current(&self) -> Arc<dyn llm_trait::LlmProvider> {
        self.0.read().unwrap().clone()
    }
}

pub struct LlmEngine {
    provider: Arc<RwLock<Arc<dyn llm_trait::LlmProvider>>>,
    event_bus: EventBus,
    /// Optional model override for all requests.
    ///
    /// When set, this overrides the model in ChatRequest.
    /// Used by sub-agents to specify their model tier (e.g., "lite").
    model_override: RwLock<Option<String>>,
    /// Runs currently executing against this engine (see [`RunGuard`]).
    active_runs: AtomicUsize,
    /// A staged provider swap waiting for the engine to go idle
    /// ([`Self::set_provider_when_idle`]).
    pending_provider: RwLock<Option<Arc<dyn llm_trait::LlmProvider>>>,
}

impl LlmEngine {
    pub(crate) fn new(provider: Arc<dyn llm_trait::LlmProvider>, event_bus: EventBus) -> Self {
        Self {
            provider: Arc::new(RwLock::new(provider)),
            event_bus,
            model_override: RwLock::new(None),
            active_runs: AtomicUsize::new(0),
            pending_provider: RwLock::new(None),
        }
    }

    /// Get a clone of the current LLM provider.
    pub fn get_provider(&self) -> Arc<dyn llm_trait::LlmProvider> {
        self.provider.read().unwrap().clone()
    }

    /// A handle that always resolves to this engine's CURRENT provider —
    /// hand it to long-lived holders (e.g. the react-loop guard) instead of a
    /// provider clone so model swaps reach them too. See [`ProviderHandle`].
    pub fn provider_handle(&self) -> ProviderHandle {
        ProviderHandle(Arc::clone(&self.provider))
    }

    /// Replace the LLM provider at runtime (e.g., model switch).
    ///
    /// Immediate: the swap lands even while a run is streaming. Callers that
    /// must not split one conversation across two models use
    /// [`Self::set_provider_when_idle`] instead. Every [`ProviderHandle`]
    /// handed out by [`Self::provider_handle`] follows the swap.
    pub fn set_provider(&self, provider: Arc<dyn llm_trait::LlmProvider>) {
        *self.provider.write().unwrap() = provider;
    }

    /// Stage a provider swap for the end of the run executing now.
    ///
    /// Hot-swapping mid-run would split one conversation across two models —
    /// the tool-continuation steps would run on the new provider with the old
    /// provider's history and identity. So the swap lands when the last active
    /// run ends; when the engine is already idle it lands at once. Returns
    /// `true` when the swap was deferred, `false` when it landed immediately.
    /// A second staged swap replaces the first (the last pick wins).
    pub fn set_provider_when_idle(&self, provider: Arc<dyn llm_trait::LlmProvider>) -> bool {
        *self.pending_provider.write().unwrap() = Some(provider);
        self.flush_pending_if_idle();
        self.pending_provider.read().unwrap().is_some()
    }

    /// Whether a run is executing against this engine right now.
    pub fn has_active_run(&self) -> bool {
        self.active_runs.load(AtomicOrdering::Acquire) > 0
    }

    /// Count one run against the engine. The returned guard decrements on
    /// drop and lands any staged swap once the count reaches zero.
    pub(crate) fn begin_run(&self) -> RunGuard<'_> {
        self.active_runs.fetch_add(1, AtomicOrdering::AcqRel);
        RunGuard { engine: self }
    }

    /// Land the staged swap if no run is in flight. Called at every point
    /// that can change the answer (a stage, a run's end) so no interleaving
    /// of the two can strand a staged swap.
    fn flush_pending_if_idle(&self) {
        if self.has_active_run() {
            return;
        }
        if let Some(provider) = self.pending_provider.write().unwrap().take() {
            self.set_provider(provider);
        }
    }

    /// Get the model override, if set.
    pub fn get_model_override(&self) -> Option<String> {
        self.model_override.read().unwrap().clone()
    }

    /// Set a model override for all requests.
    ///
    /// This is used by sub-agents to specify their model tier (e.g., "lite").
    /// The override is applied to all ChatRequests before sending to the provider.
    pub fn set_model_override(&self, model: Option<String>) {
        *self.model_override.write().unwrap() = model;
    }

    /// Replace the LLM client at runtime (e.g., model switch).
    ///
    /// Accepts an `Arc<dyn llm_trait::LlmProvider>`.
    pub fn set_client(&self, client: Arc<dyn llm_trait::LlmProvider>) {
        self.set_provider(client);
    }

    /// Await the provider's `stream()` with cancellation + TTFB timeout.
    ///
    /// reqwest's connect_timeout covers only TCP connect and read_timeout
    /// only response-body reads — the wait for response headers is covered
    /// by neither, so a silently dropped network hangs the await forever
    /// (session 20260910_e528c7a3: 1.5 h silent hang, TUI unresponsive to
    /// Ctrl+C for the same reason). Both gaps are closed here: the cancel
    /// token aborts the in-flight request, and the TTFB timeout turns a
    /// stalled connection into a regular error that the caller's retry
    /// logic can handle.
    async fn stream_cancellable(
        &self,
        cancel_token: &tokio_util::sync::CancellationToken,
        request: llm_trait::ChatRequest,
    ) -> AgentResult<Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>>> {
        // Bind the provider handle outside the select!: select! drops
        // temporaries at the end of the statement, which would free the
        // Arc clone mid-await.
        let provider = self.get_provider();
        tokio::select! {
            _ = cancel_token.cancelled() => {
                tracing::info!("LLM request cancelled while awaiting response");
                Err(crate::types::AgentError::Cancelled)
            }
            result = tokio::time::timeout(TTFB_TIMEOUT, provider.stream(request)) => {
                match result {
                    Ok(inner) => inner.map(chat_stream_to_old).map_err(Into::into),
                    Err(_) => {
                        tracing::error!(
                            timeout_secs = TTFB_TIMEOUT.as_secs(),
                            "LLM request timed out waiting for response (network stalled?)"
                        );
                        Err(crate::types::AgentError::Llm(format!(
                            "no response within {}s (network stalled or provider unreachable)",
                            TTFB_TIMEOUT.as_secs()
                        )))
                    }
                }
            }
        }
    }

    pub async fn chat_stream(
        &self,
        messages: &[ChatMessage],
        tool_definitions: &[Value],
        reasoning: Option<&ReasoningConfig>,
        response_format: Option<&crate::types::ResponseFormat>,
        thinking_disabled: bool,
        cancel_token: &tokio_util::sync::CancellationToken,
    ) -> AgentResult<Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>>> {
        tracing::info!(
            msg_count = messages.len(),
            tool_count = tool_definitions.len(),
            thinking_disabled,
            "LLM chat_stream: sending request to API"
        );
        let model_override = self.get_model_override();
        let request = build_chat_request(
            messages,
            tool_definitions,
            reasoning,
            response_format,
            thinking_disabled,
            model_override.as_deref(),
        )?;
        let result = self.stream_cancellable(cancel_token, request).await;
        match &result {
            Ok(_) => tracing::info!("LLM chat_stream: API response received"),
            Err(e) => tracing::error!(error = %e, "LLM chat_stream: API request failed"),
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn run_llm_turn_with_retry(
        &self,
        session_id: &SessionId,
        messages: &[ChatMessage],
        tool_definitions: &[Value],
        reasoning: Option<&ReasoningConfig>,
        response_format: Option<&crate::types::ResponseFormat>,
        retry: crate::types::RetryConfig,
        thinking_disabled: bool,
        cancel_token: &tokio_util::sync::CancellationToken,
    ) -> AgentResult<Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>>> {
        let model_override = self.get_model_override();
        let request = build_chat_request(
            messages,
            tool_definitions,
            reasoning,
            response_format,
            thinking_disabled,
            model_override.as_deref(),
        )?;
        let mut attempt = 1;
        let mut delay_ms = retry.initial_backoff_ms;
        loop {
            match self.stream_cancellable(cancel_token, request.clone()).await {
                Ok(chat_stream) => return Ok(chat_stream),
                Err(e) => {
                    let agent_err: crate::types::AgentError = e;
                    // Cancellation is a user decision, not a transient
                    // failure — never retry past it.
                    if agent_err.is_cancelled() {
                        return Err(agent_err);
                    }
                    if attempt > retry.max_retries {
                        tracing::error!(session_id = session_id.id, attempts = attempt, error = %agent_err, "LLM stream retry exhausted");
                        return Err(agent_err);
                    }
                    tracing::warn!(session_id = session_id.id, attempt, error = %agent_err, "LLM stream failed, retrying...");
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms = (delay_ms as f64 * retry.backoff_multiplier) as u64;
                    if delay_ms > retry.max_backoff_ms {
                        delay_ms = retry.max_backoff_ms;
                    }
                    attempt += 1;
                }
            }
        }
    }

    pub(crate) async fn process_stream<F>(
        &self,
        session_id: &SessionId,
        mut stream: Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>>,
        span: Span,
        event_rx: &mut tokio::sync::broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<std::sync::Mutex<F>>,
        cancel_token: &tokio_util::sync::CancellationToken,
    ) -> AgentResult<LlmTurnResult>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send,
    {
        let _enter = span.enter();
        let start = std::time::Instant::now();
        let mut first_token = true;
        let mut aggregator = StreamAggregator::new();
        tracing::info!(session_id = session_id.id, "LLM process_stream start");

        // Sweep self-echo UserEvent copies (dual-written by ctx.emit() during
        // middleware — compression Progress/Started/Completed) before the main
        // loop starts: they were already rendered to on_event and must not
        // double-render here. Run-local echoes are recognized by
        // `is_self_echo_user_event`; bridged child events (agent_id: Some) are
        // bus-only and get delivered like any other event.
        loop {
            match event_rx.try_recv() {
                Ok(event) if crate::engine::runtime::is_self_echo_user_event(&event) => continue,
                Ok(event) => {
                    // Non-echo (e.g. Checkpoint) — push back by processing
                    // and let the main loop handle the next ones normally.
                    if let Ok(mut cb) = on_event.lock() {
                        cb(event)?;
                    }
                    continue;
                }
                Err(tokio::sync::broadcast::error::TryRecvError::Lagged(_)) => continue,
                Err(_) => break,
            }
        }

        loop {
            tokio::select! {
                recv_result = event_rx.recv() => {
                    match recv_result {
                        Ok(event) => {
                            // Same dual-write rule as the sweep above: a
                            // run-local UserEvent copy that slipped in mid-stream
                            // must not re-render; everything else flows.
                            if crate::engine::runtime::is_self_echo_user_event(&event) {
                                continue;
                            }
                            if let Ok(mut cb) = on_event.lock() {
                                cb(event)?;
                            }
                        }
                        Err(tokio::sync::broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    }
                }
                _ = cancel_token.cancelled() => {
                    tracing::info!(session_id = session_id.id, "LLM stream cancelled");
                    return Err(crate::types::AgentError::Cancelled);
                }
                maybe_chunk = stream.next() => {
                    let Some(chunk) = maybe_chunk else {
                        break;
                    };

                    match chunk {
                        Ok(StreamChunk::Text(text)) => {
                            if first_token {
                                let ttft = start.elapsed();
                                aggregator.ttft_ms = ttft.as_millis() as u64;
                                tracing::info!(session_id = session_id.id, ?ttft, "LLM first token received");
                                first_token = false;
                            }
                            // Emit unconditionally: some providers interleave
                            // text between/after tool_call chunks. Suppressing
                            // on is_tool_call hides real content from the user
                            // while aggregator.full_text still feeds the model
                            // context — user sees ≠ model sees.
                            if !text.is_empty() {
                                self.event_bus.emit(RuntimeEvent::TextDelta {
                                    session_id: session_id.clone(),
                                    text: text.clone(),
                                    agent_id: None,
                                    trace_id: None,
                                });
                            }
                            aggregator.full_text.push_str(&text);
                        }
                        Ok(StreamChunk::Thought(text)) => {
                            tracing::debug!(session_id = session_id.id, len = text.len(), "llm thought chunk");
                            aggregator.reasoning_text.push_str(&text);
                            // Same as TextDelta: never suppress on is_tool_call —
                            // thought chunks after tool_call chunks are real.
                            if !text.is_empty() {
                                self.event_bus.emit(RuntimeEvent::ThoughtDelta {
                                    session_id: session_id.clone(),
                                    text,
                                    agent_id: None,
                                    trace_id: None,
                                });
                            }
                        }
                        Ok(StreamChunk::ToolCall(choice)) => {
                            if !aggregator.is_tool_call {
                                tracing::info!(session_id = session_id.id, "LLM stream: first tool_call chunk received");
                            }
                            aggregator.is_tool_call = true;
                            let Some(tool_calls) = choice
                                .get("delta")
                                .and_then(|d| d.get("tool_calls"))
                                .and_then(Value::as_array)
                            else {
                                tracing::debug!("tool_calls missing or not an array in LLM response");
                                continue;
                            };

                            for tool_call in tool_calls {
                                let idx = tool_call
                                    .get("index")
                                    .and_then(Value::as_u64)
                                    .unwrap_or(0) as usize;
                                // Liveness for the drafting window: tool-call
                                // args stream in as fragments and nothing else
                                // is observable until the whole turn's stream
                                // is drained — multi-KB calls (a spawn_agent
                                // task brief) leave every consumer blind for
                                // tens of seconds. Emit a coarse draft on each
                                // slot's first fragment, on function-name
                                // arrival, and on every 512-char args bucket —
                                // never the text itself, and never per
                                // fragment (the bus is a bounded broadcast).
                                const DRAFT_ARGS_BUCKET: usize = 512;
                                let fresh_slot = !aggregator.partials.contains_key(&idx);
                                let before_args_len =
                                    aggregator.partials.get(&idx).map(|e| e.2.len()).unwrap_or(0);
                                let entry = aggregator.partials.entry(idx).or_insert_with(|| (String::new(), String::new(), String::new()));
                                let mut named_now = false;
                                if let Some(id) = tool_call.get("id").and_then(Value::as_str)
                                    && !id.is_empty() {
                                        entry.0 = id.to_string();
                                    }
                                if let Some(func) = tool_call.get("function") {
                                    if let Some(name) = func.get("name").and_then(Value::as_str)
                                        && !name.is_empty() {
                                            named_now = entry.1.is_empty();
                                            entry.1 = name.to_string();
                                        }
                                    if let Some(args) = func.get("arguments").and_then(Value::as_str) {
                                        entry.2.push_str(args);
                                    }
                                }
                                let args_len = entry.2.len();
                                let grew_bucket =
                                    args_len / DRAFT_ARGS_BUCKET > before_args_len / DRAFT_ARGS_BUCKET;
                                if fresh_slot || named_now || grew_bucket {
                                    self.event_bus.emit(RuntimeEvent::ActionDraft {
                                        session_id: session_id.clone(),
                                        index: idx,
                                        name: entry.1.clone(),
                                        args_len,
                                        count: aggregator.partials.len(),
                                        agent_id: None,
                                        trace_id: None,
                                    });
                                }
                            }
                        }
                        Ok(StreamChunk::Usage(usage)) => {
                            // Usage arrives as partial events on some
                            // protocols (Anthropic splits prompt/output
                            // across message_start / message_delta), so fold
                            // field-by-field instead of replacing the whole
                            // struct — the same overwrite hazard already
                            // handled for finish_reason below.
                            tracing::debug!(
                                session_id = session_id.id,
                                input = ?usage.prompt_tokens,
                                output = ?usage.completion_tokens,
                                reasoning = ?usage.reasoning_tokens,
                                "LLM usage chunk received"
                            );
                            match &mut aggregator.usage {
                                Some(existing) => existing.merge(&usage),
                                None => aggregator.usage = Some(usage),
                            }
                        }
                        Ok(StreamChunk::Stop { finish_reason }) => {
                            // Only capture the first non-None finish_reason.
                            // OpenAI emits [DONE] after the final chunk, which
                            // would overwrite a "length" finish_reason with None,
                            // silently disabling the truncation guard.
                            if finish_reason.is_some() {
                                aggregator.finish_reason = finish_reason;
                            }
                        }
                        Err(e) => {
                            tracing::error!(session_id = session_id.id, error = %e, "LLM stream error");
                            return Err(e);
                        }
                        Ok(StreamChunk::ThinkingSignature(sig)) => {
                            aggregator.thinking_signature = Some(sig);
                        }
                        Ok(StreamChunk::Error(e)) => {
                            tracing::error!(session_id = session_id.id, error = %e, "LLM stream protocol error");
                            return Err(crate::types::AgentError::Llm(e));
                        }
                    }

                    {
                        let mut cb = on_event.lock().unwrap();
                        EventBus::drain_async_events(event_rx, &mut *cb)?;
                    }
                }
            }
        }

        let total_elapsed = start.elapsed();
        let _ttft_ms = aggregator.ttft_ms;

        // Reasoning is a separate channel from the final answer. We NEVER promote
        // reasoning_content into `content`: some models (e.g. qwen3.7-max) answer
        // entirely in the reasoning channel, but papering over that masks a stuck
        // agent as a completed run. Instead we classify and let the react loop route
        // it — `reasoning_only` (reasoning but no text and no tool call) is always a
        // degenerate state the loop nudges and eventually fails, never a fake answer.
        let full_text = aggregator.full_text;
        let reasoning_len = aggregator.reasoning_text.len();
        let reasoning_only = full_text.is_empty() && reasoning_len > 0 && !aggregator.is_tool_call;
        if reasoning_only {
            tracing::warn!(
                session_id = session_id.id,
                reasoning_len,
                "content empty but reasoning_content present — model produced reasoning only; \
                 not promoting it to the answer, the react loop will nudge/fail"
            );
        }

        tracing::info!(
            session_id = session_id.id,
            text_len = full_text.len(),
            reasoning_len,
            tool_calls = aggregator.partials.len(),
            elapsed_ms = total_elapsed.as_millis(),
            "LLM stream done"
        );

        let tool_calls = aggregator
            .partials
            .into_iter()
            // Drop phantom buckets: a fragment that carried an index but never
            // a function name is not a real tool call. Some OpenAI-compatible
            // providers (e.g. mimo) emit an empty `{index, name:""}` fragment;
            // kept, it lands in assistant history as a name-less tool_call and
            // the NEXT request is rejected with HTTP 400 "missing a function
            // name". The stream is fully drained here, so an empty name means
            // no name ever arrived — this is permanent, not a late fragment.
            .filter(|(_idx, (_id, name, _args))| !name.is_empty())
            .map(|(idx, (id, name, args))| {
                let mut tc = serde_json::Map::new();
                tc.insert("index".to_string(), idx.into());
                tc.insert("id".to_string(), id.into());
                tc.insert("type".to_string(), "function".into());
                let mut func = serde_json::Map::new();
                func.insert("name".to_string(), name.into());
                func.insert("arguments".to_string(), args.into());
                tc.insert("function".to_string(), func.into());
                serde_json::Value::Object(tc)
            })
            .collect::<Vec<_>>();

        Ok(LlmTurnResult {
            full_text,
            reasoning_text: aggregator.reasoning_text,
            is_tool_call: aggregator.is_tool_call,
            tool_calls,
            usage: aggregator.usage,
            finish_reason: aggregator.finish_reason,
            ttft_ms: aggregator.ttft_ms,
            llm_duration_ms: total_elapsed.as_millis() as u64,
            reasoning_only,
            thinking_signature: aggregator.thinking_signature,
        })
    }

    pub fn emit_text_delta(&self, session_id: &SessionId, text: String) {
        self.event_bus.emit(RuntimeEvent::TextDelta {
            session_id: session_id.clone(),
            text,
            agent_id: None,
            trace_id: None,
        });
    }
}

pub struct LlmTurnResult {
    pub full_text: String,
    /// Reasoning/thinking content accumulated from Thought stream chunks.
    /// Populated when the model uses thinking mode (e.g. qwen3.7-max).
    pub reasoning_text: String,
    pub is_tool_call: bool,
    pub tool_calls: Vec<Value>,
    pub usage: Option<UsageInfo>,
    /// Provider stop reason (e.g. "stop", "length", "tool_calls", "end_turn").
    /// `"length"` means the response hit the token limit — tool call arguments may be truncated.
    pub finish_reason: Option<String>,
    /// Time to first token in milliseconds (user-perceived latency).
    pub ttft_ms: u64,
    /// LLM stream duration in milliseconds (from stream start to end).
    pub llm_duration_ms: u64,
    /// True when the model emitted reasoning_content but no `content` and no tool
    /// call. This is always a degenerate state (the model "thought" but neither
    /// committed to a tool call nor produced an answer); the react loop nudges and
    /// eventually fails rather than promoting the reasoning into `full_text`.
    pub reasoning_only: bool,
    /// Thinking signature from the provider (e.g. Anthropic extended thinking).
    #[allow(dead_code)]
    pub thinking_signature: Option<String>,
}

struct StreamAggregator {
    pub full_text: String,
    /// Accumulated reasoning/thinking content (from StreamChunk::Thought).
    /// Kept separate from `full_text`; never promoted into the final answer.
    pub reasoning_text: String,
    pub is_tool_call: bool,
    pub partials: std::collections::HashMap<usize, (String, String, String)>,
    pub usage: Option<UsageInfo>,
    /// Provider stop reason captured from StreamChunk::Stop.
    pub finish_reason: Option<String>,
    /// Time to first token in milliseconds.
    pub ttft_ms: u64,
    /// Thinking signature from StreamChunk::ThinkingSignature.
    pub thinking_signature: Option<String>,
}

impl StreamAggregator {
    fn new() -> Self {
        Self {
            full_text: String::new(),
            reasoning_text: String::new(),
            is_tool_call: false,
            partials: std::collections::HashMap::new(),
            usage: None,
            finish_reason: None,
            ttft_ms: 0,
            thinking_signature: None,
        }
    }
}

#[cfg(test)]
mod image_tests;
#[cfg(test)]
mod tests;
