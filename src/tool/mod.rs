use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::engine::SessionStore;
use crate::types::{AgentError, AgentResult, RuntimeEvent, SessionId, UserEvent};

pub mod auto_continue;
pub mod policy;
pub mod update_plan;

pub(crate) mod schema_budget;

mod activate_tool;
mod activation;
mod deferred;

pub use activate_tool::ActivateTool;
pub use activation::ToolActivation;
pub use deferred::DeferredTool;

pub use auto_continue::AutoContinueTool;
pub use update_plan::UpdatePlanTool;

pub use policy::{DenyAllToolPolicy, ToolDecision, ToolPolicy};

// Re-export pure types from agent-types
pub use agent_types::{
    ActivationContext, Content, ToolExposure, ToolMetadata, content_details, content_images,
    content_text,
};

/// Per-call execution-clock control, shared between the tool and the
/// pipeline's timeout ([`crate::engine::DefaultPipeline`]).
///
/// The timeout measures **active** execution time: stretches spent waiting on
/// a human are excluded ([`ExecClock::pause`]), so a tool that asks the user a
/// question is never killed by its own execution budget while the user is
/// still reading. Every [`ToolContext`] built per call gets a fresh clock —
/// pauses never leak into a later call.
#[derive(Clone, Default)]
pub struct ExecClock {
    inner: Arc<ExecClockInner>,
}

#[derive(Default)]
struct ExecClockInner {
    state: std::sync::Mutex<ExecClockState>,
}

#[derive(Default)]
struct ExecClockState {
    /// Live pause guards. Non-zero ⇒ the clock is paused (nesting supported).
    live: usize,
    /// When the current pause began (`Some` while paused).
    started: Option<std::time::Instant>,
    /// Paused time already banked from closed pauses.
    total: std::time::Duration,
}

impl ExecClock {
    /// Hold the clock paused for as long as the returned guard lives.
    /// Nesting is fine: the clock resumes when the outermost guard drops.
    pub fn pause(&self) -> ExecPauseGuard {
        let mut state = self.lock();
        state.live += 1;
        if state.live == 1 {
            state.started = Some(std::time::Instant::now());
        }
        ExecPauseGuard {
            clock: self.clone(),
        }
    }

    /// Paused time so far, including the open pause if any. The pipeline's
    /// deadline is `start + budget + paused_total()`, so a pause pushes the
    /// timeout out by exactly the time the human spent.
    pub fn paused_total(&self) -> std::time::Duration {
        let state = self.lock();
        state.total
            + state
                .started
                .map_or(std::time::Duration::ZERO, |s| s.elapsed())
    }

    fn resume(&self) {
        let mut state = self.lock();
        state.live = state.live.saturating_sub(1);
        if state.live == 0
            && let Some(started) = state.started.take()
        {
            state.total += started.elapsed();
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ExecClockState> {
        // A panic while holding the clock must not poison it — worst case the
        // accounting is slightly off, never "the tool can never time out".
        self.inner
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

/// RAII pause handle — see [`ExecClock::pause`].
pub struct ExecPauseGuard {
    clock: ExecClock,
}

impl Drop for ExecPauseGuard {
    fn drop(&mut self) {
        self.clock.resume();
    }
}

#[derive(Clone)]
pub struct ToolContext {
    pub session_id: SessionId,
    /// Channel for sending user-space events (progress, sub-agent, structured).
    /// Tools should use `emit_user_event()` or `emit_progress()`.
    pub user_event_tx: mpsc::UnboundedSender<UserEvent>,
    pub llm_client: Option<Arc<dyn llm_trait::LlmProvider>>,
    pub session_store: Option<Arc<dyn SessionStore>>,
    /// Language preference for tool output.
    /// Defaults to `Language::En` if not set.
    pub language: crate::types::Language,
    /// Cancellation token for checking if the operation should be cancelled.
    pub cancel_token: tokio_util::sync::CancellationToken,
    /// Output budget for this call, in characters (set from the engine's
    /// `max_tool_output_chars`). Tools that can return large results (e.g.
    /// `read_file`) should self-truncate to this bound and mark the cut, so
    /// the engine's hard reject (§6.5) never fires for a paginated read.
    pub max_output_chars: Option<usize>,
    /// Execution clock for this call — see [`ExecClock`]. Pause it while the
    /// tool waits on a human ([`ToolContext::pause_execution_clock`]).
    pub exec_clock: ExecClock,
    /// Set by [`ToolContext::mark_waiting_for_human`] when the call left a
    /// human step pending (a pick, a confirmation) instead of finishing the
    /// work. The engine folds it into the run state after the call.
    pub waiting_for_human: Arc<AtomicBool>,
    /// Internal runtime event bus (framework tools emit `RuntimeEvent`s here).
    /// `pub(crate)` — engine-internal; user tools should use `emit_user_event()`.
    pub(crate) event_bus: crate::engine::EventBus,
}

impl ToolContext {
    /// Pause this call's execution clock for as long as the guard lives.
    ///
    /// Call this while waiting on a human (a confirmation popup, a login):
    /// the pipeline's tool timeout does not count the paused stretch.
    pub fn pause_execution_clock(&self) -> ExecPauseGuard {
        self.exec_clock.pause()
    }

    /// Mark that this call left a human step pending (a design-canvas screen
    /// awaiting a pick, a confirmation). The engine records it on the run: a
    /// run that ends here is *waiting for the user*, not an incomplete task,
    /// so the guard must not nudge "continue working" over the human's turn.
    pub fn mark_waiting_for_human(&self) {
        self.waiting_for_human.store(true, Ordering::Relaxed);
    }

    /// Send a user-space event (progress, sub-agent forwarding, structured data).
    ///
    /// Dual-write: the event goes to `user_event_tx` (the tool select arm
    /// renders it in real time) and to the event bus for external subscribers
    /// (bridge remote clients, persistence). The bus copy carries `agent_id:
    /// None` — the self-echo marker — so bus loopback drops it instead of
    /// re-rendering (see `crate::engine::runtime::is_self_echo_user_event`).
    pub fn emit_user_event(&self, event: UserEvent) {
        let _ = self.user_event_tx.send(event.clone());
        self.event_bus.emit(RuntimeEvent::UserEvent {
            session_id: self.session_id.clone(),
            event,
            agent_id: None,
            trace_id: None,
        });
    }

    /// Convenience: send a progress event with text.
    pub fn emit_progress(&self, text: impl Into<String>) {
        self.emit_user_event(UserEvent::Progress { text: text.into() });
    }

    /// Convenience: emit a partial result during long-running tool execution.
    /// `is_partial: true` means more output is coming; `false` means final.
    pub fn emit_partial_result(
        &self,
        tool_call_id: &str,
        content: impl Into<String>,
        is_partial: bool,
    ) {
        self.emit_user_event(UserEvent::ToolPartialResult {
            tool_call_id: tool_call_id.to_string(),
            content: content.into(),
            is_partial,
        });
    }

    /// Test-only constructor: a `ToolContext` with a disconnected event
    /// channel and no LLM/session backends. Lets downstream tests drop their
    /// bespoke `dummy_ctx()` helpers.
    pub fn for_test() -> Self {
        let (tx, _rx) = mpsc::unbounded_channel();
        ToolContext {
            session_id: SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: crate::types::Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            max_output_chars: None,
            exec_clock: ExecClock::default(),
            waiting_for_human: Arc::new(AtomicBool::new(false)),
            event_bus: crate::engine::EventBus::new(1),
        }
    }

    /// Host-side constructor for host-initiated tool execution outside the
    /// engine loop (e.g. a REPL driving a tool through the same approval path
    /// as the engine). User events are accepted but dropped — there is no
    /// engine event loop to render them.
    pub fn new(
        session_id: SessionId,
        llm_client: Option<Arc<dyn crate::llm_trait::LlmProvider>>,
        session_store: Option<Arc<dyn SessionStore>>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> Self {
        let (tx, _rx) = mpsc::unbounded_channel();
        ToolContext {
            session_id,
            user_event_tx: tx,
            llm_client,
            session_store,
            language: crate::types::Language::En,
            cancel_token,
            max_output_chars: None,
            exec_clock: ExecClock::default(),
            waiting_for_human: Arc::new(AtomicBool::new(false)),
            event_bus: crate::engine::EventBus::new(1),
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    /// Human-readable description of what this tool does and when to use it.
    fn description(&self) -> &'static str;
    /// JSON Schema for the tool's input arguments (MCP `inputSchema` shape,
    /// without the provider envelope).
    fn schema(&self) -> Value;
    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>>;

    /// Timeout for this tool in milliseconds.
    ///
    /// Returns `Some(ms)` to enforce a timeout, or `None` to use the
    /// framework's default timeout from `ToolConfig.default_tool_timeout_ms`.
    /// Tools that need more or less time should override this method.
    fn timeout_ms(&self) -> Option<u64> {
        None // Use framework default
    }

    /// Execution budget for a specific call, derived from its `args`.
    ///
    /// The pipeline watchdog consults this before arming: `Some(ms)` overrides
    /// [`Tool::timeout_ms`] for this invocation, and `Some(0)` means unbounded
    /// (no watchdog — the tool enforces its own limits, or the call is
    /// fire-and-forget). `None` falls back to [`Tool::timeout_ms`] / config.
    ///
    /// Tools whose arguments carry a per-call timeout (e.g. `timeout_ms`)
    /// MUST surface it here. Otherwise the pipeline arms its own watchdog
    /// from the tool-level default and can kill a call that the tool was
    /// explicitly promised more time for (session 20261018_7844215b: a
    /// 660s `execute_command` budget was cut at the 600s framework default).
    fn call_timeout_ms(&self, _args: &Value) -> Option<u64> {
        self.timeout_ms()
    }

    /// Machine-readable metadata for tool introspection.
    ///
    /// The default implementation derives `name` and `description` from
    /// [`Tool::name`] and [`Tool::description`], sets `origin` to `"custom"`,
    /// and leaves `requirements` empty. Tool authors are encouraged to
    /// override this to provide an accurate `origin` and `version`.
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name().to_string(),
            description: self.description().to_string(),
            origin: "custom".to_string(),
            version: "unknown".to_string(),
            requirements: vec![],
        }
    }

    /// Visibility level for this tool. Default: `Direct` (always visible).
    ///
    /// Override to return `Deferred` (conditionally visible) or `Hidden`
    /// (never visible to the model). `Deferred` tools are only included
    /// when [`Tool::should_activate`] returns `true`.
    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }

    /// Activation condition for `Deferred` tools.
    ///
    /// Called once per turn for each `Deferred` tool. Return `true` to
    /// include this tool in the model's tool list this turn. Ignored for
    /// `Direct` and `Hidden` tools.
    ///
    /// The default makes `Deferred` mean what it says: open only once the model
    /// has asked for this tool by name, via
    /// [`crate::tool::ActivateTool`]. `ctx.current_tools` carries the names the
    /// session has opened so far. Override only to auto-open on something else
    /// — a workspace shape, a config flag — and say so in the override.
    fn should_activate(&self, ctx: &ActivationContext) -> bool {
        ctx.current_tools.iter().any(|n| n == self.name())
    }
}

#[async_trait]
pub trait TypedTool: Send + Sync {
    type Args: serde::de::DeserializeOwned + schemars::JsonSchema;
    type Output: serde::Serialize;

    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    async fn call_typed(&self, args: Self::Args, ctx: &ToolContext) -> AgentResult<Self::Output>;

    fn format_output(&self, output: Self::Output) -> Content {
        // A `String` output is emitted verbatim; any other serializable type is
        // rendered as JSON. This avoids `serde_json::to_string` wrapping a plain
        // string in literal double quotes (`hello` → `"hello"`), which would leak
        // quotes into the LLM-visible tool result.
        match serde_json::to_value(&output) {
            Ok(serde_json::Value::String(s)) => Content::text(s),
            Ok(other) => Content::text(other.to_string()),
            Err(_) => Content::text(String::new()),
        }
    }

    /// Machine-readable origin of this tool (crate name, `"agent-base"`, or `"custom"`).
    fn origin(&self) -> &'static str {
        "custom"
    }

    /// Crate/package version, or `"unknown"` when built outside a crate.
    fn version(&self) -> &'static str {
        "unknown"
    }

    /// Visibility level for this tool. Default: `Direct` (always visible).
    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }

    /// Activation condition for `Deferred` tools. Ignored for `Direct`/`Hidden`.
    /// Defaults to the same opt-in rule as [`Tool::should_activate`].
    fn should_activate(&self, ctx: &ActivationContext) -> bool {
        ctx.current_tools.iter().any(|n| n == self.name())
    }
}

#[async_trait]
impl<T: TypedTool + Send + Sync + 'static> Tool for T {
    fn name(&self) -> &'static str {
        TypedTool::name(self)
    }

    fn description(&self) -> &'static str {
        TypedTool::description(self)
    }

    fn schema(&self) -> Value {
        // Generate a provider-safe JSON Schema: Draft 7 (not 2020-12), with
        // nested subschemas inlined (no `$ref`/`$defs`/`/definitions`) and no
        // root `$schema`/meta-schema key. OpenAI-compatible function-calling
        // rejects `$ref` and 2020-12's `$defs`, so the default 2020-12 output
        // would break any `Args` containing a nested enum or struct.
        let settings = schemars::generate::SchemaSettings::draft07().with(|s| {
            s.inline_subschemas = true;
            s.meta_schema = None;
        });
        let generator = schemars::SchemaGenerator::new(settings);
        let schema = generator.into_root_schema_for::<T::Args>();
        serde_json::to_value(schema).unwrap_or(Value::Null)
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name().to_string(),
            description: self.description().to_string(),
            origin: self.origin().to_string(),
            version: self.version().to_string(),
            requirements: vec![],
        }
    }

    fn exposure(&self) -> ToolExposure {
        TypedTool::exposure(self)
    }

    fn should_activate(&self, ctx: &ActivationContext) -> bool {
        TypedTool::should_activate(self, ctx)
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let typed_args: T::Args =
            serde_json::from_value(args.clone()).map_err(|e| AgentError::ToolArgsInvalid {
                name: self.name().to_string(),
                // Carry the serde detail ("missing field `task` …") — the model
                // can only fix what the message names. Same "{e} (args: …)"
                // shape as the phase1 JSON-parse failure site (tool_engine.rs)
                // and the contract asserted in recovery.rs.
                raw: format!("{e} (args: {args})"),
            })?;
        let output = self.call_typed(typed_args, ctx).await?;
        Ok(vec![self.format_output(output)])
    }
}

/// Render a single tool's definition into the OpenAI function-calling
/// envelope. Tools only provide `name`/`description`/`schema`; the envelope is
/// assembled here at the LLM boundary (Anthropic gets its own renderer, and
/// tool authors stay protocol-agnostic).
///
/// The schema passes through [`schema_budget`] on the way out. That is the
/// only place every definition is guaranteed to cross, so it is where a
/// per-tool ceiling belongs — see the module docs for why it is best-effort.
pub fn render_tool_definition(tool: &dyn Tool) -> Value {
    let mut parameters = tool.schema();
    match schema_budget::fit_schema_to_budget(&mut parameters) {
        schema_budget::FitOutcome::Untouched => {}
        schema_budget::FitOutcome::Compressed(pass) => {
            tracing::info!(
                tool = tool.name(),
                pass,
                "tool schema compressed to fit the request budget"
            );
        }
        schema_budget::FitOutcome::OverBudget => {
            // Kept on purpose: a tool the model cannot see, it routes around.
            tracing::warn!(
                tool = tool.name(),
                budget = schema_budget::MAX_TOOL_SCHEMA_BYTES,
                "tool schema exceeds the request budget after every pass; sent as-is"
            );
        }
    }
    json!({
        "type": "function",
        "function": {
            "name": tool.name(),
            "description": tool.description(),
            "parameters": parameters,
        }
    })
}

pub(crate) type ToolRef = Arc<dyn Tool>;

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolRef>,
    /// Names that must mount as [`DeferredTool`] whenever they appear.
    ///
    /// Recording the intent rather than wrapping at call time makes the
    /// registration order irrelevant: some tools are registered directly, some
    /// arrive from a factory at build time, some come from plugins.
    deferred: HashSet<String>,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.register_arc(Arc::new(tool));
    }

    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        let name = tool.name().to_string();
        let tool = if self.deferred.contains(&name) && tool.exposure() != ToolExposure::Deferred {
            Arc::new(DeferredTool::new(tool)) as ToolRef
        } else {
            tool
        };
        self.tools.insert(name, tool);
    }

    /// Remove a tool from the registry by name.
    pub fn remove(&mut self, name: &str) {
        self.tools.remove(name);
    }

    pub fn get(&self, name: &str) -> Option<ToolRef> {
        self.tools.get(name).cloned()
    }

    /// Mark `name` to mount as deferred whenever it appears, and wrap it now if
    /// it is already registered. Returns the rendered definition — what
    /// [`crate::tool::ActivateTool`] hands back on open — or `None` when the
    /// tool is not registered **yet**. The mark still holds, so a later
    /// registration is deferred too.
    pub fn defer(&mut self, name: &str) -> Option<Value> {
        self.deferred.insert(name.to_string());
        let inner = self.tools.get(name)?;
        let definition = render_tool_definition(inner.as_ref());
        if inner.exposure() != ToolExposure::Deferred {
            let wrapped = Arc::new(DeferredTool::new(inner.clone())) as ToolRef;
            self.tools.insert(name.to_string(), wrapped);
        }
        Some(definition)
    }

    pub fn definitions(&self) -> Vec<Value> {
        let mut tools: Vec<_> = self.tools.values().collect();
        tools.sort_by_key(|t| t.name());
        tools
            .into_iter()
            .map(|t| render_tool_definition(t.as_ref()))
            .collect()
    }

    /// Return tool definitions filtered by [`ToolExposure`].
    ///
    /// - `Direct` tools are always included.
    /// - `Deferred` tools are included only when `should_activate(ctx)` returns `true`.
    /// - `Hidden` tools are never included.
    pub fn definitions_filtered(&self, ctx: &ActivationContext) -> Vec<Value> {
        let mut tools: Vec<_> = self.tools.values().collect();
        tools.sort_by_key(|t| t.name());

        // Single sequential pass: collect Direct names first, then evaluate
        // Deferred tools in sorted order. Each Deferred tool sees the names of
        // all Direct + previously-activated Deferred tools in ctx.current_tools.
        // Evaluation order matters: a later Deferred tool can see an earlier one,
        // but not vice versa.
        //
        // `ctx.current_tools` arrives seeded with the session's openings (see
        // [`ToolActivation`]); the default `should_activate` reads them from
        // here, so they must be in the pool before the first Deferred tool is
        // asked.
        let direct_names: Vec<String> = tools
            .iter()
            .filter(|t| t.exposure() == ToolExposure::Direct)
            .map(|t| t.name().to_string())
            .collect();

        let mut activated_names = direct_names.clone();
        for name in &ctx.current_tools {
            if !activated_names.contains(name) {
                activated_names.push(name.clone());
            }
        }
        for t in &tools {
            if t.exposure() == ToolExposure::Deferred {
                let mut ctx_with_tools = ctx.clone();
                ctx_with_tools.current_tools = activated_names.clone();
                if t.should_activate(&ctx_with_tools) {
                    activated_names.push(t.name().to_string());
                }
            }
        }

        tools
            .into_iter()
            .filter(|t| match t.exposure() {
                ToolExposure::Direct => true,
                ToolExposure::Deferred => activated_names.contains(&t.name().to_string()),
                ToolExposure::Hidden => false,
            })
            .map(|t| render_tool_definition(t.as_ref()))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Collect metadata for every registered tool, sorted by name.
    ///
    /// This is the preferred introspection API for consumers — it returns a
    /// stable `ToolMetadata` struct per tool instead of having callers parse
    /// the LLM-facing JSON definitions.
    pub fn metadatas(&self) -> Vec<ToolMetadata> {
        let mut list: Vec<_> = self.tools.values().map(|tool| tool.metadata()).collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }
}

#[cfg(test)]
mod tests;
