use std::sync::{Arc, Mutex};

use futures_util::FutureExt;
use serde_json::Value;
use tokio::sync::{RwLock, broadcast, mpsc};

use crate::engine::pipeline::{DefaultPipeline, ToolExecutionPipeline};
use crate::engine::recovery::ToolErrorRecovery;
use crate::engine::runtime::event_bus::EventBus;
use crate::tool::{Content, ExecClock, ToolContext, ToolRegistry, content_details, content_text};
use crate::types::{AgentError, AgentResult, Language, RuntimeEvent, SessionId, UserEvent};

use super::{ExecutionContext, OrchestrateOutcome, ToolEngine, ToolExecutionResult, ToolFailure};

impl ToolEngine {
    /// Orchestrate a batch of tool calls: parse args, check approval, execute.
    ///
    /// Execution is **parallel** — approved tool calls run concurrently via
    /// `join_all`. A single bad tool call (invalid args or a tool execution
    /// failure) does NOT abort the whole batch — it is recorded in
    /// [`OrchestrateOutcome::failures`] and the remaining calls keep executing.
    /// Approval denial and cancellation still hard-abort (return `Err`) since
    /// they represent a user/interrupt decision, not a per-call failure that the
    /// model can recover from mid-batch.
    pub async fn orchestrate<F>(
        &self,
        session_id: &SessionId,
        tool_calls: &[(String, String, String)],
        ctx: &ExecutionContext,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<OrchestrateOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        // ── Phase 1: Parse args + approve (sequential) ──
        // Approval may prompt the user, so it must happen one at a time.
        let mut approved: Vec<(String, String, Value, String)> = Vec::new();
        let mut failures: Vec<ToolFailure> = Vec::new();

        {
            tracing::info!(
                session_id = session_id.id,
                tool_count = tool_calls.len(),
                "phase1: starting parse + approval loop"
            );
            for (id, name, args_str) in tool_calls {
                tracing::info!(
                    session_id = session_id.id,
                    tool = name,
                    args_len = args_str.len(),
                    "phase1: parsing tool args"
                );
                let args: Value = match serde_json::from_str(args_str) {
                    Ok(args) => {
                        tracing::info!(
                            session_id = session_id.id,
                            tool = name,
                            args_len = args_str.len(),
                            "phase1: args parsed ok"
                        );
                        args
                    }
                    Err(e) => {
                        tracing::warn!(
                            session_id = session_id.id,
                            tool = name,
                            error = %e,
                            args_len = args_str.len(),
                            args_preview = &args_str[..args_str.len().min(300)],
                            "phase1: args parse FAILED"
                        );
                        failures.push(ToolFailure {
                            id: id.clone(),
                            error: AgentError::ToolArgsInvalid {
                                name: name.clone(),
                                raw: format!("{} (args: {})", e, args_str),
                            },
                        });
                        continue;
                    }
                };

                tracing::info!(
                    session_id = session_id.id,
                    tool = name,
                    "phase1: checking tool exists in registry"
                );
                // Existence check under a short read lock — never hold the
                // registry lock across `process_approval`, which can wait on a
                // human and would stall every registry writer behind it.
                let exists = { self.tools.read().await.get(name).is_some() };
                if !exists {
                    tracing::warn!(
                        session_id = session_id.id,
                        tool = name,
                        "phase1: tool NOT found"
                    );
                    failures.push(ToolFailure {
                        id: id.clone(),
                        error: AgentError::tool_not_found(name),
                    });
                    continue;
                }

                tracing::info!(
                    session_id = session_id.id,
                    tool = name,
                    "phase1: processing approval"
                );
                self.process_approval(
                    session_id,
                    name,
                    &args,
                    args_str,
                    ctx,
                    event_rx,
                    on_event.clone(),
                )
                .await?;

                tracing::info!(
                    session_id = session_id.id,
                    tool = name,
                    "phase1: approved, pushing to approved list"
                );
                approved.push((id.clone(), name.clone(), args, args_str.to_string()));
            }
            tracing::info!(
                session_id = session_id.id,
                approved_count = approved.len(),
                failure_count = failures.len(),
                "phase1: loop done"
            );
        }

        if approved.is_empty() {
            return Ok(OrchestrateOutcome {
                results: Vec::new(),
                failures,
            });
        }

        // ── Phase 2: Execute all approved tools in parallel ──
        // Use resubscribe so each task has its own receiver (parallel execution).
        // The shared mutex is only held briefly for try_recv, not for the whole
        // tool call — tools run truly in parallel.
        tracing::info!(
            session_id = session_id.id,
            approved_count = approved.len(),
            "phase2: starting parallel execution"
        );
        let shared_rx = Arc::new(tokio::sync::Mutex::new(event_rx.resubscribe()));

        let futures: Vec<_> = approved
            .into_iter()
            .map(|(id, name, args, args_json)| {
                let session_id = session_id.clone();
                let ctx = ctx.clone();
                let on_event = on_event.clone();
                let shared_rx = shared_rx.clone();
                let self_tools = self.tools.clone();
                let self_pipeline = self.pipeline.clone();
                let self_error_recovery = self.error_recovery.clone();
                let self_event_bus = self.event_bus.clone();

                async move {
                    // Create a per-call event_rx by resubscribing to the shared one
                    let mut local_rx = {
                        let rx_guard = shared_rx.lock().await;
                        rx_guard.resubscribe()
                    };

                    // Build a temporary ToolEngine-like context for execute_tool.
                    // We inline the execute_tool logic here to avoid holding &self
                    // across the parallel tasks.
                    //
                    // Panic isolation: a panicking tool future must not unwind
                    // through join_all and silently kill the whole agent turn
                    // (the TUI then spins forever, and the panic text lands on
                    // the raw terminal). catch_unwind converts the panic into a
                    // per-tool failure so the session survives and the LLM sees
                    // an error result. AssertUnwindSafe is sound here: the only
                    // shared state touched across the await points is behind
                    // tokio RwLock / std Mutex guards that are released on
                    // unwind, never left poisoned-in-use by this code path.
                    let result = std::panic::AssertUnwindSafe(Self::execute_tool_static(
                        &self_tools,
                        &self_pipeline,
                        &self_error_recovery,
                        &self_event_bus,
                        &session_id,
                        &id,
                        &name,
                        &args,
                        &args_json,
                        &ctx,
                        &mut local_rx,
                        on_event.clone(),
                    ))
                    .catch_unwind()
                    .await
                    .unwrap_or_else(|payload| {
                        let panic_msg = payload
                            .downcast_ref::<&str>()
                            .map(|s| s.to_string())
                            .or_else(|| payload.downcast_ref::<String>().cloned())
                            .unwrap_or_else(|| "unknown panic".into());
                        tracing::error!(
                            session_id = session_id.id,
                            tool_name = name,
                            panic = %panic_msg,
                            "Tool panicked during execution (parallel)"
                        );
                        // Mirror the ordinary error branch: emit
                        // ActionFinished so UI panels can close the entry
                        // instead of leaking a perpetual spinner.
                        let error_summary = if ctx.language == Language::Zh {
                            format!("❌ 执行失败: tool panicked: {panic_msg}")
                        } else {
                            format!("❌ Tool execution failed: tool panicked: {panic_msg}")
                        };
                        self_event_bus.emit(RuntimeEvent::ActionFinished {
                            session_id: session_id.clone(),
                            tool_name: name.to_string(),
                            summary: error_summary,
                            agent_id: None,
                            trace_id: None,
                            denied: false,
                            details: None,
                        });
                        Err(AgentError::ToolExecution {
                            name: name.to_string(),
                            source: Box::new(AgentError::Internal(format!(
                                "tool panicked: {panic_msg}"
                            ))),
                        })
                    });

                    (id, name, result)
                }
            })
            .collect();

        let outcomes = futures_util::future::join_all(futures).await;

        // Drain the ORIGINAL event_rx after all tools complete.
        // This delivers ActionStarted/Finished events via on_event and
        // consumes them from the channel so drain_locked at the next turn
        // start won't re-deliver them. Checkpoint events that arrived
        // during orchestrate are also drained here (they only go through
        // event_bus.emit, not on_event in the tasks).
        {
            let mut cb = on_event.lock().unwrap();
            EventBus::drain_async_events(event_rx, &mut *cb)?;
        }

        // ── Phase 3: Collect results ──
        let mut results = Vec::with_capacity(outcomes.len());
        for (id, _name, result) in outcomes {
            match result {
                Ok(result) => results.push(result),
                Err(e) if e.is_cancelled() => return Err(e),
                Err(e) => failures.push(ToolFailure { id, error: e }),
            }
        }

        Ok(OrchestrateOutcome { results, failures })
    }

    /// Static version of [`execute_tool`] for use in parallel orchestration.
    ///
    /// Takes the engine's components as parameters instead of `&self`, so it can
    /// be called from concurrent tasks without shared borrows.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn execute_tool_static<F>(
        tools: &Arc<RwLock<ToolRegistry>>,
        pipeline: &DefaultPipeline,
        error_recovery: &Arc<dyn ToolErrorRecovery>,
        event_bus: &EventBus,
        session_id: &SessionId,
        id: &str,
        name: &str,
        args: &Value,
        tool_args_json: &str,
        ctx: &ExecutionContext,
        _event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<ToolExecutionResult>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send,
    {
        tracing::debug!(
            session_id = session_id.id,
            tool = name,
            args_len = tool_args_json.len(),
            "execute tool start (parallel)"
        );

        // Emit ActionStarted via internal EventBus
        event_bus.emit(RuntimeEvent::ActionStarted {
            session_id: session_id.clone(),
            tool_name: name.to_string(),
            args_json: tool_args_json.to_string(),
            agent_id: None,
            trace_id: None,
        });
        // NOTE: No drain here — orchestrate drains after join_all.

        // Build ToolContext with UserEvent channel for tool-produced events
        let (user_event_tx, mut user_event_rx) = mpsc::unbounded_channel::<UserEvent>();
        let waiting_for_human = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let tool_context = ToolContext {
            session_id: session_id.clone(),
            user_event_tx,
            llm_client: ctx.llm_client.clone(),
            session_store: Some(ctx.session_manager.session_store().clone()),
            language: ctx.language.clone(),
            cancel_token: ctx.cancel_token.clone(),
            max_output_chars: ctx.max_output_chars,
            exec_clock: ExecClock::default(),
            waiting_for_human: waiting_for_human.clone(),
            event_bus: event_bus.clone(),
        };

        tracing::info!(
            session_id = session_id.id,
            tool = name,
            "execute_tool_static: acquiring tools read lock (parallel)"
        );
        // Owned handle out, lock released before the call — see `execute_tool`:
        // the registry lock must not be held across `tool.call`, or a
        // self-registering tool deadlocks on `registry.write()`.
        let tool = { tools.read().await.get(name) };
        tracing::info!(
            session_id = session_id.id,
            tool = name,
            "execute_tool_static: tools read lock acquired (parallel)"
        );
        let tool_result = match tool {
            Some(tool) => {
                tracing::info!(
                    session_id = session_id.id,
                    tool = name,
                    "execute_tool_static: tool found, calling pipeline (parallel)"
                );

                let call_pipeline = DefaultPipeline::new(
                    pipeline.policy(),
                    ctx.tool_timeout_ms,
                    ctx.max_output_chars,
                );

                tracing::info!(
                    session_id = session_id.id,
                    tool = name,
                    "execute_tool_static: entering select loop (parallel)"
                );
                let future = call_pipeline.execute(tool.as_ref(), args, &tool_context);
                tokio::pin!(future);

                let output = loop {
                    tokio::select! {
                        result = &mut future => break result,
                        Some(user_event) = user_event_rx.recv() => {
                            if let Ok(mut cb) = on_event.lock() {
                                cb(RuntimeEvent::UserEvent {
                                    session_id: session_id.clone(),
                                    event: user_event,
                                    agent_id: None,
                                    trace_id: None,
                                })?;
                            }
                        }
                        _ = ctx.cancel_token.cancelled() => {
                            tracing::info!(session_id = session_id.id, tool = name, "tool execution cancelled");
                            return Err(crate::types::AgentError::Cancelled);
                        }
                    }
                };

                // Drain remaining UserEvents
                while let Ok(user_event) = user_event_rx.try_recv() {
                    if let Ok(mut cb) = on_event.lock() {
                        cb(RuntimeEvent::UserEvent {
                            session_id: session_id.clone(),
                            event: user_event,
                            agent_id: None,
                            trace_id: None,
                        })?;
                    }
                }

                match output {
                    Ok(output) => {
                        tracing::info!(
                            session_id = session_id.id,
                            tool = name,
                            "execute_tool_static: tool completed (parallel)"
                        );
                        output
                    }
                    Err(e) => {
                        tracing::error!(session_id = session_id.id, tool_name = name, error = %e, "Tool execution failed (parallel)");
                        let error_summary = if ctx.language == Language::Zh {
                            format!("❌ 执行失败: {}", e)
                        } else {
                            format!("❌ Tool execution failed: {}", e)
                        };
                        event_bus.emit(RuntimeEvent::ActionFinished {
                            session_id: session_id.clone(),
                            tool_name: name.to_string(),
                            summary: error_summary,
                            agent_id: None,
                            trace_id: None,
                            denied: false,
                            details: None,
                        });
                        // NOTE: No drain here — orchestrate drains after join_all.
                        return Err(AgentError::ToolExecution {
                            name: name.to_string(),
                            source: Box::new(e),
                        });
                    }
                }
            }
            None => {
                tracing::warn!(
                    session_id = session_id.id,
                    tool = name,
                    "tool not found in registry (parallel)"
                );
                vec![Content::text(if ctx.language == Language::Zh {
                    format!("工具 {} 未找到", name)
                } else {
                    format!("Tool {} not found", name)
                })]
            }
        };

        // Emit ActionFinished via internal EventBus
        event_bus.emit(RuntimeEvent::ActionFinished {
            session_id: session_id.clone(),
            tool_name: name.to_string(),
            summary: content_text(&tool_result),
            agent_id: None,
            trace_id: None,
            denied: false,
            details: content_details(&tool_result),
        });
        // NOTE: No drain here — orchestrate drains after join_all.
        // Draining here would require holding the shared event_rx lock,
        // serializing parallel tool execution.

        error_recovery.on_success(session_id, name);

        Ok(ToolExecutionResult {
            id: id.to_string(),
            name: name.to_string(),
            output: tool_result,
            waiting_for_human: waiting_for_human.load(std::sync::atomic::Ordering::Relaxed),
        })
    }
}
