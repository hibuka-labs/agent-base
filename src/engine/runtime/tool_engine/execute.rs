use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::sync::{broadcast, mpsc};

use crate::engine::pipeline::{DefaultPipeline, ToolExecutionPipeline};
use crate::engine::runtime::event_bus::EventBus;
use crate::tool::{Content, ExecClock, ToolContext, content_details, content_text};
use crate::types::{AgentError, AgentResult, Language, RuntimeEvent, SessionId, UserEvent};

use super::{ExecutionContext, ToolEngine, ToolExecutionResult};

impl ToolEngine {
    #[allow(dead_code, clippy::too_many_arguments)]
    pub async fn execute_tool<F>(
        &self,
        session_id: &SessionId,
        id: &str,
        name: &str,
        args: &Value,
        tool_args_json: &str,
        ctx: &ExecutionContext,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<ToolExecutionResult>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send,
    {
        tracing::debug!(
            session_id = session_id.id,
            tool = name,
            args_len = tool_args_json.len(),
            "execute tool start"
        );

        // Emit ActionStarted via internal EventBus
        self.event_bus.emit(RuntimeEvent::ActionStarted {
            session_id: session_id.clone(),
            tool_name: name.to_string(),
            args_json: tool_args_json.to_string(),
            agent_id: None,
            trace_id: None,
        });
        {
            let mut cb = on_event.lock().unwrap();
            EventBus::drain_async_events(event_rx, &mut *cb)?;
        }

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
            event_bus: self.event_bus.clone(),
        };

        // Lookup tool and execute via pipeline.
        // The pipeline handles: before_call hook → timeout → truncation → after_call hook.
        // ToolEngine handles: event emission and UserEvent forwarding.
        tracing::debug!(
            session_id = session_id.id,
            tool = name,
            "looking up tool in registry"
        );
        // Take an owned handle and **drop the registry lock before the call**.
        // A tool that registers another tool from inside `call` (e.g. phimint's
        // `add_tool` mounting a script tool) needs `registry.write()`, and a read
        // lock held across the call deadlocks that write forever — observed as a
        // bogus `[Tool Timeout]` after the mount had already saved its entry
        // (acceptance session 20261018_29e1e182).
        let tool = { self.tools.read().await.get(name) };
        let tool_result = match tool {
            Some(tool) => {
                tracing::debug!(
                    session_id = session_id.id,
                    tool = name,
                    "tool found, executing via pipeline"
                );

                // Per-call pipeline: inherits policy from self.pipeline, adds caller's timeout/truncation.
                let pipeline = DefaultPipeline::new(
                    self.pipeline.policy(),
                    ctx.tool_timeout_ms,
                    ctx.max_output_chars,
                );

                let future = pipeline.execute(tool.as_ref(), args, &tool_context);
                tokio::pin!(future);

                // Execute with UserEvent forwarding interleaved via tokio::select!
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

                // Drain remaining UserEvents after tool completes
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
                    Ok(output) => output,
                    Err(e) => {
                        tracing::error!(session_id = session_id.id, tool_name = name, error = %e, "Tool execution failed");
                        // Emit ActionFinished with error summary before returning error
                        let error_summary = if ctx.language == Language::Zh {
                            format!("❌ 执行失败: {}", e)
                        } else {
                            format!("❌ Tool execution failed: {}", e)
                        };
                        self.event_bus.emit(RuntimeEvent::ActionFinished {
                            session_id: session_id.clone(),
                            tool_name: name.to_string(),
                            summary: error_summary,
                            agent_id: None,
                            trace_id: None,
                            denied: false,
                            details: None,
                        });
                        // Use `let _ =` to avoid masking the original tool error
                        // if the event callback fails
                        let _ = {
                            let mut cb = on_event.lock().unwrap();
                            EventBus::drain_async_events(event_rx, &mut *cb)
                        };
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
                    "tool not found in registry"
                );
                vec![Content::text(if ctx.language == Language::Zh {
                    format!("工具 {} 未找到", name)
                } else {
                    format!("Tool {} not found", name)
                })]
            }
        };

        // Emit ActionFinished via internal EventBus
        self.event_bus.emit(RuntimeEvent::ActionFinished {
            session_id: session_id.clone(),
            tool_name: name.to_string(),
            summary: content_text(&tool_result),
            agent_id: None,
            trace_id: None,
            denied: false,
            details: content_details(&tool_result),
        });
        {
            let mut cb = on_event.lock().unwrap();
            EventBus::drain_async_events(event_rx, &mut *cb)?;
        }

        // A successful execution breaks any consecutive-failure streak for this tool.
        self.error_recovery.on_success(session_id, name);

        Ok(ToolExecutionResult {
            id: id.to_string(),
            name: name.to_string(),
            output: tool_result,
            waiting_for_human: waiting_for_human.load(std::sync::atomic::Ordering::Relaxed),
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub async fn process_approval<F>(
        &self,
        session_id: &SessionId,
        tool_name: &str,
        args: &Value,
        _tool_args_json: &str,
        ctx: &ExecutionContext,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<()>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send,
    {
        let approval_request = match self.tool_policy.as_ref() {
            Some(policy) => policy.evaluate_approval(tool_name, args).await,
            None => None,
        };

        let Some(request) = approval_request else {
            return Ok(());
        };

        let approved = if let Some(key) = request.action_key.as_deref() {
            ctx.session_manager.cached_approval(session_id, key).await
        } else {
            false
        };

        if approved {
            tracing::debug!(
                session_id = session_id.id,
                tool = tool_name,
                "approval cached, skipping"
            );
            return Ok(());
        }

        tracing::debug!(session_id = session_id.id, tool = tool_name, risk = ?request.risk_level, "requesting approval");

        self.event_bus.emit(RuntimeEvent::AwaitingApproval {
            session_id: session_id.clone(),
            request: request.clone(),
            agent_id: None,
            trace_id: None,
        });
        {
            let mut cb = on_event.lock().unwrap();
            EventBus::drain_async_events(event_rx, &mut *cb)?;
        }

        let decision = match self.approval_handler.as_ref() {
            Some(handler) => {
                let timeout = std::time::Duration::from_secs(
                    std::env::var("APPROVAL_TIMEOUT_SECS")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(300),
                );
                let result = tokio::time::timeout(
                    timeout,
                    handler.approve(request.clone(), ctx.cancel_token.clone()),
                )
                .await;
                match result {
                    Ok(result) => result.map_err(|e| {
                        AgentError::internal(format!("Approval handler failed: {e}"))
                    })?,
                    Err(_) => {
                        tracing::warn!(
                            session_id = session_id.id,
                            ?timeout,
                            "Approval timed out, defaulting to Deny"
                        );
                        crate::types::ApprovalDecision::Deny
                    }
                }
            }
            None => crate::types::ApprovalDecision::Deny,
        };

        match decision {
            crate::types::ApprovalDecision::AllowOnce => {
                tracing::info!(
                    session_id = session_id.id,
                    tool = tool_name,
                    decision = "AllowOnce",
                    "approval granted"
                );
            }
            crate::types::ApprovalDecision::AllowAlways => {
                tracing::info!(
                    session_id = session_id.id,
                    tool = tool_name,
                    decision = "AllowAlways",
                    "approval granted (cached)"
                );
                if let Some(action_key) = request.action_key.clone() {
                    ctx.session_manager
                        .cache_approval(session_id, action_key)
                        .await;
                }
            }
            crate::types::ApprovalDecision::Deny => {
                tracing::warn!(
                    session_id = session_id.id,
                    tool = tool_name,
                    decision = "Deny",
                    "approval denied"
                );
                let denial_summary =
                    format!("[Action Denied]: tool {} rejected by approval", tool_name);
                // Not recorded to session history — a user denial is a UI-layer
                // interaction; there is no need for the LLM to see it
                self.event_bus.emit(RuntimeEvent::ActionFinished {
                    session_id: session_id.clone(),
                    tool_name: tool_name.to_string(),
                    summary: denial_summary,
                    agent_id: None,
                    trace_id: None,
                    denied: true,
                    details: None,
                });
                let _ = {
                    let mut cb = on_event.lock().unwrap();
                    EventBus::drain_async_events(event_rx, &mut *cb)
                };
                return Err(AgentError::ApprovalDenied {
                    tool_name: tool_name.to_string(),
                });
            }
        }

        Ok(())
    }
}
