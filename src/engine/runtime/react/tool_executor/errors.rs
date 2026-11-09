use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use crate::engine::recovery::ToolErrorAction;
use crate::engine::runtime::react::entry::drain_locked;
use crate::types::{AgentError, AgentResult, RunOutcome, RuntimeEvent, SessionId};

use super::{ToolCallExecutor, failing_tool_names};

impl<'c> ToolCallExecutor<'c> {
    pub(super) async fn handle_tool_error<F>(
        &self,
        session_id: &SessionId,
        tool_calls: &[(String, String, String)],
        e: AgentError,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<Option<RunOutcome>>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send,
    {
        let config = self.core.config_snapshot_async().await;

        if e.is_cancelled() {
            // Don't emit RunFinished when cancelled — RunCancelled will be emitted by run_turn
            // But still drain pending events and persist session
            drain_locked(event_rx, &on_event)?;
            let session = self.core.session_manager.session_or_err(session_id).await?;
            if let Err(e) = self
                .core
                .session_manager
                .session_store()
                .save(&session)
                .await
            {
                tracing::warn!(session_id = session_id.id, error = %e, "Failed to persist session");
                if config.execution.fail_on_persist_error {
                    return Err(AgentError::internal(format!(
                        "Session persistence failed: {e}"
                    )));
                }
            }
            return Err(e);
        }

        let names = failing_tool_names(tool_calls, &e);
        let error_text = e.to_string();
        let retry_prompt_template: Option<String> = config.tool.tool_error_retry_prompt.clone();

        // User actively rejects (types n) → stop immediately, without going
        // through the recovery retry logic
        if matches!(e, AgentError::ApprovalDenied { .. }) {
            tracing::info!(
                session_id = session_id.id,
                "user rejected tool call, stopping immediately"
            );
            let error_summary = if config.language == crate::types::Language::Zh {
                format!("❌ 用户拒绝执行: {}", e)
            } else {
                format!("❌ User rejected: {}", e)
            };
            self.core
                .with_session_mut(session_id, |session| {
                    session.close_dangling_tool_calls(&error_summary);
                    session.remove_ephemeral_messages();
                })
                .await?;
            self.core.event_bus.emit(RuntimeEvent::RunFinished {
                session_id: session_id.clone(),
                agent_id: None,
                trace_id: None,
            });
            drain_locked(event_rx, &on_event)?;
            let session = self.core.session_manager.session_or_err(session_id).await?;
            if let Err(e) = self
                .core
                .session_manager
                .session_store()
                .save(&session)
                .await
            {
                tracing::warn!(session_id = session_id.id, error = %e, "Failed to persist session");
            }
            return Ok(Some(RunOutcome::Completed));
        }

        let action = self
            .core
            .tool_engine
            .error_recovery()
            .on_error(session_id, &names, &e)?;
        match action {
            ToolErrorAction::Stop => {
                // Close any dangling tool calls in the session before stopping
                let error_summary = if config.language == crate::types::Language::Zh {
                    format!("❌ 执行失败: {}", e)
                } else {
                    format!("❌ Tool execution failed: {}", e)
                };
                // Emit ActionFinished so TUI shows the failure before RunFinished
                self.core.event_bus.emit(RuntimeEvent::ActionFinished {
                    session_id: session_id.clone(),
                    tool_name: names.join(", "),
                    summary: error_summary.clone(),
                    agent_id: None,
                    trace_id: None,
                    denied: false,
                    details: None,
                });
                self.core
                    .with_session_mut(session_id, |session| {
                        session.close_dangling_tool_calls(&error_summary);
                        session.remove_ephemeral_messages();
                    })
                    .await?;
                self.core.event_bus.emit(RuntimeEvent::RunFinished {
                    session_id: session_id.clone(),
                    agent_id: None,
                    trace_id: None,
                });
                drain_locked(event_rx, &on_event)?;
                let session = self.core.session_manager.session_or_err(session_id).await?;
                if let Err(e) = self
                    .core
                    .session_manager
                    .session_store()
                    .save(&session)
                    .await
                {
                    tracing::warn!(session_id = session_id.id, error = %e, "Failed to persist session");
                    if config.execution.fail_on_persist_error {
                        return Err(AgentError::internal(format!(
                            "Session persistence failed: {e}"
                        )));
                    }
                }
                Ok(Some(RunOutcome::Failed {
                    error: format!("Tool execution failed: {}", e),
                }))
            }
            ToolErrorAction::RetryWithHistory { errors } => {
                let tool_name = names.join(", ");
                let error_history = format!(
                    "The tool `{}` has failed {} consecutive times. \
                     Error history:\n\n{}\n\n\
                     Please analyze these errors and either switch to a different \
                     approach or explain the failure to the user.",
                    tool_name,
                    errors.len(),
                    errors
                        .iter()
                        .enumerate()
                        .map(|(i, e)| format!("{}. {}", i + 1, e))
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                );

                let summary = if config.language == crate::types::Language::Zh {
                    format!(
                        "⚠️ {} 连续失败 {} 次，已将错误历史发给 LLM 评估",
                        tool_name,
                        errors.len()
                    )
                } else {
                    format!(
                        "⚠️ {} failed {} consecutive times, error history sent to LLM",
                        tool_name,
                        errors.len()
                    )
                };

                // Emit ActionFinished so TUI shows the recovery action
                self.core.event_bus.emit(RuntimeEvent::ActionFinished {
                    session_id: session_id.clone(),
                    tool_name: tool_name.clone(),
                    summary: summary.clone(),
                    agent_id: None,
                    trace_id: None,
                    denied: false,
                    details: None,
                });

                self.core
                    .with_session_mut(session_id, |session| {
                        session.close_dangling_tool_calls(&summary);
                        // Machine-composed recovery guidance: synthetic, so
                        // the rotation trail never mistakes it for the user.
                        session.push_user_synthetic(error_history);
                    })
                    .await?;
                Ok(None)
            }
            ToolErrorAction::Retry => {
                let retry_prompt = match &retry_prompt_template {
                    Some(template) => template
                        .replace("{tool_names}", &names.join(", "))
                        .replace("{error}", &error_text),
                    None => format!(
                        "Tool calls failed: {}\nError: {}\nPlease analyze the error and adjust your approach.",
                        names.join(", "),
                        error_text,
                    ),
                };

                let error_summary = if config.language == crate::types::Language::Zh {
                    format!("❌ 执行失败: {}", error_text)
                } else {
                    format!("❌ Tool execution failed: {}", error_text)
                };
                self.core
                    .with_session_mut(session_id, |session| {
                        session.close_dangling_tool_calls(&error_summary);
                        // Same synthetic notice as the recovery path above.
                        session.push_user_synthetic(retry_prompt);
                    })
                    .await?;
                Ok(None)
            }
        }
    }
}
