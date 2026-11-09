use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use crate::engine::runtime::tool_engine::ExecutionContext;
use crate::tool::{content_images, content_text};
use crate::types::{
    ActionOutcome, AgentError, AgentResult, BatchStatus, ChatMessage, Observation, RuntimeEvent,
    SessionId, ToolResultData,
};

use super::{ToolBatchReport, ToolCallExecutor};

impl<'c> ToolCallExecutor<'c> {
    /// Execute a batch of tool calls. `pre_resolved` carries results for calls
    /// that must NOT be executed (e.g. truncation-guard invalid calls) — they
    /// are rendered as tool results right after the assistant message so every
    /// tool_call_id in the assistant message gets exactly one result in order
    /// (provider pairing requirement, MiniMax error 2013).
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_tool_calls<F>(
        &self,
        session_id: &SessionId,
        tool_calls: &[(String, String, String)],
        pre_resolved: &[(String, String)],
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
        reasoning: &str,
        full_text: &str,
    ) -> AgentResult<ToolBatchReport>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        // Only execute calls that don't already have a pre-resolved result.
        let exec_calls: Vec<(String, String, String)> = tool_calls
            .iter()
            .filter(|(id, _, _)| !pre_resolved.iter().any(|(pid, _)| pid == id))
            .cloned()
            .collect();

        let tool_names: Vec<&str> = exec_calls
            .iter()
            .map(|(_, name, _)| name.as_str())
            .collect();
        tracing::debug!(
            session_id = session_id.id,
            ?tool_names,
            pre_resolved_count = pre_resolved.len(),
            "handle tool calls start"
        );

        let config = self.core.config_snapshot_async().await;

        let ctx = ExecutionContext {
            session_manager: self.core.session_manager.clone(),
            llm_client: Some(self.core.llm_engine.get_provider()),
            language: config.language.clone(),
            tool_timeout_ms: config.tool.tool_timeout_ms,
            max_output_chars: config.tool.max_tool_output_chars,
            cancel_token: self.core.cancel_token(),
        };

        // Orchestrate: approval check + execution for executable calls only.
        // Approval denial is handled inside process_approval (pushes fake tool state
        // and returns Err). We only push real tool calls on success.
        let outcome = if exec_calls.is_empty() {
            None
        } else {
            Some(
                self.core
                    .tool_engine
                    .orchestrate(session_id, &exec_calls, &ctx, event_rx, on_event.clone())
                    .await?,
            )
        };

        // Per-batch park tracking: whether this batch left a human step pending
        // (e.g. a design-canvas screen awaiting a pick). Assignment, not
        // accumulation — a later batch of real tools means the model worked on
        // after the park, so the run is no longer simply waiting. Must land
        // before the failure early-return below so a mixed batch (parked call
        // plus a failed one) still records the park.
        if let Some(ref outcome) = outcome {
            let awaiting = outcome.results.iter().any(|r| r.waiting_for_human);
            self.core
                .with_session_mut(session_id, |session| {
                    session.run_state.awaiting_human = awaiting;
                })
                .await?;
        }

        // Push assistant tool calls to session after all approvals pass — the
        // backend renders the history protocol and carries the same messages in
        // the observation for checkpoint replay (M3 seam).
        let mut history_messages: Vec<ChatMessage> = Vec::new();
        {
            let tc: Vec<(String, String, String)> = tool_calls.to_vec();
            let ft = if full_text.is_empty() {
                None
            } else {
                Some(full_text.to_string())
            };
            let rt = if reasoning.is_empty() {
                None
            } else {
                Some(reasoning.to_string())
            };
            let assistant = crate::engine::AgentSession::render_assistant_tool_calls(&tc, rt, ft);
            self.core
                .with_session_mut(session_id, |session| {
                    session.push_chat_message(assistant.clone());
                })
                .await?;
            history_messages.push(assistant);
        }

        // Push pre-resolved results (e.g. truncation-guard invalid calls) —
        // these must land immediately after the assistant message so every
        // tool_call_id gets exactly one result (provider pairing requirement).
        for (id, text) in pre_resolved {
            let msg = ChatMessage::tool(id, text);
            self.core
                .with_session_mut(session_id, |session| {
                    session.push_chat_message(msg.clone());
                })
                .await?;
            history_messages.push(msg);
        }

        // Process successful results: push to session. The pictures a tool
        // handed back ride with the words (§4.2): they are output the model
        // looks at, not metadata to summarise away.
        if let Some(ref outcome) = outcome {
            for result in &outcome.results {
                let msg = ChatMessage::tool_with_images(
                    &result.id,
                    content_text(&result.output),
                    content_images(&result.output),
                );
                self.core
                    .with_session_mut(session_id, |session| {
                        session.push_chat_message(msg.clone());
                    })
                    .await?;
                history_messages.push(msg);
            }

            // Push each failure's own error text so the model sees exactly what broke
            // (not just the first failure's message) and every tool call has a result.
            for failure in &outcome.failures {
                let msg = ChatMessage::tool(&failure.id, failure.error.to_string());
                self.core
                    .with_session_mut(session_id, |session| {
                        session.push_chat_message(msg.clone());
                    })
                    .await?;
                history_messages.push(msg);
            }
        }

        // Per-action fates in submission order (ids pair with the batch).
        let outcomes: Vec<ActionOutcome> = tool_calls
            .iter()
            .map(|(id, name, _)| {
                // Pre-resolved calls: fate comes from the pre-filled result.
                if let Some((_, text)) = pre_resolved.iter().find(|(pid, _)| pid == id) {
                    return ActionOutcome {
                        action_id: id.clone(),
                        name: name.clone(),
                        ok: true,
                        denied: false,
                        summary: text.clone(),
                        details: None,
                    };
                }
                if let Some(ref outcome) = outcome {
                    if let Some(r) = outcome.results.iter().find(|r| r.id == *id) {
                        return ActionOutcome {
                            action_id: id.clone(),
                            name: name.clone(),
                            ok: true,
                            denied: false,
                            summary: content_text(&r.output),
                            details: None,
                        };
                    }
                    if let Some(f) = outcome.failures.iter().find(|f| f.id == *id) {
                        return ActionOutcome {
                            action_id: id.clone(),
                            name: name.clone(),
                            ok: false,
                            denied: matches!(f.error, AgentError::ApprovalDenied { .. }),
                            summary: f.error.to_string(),
                            details: None,
                        };
                    }
                }
                // Defensive fallback.
                ActionOutcome {
                    action_id: id.clone(),
                    name: name.clone(),
                    ok: false,
                    denied: false,
                    summary: "no result recorded".to_string(),
                    details: None,
                }
            })
            .collect();

        let first_failure = outcome.and_then(|o| o.failures.into_iter().next().map(|f| f.error));
        Ok(ToolBatchReport {
            observation: Observation {
                outcomes,
                events: vec![],
                history_messages,
                batch_status: BatchStatus::Completed,
            },
            first_failure,
        })
    }

    /// Model-visible fate for an action the crash window cut off (OV-4) —
    /// the kernel's one vocabulary ([`crate::engine::executor::INTERRUPTED_SUMMARY`]),
    /// shared with the mounted-backend replay path (T10c).
    const INTERRUPTED_SUMMARY: &'static str = crate::engine::executor::INTERRUPTED_SUMMARY;

    /// D3 observation replay (T6) — resume never executes actions (side
    /// effects are one-shot). A completed batch replays its recorded
    /// observation verbatim; otherwise the batch is dangling (crash window)
    /// and every action gets a synthetic fate: the migration mark's legacy
    /// `results` summary when the pre-M3 payload recorded one, else
    /// [`Self::INTERRUPTED_SUMMARY`] (已中断/未执行).
    ///
    /// Contract: the restored session history ends before this batch's
    /// messages — the replay pushes exactly the batch's protocol pair
    /// (assistant tool-call message + one result per action) so provider
    /// validation stays sound.
    pub(in crate::engine::runtime::react) async fn replay_batch(
        &self,
        session_id: &SessionId,
        tool_calls: &[(String, String, String)],
        recorded: Option<Observation>,
        legacy_results: &[ToolResultData],
    ) -> AgentResult<Observation> {
        if let Some(observation) = recorded {
            self.core
                .with_session_mut(session_id, |session| {
                    for msg in &observation.history_messages {
                        session.push_chat_message(msg.clone());
                    }
                })
                .await?;
            return Ok(observation);
        }

        let mut history_messages: Vec<ChatMessage> = Vec::with_capacity(tool_calls.len() + 1);
        let assistant =
            crate::engine::AgentSession::render_assistant_tool_calls(tool_calls, None, None);
        history_messages.push(assistant);

        let mut outcomes = Vec::with_capacity(tool_calls.len());
        for (id, name, _) in tool_calls {
            let summary = legacy_results
                .iter()
                .find(|r| r.tool_call_id == *id)
                .map(|r| r.summary.as_str());
            let text = summary.unwrap_or(Self::INTERRUPTED_SUMMARY);
            history_messages.push(ChatMessage::tool(id, text));
            outcomes.push(ActionOutcome {
                action_id: id.clone(),
                name: name.clone(),
                ok: summary.is_some(),
                denied: false,
                summary: text.to_string(),
                details: None,
            });
        }

        self.core
            .with_session_mut(session_id, |session| {
                for msg in &history_messages {
                    session.push_chat_message(msg.clone());
                }
            })
            .await?;

        Ok(Observation {
            outcomes,
            events: vec![],
            history_messages,
            // Reconstructed at resume, not observed (D3).
            batch_status: BatchStatus::Interrupted,
        })
    }
}
