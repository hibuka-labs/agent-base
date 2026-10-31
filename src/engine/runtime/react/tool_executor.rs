//! Tool-call backend for the M3 `ActionExecutor` seam: the `ToolCallExecutor`
//! owns everything the tool protocol needs — envelope parse, truncation rescue,
//! batch execution, and history rendering. The kernel sees only
//! [`ParseOutcome`] / [`Observation`] / [`ExecError`].
//!
//! The `(id, name, args)` triple is internal to this backend; the kernel
//! currency is [`Action`]. `run_actions` keeps the react loop's turn-level
//! flow (breaker strikes, turn-end metrics) — full flow-face unification on
//! the trait is T10 debt (see the `ActionExecutor` impl below).

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::broadcast;

use crate::engine::executor::{ActionExecutor, ExecCtx, ExecError, ParseError, ParseOutcome};
use crate::engine::recovery::ToolErrorAction;
use crate::engine::runtime::LlmTurnResult;
use crate::engine::runtime::plan_runner::RuntimeCore;
use crate::engine::runtime::tool_engine::ExecutionContext;
use crate::tool::{content_images, content_text};
use crate::types::{
    Action, ActionFormat, ActionOutcome, AgentError, AgentResult, BatchStatus, ChatMessage,
    CheckpointData, CheckpointStep, FinishReason, MessageRole, Observation, RunOutcome,
    RuntimeEvent, SessionId, ToolResultData,
};

use super::entry::drain_locked;
use super::turn_end::TurnEndCtx;
use super::turn_guard::TurnMetrics;
use super::turn_loop::TurnFlow;

/// A finished batch: the observation (per-action fates + rendered history) and,
/// for the flow path, the first per-call failure to feed recovery.
pub(super) struct ToolBatchReport {
    pub observation: Observation,
    /// First per-call failure (an outcome, not an error — D9); `None` when
    /// every call had a success fate.
    pub first_failure: Option<AgentError>,
}

/// The tool-call action backend (M3 seam).
pub(super) struct ToolCallExecutor<'c> {
    core: &'c RuntimeCore,
    session_id: SessionId,
}

/// Internal triple currency: `Action` → backend triple.
pub(super) fn actions_to_triples(actions: &[Action]) -> Vec<(String, String, String)> {
    actions
        .iter()
        .map(|a| (a.id.clone(), a.name.clone(), a.raw_args().into_owned()))
        .collect()
}

/// Internal triple currency: backend triple → `Action` (identity recomputed
/// from the current — possibly salvaged — argument text).
pub(super) fn triples_to_actions(tool_calls: &[(String, String, String)]) -> Vec<Action> {
    tool_calls
        .iter()
        .map(|(id, name, args)| Action::from_tool_call(id.clone(), name.clone(), args.clone()))
        .collect()
}

impl<'c> ToolCallExecutor<'c> {
    pub(super) fn new(core: &'c RuntimeCore, session_id: SessionId) -> Self {
        Self { core, session_id }
    }

    /// Turn output → four-state verdict (D4+OV-7). Envelope flattening +
    /// classification only — no schema-dependent salvage here (that runs on
    /// the execute side, where schema lookup is async).
    pub(super) fn classify(&self, turn: &LlmTurnResult) -> ParseOutcome {
        classify_turn(turn)
    }

    /// Execute tool calls for one LLM turn: truncation guard, execution,
    /// checkpoint emission, and turn-end metrics.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run_actions<F>(
        &self,
        user_input: &str,
        turn_count: u32,
        turn_start: std::time::Instant,
        model: &str,
        actions: Vec<Action>,
        finish_reason: &FinishReason,
        reasoning_text: String,
        full_text: String,
        metrics: &TurnMetrics<'_>,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<TurnFlow>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        let session_id = &self.session_id;
        let mut tool_calls = actions_to_triples(&actions);
        // Truncation guard — a well-formed provider stream always yields
        // complete JSON for each tool call's arguments. When that breaks, the
        // call must NOT be executed: re-issue it instead. Three ways it breaks:
        //   1. finish_reason=length — the model hit the output token limit
        //      mid-arguments (provider reports this honestly).
        //   2. finish_reason=tool_calls (or anything else) but a *named* call's
        //      arguments don't parse as JSON — the provider truncated the
        //      argument stream and then *mislabelled* the finish as a normal
        //      tool_calls stop. mimo-v2.5-pro does exactly this under load
        //      (emits a dangling `{"agent_path": ` with finish_reason=tool_calls
        //      and never sets length). Checking the structural symptom makes
        //      the guard independent of a finish_reason we can't trust.
        //   3. arguments parse as JSON but are a `{error:"tool_call_arguments_truncated",...}`
        //      wrapper — the model echoed back truncation residue it saw earlier.
        //      Valid JSON, so case 2's check misses it; recognise it by shape.
        //   4. arguments parse as a valid *empty object* `{}` for a tool that
        //      declares required parameters. This is not provider truncation — it
        //      is the model *mimicking* our own sanitizer: truncated spawn_agent
        //      args get stored as `{}` (see session.rs, to avoid a 400 on the next
        //      request), the model sees a `spawn_agent {}` in its history and emits
        //      a stray empty twin next to a real call. It parses fine, so case 2
        //      skips it, and it fails typed schema at execution
        //      ("argument parsing failed: {}"). Detect it via the tool schema.
        // Left to fall through, cases (2)/(3)/(4) become a ToolArgsInvalid
        // execution failure that feeds ConsecutiveFailureRecovery and aborts the
        // run ("failed 3 consecutive times"). Routing them here degrades to a
        // benign retry that never increments the failure counter.
        // ── Per-call validity check ──
        // Check each tool_call individually so we can execute the valid ones
        // and only re-issue the broken ones (instead of discarding the whole batch).
        let truncated_by_limit = finish_reason.is_truncated();
        let mut invalid_indices: Vec<usize> = Vec::new();
        for (i, (_id, name, args)) in tool_calls.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            // Empty args for a named tool = truncation (provider cut off
            // before generating any arguments). Treat as invalid so the
            // truncation guard / circuit breaker handles it.
            if args.trim().is_empty() {
                invalid_indices.push(i);
                continue;
            }
            let dominated = serde_json::from_str::<Value>(args)
                .map(|v| is_truncation_wrapper(&v))
                .unwrap_or(true); // parse failure = truncated
            if dominated {
                invalid_indices.push(i);
                continue;
            }
            // Case 4: empty `{}` for a tool with required fields
            let empty_object = serde_json::from_str::<Value>(args)
                .map(|v| v.as_object().is_some_and(|o| o.is_empty()))
                .unwrap_or(false);
            if empty_object && self.core.tool_engine.tool_requires_params(name).await {
                invalid_indices.push(i);
                continue;
            }
            // Valid — no issue found
        }
        // finish_reason=length means the whole stream was cut — mark all as invalid
        if truncated_by_limit {
            invalid_indices = (0..tool_calls.len()).collect();
        }

        // ── Markup salvage: recover parameters from template markup leaked
        // through the text channel ──
        // mimo-family providers occasionally write the same call on two
        // channels: the native tool_call stream's arguments are truncated
        // mid-way (finish falsely reports Stop), while the content text leaks
        // the chat-template markup `<tool_call><function=…><parameter=…>…`
        // (observed in session 20260923_263149bf: the native stream died on a
        // 15-byte `{"fork_turns":"`, while the leaked text carried the complete
        // task string). The leaked copy is the model's real intent and takes
        // priority over burning a re-issue strike (a re-send will most likely
        // be truncated again). Skip when finish_reason=length: generation was
        // cut off wholesale by the budget there, so markup and args are
        // truncated together and salvaging is pointless. Leaks that fail schema
        // validation (required fields present + type coercion) still go to
        // re-issue, to avoid swapping the death-spiral entry from "truncation"
        // to "missing-argument execution failure".
        if !truncated_by_limit && !invalid_indices.is_empty() {
            let leaked = super::text_toolcall::harvest(&full_text);
            if !leaked.is_empty() {
                let mut salvaged: Vec<usize> = Vec::new();
                for &i in &invalid_indices {
                    let name = tool_calls[i].1.clone();
                    let Some(candidate) = leaked.iter().find(|c| c.name == name) else {
                        continue;
                    };
                    let schema = self.core.tool_engine.tool_schema(&name).await;
                    let Some(args_json) =
                        super::text_toolcall::build_arguments(candidate, schema.as_ref())
                    else {
                        tracing::debug!(
                            session_id = session_id.id,
                            turn = turn_count,
                            tool = %name,
                            "leaked markup found but arguments unusable (missing required \
                             fields) — keeping re-issue path"
                        );
                        continue;
                    };
                    if serde_json::from_str::<Value>(&args_json)
                        .map(|v| is_truncation_wrapper(&v))
                        .unwrap_or(true)
                    {
                        continue;
                    }
                    tool_calls[i].2 = args_json;
                    salvaged.push(i);
                    tracing::info!(
                        session_id = session_id.id,
                        turn = turn_count,
                        tool = %name,
                        recovered_args_len = tool_calls[i].2.len(),
                        "salvaged truncated tool call from leaked text-channel markup — \
                         executing recovered arguments instead of re-issuing"
                    );
                }
                if !salvaged.is_empty() {
                    invalid_indices.retain(|i| !salvaged.contains(i));
                }
            }
        }

        if !invalid_indices.is_empty() {
            let has_valid = invalid_indices.len() < tool_calls.len();
            let is_truncated_call = |args: &str| -> bool {
                serde_json::from_str::<Value>(args)
                    .map(|v| is_truncation_wrapper(&v))
                    .unwrap_or(true)
            };
            let has_truncated = invalid_indices
                .iter()
                .any(|&i| is_truncated_call(&tool_calls[i].2));

            tracing::warn!(
                session_id = session_id.id,
                turn = turn_count,
                tool_count = tool_calls.len(),
                invalid_count = invalid_indices.len(),
                finish_reason = ?finish_reason,
                has_valid_calls = has_valid,
                "some tool-call arguments incomplete — re-issuing invalid, executing valid"
            );

            // ── Circuit breaker (only when ALL calls are invalid) ──
            const TRUNCATION_STRIKE_LIMIT: usize = 3;
            const TRUNCATION_HARD_LIMIT: usize = 5;
            let strikes_after = if has_valid {
                0 // valid calls will reset the counter
            } else {
                let current = self
                    .core
                    .with_session_mut(session_id, |s| s.run_state.truncation_strikes)
                    .await?;
                current + 1
            };

            if !has_valid && strikes_after >= TRUNCATION_HARD_LIMIT {
                // Model kept calling the same tool even after redirect — hard stop.
                let failed_tools: Vec<String> = tool_calls
                    .iter()
                    .map(|(_, name, _)| name.clone())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                let error_msg = format!(
                    "Tool [{}] repeatedly truncated ({} attempts). Model did not \
                     follow redirect instruction. Stopping run.",
                    failed_tools.join(", "),
                    strikes_after,
                );
                tracing::error!(
                    session_id = session_id.id,
                    turn = turn_count,
                    strikes = strikes_after,
                    tools = ?failed_tools,
                    "truncation hard limit reached, stopping run"
                );
                // Record the final strike and close dangling tool calls.
                self.core
                    .with_session_mut(session_id, |session| {
                        session.push_assistant_tool_calls(
                            &tool_calls,
                            Some(reasoning_text),
                            Some(full_text),
                        );
                        for &i in &invalid_indices {
                            session.push_tool_result(&tool_calls[i].0, &error_msg);
                        }
                        session.run_state.record_truncation();
                    })
                    .await?;
                return Ok(TurnFlow::Done(RunOutcome::Failed { error: error_msg }));
            }

            if !has_valid && strikes_after >= TRUNCATION_STRIKE_LIMIT {
                let failed_tools: Vec<String> = tool_calls
                    .iter()
                    .map(|(_, name, _)| name.clone())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                tracing::warn!(
                    session_id = session_id.id,
                    turn = turn_count,
                    strikes = strikes_after,
                    tools = ?failed_tools,
                    "truncation circuit breaker tripped — redirecting model away from truncated tool"
                );
                let redirect_msg = format!(
                    "CRITICAL: The tool [{}] has had its arguments truncated {} \
                     consecutive times. Its parameter schema is too complex for \
                     this model to generate. You MUST NOT call [{}] again in this \
                     session. Instead, perform the task directly using other \
                     available tools (bash, read_file, write_file, etc.). \
                     Do not attempt [{}] again.",
                    failed_tools.join(", "),
                    strikes_after,
                    failed_tools.join(", "),
                    failed_tools.join(", "),
                );
                self.core
                    .with_session_mut(session_id, |session| {
                        session.push_assistant_tool_calls(
                            &tool_calls,
                            Some(reasoning_text),
                            Some(full_text),
                        );
                        for &i in &invalid_indices {
                            session.push_tool_result(&tool_calls[i].0, &redirect_msg);
                        }
                        session.run_state.record_truncation();
                    })
                    .await?;
                return Ok(TurnFlow::Continue);
            }

            // ── Choose guidance text ──
            let guidance: String = if strikes_after >= 2 {
                // Stronger guidance on second+ attempt
                if has_truncated || truncated_by_limit {
                    format!(
                        "Tool call was not executed (attempt {}): arguments are still \
                         truncated. STOP trying this tool — its argument schema is too \
                         complex for your output budget. Use a simpler alternative tool \
                         or explain to the user what you were trying to do.",
                        strikes_after,
                    )
                } else {
                    format!(
                        "Tool call was not executed (attempt {}): empty {{}} arguments. \
                         STOP trying this tool. Use a simpler alternative or explain \
                         the failure to the user.",
                        strikes_after,
                    )
                }
            } else if truncated_by_limit {
                "Tool call was not executed: the response hit the output token limit, \
                 so its arguments may be truncated. Re-issue the tool call with complete arguments."
                    .to_string()
            } else if has_truncated {
                "Tool call was not executed: the provider truncated the argument stream \
                 mid-generation, so its arguments are incomplete. Re-issue the tool call — \
                 emit ONE tool call per turn, or shorten long string fields (e.g. message, \
                 system_prompt) so the full arguments fit in a single response."
                    .to_string()
            } else {
                "Tool call was not executed: it carried an empty argument object `{}`, but \
                 this tool requires fields. Re-issue it with every required field filled in \
                 — never send a bare `{}` placeholder. If you need several calls, emit each \
                 one with complete arguments."
                    .to_string()
            };

            if has_valid {
                // Build pre-resolved results for invalid calls (guidance text).
                // `handle_tool_calls` pushes ONE assistant message with ALL
                // tool_calls, then these results, then executes the valid ones —
                // every tool_call_id gets exactly one result in order.
                let pre_resolved: Vec<(String, String)> = invalid_indices
                    .iter()
                    .map(|&i| (tool_calls[i].0.clone(), guidance.clone()))
                    .collect();
                self.handle_tool_calls(
                    session_id,
                    &tool_calls,
                    &pre_resolved,
                    event_rx,
                    on_event,
                    &reasoning_text,
                    &full_text,
                )
                .await?;
                // Valid calls executed — reset truncation counter.
                self.core
                    .with_session_mut(session_id, |session| {
                        session.run_state.truncation_strikes = 0;
                    })
                    .await?;
            } else {
                // All calls invalid — push assistant + guidance results directly.
                self.core
                    .with_session_mut(session_id, |session| {
                        session.push_assistant_tool_calls(
                            &tool_calls,
                            Some(reasoning_text),
                            Some(full_text),
                        );
                        for &i in &invalid_indices {
                            session.push_tool_result(&tool_calls[i].0, &guidance);
                        }
                        session.run_state.record_truncation();
                    })
                    .await?;
            }
            return Ok(TurnFlow::Continue);
        }

        tracing::info!(
            session_id = session_id.id,
            turn = turn_count,
            tool_count = tool_calls.len(),
            "handling tool calls"
        );
        // 账+数 in one write (D5+OV-11): the step and the counting state that
        // goes with it land on the journal together.
        let run_state = self
            .core
            .with_session_mut(session_id, |session| session.run_state.clone())
            .await?;
        self.core.event_bus.emit(RuntimeEvent::Checkpoint {
            session_id: session_id.clone(),
            checkpoint: Box::new(CheckpointData::new(
                session_id.clone(),
                user_input.to_string(),
                CheckpointStep::BeforeActions {
                    actions: triples_to_actions(&tool_calls),
                },
                turn_count,
                ActionFormat::ToolCall,
                run_state,
            )),
            agent_id: None,
            trace_id: None,
        });

        let tool_start = std::time::Instant::now();
        let tool_call_count = tool_calls.len() as u32;
        let tool_names: Vec<String> = tool_calls.iter().map(|(_, name, _)| name.clone()).collect();

        let batch_result = self
            .handle_tool_calls(
                session_id,
                &tool_calls,
                &[],
                event_rx,
                on_event.clone(),
                &reasoning_text,
                &full_text,
            )
            .await;

        // Per-call failures are outcomes (D9); the flow path still runs the
        // legacy recovery on the first one. Orchestrate-level errors (approval
        // denial, cancellation) arrive the same way they always did.
        let (observation, first_failure) = match batch_result {
            Ok(batch) => (Some(batch.observation), batch.first_failure),
            Err(e) => (None, Some(e)),
        };

        if let Some(e) = first_failure {
            let tool_duration_ms = tool_start.elapsed().as_millis() as u64;
            let error_msg = e.to_string();
            if let Some(outcome) = self
                .handle_tool_error(session_id, &tool_calls, e, event_rx, on_event)
                .await?
            {
                self.core
                    .fire_turn_end(TurnEndCtx {
                        ttft_ms: metrics.ttft_ms,
                        llm_duration_ms: metrics.llm_duration_ms,
                        tool_duration_ms,
                        usage: metrics.usage,
                        text_length: metrics.text_len,
                        has_thinking: metrics.has_thinking,
                        thinking_bytes: metrics.thinking_bytes,
                        tool_call_count,
                        tools_used: &tool_names,
                        tool_failed: tool_call_count,
                        error_message: Some(&error_msg),
                        llm_calls: 1,
                        ..TurnEndCtx::new(
                            session_id,
                            turn_count,
                            turn_start,
                            model,
                            user_input,
                            RunOutcome::Failed {
                                error: error_msg.clone(),
                            },
                        )
                    })
                    .await;
                return Ok(TurnFlow::Done(outcome));
            }
            // Retry: record metrics for the failed attempt
            self.core
                .fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    tool_duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    thinking_bytes: metrics.thinking_bytes,
                    tool_call_count,
                    tools_used: &tool_names,
                    tool_failed: tool_call_count,
                    error_message: Some(&error_msg),
                    llm_calls: 1,
                    ..TurnEndCtx::new(
                        session_id,
                        turn_count,
                        turn_start,
                        model,
                        user_input,
                        RunOutcome::Failed {
                            error: error_msg.clone(),
                        },
                    )
                })
                .await;
            return Ok(TurnFlow::Continue);
        }

        let tool_duration_ms = tool_start.elapsed().as_millis() as u64;
        self.core
            .fire_turn_end(TurnEndCtx {
                ttft_ms: metrics.ttft_ms,
                llm_duration_ms: metrics.llm_duration_ms,
                tool_duration_ms,
                usage: metrics.usage,
                text_length: metrics.text_len,
                has_thinking: metrics.has_thinking,
                thinking_bytes: metrics.thinking_bytes,
                tool_call_count,
                tools_used: &tool_names,
                tool_success: tool_call_count,
                llm_calls: 1,
                ..TurnEndCtx::new(
                    session_id,
                    turn_count,
                    turn_start,
                    model,
                    user_input,
                    RunOutcome::Completed,
                )
            })
            .await;
        tracing::info!(
            session_id = session_id.id,
            turn = turn_count,
            "tool calls done, continuing loop"
        );
        let n = tool_calls.len();
        // One counting entry point (D5): record_actions bumps both the turn
        // and the session total. The AfterActions line co-writes the post-bump
        // state in the same closure, so the 账 on the line can never disagree
        // with the 数 (crash-window consistency, OV-11).
        let run_state = self
            .core
            .with_session_mut(session_id, |session| {
                session.run_state.record_actions(n);
                session.run_state.clone()
            })
            .await?;
        self.core.event_bus.emit(RuntimeEvent::Checkpoint {
            session_id: session_id.clone(),
            checkpoint: Box::new(CheckpointData::new(
                session_id.clone(),
                user_input.to_string(),
                CheckpointStep::AfterActions {
                    actions: triples_to_actions(&tool_calls),
                    observation,
                },
                turn_count,
                ActionFormat::ToolCall,
                run_state,
            )),
            agent_id: None,
            trace_id: None,
        });
        Ok(TurnFlow::Continue)
    }

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
                        session.push_message(MessageRole::User, error_history);
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
                        session.push_message(MessageRole::User, retry_prompt);
                    })
                    .await?;
                Ok(None)
            }
        }
    }

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
    pub(super) async fn replay_batch(
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

/// Four-state classification of one turn (D4+OV-7). State-free: pure over the
/// provider envelope.
fn classify_turn(turn: &LlmTurnResult) -> ParseOutcome {
    let actions = triples_to_actions(&flatten_envelope(turn));
    if turn.is_tool_call {
        if actions.is_empty() {
            // Tool intent signalled but nothing materialized.
            ParseOutcome::Incomplete
        } else {
            ParseOutcome::Actions(actions)
        }
    } else if !turn.full_text.is_empty() {
        let finish = FinishReason::from_raw(turn.finish_reason.as_deref());
        // Intent without a call: truncated finish or a ToolUse mislabel —
        // the old finish-anomaly branch.
        if finish.is_truncated() || matches!(finish, FinishReason::ToolUse) {
            ParseOutcome::Malformed
        } else {
            ParseOutcome::Empty
        }
    } else {
        ParseOutcome::Empty
    }
}

/// Flatten the provider envelope's tool calls into backend triples.
fn flatten_envelope(turn: &LlmTurnResult) -> Vec<(String, String, String)> {
    turn.tool_calls
        .iter()
        .map(|tc| {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("")
                .to_string();
            (id, name, args)
        })
        .collect()
}

/// Names of the tools that actually failed in a batch of tool calls.
///
/// Tool-level errors carry the failing tool's name on the error itself, so a single
/// bad call in a large batch yields exactly one name (not every tool in the batch,
/// which would inflate the consecutive-failure counter and trip the Stop threshold).
/// Unknown error types conservatively fall back to the full batch.
fn failing_tool_names(tool_calls: &[(String, String, String)], error: &AgentError) -> Vec<String> {
    match error {
        AgentError::ToolArgsInvalid { name, .. } => vec![name.clone()],
        AgentError::ToolExecution { name, .. } => vec![name.clone()],
        _ => tool_calls.iter().map(|(_, n, _)| n.clone()).collect(),
    }
}

/// True when a tool call's (valid) JSON arguments are actually a truncation
/// error wrapper the model echoed back from history, rather than a real call.
/// Such an object parses fine, so the structural (invalid-JSON) guard can't see
/// it, but executing it only reproduces a ToolArgsInvalid failure. Detected by
/// its distinctive marker key (plus the preview field it always carried).
fn is_truncation_wrapper(args: &Value) -> bool {
    let obj = match args.as_object() {
        Some(o) => o,
        None => return false,
    };
    let marker = obj
        .get("error")
        .and_then(Value::as_str)
        .is_some_and(|s| s == "tool_call_arguments_truncated");
    let preview = obj.contains_key("original_args_preview");
    marker || (preview && obj.contains_key("message"))
}

fn noop_event(_event: RuntimeEvent) -> AgentResult<()> {
    Ok(())
}

/// The seam face (M3). `parse` is the live four-state classifier; `execute` is
/// the observation-shaped batch core. Flow-face debt: the react loop still
/// drives turns through `run_actions` (truncation rescue, breaker strikes, and
/// run-level flow signals are turn plumbing `ExecCtx` does not yet carry —
/// T8 wired the interaction channel and its guards, T10 switches the loop to
/// this trait).
#[async_trait]
impl<'c> ActionExecutor for ToolCallExecutor<'c> {
    fn parse(&self, turn: &LlmTurnResult) -> Result<ParseOutcome, ParseError> {
        Ok(self.classify(turn))
    }

    async fn execute(
        &mut self,
        actions: Vec<Action>,
        ctx: &ExecCtx,
    ) -> Result<Observation, ExecError> {
        if actions.is_empty() {
            return Err(ExecError::NothingRan {
                reason: "empty batch".into(),
            });
        }
        // Per-action fates are outcomes (D9); ExecError is reserved for "not a
        // single action got off the ground" (D15). Interactions from inside
        // the batch ride `ctx` (T8: consent/ask_user suspend there, on the
        // run's cancel token); the batch's own cancel wiring and live event
        // streaming need run plumbing (T10) — events ride a sinkless channel
        // here.
        let _ = ctx;
        let triples = actions_to_triples(&actions);
        let (_event_tx, mut event_rx) = broadcast::channel(8);
        let on_event: Arc<Mutex<fn(RuntimeEvent) -> AgentResult<()>>> =
            Arc::new(Mutex::new(noop_event));
        match self
            .handle_tool_calls(
                &self.session_id,
                &triples,
                &[],
                &mut event_rx,
                on_event,
                "",
                "",
            )
            .await
        {
            Ok(batch) => Ok(batch.observation),
            Err(e) => Err(ExecError::NothingRan {
                reason: e.to_string(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failing_tool_names_isolates_single_failing_tool() {
        let tool_calls = vec![
            (
                "id1".to_string(),
                "write_file".to_string(),
                "{}".to_string(),
            ),
            (
                "id2".to_string(),
                "write_file".to_string(),
                "{}".to_string(),
            ),
            (
                "id3".to_string(),
                "write_file".to_string(),
                "{}".to_string(),
            ),
        ];

        // One bad call in a batch of three must yield one name, not three.
        let err = AgentError::ToolArgsInvalid {
            name: "write_file".to_string(),
            raw: "truncated-json".to_string(),
        };
        assert_eq!(
            failing_tool_names(&tool_calls, &err),
            vec!["write_file".to_string()]
        );
    }

    #[test]
    fn failing_tool_names_extracts_execution_failure_name() {
        let tool_calls = vec![
            ("id1".to_string(), "bash".to_string(), "{}".to_string()),
            ("id2".to_string(), "bash".to_string(), "{}".to_string()),
        ];
        let err = AgentError::ToolExecution {
            name: "bash".to_string(),
            source: Box::new(AgentError::internal("boom")),
        };
        assert_eq!(
            failing_tool_names(&tool_calls, &err),
            vec!["bash".to_string()]
        );
    }

    #[test]
    fn failing_tool_names_falls_back_to_full_batch_for_unknown_error() {
        let tool_calls = vec![
            ("id1".to_string(), "a".to_string(), "{}".to_string()),
            ("id2".to_string(), "b".to_string(), "{}".to_string()),
        ];
        let err = AgentError::internal("unexpected");
        assert_eq!(
            failing_tool_names(&tool_calls, &err),
            vec!["a".to_string(), "b".to_string()]
        );
    }

    /// Four-state verdict of the envelope classifier (D4+OV-7 mapping).
    #[test]
    fn classify_maps_envelope_states() {
        let mut turn = LlmTurnResult {
            full_text: String::new(),
            reasoning_text: String::new(),
            is_tool_call: true,
            tool_calls: vec![serde_json::json!({
                "id": "call_1", "name": "echo", "arguments": "{\"x\":1}"
            })],
            usage: None,
            finish_reason: Some("tool_calls".into()),
            ttft_ms: 1,
            llm_duration_ms: 2,
            reasoning_only: false,
            thinking_signature: None,
        };

        // Calls materialized → Actions.
        let out = classify_turn(&turn);
        assert!(matches!(out, ParseOutcome::Actions(ref a) if a.len() == 1 && a[0].id == "call_1"));

        // Tool intent, nothing parsed → Incomplete.
        turn.tool_calls = vec![];
        assert_eq!(classify_turn(&turn), ParseOutcome::Incomplete);

        // Truncated text without call flag → Malformed.
        turn.is_tool_call = false;
        turn.full_text = "partial...".into();
        turn.finish_reason = Some("length".into());
        assert_eq!(classify_turn(&turn), ParseOutcome::Malformed);

        // ToolUse mislabel with text → Malformed.
        turn.finish_reason = Some("tool_use".into());
        assert_eq!(classify_turn(&turn), ParseOutcome::Malformed);

        // Clean text → Empty (the loop's text path).
        turn.finish_reason = Some("stop".into());
        assert_eq!(classify_turn(&turn), ParseOutcome::Empty);

        // Empty text wins over the finish reason → Empty (old empty branch).
        turn.full_text = String::new();
        turn.finish_reason = Some("length".into());
        assert_eq!(classify_turn(&turn), ParseOutcome::Empty);
    }
}

#[cfg(test)]
mod parse_fixtures;
