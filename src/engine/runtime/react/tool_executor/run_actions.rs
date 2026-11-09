use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::sync::broadcast;

use crate::engine::runtime::react::turn_end::TurnEndCtx;
use crate::engine::runtime::react::turn_guard::TurnMetrics;
use crate::engine::runtime::react::turn_loop::TurnFlow;
use crate::types::{
    Action, ActionFormat, AgentResult, CheckpointData, CheckpointStep, FinishReason, RunOutcome,
    RuntimeEvent,
};

use super::{ToolCallExecutor, actions_to_triples, is_truncation_wrapper, triples_to_actions};

impl<'c> ToolCallExecutor<'c> {
    /// Execute tool calls for one LLM turn: truncation guard, execution,
    /// checkpoint emission, and turn-end metrics.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::engine::runtime::react) async fn run_actions<F>(
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
            let leaked = super::super::text_toolcall::harvest(&full_text);
            if !leaked.is_empty() {
                let mut salvaged: Vec<usize> = Vec::new();
                for &i in &invalid_indices {
                    let name = tool_calls[i].1.clone();
                    let Some(candidate) = leaked.iter().find(|c| c.name == name) else {
                        continue;
                    };
                    let schema = self.core.tool_engine.tool_schema(&name).await;
                    let Some(args_json) =
                        super::super::text_toolcall::build_arguments(candidate, schema.as_ref())
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
}
