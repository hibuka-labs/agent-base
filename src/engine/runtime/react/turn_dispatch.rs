use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use crate::engine::executor::{ExecError, ParseOutcome};
use crate::engine::runtime::llm_engine::LlmTurnResult;
use crate::engine::runtime::plan_runner::RuntimeCore;
use crate::types::{
    Action, ActionFormat, ActionOutcome, AgentError, AgentResult, BatchStatus, CheckpointData,
    CheckpointStep, FinishReason, MessageRole, Observation, RunOutcome, RuntimeEvent, SessionId,
};

use super::entry::drain_locked;
use super::tool_executor::ToolCallExecutor;
use super::turn_end::TurnEndCtx;
use super::turn_guard::{TurnCtx, TurnMetrics};
use super::turn_loop::PostLlmMwResult;
use super::turn_loop::TurnFlow;

impl RuntimeCore {
    // ── Dispatch ────────────────────────────────────────────────────────

    /// Dispatch one LLM turn result: push messages, run tool calls, fire the
    /// turn-end callback, and decide whether the loop continues or the turn ends.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn handle_llm_turn<F>(
        &self,
        session_id: &SessionId,
        user_input_owned: &str,
        tool_definitions: &[serde_json::Value],
        turn_count: u32,
        turn_start: std::time::Instant,
        model: &str,
        result: AgentResult<LlmTurnResult>,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
        all_user_inputs: &[String],
        max_turns: u32,
    ) -> AgentResult<TurnFlow>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        let turn = match result {
            Ok(r) => r,
            Err(e) => {
                return self
                    .handle_llm_error(
                        session_id,
                        turn_count,
                        turn_start,
                        model,
                        user_input_owned,
                        e,
                    )
                    .await;
            }
        };

        let finish_reason = FinishReason::from_raw(turn.finish_reason.as_deref());
        tracing::info!(
            session_id = session_id.id,
            turn = turn_count,
            text_len = turn.full_text.len(),
            is_tool_call = turn.is_tool_call,
            tool_call_count = turn.tool_calls.len(),
            "LLM turn result"
        );

        let text_len = turn.full_text.len() as u64;
        let has_thinking = !turn.reasoning_text.is_empty();
        let thinking_bytes = turn.reasoning_text.len() as u64;
        // Four-state parse (M3 seam, D4+OV-7). The built-in tool-call backend
        // owns envelope turns; a text turn goes to the mounted action backend
        // (T10c) — it claims its own shape. A backend that cannot classify a
        // turn its own parse routed to it is a routing bug and fails the run
        // loudly (never a silent nudge).
        let parse = if turn.is_tool_call {
            ToolCallExecutor::new(self, session_id.clone()).classify(&turn)
        } else if let Some(backend) = &self.action_backend {
            match backend.lock().await.parse(&turn) {
                Ok(parse) => parse,
                Err(e) => {
                    return Err(AgentError::internal(format!(
                        "action backend failed to classify a text turn: {e}"
                    )));
                }
            }
        } else {
            ToolCallExecutor::new(self, session_id.clone()).classify(&turn)
        };
        let available_tools: Vec<String> = tool_definitions
            .iter()
            .filter_map(|d| {
                d.get("function")?
                    .get("name")?
                    .as_str()
                    .map(|s| s.to_string())
            })
            .collect();

        let LlmTurnResult {
            full_text,
            reasoning_text,
            usage,
            ttft_ms,
            llm_duration_ms,
            reasoning_only,
            is_tool_call: _,
            tool_calls: _,
            finish_reason: _,
            thinking_signature: _,
        } = turn;

        let turn_ctx = TurnCtx {
            session_id,
            turn_count,
            user_input: user_input_owned,
            finish_reason: &finish_reason.clone(),
            available_tools: &available_tools,
            turn_start,
            model,
            all_user_inputs,
            max_turns,
        };
        let metrics = TurnMetrics {
            ttft_ms,
            llm_duration_ms,
            usage: &usage,
            text_len,
            has_thinking,
            thinking_bytes,
        };

        if reasoning_only {
            return self.handle_reasoning_only(&turn_ctx, &metrics).await;
        }

        self.handle_post_llm_result(
            &turn_ctx,
            &metrics,
            full_text,
            reasoning_text,
            parse,
            &available_tools,
            turn_count,
            finish_reason,
            event_rx,
            on_event,
        )
        .await
    }

    // ── Branch handlers ─────────────────────────────────────────────────

    /// Reasoning-only: the model emitted thinking content but no answer and no
    /// tool call. Nudge it to commit; fail after consecutive strikes.
    async fn handle_reasoning_only(
        &self,
        turn_ctx: &TurnCtx<'_>,
        metrics: &TurnMetrics<'_>,
    ) -> AgentResult<TurnFlow> {
        let (strikes, thinking_disabled) = self
            .with_session_mut(turn_ctx.session_id, |session| {
                let strikes = session.run_state.record_reasoning_only();
                (strikes, session.run_state.thinking_disabled_for_rest_of_run)
            })
            .await?;
        if thinking_disabled {
            // The observability for the strike-3 flip lives here: RunState is
            // a pure type (agent-types stays I/O-free).
            tracing::info!(
                strikes,
                "reasoning_only_strikes reached 3, disabling thinking for rest of run"
            );
        }
        tracing::warn!(
            session_id = turn_ctx.session_id.id,
            turn = turn_ctx.turn_count,
            strikes,
            thinking_disabled,
            "reasoning-only response with tools available — nudging the model to commit"
        );

        let guard_ctx = self
            .build_guard_ctx(turn_ctx, "", true, false, false, thinking_disabled)
            .await;
        let decision = self.guard.on_turn(&guard_ctx).await;
        self.execute_guard_decision(decision, turn_ctx, metrics)
            .await
    }

    /// Normal LLM response: apply post-LLM middleware, push messages, then
    /// dispatch on the backend's four-state parse verdict (M3 seam).
    #[allow(clippy::too_many_arguments)]
    async fn handle_post_llm_result<F>(
        &self,
        turn_ctx: &TurnCtx<'_>,
        metrics: &TurnMetrics<'_>,
        full_text: String,
        reasoning_text: String,
        parse: ParseOutcome,
        available_tools: &[String],
        turn_count: u32,
        finish_reason: FinishReason,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<TurnFlow>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        let result = self
            .apply_post_llm_mw(
                turn_ctx.session_id,
                full_text,
                parse,
                available_tools,
                turn_count,
                finish_reason.clone(),
            )
            .await?;

        let PostLlmMwResult {
            full_text,
            parse,
            skip_push,
            follow_up_message,
        } = result;

        // Push text-only assistant message (when there is no action intent).
        if !skip_push && !full_text.is_empty() && !parse.has_action_intent() {
            let reasoning = reasoning_text.clone();
            self.with_session_mut(turn_ctx.session_id, |session| {
                if !reasoning.is_empty() {
                    session.push_assistant_with_reasoning(&full_text, &reasoning);
                } else {
                    session.push_message(MessageRole::Assistant, &full_text);
                }
            })
            .await?;
        }

        if let Some(follow_up) = follow_up_message {
            self.with_session_mut(turn_ctx.session_id, |session| {
                session.push_message(MessageRole::User, &follow_up);
            })
            .await?;
            return Ok(TurnFlow::Continue);
        }

        // Four-state dispatch (D4+OV-7) — behavior-identical to the old
        // is_tool_call/tool_calls chain. Empty text wins over Malformed: a
        // truncated finish with no text still takes the old empty branch.
        match parse {
            ParseOutcome::Actions(actions) if !actions.is_empty() => {
                // Tool-format actions run the tool flow (the built-in
                // backend's own protocol: truncation rescue, breaker,
                // recovery). Any other format came from the mounted action
                // backend and runs its flow (T10c).
                if actions[0].format == ActionFormat::ToolCall {
                    self.handle_tool_call_branch(
                        turn_ctx,
                        metrics,
                        actions,
                        full_text,
                        reasoning_text,
                        event_rx,
                        on_event,
                    )
                    .await
                } else {
                    self.handle_backend_action_branch(
                        turn_ctx, metrics, actions, event_rx, on_event,
                    )
                    .await
                }
            }
            ParseOutcome::Incomplete | ParseOutcome::Actions(_) => {
                self.handle_incomplete_tool_call(turn_ctx, metrics).await
            }
            ParseOutcome::Empty | ParseOutcome::Malformed => {
                if full_text.is_empty() {
                    self.handle_empty_response(turn_ctx, metrics).await
                } else {
                    self.handle_finish_anomaly(turn_ctx, metrics, &finish_reason, &full_text)
                        .await
                }
            }
        }
    }

    /// Empty response: no text, no reasoning, no tool call. Nudge the model;
    /// fail after consecutive strikes.
    async fn handle_empty_response(
        &self,
        turn_ctx: &TurnCtx<'_>,
        metrics: &TurnMetrics<'_>,
    ) -> AgentResult<TurnFlow> {
        let strikes = self
            .with_session_mut(turn_ctx.session_id, |session| {
                session.run_state.record_empty_response()
            })
            .await?;
        tracing::warn!(
            session_id = turn_ctx.session_id.id,
            turn = turn_ctx.turn_count,
            strikes,
            "empty LLM response (no text, no reasoning, no tool call)"
        );

        let guard_ctx = self
            .build_guard_ctx(turn_ctx, "", false, true, false, false)
            .await;
        let decision = self.guard.on_turn(&guard_ctx).await;
        self.execute_guard_decision(decision, turn_ctx, metrics)
            .await
    }

    /// Tool call branch: notify guard, handle RestoreThinking, then execute.
    #[allow(clippy::too_many_arguments)]
    async fn handle_tool_call_branch<F>(
        &self,
        turn_ctx: &TurnCtx<'_>,
        metrics: &TurnMetrics<'_>,
        actions: Vec<Action>,
        full_text: String,
        reasoning_text: String,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<TurnFlow>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        // 1. Notify guard about tool call
        let thinking_disabled = self
            .session_manager
            .session_or_err(turn_ctx.session_id)
            .await
            .map(|s| s.run_state.thinking_disabled_for_rest_of_run)
            .unwrap_or(false);
        let guard_ctx = self
            .build_guard_ctx(turn_ctx, "", false, false, false, thinking_disabled)
            .await;
        let decision = self.guard.on_tool_call(&guard_ctx).await;

        // 2. Handle RestoreThinking (update RunState)
        if matches!(
            decision,
            crate::engine::react_loop_guard::GuardDecision::RestoreThinking
        ) {
            self.execute_guard_decision(decision, turn_ctx, metrics)
                .await?;
        }

        // 3. Execute tools regardless of guard decision
        ToolCallExecutor::new(self, turn_ctx.session_id.clone())
            .run_actions(
                turn_ctx.user_input,
                turn_ctx.turn_count,
                turn_ctx.turn_start,
                turn_ctx.model,
                actions,
                turn_ctx.finish_reason,
                reasoning_text,
                full_text,
                metrics,
                event_rx,
                on_event,
            )
            .await
    }

    /// Mounted-backend action branch (T10c): the same turn plumbing the tool
    /// backend gets — guard notification, 账+数 checkpoints, turn-end metrics —
    /// around the backend's own `execute` and history protocol. The tool
    /// backend's truncation rescue / markup salvage / circuit breaker is its
    /// protocol and does not run here (flow-face unification stays the named
    /// T10 debt). An interrupt mid-batch leaves the batch dangling from
    /// `BeforeActions` on — resume synthesizes its fates (OV-4).
    #[allow(clippy::too_many_arguments)]
    async fn handle_backend_action_branch<F>(
        &self,
        turn_ctx: &TurnCtx<'_>,
        metrics: &TurnMetrics<'_>,
        actions: Vec<Action>,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<TurnFlow>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        let session_id = turn_ctx.session_id;
        let Some(backend) = &self.action_backend else {
            return Err(AgentError::internal(
                "non-tool actions with no action backend mounted (routing bug)",
            ));
        };

        // 1. Notify the guard about the action batch (D5 counts actions the
        //    same way calls are counted) and honor RestoreThinking — the same
        //    turn plumbing the tool branch runs.
        let thinking_disabled = self
            .session_manager
            .session_or_err(session_id)
            .await
            .map(|s| s.run_state.thinking_disabled_for_rest_of_run)
            .unwrap_or(false);
        let guard_ctx = self
            .build_guard_ctx(turn_ctx, "", false, false, false, thinking_disabled)
            .await;
        let decision = self.guard.on_tool_call(&guard_ctx).await;
        if matches!(
            decision,
            crate::engine::react_loop_guard::GuardDecision::RestoreThinking
        ) {
            self.execute_guard_decision(decision, turn_ctx, metrics)
                .await?;
        }

        let mut slot = backend.lock().await;
        // 2. Session liveness (D10): one rebuild attempt, never a loop.
        if !slot.is_alive() {
            slot.rebuild().await.map_err(|e| {
                AgentError::internal(format!("action backend session rebuild failed: {e}"))
            })?;
        }
        let format = slot.format();

        // 3. 账+数 in one write (D5+OV-11): the step and the counting state
        //    that goes with it land on the journal together.
        let run_state = self
            .with_session_mut(session_id, |session| session.run_state.clone())
            .await?;
        self.event_bus.emit(RuntimeEvent::Checkpoint {
            session_id: session_id.clone(),
            checkpoint: Box::new(CheckpointData::new(
                session_id.clone(),
                turn_ctx.user_input.to_string(),
                CheckpointStep::BeforeActions {
                    actions: actions.clone(),
                },
                turn_ctx.turn_count,
                format,
                run_state,
            )),
            agent_id: None,
            trace_id: None,
        });

        // 4. Execute against the run's ExecCtx (cancel + interaction). A
        //    user interrupt releases the batch mid-compute.
        let tool_start = std::time::Instant::now();
        let ctx = self.exec_ctx();
        let executed = tokio::select! {
            biased;
            out = slot.execute(actions.clone(), &ctx) => Some(out),
            _ = ctx.cancel.cancelled() => None,
        };
        let n = actions.len() as u32;
        let names: Vec<String> = actions.iter().map(|a| a.name.clone()).collect();
        let duration_ms = tool_start.elapsed().as_millis() as u64;
        match executed {
            None => {
                // Interrupt: nothing after BeforeActions is written, so the
                // batch stays dangling and resume reconstructs its fates (D3).
                // Same terminal shape as the tool flow's cancelled error:
                // drain, persist, ride up (RunCancelled comes from run_turn).
                drop(slot);
                drain_locked(event_rx, &on_event)?;
                let session = self.session_manager.session_or_err(session_id).await?;
                if let Err(e) = self.session_manager.session_store().save(&session).await {
                    tracing::warn!(session_id = session_id.id, error = %e, "Failed to persist session");
                    let config = self.config_snapshot_async().await;
                    if config.execution.fail_on_persist_error {
                        return Err(AgentError::internal(format!(
                            "Session persistence failed: {e}"
                        )));
                    }
                }
                Err(AgentError::Cancelled)
            }
            Some(Ok(observation)) => {
                // The backend's protocol pair, pushed verbatim (D7 — the
                // kernel never interprets history_messages).
                self.with_session_mut(session_id, |session| {
                    for msg in &observation.history_messages {
                        session.push_chat_message(msg.clone());
                    }
                })
                .await?;
                // 5. Counting + AfterActions in one write (D5+OV-11): the
                //    batch ran to fates (D15), so it counts as one batch.
                let run_state = self
                    .with_session_mut(session_id, |session| {
                        session.run_state.record_actions(actions.len());
                        session.run_state.clone()
                    })
                    .await?;
                let ok = observation.outcomes.iter().filter(|o| o.ok).count() as u32;
                self.event_bus.emit(RuntimeEvent::Checkpoint {
                    session_id: session_id.clone(),
                    checkpoint: Box::new(CheckpointData::new(
                        session_id.clone(),
                        turn_ctx.user_input.to_string(),
                        CheckpointStep::AfterActions {
                            actions,
                            observation: Some(observation),
                        },
                        turn_ctx.turn_count,
                        format,
                        run_state,
                    )),
                    agent_id: None,
                    trace_id: None,
                });
                self.fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    tool_duration_ms: duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    thinking_bytes: metrics.thinking_bytes,
                    tool_call_count: n,
                    tools_used: &names,
                    tool_success: ok,
                    tool_failed: n - ok,
                    llm_calls: 1,
                    ..TurnEndCtx::new(
                        session_id,
                        turn_ctx.turn_count,
                        turn_ctx.turn_start,
                        turn_ctx.model,
                        turn_ctx.user_input,
                        RunOutcome::Completed,
                    )
                })
                .await;
                Ok(TurnFlow::Continue)
            }
            Some(Err(ExecError::NothingRan { reason })) => {
                // D15: not one action got off the ground. The fates say so —
                // the backend renders them in its own shape — and the batch
                // still lands as a checkpoint pair, so resume replays this
                // story verbatim instead of synthesizing a second one. The
                // tool backend's consecutive-failure recovery is tool
                // protocol and does not apply here; the model reads the
                // reason and adapts.
                let fates: Vec<ActionOutcome> = actions
                    .iter()
                    .map(|a| ActionOutcome {
                        action_id: a.id.clone(),
                        name: a.name.clone(),
                        ok: false,
                        denied: false,
                        summary: format!("not run ({reason})"),
                        details: None,
                    })
                    .collect();
                let history_messages = slot.render_fates(&actions, &fates);
                drop(slot);
                self.with_session_mut(session_id, |session| {
                    for msg in &history_messages {
                        session.push_chat_message(msg.clone());
                    }
                })
                .await?;
                let run_state = self
                    .with_session_mut(session_id, |session| session.run_state.clone())
                    .await?;
                self.event_bus.emit(RuntimeEvent::Checkpoint {
                    session_id: session_id.clone(),
                    checkpoint: Box::new(CheckpointData::new(
                        session_id.clone(),
                        turn_ctx.user_input.to_string(),
                        CheckpointStep::AfterActions {
                            actions,
                            observation: Some(Observation {
                                outcomes: fates,
                                events: vec![],
                                history_messages,
                                batch_status: BatchStatus::Completed,
                            }),
                        },
                        turn_ctx.turn_count,
                        format,
                        run_state,
                    )),
                    agent_id: None,
                    trace_id: None,
                });
                self.fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    tool_duration_ms: duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    thinking_bytes: metrics.thinking_bytes,
                    tool_call_count: n,
                    tools_used: &names,
                    tool_failed: n,
                    error_message: Some(&reason),
                    llm_calls: 1,
                    ..TurnEndCtx::new(
                        session_id,
                        turn_ctx.turn_count,
                        turn_ctx.turn_start,
                        turn_ctx.model,
                        turn_ctx.user_input,
                        RunOutcome::Failed {
                            error: reason.clone(),
                        },
                    )
                })
                .await;
                Ok(TurnFlow::Continue)
            }
        }
    }

    /// Incomplete tool call: model signalled tool_use but no complete tool calls
    /// were parsed (stream cut off mid-delta, JSON incomplete, etc.).
    async fn handle_incomplete_tool_call(
        &self,
        turn_ctx: &TurnCtx<'_>,
        metrics: &TurnMetrics<'_>,
    ) -> AgentResult<TurnFlow> {
        let strikes = self
            .with_session_mut(turn_ctx.session_id, |session| {
                session.run_state.record_empty_response()
            })
            .await?;
        tracing::warn!(
            session_id = turn_ctx.session_id.id,
            turn = turn_ctx.turn_count,
            strikes,
            "model signalled tool call but no complete tool calls were parsed"
        );
        let guard_ctx = self
            .build_guard_ctx(turn_ctx, "", false, true, false, false)
            .await;
        let decision = self.guard.on_turn(&guard_ctx).await;
        self.execute_guard_decision(decision, turn_ctx, metrics)
            .await
    }

    /// Finish-reason anomaly: tool_use with no parsed tool calls, or truncated.
    async fn handle_finish_anomaly(
        &self,
        turn_ctx: &TurnCtx<'_>,
        metrics: &TurnMetrics<'_>,
        finish_reason: &FinishReason,
        full_text: &str,
    ) -> AgentResult<TurnFlow> {
        match finish_reason {
            FinishReason::ToolUse => {
                let error = "finish_reason=tool_use but no tool calls were parsed";
                tracing::warn!(
                    session_id = turn_ctx.session_id.id,
                    turn = turn_ctx.turn_count,
                    error,
                    "model signalled tool_use but no tool calls were parsed"
                );
                self.fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    thinking_bytes: metrics.thinking_bytes,
                    llm_calls: 1,
                    error_message: Some(error),
                    ..TurnEndCtx::new(
                        turn_ctx.session_id,
                        turn_ctx.turn_count,
                        turn_ctx.turn_start,
                        turn_ctx.model,
                        turn_ctx.user_input,
                        RunOutcome::Failed {
                            error: error.to_string(),
                        },
                    )
                })
                .await;
                Ok(TurnFlow::Done(RunOutcome::Failed {
                    error: error.to_string(),
                }))
            }
            FinishReason::Truncated { reason } => {
                let error = format!("response truncated ({:?})", reason);
                tracing::warn!(
                    session_id = turn_ctx.session_id.id,
                    turn = turn_ctx.turn_count,
                    ?reason,
                    "text-only response truncated by token limit"
                );
                self.fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    thinking_bytes: metrics.thinking_bytes,
                    llm_calls: 1,
                    error_message: Some(&error),
                    ..TurnEndCtx::new(
                        turn_ctx.session_id,
                        turn_ctx.turn_count,
                        turn_ctx.turn_start,
                        turn_ctx.model,
                        turn_ctx.user_input,
                        RunOutcome::Failed {
                            error: error.clone(),
                        },
                    )
                })
                .await;
                Ok(TurnFlow::Done(RunOutcome::Failed { error }))
            }
            _ => {
                // Text-only branch — use the guard to determine completion.
                let guard_ctx = self
                    .build_guard_ctx(turn_ctx, full_text, false, false, true, false)
                    .await;
                let decision = self.guard.on_turn(&guard_ctx).await;
                self.execute_guard_decision(decision, turn_ctx, metrics)
                    .await
            }
        }
    }

    /// LLM stream error: fire turn-end and persist session on cancellation.
    async fn handle_llm_error(
        &self,
        session_id: &SessionId,
        turn_count: u32,
        turn_start: std::time::Instant,
        model: &str,
        user_input_owned: &str,
        e: crate::types::AgentError,
    ) -> AgentResult<TurnFlow> {
        let stream_outcome = if e.is_cancelled() {
            RunOutcome::Cancelled
        } else {
            RunOutcome::Failed {
                error: e.to_string(),
            }
        };
        self.fire_turn_end(TurnEndCtx {
            error_message: Some(&e.to_string()),
            ..TurnEndCtx::new(
                session_id,
                turn_count,
                turn_start,
                model,
                user_input_owned,
                stream_outcome,
            )
        })
        .await;
        // Persist session on cancellation (LLM-stream path bypasses handle_tool_error)
        if e.is_cancelled()
            && let Ok(session) = self.session_manager.session_or_err(session_id).await
        {
            let _ = self.session_manager.session_store().save(&session).await;
        }
        Err(e)
    }

    // ── Guard decision execution ────────────────────────────────────────

    /// Execute a guard decision — fire events, inject nudge, or terminate.
    pub(super) async fn execute_guard_decision(
        &self,
        decision: crate::engine::react_loop_guard::GuardDecision,
        turn_ctx: &TurnCtx<'_>,
        metrics: &TurnMetrics<'_>,
    ) -> AgentResult<TurnFlow> {
        use crate::engine::react_loop_guard::GuardDecision;
        match decision {
            GuardDecision::Continue { nudge } => {
                tracing::info!(
                    session_id = turn_ctx.session_id.id,
                    turn = turn_ctx.turn_count,
                    "guard decided to continue"
                );
                self.fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    thinking_bytes: metrics.thinking_bytes,
                    llm_calls: 1,
                    ..TurnEndCtx::new(
                        turn_ctx.session_id,
                        turn_ctx.turn_count,
                        turn_ctx.turn_start,
                        turn_ctx.model,
                        turn_ctx.user_input,
                        RunOutcome::Continuing,
                    )
                })
                .await;
                if let Some(msg) = nudge {
                    self.with_session_mut(turn_ctx.session_id, |session| {
                        session.push_message(MessageRole::User, &msg);
                    })
                    .await?;
                }
                Ok(TurnFlow::Continue)
            }
            GuardDecision::Complete => {
                tracing::info!(
                    session_id = turn_ctx.session_id.id,
                    turn = turn_ctx.turn_count,
                    "guard decided task is complete"
                );
                self.fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    thinking_bytes: metrics.thinking_bytes,
                    llm_calls: 1,
                    ..TurnEndCtx::new(
                        turn_ctx.session_id,
                        turn_ctx.turn_count,
                        turn_ctx.turn_start,
                        turn_ctx.model,
                        turn_ctx.user_input,
                        RunOutcome::Completed,
                    )
                })
                .await;
                Ok(TurnFlow::Done(RunOutcome::Completed))
            }
            GuardDecision::Fail { error } => {
                tracing::warn!(
                    session_id = turn_ctx.session_id.id,
                    turn = turn_ctx.turn_count,
                    error = %error,
                    "guard decided to fail"
                );
                self.fire_guard_fail(turn_ctx, metrics, &error).await;
                Ok(TurnFlow::Done(RunOutcome::Failed { error }))
            }
            GuardDecision::DisableThinking { nudge } => {
                self.with_session_mut(turn_ctx.session_id, |session| {
                    session.run_state.thinking_disabled_for_rest_of_run = true;
                    session.push_message(MessageRole::User, &nudge);
                })
                .await?;

                self.fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    thinking_bytes: metrics.thinking_bytes,
                    llm_calls: 1,
                    ..TurnEndCtx::new(
                        turn_ctx.session_id,
                        turn_ctx.turn_count,
                        turn_ctx.turn_start,
                        turn_ctx.model,
                        turn_ctx.user_input,
                        RunOutcome::Continuing,
                    )
                })
                .await;

                tracing::info!(
                    session_id = turn_ctx.session_id.id,
                    turn = turn_ctx.turn_count,
                    "thinking disabled by guard"
                );
                Ok(TurnFlow::Continue)
            }
            GuardDecision::RestoreThinking => {
                let original = self
                    .session_manager
                    .session_or_err(turn_ctx.session_id)
                    .await
                    .map(|s| s.run_state.original_thinking_enabled)
                    .unwrap_or(false);

                self.with_session_mut(turn_ctx.session_id, |session| {
                    session.run_state.thinking_disabled_for_rest_of_run = !original;
                })
                .await?;

                tracing::info!(
                    session_id = turn_ctx.session_id.id,
                    turn = turn_ctx.turn_count,
                    thinking_enabled = original,
                    "thinking restored by guard"
                );
                Ok(TurnFlow::Continue)
            }
        }
    }
}
