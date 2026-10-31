use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use crate::engine::context::ContextWindowManager;
use crate::engine::executor::ParseOutcome;
use crate::engine::middleware::{PostLlmCtx, PreLlmCtx};
use crate::engine::runtime::plan_runner::RuntimeCore;
use crate::types::{
    ActionFormat, AgentResult, CheckpointData, CheckpointStep, FinishReason, MessageRole,
    RunOutcome, RuntimeEvent, SessionId, UserEvent, default_convert_to_llm,
};

use super::entry::drain_locked;
use super::turn_end::TurnEndCtx;

/// Control-flow result of one `handle_llm_turn` call: keep looping, or
/// terminate the turn with a final outcome.
pub(super) enum TurnFlow {
    Continue,
    Done(RunOutcome),
}

pub(super) struct PostLlmMwResult {
    pub full_text: String,
    /// The action backend's verdict for this turn (M3 seam) — middleware may
    /// reshape it (demotion clears it to [`ParseOutcome::Empty`]).
    pub parse: ParseOutcome,
    pub skip_push: bool,
    pub follow_up_message: Option<String>,
}

impl RuntimeCore {
    async fn apply_pre_llm_mw<F>(
        &self,
        session_id: &SessionId,
        messages: Vec<crate::types::ChatMessage>,
        tools: Vec<serde_json::Value>,
        on_event: Arc<Mutex<F>>,
        turn_count: u32,
        max_turns: u32,
    ) -> AgentResult<(Vec<crate::types::ChatMessage>, Vec<serde_json::Value>)>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        // Build a unified emit function that sends UserEvents to both:
        // 1. on_event (real-time renderer callback)
        // 2. event_bus (persistence, event_log, checkpoint)
        let emit_fn: Box<dyn Fn(UserEvent) + Send + Sync> = {
            let eb = self.event_bus.clone();
            let sid = session_id.clone();
            let cb = on_event.clone();
            Box::new(move |ev: UserEvent| {
                // Real-time render
                if let Ok(mut cb) = cb.lock() {
                    let _ = cb(RuntimeEvent::UserEvent {
                        session_id: sid.clone(),
                        event: ev.clone(),
                        agent_id: None,
                        trace_id: None,
                    });
                }
                // Persistence
                eb.emit(RuntimeEvent::UserEvent {
                    session_id: sid.clone(),
                    event: ev,
                    agent_id: None,
                    trace_id: None,
                });
            })
        };

        let mut ctx = PreLlmCtx {
            session_id: session_id.clone(),
            messages,
            tools,
            emit_fn: Some(emit_fn),
            turn_count,
            max_turns,
        };

        for mw in &self.middlewares {
            mw.on_pre_llm(&mut ctx).await?;
        }

        // No drain needed — ctx.emit() already delivered events directly.
        Ok((ctx.messages, ctx.tools))
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn apply_post_llm_mw(
        &self,
        session_id: &SessionId,
        full_text: String,
        parse: ParseOutcome,
        available_tools: &[String],
        turn_count: u32,
        finish_reason: FinishReason,
    ) -> AgentResult<PostLlmMwResult> {
        let session = self.session_manager.session_or_err(session_id).await?;
        // Counting state (D5: 口径=动作) — all of it lives in RunState.
        let total_actions = session.run_state.total_actions;
        let nudge_count = session.run_state.nudge_count;
        let turn_actions = session.run_state.turn_actions;
        drop(session);
        let mut ctx = PostLlmCtx {
            session_id: session_id.clone(),
            full_text,
            parse,
            available_tools: available_tools.to_vec(),
            turn_count,
            total_actions,
            nudge_count,
            turn_actions,
            skip_push: false,
            follow_up_message: None,
            finish_reason,
        };
        for mw in &self.middlewares {
            mw.on_post_llm(&mut ctx).await?;
        }
        // Write back nudge_count if middleware modified it
        if ctx.nudge_count != nudge_count {
            self.with_session_mut(session_id, |session| {
                session.run_state.nudge_count = ctx.nudge_count;
            })
            .await?;
        }
        Ok(PostLlmMwResult {
            full_text: ctx.full_text,
            parse: ctx.parse,
            skip_push: ctx.skip_push,
            follow_up_message: ctx.follow_up_message,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run_react_loop<F>(
        &self,
        session_id: &SessionId,
        user_input_owned: &str,
        mut turn_count: u32,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
        all_user_inputs: &[String],
    ) -> AgentResult<(RunOutcome, u32)>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        let config = self.config_snapshot_async().await;
        let max_turns = config
            .execution
            .max_turns
            .unwrap_or(crate::engine::runtime::DEFAULT_MAX_TURNS);

        tracing::debug!(session_id = session_id.id, max_turns, "run turn loop start");

        loop {
            turn_count += 1;
            let turn_start = std::time::Instant::now();
            let model = self.llm_engine.get_provider().info().model.clone();

            // Check for cancellation at the top of each iteration
            if self.is_cancelled() {
                tracing::info!(session_id = session_id.id, "run_react_loop cancelled");
                self.fire_turn_end(TurnEndCtx::new(
                    session_id,
                    turn_count,
                    turn_start,
                    &model,
                    user_input_owned,
                    RunOutcome::Cancelled,
                ))
                .await;
                return Ok((RunOutcome::Cancelled, turn_count));
            }

            // Drain steering messages (P2) — injected mid-run by steer().
            // These are pushed as user messages and processed in this iteration.
            {
                let steering_msgs = self.message_queue.drain_steering();
                if !steering_msgs.is_empty() {
                    tracing::info!(
                        session_id = session_id.id,
                        count = steering_msgs.len(),
                        "drained steering messages"
                    );
                    for msg in steering_msgs {
                        self.with_session_mut(session_id, |session| {
                            session.push_message(MessageRole::User, &msg);
                        })
                        .await?;
                    }
                }
            }

            if turn_count > max_turns {
                tracing::warn!(
                    session_id = session_id.id,
                    turn_count,
                    max_turns,
                    "max turns exceeded"
                );
                self.fire_turn_end(TurnEndCtx {
                    error_message: Some("max turns exceeded"),
                    ..TurnEndCtx::new(
                        session_id,
                        turn_count,
                        turn_start,
                        &model,
                        user_input_owned,
                        RunOutcome::MaxTurnsExceeded { turns: turn_count },
                    )
                })
                .await;
                return Ok((
                    RunOutcome::MaxTurnsExceeded { turns: turn_count },
                    turn_count,
                ));
            }

            drain_locked(event_rx, &on_event)?;

            // Turn-boundary notice drain: a notice sent during the previous
            // turn (guard degradation at `on_turn`, ...) is delivered here —
            // before this turn emits anything — instead of waiting for the
            // terminal drain. Pure synchronous call, no select-arm wiring
            // (the fan-in loops stay untouched).
            self.drain_notices(session_id, &on_event)?;

            let turn_span =
                tracing::info_span!("turn", session_id = session_id.id, turn = turn_count);
            let _turn_guard = turn_span.enter();

            let session = self.session_manager.session_or_err(session_id).await?;
            let messages: Vec<_> = session.chat_messages().to_vec();
            let thinking_disabled = session.run_state.thinking_disabled_for_rest_of_run;

            drop(session);

            // Apply message conversion before sending to LLM.
            // Default: strip Custom messages that providers don't understand.
            let mut messages = match &self.convert_to_llm {
                Some(convert) => convert(&messages),
                None => default_convert_to_llm(&messages),
            };

            // Refresh tool definitions each iteration (supports MCP dynamic tools)
            // Respects ToolExposure: Direct always visible, Deferred conditional, Hidden never.
            let activation_ctx = crate::tool::ActivationContext {
                session_id: session_id.clone(),
                current_tools: vec![],
                workspace: std::env::current_dir().unwrap_or_default(),
            };
            let tool_definitions = self.tool_engine.definitions_filtered(&activation_ctx).await;
            let tools_for_turn = tool_definitions.clone();

            if let Some(ref ctx_mgr) = self.context_manager {
                let before = messages.len();
                ctx_mgr.trim(&mut messages);
                tracing::debug!(
                    session_id = session_id.id,
                    turn = turn_count,
                    before,
                    after = messages.len(),
                    "context trimmed"
                );
            }

            let (messages, tools_for_turn) = self
                .apply_pre_llm_mw(
                    session_id,
                    messages,
                    tools_for_turn,
                    on_event.clone(),
                    turn_count,
                    max_turns,
                )
                .await?;

            // Sweep self-echo UserEvent copies from event_rx — dual-written by
            // ctx.emit() (and other run-local publishers) whose direct on_event
            // copy was already rendered. Leaving them in the bus would cause
            // double-rendering in process_stream or orphaning on error paths
            // (e.g. LLM call fails before process_stream runs). Bridged child
            // events (agent_id: Some) are bus-only and are delivered normally.
            loop {
                match event_rx.try_recv() {
                    Ok(event) if crate::engine::runtime::is_self_echo_user_event(&event) => {
                        continue;
                    }
                    Ok(event) => {
                        // Non-echo — deliver it (e.g. Checkpoint from
                        // middleware that only went through event_bus.emit).
                        if let Ok(mut cb) = on_event.lock() {
                            cb(event)?;
                        }
                        continue;
                    }
                    Err(_) => break,
                }
            }

            // 账+数 in one write (D5+OV-11).
            let run_state = self
                .with_session_mut(session_id, |session| session.run_state.clone())
                .await?;
            self.event_bus.emit(RuntimeEvent::Checkpoint {
                session_id: session_id.clone(),
                checkpoint: Box::new(CheckpointData::new(
                    session_id.clone(),
                    user_input_owned.to_string(),
                    CheckpointStep::BeforeLlm {
                        messages: messages.clone(),
                        tools: tools_for_turn.clone(),
                    },
                    turn_count,
                    ActionFormat::ToolCall,
                    run_state,
                )),
                agent_id: None,
                trace_id: None,
            });

            tracing::info!(
                session_id = session_id.id,
                turn = turn_count,
                msg_count = messages.len(),
                tool_count = tools_for_turn.len(),
                "calling LLM"
            );

            let result = self
                .llm_call_with_retry(
                    session_id,
                    &messages,
                    &tools_for_turn,
                    &config,
                    thinking_disabled,
                    turn_count,
                    event_rx,
                    on_event.clone(),
                )
                .await;

            match self
                .handle_llm_turn(
                    session_id,
                    user_input_owned,
                    &tool_definitions,
                    turn_count,
                    turn_start,
                    &model,
                    result,
                    event_rx,
                    on_event.clone(),
                    all_user_inputs,
                    max_turns,
                )
                .await?
            {
                TurnFlow::Continue => {
                    // Inline compaction check — context may have grown after
                    // tool execution. The compactor owns ALL threshold
                    // judgment (token-budget phases live in `compact()`),
                    // so it runs every iteration and returns None when no
                    // action is needed. The previous gate (`token_count >
                    // max_message_tokens`) starved budget-based compactors:
                    // that config defaults to 128K and doubles as the
                    // single-message safety valve — it is not a compaction
                    // trigger. Registered compactors self-gate cheaply.
                    if let Some(ref compactor) = self.context_compactor {
                        let messages = {
                            let session = self.session_manager.session_or_err(session_id).await;
                            session.ok().map(|s| s.chat_messages().to_vec())
                        };
                        if let Some(msgs) = messages {
                            let token_count: usize =
                                msgs.iter().map(ContextWindowManager::message_tokens).sum();
                            if let Some(outcome) = compactor.compact(session_id, &msgs).await {
                                // One grep-able line per action kind (issue #33):
                                // a nudge append must never read as a compaction —
                                // only a real replacement may. agent-base stays
                                // strategy-free and merely relays the kind.
                                tracing::info!(
                                    session_id = session_id.id,
                                    turn = turn_count,
                                    token_count,
                                    "{}",
                                    outcome.kind.trigger_log()
                                );
                                self.with_session_mut(session_id, |session| {
                                    if let Err(e) = session.set_chat_messages(outcome.messages) {
                                        tracing::warn!(
                                            session_id = session_id.id,
                                            error = %e,
                                            "compactor output rejected: invalid message sequence, discarding"
                                        );
                                    }
                                })
                                .await?;
                                if let Some(done) = outcome.kind.completion_log() {
                                    tracing::info!(session_id = session_id.id, "{}", done);
                                }
                            }
                        }
                    }
                    continue;
                }
                TurnFlow::Done(outcome) => return Ok((outcome, turn_count)),
            }
        }
    }
}
