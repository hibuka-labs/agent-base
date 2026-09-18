use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use crate::engine::middleware::UserMessageCtx;
use crate::engine::runtime::plan_runner::RuntimeCore;
use crate::types::{
    AgentResult, CheckpointData, CheckpointStep, MessageRole, RunOutcome, RuntimeEvent, SessionId,
};

/// Drain events from a `broadcast::Receiver` and invoke a `Mutex<FnMut>` callback
/// for each one.
pub(super) fn drain_locked<F>(
    event_rx: &mut broadcast::Receiver<RuntimeEvent>,
    on_event: &Mutex<F>,
) -> AgentResult<()>
where
    F: FnMut(RuntimeEvent) -> AgentResult<()>,
{
    let events: Vec<RuntimeEvent> = {
        let mut buf = Vec::new();
        loop {
            match event_rx.try_recv() {
                Ok(ev) => buf.push(ev),
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                    tracing::warn!(skipped = n, "EventBus consumer lagged, events dropped");
                    continue;
                }
                Err(broadcast::error::TryRecvError::Closed) => break,
            }
        }
        buf
    };
    if !events.is_empty()
        && let Ok(mut cb) = on_event.lock()
    {
        for ev in events {
            cb(ev)?;
        }
    }
    Ok(())
}

impl RuntimeCore {
    pub async fn run<F>(&self, session_id: SessionId, on_event: F) -> AgentResult<RunOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        // Reset cancel token for this run
        self.reset_cancel();

        let span = tracing::info_span!("agent_run", session_id = session_id.id);
        let _enter = span.enter();

        let mut event_rx = self.event_bus.subscribe();
        let on_event = Arc::new(Mutex::new(on_event));

        if let Err(e) = self.validate_session(&session_id).await {
            tracing::warn!(session_id = session_id.id, error = %e, "session validation failed");
            self.event_bus.emit(RuntimeEvent::RunFinished {
                session_id: session_id.clone(),
                agent_id: None,
                trace_id: None,
            });
            drain_locked(&mut event_rx, &on_event)?;
            return Err(e);
        }

        tracing::debug!(session_id = session_id.id, "agent run start");
        let user_input_owned = self
            .with_session_mut(&session_id, |session| {
                session
                    .chat_messages()
                    .last()
                    .and_then(|m| match m {
                        crate::types::ChatMessage::User { content, .. } => Some(content.clone()),
                        _ => None,
                    })
                    .unwrap_or_default()
            })
            .await?;

        // Apply user message middleware (same as run_turn)
        let user_input_owned = self
            .apply_user_message_mw(&session_id, user_input_owned)
            .await?;

        // Reset run state for the new turn
        self.with_session_mut(&session_id, |session| {
            session.run_state.reset_for_new_run();
        })
        .await?;

        let all_user_inputs = self.collect_user_inputs(&session_id).await?;

        let result = self
            .run_react_loop(
                &session_id,
                &user_input_owned,
                0,
                &mut event_rx,
                on_event.clone(),
                &all_user_inputs,
            )
            .await;

        // Emit RunCancelled if cancelled, RunFinished otherwise (same as run_turn)
        match &result {
            Ok((RunOutcome::Cancelled, _)) => {
                if let Ok(mut cb) = on_event.lock() {
                    cb(RuntimeEvent::RunCancelled {
                        session_id: session_id.clone(),
                        agent_id: None,
                        trace_id: None,
                    })?;
                }
            }
            Err(e) if e.is_cancelled() => {
                if let Ok(mut cb) = on_event.lock() {
                    cb(RuntimeEvent::RunCancelled {
                        session_id: session_id.clone(),
                        agent_id: None,
                        trace_id: None,
                    })?;
                }
            }
            _ => {
                self.event_bus.emit(RuntimeEvent::RunFinished {
                    session_id: session_id.clone(),
                    agent_id: None,
                    trace_id: None,
                });
                drain_locked(&mut event_rx, &on_event)?;
            }
        }

        // Turn 结束：清理临时消息（包括错误路径，避免 ephemeral 残留到持久化）
        if let Err(e) = self
            .with_session_mut(&session_id, |session| {
                session.remove_ephemeral_messages();
            })
            .await
        {
            tracing::warn!(error = %e, "failed to clean up ephemeral messages");
        }

        let (outcome, _turn_count) = result?;
        Ok(outcome)
    }

    pub async fn run_turn<F>(
        &self,
        session_id: SessionId,
        user_input: &str,
        on_event: F,
    ) -> AgentResult<RunOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        self.run_turn_impl(session_id, user_input, false, on_event)
            .await
    }

    /// Like `run_turn`, but the user input is pushed as an **ephemeral**
    /// message: the LLM sees it for this turn only, then the existing
    /// turn-end cleanup removes it from memory and persistence. Used for
    /// skill-body injection. Note: the turn leaves NO trace in conversation
    /// history — neither the body nor any command (hosts that need a trail
    /// keep it in their own turn log).
    pub async fn run_turn_ephemeral_input<F>(
        &self,
        session_id: SessionId,
        user_input: &str,
        on_event: F,
    ) -> AgentResult<RunOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        self.run_turn_impl(session_id, user_input, true, on_event)
            .await
    }

    async fn run_turn_impl<F>(
        &self,
        session_id: SessionId,
        user_input: &str,
        ephemeral_input: bool,
        on_event: F,
    ) -> AgentResult<RunOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        // Reset cancel token for this turn
        self.reset_cancel();

        let span = tracing::Span::current();
        let _guard = span.enter();
        tracing::info!(session_id = session_id.id, user_input = %user_input, "agent turn start");
        drop(_guard);

        let mut event_rx = self.event_bus.subscribe();
        let on_event = Arc::new(Mutex::new(on_event));

        let user_input_owned = match self
            .apply_user_message_mw(&session_id, user_input.to_string())
            .await
        {
            Ok(u) => u,
            Err(e) => {
                let _ = on_event.lock().unwrap()(RuntimeEvent::RunFinished {
                    session_id: session_id.clone(),
                    agent_id: None,
                    trace_id: None,
                });
                return Err(e);
            }
        };

        // Reset nudge_count, turn_tool_calls, and run_tool_calls for the new turn
        if let Err(e) = self
            .with_session_mut(&session_id, |session| {
                session.run_state.reset_for_new_run();
            })
            .await
        {
            let _ = on_event.lock().unwrap()(RuntimeEvent::RunFinished {
                session_id: session_id.clone(),
                agent_id: None,
                trace_id: None,
            });
            return Err(e);
        }

        if let Err(e) = self
            .with_session_mut(&session_id, |session| {
                if ephemeral_input {
                    session.push_message_ephemeral(MessageRole::User, &user_input_owned);
                } else {
                    session.push_message(MessageRole::User, &user_input_owned);
                }
            })
            .await
        {
            let _ = on_event.lock().unwrap()(RuntimeEvent::RunFinished {
                session_id: session_id.clone(),
                agent_id: None,
                trace_id: None,
            });
            return Err(e);
        }

        self.event_bus.emit(RuntimeEvent::Checkpoint {
            session_id: session_id.clone(),
            checkpoint: CheckpointData {
                session_id: session_id.clone(),
                user_input: user_input_owned.clone(),
                step: CheckpointStep::AfterUserInput,
                turn_count: 0,
            },
            agent_id: None,
            trace_id: None,
        });

        tracing::info!(
            session_id = session_id.id,
            "run_turn: entering run_react_loop"
        );
        let all_user_inputs = self.collect_user_inputs(&session_id).await?;
        let result = self
            .run_react_loop(
                &session_id,
                &user_input_owned,
                0,
                &mut event_rx,
                on_event.clone(),
                &all_user_inputs,
            )
            .await;

        // Emit RunCancelled event if cancelled
        match &result {
            Ok((RunOutcome::Cancelled, _)) => {
                if let Ok(mut cb) = on_event.lock() {
                    cb(RuntimeEvent::RunCancelled {
                        session_id: session_id.clone(),
                        agent_id: None,
                        trace_id: None,
                    })?;
                }
            }
            Err(e) if e.is_cancelled() => {
                if let Ok(mut cb) = on_event.lock() {
                    cb(RuntimeEvent::RunCancelled {
                        session_id: session_id.clone(),
                        agent_id: None,
                        trace_id: None,
                    })?;
                }
            }
            _ => {}
        }

        // Clean up ephemeral messages
        if let Err(e) = self
            .with_session_mut(&session_id, |session| {
                session.remove_ephemeral_messages();
            })
            .await
        {
            tracing::warn!(error = %e, "failed to clean up ephemeral messages");
        }

        // Emit RunFinished on any non-cancelled error so event listeners
        // know the turn ended.
        let (outcome, turn_count) = match result {
            Ok((RunOutcome::Cancelled, _)) => return Ok(RunOutcome::Cancelled),
            Ok(tuple) => {
                // The loop no longer emits RunFinished on completion; emit it
                // here once so listeners see a single terminal event.
                self.event_bus.emit(RuntimeEvent::RunFinished {
                    session_id: session_id.clone(),
                    agent_id: None,
                    trace_id: None,
                });
                drain_locked(&mut event_rx, &on_event)?;
                tuple
            }
            Err(e) if e.is_cancelled() => {
                return Err(e);
            }
            Err(e) => {
                let _ = on_event.lock().unwrap()(RuntimeEvent::RunFinished {
                    session_id: session_id.clone(),
                    agent_id: None,
                    trace_id: None,
                });
                return Err(e);
            }
        };

        tracing::info!(
            session_id = session_id.id,
            turn_count,
            "agent turn completed"
        );
        Ok(outcome)
    }

    pub async fn run_turn_collect(
        &self,
        session_id: SessionId,
        user_input: &str,
    ) -> AgentResult<(Vec<RuntimeEvent>, RunOutcome)> {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();
        let outcome = self
            .run_turn(session_id, user_input, move |event| {
                events_clone.lock().unwrap().push(event);
                Ok(())
            })
            .await?;
        let events = Arc::try_unwrap(events).unwrap().into_inner().unwrap();
        Ok((events, outcome))
    }

    #[allow(dead_code)]
    pub async fn resume_from_checkpoint<F>(
        &self,
        checkpoint: CheckpointData,
        on_event: F,
    ) -> AgentResult<RunOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        // Reset cancel token for this resume
        self.reset_cancel();

        let session_id = checkpoint.session_id.clone();
        let user_input = checkpoint.user_input.clone();
        let turn_count = checkpoint.turn_count;

        tracing::info!(session_id = session_id.id, turn_count, step = ?checkpoint.step, "resuming from checkpoint");

        let mut event_rx = self.event_bus.subscribe();
        let on_event = Arc::new(Mutex::new(on_event));

        if let CheckpointStep::BeforeToolCalls { tool_calls } = checkpoint.step {
            match self
                .handle_tool_calls(
                    &session_id,
                    &tool_calls,
                    &mut event_rx,
                    on_event.clone(),
                    "",
                    "",
                )
                .await
            {
                Ok(()) => {}
                Err(e) => {
                    if let Some(outcome) = self
                        .handle_tool_error(
                            &session_id,
                            &tool_calls,
                            e,
                            &mut event_rx,
                            on_event.clone(),
                        )
                        .await?
                    {
                        return Ok(outcome);
                    }
                }
            }
        }

        let all_user_inputs = self.collect_user_inputs(&session_id).await?;

        let result = self
            .run_react_loop(
                &session_id,
                &user_input,
                turn_count,
                &mut event_rx,
                on_event.clone(),
                &all_user_inputs,
            )
            .await;

        // Emit RunCancelled if cancelled, same as run_turn
        match &result {
            Ok((RunOutcome::Cancelled, _)) => {
                if let Ok(mut cb) = on_event.lock() {
                    cb(RuntimeEvent::RunCancelled {
                        session_id: session_id.clone(),
                        agent_id: None,
                        trace_id: None,
                    })?;
                }
            }
            Err(e) if e.is_cancelled() => {
                if let Ok(mut cb) = on_event.lock() {
                    cb(RuntimeEvent::RunCancelled {
                        session_id: session_id.clone(),
                        agent_id: None,
                        trace_id: None,
                    })?;
                }
            }
            _ => {}
        }

        // Turn 结束：清理临时消息（包括错误路径）
        if let Err(e) = self
            .with_session_mut(&session_id, |session| {
                session.remove_ephemeral_messages();
            })
            .await
        {
            tracing::warn!(error = %e, "failed to clean up ephemeral messages");
        }

        let (outcome, _final_turn_count) = result?;
        Ok(outcome)
    }

    async fn apply_user_message_mw(
        &self,
        session_id: &SessionId,
        user_input: String,
    ) -> AgentResult<String> {
        let mut ctx = UserMessageCtx {
            session_id: session_id.clone(),
            user_input,
        };
        for mw in &self.middlewares {
            mw.on_user_message(&mut ctx).await?;
        }
        Ok(ctx.user_input)
    }

    /// Managed run with follow-up queue support (P2).
    ///
    /// After the inner turn loop completes naturally (text response, no tool calls),
    /// follow-up messages are drained and a new inner loop is started. This repeats
    /// until no more follow-up messages are queued.
    pub(crate) async fn run_managed<F>(
        &self,
        session_id: SessionId,
        user_input: &str,
        on_event: F,
    ) -> AgentResult<RunOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        // Reset cancel token for this managed run
        self.reset_cancel();

        let span = tracing::info_span!("agent_managed_run", session_id = session_id.id);
        let _enter = span.enter();

        let mut event_rx = self.event_bus.subscribe();
        let on_event = Arc::new(Mutex::new(on_event));

        if let Err(e) = self.validate_session(&session_id).await {
            tracing::warn!(session_id = session_id.id, error = %e, "session validation failed");
            self.event_bus.emit(RuntimeEvent::RunFinished {
                session_id: session_id.clone(),
                agent_id: None,
                trace_id: None,
            });
            drain_locked(&mut event_rx, &on_event)?;
            return Err(e);
        }

        tracing::debug!(session_id = session_id.id, "managed run start");

        // Push initial user message
        self.with_session_mut(&session_id, |session| {
            session.push_message(MessageRole::User, user_input);
        })
        .await?;

        // Apply user message middleware
        let user_input_owned = self
            .apply_user_message_mw(&session_id, user_input.to_string())
            .await?;

        // Reset run state for the new turn
        self.with_session_mut(&session_id, |session| {
            session.run_state.reset_for_new_run();
        })
        .await?;

        // Outer follow-up loop
        let mut current_input = user_input_owned;
        let mut final_outcome;
        let mut total_turns = 0u32;

        let config = self.config_snapshot_async().await;
        let max_turns = config
            .execution
            .max_turns
            .unwrap_or(crate::engine::runtime::DEFAULT_MAX_TURNS);

        loop {
            // Guard against unbounded execution: if follow-up messages
            // keep arriving, break out when the cumulative turn budget is
            // exhausted.
            if total_turns >= max_turns {
                tracing::warn!(
                    session_id = session_id.id,
                    total_turns,
                    max_turns,
                    "managed run: global turn cap reached"
                );
                final_outcome = RunOutcome::MaxTurnsExceeded { turns: total_turns };
                break;
            }

            // Run inner turn loop (steering drained at each iteration inside)
            let all_user_inputs = self.collect_user_inputs(&session_id).await?;
            let result = self
                .run_react_loop(
                    &session_id,
                    &current_input,
                    total_turns,
                    &mut event_rx,
                    on_event.clone(),
                    &all_user_inputs,
                )
                .await;

            // Emit RunCancelled if cancelled
            let is_cancelled = matches!(&result, Err(e) if e.is_cancelled());
            if is_cancelled {
                if let Ok(mut cb) = on_event.lock() {
                    cb(RuntimeEvent::RunCancelled {
                        session_id: session_id.clone(),
                        agent_id: None,
                        trace_id: None,
                    })?;
                }
                self.cleanup_ephemeral(&session_id).await;
                // Safe: we just checked it's Err and cancelled
                return Err(result.unwrap_err());
            }

            // Emit RunCancelled for Ok(Cancelled) outcome
            if let Ok((RunOutcome::Cancelled, _)) = &result {
                if let Ok(mut cb) = on_event.lock() {
                    cb(RuntimeEvent::RunCancelled {
                        session_id: session_id.clone(),
                        agent_id: None,
                        trace_id: None,
                    })?;
                }
                self.cleanup_ephemeral(&session_id).await;
                return Ok(RunOutcome::Cancelled);
            }

            let (outcome, turns) = result?;
            total_turns += turns;
            final_outcome = outcome;

            // Check for follow-up messages
            let follow_up_msgs = self.message_queue.drain_follow_up();
            if follow_up_msgs.is_empty() {
                break;
            }

            tracing::info!(
                session_id = session_id.id,
                count = follow_up_msgs.len(),
                "drained follow-up messages, starting new inner loop"
            );

            for msg in follow_up_msgs {
                self.with_session_mut(&session_id, |session| {
                    session.push_message(MessageRole::User, &msg);
                })
                .await?;
            }

            // Use an empty input for follow-up continuation (messages are already in session)
            current_input = String::new();
        }

        // Emit RunFinished
        self.event_bus.emit(RuntimeEvent::RunFinished {
            session_id: session_id.clone(),
            agent_id: None,
            trace_id: None,
        });
        drain_locked(&mut event_rx, &on_event)?;

        // Clean up ephemeral messages
        self.cleanup_ephemeral(&session_id).await;

        Ok(final_outcome)
    }

    /// Collect all user messages from the session (oldest-first).
    async fn collect_user_inputs(&self, session_id: &SessionId) -> AgentResult<Vec<String>> {
        self.with_session_mut(session_id, |session| {
            session
                .chat_messages()
                .iter()
                .filter_map(|m| match m {
                    crate::types::ChatMessage::User { content, .. } => Some(content.clone()),
                    _ => None,
                })
                .collect()
        })
        .await
    }

    /// Remove ephemeral messages from the session, best-effort (warn on failure).
    async fn cleanup_ephemeral(&self, session_id: &SessionId) {
        if let Err(e) = self
            .with_session_mut(session_id, |session| {
                session.remove_ephemeral_messages();
            })
            .await
        {
            tracing::warn!(error = %e, "failed to clean up ephemeral messages");
        }
    }
}
