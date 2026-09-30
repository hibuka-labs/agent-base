use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use tokio::sync::broadcast;

use crate::types::{AgentResult, RuntimeEvent};

/// Internal event bus broadcasting [`RuntimeEvent`]s within the runtime.
#[derive(Clone)]
pub(crate) struct EventBus {
    sender: broadcast::Sender<RuntimeEvent>,
    /// Count of PlanUpdated events emitted since last reset.
    plan_updates: Arc<AtomicU32>,
    /// Count of AwaitingApproval events emitted since last reset.
    approval_count: Arc<AtomicU32>,
}

/// Is this bus event a self-echo of a [`crate::types::UserEvent`] the run has
/// already delivered to `on_event`?
///
/// Dual-write contract: run-local `UserEvent` publishers
/// ([`crate::engine::middleware::PreLlmCtx::emit`],
/// [`crate::tool::ToolContext::emit_user_event`], the notice pump) deliver to
/// `on_event` directly **and** emit a copy to the event bus so external
/// subscribers (bridge remote clients, persistence) can see it. Those bus
/// copies carry `agent_id: None` and must never loop back into `on_event`.
///
/// `agent_id: Some(_)` marks events bridged from a child agent (the
/// multi-agent bridge re-publishes them via `Runtime::emit_event` with
/// `with_agent_id`): they have no direct path into the parent's `on_event`
/// and must be delivered on loopback.
pub(crate) fn is_self_echo_user_event(event: &RuntimeEvent) -> bool {
    matches!(event, RuntimeEvent::UserEvent { agent_id: None, .. })
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            plan_updates: Arc::new(AtomicU32::new(0)),
            approval_count: Arc::new(AtomicU32::new(0)),
        }
    }

    pub fn subscribe(&self) -> broadcast::Receiver<RuntimeEvent> {
        self.sender.subscribe()
    }

    pub fn emit(&self, event: RuntimeEvent) {
        match &event {
            RuntimeEvent::PlanUpdated { .. } => {
                self.plan_updates.fetch_add(1, Ordering::Relaxed);
            }
            RuntimeEvent::AwaitingApproval { .. } => {
                self.approval_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
        let _ = self.sender.send(event);
    }

    /// Take and reset the plan-update counter for the current turn.
    pub fn take_plan_updates(&self) -> u32 {
        self.plan_updates.swap(0, Ordering::Relaxed)
    }

    /// Take and reset the approval counter for the current turn.
    pub fn take_approval_count(&self) -> u32 {
        self.approval_count.swap(0, Ordering::Relaxed)
    }

    /// Drain pending events from the broadcast receiver, forwarding each
    /// [`RuntimeEvent`] to the callback.
    ///
    /// Run-local `UserEvent` self-echoes (see [`is_self_echo_user_event`]) are
    /// dropped: the publisher already rendered them to `on_event`, and the bus
    /// copy is for external subscribers only.
    pub fn drain_async_events<F>(
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: &mut F,
    ) -> AgentResult<()>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + ?Sized,
    {
        loop {
            match event_rx.try_recv() {
                Ok(event) => {
                    if is_self_echo_user_event(&event) {
                        continue;
                    }
                    on_event(event)?;
                }
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                    tracing::warn!(skipped = n, "EventBus consumer lagged, events dropped");
                    continue;
                }
                Err(broadcast::error::TryRecvError::Closed) => break,
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{SessionId, UserEvent};

    fn progress_event(agent_id: Option<&str>) -> RuntimeEvent {
        RuntimeEvent::UserEvent {
            session_id: SessionId::new(1),
            event: UserEvent::Progress {
                text: "working".to_string(),
            },
            agent_id: agent_id.map(str::to_string),
            trace_id: None,
        }
    }

    // Dual-write contract: a run-local UserEvent (`agent_id: None`) was already
    // rendered at the publisher — its bus copy is a self-echo and loopback must
    // drop it. A bridged child event (`agent_id: Some`) has no direct path into
    // the parent's on_event and must loop back.
    #[test]
    fn drain_async_events_skips_self_echo_user_events_but_delivers_child_ones() {
        let bus = EventBus::new(8);
        let mut rx = bus.subscribe();
        bus.emit(progress_event(None));
        bus.emit(progress_event(Some("child-a")));
        bus.emit(RuntimeEvent::RunFinished {
            session_id: SessionId::new(1),
            agent_id: None,
            trace_id: None,
        });

        let mut seen = Vec::new();
        {
            let mut cb = |ev: RuntimeEvent| -> AgentResult<()> {
                seen.push(ev);
                Ok(())
            };
            EventBus::drain_async_events(&mut rx, &mut cb).unwrap();
        }

        assert_eq!(seen.len(), 2, "got: {seen:?}");
        assert!(
            matches!(&seen[0], RuntimeEvent::UserEvent { agent_id: Some(a), .. } if a == "child-a"),
            "child UserEvent must loop back: {:?}",
            seen[0]
        );
        assert!(matches!(&seen[1], RuntimeEvent::RunFinished { .. }));
    }
}
