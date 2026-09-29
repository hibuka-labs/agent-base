//! Run-level notice channel: NoticeHandle + engine pump.
//!
//! Engine mechanisms (guard, judge, future reaper) hold a cloneable
//! [`NoticeHandle`] and send display-only [`Notice`]s. The pump (see
//! [`RuntimeCore::drain_notices`]) stamps each notice with the current
//! session id and forwards it into the event stream as
//! `RuntimeEvent::UserEvent { event: UserEvent::Notice { .. } }`.
//!
//! Delivery is deliberately synchronous (try_recv drain when the react loop
//! completes and before each terminal event) rather than a spawned task: the
//! guard decides during the final turn of a run, so a synchronous drain
//! guarantees the notice is delivered before `RunFinished`/`RunCancelled`
//! with no task-scheduling races — and touches none of the fan-in select
//! loops. Consequence: a `Progress` notice surfaces at loop completion, not
//! in real time (turn-boundary drain points are future work).
//!
//! Lifecycle: the channel is created once per engine. Any notice still
//! pending when a new run starts is discarded (`discard_pending_notices`) so
//! it can never be stamped with a later run's session id. Sends outside a
//! run are therefore best-effort and may be dropped.

use std::sync::Mutex;

use tokio::sync::mpsc;

use crate::types::{AgentResult, Notice, NoticeKind, RuntimeEvent, SessionId, UserEvent};

use super::plan_runner::RuntimeCore;

/// Cloneable publish handle for engine-internal notices.
///
/// Publishers don't know whether a session/engine exists — sending into a
/// closed channel (run over, runtime dropped) silently discards the notice:
/// it is a best-effort display signal and must never affect execution.
#[derive(Clone)]
pub struct NoticeHandle {
    tx: mpsc::UnboundedSender<Notice>,
}

impl NoticeHandle {
    pub(crate) fn new(tx: mpsc::UnboundedSender<Notice>) -> Self {
        Self { tx }
    }

    /// Create a standalone handle + receiver pair.
    ///
    /// The runtime builds its own channel internally; this constructor exists
    /// for hosts and tests that want to drive a guard (or any publisher)
    /// outside a full engine build and observe the raw notices.
    pub fn new_channel() -> (Self, mpsc::UnboundedReceiver<Notice>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self { tx }, rx)
    }

    /// Send a notice. Never fails from the caller's perspective — a closed
    /// receiver drops the notice silently.
    ///
    /// `source` takes `impl Into<String>` so dynamic emitters ("tool:<name>")
    /// work alongside static ones ("guard").
    pub fn send(&self, kind: NoticeKind, source: impl Into<String>, text: impl Into<String>) {
        let _ = self.tx.send(Notice {
            kind,
            source: source.into(),
            text: text.into(),
        });
    }
}

impl RuntimeCore {
    /// Drain pending notices, stamping each with `session_id` and delivering
    /// it to the run's event callback as a `UserEvent::Notice`.
    ///
    /// Called after the react loop completes and before every terminal event
    /// (`RunFinished`/`RunCancelled`) — a notice is therefore always
    /// delivered before the terminal event on every exit path, Ok or Err.
    ///
    /// Display-only contract: a consumer callback error is logged and the
    /// notice dropped — it never fails the run nor blocks later notices.
    pub(crate) fn drain_notices<F>(
        &self,
        session_id: &SessionId,
        on_event: &Mutex<F>,
    ) -> AgentResult<()>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()>,
    {
        let mut rx = self.notice_rx.lock().unwrap();
        while let Ok(notice) = rx.try_recv() {
            let event = RuntimeEvent::UserEvent {
                session_id: session_id.clone(),
                event: UserEvent::Notice {
                    kind: notice.kind,
                    source: notice.source,
                    text: notice.text,
                },
                agent_id: None,
                trace_id: None,
            };
            if let Ok(mut cb) = on_event.lock()
                && let Err(e) = cb(event)
            {
                tracing::warn!(error = %e, "notice consumer callback failed — dropping notice");
            }
        }
        Ok(())
    }

    /// Drop any notices still pending from before this run.
    ///
    /// Called at run entry: the notice channel is engine-level, so without
    /// this a notice sent outside a run (or left over from an aborted one)
    /// would surface later stamped with the wrong run's session id.
    pub(crate) fn discard_pending_notices(&self) {
        let mut rx = self.notice_rx.lock().unwrap();
        while rx.try_recv().is_ok() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Design doc §9: receiver closed (run over) — send must not panic and
    // must not affect the caller.
    #[test]
    fn send_after_receiver_closed_is_silent_no_panic() {
        let (tx, rx) = mpsc::unbounded_channel::<Notice>();
        let handle = NoticeHandle::new(tx);
        drop(rx);
        handle.send(NoticeKind::Warning, "guard", "nobody listening");
        handle.send(NoticeKind::Info, "judge", "also dropped");
    }

    #[test]
    fn send_stores_kind_source_text() {
        let (tx, mut rx) = mpsc::unbounded_channel::<Notice>();
        let handle = NoticeHandle::new(tx);
        handle.send(
            NoticeKind::Warning,
            "guard",
            "judge unparsed — treating as complete",
        );
        let notice = rx.try_recv().unwrap();
        assert_eq!(notice.kind, NoticeKind::Warning);
        assert_eq!(notice.source, "guard");
        assert_eq!(notice.text, "judge unparsed — treating as complete");
    }
}
