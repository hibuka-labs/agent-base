use std::collections::VecDeque;
use std::sync::Mutex;

/// Controls how queued messages are drained each turn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QueueMode {
    /// Drain all queued messages at once.
    #[default]
    All,
    /// Drain one message per iteration (the rest wait for the next turn).
    OneAtATime,
}

// (no separate impl Default needed — derived above)

/// Dual-queue message system for steering and follow-up messages.
///
/// Inspired by Pi's `steeringQueue` / `followUpQueue` pattern:
/// - **Steering**: messages injected mid-run, processed at the next turn.
/// - **Follow-up**: messages processed after the agent stops naturally.
///
/// `QueueMode` controls how many messages are drained per turn.
pub struct MessageQueue {
    steering: Mutex<VecDeque<String>>,
    follow_up: Mutex<VecDeque<String>>,
    mode: Mutex<QueueMode>,
}

impl MessageQueue {
    pub fn new() -> Self {
        Self {
            steering: Mutex::new(VecDeque::new()),
            follow_up: Mutex::new(VecDeque::new()),
            mode: Mutex::new(QueueMode::default()),
        }
    }

    #[allow(dead_code)]
    pub fn with_mode(mode: QueueMode) -> Self {
        Self {
            steering: Mutex::new(VecDeque::new()),
            follow_up: Mutex::new(VecDeque::new()),
            mode: Mutex::new(mode),
        }
    }

    /// Set the drain mode for both queues at runtime.
    pub fn set_mode(&self, mode: QueueMode) {
        *self.mode.lock().unwrap_or_else(|e| e.into_inner()) = mode;
    }

    /// Get the current drain mode.
    #[allow(dead_code)]
    pub fn mode(&self) -> QueueMode {
        *self.mode.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Push a steering message — will be processed at the start of the next turn.
    pub fn steer(&self, message: String) {
        self.steering
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(message);
    }

    /// Push a follow-up message — will be processed after the agent stops.
    pub fn follow_up(&self, message: String) {
        self.follow_up
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push_back(message);
    }

    /// Drain steering messages according to the current `QueueMode`.
    pub fn drain_steering(&self) -> Vec<String> {
        let mode = *self.mode.lock().unwrap_or_else(|e| e.into_inner());
        let mut queue = self.steering.lock().unwrap_or_else(|e| e.into_inner());
        match mode {
            QueueMode::All => queue.drain(..).collect(),
            QueueMode::OneAtATime => queue.pop_front().into_iter().collect(),
        }
    }

    /// Snapshot the pending steering messages in FIFO order — a host UI's
    /// source of truth for "what is still queued" (display + take-back),
    /// independent of the drain mode.
    pub fn snapshot_steering(&self) -> Vec<String> {
        self.steering
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// Drain **every** pending steering message and return them in FIFO order.
    /// Take-back and turn-end flush must grab the whole queue regardless of
    /// `QueueMode` (a one-at-a-time drain would strand the rest).
    pub fn take_steering(&self) -> Vec<String> {
        self.steering
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .drain(..)
            .collect()
    }

    /// Drain follow-up messages according to the current `QueueMode`.
    pub fn drain_follow_up(&self) -> Vec<String> {
        let mode = *self.mode.lock().unwrap_or_else(|e| e.into_inner());
        let mut queue = self.follow_up.lock().unwrap_or_else(|e| e.into_inner());
        match mode {
            QueueMode::All => queue.drain(..).collect(),
            QueueMode::OneAtATime => queue.pop_front().into_iter().collect(),
        }
    }

    /// Check if the steering queue has any pending messages.
    #[allow(dead_code)]
    pub fn has_steering(&self) -> bool {
        !self
            .steering
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── construction ──

    #[test]
    fn new_uses_default_mode() {
        let mq = MessageQueue::new();
        assert_eq!(mq.mode(), QueueMode::All);
    }

    #[test]
    fn with_mode_sets_correct_mode() {
        let mq = MessageQueue::with_mode(QueueMode::OneAtATime);
        assert_eq!(mq.mode(), QueueMode::OneAtATime);
    }

    #[test]
    fn set_mode_changes_at_runtime() {
        let mq = MessageQueue::new();
        assert_eq!(mq.mode(), QueueMode::All);
        mq.set_mode(QueueMode::OneAtATime);
        assert_eq!(mq.mode(), QueueMode::OneAtATime);
    }

    // ── steering queue ──

    #[test]
    fn steer_and_drain_all() {
        let mq = MessageQueue::new();
        mq.steer("msg1".into());
        mq.steer("msg2".into());
        mq.steer("msg3".into());

        let drained = mq.drain_steering();
        assert_eq!(drained, vec!["msg1", "msg2", "msg3"]);
    }

    #[test]
    fn steer_and_drain_one_at_a_time() {
        let mq = MessageQueue::with_mode(QueueMode::OneAtATime);
        mq.steer("msg1".into());
        mq.steer("msg2".into());
        mq.steer("msg3".into());

        assert_eq!(mq.drain_steering(), vec!["msg1"]);
        assert_eq!(mq.drain_steering(), vec!["msg2"]);
        assert_eq!(mq.drain_steering(), vec!["msg3"]);
        assert!(mq.drain_steering().is_empty());
    }

    #[test]
    fn drain_empty_steering_returns_empty() {
        let mq = MessageQueue::new();
        assert!(mq.drain_steering().is_empty());
    }

    #[test]
    fn drain_empty_steering_one_at_a_time_returns_empty() {
        let mq = MessageQueue::with_mode(QueueMode::OneAtATime);
        assert!(mq.drain_steering().is_empty());
    }

    #[test]
    fn has_steering_reports_correctly() {
        let mq = MessageQueue::new();
        assert!(!mq.has_steering());
        mq.steer("msg".into());
        assert!(mq.has_steering());
        mq.drain_steering();
        assert!(!mq.has_steering());
    }

    #[test]
    fn snapshot_steering_is_fifo_and_non_destructive() {
        let mq = MessageQueue::new();
        mq.steer("msg1".into());
        mq.steer("msg2".into());

        assert_eq!(mq.snapshot_steering(), vec!["msg1", "msg2"]);
        // A snapshot leaves the queue untouched.
        assert_eq!(mq.snapshot_steering(), vec!["msg1", "msg2"]);
        assert_eq!(mq.drain_steering(), vec!["msg1", "msg2"]);
    }

    #[test]
    fn take_steering_drains_everything_even_in_one_at_a_time_mode() {
        let mq = MessageQueue::with_mode(QueueMode::OneAtATime);
        mq.steer("msg1".into());
        mq.steer("msg2".into());
        mq.steer("msg3".into());

        // Take-back / turn-end flush must not strand the tail behind a
        // one-at-a-time drain.
        assert_eq!(mq.take_steering(), vec!["msg1", "msg2", "msg3"]);
        assert!(mq.snapshot_steering().is_empty());
    }

    // ── follow-up queue ──

    #[test]
    fn follow_up_and_drain_all() {
        let mq = MessageQueue::new();
        mq.follow_up("f1".into());
        mq.follow_up("f2".into());

        assert_eq!(mq.drain_follow_up(), vec!["f1", "f2"]);
    }

    #[test]
    fn follow_up_and_drain_one_at_a_time() {
        let mq = MessageQueue::with_mode(QueueMode::OneAtATime);
        mq.follow_up("f1".into());
        mq.follow_up("f2".into());

        assert_eq!(mq.drain_follow_up(), vec!["f1"]);
        assert_eq!(mq.drain_follow_up(), vec!["f2"]);
        assert!(mq.drain_follow_up().is_empty());
    }

    #[test]
    fn drain_empty_follow_up_returns_empty() {
        let mq = MessageQueue::new();
        assert!(mq.drain_follow_up().is_empty());
    }

    // ── independence between queues ──

    #[test]
    fn steering_and_follow_up_are_independent() {
        let mq = MessageQueue::new();
        mq.steer("s1".into());
        mq.follow_up("f1".into());

        // Draining steering does not affect follow-up
        assert_eq!(mq.drain_steering(), vec!["s1"]);
        assert_eq!(mq.drain_follow_up(), vec!["f1"]);
    }

    #[test]
    fn mode_switch_mid_queue() {
        let mq = MessageQueue::new(); // All mode
        mq.steer("s1".into());
        mq.steer("s2".into());

        // Drain all first
        assert_eq!(mq.drain_steering(), vec!["s1", "s2"]);

        // Switch to OneAtATime
        mq.set_mode(QueueMode::OneAtATime);
        mq.steer("s3".into());
        mq.steer("s4".into());
        assert_eq!(mq.drain_steering(), vec!["s3"]);
        assert_eq!(mq.drain_steering(), vec!["s4"]);
    }

    // ── thread safety (basic) ──

    #[test]
    fn message_queue_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MessageQueue>();
    }
}
