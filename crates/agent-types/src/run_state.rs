//! Run-state pure types: [`RunState`] — the counting and strike state of a run.
//!
//! M3 D5+OV-11: this is THE counting evidence. Restores trust its persisted
//! value ("恢复计数信 RunState 持久值"); it rides in [`crate::CheckpointData`]
//! so it is co-written with the checkpoint/journal in one write ("与
//! checkpoint/journal 事务性同写") — the recorded batch and the counts can
//! never disagree across a crash window. The message-lens recount stays as
//! old-format compatibility only.
//!
//! Counting unit = **actions** (D5: 口径=动作) — tool calls today, CodeAct
//! cells tomorrow — so the fields and methods say `actions`, not `tool_calls`.
//! Old wire keys (`turn_tool_calls`, `run_has_tool_calls`, `total_tool_calls`)
//! are read through serde aliases.

use serde::{Deserialize, Serialize};

/// Run-level state tracking for the react loop.
///
/// Manages counters and flags that track the current run's progress.
/// Most fields are reset at the start of each run (when a new user message
/// arrives); [`total_actions`](Self::total_actions) and
/// [`original_thinking_enabled`](Self::original_thinking_enabled) are
/// session-lifetime and survive run boundaries.
///
/// # Backward compatibility
///
/// Before this struct existed, `nudge_count` and `turn_tool_calls` were flat
/// fields on `AgentSession` (and the session total lived in a top-level
/// `total_tool_calls`). The `RawAgentSession` deserialization shim migrates
/// them automatically; the field aliases below keep pre-T7 `run_state` wire
/// keys readable.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunState {
    /// Session-lifetime count of actions executed (across all runs).
    /// Not reset by [`reset_for_new_run`](Self::reset_for_new_run) — a session
    /// that already acted stays "not first turn" across run boundaries.
    /// Used by `ToolEnforcementMiddleware` for "first_turn_only" enforcement.
    #[serde(default, alias = "total_tool_calls")]
    pub total_actions: usize,
    /// Number of actions already executed in the current turn.
    /// Reset to 0 at the start of each turn (when a new user message arrives).
    /// Used by `TurnToolLimitMiddleware` to enforce per-turn action limits.
    #[serde(default, alias = "turn_tool_calls")]
    pub turn_actions: usize,
    /// Whether any actions ran in the current run.
    /// Reset to false at the start of each run. Used by the completion judge
    /// to determine if actions were used before a text-only response.
    #[serde(default, alias = "run_has_tool_calls")]
    pub run_has_actions: bool,
    /// Number of consecutive LLM turns that produced only reasoning_content
    /// (no text, no tool call). Reset to 0 when a normal response or tool call
    /// is produced. Used by the react loop to fail instead of looping forever
    /// on a reasoning-model runaway.
    pub reasoning_only_strikes: usize,
    /// Number of consecutive LLM turns that produced a completely empty response
    /// (no text, no reasoning, no tool call). Reset to 0 when a normal response
    /// or tool call is produced. Used by the react loop to retry a bounded
    /// number of times, then fail instead of looping forever.
    pub empty_response_strikes: usize,
    /// Number of tool-enforcement nudges issued in the current turn.
    /// Reset to 0 at the start of each turn (when a new user message arrives).
    /// Used by `ToolEnforcementMiddleware` to cap nudge attempts per turn.
    pub nudge_count: usize,
    /// Thinking is disabled for the rest of the current run.
    /// Set to true when reasoning_only_strikes reaches the maximum (3).
    /// Reset to false at the start of each new run (when a new user message arrives).
    /// This allows the model to continue without thinking after too many
    /// reasoning-only responses.
    #[serde(default)]
    pub thinking_disabled_for_rest_of_run: bool,

    // ─── New fields for thinking guard ─────────────────────────────
    /// Original thinking configuration (for restoration)
    ///
    /// Records whether thinking was originally enabled when the session started.
    /// Used to restore thinking to its original state after DisableThinking → RestoreThinking cycle.
    #[serde(default)]
    pub original_thinking_enabled: bool,

    /// Number of consecutive LLM turns where the truncation guard fired
    /// (all tool_calls had invalid/truncated arguments). Reset to 0 when
    /// at least one valid tool call executes successfully. Used to break
    /// the re-issue death spiral: after N consecutive truncations, stop
    /// retrying and fail with an actionable message.
    #[serde(default)]
    pub truncation_strikes: usize,

    /// The latest tool batch parked on a human step (a design-canvas screen
    /// awaiting a pick) and no tool ran after it. Overwritten per batch by
    /// `handle_tool_calls`. When true at the end of a run, the run is
    /// *waiting for the user* — the guard must complete instead of nudging
    /// "continue working" over their pending answer.
    #[serde(default)]
    pub awaiting_human: bool,
}

impl RunState {
    /// Reset all run-level state for a new run (when a new user message arrives).
    /// Session-lifetime state (`total_actions`, `original_thinking_enabled`)
    /// is not reset.
    pub fn reset_for_new_run(&mut self) {
        self.turn_actions = 0;
        self.run_has_actions = false;
        self.reasoning_only_strikes = 0;
        self.empty_response_strikes = 0;
        self.nudge_count = 0;
        self.thinking_disabled_for_rest_of_run = false;
        self.truncation_strikes = 0;
        self.awaiting_human = false;
        // Note: original_thinking_enabled is NOT reset here
        // It should be set once when the session starts
    }

    /// Record executed actions (branch 3: actions ran).
    /// Counts them against both the turn and the session total (D5: one
    /// counting entry point, unit = actions) and resets reasoning_only_strikes,
    /// empty_response_strikes, and truncation_strikes.
    pub fn record_actions(&mut self, n: usize) {
        self.turn_actions += n;
        self.total_actions += n;
        self.run_has_actions = true;
        self.reasoning_only_strikes = 0;
        self.empty_response_strikes = 0;
        self.truncation_strikes = 0;
    }

    /// Record reasoning-only response (branch 1).
    /// Resets empty_response_strikes.
    /// Returns the new strike count.
    /// If strikes reach 3, disables thinking for the rest of the run (the
    /// caller owns the observability for that flip — types stay I/O-free).
    pub fn record_reasoning_only(&mut self) -> usize {
        self.empty_response_strikes = 0;
        self.reasoning_only_strikes += 1;
        if self.reasoning_only_strikes >= 3 {
            self.thinking_disabled_for_rest_of_run = true;
        }
        self.reasoning_only_strikes
    }

    /// Record empty response (branch 2).
    /// Resets reasoning_only_strikes.
    /// Returns the new strike count.
    pub fn record_empty_response(&mut self) -> usize {
        self.reasoning_only_strikes = 0;
        self.empty_response_strikes += 1;
        self.empty_response_strikes
    }

    /// Record a truncation guard firing (all tool_calls had invalid arguments).
    /// Returns the new strike count. Caller should check against threshold.
    pub fn record_truncation(&mut self) -> usize {
        self.truncation_strikes += 1;
        self.truncation_strikes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_state_default() {
        let rs = RunState::default();
        assert_eq!(rs.total_actions, 0);
        assert_eq!(rs.turn_actions, 0);
        assert!(!rs.run_has_actions);
        assert_eq!(rs.reasoning_only_strikes, 0);
        assert_eq!(rs.empty_response_strikes, 0);
        assert_eq!(rs.nudge_count, 0);
    }

    #[test]
    fn run_state_reset_for_new_run() {
        let mut rs = RunState {
            total_actions: 9,
            turn_actions: 5,
            run_has_actions: true,
            reasoning_only_strikes: 2,
            empty_response_strikes: 1,
            nudge_count: 3,
            thinking_disabled_for_rest_of_run: true,
            original_thinking_enabled: true,
            truncation_strikes: 4,
            awaiting_human: true,
        };

        rs.reset_for_new_run();

        // Session-lifetime state survives the run boundary.
        assert_eq!(rs.total_actions, 9);
        assert_eq!(rs.turn_actions, 0);
        assert!(!rs.run_has_actions);
        assert_eq!(rs.reasoning_only_strikes, 0);
        assert_eq!(rs.empty_response_strikes, 0);
        assert_eq!(rs.nudge_count, 0);
        assert_eq!(rs.truncation_strikes, 0);
        assert!(!rs.thinking_disabled_for_rest_of_run);
        assert!(!rs.awaiting_human);
        // original_thinking_enabled is NOT reset
        assert!(rs.original_thinking_enabled);
    }

    #[test]
    fn run_state_record_actions() {
        let mut rs = RunState {
            reasoning_only_strikes: 2,
            empty_response_strikes: 1,
            ..RunState::default()
        };

        rs.record_actions(3);

        assert_eq!(rs.turn_actions, 3);
        assert_eq!(rs.total_actions, 3);
        assert!(rs.run_has_actions);
        assert_eq!(rs.reasoning_only_strikes, 0); // reset
        assert_eq!(rs.empty_response_strikes, 0); // reset
    }

    #[test]
    fn run_state_record_actions_accumulates() {
        let mut rs = RunState::default();
        rs.record_actions(2);
        rs.record_actions(3);

        assert_eq!(rs.turn_actions, 5);
        assert_eq!(rs.total_actions, 5);
        assert!(rs.run_has_actions);
    }

    #[test]
    fn run_state_record_reasoning_only() {
        let mut rs = RunState {
            empty_response_strikes: 2,
            ..RunState::default()
        };

        let strikes = rs.record_reasoning_only();

        assert_eq!(strikes, 1);
        assert_eq!(rs.reasoning_only_strikes, 1);
        assert_eq!(rs.empty_response_strikes, 0); // reset
    }

    #[test]
    fn run_state_record_reasoning_only_consecutive() {
        let mut rs = RunState::default();

        assert_eq!(rs.record_reasoning_only(), 1);
        assert_eq!(rs.record_reasoning_only(), 2);
        assert_eq!(rs.record_reasoning_only(), 3);
    }

    #[test]
    fn run_state_record_empty_response() {
        let mut rs = RunState {
            reasoning_only_strikes: 2,
            ..RunState::default()
        };

        let strikes = rs.record_empty_response();

        assert_eq!(strikes, 1);
        assert_eq!(rs.empty_response_strikes, 1);
        assert_eq!(rs.reasoning_only_strikes, 0); // reset
    }

    #[test]
    fn run_state_record_empty_response_consecutive() {
        let mut rs = RunState::default();

        assert_eq!(rs.record_empty_response(), 1);
        assert_eq!(rs.record_empty_response(), 2);
        assert_eq!(rs.record_empty_response(), 3);
    }

    #[test]
    fn run_state_branch_cross_reset() {
        // Simulate: reasoning only → actions → reasoning only
        let mut rs = RunState::default();

        // Branch 1: reasoning only
        rs.record_reasoning_only();
        assert_eq!(rs.reasoning_only_strikes, 1);

        // Branch 3: actions (should reset reasoning_only_strikes)
        rs.record_actions(2);
        assert_eq!(rs.reasoning_only_strikes, 0);
        assert_eq!(rs.turn_actions, 2);

        // Branch 1 again: reasoning only (should start from 1, not 2)
        let strikes = rs.record_reasoning_only();
        assert_eq!(strikes, 1);
    }

    #[test]
    fn run_state_empty_to_reasoning_reset() {
        // Simulate: empty → empty → reasoning only (should reset empty strikes)
        let mut rs = RunState::default();

        rs.record_empty_response();
        rs.record_empty_response();
        assert_eq!(rs.empty_response_strikes, 2);

        // Branch 1: reasoning only (should reset empty_response_strikes)
        rs.record_reasoning_only();
        assert_eq!(rs.empty_response_strikes, 0);
        assert_eq!(rs.reasoning_only_strikes, 1);
    }

    #[test]
    fn run_state_thinking_disabled_default() {
        let rs = RunState::default();
        assert!(!rs.thinking_disabled_for_rest_of_run);
    }

    #[test]
    fn run_state_thinking_disabled_after_3_strikes() {
        let mut rs = RunState::default();

        // After 1st reasoning-only: not disabled
        rs.record_reasoning_only();
        assert!(!rs.thinking_disabled_for_rest_of_run);
        assert_eq!(rs.reasoning_only_strikes, 1);

        // After 2nd reasoning-only: not disabled
        rs.record_reasoning_only();
        assert!(!rs.thinking_disabled_for_rest_of_run);
        assert_eq!(rs.reasoning_only_strikes, 2);

        // After 3rd reasoning-only: disabled!
        rs.record_reasoning_only();
        assert!(rs.thinking_disabled_for_rest_of_run);
        assert_eq!(rs.reasoning_only_strikes, 3);
    }

    #[test]
    fn run_state_thinking_disabled_resets_on_new_run() {
        let mut rs = RunState::default();

        // Simulate 3 reasoning-only responses
        rs.record_reasoning_only();
        rs.record_reasoning_only();
        rs.record_reasoning_only();
        assert!(rs.thinking_disabled_for_rest_of_run);
        assert_eq!(rs.reasoning_only_strikes, 3);

        // Reset for new run (new user message)
        rs.reset_for_new_run();
        assert!(!rs.thinking_disabled_for_rest_of_run);
        assert_eq!(rs.reasoning_only_strikes, 0);
    }

    #[test]
    fn run_state_thinking_disabled_stays_after_actions() {
        let mut rs = RunState::default();

        // Simulate 3 reasoning-only responses
        rs.record_reasoning_only();
        rs.record_reasoning_only();
        rs.record_reasoning_only();
        assert!(rs.thinking_disabled_for_rest_of_run);

        // Actions should NOT reset thinking_disabled_for_rest_of_run
        // (it should stay disabled for the rest of the run)
        rs.record_actions(2);
        assert!(rs.thinking_disabled_for_rest_of_run);
        assert_eq!(rs.reasoning_only_strikes, 0); // strikes reset, but thinking stays disabled
    }

    #[test]
    fn pre_t7_wire_keys_are_read_through_aliases() {
        // A P2-era `run_state` payload: the counting fields still spelled
        // "tool calls". The aliases keep that corpus readable verbatim.
        let json = r#"{
            "turn_tool_calls": 4,
            "run_has_tool_calls": true,
            "reasoning_only_strikes": 0,
            "empty_response_strikes": 1,
            "nudge_count": 2
        }"#;
        let rs: RunState = serde_json::from_str(json).unwrap();
        assert_eq!(rs.turn_actions, 4);
        assert!(rs.run_has_actions);
        assert_eq!(rs.empty_response_strikes, 1);
        assert_eq!(rs.nudge_count, 2);
    }

    #[test]
    fn run_state_round_trips_under_the_new_wire_keys() {
        let mut rs = RunState::default();
        rs.record_actions(3);
        let json = serde_json::to_string(&rs).unwrap();
        assert!(json.contains("\"total_actions\":3"), "new key: {json}");
        assert!(json.contains("\"turn_actions\":3"), "new key: {json}");
        assert!(json.contains("\"run_has_actions\":true"), "new key: {json}");
        let decoded: RunState = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.turn_actions, 3);
        assert_eq!(decoded.total_actions, 3);
        assert!(decoded.run_has_actions);
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn reset_for_new_run_zeros_run_fields(
            total_actions in 0usize..1000,
            turn_actions in 0usize..1000,
            run_has_actions in proptest::bool::ANY,
            reasoning_only_strikes in 0usize..100,
            empty_response_strikes in 0usize..100,
            nudge_count in 0usize..100,
        ) {
            let mut rs = RunState {
                total_actions,
                turn_actions,
                run_has_actions,
                reasoning_only_strikes,
                empty_response_strikes,
                nudge_count,
                thinking_disabled_for_rest_of_run: true,
                original_thinking_enabled: true,
                truncation_strikes: 5,
                awaiting_human: true,
            };
            rs.reset_for_new_run();
            // Session-lifetime count survives; run-level state is zeroed.
            assert_eq!(rs.total_actions, total_actions);
            assert_eq!(rs.turn_actions, 0);
            assert!(!rs.run_has_actions);
            assert_eq!(rs.reasoning_only_strikes, 0);
            assert_eq!(rs.empty_response_strikes, 0);
            assert_eq!(rs.nudge_count, 0);
            assert_eq!(rs.truncation_strikes, 0);
            assert!(!rs.thinking_disabled_for_rest_of_run);
            assert!(!rs.awaiting_human);
            // original_thinking_enabled is NOT reset
            assert!(rs.original_thinking_enabled);
        }

        #[test]
        fn record_actions_accumulates(n in 0usize..100) {
            let mut rs = RunState::default();
            rs.record_actions(n);
            assert_eq!(rs.turn_actions, n);
            assert_eq!(rs.total_actions, n);
            assert!(rs.run_has_actions);
            assert_eq!(rs.reasoning_only_strikes, 0);
            assert_eq!(rs.empty_response_strikes, 0);
        }

        #[test]
        fn record_reasoning_only_increments(count in 1usize..50) {
            let mut rs = RunState::default();
            for i in 1..=count {
                let strikes = rs.record_reasoning_only();
                assert_eq!(strikes, i);
                assert_eq!(rs.empty_response_strikes, 0);
            }
        }

        #[test]
        fn record_empty_response_increments(count in 1usize..50) {
            let mut rs = RunState::default();
            for i in 1..=count {
                let strikes = rs.record_empty_response();
                assert_eq!(strikes, i);
                assert_eq!(rs.reasoning_only_strikes, 0);
            }
        }
    }
}
