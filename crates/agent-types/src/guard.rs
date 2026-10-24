//! Guard-related pure types: GuardDecision, GuardCtx.

use serde::{Deserialize, Serialize};

use crate::execution::FinishReason;
use crate::session::SessionId;

/// Guard decision — returned by guard, executed by base loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuardDecision {
    /// Continue loop, optionally inject nudge message
    Continue { nudge: Option<String> },
    /// Normal completion (fire_turn_end + RunOutcome::Completed)
    Complete,
    /// Abnormal termination (fire_guard_fail + RunOutcome::Failed)
    Fail { error: String },

    // ─── Thinking control ─────────────────────────────────────
    /// Temporarily disable thinking functionality
    ///
    /// Used for reasoning-only loop scenarios: model keeps thinking but produces no output.
    /// After calling, runtime will:
    /// 1. Set thinking_disabled_for_rest_of_run = true
    /// 2. Inject nudge message
    /// 3. Continue loop
    DisableThinking { nudge: String },

    /// Restore thinking functionality to previous state
    ///
    /// Used for thinking recovery scenarios: model starts working normally (has text or tool call).
    /// After calling, runtime will:
    /// 1. Restore thinking_disabled_for_rest_of_run to original state
    /// 2. Reset related counters
    RestoreThinking,
}

/// Guard context information — built by runtime, passed to guard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardCtx {
    pub session_id: SessionId,
    pub turn_count: u32,
    pub user_input: String,
    pub model_response: String,
    pub finish_reason: FinishReason,
    pub available_tools: Vec<String>,
    // RunState information
    pub reasoning_only_strikes: usize,
    pub empty_response_strikes: usize,
    pub run_has_tool_calls: bool,
    /// The most recent assistant tool calls were ALL rejected before
    /// execution (truncated/invalid arguments) and the model was told to
    /// re-issue them (`RunState.truncation_strikes > 0`). While this is
    /// true, a text-only response cannot be task completion — the work the
    /// text describes was never executed (session 20260904_efad759c: a
    /// fabricated success narrative passed the completion judge).
    pub last_tool_calls_invalid: bool,
    /// All user messages in the current session, ordered oldest-first.
    /// Guards can use this to reconstruct full conversation context
    /// (e.g. "continue" after a multi-turn discussion).
    pub all_user_inputs: Vec<String>,
    // Scene hints (runtime detected, guard can trust or ignore)
    pub is_reasoning_only: bool,
    pub is_empty_response: bool,
    pub is_text_only: bool,
    /// The run ended with a human step pending: the latest tool batch parked
    /// on human input (a design-canvas screen awaiting a pick) and no tool ran
    /// after it. Ending here is *waiting for the user*, not an incomplete task
    /// — the next input is their answer. A "continue working" nudge in this
    /// state steamrolls interactive flows (session 20261022_bb8c2dc3: the
    /// completion judge pushed the model on and it decided the user's design
    /// choices for them).
    #[serde(default)]
    pub awaiting_human: bool,
    // Environment state
    pub thinking_disabled: bool,
    /// Original thinking configuration (for restoration)
    ///
    /// From RunState.original_thinking_enabled
    pub original_thinking_enabled: bool,
    /// Remaining turns before hitting max_turns limit
    ///
    /// Guards can use this to nudge the model to wrap up when running low on turns.
    /// When remaining_turns == 0, the run will be terminated with MaxTurnsExceeded.
    pub remaining_turns: u32,
}
