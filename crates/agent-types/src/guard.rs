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

    // ─── Turn-as-transaction (stream integrity) ───────────────
    /// Discard this response whole and re-ask the LLM from the same state.
    ///
    /// For silent rounds whose tool-call batch is structurally corrupt (wire
    /// truncation / misplaced args fragments). The kernel executes the
    /// discard: NO history commit, NO tool execution — the response never
    /// happened. The retry stays inside the current turn (`turn_count`
    /// unchanged) and is bounded by the policy budget carried in
    /// [`GuardCtx::silent_retries_in_round`] plus a kernel hard cap.
    /// Never returned for rounds whose text already reached the user.
    RetryTurn,
}

/// Structural integrity facts about one assembled tool-call batch.
///
/// Detection is shape-based (never provider metadata — stop reasons lie):
/// a named call whose args are non-empty but not valid JSON is wire
/// truncation; args fragments arriving on a name-less bucket are a misplaced
/// tail (the provider leaked them past the owning block, e.g. into a
/// thinking block's index).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamIntegrity {
    /// Named tool calls with non-empty, unparseable args.
    #[serde(default)]
    pub invalid_args: usize,
    /// Bytes of args fragments that landed on name-less buckets.
    #[serde(default)]
    pub orphan_args_bytes: usize,
}

impl StreamIntegrity {
    pub fn is_clean(&self) -> bool {
        self.invalid_args == 0 && self.orphan_args_bytes == 0
    }
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
    #[serde(alias = "run_has_tool_calls")]
    pub run_has_actions: bool,
    /// The tools this run has called so far, in order, by name. A completion
    /// judge handed only `user_input` and `model_response` cannot tell a report
    /// from a shrug — both are a short text-only turn. This is the work it can
    /// check the response against. Names only; nothing sensitive rides along.
    #[serde(default)]
    pub run_actions: Vec<String>,
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
    /// Structural facts about this turn's assembled tool-call batch (wire
    /// corruption detection). Clean on healthy streams.
    #[serde(default)]
    pub integrity: StreamIntegrity,
    /// The response carried visible text (already streamed to the user) —
    /// a round with text can never be silently discarded.
    #[serde(default)]
    pub has_visible_text: bool,
    /// The response carried thinking content (tolerated as "silent" by
    /// default policy; tighten via config).
    #[serde(default)]
    pub has_thinking: bool,
    /// Silent retries already spent on the current logical round (a retry
    /// keeps the same round). Policy budget compares against this; the kernel
    /// enforces its own hard cap on top.
    #[serde(default)]
    pub silent_retries_in_round: u32,
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
