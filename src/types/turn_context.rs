//! Turn context — pure data struct passed to turn-end hooks.
//!
//! agent-base does not build, aggregate, or persist metrics. It only exposes
//! raw turn data through the [`on_turn_end`](crate::AgentRuntime::on_turn_end)
//! hook. Consumers (e.g. phi-telemetry) build their own metrics from this context.

use crate::llm::UsageInfo;
use crate::types::RunOutcome;

/// Pure-data snapshot of a completed turn iteration, passed to turn-end hooks.
///
/// Contains all raw data a consumer needs to build turn-level metrics —
/// agent-base itself performs no aggregation, no persistence, and no
/// business-level interpretation of this data.
#[derive(Clone, Debug)]
pub struct TurnContext {
    /// Numeric session identifier (agent-base internal).
    pub session_id: u64,
    /// 1-based turn number within the session.
    pub turn_number: u32,
    /// Time-to-first-token in milliseconds (user-perceived latency).
    pub ttft_ms: u64,
    /// LLM stream duration in milliseconds.
    pub llm_duration_ms: u64,
    /// Total wall-clock turn duration in milliseconds (llm + tool + overhead).
    pub duration_ms: u64,
    /// Total tool execution duration in milliseconds.
    pub tool_duration_ms: u64,
    /// Token usage from the LLM response, if available.
    pub usage: Option<UsageInfo>,
    /// Length of the full assistant text response in bytes.
    pub full_text_len: u64,
    /// Whether the response included thinking/reasoning content.
    pub has_thinking: bool,
    /// Byte length of reasoning/thinking content. Always available even when
    /// the provider does not report reasoning token counts (e.g. mimo-v2.5-pro).
    pub thinking_bytes: u64,
    /// Tool names called in this turn iteration.
    pub tools_used: Vec<String>,
    /// Total number of tool calls made.
    pub tool_call_count: u32,
    /// Number of tools that succeeded.
    pub tool_success: u32,
    /// Number of tools that failed.
    pub tool_failed: u32,
    /// Outcome of this turn iteration (Completed / Failed / Cancelled / MaxTurnsExceeded).
    pub outcome: RunOutcome,
    /// Error message if the turn errored.
    pub error_message: Option<String>,
    /// The user's input text (may be truncated).
    pub user_input: String,
    /// Model name used for the LLM call.
    pub model: String,
    /// Plan-update events emitted during this turn iteration (taken from EventBus).
    pub plan_updates: u32,
    /// Approval-request events emitted during this turn iteration (taken from EventBus).
    pub approval_count: u32,
    /// Number of LLM calls made (≥ 1; includes retries).
    pub llm_calls: u32,
    /// Session-lifetime count of silent retries spent on corrupt rounds.
    pub silent_retries: u32,
    /// Session-lifetime count of structurally corrupt tool-call batches seen.
    pub integrity_failures: u32,
}
