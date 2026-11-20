use crate::engine::runtime::plan_runner::RuntimeCore;
use crate::types::{RunOutcome, SessionId};

impl RuntimeCore {
    /// Build a TurnContext and fire all registered turn-end callbacks.
    /// agent-base does NOT store, aggregate, or persist metrics — consumers
    /// (e.g. phi-telemetry) do that via their registered callback.
    pub(super) async fn fire_turn_end(&self, ctx: TurnEndCtx<'_>) {
        let duration_ms = ctx.turn_start.elapsed().as_millis() as u64;
        // Session-lifetime integrity counters (metrics seam): how many
        // corrupt batches this session saw and how many silent retries it
        // spent. Read at turn end so the numbers reflect the round that just
        // committed.
        let (silent_retries, integrity_failures) = self
            .session_manager
            .session_or_err(ctx.session_id)
            .await
            .map(|s| (s.run_state.silent_retries, s.run_state.integrity_failures))
            .unwrap_or((0, 0));

        let turn_ctx = crate::types::TurnContext {
            session_id: ctx.session_id.id,
            turn_number: ctx.turn_number,
            ttft_ms: ctx.ttft_ms,
            llm_duration_ms: ctx.llm_duration_ms,
            duration_ms,
            tool_duration_ms: ctx.tool_duration_ms,
            usage: ctx.usage.clone(),
            full_text_len: ctx.text_length,
            has_thinking: ctx.has_thinking,
            thinking_bytes: ctx.thinking_bytes,
            tools_used: ctx.tools_used.to_vec(),
            tool_call_count: ctx.tool_call_count,
            tool_success: ctx.tool_success,
            tool_failed: ctx.tool_failed,
            outcome: ctx.outcome,
            error_message: ctx.error_message.map(|s| s.to_string()),
            user_input: truncate_for_context(ctx.user_input),
            model: ctx.model.to_string(),
            plan_updates: self.event_bus.take_plan_updates(),
            approval_count: self.event_bus.take_approval_count(),
            llm_calls: ctx.llm_calls,
            silent_retries,
            integrity_failures,
        };

        let callbacks = self.turn_end_callbacks.read().unwrap();
        for cb in callbacks.iter() {
            cb(&turn_ctx);
        }
        drop(callbacks);
    }
}

/// Collapsed argument bundle for `fire_turn_end` — a borrowed struct so the 7
/// call sites name fields instead of passing an opaque run of `0,0,0,&None,...`.
pub(super) struct TurnEndCtx<'a> {
    pub(super) session_id: &'a SessionId,
    pub(super) turn_number: u32,
    pub(super) turn_start: std::time::Instant,
    pub(super) model: &'a str,
    pub(super) user_input: &'a str,
    pub(super) ttft_ms: u64,
    pub(super) llm_duration_ms: u64,
    pub(super) tool_duration_ms: u64,
    pub(super) usage: &'a Option<crate::llm::UsageInfo>,
    pub(super) text_length: u64,
    pub(super) has_thinking: bool,
    /// Byte length of reasoning/thinking content.
    pub(super) thinking_bytes: u64,
    pub(super) tool_call_count: u32,
    pub(super) tools_used: &'a [String],
    pub(super) tool_success: u32,
    pub(super) tool_failed: u32,
    pub(super) outcome: RunOutcome,
    pub(super) error_message: Option<&'a str>,
    pub(super) llm_calls: u32,
}

impl<'a> TurnEndCtx<'a> {
    /// Fill the six always-provided fields; the rest default to "zero" metrics
    /// (0 / false / `&None` / `&[]` / `None`), matching the old positional calls.
    pub(super) fn new(
        session_id: &'a SessionId,
        turn_number: u32,
        turn_start: std::time::Instant,
        model: &'a str,
        user_input: &'a str,
        outcome: RunOutcome,
    ) -> Self {
        Self {
            session_id,
            turn_number,
            turn_start,
            model,
            user_input,
            ttft_ms: 0,
            llm_duration_ms: 0,
            tool_duration_ms: 0,
            usage: &None,
            text_length: 0,
            has_thinking: false,
            thinking_bytes: 0,
            tool_call_count: 0,
            tools_used: &[],
            tool_success: 0,
            tool_failed: 0,
            outcome,
            error_message: None,
            llm_calls: 0,
        }
    }
}

/// Truncate a string to 80 characters (respecting UTF-8 boundaries),
/// appending "..." if truncated.
fn truncate_for_context(s: &str) -> String {
    if s.chars().count() > 80 {
        let truncated: String = s.chars().take(80).collect();
        format!("{}...", truncated)
    } else {
        s.to_string()
    }
}
