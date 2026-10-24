use std::time::Instant;

use crate::engine::react_loop_guard::GuardCtx;
use crate::engine::runtime::plan_runner::RuntimeCore;
use crate::types::{FinishReason, RunOutcome, SessionId};

use super::turn_end::TurnEndCtx;

/// LLM turn metrics needed to fire turn-end callbacks in guard dispatch.
pub(super) struct TurnMetrics<'a> {
    pub ttft_ms: u64,
    pub llm_duration_ms: u64,
    pub usage: &'a Option<crate::llm::UsageInfo>,
    pub text_len: u64,
    pub has_thinking: bool,
    /// Byte length of reasoning/thinking content (always available, even when
    /// the provider does not report reasoning token counts).
    pub thinking_bytes: u64,
}

/// Bundle of per-turn context shared across all guard dispatch paths.
pub(super) struct TurnCtx<'a> {
    pub session_id: &'a SessionId,
    pub turn_count: u32,
    pub user_input: &'a str,
    pub finish_reason: &'a FinishReason,
    pub available_tools: &'a [String],
    pub turn_start: Instant,
    pub model: &'a str,
    pub all_user_inputs: &'a [String],
    pub max_turns: u32,
}

impl RuntimeCore {
    // ── Guard context construction ──────────────────────────────────────

    /// Build a GuardCtx from turn context and scene flags.
    ///
    /// Scene flags (`is_reasoning_only`, `is_empty_response`, `is_text_only`)
    /// are detected by runtime and passed as hints to the guard.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn build_guard_ctx(
        &self,
        turn_ctx: &TurnCtx<'_>,
        model_response: &str,
        is_reasoning_only: bool,
        is_empty_response: bool,
        is_text_only: bool,
        thinking_disabled: bool,
    ) -> GuardCtx {
        let session_data = self
            .session_manager
            .session_or_err(turn_ctx.session_id)
            .await
            .map(|s| {
                (
                    s.run_state.reasoning_only_strikes,
                    s.run_state.empty_response_strikes,
                    s.run_state.run_has_tool_calls,
                    s.run_state.original_thinking_enabled,
                    s.run_state.truncation_strikes > 0,
                    s.run_state.awaiting_human,
                )
            })
            .unwrap_or((0, 0, false, false, false, false));

        GuardCtx {
            session_id: turn_ctx.session_id.clone(),
            turn_count: turn_ctx.turn_count,
            user_input: turn_ctx.user_input.to_string(),
            model_response: model_response.to_string(),
            finish_reason: turn_ctx.finish_reason.clone(),
            available_tools: turn_ctx.available_tools.to_vec(),
            reasoning_only_strikes: session_data.0,
            empty_response_strikes: session_data.1,
            run_has_tool_calls: session_data.2,
            last_tool_calls_invalid: session_data.4,
            all_user_inputs: turn_ctx.all_user_inputs.to_vec(),
            is_reasoning_only,
            is_empty_response,
            is_text_only,
            awaiting_human: session_data.5,
            thinking_disabled,
            original_thinking_enabled: session_data.3,
            remaining_turns: turn_ctx.max_turns.saturating_sub(turn_ctx.turn_count),
        }
    }

    // ── Guard fail helper ───────────────────────────────────────────────

    /// Fire turn-end with a `Failed` outcome — used by guard decision execution.
    pub(super) async fn fire_guard_fail(
        &self,
        turn_ctx: &TurnCtx<'_>,
        metrics: &TurnMetrics<'_>,
        error: &str,
    ) {
        self.fire_turn_end(TurnEndCtx {
            ttft_ms: metrics.ttft_ms,
            llm_duration_ms: metrics.llm_duration_ms,
            usage: metrics.usage,
            text_length: metrics.text_len,
            has_thinking: metrics.has_thinking,
            thinking_bytes: metrics.thinking_bytes,
            llm_calls: 1,
            error_message: Some(error),
            ..TurnEndCtx::new(
                turn_ctx.session_id,
                turn_ctx.turn_count,
                turn_ctx.turn_start,
                turn_ctx.model,
                turn_ctx.user_input,
                RunOutcome::Failed {
                    error: error.to_string(),
                },
            )
        })
        .await;
    }
}

#[cfg(test)]
mod tests {
    use crate::engine::react_loop_guard::{GuardCtx, GuardDecision, ReactLoopGuard};
    use crate::types::{FinishReason, SessionId};

    /// A mock guard that returns a configurable decision.
    struct MockGuard {
        decision: GuardDecision,
    }

    impl MockGuard {
        fn new(decision: GuardDecision) -> Self {
            Self { decision }
        }
    }

    #[async_trait::async_trait]
    impl ReactLoopGuard for MockGuard {
        async fn on_turn(&self, _ctx: &GuardCtx) -> GuardDecision {
            self.decision.clone()
        }
    }

    #[test]
    fn test_guard_decision_clone() {
        let decision = GuardDecision::Continue {
            nudge: Some("test".to_string()),
        };
        let cloned = decision.clone();
        match cloned {
            GuardDecision::Continue { nudge } => assert_eq!(nudge.unwrap(), "test"),
            _ => panic!("Expected Continue"),
        }

        let decision = GuardDecision::Complete;
        let cloned = decision.clone();
        assert!(matches!(cloned, GuardDecision::Complete));

        let decision = GuardDecision::Fail {
            error: "error".to_string(),
        };
        let cloned = decision.clone();
        match cloned {
            GuardDecision::Fail { error } => assert_eq!(error, "error"),
            _ => panic!("Expected Fail"),
        }
    }

    #[test]
    fn test_guard_ctx_debug() {
        let ctx = GuardCtx {
            session_id: SessionId::new(1),
            turn_count: 1,
            user_input: "test".to_string(),
            model_response: "response".to_string(),
            finish_reason: FinishReason::Stop,
            available_tools: vec!["tool1".to_string()],
            reasoning_only_strikes: 0,
            empty_response_strikes: 0,
            run_has_tool_calls: false,
            last_tool_calls_invalid: false,
            all_user_inputs: vec!["test".to_string()],
            is_reasoning_only: false,
            is_empty_response: false,
            is_text_only: false,
            awaiting_human: false,
            thinking_disabled: false,
            original_thinking_enabled: true,
            remaining_turns: 50,
        };
        let debug_str = format!("{:?}", ctx);
        assert!(debug_str.contains("session_id"));
        assert!(debug_str.contains("turn_count"));
        assert!(debug_str.contains("user_input"));
    }

    #[test]
    fn test_guard_ctx_scene_flags() {
        let ctx = GuardCtx {
            session_id: SessionId::new(1),
            turn_count: 1,
            user_input: "test".to_string(),
            model_response: "".to_string(),
            finish_reason: FinishReason::Stop,
            available_tools: vec![],
            reasoning_only_strikes: 1,
            empty_response_strikes: 0,
            run_has_tool_calls: false,
            last_tool_calls_invalid: false,
            all_user_inputs: vec!["test".to_string()],
            is_reasoning_only: true,
            is_empty_response: false,
            is_text_only: false,
            awaiting_human: false,
            thinking_disabled: true,
            original_thinking_enabled: true,
            remaining_turns: 50,
        };
        assert!(ctx.is_reasoning_only);
        assert!(!ctx.is_empty_response);
        assert!(!ctx.is_text_only);
        assert!(ctx.thinking_disabled);
    }

    #[tokio::test]
    async fn test_mock_guard_returns_configured_decision() {
        let guard = MockGuard::new(GuardDecision::Continue {
            nudge: Some("nudge".to_string()),
        });

        let ctx = GuardCtx {
            session_id: SessionId::new(1),
            turn_count: 1,
            user_input: "test".to_string(),
            model_response: "response".to_string(),
            finish_reason: FinishReason::Stop,
            available_tools: vec![],
            reasoning_only_strikes: 0,
            empty_response_strikes: 0,
            run_has_tool_calls: false,
            last_tool_calls_invalid: false,
            all_user_inputs: vec!["test".to_string()],
            is_reasoning_only: false,
            is_empty_response: false,
            is_text_only: false,
            awaiting_human: false,
            thinking_disabled: false,
            original_thinking_enabled: true,
            remaining_turns: 50,
        };

        let decision = guard.on_turn(&ctx).await;
        assert!(matches!(
            decision,
            GuardDecision::Continue { nudge: Some(_) }
        ));
    }
}
