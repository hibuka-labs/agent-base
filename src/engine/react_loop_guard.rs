use async_trait::async_trait;

// Re-export from agent-types (single source of truth).
pub use agent_types::{GuardCtx, GuardDecision};

use crate::engine::runtime::NoticeHandle;

/// React Loop Guard trait — single unified entry point.
///
/// Runtime builds GuardCtx (with scene hints), guard decides what to do.
/// The guard has full control: it can trust the hints or re-detect.
#[async_trait]
pub trait ReactLoopGuard: Send + Sync {
    /// Unified entry point — guard judges the scene and returns a decision.
    async fn on_turn(&self, ctx: &GuardCtx) -> GuardDecision;

    /// Callback when model calls a tool (new)
    ///
    /// Default implementation: returns Complete (let other logic continue)
    ///
    /// Usage:
    /// - DefaultGuard can return RestoreThinking here
    /// - Other guards can record tool call history
    ///
    /// Note: This callback is called before tool execution, Guard cannot prevent tool execution
    async fn on_tool_call(&self, _ctx: &GuardCtx) -> GuardDecision {
        GuardDecision::Complete
    }

    /// Receive the engine's notice handle (Batch E). Called once by the
    /// runtime when the guard is registered, before any run starts. Guards
    /// that want to surface display-only notices (e.g. judge degradation)
    /// store the handle; the default implementation ignores it, so existing
    /// implementers are unaffected.
    fn set_notice(&self, _handle: NoticeHandle) {}
}

/// Default guard — fails on degenerate states, completes on normal flow.
///
/// This is the default guard injected when no custom guard is set.
/// It provides basic safety: reasoning-only and empty responses fail,
/// text-only responses complete normally.
pub struct NoopGuard;

#[async_trait]
impl ReactLoopGuard for NoopGuard {
    async fn on_turn(&self, ctx: &GuardCtx) -> GuardDecision {
        if ctx.is_reasoning_only || ctx.is_empty_response {
            GuardDecision::Fail {
                error: if ctx.is_reasoning_only {
                    "model produced only reasoning, no output".to_string()
                } else {
                    "model returned empty response".to_string()
                },
            }
        } else {
            GuardDecision::Complete
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_types::{FinishReason, SessionId};

    #[tokio::test]
    async fn test_noop_guard_returns_complete() {
        let guard = NoopGuard;
        let ctx = GuardCtx {
            session_id: SessionId {
                id: 1,
                external_id: None,
            },
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
            thinking_disabled: false,
            original_thinking_enabled: true,
            remaining_turns: 50,
        };

        assert!(matches!(guard.on_turn(&ctx).await, GuardDecision::Complete));
    }

    #[tokio::test]
    async fn test_noop_guard_handles_degenerate_states() {
        let guard = NoopGuard;

        // reasoning-only → Fail
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
            thinking_disabled: false,
            original_thinking_enabled: true,
            remaining_turns: 50,
        };
        assert!(matches!(
            guard.on_turn(&ctx).await,
            GuardDecision::Fail { .. }
        ));

        // empty response → Fail
        let mut ctx2 = ctx.clone();
        ctx2.is_reasoning_only = false;
        ctx2.is_empty_response = true;
        assert!(matches!(
            guard.on_turn(&ctx2).await,
            GuardDecision::Fail { .. }
        ));

        // text-only → Complete
        let mut ctx3 = ctx.clone();
        ctx3.is_reasoning_only = false;
        ctx3.is_empty_response = false;
        ctx3.is_text_only = true;
        assert!(matches!(
            guard.on_turn(&ctx3).await,
            GuardDecision::Complete
        ));

        // no flags → Complete
        let mut ctx4 = ctx.clone();
        ctx4.is_reasoning_only = false;
        ctx4.is_text_only = false;
        assert!(matches!(
            guard.on_turn(&ctx4).await,
            GuardDecision::Complete
        ));
    }
}
