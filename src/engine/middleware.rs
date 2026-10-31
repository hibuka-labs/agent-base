use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::engine::executor::ParseOutcome;
use crate::types::{Action, AgentResult, ChatMessage, FinishReason, SessionId, UserEvent};

#[derive(Clone)]
pub struct UserMessageCtx {
    pub session_id: SessionId,
    pub user_input: String,
}

pub struct PreLlmCtx {
    pub session_id: SessionId,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<Value>,
    /// Unified emit function: sends [`UserEvent`] to both the renderer (real-time)
    /// and the event bus (persistence).  Set by the react loop; middleware calls
    /// [`emit()`](Self::emit) which delegates to this closure.
    ///
    /// The closure captures the renderer callback and the event bus, so middleware
    /// never needs to know about broadcast channels or drain mechanics.
    pub emit_fn: Option<Box<dyn Fn(UserEvent) + Send + Sync>>,
    /// Current turn count (1-indexed).
    pub turn_count: u32,
    /// Maximum turns allowed for this run.
    pub max_turns: u32,
}

impl PreLlmCtx {
    /// Emit a [`UserEvent`] to both the renderer and the event bus.
    ///
    /// This is the single entry point for middleware to send events.  The event
    /// is delivered to the renderer callback (real-time display) and to the
    /// event bus (persistence, event_log, checkpoint) in one call.
    ///
    /// Does nothing if the emit function was not set (e.g. in tests).
    /// Panics inside the emit function are caught and logged as warnings.
    pub fn emit(&self, event: UserEvent) {
        if let Some(ref f) = self.emit_fn
            && let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                f(event);
            }))
        {
            tracing::warn!("emit failed: {:?}", e);
        }
    }
}

#[derive(Clone)]
pub struct PostLlmCtx {
    pub session_id: SessionId,
    pub full_text: String,
    /// The action backend's four-state verdict for this turn (M3 seam). The
    /// turn's action list rides inside [`ParseOutcome::Actions`]; middleware
    /// discards pending work with [`PostLlmCtx::demote_to_text`]. Tool calls
    /// and CodeAct cells both surface here as [`Action`]s — no protocol shape.
    pub parse: ParseOutcome,
    pub available_tools: Vec<String>,
    pub turn_count: u32,
    pub total_actions: usize,
    /// Number of tool-enforcement nudges issued in the current turn.
    /// Read from session; middleware may increment this to track nudge attempts.
    pub nudge_count: usize,
    /// Number of tool calls already executed in the current turn.
    /// Used by `TurnToolLimitMiddleware` to enforce per-turn tool call limits.
    pub turn_actions: usize,
    pub skip_push: bool,
    pub follow_up_message: Option<String>,
    /// Semantic finish reason from the LLM (Stop / ToolUse / Truncated / Other).
    /// Middleware can inspect this to implement custom continuation logic
    /// (e.g. auto-continue on truncation).
    pub finish_reason: FinishReason,
}

impl PostLlmCtx {
    /// The turn's pending actions (empty when the turn has none or was demoted).
    pub fn actions(&self) -> &[Action] {
        match &self.parse {
            ParseOutcome::Actions(actions) => actions,
            _ => &[],
        }
    }

    /// Mutable access to the pending actions (e.g. to drop selected ones).
    pub fn actions_mut(&mut self) -> Option<&mut Vec<Action>> {
        match &mut self.parse {
            ParseOutcome::Actions(actions) => Some(actions),
            _ => None,
        }
    }

    /// Discard the turn's pending actions and demote it to text-only. A
    /// demoted turn dispatches exactly like [`ParseOutcome::Empty`] (nudge on
    /// empty text, finish-anomaly handling on text).
    pub fn demote_to_text(&mut self) {
        self.parse = ParseOutcome::Empty;
    }

    /// Whether the backend saw action intent this turn — true for
    /// [`ParseOutcome::Actions`] and [`ParseOutcome::Incomplete`] (a call the
    /// truncation guard left with zero entries still counts as intent).
    pub fn has_action_intent(&self) -> bool {
        self.parse.has_action_intent()
    }
}

#[async_trait]
pub trait Middleware: Send + Sync {
    async fn on_user_message(&self, _ctx: &mut UserMessageCtx) -> AgentResult<()> {
        Ok(())
    }

    async fn on_pre_llm(&self, _ctx: &mut PreLlmCtx) -> AgentResult<()> {
        Ok(())
    }

    async fn on_post_llm(&self, _ctx: &mut PostLlmCtx) -> AgentResult<()> {
        Ok(())
    }
}

pub(crate) type MiddlewareRef = Arc<dyn Middleware>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SessionId;

    #[test]
    fn test_post_llm_ctx_new_fields_defaults() {
        let ctx = PostLlmCtx {
            session_id: SessionId {
                id: 1,
                external_id: None,
            },
            full_text: "test".to_string(),
            parse: ParseOutcome::Empty,
            available_tools: vec![],
            turn_count: 0,
            total_actions: 0,
            nudge_count: 0,
            turn_actions: 0,
            skip_push: false,
            follow_up_message: None,
            finish_reason: FinishReason::Stop,
        };
        assert!(ctx.available_tools.is_empty());
        assert_eq!(ctx.turn_count, 0);
        assert_eq!(ctx.total_actions, 0);
        assert!(!ctx.skip_push);
        assert!(ctx.follow_up_message.is_none());
        assert_eq!(ctx.finish_reason, FinishReason::Stop);
    }

    #[test]
    fn test_post_llm_ctx_skip_push_follow_up_set() {
        let ctx = PostLlmCtx {
            session_id: SessionId {
                id: 2,
                external_id: None,
            },
            full_text: "I will execute...".to_string(),
            parse: ParseOutcome::Empty,
            available_tools: vec!["echo".to_string()],
            turn_count: 1,
            total_actions: 0,
            nudge_count: 0,
            turn_actions: 0,
            skip_push: true,
            follow_up_message: Some("Please call tools now.".to_string()),
            finish_reason: FinishReason::Stop,
        };
        assert!(ctx.skip_push);
        assert_eq!(
            ctx.follow_up_message,
            Some("Please call tools now.".to_string())
        );
        assert_eq!(ctx.available_tools, vec!["echo".to_string()]);
        assert_eq!(ctx.total_actions, 0);
    }

    #[test]
    fn test_post_llm_ctx_clone_preserves_new_fields() {
        let ctx = PostLlmCtx {
            session_id: SessionId {
                id: 3,
                external_id: None,
            },
            full_text: "hello".to_string(),
            parse: ParseOutcome::Empty,
            available_tools: vec!["add".to_string(), "subtract".to_string()],
            turn_count: 5,
            total_actions: 3,
            nudge_count: 0,
            turn_actions: 2,
            skip_push: true,
            follow_up_message: Some("nudge".to_string()),
            finish_reason: FinishReason::Truncated {
                reason: Some("max_tokens".into()),
            },
        };
        let cloned = ctx.clone();
        assert_eq!(cloned.available_tools, vec!["add", "subtract"]);
        assert_eq!(cloned.turn_count, 5);
        assert_eq!(cloned.total_actions, 3);
        assert_eq!(cloned.turn_actions, 2);
        assert!(cloned.skip_push);
        assert_eq!(cloned.follow_up_message, Some("nudge".to_string()));
        assert!(cloned.finish_reason.is_truncated());
    }
}
