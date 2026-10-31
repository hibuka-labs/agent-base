use async_trait::async_trait;

use crate::engine::middleware::{Middleware, PostLlmCtx};
use crate::types::{AgentResult, SafetyConfig};

/// Middleware that enforces a hard limit on tool calls per turn.
///
/// Mounted on `on_post_llm` hook. When the turn's tool call count reaches
/// `max_tool_calls_per_turn`, this middleware:
/// 1. Demotes the turn to text-only (discards pending actions)
/// 2. Injects a `follow_up_message` forcing the LLM to summarize
///
/// This is a **hard constraint** — unlike prompt rules, the model cannot bypass it.
pub struct TurnToolLimitMiddleware {
    max_tool_calls_per_turn: usize,
}

impl TurnToolLimitMiddleware {
    pub fn new(max_tool_calls_per_turn: usize) -> Self {
        Self {
            max_tool_calls_per_turn,
        }
    }

    pub fn from_config(config: &SafetyConfig) -> Self {
        Self::new(config.max_tool_calls_per_turn)
    }
}

#[async_trait]
impl Middleware for TurnToolLimitMiddleware {
    async fn on_post_llm(&self, ctx: &mut PostLlmCtx) -> AgentResult<()> {
        // Only intercept when the LLM is trying to call tools
        if ctx.actions().is_empty() {
            return Ok(());
        }

        // Check if this turn has already hit the limit
        if ctx.turn_actions >= self.max_tool_calls_per_turn {
            tracing::warn!(
                turn_actions = ctx.turn_actions,
                max = self.max_tool_calls_per_turn,
                pending_calls = ctx.actions().len(),
                "TurnToolLimit: blocking tool calls — limit reached"
            );

            // Discard all pending tool calls
            ctx.demote_to_text();

            // Force LLM to summarize
            ctx.follow_up_message = Some(
                "本轮工具调用已达上限。请根据已有结果总结并向用户报告。不要再调用工具。"
                    .to_string(),
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor::ParseOutcome;
    use crate::types::{Action, ActionFormat, FinishReason, SessionId};

    fn make_ctx(turn_actions: usize, pending_calls: usize) -> PostLlmCtx {
        let actions = (0..pending_calls)
            .map(|i| Action {
                id: format!("call_{}", i),
                name: "test_tool".to_string(),
                args: serde_json::json!({}),
                identity: "test_tool|{}".to_string(),
                format: ActionFormat::ToolCall,
            })
            .collect::<Vec<Action>>();
        let parse = if actions.is_empty() {
            ParseOutcome::Empty
        } else {
            ParseOutcome::Actions(actions)
        };
        PostLlmCtx {
            session_id: SessionId::new(1),
            full_text: String::new(),
            parse,
            available_tools: vec!["test_tool".to_string()],
            turn_count: 1,
            total_actions: 0,
            nudge_count: 0,
            turn_actions,
            skip_push: false,
            follow_up_message: None,
            finish_reason: FinishReason::Stop,
        }
    }

    #[tokio::test]
    async fn allows_calls_under_limit() {
        let mw = TurnToolLimitMiddleware::new(8);
        let mut ctx = make_ctx(3, 2);
        mw.on_post_llm(&mut ctx).await.unwrap();
        assert_eq!(ctx.actions().len(), 2);
        assert!(ctx.follow_up_message.is_none());
    }

    #[tokio::test]
    async fn blocks_calls_at_limit() {
        let mw = TurnToolLimitMiddleware::new(8);
        let mut ctx = make_ctx(8, 3);
        mw.on_post_llm(&mut ctx).await.unwrap();
        assert!(ctx.actions().is_empty());
        assert!(!ctx.has_action_intent());
        assert!(ctx.follow_up_message.is_some());
    }

    #[tokio::test]
    async fn blocks_calls_over_limit() {
        let mw = TurnToolLimitMiddleware::new(8);
        let mut ctx = make_ctx(12, 2);
        mw.on_post_llm(&mut ctx).await.unwrap();
        assert!(ctx.actions().is_empty());
        assert!(ctx.follow_up_message.is_some());
    }

    #[tokio::test]
    async fn ignores_text_only_responses() {
        let mw = TurnToolLimitMiddleware::new(8);
        let mut ctx = make_ctx(10, 0); // no tool calls
        ctx.demote_to_text();
        ctx.full_text = "just text".to_string();
        mw.on_post_llm(&mut ctx).await.unwrap();
        assert!(ctx.follow_up_message.is_none());
    }

    #[tokio::test]
    async fn custom_limit() {
        let mw = TurnToolLimitMiddleware::new(3);
        let mut ctx = make_ctx(3, 1);
        mw.on_post_llm(&mut ctx).await.unwrap();
        assert!(ctx.actions().is_empty());
        assert!(ctx.follow_up_message.is_some());
    }
}
