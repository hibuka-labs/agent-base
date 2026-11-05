use async_trait::async_trait;

use crate::types::AgentResult;

use super::middleware::{Middleware, PreLlmCtx};

/// Configuration for the max-turns nudge middleware.
#[derive(Clone, Debug)]
pub struct MaxTurnsNudgeConfig {
    /// When remaining turns <= this threshold, inject a nudge message.
    /// Set to 0 to disable the nudge.
    pub threshold: u32,
    /// The nudge message to inject as a user turn.
    pub message: String,
}

impl Default for MaxTurnsNudgeConfig {
    fn default() -> Self {
        Self {
            threshold: 3,
            message: "You are approaching the maximum number of turns. \
                Please wrap up your current work and provide a final answer. \
                Summarize what you've accomplished and any remaining tasks."
                .to_string(),
        }
    }
}

/// Middleware that injects a nudge message when approaching the max turns limit.
///
/// This is a **soft intervention**: the LLM can choose to continue working,
/// but the nudge encourages it to wrap up gracefully before hitting the hard
/// limit (which causes an error).
///
/// # How it works
/// - On each LLM call, checks `remaining_turns = max_turns - turn_count`
/// - If `remaining_turns <= threshold`, injects the nudge as a user message
/// - The LLM sees the nudge **before** generating its response (this turn)
///
/// # Usage
/// ```ignore
/// use agent_base::engine::{AgentBuilder, max_turns_nudge::{MaxTurnsNudgeMiddleware, MaxTurnsNudgeConfig}};
///
/// let runtime = AgentBuilder::new(client)
///     .middleware(MaxTurnsNudgeMiddleware::new(MaxTurnsNudgeConfig {
///         threshold: 3,
///         message: "Please wrap up soon.".to_string(),
///     }))
///     .build()?;
/// ```
pub struct MaxTurnsNudgeMiddleware {
    config: MaxTurnsNudgeConfig,
}

impl MaxTurnsNudgeMiddleware {
    pub fn new(config: MaxTurnsNudgeConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Middleware for MaxTurnsNudgeMiddleware {
    async fn on_pre_llm(&self, ctx: &mut PreLlmCtx) -> AgentResult<()> {
        if self.config.threshold == 0 {
            return Ok(());
        }

        let remaining_turns = ctx.max_turns.saturating_sub(ctx.turn_count);
        if remaining_turns <= self.config.threshold {
            tracing::info!(
                session_id = ctx.session_id.id,
                turn = ctx.turn_count,
                remaining_turns,
                max_turns = ctx.max_turns,
                "max turns nudge: injecting nudge message"
            );
            // Machine-composed nudge: synthetic, so the rotation trail never
            // mistakes it for the user's own words.
            ctx.messages.push(crate::types::ChatMessage::user_synthetic(
                self.config.message.clone(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::SessionId;

    fn make_ctx(turn_count: u32, max_turns: u32) -> PreLlmCtx {
        PreLlmCtx {
            session_id: SessionId::new(1),
            messages: vec![],
            tools: vec![],
            emit_fn: None,
            turn_count,
            max_turns,
        }
    }

    #[tokio::test]
    async fn test_nudge_not_injected_when_far_from_limit() {
        let mw = MaxTurnsNudgeMiddleware::new(MaxTurnsNudgeConfig {
            threshold: 3,
            message: "nudge".to_string(),
        });
        let mut ctx = make_ctx(1, 10);
        mw.on_pre_llm(&mut ctx).await.unwrap();
        assert!(ctx.messages.is_empty());
    }

    #[tokio::test]
    async fn test_nudge_injected_at_threshold() {
        let mw = MaxTurnsNudgeMiddleware::new(MaxTurnsNudgeConfig {
            threshold: 3,
            message: "nudge".to_string(),
        });
        let mut ctx = make_ctx(8, 10); // remaining = 2 <= 3
        mw.on_pre_llm(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 1);
        match &ctx.messages[0] {
            crate::types::ChatMessage::User { content, .. } => assert_eq!(content, "nudge"),
            _ => panic!("Expected User message"),
        }
    }

    #[tokio::test]
    async fn test_nudge_injected_at_exact_threshold() {
        let mw = MaxTurnsNudgeMiddleware::new(MaxTurnsNudgeConfig {
            threshold: 3,
            message: "nudge".to_string(),
        });
        let mut ctx = make_ctx(7, 10); // remaining = 3 <= 3
        mw.on_pre_llm(&mut ctx).await.unwrap();
        assert_eq!(ctx.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_nudge_disabled_when_threshold_zero() {
        let mw = MaxTurnsNudgeMiddleware::new(MaxTurnsNudgeConfig {
            threshold: 0,
            message: "nudge".to_string(),
        });
        let mut ctx = make_ctx(10, 10); // remaining = 0
        mw.on_pre_llm(&mut ctx).await.unwrap();
        assert!(ctx.messages.is_empty());
    }
}
