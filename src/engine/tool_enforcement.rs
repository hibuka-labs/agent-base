use async_trait::async_trait;

use crate::engine::middleware::{Middleware, PostLlmCtx};
use crate::types::AgentResult;

pub struct ToolEnforcementConfig {
    pub max_nudges: usize,
    pub nudge_message: String,
    pub first_turn_only: bool,
    pub min_tools_threshold: usize,
}

impl Default for ToolEnforcementConfig {
    fn default() -> Self {
        Self {
            max_nudges: 3,
            nudge_message: "CRITICAL: You have tools available but did not call any. \
                             Call the appropriate tool NOW. \
                             关键提示：你有可用的工具但没有调用。立即使用工具执行。"
                .to_string(),
            first_turn_only: true,
            min_tools_threshold: 1,
        }
    }
}

pub struct ToolEnforcementMiddleware {
    config: ToolEnforcementConfig,
}

impl ToolEnforcementMiddleware {
    pub fn new(config: ToolEnforcementConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl Middleware for ToolEnforcementMiddleware {
    async fn on_post_llm(&self, ctx: &mut PostLlmCtx) -> AgentResult<()> {
        if ctx.available_tools.len() < self.config.min_tools_threshold {
            return Ok(());
        }
        if self.config.first_turn_only && ctx.total_actions > 0 {
            return Ok(());
        }
        if ctx.has_action_intent() {
            return Ok(());
        }
        if ctx.full_text.is_empty() {
            return Ok(());
        }

        if ctx.nudge_count >= self.config.max_nudges {
            return Ok(());
        }

        // Increment nudge_count in the context; the caller will write it back to the session
        ctx.nudge_count += 1;

        tracing::info!(
            nudge_count = ctx.nudge_count,
            full_text_len = ctx.full_text.len(),
            "ToolEnforcement: suppressing text response, injecting nudge"
        );

        ctx.skip_push = true;
        ctx.follow_up_message = Some(self.config.nudge_message.clone());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor::ParseOutcome;
    use crate::types::{FinishReason, SessionId};

    fn ctx(available_tools: Vec<String>) -> PostLlmCtx {
        PostLlmCtx {
            session_id: SessionId::new(1),
            full_text: "I will do it.".to_string(),
            parse: ParseOutcome::Empty,
            available_tools,
            turn_count: 1,
            total_actions: 0,
            nudge_count: 0,
            turn_actions: 0,
            skip_push: false,
            follow_up_message: None,
            finish_reason: FinishReason::Stop,
        }
    }

    #[test]
    fn config_defaults() {
        let cfg = ToolEnforcementConfig::default();
        assert_eq!(cfg.max_nudges, 3);
        assert!(cfg.first_turn_only);
        assert_eq!(cfg.min_tools_threshold, 1);
        assert!(cfg.nudge_message.contains("CRITICAL"));
    }

    #[tokio::test]
    async fn nudges_when_no_tool_call() {
        let config = ToolEnforcementConfig::default();
        let expected_nudge = config.nudge_message.clone();
        let mw = ToolEnforcementMiddleware::new(config);

        let mut c = ctx(vec!["read".to_string()]);
        mw.on_post_llm(&mut c).await.unwrap();

        assert!(c.skip_push);
        assert_eq!(c.nudge_count, 1);
        assert_eq!(c.follow_up_message, Some(expected_nudge));
    }

    #[tokio::test]
    async fn skips_when_below_tools_threshold() {
        let mw = ToolEnforcementMiddleware::new(ToolEnforcementConfig::default());
        let mut c = ctx(vec![]);
        mw.on_post_llm(&mut c).await.unwrap();

        assert!(!c.skip_push);
        assert_eq!(c.nudge_count, 0);
        assert!(c.follow_up_message.is_none());
    }

    #[tokio::test]
    async fn skips_when_is_tool_call() {
        let mw = ToolEnforcementMiddleware::new(ToolEnforcementConfig::default());
        let mut c = ctx(vec!["read".to_string()]);
        // Action intent (even with no materialized calls) is tool work —
        // never text-nudge the model over it.
        c.parse = ParseOutcome::Incomplete;
        mw.on_post_llm(&mut c).await.unwrap();

        assert!(!c.skip_push);
        assert_eq!(c.nudge_count, 0);
    }

    #[tokio::test]
    async fn skips_when_full_text_empty() {
        let mw = ToolEnforcementMiddleware::new(ToolEnforcementConfig::default());
        let mut c = ctx(vec!["read".to_string()]);
        c.full_text = String::new();
        mw.on_post_llm(&mut c).await.unwrap();

        assert!(!c.skip_push);
        assert_eq!(c.nudge_count, 0);
    }

    #[tokio::test]
    async fn skips_when_max_nudges_reached() {
        let config = ToolEnforcementConfig {
            max_nudges: 1,
            ..ToolEnforcementConfig::default()
        };
        let mw = ToolEnforcementMiddleware::new(config);

        let mut c = ctx(vec!["read".to_string()]);
        c.nudge_count = 1;
        mw.on_post_llm(&mut c).await.unwrap();

        assert!(!c.skip_push);
        assert_eq!(c.nudge_count, 1);
        assert!(c.follow_up_message.is_none());
    }

    #[tokio::test]
    async fn first_turn_only_skips_after_tool_calls() {
        let mw = ToolEnforcementMiddleware::new(ToolEnforcementConfig::default());
        let mut c = ctx(vec!["read".to_string()]);
        c.total_actions = 1;
        mw.on_post_llm(&mut c).await.unwrap();

        assert!(!c.skip_push);
        assert_eq!(c.nudge_count, 0);
    }
}
