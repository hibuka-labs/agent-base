use async_trait::async_trait;

use crate::types::AgentResult;

use super::middleware::{Middleware, PostLlmCtx};

/// Middleware that automatically injects a continuation message when the LLM
/// response is truncated by the token limit (e.g. `max_tokens` / `length`)
/// and no tool calls are present.
///
/// This turns a silent truncation into an automatic retry: the react loop
/// will push the follow-up as a user message and call the LLM again.
///
/// # When it triggers
/// - `finish_reason` is `Truncated` **and**
/// - `actions` is empty (tool-call truncation is already handled by the
///   react-loop's built-in truncation guard, which injects error results
///   instead of executing incomplete tool calls).
///
/// # Usage
/// ```ignore
/// use agent_base::engine::{AgentBuilder, auto_continue::AutoContinueMiddleware};
///
/// let runtime = AgentBuilder::new(client)
///     .middleware(AutoContinueMiddleware::new())
///     .build()?;
/// ```
pub struct AutoContinueMiddleware {
    /// The message to inject as a user turn when truncation is detected.
    prompt: String,
}

impl AutoContinueMiddleware {
    /// Create a new middleware with the default continuation prompt.
    pub fn new() -> Self {
        Self {
            prompt: "Please continue.".to_string(),
        }
    }

    /// Create a new middleware with a custom continuation prompt.
    pub fn with_prompt(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
        }
    }
}

impl Default for AutoContinueMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for AutoContinueMiddleware {
    async fn on_post_llm(&self, ctx: &mut PostLlmCtx) -> AgentResult<()> {
        if ctx.finish_reason.is_truncated() && ctx.actions().is_empty() {
            tracing::info!(
                session_id = ctx.session_id.id,
                turn = ctx.turn_count,
                "text-only response truncated — injecting auto-continue prompt"
            );
            ctx.follow_up_message = Some(self.prompt.clone());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor::ParseOutcome;
    use crate::types::{Action, ActionFormat, FinishReason, SessionId};

    fn ctx(finish_reason: FinishReason, tool_calls: Vec<(String, String, String)>) -> PostLlmCtx {
        let actions = tool_calls
            .into_iter()
            .map(|(id, name, args)| Action {
                id,
                name,
                args: serde_json::Value::String(args),
                identity: String::new(),
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
            full_text: "partial answer...".to_string(),
            parse,
            available_tools: vec![],
            turn_count: 1,
            total_actions: 0,
            nudge_count: 0,
            turn_actions: 0,
            skip_push: false,
            follow_up_message: None,
            finish_reason,
        }
    }

    #[tokio::test]
    async fn triggers_on_truncated_text_only() {
        let mw = AutoContinueMiddleware::new();
        let mut c = ctx(
            FinishReason::Truncated {
                reason: Some("max_tokens".into()),
            },
            vec![],
        );
        mw.on_post_llm(&mut c).await.unwrap();
        assert_eq!(c.follow_up_message, Some("Please continue.".to_string()));
    }

    #[tokio::test]
    async fn triggers_on_truncated_no_reason() {
        let mw = AutoContinueMiddleware::new();
        let mut c = ctx(FinishReason::Truncated { reason: None }, vec![]);
        mw.on_post_llm(&mut c).await.unwrap();
        assert_eq!(c.follow_up_message, Some("Please continue.".to_string()));
    }

    #[tokio::test]
    async fn skips_when_tool_calls_present() {
        let mw = AutoContinueMiddleware::new();
        let mut c = ctx(
            FinishReason::Truncated {
                reason: Some("length".into()),
            },
            vec![("id".into(), "shell".into(), "{}".into())],
        );
        mw.on_post_llm(&mut c).await.unwrap();
        assert!(c.follow_up_message.is_none());
    }

    #[tokio::test]
    async fn skips_when_stop() {
        let mw = AutoContinueMiddleware::new();
        let mut c = ctx(FinishReason::Stop, vec![]);
        mw.on_post_llm(&mut c).await.unwrap();
        assert!(c.follow_up_message.is_none());
    }

    #[tokio::test]
    async fn skips_when_tool_use() {
        let mw = AutoContinueMiddleware::new();
        let mut c = ctx(FinishReason::ToolUse, vec![]);
        mw.on_post_llm(&mut c).await.unwrap();
        assert!(c.follow_up_message.is_none());
    }

    #[tokio::test]
    async fn custom_prompt() {
        let mw = AutoContinueMiddleware::with_prompt("Continue please.");
        let mut c = ctx(
            FinishReason::Truncated {
                reason: Some("max_tokens".into()),
            },
            vec![],
        );
        mw.on_post_llm(&mut c).await.unwrap();
        assert_eq!(c.follow_up_message, Some("Continue please.".to_string()));
    }
}
