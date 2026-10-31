use std::sync::Mutex;

use async_trait::async_trait;

use crate::engine::middleware::{Middleware, PostLlmCtx, UserMessageCtx};
use crate::types::{AgentResult, Language};

/// Turn fact summary middleware — injects structured facts from the previous
/// turn's tool results into the next user message.
///
/// This prevents long-conversation attention drift: the LLM sees deterministic
/// facts (tool call names) instead of relying on fuzzy memory of tool outputs
/// buried 20+ turns ago.
///
/// # Design
///
/// - After each LLM turn with tool calls, collects which tools were called
///   and stores them in a buffer.
/// - At the start of the next user message, prepends the buffered facts as a
///   structured prefix, then clears the buffer.
/// - Facts are derived from tool names only (not parsing output text), keeping
///   the logic simple and model-agnostic.
/// - The prefix language can be configured via [`Language`]; defaults to
///   [`Language::Zh`] for backward compatibility.
pub struct TurnFactMiddleware {
    pending_facts: Mutex<Vec<String>>,
    language: Language,
}

impl TurnFactMiddleware {
    /// Create a new middleware with the default language (Chinese).
    pub fn new() -> Self {
        Self {
            pending_facts: Mutex::new(Vec::new()),
            language: Language::Zh,
        }
    }

    /// Create a new middleware with the specified language for the prefix text.
    pub fn with_language(language: Language) -> Self {
        Self {
            pending_facts: Mutex::new(Vec::new()),
            language,
        }
    }
}

impl Default for TurnFactMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for TurnFactMiddleware {
    async fn on_user_message(&self, ctx: &mut UserMessageCtx) -> AgentResult<()> {
        let facts = {
            let mut guard = self.pending_facts.lock().unwrap();
            if guard.is_empty() {
                return Ok(());
            }
            std::mem::take(&mut *guard)
        };

        // Prepend facts to user message as structured context
        let prefix = match self.language {
            Language::Zh => format!(
                "[本轮工具调用摘要 — 以下为确定性事实，请以此为准]\n{}\n",
                facts.join("\n")
            ),
            Language::En => format!(
                "[Previous turn tool-call summary — treat these as ground truth]\n{}\n",
                facts.join("\n")
            ),
        };
        ctx.user_input = format!("{prefix}\n{original}", original = ctx.user_input);

        Ok(())
    }

    async fn on_post_llm(&self, ctx: &mut PostLlmCtx) -> AgentResult<()> {
        if ctx.actions().is_empty() {
            return Ok(());
        }

        let mut facts = Vec::new();
        for action in ctx.actions() {
            // Record which tools were called — the actual results are in the
            // session history, but a compact reminder helps the LLM stay grounded.
            let fact = match self.language {
                Language::Zh => format!("- 调用了工具: {}", action.name),
                Language::En => format!("- Called tool: {}", action.name),
            };
            facts.push(fact);
        }

        let mut guard = self.pending_facts.lock().unwrap();
        guard.extend(facts);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::executor::ParseOutcome;
    use crate::types::{Action, ActionFormat, FinishReason, SessionId};

    fn tool_actions(calls: &[(&str, &str)]) -> Vec<Action> {
        calls
            .iter()
            .enumerate()
            .map(|(i, (name, args))| Action {
                id: format!("id{}", i + 1),
                name: (*name).into(),
                args: serde_json::Value::String((*args).into()),
                identity: format!("{name}::{args}"),
                format: ActionFormat::ToolCall,
            })
            .collect()
    }

    fn make_post_ctx(actions: Vec<Action>) -> PostLlmCtx {
        PostLlmCtx {
            session_id: SessionId::new(1),
            full_text: String::new(),
            parse: ParseOutcome::Actions(actions),
            available_tools: vec![],
            turn_count: 1,
            total_actions: 0,
            nudge_count: 0,
            turn_actions: 0,
            skip_push: false,
            follow_up_message: None,
            finish_reason: FinishReason::Stop,
        }
    }

    #[tokio::test]
    async fn test_no_facts_no_prefix() {
        let mw = TurnFactMiddleware::new();
        let mut ctx = UserMessageCtx {
            session_id: SessionId::new(1),
            user_input: "hello".to_string(),
        };
        mw.on_user_message(&mut ctx).await.unwrap();
        assert_eq!(ctx.user_input, "hello");
    }

    #[tokio::test]
    async fn test_facts_injected_on_next_user_message() {
        let mw = TurnFactMiddleware::new();

        // Simulate a tool call turn
        let mut post_ctx = make_post_ctx(tool_actions(&[
            ("execute_ssh_command", "{}"),
            ("start_interactive_task", "{}"),
        ]));
        mw.on_post_llm(&mut post_ctx).await.unwrap();

        // Next user message should have facts prepended
        let mut user_ctx = UserMessageCtx {
            session_id: SessionId::new(1),
            user_input: "继续执行".to_string(),
        };
        mw.on_user_message(&mut user_ctx).await.unwrap();

        assert!(user_ctx.user_input.contains("本轮工具调用摘要"));
        assert!(user_ctx.user_input.contains("execute_ssh_command"));
        assert!(user_ctx.user_input.contains("start_interactive_task"));
        assert!(user_ctx.user_input.contains("继续执行"));
    }

    #[tokio::test]
    async fn test_facts_cleared_after_injection() {
        let mw = TurnFactMiddleware::new();

        // First turn with tool calls
        let mut post_ctx = make_post_ctx(tool_actions(&[("docker", "{}")]));
        mw.on_post_llm(&mut post_ctx).await.unwrap();

        // First user message gets facts
        let mut ctx1 = UserMessageCtx {
            session_id: SessionId::new(1),
            user_input: "next".into(),
        };
        mw.on_user_message(&mut ctx1).await.unwrap();
        assert!(ctx1.user_input.contains("本轮工具调用摘要"));

        // Second user message should NOT have facts (cleared)
        let mut ctx2 = UserMessageCtx {
            session_id: SessionId::new(1),
            user_input: "again".into(),
        };
        mw.on_user_message(&mut ctx2).await.unwrap();
        assert_eq!(ctx2.user_input, "again");
    }

    #[tokio::test]
    async fn test_english_language_prefix() {
        let mw = TurnFactMiddleware::with_language(Language::En);

        let mut post_ctx = make_post_ctx(tool_actions(&[("docker", "{}")]));
        mw.on_post_llm(&mut post_ctx).await.unwrap();

        let mut ctx = UserMessageCtx {
            session_id: SessionId::new(1),
            user_input: "continue".into(),
        };
        mw.on_user_message(&mut ctx).await.unwrap();

        assert!(ctx.user_input.contains("Previous turn tool-call summary"));
        assert!(ctx.user_input.contains("ground truth"));
        assert!(ctx.user_input.contains("continue"));
    }
}
