use async_trait::async_trait;
use serde_json::Value;

use super::{Content, ToolContext};
use crate::types::{AgentResult, ApprovalRequest, RiskLevel};

/// Decision returned by [`ToolPolicy::before_call`] to control tool execution.
///
/// This extends the simple Ok/Err model with the ability to modify tool
/// arguments before execution — enabling hooks that auto-inject flags,
/// rewrite paths, or sanitize inputs without blocking the call.
#[derive(Clone, Debug)]
pub enum ToolDecision {
    /// Proceed with the original arguments.
    Proceed,
    /// Block the tool call entirely. The string is surfaced to the LLM as an
    /// error message so it can correct its approach.
    Block(String),
    /// Proceed, but replace the tool's arguments with the provided value.
    /// The tool executes as if the LLM had sent `modified_args` originally.
    Modify(Value),
}

/// Policy-based control over tool execution.
///
/// Implement this trait to customise how tools are approved, monitored, and
/// validated during an agent run. The pipeline calls each hook at a specific
/// point in the tool lifecycle:
///
/// ```text
/// evaluate_approval  →  before_call  →  (tool executes)  →  after_call
/// ```
///
/// # Example: auto-approve read-only tools
///
/// ```ignore
/// struct ReadOnlyPolicy;
///
/// #[async_trait]
/// impl ToolPolicy for ReadOnlyPolicy {
///     async fn evaluate_approval(&self, tool_name: &str, _args: &Value) -> Option<ApprovalRequest> {
///         if tool_name == "read_file" || tool_name == "search" {
///             None  // auto-approve — no prompt for user
///         } else {
///             Some(ApprovalRequest { message: format!("Allow {}?", tool_name) })
///         }
///     }
/// }
/// ```
///
/// All hooks have default no-op implementations, so you only need to override
/// the ones you care about.
#[async_trait]
pub trait ToolPolicy: Send + Sync {
    /// Called **before** every tool call.
    ///
    /// Return `None` to auto-approve (skip the approval handler entirely).
    /// Return `Some(ApprovalRequest)` to defer to the configured
    /// [`ApprovalHandler`](crate::ApprovalHandler).
    ///
    /// This is the primary hook for implementing permission guards — for
    /// example, auto-approving read-only operations while prompting for
    /// destructive ones.
    async fn evaluate_approval(&self, tool_name: &str, args: &Value) -> Option<ApprovalRequest>;

    /// Called immediately **before** a tool executes, after approval has been
    /// granted (or auto-approved).
    ///
    /// Return a [`ToolDecision`] to control execution:
    /// - `ToolDecision::Proceed` — execute with the original arguments (default).
    /// - `ToolDecision::Block(msg)` — cancel the call; `msg` is sent to the LLM.
    /// - `ToolDecision::Modify(new_args)` — execute with replacement arguments.
    ///
    /// Use `Modify` for:
    /// - Auto-injecting flags (e.g. `--no-color` to shell commands).
    /// - Path normalization or sandboxing.
    /// - Sanitizing inputs before execution.
    fn before_call(
        &self,
        tool_name: &str,
        args: &Value,
        ctx: &ToolContext,
    ) -> AgentResult<ToolDecision> {
        let _ = (tool_name, args, ctx);
        Ok(ToolDecision::Proceed)
    }

    /// Called immediately **after** a tool executes, before the result is
    /// returned to the LLM.
    ///
    /// **Note:** When [`ToolDecision::Modify`] is used in `before_call`, the
    /// `args` parameter here reflects the **modified** arguments, not the
    /// original ones from the LLM.
    ///
    /// Use this for:
    /// - Output scrubbing / redaction (strip secrets from tool results).
    /// - Truncation or formatting of large outputs.
    /// - Recording metrics or audit trails.
    ///
    /// The `result` is the raw [`Vec<Content>`](Content) produced by the tool.
    /// You can inspect it but not modify it through this hook — if you need to
    /// transform the output, use a middleware instead.
    ///
    /// Return an `Err` to **reject** the result. The error message is surfaced
    /// to the LLM.
    fn after_call(
        &self,
        tool_name: &str,
        args: &Value,
        result: &[Content],
        ctx: &ToolContext,
    ) -> AgentResult<()> {
        let _ = (tool_name, args, result, ctx);
        Ok(())
    }
}

/// A [`ToolPolicy`] that requires approval for every tool call.
///
/// Combined with a deny-all approval handler, this denies every tool for the
/// agent that carries it. Used by multi-agent runtimes as a fallback when a
/// child should have "no permission" but its parent has no [`ToolPolicy`] of
/// its own to inherit (so there is no notion of which tools are "dangerous").
#[derive(Debug, Clone, Default)]
pub struct DenyAllToolPolicy;

#[async_trait]
impl ToolPolicy for DenyAllToolPolicy {
    async fn evaluate_approval(&self, tool_name: &str, _args: &Value) -> Option<ApprovalRequest> {
        Some(ApprovalRequest {
            title: format!("Permission required: {tool_name}"),
            message: format!("This agent has no permission to call `{tool_name}`."),
            action_key: None,
            risk_level: RiskLevel::Destructive,
            raw: None,
            source: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Minimal policy that auto-approves everything; exercises the default
    /// no-op `before_call` / `after_call` hooks.
    struct NoopPolicy;

    #[async_trait]
    impl ToolPolicy for NoopPolicy {
        async fn evaluate_approval(
            &self,
            _tool_name: &str,
            _args: &Value,
        ) -> Option<ApprovalRequest> {
            None
        }
    }

    #[test]
    fn default_hooks_are_noop() {
        let p = NoopPolicy;
        let ctx = ToolContext::for_test();
        let args = json!({ "x": 1 });
        // before_call now returns ToolDecision::Proceed by default
        let decision = p.before_call("echo", &args, &ctx).unwrap();
        assert!(matches!(decision, ToolDecision::Proceed));
        assert!(p.after_call("echo", &args, &[], &ctx).is_ok());
    }

    #[tokio::test]
    async fn deny_all_policy_requires_approval_for_every_tool() {
        let p = DenyAllToolPolicy;
        let req = p
            .evaluate_approval("any_tool", &json!({}))
            .await
            .expect("every tool should require approval");
        assert_eq!(req.risk_level, RiskLevel::Destructive);
        assert!(req.title.contains("any_tool"));
    }
}
