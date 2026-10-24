use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::tool::{Content, Tool, ToolContext, ToolDecision, ToolPolicy, content_text};
use crate::types::{AgentError, AgentResult, DEFAULT_TOOL_TIMEOUT_MS};

/// Pure execution pipeline — cares about *how* to safely execute a tool.
///
/// Responsibilities: policy hooks, timeout, output size limit.
/// Does NOT do: tool lookup, event emission, user-event forwarding.
#[async_trait]
pub trait ToolExecutionPipeline: Send + Sync {
    async fn execute(
        &self,
        tool: &dyn Tool,
        args: &Value,
        ctx: &ToolContext,
    ) -> AgentResult<Vec<Content>>;
}

/// Default pipeline: ToolPolicy hooks + timeout + output size limit.
#[derive(Clone)]
pub struct DefaultPipeline {
    tool_policy: Option<Arc<dyn ToolPolicy>>,
    default_tool_timeout_ms: u64,
    tool_timeout_ms: Option<u64>,
    max_output_chars: Option<usize>,
}

impl DefaultPipeline {
    pub fn new(
        tool_policy: Option<Arc<dyn ToolPolicy>>,
        tool_timeout_ms: Option<u64>,
        max_output_chars: Option<usize>,
    ) -> Self {
        Self {
            tool_policy,
            default_tool_timeout_ms: DEFAULT_TOOL_TIMEOUT_MS,
            tool_timeout_ms,
            max_output_chars,
        }
    }

    pub fn with_default_timeout(mut self, timeout_ms: u64) -> Self {
        self.default_tool_timeout_ms = timeout_ms;
        self
    }

    pub fn policy(&self) -> Option<Arc<dyn ToolPolicy>> {
        self.tool_policy.clone()
    }
}

#[async_trait]
impl ToolExecutionPipeline for DefaultPipeline {
    async fn execute(
        &self,
        tool: &dyn Tool,
        args: &Value,
        ctx: &ToolContext,
    ) -> AgentResult<Vec<Content>> {
        // 1. before_call hook — may modify args or block the call entirely
        let effective_args: std::borrow::Cow<'_, Value> = if let Some(policy) = &self.tool_policy {
            match policy.before_call(tool.name(), args, ctx)? {
                ToolDecision::Proceed => std::borrow::Cow::Borrowed(args),
                ToolDecision::Block(msg) => {
                    return Err(AgentError::ToolBlocked(msg));
                }
                ToolDecision::Modify(modified) => std::borrow::Cow::Owned(modified),
            }
        } else {
            std::borrow::Cow::Borrowed(args)
        };
        let args = &*effective_args;

        // 2. Execute with an **active-time** timeout: per-call → tool → global
        // config → framework default. The budget measures active execution —
        // stretches the tool paused while waiting on a human (see `ExecClock`,
        // e.g. a confirmation popup) push the deadline out by exactly the
        // paused time. Wall-clock timeouts here produce bogus `[Tool Timeout]`
        // results while the user is still reading, with the side effects
        // half-done. `0` is unbounded: the tool arms its own fuse (or the call
        // is fire-and-forget), so no watchdog is needed.
        let timeout_ms = tool
            .call_timeout_ms(args)
            .or(self.tool_timeout_ms)
            .unwrap_or(self.default_tool_timeout_ms);
        let budget = (timeout_ms > 0).then(|| Duration::from_millis(timeout_ms));
        let started = std::time::Instant::now();
        let call = tool.call(args, ctx);
        tokio::pin!(call);
        let output = loop {
            let paused = ctx.exec_clock.paused_total();
            let Some(deadline) = budget.and_then(|b| started.checked_add(b.saturating_add(paused)))
            else {
                // Unbounded budget, or an absurd one — run to completion
                // (cancel still applies).
                break call.as_mut().await;
            };
            tokio::select! {
                result = &mut call => break result,
                _ = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
                    // The deadline moves while the clock is paused, and a pause
                    // can start or end after this sleep was armed. Only a
                    // deadline that still holds with fresh accounting is a real
                    // overrun.
                    let holds = budget
                        .and_then(|b| {
                            started.checked_add(b.saturating_add(ctx.exec_clock.paused_total()))
                        })
                        .is_some_and(|d| std::time::Instant::now() >= d);
                    if !holds {
                        continue;
                    }
                    tracing::warn!(
                        tool = tool.name(),
                        timeout_ms = timeout_ms,
                        "tool execution timed out"
                    );
                    return Ok(vec![Content::text("[Tool Timeout]")]);
                }
            }
        };
        let output = output?;

        // 3. Output size limit — reject by default rather than silently
        // truncating (design §6.5). Tools that want to return a bounded
        // subset should do their own explicit truncation before returning.
        if let Some(max_chars) = self.max_output_chars {
            let text_len = content_text(&output).chars().count();
            if text_len > max_chars {
                return Err(AgentError::ToolOutputTooLarge {
                    name: tool.name().to_string(),
                    max_chars,
                });
            }
        }

        // 4. after_call hook
        if let Some(policy) = &self.tool_policy {
            policy.after_call(tool.name(), args, &output, ctx)?;
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::types::{ApprovalRequest, Language, SessionId};

    // ── Test helpers ──

    use tokio::sync::mpsc;

    fn test_ctx() -> ToolContext {
        let (tx, _rx) = mpsc::unbounded_channel();
        ToolContext {
            session_id: SessionId::new(1),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            max_output_chars: None,
            exec_clock: crate::tool::ExecClock::default(),
            waiting_for_human: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            event_bus: crate::engine::EventBus::new(1),
        }
    }

    struct EchoTool;
    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &'static str {
            "echo"
        }
        fn description(&self) -> &'static str {
            "echo"
        }
        fn schema(&self) -> Value {
            json!({})
        }
        async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
            Ok(vec![Content::text(
                args.get("msg").and_then(|v| v.as_str()).unwrap_or("ok"),
            )])
        }
    }

    struct SlowTool;
    #[async_trait]
    impl Tool for SlowTool {
        fn name(&self) -> &'static str {
            "slow"
        }
        fn description(&self) -> &'static str {
            "slow"
        }
        fn schema(&self) -> Value {
            json!({})
        }
        fn timeout_ms(&self) -> Option<u64> {
            Some(50) // 50ms for testing
        }
        async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Ok(vec![Content::text("done")])
        }
    }

    struct FailingTool;
    #[async_trait]
    impl Tool for FailingTool {
        fn name(&self) -> &'static str {
            "fail"
        }
        fn description(&self) -> &'static str {
            "fail"
        }
        fn schema(&self) -> Value {
            json!({})
        }
        async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
            Err(AgentError::tool_not_found("intentional"))
        }
    }

    struct TrackingPolicy {
        before_count: std::sync::atomic::AtomicU32,
        after_count: std::sync::atomic::AtomicU32,
        fail_before: bool,
    }
    impl TrackingPolicy {
        fn new() -> Self {
            Self {
                before_count: std::sync::atomic::AtomicU32::new(0),
                after_count: std::sync::atomic::AtomicU32::new(0),
                fail_before: false,
            }
        }
        fn fail_before_call() -> Self {
            Self {
                before_count: std::sync::atomic::AtomicU32::new(0),
                after_count: std::sync::atomic::AtomicU32::new(0),
                fail_before: true,
            }
        }
    }
    #[async_trait]
    impl ToolPolicy for TrackingPolicy {
        async fn evaluate_approval(&self, _: &str, _: &Value) -> Option<ApprovalRequest> {
            None
        }
        fn before_call(
            &self,
            _name: &str,
            _args: &Value,
            _ctx: &ToolContext,
        ) -> AgentResult<ToolDecision> {
            self.before_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if self.fail_before {
                return Err(AgentError::ToolBlocked("before_call denied".into()));
            }
            Ok(ToolDecision::Proceed)
        }
        fn after_call(
            &self,
            _name: &str,
            _args: &Value,
            _output: &[Content],
            _ctx: &ToolContext,
        ) -> AgentResult<()> {
            self.after_count
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
    }

    // ── DefaultPipeline tests ──

    #[tokio::test]
    async fn basic_execution() {
        let pipeline = DefaultPipeline::new(None, None, None);
        let output = pipeline
            .execute(&EchoTool, &json!({"msg": "hello"}), &test_ctx())
            .await
            .unwrap();
        assert_eq!(content_text(&output), "hello");
    }

    #[tokio::test]
    async fn policy_before_and_after_called() {
        let policy = Arc::new(TrackingPolicy::new());
        let pipeline = DefaultPipeline::new(Some(policy.clone()), None, None);

        pipeline
            .execute(&EchoTool, &json!({}), &test_ctx())
            .await
            .unwrap();

        assert_eq!(
            policy
                .before_count
                .load(std::sync::atomic::Ordering::SeqCst),
            1
        );
        assert_eq!(
            policy.after_count.load(std::sync::atomic::Ordering::SeqCst),
            1
        );
    }

    #[tokio::test]
    async fn policy_before_call_aborts() {
        let policy = Arc::new(TrackingPolicy::fail_before_call());
        let pipeline = DefaultPipeline::new(Some(policy.clone()), None, None);

        let result = pipeline.execute(&EchoTool, &json!({}), &test_ctx()).await;
        assert!(result.is_err());
        // after_call should NOT be called
        assert_eq!(
            policy.after_count.load(std::sync::atomic::Ordering::SeqCst),
            0
        );
    }

    #[tokio::test]
    async fn timeout_fires() {
        let pipeline = DefaultPipeline::new(None, Some(50), None); // 50ms timeout
        let output = pipeline
            .execute(&SlowTool, &json!({}), &test_ctx())
            .await
            .unwrap();
        assert_eq!(content_text(&output), "[Tool Timeout]");
    }

    /// Sleeps `delay`; its budget comes from a per-call `timeout_ms` arg via
    /// [`Tool::call_timeout_ms`], mirroring `execute_command`.
    struct PerCallBudgetTool {
        delay: Duration,
    }
    #[async_trait]
    impl Tool for PerCallBudgetTool {
        fn name(&self) -> &'static str {
            "per_call_budget"
        }
        fn description(&self) -> &'static str {
            "per_call_budget"
        }
        fn schema(&self) -> Value {
            json!({})
        }
        async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
            tokio::time::sleep(self.delay).await;
            Ok(vec![Content::text("done")])
        }
        fn call_timeout_ms(&self, args: &Value) -> Option<u64> {
            args.get("timeout_ms").and_then(Value::as_u64)
        }
    }

    #[tokio::test]
    async fn call_timeout_overrides_framework_default() {
        // Framework default is 50ms; the per-call budget of 5s must win, or the
        // 200ms sleep gets cut (session 20261018_7844215b: a 660s
        // `execute_command` budget was killed at the 600s default).
        let pipeline = DefaultPipeline::new(None, Some(50), None);
        let output = pipeline
            .execute(
                &PerCallBudgetTool {
                    delay: Duration::from_millis(200),
                },
                &json!({"timeout_ms": 5000}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert_eq!(content_text(&output), "done");
    }

    #[tokio::test]
    async fn call_timeout_zero_is_unbounded() {
        // 0 = "never kill": no watchdog, even though the default is 50ms.
        let pipeline = DefaultPipeline::new(None, Some(50), None);
        let output = pipeline
            .execute(
                &PerCallBudgetTool {
                    delay: Duration::from_millis(200),
                },
                &json!({"timeout_ms": 0}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert_eq!(content_text(&output), "done");
    }

    #[tokio::test]
    async fn call_timeout_absent_falls_back_to_default() {
        // No per-call budget → the 50ms default cuts the 200ms sleep.
        let pipeline = DefaultPipeline::new(None, Some(50), None);
        let output = pipeline
            .execute(
                &PerCallBudgetTool {
                    delay: Duration::from_millis(200),
                },
                &json!({}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert_eq!(content_text(&output), "[Tool Timeout]");
    }

    /// Pauses the execution clock for a wall-clock stretch (a human wait),
    /// then returns.
    struct HumanWaitTool {
        wait: Duration,
    }
    #[async_trait]
    impl Tool for HumanWaitTool {
        fn name(&self) -> &'static str {
            "human_wait"
        }
        fn description(&self) -> &'static str {
            "human_wait"
        }
        fn schema(&self) -> Value {
            json!({})
        }
        fn timeout_ms(&self) -> Option<u64> {
            Some(200) // 200ms of *active* time
        }
        async fn call(&self, _args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
            let _paused = ctx.pause_execution_clock();
            tokio::time::sleep(self.wait).await;
            Ok(vec![Content::text("done")])
        }
    }

    /// Burns `active` after an unpaused-start `lead`, with a `wait` pause in
    /// between — models "tool works, pauses for the human, works again".
    struct PauseThenRunTool {
        lead: Duration,
        wait: Duration,
        active: Duration,
    }
    #[async_trait]
    impl Tool for PauseThenRunTool {
        fn name(&self) -> &'static str {
            "pause_then_run"
        }
        fn description(&self) -> &'static str {
            "pause_then_run"
        }
        fn schema(&self) -> Value {
            json!({})
        }
        fn timeout_ms(&self) -> Option<u64> {
            Some(200)
        }
        async fn call(&self, _args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
            tokio::time::sleep(self.lead).await;
            {
                let _paused = ctx.pause_execution_clock();
                tokio::time::sleep(self.wait).await;
            }
            tokio::time::sleep(self.active).await;
            Ok(vec![Content::text("done")])
        }
    }

    #[tokio::test]
    async fn timeout_excludes_human_wait() {
        // Wall clock: 600ms against a 200ms budget. Active time: ~0ms.
        // The human wait must not count — no `[Tool Timeout]`.
        let pipeline = DefaultPipeline::new(None, None, None);
        let output = pipeline
            .execute(
                &HumanWaitTool {
                    wait: Duration::from_millis(600),
                },
                &json!({}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert_eq!(content_text(&output), "done");
    }

    #[tokio::test]
    async fn timeout_still_counts_active_time_after_a_pause() {
        // 30ms active + 400ms paused + 900ms active: the second active stretch
        // blows the 200ms budget and must time out even though the pause was
        // longer than everything else.
        let pipeline = DefaultPipeline::new(None, None, None);
        let output = pipeline
            .execute(
                &PauseThenRunTool {
                    lead: Duration::from_millis(30),
                    wait: Duration::from_millis(400),
                    active: Duration::from_millis(900),
                },
                &json!({}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert_eq!(content_text(&output), "[Tool Timeout]");
    }

    #[tokio::test]
    async fn pause_mid_call_keeps_the_rest_of_the_budget() {
        // 30ms active before the pause, 50ms after — both inside the 200ms
        // budget once the 400ms human wait is excluded.
        let pipeline = DefaultPipeline::new(None, None, None);
        let output = pipeline
            .execute(
                &PauseThenRunTool {
                    lead: Duration::from_millis(30),
                    wait: Duration::from_millis(400),
                    active: Duration::from_millis(50),
                },
                &json!({}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert_eq!(content_text(&output), "done");
    }

    #[tokio::test]
    async fn no_timeout_when_tool_fast() {
        let pipeline = DefaultPipeline::new(None, Some(5000), None);
        let output = pipeline
            .execute(&EchoTool, &json!({"msg": "fast"}), &test_ctx())
            .await
            .unwrap();
        assert_eq!(content_text(&output), "fast");
    }

    #[tokio::test]
    async fn output_over_limit_is_rejected() {
        let pipeline = DefaultPipeline::new(None, None, Some(10)); // max 10 chars
        let result = pipeline
            .execute(
                &EchoTool,
                &json!({"msg": "this is a very long message"}),
                &test_ctx(),
            )
            .await;
        assert!(matches!(
            result,
            Err(AgentError::ToolOutputTooLarge { max_chars: 10, .. })
        ));
    }

    #[tokio::test]
    async fn no_rejection_when_short() {
        let pipeline = DefaultPipeline::new(None, None, Some(100));
        let output = pipeline
            .execute(&EchoTool, &json!({"msg": "short"}), &test_ctx())
            .await
            .unwrap();
        assert_eq!(content_text(&output), "short");
    }

    #[tokio::test]
    async fn output_over_limit_cjk_rejected() {
        // CJK chars are 3 bytes each. The size check counts chars, not bytes,
        // so a long Chinese string must still be rejected without panicking.
        let pipeline = DefaultPipeline::new(None, None, Some(20));
        let result = pipeline
            .execute(
                &EchoTool,
                &json!({"msg": "这是一个很长的中文消息，用于测试多字节字符的超限处理"}),
                &test_ctx(),
            )
            .await;
        assert!(matches!(
            result,
            Err(AgentError::ToolOutputTooLarge { max_chars: 20, .. })
        ));
    }

    #[tokio::test]
    async fn cjk_within_char_limit_is_not_rejected() {
        // 17 CJK chars = 51 bytes. With a 20-char limit, byte-counting would
        // wrongly reject (51 > 20), but char-counting must accept (17 <= 20).
        let pipeline = DefaultPipeline::new(None, None, Some(20));
        let output = pipeline
            .execute(
                &EchoTool,
                &json!({"msg": "这是一个用于验证字符计数的中文消息"}),
                &test_ctx(),
            )
            .await
            .unwrap();
        assert_eq!(content_text(&output).chars().count(), 17);
    }

    #[tokio::test]
    async fn timeout_plus_limit() {
        let pipeline = DefaultPipeline::new(None, Some(50), Some(100));
        let output = pipeline
            .execute(&SlowTool, &json!({}), &test_ctx())
            .await
            .unwrap();
        // timeout output is short, so it does not trip the size limit
        assert_eq!(content_text(&output), "[Tool Timeout]");
    }

    #[tokio::test]
    async fn tool_error_propagates() {
        let pipeline = DefaultPipeline::new(None, None, None);
        let result = pipeline
            .execute(&FailingTool, &json!({}), &test_ctx())
            .await;
        assert!(result.is_err());
    }

    // ── ToolDecision tests ──

    struct ModifyPolicy;
    #[async_trait]
    impl ToolPolicy for ModifyPolicy {
        async fn evaluate_approval(&self, _: &str, _: &Value) -> Option<ApprovalRequest> {
            None
        }
        fn before_call(
            &self,
            _name: &str,
            _args: &Value,
            _ctx: &ToolContext,
        ) -> AgentResult<ToolDecision> {
            // Replace the msg argument with "modified"
            Ok(ToolDecision::Modify(json!({"msg": "modified"})))
        }
    }

    struct BlockPolicy;
    #[async_trait]
    impl ToolPolicy for BlockPolicy {
        async fn evaluate_approval(&self, _: &str, _: &Value) -> Option<ApprovalRequest> {
            None
        }
        fn before_call(
            &self,
            name: &str,
            _args: &Value,
            _ctx: &ToolContext,
        ) -> AgentResult<ToolDecision> {
            Ok(ToolDecision::Block(format!("{name} is blocked")))
        }
    }

    #[tokio::test]
    async fn modify_decision_replaces_args() {
        let pipeline = DefaultPipeline::new(Some(Arc::new(ModifyPolicy)), None, None);
        let output = pipeline
            .execute(&EchoTool, &json!({"msg": "original"}), &test_ctx())
            .await
            .unwrap();
        // The policy replaced args with {"msg": "modified"}
        assert_eq!(content_text(&output), "modified");
    }

    #[tokio::test]
    async fn block_decision_returns_error() {
        let pipeline = DefaultPipeline::new(Some(Arc::new(BlockPolicy)), None, None);
        let result = pipeline
            .execute(&EchoTool, &json!({"msg": "hello"}), &test_ctx())
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("is blocked"),
            "expected block message, got: {err}"
        );
    }
}
