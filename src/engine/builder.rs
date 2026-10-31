use std::sync::Arc;
use std::time::Duration;

use crate::llm::ReasoningConfig;
use crate::tool::{Tool, ToolPolicy, ToolRegistry};
use crate::types::{
    AgentConfig, ConvertToLlmFn, ResponseFormat, RetryConfig, SessionIdGenerator,
    shared_session_ids,
};

use super::AgentRuntime;
use super::approval::ApprovalHandler;
use super::context::{ContextCompaction, ContextWindowManager};
use super::executor::{
    ActionBackend, DEFAULT_INTERACT_TIMEOUT, FailClosedInteract, InteractionChannel,
};
use super::middleware::{Middleware, MiddlewareRef};
use super::react_loop_guard::ReactLoopGuard;
use super::recovery::{StopOnError, ToolErrorRecovery};
use super::session_store::{InMemorySessionStore, SessionStore};

pub struct AgentBuilder {
    provider: Arc<dyn llm_trait::LlmProvider>,
    config: AgentConfig,
    tools: ToolRegistry,
    approval_handler: Option<Arc<dyn ApprovalHandler>>,
    tool_policy: Option<Arc<dyn ToolPolicy>>,
    middlewares: Vec<MiddlewareRef>,
    context_manager: Option<ContextWindowManager>,
    session_store: Option<Arc<dyn SessionStore>>,
    error_recovery: Option<Arc<dyn ToolErrorRecovery>>,
    event_bus_capacity: usize,
    session_id_generator: Option<Arc<dyn SessionIdGenerator>>,
    convert_to_llm: Option<ConvertToLlmFn>,
    guard: Option<Arc<dyn ReactLoopGuard>>,
    context_compactor: Option<Arc<dyn ContextCompaction>>,
    action_backend: Option<Box<dyn ActionBackend>>,
    interact: Option<Arc<dyn InteractionChannel>>,
    /// Outer `None` = unset (the default watchdog); inner `None` = explicitly
    /// unbounded human time.
    interact_timeout: Option<Option<Duration>>,
}

impl AgentBuilder {
    /// Create a new builder with an `LlmProvider`.
    pub fn new(provider: Arc<dyn llm_trait::LlmProvider>) -> Self {
        Self {
            provider,
            config: AgentConfig::default(),
            tools: ToolRegistry::default(),
            approval_handler: None,
            tool_policy: None,
            middlewares: Vec::new(),
            context_manager: None,
            session_store: None,
            error_recovery: None,
            event_bus_capacity: 2048,
            session_id_generator: None,
            convert_to_llm: None,
            guard: None,
            context_compactor: None,
            action_backend: None,
            interact: None,
            interact_timeout: None,
        }
    }

    pub fn event_bus_capacity(mut self, capacity: usize) -> Self {
        self.event_bus_capacity = capacity;
        self
    }

    pub fn session_id_generator(mut self, generator: Arc<dyn SessionIdGenerator>) -> Self {
        self.session_id_generator = Some(generator);
        self
    }

    /// Set a callback to transform messages before they are sent to the LLM.
    ///
    /// The default behavior (when `None`) is to filter out
    /// `ChatMessage::Custom` variants, which most providers don't understand.
    /// Override this to inject custom serialization logic for application-specific
    /// message types.
    pub fn convert_to_llm(mut self, cb: ConvertToLlmFn) -> Self {
        self.convert_to_llm = Some(cb);
        self
    }

    pub fn guard(mut self, guard: impl ReactLoopGuard + 'static) -> Self {
        self.guard = Some(Arc::new(guard));
        self
    }

    /// Set a guard from a pre-built `Arc<dyn ReactLoopGuard>`.
    ///
    /// Use this when you need to share the guard reference (e.g. for inspecting
    /// recorded calls in tests).
    pub fn guard_dyn(mut self, guard: Arc<dyn ReactLoopGuard>) -> Self {
        self.guard = Some(guard);
        self
    }

    /// Set an inline context compactor for the react loop.
    ///
    /// When set, the react loop will check token count after each tool turn
    /// and compact the session if it exceeds the configured threshold.
    pub fn context_compactor(mut self, compactor: Arc<dyn ContextCompaction>) -> Self {
        self.context_compactor = Some(compactor);
        self
    }

    /// Mount this runtime's action backend (T10c): text turns route to it,
    /// tool-envelope turns keep the built-in tool-call backend. One backend
    /// per runtime — build one runtime per session when the backend holds a
    /// resident engine (the product's assembly-time choice, sdk-assembly
    /// §3.4).
    pub fn action_backend(mut self, backend: Box<dyn ActionBackend>) -> Self {
        self.action_backend = Some(backend);
        self
    }

    /// The product's interaction surface for the action backend's `ExecCtx`
    /// (OV-6): consent and ask_user suspend through it. Without one, every
    /// interaction fails closed (`FailClosedInteract`) — a cell reaching for
    /// the human refuses, it never hangs or silently allows.
    pub fn interaction(mut self, channel: Arc<dyn InteractionChannel>) -> Self {
        self.interact = Some(channel);
        self
    }

    /// Per-interaction deadline for the action backend's suspensions (`None`
    /// = unbounded human time, the run's cancel token is the only guard —
    /// phimint's TUI shape). Unset = [`DEFAULT_INTERACT_TIMEOUT`].
    pub fn interaction_timeout(mut self, interact_timeout: Option<Duration>) -> Self {
        self.interact_timeout = Some(interact_timeout);
        self
    }

    /// Check if a guard has been set.
    ///
    /// Used by agent-works to inject a default guard if none was set.
    pub fn get_guard(&self) -> Option<&Arc<dyn ReactLoopGuard>> {
        self.guard.as_ref()
    }

    pub fn system_prompt(mut self, system_prompt: impl Into<String>) -> Self {
        self.config.system_prompt = Some(system_prompt.into());
        self
    }

    /// Set whether to include the reasoning content in LLM responses.
    ///
    /// Controls whether the `reasoning_content` field from the LLM is forwarded
    /// to consumers (i.e., "show the thinking process").
    /// See [`AgentConfig::enable_thought`] for the distinction from `enable_thinking()`.
    pub fn enable_thought(mut self, enable: bool) -> Self {
        self.config.enable_thought = enable;
        self
    }

    pub fn reasoning(mut self, config: ReasoningConfig) -> Self {
        self.config.reasoning = Some(config);
        self
    }

    /// Set whether to enable the model's extended thinking / reasoning mode.
    ///
    /// Controls whether the model performs deep reasoning (i.e., "enable thinking mode").
    /// See [`AgentConfig::enable_thought`] for the distinction from `enable_thought()`.
    pub fn enable_thinking(mut self, enable: bool) -> Self {
        let mut config = self.config.reasoning.take().unwrap_or_default();
        config.enabled = Some(enable);
        self.config.reasoning = Some(config);
        self
    }

    pub fn thinking_budget(mut self, budget: u64) -> Self {
        let mut config = self.config.reasoning.take().unwrap_or_default();
        config.budget_tokens = Some(budget);
        self.config.reasoning = Some(config);
        self
    }

    pub fn tool_timeout(mut self, timeout_ms: u64) -> Self {
        self.config.tool.tool_timeout_ms = Some(timeout_ms);
        self
    }

    pub fn max_tool_output_chars(mut self, max_chars: usize) -> Self {
        self.config.tool.max_tool_output_chars = Some(max_chars);
        self
    }

    pub fn register_tool(mut self, tool: impl Tool + 'static) -> Self {
        self.tools.register(tool);
        self
    }

    pub fn register_tool_arc(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.register_arc(tool);
        self
    }

    pub fn approval_handler(mut self, handler: Arc<dyn ApprovalHandler>) -> Self {
        self.approval_handler = Some(handler);
        self
    }

    pub fn tool_policy(mut self, policy: Arc<dyn ToolPolicy>) -> Self {
        self.tool_policy = Some(policy);
        self
    }

    pub fn middleware(mut self, mw: impl Middleware + 'static) -> Self {
        self.middlewares.push(Arc::new(mw));
        self
    }

    pub fn context_window(mut self, max_tokens: usize) -> Self {
        self.context_manager = Some(ContextWindowManager::new(max_tokens));
        self
    }

    pub fn context_window_manager(mut self, manager: ContextWindowManager) -> Self {
        self.context_manager = Some(manager);
        self
    }

    pub fn response_format(mut self, format: ResponseFormat) -> Self {
        self.config.llm.response_format = Some(format);
        self
    }

    pub fn llm_retry(mut self, retry: RetryConfig) -> Self {
        self.config.llm.llm_retry = Some(retry);
        self
    }

    pub fn session_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.session_store = Some(store);
        self
    }

    pub fn error_recovery(mut self, recovery: Arc<dyn ToolErrorRecovery>) -> Self {
        self.error_recovery = Some(recovery);
        self
    }

    pub fn max_sessions(mut self, max: usize) -> Self {
        self.config.session.max_sessions = Some(max);
        self
    }

    pub fn max_turns_per_session(mut self, max: usize) -> Self {
        self.config.session.max_turns_per_session = Some(max);
        self
    }

    /// Cap the number of react-loop iterations allowed for a *single* run (one user
    /// input). Distinct from [`Self::max_turns_per_session`], which caps turns across
    /// the whole session. When unset, falls back to `DEFAULT_MAX_TURNS` (50).
    pub fn execution_max_turns(mut self, max: u32) -> Self {
        self.config.execution.max_turns = Some(max);
        self
    }

    pub fn max_message_tokens(mut self, max: usize) -> Self {
        self.config.session.max_message_tokens = Some(max);
        self
    }

    pub fn tool_error_retry_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.config.tool.tool_error_retry_prompt = Some(prompt.into());
        self
    }

    pub fn language(mut self, language: crate::types::Language) -> Self {
        self.config.language = language;
        self
    }

    /// Conditionally chain a builder call: apply `f` only when `value` is `Some`.
    ///
    /// # Example
    /// ```ignore
    /// let builder = AgentBuilder::new(client)
    ///     .apply_if(config.timeout, |b, t| b.tool_timeout(t))
    ///     .apply_if(config.max_chars, |b, c| b.max_tool_output_chars(c));
    /// ```
    pub fn apply_if<T>(self, value: Option<T>, f: impl FnOnce(Self, T) -> Self) -> Self {
        match value {
            Some(v) => f(self, v),
            None => self,
        }
    }

    pub fn build(self) -> crate::types::AgentResult<AgentRuntime> {
        self.config.validate()?;

        tracing::info!(
            tool_count = self.tools.len(),
            middleware_count = self.middlewares.len(),
            has_approval = self.approval_handler.is_some(),
            has_context_window = self.context_manager.is_some(),
            "building agent runtime"
        );

        let event_bus = super::runtime::EventBus::new(self.event_bus_capacity);

        let session_store = self
            .session_store
            .unwrap_or_else(|| Arc::new(InMemorySessionStore::new()));
        let error_recovery = self.error_recovery.unwrap_or_else(|| Arc::new(StopOnError));
        // Default numbering is process-global: a host and its sub-agents are
        // several engines in one process, and per-engine counters handed every
        // one of them a "session 1".
        let session_id_generator = self.session_id_generator.unwrap_or_else(shared_session_ids);

        let session_manager = super::runtime::SessionManager::new(
            session_id_generator,
            session_store,
            self.config.session.clone(),
        );

        let llm_engine = super::runtime::LlmEngine::new(self.provider.clone(), event_bus.clone());

        let tool_engine = super::runtime::ToolEngine::new(
            self.tools,
            self.approval_handler,
            self.tool_policy,
            error_recovery,
            event_bus.clone(),
        );

        // Interaction defaults: fail-closed when the product mounts no
        // channel; the standard watchdog when it sets no deadline.
        let interact = self
            .interact
            .unwrap_or_else(|| Arc::new(FailClosedInteract));
        let interact_timeout = self
            .interact_timeout
            .unwrap_or(Some(DEFAULT_INTERACT_TIMEOUT));

        let runner = Arc::new(super::runtime::RuntimeCore::new(
            self.config,
            llm_engine,
            tool_engine,
            session_manager,
            event_bus,
            self.context_manager,
            self.middlewares,
            self.convert_to_llm,
            self.guard,
            self.context_compactor,
            self.action_backend,
            interact,
            interact_timeout,
        ));

        Ok(AgentRuntime { runner })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::DenyAllApprovalHandler;
    use crate::llm::ReasoningEffort;
    use crate::tool::{Content, ToolContext};
    use crate::types::{AgentError, AgentResult, ApprovalRequest, Language, ResponseFormat};
    use async_trait::async_trait;
    use llm_trait::{Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, ProviderInfo};
    use serde_json::Value;

    struct DummyProvider;

    #[async_trait]
    impl llm_trait::LlmProvider for DummyProvider {
        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            Ok(ChatStream::new(Box::pin(futures_util::stream::empty())))
        }

        async fn chat(&self, _request: ChatRequest) -> Result<ChatResponse, LlmError> {
            Ok(ChatResponse {
                content: String::new(),
                reasoning_content: None,
                tool_calls: vec![],
                usage: Default::default(),
                finish_reason: llm_trait::FinishReason::Stop,
                raw: None,
                thinking_signature: None,
            })
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }

        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                name: "dummy".to_string(),
                model: "dummy".to_string(),
                version: None,
            }
        }
    }

    #[test]
    fn execution_max_turns_writes_per_run_config() {
        let builder = AgentBuilder::new(Arc::new(DummyProvider)).execution_max_turns(200);
        assert_eq!(builder.config.execution.max_turns, Some(200));
    }

    #[test]
    fn execution_max_turns_defaults_to_none() {
        let builder = AgentBuilder::new(Arc::new(DummyProvider));
        assert_eq!(builder.config.execution.max_turns, None);
    }

    fn b() -> AgentBuilder {
        AgentBuilder::new(Arc::new(DummyProvider))
    }

    struct NoopTool;

    #[async_trait]
    impl Tool for NoopTool {
        fn name(&self) -> &'static str {
            "noop"
        }
        fn description(&self) -> &'static str {
            "noop tool"
        }
        fn schema(&self) -> Value {
            serde_json::json!({"type": "object"})
        }
        async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
            Ok(vec![Content::text("ok")])
        }
    }

    struct AutoApprovePolicy;

    #[async_trait]
    impl ToolPolicy for AutoApprovePolicy {
        async fn evaluate_approval(
            &self,
            _tool_name: &str,
            _args: &Value,
        ) -> Option<ApprovalRequest> {
            None
        }
    }

    struct NoopMiddleware;

    impl Middleware for NoopMiddleware {}

    #[test]
    fn system_prompt_sets_config() {
        assert_eq!(
            b().system_prompt("be helpful")
                .config
                .system_prompt
                .as_deref(),
            Some("be helpful")
        );
    }

    #[test]
    fn enable_thought_sets_config() {
        assert!(b().enable_thought(true).config.enable_thought);
    }

    #[test]
    fn reasoning_sets_config() {
        let rc = ReasoningConfig {
            enabled: Some(true),
            budget_tokens: Some(64),
            effort: Some(ReasoningEffort::Medium),
        };
        let builder = b().reasoning(rc);
        let got = builder.config.reasoning.as_ref().unwrap();
        assert_eq!(got.enabled, Some(true));
        assert_eq!(got.budget_tokens, Some(64));
        assert!(matches!(got.effort.as_ref(), Some(ReasoningEffort::Medium)));
    }

    #[test]
    fn enable_thinking_and_budget_set_reasoning() {
        let builder = b().enable_thinking(true).thinking_budget(128);
        let got = builder.config.reasoning.as_ref().unwrap();
        assert_eq!(got.enabled, Some(true));
        assert_eq!(got.budget_tokens, Some(128));
    }

    #[test]
    fn tool_limits_set_config() {
        let builder = b().tool_timeout(5_000).max_tool_output_chars(1_024);
        assert_eq!(builder.config.tool.tool_timeout_ms, Some(5_000));
        assert_eq!(builder.config.tool.max_tool_output_chars, Some(1_024));
    }

    #[test]
    fn register_tool_adds_to_registry() {
        assert_eq!(b().register_tool(NoopTool).tools.len(), 1);
    }

    #[test]
    fn approval_handler_and_tool_policy_are_set() {
        let builder = b()
            .approval_handler(Arc::new(DenyAllApprovalHandler))
            .tool_policy(Arc::new(AutoApprovePolicy));
        assert!(builder.approval_handler.is_some());
        assert!(builder.tool_policy.is_some());
    }

    #[test]
    fn middleware_and_context_window_are_set() {
        let builder = b().middleware(NoopMiddleware).context_window(8_000);
        assert_eq!(builder.middlewares.len(), 1);
        assert!(builder.context_manager.is_some());
    }

    #[test]
    fn response_format_and_retry_set_config() {
        let builder = b()
            .response_format(ResponseFormat::JsonObject)
            .llm_retry(RetryConfig::default().max_retries(5));
        assert!(builder.config.llm.response_format.is_some());
        assert_eq!(
            builder.config.llm.llm_retry.as_ref().unwrap().max_retries,
            5
        );
    }

    #[test]
    fn session_store_and_error_recovery_are_set() {
        let builder = b()
            .session_store(Arc::new(InMemorySessionStore::new()))
            .error_recovery(Arc::new(StopOnError));
        assert!(builder.session_store.is_some());
        assert!(builder.error_recovery.is_some());
    }

    #[test]
    fn session_limits_set_config() {
        let builder = b()
            .max_sessions(10)
            .max_turns_per_session(20)
            .max_message_tokens(30);
        assert_eq!(builder.config.session.max_sessions, Some(10));
        assert_eq!(builder.config.session.max_turns_per_session, Some(20));
        assert_eq!(builder.config.session.max_message_tokens, Some(30));
    }

    #[test]
    fn tool_error_retry_prompt_and_language_set_config() {
        let builder = b()
            .tool_error_retry_prompt("try again")
            .language(Language::Zh);
        assert_eq!(
            builder.config.tool.tool_error_retry_prompt.as_deref(),
            Some("try again")
        );
        assert_eq!(builder.config.language, Language::Zh);
    }

    #[test]
    fn apply_if_applies_when_some_and_skips_when_none() {
        let applied = b().apply_if(Some(3_000_u64), |b, t| b.tool_timeout(t));
        assert_eq!(applied.config.tool.tool_timeout_ms, Some(3_000));

        let skipped = b().apply_if(None, |b, t| b.tool_timeout(t));
        assert_eq!(skipped.config.tool.tool_timeout_ms, None);
    }

    #[test]
    fn build_ok_with_defaults() {
        assert!(b().build().is_ok());
    }

    #[test]
    fn build_err_on_invalid_config() {
        assert!(matches!(
            b().execution_max_turns(0).build(),
            Err(AgentError::ConfigError(_))
        ));
    }
}
