use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::engine::AgentSession;
use crate::engine::session_store::SessionStore;
use crate::types::{
    AgentConfig, AgentError, AgentResult, CheckpointData, MessageRole, RunOutcome, RuntimeEvent,
    SessionId, TurnContext,
};

use super::approval::ApprovalHandler;
use crate::tool::ToolPolicy;

mod event_bus;
pub(crate) use event_bus::EventBus;
mod llm_engine;
mod message_queue;
mod plan_runner;
mod react;
mod session_manager;
mod tool_engine;

pub(super) const DEFAULT_MAX_TURNS: u32 = 160;

pub use llm_engine::LlmEngine;
pub use message_queue::QueueMode;
pub(crate) use plan_runner::RuntimeCore;
pub use session_manager::SessionManager;
pub(crate) use tool_engine::ToolEngine;

#[derive(Clone)]
pub struct AgentRuntime {
    pub(crate) runner: Arc<RuntimeCore>,
}

impl AgentRuntime {
    pub async fn create_session(&self) -> SessionId {
        let config = self.runner.config.read().await;
        self.runner
            .session_manager
            .create_session(config.system_prompt.as_deref())
            .await
    }

    pub async fn restore_session(&self, session_id: &SessionId) -> Option<AgentSession> {
        self.runner
            .session_manager
            .restore_session(session_id)
            .await
    }

    pub async fn session(&self, session_id: &SessionId) -> Option<AgentSession> {
        self.runner.session_manager.session(session_id).await
    }

    pub async fn session_or_err(&self, session_id: &SessionId) -> AgentResult<AgentSession> {
        self.runner.session_manager.session_or_err(session_id).await
    }

    pub async fn with_session_mut<F, R>(&self, session_id: &SessionId, f: F) -> AgentResult<R>
    where
        F: FnOnce(&mut AgentSession) -> R,
    {
        self.runner
            .session_manager
            .with_session_mut(session_id, f)
            .await
    }

    pub fn emit_event(&self, event: RuntimeEvent) {
        self.runner.event_bus.emit(event);
    }

    /// Subscribe to runtime events from the internal broadcast channel.
    ///
    /// Events are delivered directly from the runtime's event bus (capacity 2048).
    /// Slow consumers may receive `Lagged(n)` errors if they cannot keep up —
    /// ensure the receiver loop processes events promptly or use a buffering
    /// layer in the consumer if backpressure is a concern.
    pub fn subscribe_runtime_events(&self) -> tokio::sync::broadcast::Receiver<RuntimeEvent> {
        self.runner.event_bus.subscribe()
    }

    pub fn session_manager(&self) -> &SessionManager {
        &self.runner.session_manager
    }

    pub fn llm_engine(&self) -> &LlmEngine {
        &self.runner.llm_engine
    }

    pub fn provider(&self) -> Arc<dyn llm_trait::LlmProvider> {
        self.runner.llm_engine.get_provider()
    }

    /// Replace the LLM provider at runtime (e.g., model switch).
    /// Requires `&mut self` — obtain via `runtime.lock().await`.
    pub fn set_client(&mut self, provider: Arc<dyn llm_trait::LlmProvider>) {
        self.runner.llm_engine.set_provider(provider);
    }

    /// Get the model override, if set.
    pub fn get_model_override(&self) -> Option<String> {
        self.runner.llm_engine.get_model_override()
    }

    /// Set a model override for all requests.
    ///
    /// This is used by sub-agents to specify their model tier (e.g., "lite").
    /// The override is applied to all ChatRequests before sending to the provider.
    pub fn set_model_override(&self, model: Option<String>) {
        self.runner.llm_engine.set_model_override(model);
    }

    pub fn tools_mut(&self) -> Arc<tokio::sync::RwLock<crate::tool::ToolRegistry>> {
        self.runner.tool_engine.tools_arc()
    }

    pub fn config(&self) -> tokio::sync::RwLockReadGuard<'_, AgentConfig> {
        self.runner.config.blocking_read()
    }

    /// 设置 reasoning effort（异步版本）
    pub async fn set_reasoning_effort(&self, effort: crate::llm::ReasoningEffort) {
        let mut config = self.runner.config.write().await;
        let mut reasoning = config.reasoning.take().unwrap_or_default();
        reasoning.effort = Some(effort);
        config.reasoning = Some(reasoning);
    }

    /// 设置 reasoning effort（同步版本，只在同步上下文中使用）
    pub fn set_reasoning_effort_sync(&self, effort: crate::llm::ReasoningEffort) {
        let mut config = self.runner.config.blocking_write();
        let mut reasoning = config.reasoning.take().unwrap_or_default();
        reasoning.effort = Some(effort);
        config.reasoning = Some(reasoning);
    }

    pub fn approval_handler(&self) -> Option<&Arc<dyn ApprovalHandler>> {
        self.runner.tool_engine.approval_handler()
    }

    pub fn tool_policy(&self) -> Option<&Arc<dyn ToolPolicy>> {
        self.runner.tool_engine.tool_policy()
    }

    pub async fn cached_approval(&self, session_id: &SessionId, action_key: &str) -> bool {
        self.runner
            .session_manager
            .cached_approval(session_id, action_key)
            .await
    }

    pub async fn cache_approval(&self, session_id: &SessionId, action_key: String) {
        self.runner
            .session_manager
            .cache_approval(session_id, action_key)
            .await
    }

    pub async fn save_checkpoint(
        &self,
        session_id: &SessionId,
        checkpoint: CheckpointData,
    ) -> AgentResult<()> {
        self.emit_event(RuntimeEvent::Checkpoint {
            session_id: session_id.clone(),
            checkpoint,
            agent_id: None,
            trace_id: None,
        });
        Ok(())
    }

    pub async fn load_checkpoint(
        &self,
        _session_id: &SessionId,
        _checkpoint: &CheckpointData,
    ) -> AgentResult<Option<CheckpointData>> {
        Ok(None)
    }

    pub async fn run<F>(&self, session_id: SessionId, on_event: F) -> AgentResult<RunOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        self.runner.run(session_id, on_event).await
    }

    pub async fn run_turn<F>(
        &self,
        session_id: SessionId,
        user_input: &str,
        on_event: F,
    ) -> AgentResult<RunOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        self.runner.run_turn(session_id, user_input, on_event).await
    }

    pub async fn run_turn_collect(
        &self,
        session_id: SessionId,
        user_input: &str,
    ) -> AgentResult<(Vec<RuntimeEvent>, RunOutcome)> {
        self.runner.run_turn_collect(session_id, user_input).await
    }

    pub async fn add_user_message(
        &self,
        session_id: &SessionId,
        text: impl Into<String>,
    ) -> AgentResult<()> {
        let text = text.into();
        self.with_session_mut(session_id, |session| {
            session.push_message(MessageRole::User, &text);
        })
        .await
    }

    pub async fn add_system_message(
        &self,
        session_id: &SessionId,
        text: impl Into<String>,
    ) -> AgentResult<()> {
        let text = text.into();
        self.with_session_mut(session_id, |session| {
            session.push_message(MessageRole::System, &text);
        })
        .await
    }

    pub async fn add_tool_result(
        &self,
        session_id: &SessionId,
        tool_call_id: &str,
        summary: impl Into<String>,
    ) -> AgentResult<()> {
        let summary = summary.into();
        self.with_session_mut(session_id, |session| {
            session.push_tool_result(tool_call_id, summary.clone());
        })
        .await
    }

    pub async fn get_messages(
        &self,
        session_id: &SessionId,
    ) -> AgentResult<Vec<crate::types::ChatMessage>> {
        let session = self.session_or_err(session_id).await?;
        Ok(session.chat_messages().to_vec())
    }

    /// Replace the chat messages for a session — only for persistence restore.
    /// Validates message sequence before applying.
    ///
    /// 仅供持久化恢复使用。
    pub async fn set_messages(
        &self,
        session_id: &SessionId,
        messages: Vec<crate::types::ChatMessage>,
    ) -> AgentResult<()> {
        self.with_session_mut(session_id, |session| session.set_chat_messages(messages))
            .await?
            .map_err(AgentError::internal)
    }

    pub async fn validate_session(&self, session_id: &SessionId) -> AgentResult<()> {
        if self
            .runner
            .session_manager
            .session(session_id)
            .await
            .is_none()
        {
            return Err(AgentError::session_not_found(session_id.id));
        }
        Ok(())
    }

    pub fn session_store(&self) -> Arc<dyn SessionStore> {
        self.runner.session_manager.session_store().clone()
    }

    // ── Observability hook ──

    /// Register a turn-end callback. The callback receives a [`TurnContext`]
    /// with raw data about the completed turn iteration. Consumers (e.g.
    /// phi-telemetry) use this to build their own metrics without agent-base
    /// knowing anything about metrics.
    pub fn on_turn_end<F>(&self, f: F)
    where
        F: Fn(&TurnContext) + Send + Sync + 'static,
    {
        self.runner
            .turn_end_callbacks
            .write()
            .unwrap()
            .push(Arc::new(f));
    }

    // --- Cancellation support ---

    /// Cancel the currently executing run_turn / run.
    /// No-op if there is no current execution.
    pub fn cancel(&self) {
        self.runner.cancel();
    }

    /// Reset the cancel token (called automatically before each run_turn)
    pub fn reset_cancel(&self) {
        self.runner.reset_cancel();
    }

    /// Get a clone of the cancel token
    pub fn cancel_token(&self) -> CancellationToken {
        self.runner.cancel_token()
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.runner.is_cancelled()
    }

    // ── Message Queue (P2) ──

    /// Push a steering message — will be processed at the start of the next turn
    /// in the current `run_managed()` loop.
    pub fn steer(&self, message: String) {
        self.runner.message_queue.steer(message);
    }

    /// Push a follow-up message — will be processed after the inner turn loop
    /// stops naturally (no tool calls or max turns).
    pub fn follow_up(&self, message: String) {
        self.runner.message_queue.follow_up(message);
    }

    /// Run the agent in managed mode with message queue support.
    ///
    /// This wraps `run_turn()` (or `run()`) in an outer loop: after the inner
    /// turn loop completes, any follow-up messages are drained and a new inner
    /// loop is started. Steering messages are drained automatically at each
    /// iteration of the inner turn loop.
    ///
    /// The `on_event` callback receives all events from every inner run.
    pub async fn run_managed<F>(
        &self,
        session_id: SessionId,
        user_input: &str,
        on_event: F,
    ) -> AgentResult<RunOutcome>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        self.runner
            .run_managed(session_id, user_input, on_event)
            .await
    }

    /// Set the drain mode for the message queues.
    pub fn set_queue_mode(&self, mode: crate::engine::runtime::message_queue::QueueMode) {
        self.runner.message_queue.set_mode(mode);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ReasoningEffort;
    use crate::types::{ChatMessage, RuntimeEvent, SessionId};
    use async_trait::async_trait;
    use llm_trait::{Capabilities, ChatRequest, ChatResponse, ChatStream, LlmError, ProviderInfo};

    struct StubProvider;

    #[async_trait]
    impl llm_trait::LlmProvider for StubProvider {
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
                name: "stub".to_string(),
                model: "stub".to_string(),
                version: None,
            }
        }
    }

    fn runtime() -> AgentRuntime {
        crate::engine::AgentBuilder::new(Arc::new(StubProvider))
            .build()
            .unwrap()
    }

    #[tokio::test]
    async fn create_session_and_lookup() {
        let rt = runtime();
        let id = rt.create_session().await;
        assert_eq!(id.id, 1);
        assert!(rt.session(&id).await.is_some());
        assert!(rt.session_or_err(&id).await.is_ok());
    }

    #[tokio::test]
    async fn add_messages_and_get() {
        let rt = runtime();
        let id = rt.create_session().await;
        rt.add_system_message(&id, "sys").await.unwrap();
        rt.add_user_message(&id, "hello").await.unwrap();
        let msgs = rt.get_messages(&id).await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert!(matches!(msgs[0], ChatMessage::System { .. }));
        assert!(matches!(msgs[1], ChatMessage::User { .. }));
    }

    #[tokio::test]
    async fn add_tool_result_appends_tool_message() {
        let rt = runtime();
        let id = rt.create_session().await;
        rt.add_tool_result(&id, "call_1", "done").await.unwrap();
        let msgs = rt.get_messages(&id).await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert!(matches!(msgs[0], ChatMessage::Tool { .. }));
    }

    #[tokio::test]
    async fn set_messages_replaces_history() {
        let rt = runtime();
        let id = rt.create_session().await;
        rt.add_user_message(&id, "old").await.unwrap();
        rt.set_messages(
            &id,
            vec![ChatMessage::system("sys"), ChatMessage::user("new")],
        )
        .await
        .unwrap();
        let msgs = rt.get_messages(&id).await.unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[tokio::test]
    async fn validate_session_errors_for_unknown() {
        let rt = runtime();
        let id = rt.create_session().await;
        assert!(rt.validate_session(&id).await.is_ok());
        let err = rt.validate_session(&SessionId::new(999)).await.unwrap_err();
        assert!(matches!(err, AgentError::SessionNotFound(_)));
    }

    #[test]
    fn config_and_set_reasoning_effort() {
        let rt = runtime();
        assert!(rt.config().system_prompt.is_none());

        rt.set_reasoning_effort_sync(ReasoningEffort::High);
        let cfg = rt.config();
        let effort = cfg.reasoning.as_ref().and_then(|r| r.effort.as_ref());
        assert!(matches!(effort, Some(ReasoningEffort::High)));
    }

    #[tokio::test]
    async fn session_store_is_available() {
        let rt = runtime();
        assert!(rt.session_store().list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn emit_and_subscribe_event() {
        let rt = runtime();
        let mut rx = rt.subscribe_runtime_events();
        rt.emit_event(RuntimeEvent::TextDelta {
            session_id: SessionId::new(1),
            text: "hi".into(),
            agent_id: None,
            trace_id: None,
        });
        let ev = rx.recv().await.unwrap();
        assert!(matches!(ev, RuntimeEvent::TextDelta { .. }));
    }

    #[tokio::test]
    async fn cancel_reset_and_is_cancelled() {
        let rt = runtime();
        assert!(!rt.is_cancelled());
        rt.cancel();
        assert!(rt.is_cancelled());
        rt.reset_cancel();
        assert!(!rt.is_cancelled());
    }
}
