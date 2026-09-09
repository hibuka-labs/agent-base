use std::sync::{Arc, Mutex, RwLock as StdRwLock};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;

use crate::engine::context::{ContextCompaction, ContextWindowManager};
use crate::engine::middleware::MiddlewareRef;
use crate::engine::react_loop_guard::{NoopGuard, ReactLoopGuard};
use crate::engine::runtime::event_bus::EventBus;
use crate::engine::runtime::llm_engine::LlmEngine;
use crate::engine::runtime::message_queue::MessageQueue;
use crate::engine::runtime::session_manager::SessionManager;
use crate::engine::runtime::tool_engine::ToolEngine;
use crate::types::ConvertToLlmFn;
use crate::types::{AgentConfig, AgentError, AgentResult, SessionId, TurnContext};

/// Turn-end callback type: receives TurnContext, no return value.
pub type TurnEndCallback = Arc<dyn Fn(&TurnContext) + Send + Sync>;

pub(crate) struct RuntimeCore {
    pub(crate) config: Arc<RwLock<AgentConfig>>,
    pub(crate) llm_engine: LlmEngine,
    pub(crate) tool_engine: ToolEngine,
    pub(crate) session_manager: SessionManager,
    pub(crate) event_bus: EventBus,
    pub(crate) context_manager: Option<ContextWindowManager>,
    pub(crate) middlewares: Vec<MiddlewareRef>,
    pub(crate) cancel_token: Mutex<CancellationToken>,
    /// Turn-end callbacks registered by consumers.
    /// Uses std::sync::RwLock — registration is cold-path and read-lock is brief.
    pub(crate) turn_end_callbacks: StdRwLock<Vec<TurnEndCallback>>,
    /// Dual-queue message system (steering + follow-up).
    pub(crate) message_queue: MessageQueue,
    /// Optional callback to transform messages before sending to LLM.
    pub(crate) convert_to_llm: Option<ConvertToLlmFn>,
    /// Guard for react loop completion detection.
    pub(crate) guard: Arc<dyn ReactLoopGuard>,
    /// Optional inline context compactor (e.g., agent-works ContextCompactor).
    pub(crate) context_compactor: Option<Arc<dyn ContextCompaction>>,
}

impl RuntimeCore {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: AgentConfig,
        llm_engine: LlmEngine,
        tool_engine: ToolEngine,
        session_manager: SessionManager,
        event_bus: EventBus,
        context_manager: Option<ContextWindowManager>,
        middlewares: Vec<MiddlewareRef>,
        convert_to_llm: Option<ConvertToLlmFn>,
        guard: Option<Arc<dyn ReactLoopGuard>>,
        context_compactor: Option<Arc<dyn ContextCompaction>>,
    ) -> Self {
        Self {
            config: Arc::new(RwLock::new(config)),
            llm_engine,
            tool_engine,
            session_manager,
            event_bus,
            context_manager,
            middlewares,
            cancel_token: Mutex::new(CancellationToken::new()),
            turn_end_callbacks: StdRwLock::new(Vec::new()),
            message_queue: MessageQueue::new(),
            convert_to_llm,
            guard: guard.unwrap_or_else(|| Arc::new(NoopGuard)),
            context_compactor,
        }
    }

    /// Get a clone of the current cancel token
    pub fn cancel_token(&self) -> CancellationToken {
        self.cancel_token.lock().unwrap().clone()
    }

    /// Reset the cancel token (called before each run_turn)
    pub fn reset_cancel(&self) {
        *self.cancel_token.lock().unwrap() = CancellationToken::new();
    }

    /// Send the cancellation signal
    pub fn cancel(&self) {
        self.cancel_token.lock().unwrap().cancel();
    }

    /// Check if cancellation has been requested
    pub fn is_cancelled(&self) -> bool {
        self.cancel_token.lock().unwrap().is_cancelled()
    }

    /// Get a clone of config (async version)
    pub async fn config_snapshot_async(&self) -> AgentConfig {
        self.config.read().await.clone()
    }

    pub async fn validate_session(&self, session_id: &SessionId) -> AgentResult<()> {
        if self.session_manager.session(session_id).await.is_none() {
            return Err(AgentError::session_not_found(session_id.id));
        }
        Ok(())
    }

    pub async fn with_session_mut<F, R>(&self, session_id: &SessionId, f: F) -> AgentResult<R>
    where
        F: FnOnce(&mut crate::engine::AgentSession) -> R,
    {
        self.session_manager.with_session_mut(session_id, f).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancel_token_initial_state() {
        // Create a simple RuntimeCore for testing
        // Note: we only test the cancel_token field behavior
        let token = CancellationToken::new();
        assert!(!token.is_cancelled());

        let token_clone = token.clone();
        assert!(!token_clone.is_cancelled());

        token.cancel();
        assert!(token.is_cancelled());
        assert!(token_clone.is_cancelled());
    }

    #[test]
    fn test_cancel_token_reset() {
        let token = CancellationToken::new();
        token.cancel();
        assert!(token.is_cancelled());

        // After reset, should be a new token
        let new_token = CancellationToken::new();
        assert!(!new_token.is_cancelled());
    }

    #[tokio::test]
    async fn test_cancel_token_select() {
        let token = CancellationToken::new();
        let token_clone = token.clone();

        // Spawn a task to cancel the token
        let handle = tokio::spawn(async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
            token_clone.cancel();
        });

        // Wait for cancellation
        token.cancelled().await;
        assert!(token.is_cancelled());

        handle.await.unwrap();
    }
}
