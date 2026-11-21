use std::sync::Arc;

use tokio::sync::RwLock;

use crate::engine::approval::ApprovalHandler;
use crate::engine::pipeline::DefaultPipeline;
use crate::engine::recovery::{InvalidArgsHandler, ToolErrorRecovery};
use crate::engine::runtime::event_bus::EventBus;
use crate::engine::runtime::session_manager::SessionManager;
use crate::tool::{Content, ToolPolicy, ToolRegistry};
use crate::types::{AgentError, Language};

pub(crate) struct ToolEngine {
    pub(super) tools: Arc<RwLock<ToolRegistry>>,
    pub(super) approval_handler: Option<Arc<dyn ApprovalHandler>>,
    pub(super) tool_policy: Option<Arc<dyn ToolPolicy>>,
    pub(super) error_recovery: Arc<dyn ToolErrorRecovery>,
    pub(super) invalid_args_handler: Arc<dyn InvalidArgsHandler>,
    pub(super) event_bus: EventBus,
    pub(super) pipeline: DefaultPipeline,
}

#[derive(Debug)]
pub struct ToolExecutionResult {
    pub id: String,
    #[allow(dead_code)]
    pub name: String,
    pub output: Vec<Content>,
    /// The call left a human step pending (see
    /// [`crate::tool::ToolContext::mark_waiting_for_human`]).
    pub waiting_for_human: bool,
}

/// One failed tool call within an orchestrated batch.
#[derive(Debug)]
pub struct ToolFailure {
    pub id: String,
    pub error: AgentError,
}

/// Result of [`ToolEngine::orchestrate`]: successful tool calls plus any per-call
/// failures collected without aborting the rest of the batch.
#[derive(Debug)]
pub struct OrchestrateOutcome {
    pub results: Vec<ToolExecutionResult>,
    pub failures: Vec<ToolFailure>,
}

/// Grouped context passed through the tool execution call chain.
/// Reduces parameter count on `execute_tool` and `process_approval`.
#[derive(Clone)]
pub(crate) struct ExecutionContext {
    pub session_manager: SessionManager,
    pub llm_client: Option<Arc<dyn llm_trait::LlmProvider>>,
    pub language: Language,
    pub tool_timeout_ms: Option<u64>,
    pub max_output_chars: Option<usize>,
    pub cancel_token: tokio_util::sync::CancellationToken,
}
