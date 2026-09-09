pub mod engine;
pub mod llm;
pub mod tool;
pub mod types;

// Re-export llm_trait for downstream use.
pub use llm_trait;

// ---------------------------------------------------------------------------
// Prelude — most commonly used types for `use agent_base::prelude::*`
// ---------------------------------------------------------------------------
pub mod prelude {
    pub use crate::engine::{
        AgentBuilder, AgentRuntime, AgentSession, AllowAllApprovalHandler, DenyAllApprovalHandler,
    };
    pub use crate::llm::ChatRequest;
    pub use crate::llm::LlmProvider as LlmProviderTrait;
    pub use crate::llm::{ChatResponse, ChatStream};
    pub use crate::tool::{
        AutoContinueTool, Content, DenyAllToolPolicy, Tool, ToolContext, ToolDecision,
        ToolMetadata, ToolPolicy, ToolRegistry, TypedTool,
    };
    pub use crate::types::{
        AgentConfig, AgentError, AgentResult, ChatMessage, FinishReason, Language, Message,
        MessageRole, RunOutcome, RuntimeEvent, SessionId, TurnContext, UserEvent,
    };
}

// ---------------------------------------------------------------------------
// Agent Runtime
// ---------------------------------------------------------------------------
pub use engine::{
    AgentBuilder, AgentRuntime, AgentSession, DefaultPipeline, InMemorySessionStore, QueueMode,
    SessionId, SessionStore, ToolExecutionPipeline,
};

// ---------------------------------------------------------------------------
// LLM Provider (llm-trait re-exports)
// ---------------------------------------------------------------------------
pub use llm::{ReasoningConfig, ReasoningEffort, StreamChunk, UsageInfo};

// ---------------------------------------------------------------------------
// Approval
// ---------------------------------------------------------------------------
pub use engine::{
    AllowAllApprovalHandler, ApprovalDecision, ApprovalHandler, ApprovalRequest,
    ConsecutiveFailureRecovery, ContextCompaction, ContextWindowManager, DenyAllApprovalHandler,
    GuardCtx, GuardDecision, Middleware, NoopGuard, PostLlmCtx, PreLlmCtx, ReactLoopGuard,
    RetryOnError, RiskLevel, StopOnError, ToolEnforcementConfig, ToolEnforcementMiddleware,
    ToolErrorAction, ToolErrorRecovery, TurnFactMiddleware, TurnToolLimitMiddleware, UserMessageCtx,
    estimate_messages_tokens, first_system_prompt,
};

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------
pub use tool::{
    ActivationContext, AutoContinueTool, Content, DenyAllToolPolicy, Tool, ToolContext,
    ToolDecision, ToolExposure, ToolMetadata, ToolPolicy, ToolRegistry, TypedTool, UpdatePlanTool,
};

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------
pub use types::{RuntimeEvent, UserEvent};

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------
pub use types::{AgentError, AgentResult, ErrorKind};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
pub use types::{
    AgentConfig, ChatMessage, CheckpointData, CheckpointStep, FinishReason, ImageAttachment,
    ImageDetail, Language, Message, MessageRole, PlanItem, PlanStepStatus, ResponseFormat,
    RetryConfig, RunOutcome, SafetyConfig, SessionConfig, ToolCallMessage, ToolResultData,
    TurnContext, UpdatePlanArgs,
};
