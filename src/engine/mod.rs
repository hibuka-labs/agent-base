mod approval;
pub mod auto_continue;
mod builder;
mod context;
pub mod max_turns_nudge;
mod middleware;
mod pipeline;
pub mod react_loop_guard;

mod recovery;
mod runtime;
mod safety;
mod session;
mod session_store;
mod tool_enforcement;
mod turn_facts;

pub use approval::{AllowAllApprovalHandler, ApprovalHandler, DenyAllApprovalHandler};
pub use builder::AgentBuilder;
pub use context::{
    ContextCompaction, ContextWindowManager, estimate_messages_tokens, first_system_prompt,
};
pub use middleware::{Middleware, PostLlmCtx, PreLlmCtx, UserMessageCtx};
pub use pipeline::{DefaultPipeline, ToolExecutionPipeline};
pub use react_loop_guard::{GuardCtx, GuardDecision, NoopGuard, ReactLoopGuard};
pub(crate) use runtime::EventBus;

pub use crate::types::{AgentResult, ApprovalDecision, ApprovalRequest, RiskLevel, SessionId};
pub use recovery::{
    ConsecutiveFailureRecovery, RetryOnError, StopOnError, ToolErrorAction, ToolErrorRecovery,
};
pub use runtime::AgentRuntime;
pub use runtime::QueueMode;
pub use runtime::SessionManager;
pub use safety::TurnToolLimitMiddleware;
pub use session::AgentSession;
#[cfg(feature = "fuzzing")]
pub use session::validate_message_sequence;
pub use session_store::{InMemorySessionStore, SessionStore};

pub use auto_continue::AutoContinueMiddleware;
pub use max_turns_nudge::{MaxTurnsNudgeConfig, MaxTurnsNudgeMiddleware};
#[cfg(feature = "sqlite-session")]
pub use session_store::SqliteSessionStore;
pub use tool_enforcement::{ToolEnforcementConfig, ToolEnforcementMiddleware};
pub use turn_facts::TurnFactMiddleware;
