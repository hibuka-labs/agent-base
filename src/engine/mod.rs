mod approval;
pub mod auto_continue;
mod builder;
mod context;
pub mod executor;
pub mod max_turns_nudge;
mod middleware;
mod pipeline;
pub mod react_loop_guard;
mod recovery;
pub mod repeat_tool_limit;
mod runtime;
mod safety;
mod session;
mod session_store;
mod tool_enforcement;
mod turn_facts;

pub use approval::{AllowAllApprovalHandler, ApprovalHandler, DenyAllApprovalHandler};
pub use builder::AgentBuilder;
pub use context::{
    CompactionKind, CompactionOutcome, ContextCompaction, ContextWindowManager,
    estimate_messages_tokens, first_system_prompt,
};
pub use executor::{
    ActionBackend, ActionExecutor, DEFAULT_INTERACT_TIMEOUT, ExecCtx, ExecError,
    INTERRUPTED_SUMMARY, InteractionChannel, InteractionError, ParseError, ParseOutcome,
    SessionError, SessionHandle,
};
pub use middleware::{Middleware, PostLlmCtx, PreLlmCtx, UserMessageCtx};
pub use pipeline::{DefaultPipeline, ToolExecutionPipeline};
pub use react_loop_guard::{GuardCtx, GuardDecision, NoopGuard, ReactLoopGuard};
pub(crate) use runtime::EventBus;
pub use runtime::NoticeHandle;

pub use crate::types::{AgentResult, ApprovalDecision, ApprovalRequest, RiskLevel, SessionId};
pub use recovery::{
    ConsecutiveFailureRecovery, RetryOnError, StopOnError, ToolErrorAction, ToolErrorRecovery,
};
pub use runtime::AgentRuntime;
// The seam's input type — exported so out-of-kernel ActionExecutor
// implementations can name it (M3 T10).
pub use runtime::LlmTurnResult;
pub use runtime::QueueMode;
pub use runtime::SessionManager;
pub use safety::TurnToolLimitMiddleware;
pub use session::AgentSession;
#[cfg(feature = "fuzzing")]
pub use session::validate_message_sequence;
pub use session_store::{InMemorySessionStore, SessionStore};

pub use auto_continue::AutoContinueMiddleware;
pub use max_turns_nudge::{MaxTurnsNudgeConfig, MaxTurnsNudgeMiddleware};
pub use repeat_tool_limit::{RepeatToolLimitConfig, RepeatToolLimitMiddleware};
#[cfg(feature = "sqlite-session")]
pub use session_store::SqliteSessionStore;
pub use tool_enforcement::{ToolEnforcementConfig, ToolEnforcementMiddleware};
pub use turn_facts::TurnFactMiddleware;
