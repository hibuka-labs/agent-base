//! Pure type definitions for the agent-base runtime.
//!
//! This crate contains data structures and enums with **zero runtime
//! dependencies** — no async, no I/O, no trait objects. It exists so that
//! downstream crates (SDKs, protocol layers, tool definitions) can depend on
//! just the types without pulling in the full agent-base runtime.
//!
//! agent-base re-exports everything from this crate, so existing
//! `use agent_base::Content` continues to work.

mod action;
mod approval;
mod checkpoint;
mod execution;
mod guard;
mod interaction;
mod message;
mod notify;
mod plan;
mod run_state;
mod session;
mod tool;

pub use action::{Action, ActionFormat, ActionOutcome, BatchStatus, Observation, ObservationEvent};
pub use approval::{ApprovalDecision, ApprovalRequest, RiskLevel};
pub use checkpoint::{
    CHECKPOINT_VERSION, CheckpointData, CheckpointStep, MigrationKind, MigrationSource,
    ToolResultData,
};
pub use execution::{FinishReason, RunOutcome};
pub use guard::{GuardCtx, GuardDecision};
pub use interaction::{AskAnswer, AskOption};
pub use message::MessageRole;
pub use notify::{Notice, NoticeKind};
pub use plan::{PlanItem, PlanStepStatus, UpdatePlanArgs};
pub use run_state::RunState;
pub use session::SessionId;
pub use tool::{
    ActivationContext, Content, ToolExposure, ToolMetadata, content_details, content_images,
    content_text,
};
