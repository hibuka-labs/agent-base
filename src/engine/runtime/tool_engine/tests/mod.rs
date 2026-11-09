use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::sync::RwLock;

use crate::engine::pipeline::DefaultPipeline;
use crate::engine::recovery::ToolErrorRecovery;
use crate::engine::runtime::event_bus::EventBus;
use crate::engine::runtime::session_manager::SessionManager;
use crate::engine::{
    AllowAllApprovalHandler, DenyAllApprovalHandler, InMemorySessionStore, StopOnError,
};
use crate::tool::{Content, Tool, ToolContext, ToolPolicy, ToolRegistry, content_text};
use crate::types::{
    AgentError, AgentResult, ApprovalRequest, AtomicU64SessionIdGenerator, Language, RiskLevel,
    RuntimeEvent, SessionConfig, SessionId, UserEvent,
};
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use super::*;

mod approval;
mod execution;
mod orchestrate;

/// Minimal tool that echoes its `text` argument back as `echo: <text>`.
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn description(&self) -> &'static str {
        "Echo back the provided text"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": { "text": { "type": "string" } }
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let text = args.get("text").and_then(Value::as_str).unwrap_or("");
        Ok(vec![Content::text(format!("echo: {text}"))])
    }
}

struct FailingTool;
#[async_trait]
impl Tool for FailingTool {
    fn name(&self) -> &'static str {
        "failing"
    }
    fn description(&self) -> &'static str {
        ""
    }
    fn schema(&self) -> Value {
        serde_json::json!({})
    }
    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        Err(AgentError::internal("simulated tool failure"))
    }
}

fn session_manager() -> SessionManager {
    SessionManager::new(
        Arc::new(AtomicU64SessionIdGenerator::default()),
        Arc::new(InMemorySessionStore::new()),
        SessionConfig::default(),
    )
}

fn ctx(session_manager: &SessionManager) -> ExecutionContext {
    ExecutionContext {
        session_manager: session_manager.clone(),
        llm_client: None,
        language: Language::En,
        tool_timeout_ms: None,
        max_output_chars: None,
        cancel_token: CancellationToken::new(),
    }
}
