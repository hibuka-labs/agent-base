use crate::engine::executor::{
    ActionBackend, ActionExecutor, ExecCtx, ExecError, ParseError, ParseOutcome, SessionError,
    SessionHandle,
};
use crate::engine::middleware::{Middleware, PostLlmCtx, PreLlmCtx, UserMessageCtx};
use crate::engine::runtime::LlmTurnResult;
use crate::engine::{AgentBuilder, AgentRuntime, DenyAllApprovalHandler, RetryOnError};
use crate::llm::StreamChunk;
use crate::tool::{Content, Tool, ToolContext, ToolPolicy};
use crate::types::{
    Action, ActionFormat, ActionOutcome, AgentError, AgentResult, ApprovalRequest, BatchStatus,
    ChatMessage, CheckpointData, CheckpointStep, MigrationKind, MigrationSource, Observation,
    RiskLevel, RunOutcome, RunState, RuntimeEvent, SessionId, ToolResultData, TurnContext,
};
use async_trait::async_trait;
use futures_core::Stream;
use llm_trait::LlmProvider;
use serde_json::Value;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::task::{Context, Poll};

mod cancel_paths;
mod ephemeral;
mod event_order;
mod guards;
mod integrity_retry;
mod middleware;
mod notice_pump;
mod resume_code_backend;
mod resume_recovery;
mod run_errors;
mod terminal_events;
mod truncation;
mod truncation_partial;
mod truncation_strikes;

// Shared fixtures used by the themed test modules below.

/// Minimal LLM provider for tests that don't need LLM calls.
struct DummyProvider;

#[async_trait]
impl LlmProvider for DummyProvider {
    async fn stream(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatStream, llm_trait::LlmError> {
        unimplemented!("not used")
    }

    async fn chat(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatResponse, llm_trait::LlmError> {
        Ok(llm_trait::ChatResponse {
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![],
            usage: Default::default(),
            finish_reason: llm_trait::response::FinishReason::Stop,
            raw: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> llm_trait::Capabilities {
        Default::default()
    }

    fn info(&self) -> llm_trait::ProviderInfo {
        llm_trait::ProviderInfo {
            name: "test".to_string(),
            model: "test".to_string(),
            version: None,
        }
    }
}

/// LLM provider whose stream immediately yields a cancellation error, exercising
/// the `e.is_cancelled()` branch of the LLM-stream error path.
struct CancelledStreamProvider;

#[async_trait]
impl LlmProvider for CancelledStreamProvider {
    async fn stream(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatStream, llm_trait::LlmError> {
        struct CancelledStream;
        impl Stream for CancelledStream {
            type Item = Result<StreamChunk, llm_trait::LlmError>;
            fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
                Poll::Ready(Some(Err(llm_trait::LlmError::llm("cancelled"))))
            }
        }
        Ok(llm_trait::ChatStream::new(Box::pin(CancelledStream)))
    }

    async fn chat(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatResponse, llm_trait::LlmError> {
        Ok(llm_trait::ChatResponse {
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![],
            usage: Default::default(),
            finish_reason: llm_trait::response::FinishReason::Stop,
            raw: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> llm_trait::Capabilities {
        Default::default()
    }

    fn info(&self) -> llm_trait::ProviderInfo {
        llm_trait::ProviderInfo {
            name: "test".to_string(),
            model: "test".to_string(),
            version: None,
        }
    }
}

/// Mock LLM that returns scripted responses — one Vec<StreamChunk> per call.
struct ScriptedProvider {
    script: Mutex<std::vec::IntoIter<Vec<StreamChunk>>>,
}

impl ScriptedProvider {
    fn new(script: Vec<Vec<StreamChunk>>) -> Self {
        Self {
            script: Mutex::new(script.into_iter()),
        }
    }
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn stream(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatStream, llm_trait::LlmError> {
        let chunks: Vec<Result<StreamChunk, llm_trait::LlmError>> = self
            .script
            .lock()
            .unwrap()
            .next()
            .unwrap_or_default()
            .into_iter()
            .map(Ok)
            .collect();
        Ok(llm_trait::ChatStream::new(Box::pin(
            futures_util::stream::iter(chunks),
        )))
    }

    async fn chat(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatResponse, llm_trait::LlmError> {
        Ok(llm_trait::ChatResponse {
            content: String::new(),
            reasoning_content: None,
            tool_calls: vec![],
            usage: Default::default(),
            finish_reason: llm_trait::response::FinishReason::Stop,
            raw: None,
            thinking_signature: None,
        })
    }

    fn capabilities(&self) -> llm_trait::Capabilities {
        llm_trait::Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_thinking: false,
            max_context_tokens: None,
            max_output_tokens: None,
        }
    }

    fn info(&self) -> llm_trait::ProviderInfo {
        llm_trait::ProviderInfo {
            name: "test".to_string(),
            model: "test".to_string(),
            version: None,
        }
    }
}

/// Minimal tool whose schema declares required fields — mirrors `spawn_agent`.
struct SpawnLikeTool;

#[async_trait]
impl Tool for SpawnLikeTool {
    fn name(&self) -> &'static str {
        "spawn_agent"
    }

    fn description(&self) -> &'static str {
        "Spawn a sub-agent"
    }

    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task_name": { "type": "string" },
                "message": { "type": "string" },
            },
            "required": ["task_name", "message"]
        })
    }

    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        Ok(vec![Content::text("spawned")])
    }
}

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

/// Tool that cancels the current run's token — exercises the cancellation path.
struct CancelTool;

#[async_trait]
impl Tool for CancelTool {
    fn name(&self) -> &'static str {
        "cancel"
    }

    fn description(&self) -> &'static str {
        "Cancel the current run"
    }

    fn schema(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        ctx.cancel_token.cancel();
        Ok(vec![Content::text("cancelled")])
    }
}

/// Tool that cancels the run's token and then blocks forever, so
/// `execute_tool`'s `cancel_token.cancelled()` branch fires and propagates a raw
/// `AgentError::Cancelled` (not wrapped in `ToolExecution`) to `handle_tool_error`.
struct CancelPendingTool;

#[async_trait]
impl Tool for CancelPendingTool {
    fn name(&self) -> &'static str {
        "cancel_pending"
    }

    fn description(&self) -> &'static str {
        "Cancel the current run mid-execution"
    }

    fn schema(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        ctx.cancel_token.cancel();
        // Never resolve — let execute_tool's cancellation branch fire instead.
        std::future::pending::<()>().await;
        unreachable!("cancelled tool should never resolve")
    }
}

/// Single-turn text-only script.
fn text_script(text: &str) -> Vec<Vec<StreamChunk>> {
    vec![vec![
        StreamChunk::Text(text.to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]]
}
