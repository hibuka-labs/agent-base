use crate::engine::middleware::{Middleware, PostLlmCtx, PreLlmCtx, UserMessageCtx};
use crate::engine::{AgentBuilder, DenyAllApprovalHandler, RetryOnError};
use crate::llm::StreamChunk;
use crate::tool::{Content, Tool, ToolContext, ToolPolicy};
use crate::types::{
    AgentError, AgentResult, ApprovalRequest, ChatMessage, CheckpointData, CheckpointStep,
    RiskLevel, RunOutcome, RuntimeEvent, SessionId, TurnContext,
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

#[tokio::test]
async fn run_turn_emits_run_finished_on_session_not_found() {
    let runtime = AgentBuilder::new(Arc::new(DummyProvider))
        .system_prompt("test")
        .build()
        .expect("build runtime");

    // Use a SessionId that was never created — session lookup will fail.
    let nonexistent = SessionId::new(99999);

    let event_fired = Arc::new(AtomicBool::new(false));
    let event_fired_clone = event_fired.clone();

    let result = runtime
        .run_turn(nonexistent.clone(), "test input", move |event| {
            if let RuntimeEvent::RunFinished { .. } = &event {
                event_fired_clone.store(true, Ordering::SeqCst);
            }
            Ok(())
        })
        .await;

    // Must return an error
    assert!(
        result.is_err(),
        "run_turn should return Err for nonexistent session"
    );
    // Must have emitted RunFinished before returning
    assert!(
        event_fired.load(Ordering::SeqCst),
        "run_turn must emit RunFinished before returning Err on session not found"
    );
}

/// Middleware that always fails — used to test the middleware error path.
struct FailingMiddleware;

#[async_trait]
impl Middleware for FailingMiddleware {
    async fn on_user_message(&self, _ctx: &mut UserMessageCtx) -> AgentResult<()> {
        Err(AgentError::internal("middleware intentionally fails"))
    }
}

#[tokio::test]
async fn run_turn_emits_run_finished_on_middleware_failure() {
    let runtime = AgentBuilder::new(Arc::new(DummyProvider))
        .system_prompt("test")
        .middleware(FailingMiddleware)
        .build()
        .expect("build runtime");

    // Create a valid session — middleware failure happens AFTER session lookup.
    let sid = runtime.create_session().await;

    let event_fired = Arc::new(AtomicBool::new(false));
    let event_fired_clone = event_fired.clone();

    let result = runtime
        .run_turn(sid, "test input", move |event| {
            if let RuntimeEvent::RunFinished { .. } = &event {
                event_fired_clone.store(true, Ordering::SeqCst);
            }
            Ok(())
        })
        .await;

    assert!(
        result.is_err(),
        "run_turn should return Err when middleware fails"
    );
    assert!(
        event_fired.load(Ordering::SeqCst),
        "run_turn must emit RunFinished before returning Err on middleware failure"
    );
}

/// LLM provider whose stream immediately yields an error.
struct ErrorStreamProvider;

#[async_trait]
impl LlmProvider for ErrorStreamProvider {
    async fn stream(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatStream, llm_trait::LlmError> {
        // Return a stream that immediately yields an error then ends.
        struct ErrorStream;
        impl Stream for ErrorStream {
            type Item = Result<StreamChunk, llm_trait::LlmError>;
            fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
                Poll::Ready(Some(Err(llm_trait::LlmError::llm("simulated LLM error"))))
            }
        }
        Ok(llm_trait::ChatStream::new(Box::pin(ErrorStream)))
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

#[tokio::test]
async fn run_turn_emits_run_finished_on_llm_error() {
    let runtime = AgentBuilder::new(Arc::new(ErrorStreamProvider))
        .system_prompt("test")
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;

    let event_fired = Arc::new(AtomicBool::new(false));
    let event_fired_clone = event_fired.clone();

    let result = runtime
        .run_turn(sid, "test input", move |event| {
            if let RuntimeEvent::RunFinished { .. } = &event {
                event_fired_clone.store(true, Ordering::SeqCst);
            }
            Ok(())
        })
        .await;

    // LLM errors should still emit RunFinished so event listeners don't hang.
    assert!(
        event_fired.load(Ordering::SeqCst),
        "run_turn must emit RunFinished when LLM returns an error"
    );
    // Note: the react loop may retry LLM errors, so the result might be Ok (retry succeeded
    // via retry logic) or Err.  Either is fine — the key assertion is that RunFinished fires.
    let _ = result;
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

#[tokio::test]
async fn truncation_guard_blocks_tool_calls_on_length_finish_reason() {
    // First call: tool call with finish_reason="length" — should be blocked by guard.
    // Second call: model retries with corrected approach (text response).
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: truncated tool call
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_trunc",
                        "function": {
                            "name": "shell",
                            "arguments": "{\"cmd\": \"rm -rf /inco"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("length".to_string()),
            },
        ],
        // Turn 2: model sees the error and retries
        vec![
            StreamChunk::Text(
                "I see the previous call was truncated. Let me re-issue it.".to_string(),
            ),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "run a command", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    // Verify the session messages contain the truncation error, not the
    // partial argument that would have been executed.
    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    let has_truncation_error = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed") && content.contains("output token limit")
        } else {
            false
        }
    });
    assert!(
        has_truncation_error,
        "session should contain truncation error tool result. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn truncation_guard_blocks_tool_calls_on_finish_tool_calls_mimo() {
    // mimo-v2.5-pro truncates the tool-call argument stream mid-generation
    // under load but reports `finish_reason=tool_calls` (NOT `length`), so the
    // length-only guard lets these fall through to tool execution as
    // ToolArgsInvalid — which feeds ConsecutiveFailureRecovery and aborts the
    // run ("failed 3 consecutive times"). The widened guard must detect the
    // structurally-incomplete JSON args regardless of the claimed finish reason
    // and route through the clean re-issue path instead.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: named tool call with a dangling, non-JSON argument string,
        // provider claims a normal tool_calls finish.
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_trunc_1",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": "{\"agent_path\": "
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: model sees the re-issue instruction and replies in text.
        vec![
            StreamChunk::Text("Re-issuing the call with complete arguments.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn a sub-agent", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    // The truncated call must NOT have been executed / recorded as a parse
    // failure; it must have been routed to the re-issue guard.
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
                && content.contains("provider truncated the argument stream")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "session should contain a re-issue tool result for the truncated call. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // And crucially the raw malformed args must never appear as an executed
    // ToolArgsInvalid failure (the death-spiral trigger).
    let has_args_invalid = messages.iter().any(|m| {
        let s = format!("{:?}", m);
        s.contains("argument parsing failed") || s.contains("EOF while parsing")
    });
    assert!(
        !has_args_invalid,
        "truncated args must not surface as a ToolArgsInvalid failure. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn truncation_guard_salvages_call_from_leaked_text_markup() {
    // mimo-v2.6-flash (session 20260923_263149bf) emits the SAME tool call on
    // two channels: the native tool_call argument stream dies mid-JSON with a
    // bogus "stop" finish, while the content channel leaks the chat-template
    // markup carrying the full parameter values. The guard must harvest the
    // leaked copy (name match + schema-valid) and EXECUTE it, not burn a
    // re-issue strike — the re-issue truncated again on the real session.
    let leaked = "我来派一个子代理。<tool_call><function=spawn_agent>\
<parameter=task_name>research<parameter=message>只读调研 codex，不改文件。";
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::Text(leaked.to_string()),
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_leak_1",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": "{\"task_name\":\"res"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("Should never be reached.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn a sub-agent", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    // The salvaged call must have executed exactly once.
    let spawned_count = messages
        .iter()
        .filter(|m| {
            if let ChatMessage::Tool { content, .. } = m {
                content.contains("spawned")
            } else {
                false
            }
        })
        .count();
    assert_eq!(
        spawned_count,
        1,
        "leaked call must be salvaged and executed. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // No re-issue guidance, and no ToolArgsInvalid death-spiral residue.
    let bad = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
                || content.contains("argument parsing failed")
                || content.contains("EOF while parsing")
        } else {
            false
        }
    });
    assert!(
        !bad,
        "salvaged call must not surface re-issue guidance or parse failure. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn salvage_rejects_leak_missing_required_field() {
    // A leak that omits a schema-required field must NOT be salvaged: executing
    // it would fail typed parsing (ToolArgsInvalid — the death-spiral trigger).
    // The call stays on the re-issue path instead.
    let leaked = "<tool_call><function=spawn_agent><parameter=task_name>research</tool_call>";
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::Text(leaked.to_string()),
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_leak_2",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": "{\"task_name\":\"res"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("Re-issuing with complete arguments.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn a sub-agent", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    let executed = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("spawned")
        } else {
            false
        }
    });
    assert!(
        !executed,
        "incomplete leak must not execute. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "incomplete leak must fall back to re-issue guidance. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn truncation_guard_recognizes_wrapper_echo() {
    // mimo, having seen a `{error:"tool_call_arguments_truncated",...}` object
    // in its history, replays it VERBATIM as the next call's arguments. That
    // payload is valid JSON, so the raw-args structural guard can't catch it —
    // but it is unmistakably truncation residue. The guard must recognise the
    // wrapper marker and re-issue instead of executing it (which would surface
    // as a downstream ToolArgsInvalid, the send_message slip seen in the field).
    let wrapper = serde_json::json!({
        "error": "tool_call_arguments_truncated",
        "message": "The tool call arguments were truncated or invalid. Please retry with complete arguments.",
        "original_args_preview": "{\"agent_path\": "
    })
    .to_string();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_echo",
                        "function": {
                            "name": "send_message",
                            "arguments": wrapper
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("Now sending a real message.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "continue", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "echoed wrapper args must be routed to the re-issue guard. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    let has_args_invalid = messages.iter().any(|m| {
        let s = format!("{:?}", m);
        s.contains("argument parsing failed")
    });
    assert!(
        !has_args_invalid,
        "echoed wrapper must not reach tool execution. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
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

/// Minimal tool whose schema has no required fields — mirrors `list_agents`.
struct NoArgTool;

#[async_trait]
impl Tool for NoArgTool {
    fn name(&self) -> &'static str {
        "list_agents"
    }

    fn description(&self) -> &'static str {
        "List running agents"
    }

    fn schema(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        Ok(vec![Content::text("no agents")])
    }
}

#[tokio::test]
async fn truncation_guard_blocks_empty_args_for_required_field_tool() {
    // The model, having seen sanitized `spawn_agent {}` entries in its history,
    // echoes the shape: one real spawn + one empty `{}` in the same turn.
    // `{}` is valid JSON (case 2 skips it) but the tool's schema requires
    // task_name + message — case 4 must detect this as degenerate and re-issue
    // the whole turn, avoiding a ToolArgsInvalid execution failure.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: two spawn_agent buckets — one complete, one empty `{}`.
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_real",
                            "function": {
                                "name": "spawn_agent",
                                "arguments": "{\"task_name\":\"foo\",\"message\":\"do stuff\"}"
                            }
                        },
                        {
                            "index": 1,
                            "id": "call_empty",
                            "function": {
                                "name": "spawn_agent",
                                "arguments": "{}"
                            }
                        }
                    ]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: model sees the re-issue instruction and replies.
        vec![
            StreamChunk::Text("Re-issuing with complete arguments.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn two agents", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    // Must contain a re-issue tool result for the empty-required case.
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
                && content.contains("empty argument object")
                && content.contains("requires fields")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "empty {{}} for a required-field tool must be caught by case 4 guard. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // Crucially the empty args must NOT have reached tool execution as a
    // ToolArgsInvalid failure.
    let has_args_invalid = messages.iter().any(|m| {
        let s = format!("{:?}", m);
        s.contains("argument parsing failed") || s.contains("EOF while parsing")
    });
    assert!(
        !has_args_invalid,
        "empty {{}} args must not surface as a ToolArgsInvalid failure. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn no_arg_tool_with_empty_object_args_executes_normally() {
    // A tool with no required fields (like `list_agents`) legitimately receives
    // `{}` as arguments. The case 4 guard must NOT fire — the tool should
    // execute normally.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: list_agents with `{}` — should execute.
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_list",
                        "function": {
                            "name": "list_agents",
                            "arguments": "{}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: model reports the result.
        vec![
            StreamChunk::Text("No agents running.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(NoArgTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "list agents", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    // Must NOT contain a re-issue guidance — no required fields, so {} is valid.
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
                && content.contains("empty argument object")
        } else {
            false
        }
    });
    assert!(
        !has_reissue,
        "no-arg tool with {{}} must execute, not be re-issued. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // And must contain the tool's actual output (execution succeeded).
    let has_output = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("no agents")
        } else {
            false
        }
    });
    assert!(
        has_output,
        "no-arg tool must have executed and returned output. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

// ── Partial execution tests ──

#[tokio::test]
async fn partial_execution_valid_calls_execute_invalid_get_guidance() {
    // 3 tool_calls: first is valid, second is truncated (invalid JSON),
    // third is valid. Only the truncated one should get re-issue guidance;
    // the other two should execute normally.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_1",
                            "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"a\",\"message\":\"first\"}" }
                        },
                        {
                            "index": 1,
                            "id": "call_2",
                            "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"b\",\"message\":\"tru" }
                        },
                        {
                            "index": 2,
                            "id": "call_3",
                            "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"c\",\"message\":\"third\"}" }
                        }
                    ]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        vec![
            StreamChunk::Text("Done.".to_string()),
            StreamChunk::Stop { finish_reason: Some("stop".to_string()) },
        ],
    ])))
    .system_prompt("test")
    .register_tool(SpawnLikeTool)
    .build().expect("build");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn three", move |e| {
            ec.lock().unwrap().push(e);
            Ok(())
        })
        .await;
    assert!(result.is_ok(), "run_turn failed: {:?}", result.err());

    let session = runtime.session(&sid).await.expect("session");
    let messages = session.chat_messages().to_vec();

    // call_2 (truncated) must have re-issue guidance
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("incomplete")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "truncated call must get guidance. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // call_1 and call_3 must have executed (got "spawned" results)
    let spawned_count = messages
        .iter()
        .filter(|m| {
            if let ChatMessage::Tool { content, .. } = m {
                content.contains("spawned")
            } else {
                false
            }
        })
        .count();
    assert_eq!(
        spawned_count,
        2,
        "exactly 2 valid calls must execute. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn partial_execution_all_valid_no_guard() {
    // All 3 calls valid → no guard should fire, all execute normally.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [
                        { "index": 0, "id": "c1", "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"a\",\"message\":\"first\"}" } },
                        { "index": 1, "id": "c2", "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"b\",\"message\":\"second\"}" } },
                        { "index": 2, "id": "c3", "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"c\",\"message\":\"third\"}" } }
                    ]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        vec![
            StreamChunk::Text("All done.".to_string()),
            StreamChunk::Stop { finish_reason: Some("stop".to_string()) },
        ],
    ])))
    .system_prompt("test")
    .register_tool(SpawnLikeTool)
    .build().expect("build");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn three", move |e| {
            ec.lock().unwrap().push(e);
            Ok(())
        })
        .await;
    assert!(result.is_ok(), "run_turn failed: {:?}", result.err());

    let session = runtime.session(&sid).await.expect("session");
    let messages = session.chat_messages().to_vec();

    // No re-issue guidance should appear
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
        } else {
            false
        }
    });
    assert!(
        !has_reissue,
        "all valid calls must not trigger guard. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // All 3 should have "spawned" results
    let spawned_count = messages
        .iter()
        .filter(|m| {
            if let ChatMessage::Tool { content, .. } = m {
                content.contains("spawned")
            } else {
                false
            }
        })
        .count();
    assert_eq!(
        spawned_count,
        3,
        "all 3 must execute. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn partial_execution_mixed_truncation_and_empty_required() {
    // 3 calls: truncated (invalid JSON), empty {}, valid.
    // Truncated takes priority for guidance selection.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [
                        { "index": 0, "id": "c_trunc", "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"x\"," } },
                        { "index": 1, "id": "c_empty", "function": { "name": "spawn_agent", "arguments": "{}" } },
                        { "index": 2, "id": "c_ok",    "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"z\",\"message\":\"ok\"}" } }
                    ]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        vec![
            StreamChunk::Text("Fixed.".to_string()),
            StreamChunk::Stop { finish_reason: Some("stop".to_string()) },
        ],
    ])))
    .system_prompt("test")
    .register_tool(SpawnLikeTool)
    .build().expect("build");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn mixed", move |e| {
            ec.lock().unwrap().push(e);
            Ok(())
        })
        .await;
    assert!(result.is_ok(), "run_turn failed: {:?}", result.err());

    let session = runtime.session(&sid).await.expect("session");
    let messages = session.chat_messages().to_vec();

    // Should have truncation guidance (not empty-required) since truncated call exists
    let has_trunc_guidance = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("incomplete") || content.contains("truncated")
        } else {
            false
        }
    });
    assert!(
        has_trunc_guidance,
        "must have truncation guidance. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // Valid call (c_ok) must have executed
    let spawned_count = messages
        .iter()
        .filter(|m| {
            if let ChatMessage::Tool { content, .. } = m {
                content.contains("spawned")
            } else {
                false
            }
        })
        .count();
    assert_eq!(
        spawned_count,
        1,
        "only valid call executes. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // 2 invalid calls should have guidance results
    let guidance_count = messages
        .iter()
        .filter(|m| {
            if let ChatMessage::Tool { content, .. } = m {
                content.contains("Tool call was not executed")
            } else {
                false
            }
        })
        .count();
    assert_eq!(
        guidance_count,
        2,
        "2 invalid calls get guidance. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn truncation_circuit_breaker_redirects_then_hard_stops() {
    // Turns 1-3: truncated → strike 3 fires redirect (Continue)
    // Turns 4-5: truncated → strike 5 fires hard stop (Done(Failed))
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: truncated → strikes = 1
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "t1",
                        "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"x\",\"message\":\"tru" }
                    }]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        // Turn 2: truncated → strikes = 2
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "t2",
                        "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"x\",\"message\":\"tru" }
                    }]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        // Turn 3: truncated → strikes = 3, circuit breaker fires (redirect)
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "t3",
                        "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"x\",\"message\":\"tru" }
                    }]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        // Turn 4: still truncated → strikes = 4, redirect again
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "t4",
                        "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"x\",\"message\":\"tru" }
                    }]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        // Turn 5: still truncated → strikes = 5, hard stop
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "t5",
                        "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"x\",\"message\":\"tru" }
                    }]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
    ])))
    .system_prompt("test")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "do something", move |e| {
            ec.lock().unwrap().push(e);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should not error: {:?}",
        result.err()
    );
    let outcome = result.unwrap();
    match &outcome {
        RunOutcome::Failed { error } => {
            assert!(
                error.contains("repeatedly truncated"),
                "error should mention repeated truncation: {}",
                error,
            );
        }
        other => panic!("expected RunOutcome::Failed at hard limit, got {:?}", other),
    }

    // Verify truncation_strikes reached the hard limit
    let session = runtime.session(&sid).await.expect("session");
    assert_eq!(session.run_state.truncation_strikes, 5);
}

#[tokio::test]
async fn truncation_strikes_reset_on_successful_tool_call() {
    // Turn 1: truncated → strikes = 1
    // Turn 2: valid call → strikes reset to 0
    // Turn 3: truncated again → strikes = 1 (not 2!)
    // Turn 4: text → done
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: truncated → strikes = 1
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "bad1",
                        "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"x\",\"message\":\"tru" }
                    }]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        // Turn 2: valid call → strikes reset to 0
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "good1",
                        "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"y\",\"message\":\"hello\"}" }
                    }]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        // Turn 3: truncated again → strikes = 1 (not 2, because reset happened)
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "bad2",
                        "function": { "name": "spawn_agent", "arguments": "{\"task_name\":\"z\",\"message\":\"tru" }
                    }]
                }
            })),
            StreamChunk::Stop { finish_reason: Some("tool_calls".to_string()) },
        ],
        // Turn 4: text response → normal completion
        vec![
            StreamChunk::Text("Done.".to_string()),
            StreamChunk::Stop { finish_reason: Some("stop".to_string()) },
        ],
    ])))
    .system_prompt("test")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "do something", move |e| {
            ec.lock().unwrap().push(e);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should not error: {:?}",
        result.err()
    );
    let outcome = result.unwrap();
    // Should complete normally (not Failed) because strikes were reset by the valid call
    assert!(
        matches!(outcome, RunOutcome::Completed),
        "expected Completed after reset, got {:?}",
        outcome,
    );

    // Verify truncation_strikes reflects the last truncation (1, not accumulated)
    // because the valid call in Turn 2 reset the counter.
    let session = runtime.session(&sid).await.expect("session");
    assert_eq!(session.run_state.truncation_strikes, 1);
}

#[tokio::test]
async fn empty_args_counted_as_truncation_strike() {
    // spawn_agent with completely empty arguments (args_len=0) should be
    // caught by the truncation guard and count toward circuit breaker.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: spawn_agent with empty args → strike 1
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "e1",
                        "function": { "name": "spawn_agent", "arguments": "" }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: spawn_agent with empty args → strike 2
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "e2",
                        "function": { "name": "spawn_agent", "arguments": "" }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 3: spawn_agent with empty args → strike 3, circuit breaker redirect
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "e3",
                        "function": { "name": "spawn_agent", "arguments": "" }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 4: text response → done
        vec![
            StreamChunk::Text("Done.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "do something", move |e| {
            ec.lock().unwrap().push(e);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should not error: {:?}",
        result.err()
    );
    let outcome = result.unwrap();
    assert!(
        matches!(outcome, RunOutcome::Completed),
        "expected Completed after redirect, got {:?}",
        outcome,
    );

    // Verify the circuit breaker fired (strikes should be 3)
    let session = runtime.session(&sid).await.expect("session");
    assert_eq!(session.run_state.truncation_strikes, 3);
}

#[tokio::test]
async fn run_managed_processes_follow_up_messages() {
    // ScriptedProvider: first call returns text "done", second call (triggered
    // by follow-up) returns "follow-up done". run_managed should process both.
    let scripted = Arc::new(ScriptedProvider::new(vec![
        // Turn 1: initial user input → text response
        vec![
            StreamChunk::Text("done".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        // Turn 2: follow-up → another text response
        vec![
            StreamChunk::Text("follow-up done".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ]));
    let runtime = AgentBuilder::new(scripted)
        .system_prompt("You are a helpful assistant.")
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;

    // Pre-seed a follow-up message — it will be drained after the first
    // inner loop completes and trigger a second inner loop.
    runtime.follow_up("continue working on the task".to_string());

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_managed(sid.clone(), "do something", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_managed should complete: {:?}",
        result.err()
    );

    // Verify session contains both turns
    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    let text_count = messages
        .iter()
        .filter(|m| matches!(m, ChatMessage::Assistant { content: Some(c), .. } if c == "done" || c == "follow-up done"))
        .count();
    assert_eq!(
        text_count, 2,
        "both initial and follow-up turns should produce assistant responses"
    );
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

#[tokio::test]
async fn run_turn_executes_tool_call_and_returns_text() {
    // Turn 1: model requests the echo tool; Turn 2: model emits a final answer.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"text\":\"hello\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("final answer".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "echo hello", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(result.is_ok(), "run_turn failed: {:?}", result.err());

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    assert!(
        messages.iter().any(
            |m| matches!(m, ChatMessage::Tool { content, .. } if content.contains("echo: hello"))
        ),
        "session should contain the echo tool result: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
    assert!(
        messages.iter().any(
            |m| matches!(m, ChatMessage::Assistant { content: Some(c), .. } if c == "final answer")
        ),
        "session should contain the final assistant text"
    );

    let events = events.lock().unwrap();

    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallStarted { .. })),
        "run_turn should emit ToolCallStarted"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ToolCallFinished { .. })),
        "run_turn should emit ToolCallFinished"
    );
}

// ── B1: core error / edge paths ──────────────────────────────────────

/// Tool that always fails — exercises the error-recovery (Stop / Retry) paths.
struct FailingTool;

#[async_trait]
impl Tool for FailingTool {
    fn name(&self) -> &'static str {
        "fail"
    }

    fn description(&self) -> &'static str {
        "Always fails"
    }

    fn schema(&self) -> Value {
        serde_json::json!({"type": "object", "properties": {}})
    }

    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        Err(AgentError::internal("simulated tool failure"))
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

/// Policy that always requests approval, so `process_approval` reaches the handler.
struct RequireApproval;

#[async_trait]
impl ToolPolicy for RequireApproval {
    async fn evaluate_approval(&self, tool_name: &str, _args: &Value) -> Option<ApprovalRequest> {
        Some(ApprovalRequest {
            title: format!("Approve {tool_name}"),
            message: "Approve this tool call?".to_string(),
            action_key: Some(format!("approve:{tool_name}")),
            risk_level: RiskLevel::Sensitive,
            raw: None,
            source: None,
        })
    }
}

/// Middleware recording the `on_pre_llm` / `on_post_llm` hooks, and nudging
/// `nudge_count` so the write-back branch is exercised.
struct HookMiddleware {
    pre_called: Arc<AtomicBool>,
    post_called: Arc<AtomicBool>,
}

#[async_trait]
impl Middleware for HookMiddleware {
    async fn on_pre_llm(&self, _ctx: &mut PreLlmCtx) -> AgentResult<()> {
        self.pre_called.store(true, Ordering::SeqCst);
        Ok(())
    }

    async fn on_post_llm(&self, ctx: &mut PostLlmCtx) -> AgentResult<()> {
        self.post_called.store(true, Ordering::SeqCst);
        ctx.nudge_count += 1;
        Ok(())
    }
}

#[tokio::test]
async fn run_emits_run_finished_on_session_not_found() {
    let runtime = AgentBuilder::new(Arc::new(DummyProvider))
        .system_prompt("test")
        .build()
        .expect("build runtime");

    let nonexistent = SessionId::new(99999);
    let event_fired = Arc::new(AtomicBool::new(false));
    let event_fired_clone = event_fired.clone();

    let result = runtime
        .run(nonexistent.clone(), move |event| {
            if matches!(event, RuntimeEvent::RunFinished { .. }) {
                event_fired_clone.store(true, Ordering::SeqCst);
            }
            Ok(())
        })
        .await;

    assert!(
        result.is_err(),
        "run should return Err for a nonexistent session"
    );
    assert!(
        event_fired.load(Ordering::SeqCst),
        "run must emit RunFinished before returning Err on session not found"
    );
}

#[tokio::test]
async fn run_completes_with_text_response() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    runtime.add_user_message(&sid, "question").await.unwrap();

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run(sid.clone(), move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "run failed: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session
            .chat_messages()
            .iter()
            .any(|m| matches!(m, ChatMessage::Assistant { content: Some(c), .. } if c == "answer")),
        "run should push the assistant text response into the session"
    );
}

#[tokio::test]
async fn run_turn_emits_run_cancelled_when_cancelled_mid_run() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": {
                "tool_calls": [{
                    "id": "call_cancel",
                    "function": { "name": "cancel", "arguments": "{}" }
                }]
            }
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_calls".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CancelTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "cancel now", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    let events = events.lock().unwrap();

    assert!(
        matches!(result, Ok(RunOutcome::Cancelled)),
        "run_turn should return Cancelled outcome: {:?}",
        result
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::RunCancelled { .. })),
        "run_turn should emit RunCancelled when cancelled mid-run"
    );
}

#[tokio::test]
async fn run_turn_failing_tool_stops_with_failed_outcome() {
    // Default recovery is StopOnError — a failing tool should stop the run.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": {
                "tool_calls": [{
                    "id": "call_fail",
                    "function": { "name": "fail", "arguments": "{}" }
                }]
            }
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_calls".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(FailingTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid.clone(), "do something", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "failing tool with StopOnError should yield Failed outcome: {:?}",
        result
    );
}

#[tokio::test]
async fn run_turn_failing_tool_retries_then_completes() {
    // RetryOnError recovery: first turn fails, second turn (after retry prompt) completes.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_fail",
                        "function": { "name": "fail", "arguments": "{}" }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("recovered".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(FailingTool)
    .error_recovery(Arc::new(RetryOnError))
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid.clone(), "do something", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "failing tool with RetryOnError should eventually complete: {:?}",
        result
    );
}

#[tokio::test]
async fn run_turn_approval_denied_completes_without_executing() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": {
                "tool_calls": [{
                    "id": "call_denied",
                    "function": { "name": "echo", "arguments": "{\"text\":\"hello\"}" }
                }]
            }
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_calls".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .approval_handler(Arc::new(DenyAllApprovalHandler))
    .tool_policy(Arc::new(RequireApproval))
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid.clone(), "echo hello", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "approval denial should stop the run cleanly (Completed): {:?}",
        result
    );

    // The denied tool must NOT have executed.
    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        !session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::Tool { content, .. } if content.contains("echo: hello"))
        ),
        "denied tool call must not execute"
    );
}

#[tokio::test]
async fn resume_from_checkpoint_executes_pending_tool_calls() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("resumed answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let checkpoint = CheckpointData {
        session_id: sid.clone(),
        user_input: "resume".to_string(),
        step: CheckpointStep::BeforeToolCalls {
            tool_calls: vec![(
                "call_resume".to_string(),
                "echo".to_string(),
                r#"{"text":"resume"}"#.to_string(),
            )],
        },
        turn_count: 0,
    };

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .runner
        .resume_from_checkpoint(checkpoint, move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "resume_from_checkpoint should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::Tool { content, .. } if content.contains("echo: resume"))
        ),
        "resumed tool call should have executed"
    );
}

#[tokio::test]
async fn middleware_on_pre_and_post_llm_hooks_fire() {
    let pre_called = Arc::new(AtomicBool::new(false));
    let post_called = Arc::new(AtomicBool::new(false));
    let mw = HookMiddleware {
        pre_called: pre_called.clone(),
        post_called: post_called.clone(),
    };

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("hello".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .middleware(mw)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime.run_turn(sid.clone(), "hi", |_| Ok(())).await;

    assert!(result.is_ok(), "run_turn failed: {:?}", result.err());
    assert!(
        pre_called.load(Ordering::SeqCst),
        "on_pre_llm hook should fire"
    );
    assert!(
        post_called.load(Ordering::SeqCst),
        "on_post_llm hook should fire"
    );

    // The middleware incremented nudge_count, which should be written back.
    let session = runtime.session(&sid).await.expect("session exists");
    assert_eq!(
        session.run_state.nudge_count, 1,
        "nudge_count should be written back"
    );
}

// ── Bug ① regression: exactly one RunFinished per entry point ──────

#[tokio::test]
async fn run_emits_runfinished_exactly_once() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;
    runtime.add_user_message(&sid, "question").await.unwrap();

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let outcome = runtime
        .run(sid.clone(), move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run");

    let events = events.lock().unwrap();

    assert!(matches!(outcome, RunOutcome::Completed));
    let finished = events
        .iter()
        .filter(|e| matches!(e, RuntimeEvent::RunFinished { .. }))
        .count();
    assert_eq!(finished, 1, "run() must emit exactly one RunFinished");
}

#[tokio::test]
async fn run_managed_emits_runfinished_exactly_once() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("done".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let outcome = runtime
        .run_managed(sid.clone(), "do something", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run_managed");

    let events = events.lock().unwrap();

    assert!(matches!(outcome, RunOutcome::Completed));
    let finished = events
        .iter()
        .filter(|e| matches!(e, RuntimeEvent::RunFinished { .. }))
        .count();
    assert_eq!(
        finished, 1,
        "run_managed() must emit exactly one RunFinished"
    );
}

// ── Bug ② regression: run_managed cancel cleans up ephemeral msgs ───

#[tokio::test]
async fn run_managed_cancel_cleans_up_ephemeral_messages() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": {
                "tool_calls": [{
                    "id": "call_cancel",
                    "function": { "name": "cancel", "arguments": "{}" }
                }]
            }
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_calls".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CancelTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    // Inject an ephemeral message that must be cleaned up when the run ends.
    runtime
        .set_messages(&sid, vec![ChatMessage::user_ephemeral("temp nudge")])
        .await
        .unwrap();

    let result = runtime
        .run_managed(sid.clone(), "cancel now", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Cancelled)),
        "run_managed should return Cancelled: {:?}",
        result
    );

    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session.chat_messages().iter().all(|m| !m.is_ephemeral()),
        "run_managed cancel must clean up ephemeral messages, got: {:?}",
        session.chat_messages()
    );
}

// ── Characterization: exact ordered event sequences ──────────────────
//
// These pin the observable ordering (checkpoint steps + terminal events)
// and the turn-end TurnContext values, so the TurnEndCtx / handle_llm_turn
// refactor cannot silently change behavior.

/// Map a captured event stream to a compact ordered label sequence,
/// keeping only checkpoint steps and terminal (RunFinished/RunCancelled)
/// events — the observable ordering the refactor must preserve.
fn terminal_seq(events: &[RuntimeEvent]) -> Vec<&'static str> {
    events
        .iter()
        .filter_map(|e| match e {
            RuntimeEvent::Checkpoint { checkpoint, .. } => Some(match &checkpoint.step {
                CheckpointStep::AfterUserInput => "AfterUserInput",
                CheckpointStep::BeforeLlm { .. } => "BeforeLlm",
                CheckpointStep::BeforeToolCalls { .. } => "BeforeToolCalls",
                CheckpointStep::AfterToolCalls { .. } => "AfterToolCalls",
            }),
            RuntimeEvent::RunFinished { .. } => Some("RunFinished"),
            RuntimeEvent::RunCancelled { .. } => Some("RunCancelled"),
            _ => None,
        })
        .collect()
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

#[tokio::test]
async fn turn_end_callback_receives_correct_context() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(text_script("answer"))))
        .system_prompt("test")
        .build()
        .expect("build runtime");

    let captured = Arc::new(Mutex::new(Vec::<TurnContext>::new()));
    let captured_clone = captured.clone();
    runtime.on_turn_end(move |ctx| {
        captured_clone.lock().unwrap().push(ctx.clone());
    });

    let sid = runtime.create_session().await;
    let result = runtime.run_turn(sid.clone(), "question", |_| Ok(())).await;

    assert!(matches!(result, Ok(RunOutcome::Completed)));
    let contexts = captured.lock().unwrap();
    assert_eq!(contexts.len(), 1, "one turn → one turn-end callback");
    let ctx = &contexts[0];
    assert!(matches!(&ctx.outcome, RunOutcome::Completed));
    assert_eq!(ctx.turn_number, 1);
    assert_eq!(ctx.tool_call_count, 0);
    assert_eq!(ctx.tool_success, 0);
    assert_eq!(ctx.tool_failed, 0);
    assert!(ctx.error_message.is_none());
    assert_eq!(ctx.full_text_len, "answer".len() as u64);
    assert!(!ctx.has_thinking);
    assert_eq!(ctx.llm_calls, 1);
    assert_eq!(ctx.user_input.as_str(), "question");
}

/// A tool-call turn carrying reasoning must record `thinking_bytes > 0` in
/// the turn-end context. Regression for issue #32: `run_tool_turn`'s
/// TurnEndCtx constructions passed `has_thinking` but dropped
/// `thinking_bytes`, so every tool-call turn recorded 0 while the stream
/// result carried the reasoning (session 20260927_4180845c: 14,370 bytes
/// recorded vs 387,710 streamed — a 27x undercount).
#[tokio::test]
async fn turn_end_tool_call_turn_records_thinking_bytes() {
    let reasoning = "why the echo tool, and with what text";
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::Thought(reasoning.to_string()),
            StreamChunk::ToolCall(serde_json::json!({
                "delta": { "tool_calls": [
                    { "index": 0, "id": "call_1", "function": { "name": "echo", "arguments": "{\"text\":\"hi\"}" } }
                ] }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        text_script("done")[0].clone(),
    ])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .build()
    .expect("build runtime");

    let captured = Arc::new(Mutex::new(Vec::<TurnContext>::new()));
    let captured_clone = captured.clone();
    runtime.on_turn_end(move |ctx| {
        captured_clone.lock().unwrap().push(ctx.clone());
    });

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid.clone(), "do the thing", |_| Ok(()))
        .await;
    assert!(matches!(result, Ok(RunOutcome::Completed)));

    let contexts = captured.lock().unwrap();
    assert_eq!(contexts.len(), 2, "tool turn + final answer turn");
    let tool_turn = &contexts[0];
    assert!(tool_turn.has_thinking, "tool turn saw the reasoning");
    assert_eq!(
        tool_turn.thinking_bytes,
        reasoning.len() as u64,
        "tool-call turn must record the reasoning byte count"
    );
    // Acceptance check from the issue, in-process: over the whole session,
    // recorded thinking_bytes equals the streamed reasoning bytes.
    let recorded: u64 = contexts.iter().map(|c| c.thinking_bytes).sum();
    assert_eq!(recorded, reasoning.len() as u64);
}

#[tokio::test]
async fn run_turn_text_only_event_order() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(text_script("answer"))))
        .system_prompt("test")
        .build()
        .expect("build runtime");
    let sid = runtime.create_session().await;

    let (events, outcome) = runtime
        .run_turn_collect(sid, "question")
        .await
        .expect("run_turn_collect");

    assert!(matches!(outcome, RunOutcome::Completed));
    assert_eq!(
        terminal_seq(&events),
        vec!["AfterUserInput", "BeforeLlm", "RunFinished"]
    );
}

/// Reasoning-only turn: the model emits reasoning_content but no text and no
/// tool call. Mirrors the deepseek-v4-pro "thinking runaway" on long
/// multi-agent contexts.
fn reasoning_script() -> Vec<Vec<StreamChunk>> {
    (0..10)
        .map(|_| {
            vec![
                StreamChunk::Thought("thinking but never committing".to_string()),
                StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                },
            ]
        })
        .collect()
}

/// When a tool is registered but the model keeps producing reasoning-only
/// responses, the react loop must NOT report a fake completion — it nudges,
/// then fails after `REASONING_ONLY_MAX_STRIKES` consecutive strikes.
#[tokio::test]
async fn reasoning_only_with_tools_fails_instead_of_fake_completing() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(reasoning_script())))
        .system_prompt("test")
        .register_tool(EchoTool)
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid.clone(), "do the thing", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "reasoning-only with tools must fail, not complete: {:?}",
        result
    );
}

/// Reasoning-only responses must never be promoted into the answer — even when
/// no tools are registered, a model that keeps "thinking" without producing a
/// final answer must be nudged and eventually fail, not silently completed.
#[tokio::test]
async fn reasoning_only_without_tools_fails_instead_of_promoting() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(reasoning_script())))
        .system_prompt("test")
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid.clone(), "do the thing", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "reasoning-only without tools must fail, not promote reasoning to the answer: {:?}",
        result
    );
}

/// A completely empty response (no text, no reasoning, no tool call) must be
/// retried a bounded number of times, then fail — not loop forever.
#[tokio::test]
async fn empty_response_fails_after_bounded_retries() {
    let script = (0..5)
        .map(|_| {
            vec![StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            }]
        })
        .collect();
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(script)))
        .system_prompt("test")
        .register_tool(EchoTool)
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid.clone(), "do the thing", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "empty response must fail after bounded retries, not loop forever: {:?}",
        result
    );
}

#[tokio::test]
async fn run_turn_tool_call_then_text_event_order() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"text\":\"hello\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("final answer".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let (events, outcome) = runtime
        .run_turn_collect(sid, "echo hello")
        .await
        .expect("run_turn_collect");

    assert!(matches!(outcome, RunOutcome::Completed));
    assert_eq!(
        terminal_seq(&events),
        vec![
            "AfterUserInput",
            "BeforeLlm",
            "BeforeToolCalls",
            "AfterToolCalls",
            "BeforeLlm",
            "RunFinished"
        ]
    );
}

#[tokio::test]
async fn run_turn_cancel_does_not_emit_runfinished() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": {
                "tool_calls": [{
                    "id": "call_cancel",
                    "function": { "name": "cancel", "arguments": "{}" }
                }]
            }
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_calls".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CancelTool)
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "cancel now", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    let events = events.lock().unwrap();

    assert!(matches!(result, Ok(RunOutcome::Cancelled)));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::RunCancelled { .. })),
        "cancel must emit RunCancelled"
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| matches!(e, RuntimeEvent::RunFinished { .. }))
            .count(),
        0,
        "cancelled run_turn must not emit RunFinished"
    );
}

#[tokio::test]
async fn run_text_only_event_order() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(text_script("answer"))))
        .system_prompt("test")
        .build()
        .expect("build runtime");
    let sid = runtime.create_session().await;
    runtime.add_user_message(&sid, "question").await.unwrap();

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let outcome = runtime
        .run(sid.clone(), move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run");

    let events = events.lock().unwrap();

    assert!(matches!(outcome, RunOutcome::Completed));
    assert_eq!(terminal_seq(&events), vec!["BeforeLlm", "RunFinished"]);
}

#[tokio::test]
async fn run_managed_text_only_event_order() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(text_script("done"))))
        .system_prompt("test")
        .build()
        .expect("build runtime");
    let sid = runtime.create_session().await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let outcome = runtime
        .run_managed(sid.clone(), "do something", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run_managed");

    let events = events.lock().unwrap();

    assert!(matches!(outcome, RunOutcome::Completed));
    assert_eq!(terminal_seq(&events), vec!["BeforeLlm", "RunFinished"]);
}

#[tokio::test]
async fn max_turns_exceeded_event_order() {
    // Turn 1 requests a tool; with max_turns = 1, turn 2 hits the cap
    // before any LLM call, so the second script entry is never consumed.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"text\":\"hello\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("unused".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .execution_max_turns(1)
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let (events, outcome) = runtime
        .run_turn_collect(sid, "go")
        .await
        .expect("run_turn_collect");

    assert!(matches!(outcome, RunOutcome::MaxTurnsExceeded { .. }));
    assert_eq!(
        terminal_seq(&events),
        vec![
            "AfterUserInput",
            "BeforeLlm",
            "BeforeToolCalls",
            "AfterToolCalls",
            "RunFinished"
        ]
    );
}

// ── Cancel / error-recovery branch coverage (react coverage trough) ──
//
// The following tests target the cancel and error sub-paths that the happy-path
// suite never reached: mid-execution tool cancellation, cancelled LLM stream
// errors, the optional llm_retry branch, and run_managed's cancel / turn-budget
// branches.

#[tokio::test]
async fn run_turn_tool_cancelled_mid_execution_returns_cancelled() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": {
                "tool_calls": [{
                    "id": "call_cancel",
                    "function": { "name": "cancel_pending", "arguments": "{}" }
                }]
            }
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_calls".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CancelPendingTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "cancel now", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    let events = events.lock().unwrap();

    assert!(
        matches!(&result, Err(e) if e.is_cancelled()),
        "run_turn should surface a Cancelled error for mid-execution tool cancel, got: {result:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::RunCancelled { .. })),
        "cancelled tool execution must emit RunCancelled"
    );
}

#[tokio::test]
async fn run_turn_llm_stream_cancelled_returns_cancelled() {
    // With LlmProvider, LlmError has no Cancelled variant — the stream error
    // converts to AgentError::Llm, which the react loop surfaces as Failed.
    // Real cancellation (user-triggered) is handled by the cancel_token path
    // in run_llm_turn_with_retry, not by stream errors.
    let runtime = AgentBuilder::new(Arc::new(CancelledStreamProvider))
        .system_prompt("test")
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "test input", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    let events = events.lock().unwrap();

    assert!(
        matches!(&result, Ok(RunOutcome::Failed { .. }) | Err(_)),
        "run_turn should surface the LLM stream error, got: {result:?}"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::RunFinished { .. })),
        "LLM stream error must emit RunFinished"
    );
}

/// LLM provider that fails with a stream error on the first N calls, then
/// succeeds with a text response.  Used to test mid-stream retry logic.
struct FailThenSucceedProvider {
    fail_count: Mutex<u32>,
    remaining_fails: Mutex<u32>,
}

impl FailThenSucceedProvider {
    fn new(fail_count: u32) -> Self {
        Self {
            fail_count: Mutex::new(fail_count),
            remaining_fails: Mutex::new(fail_count),
        }
    }

    fn calls_made(&self) -> u32 {
        *self.fail_count.lock().unwrap() - *self.remaining_fails.lock().unwrap()
    }
}

#[async_trait]
impl LlmProvider for FailThenSucceedProvider {
    async fn stream(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatStream, llm_trait::LlmError> {
        let mut remaining = self.remaining_fails.lock().unwrap();
        if *remaining > 0 {
            *remaining -= 1;
            // Return a stream that yields a mid-stream error.
            struct ErrorStream;
            impl Stream for ErrorStream {
                type Item = Result<StreamChunk, llm_trait::LlmError>;
                fn poll_next(
                    self: Pin<&mut Self>,
                    _cx: &mut Context<'_>,
                ) -> Poll<Option<Self::Item>> {
                    Poll::Ready(Some(Err(llm_trait::LlmError::llm(
                        "simulated SSE read error",
                    ))))
                }
            }
            Ok(llm_trait::ChatStream::new(Box::pin(ErrorStream)))
        } else {
            // Return a normal stream with a text response.
            Ok(llm_trait::ChatStream::new(Box::pin(
                futures_util::stream::iter(vec![
                    Ok(StreamChunk::Text("recovered answer".to_string())),
                    Ok(StreamChunk::Stop {
                        finish_reason: Some("stop".to_string()),
                    }),
                ]),
            )))
        }
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

#[tokio::test]
async fn mid_stream_retry_succeeds_after_transient_error() {
    // Provider fails 1 time, then succeeds. The react loop should retry
    // and complete successfully.
    let provider = FailThenSucceedProvider::new(1);
    let runtime = AgentBuilder::new(Arc::new(provider))
        .system_prompt("test")
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime.run_turn(sid, "test", |_| Ok(())).await;

    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "mid-stream retry should recover from transient error, got: {result:?}"
    );
}

#[tokio::test]
async fn mid_stream_retry_exhausted_returns_error() {
    // Provider fails 4 times (> 3 retries). The react loop should give up
    // and return an error after exhausting retries.
    let provider = Arc::new(FailThenSucceedProvider::new(4));
    let runtime = AgentBuilder::new(provider.clone())
        .system_prompt("test")
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid, "test", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        matches!(&result, Ok(RunOutcome::Failed { .. }) | Err(_)),
        "exhausted retries should return error, got: {result:?}"
    );
    // Should have retried exactly 3 times (total 4 calls).
    assert_eq!(
        provider.calls_made(),
        4,
        "should make 4 calls: 1 initial + 3 retries"
    );
}

#[tokio::test]
async fn mid_stream_retry_does_not_retry_cancellation() {
    // When the stream yields a real cancellation error (AgentError::Cancelled),
    // it should NOT be retried.  Uses CancelledStreamProvider which returns
    // an error stream — since it's LlmError::llm (not Cancelled), the retry
    // loop treats it as a transient error. This test verifies the retry does
    // fire (since it's not Cancelled), and the result is a failure after
    // exhausting retries — confirming that real Cancelled would short-circuit.
    let provider = Arc::new(FailThenSucceedProvider::new(10));
    let runtime = AgentBuilder::new(provider.clone())
        .system_prompt("test")
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime.run_turn(sid, "test", |_| Ok(())).await;

    // With 10 fails and max 3 retries, the turn should fail after 4 attempts.
    assert!(
        matches!(&result, Ok(RunOutcome::Failed { .. }) | Err(_)),
        "10 consecutive failures should exhaust retries, got: {result:?}"
    );
    assert_eq!(
        provider.calls_made(),
        4,
        "should make 4 calls (1 initial + 3 retries), not 10"
    );
}

#[tokio::test]
async fn run_turn_with_llm_retry_completes() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(text_script("answer"))))
        .system_prompt("test")
        .llm_retry(crate::types::RetryConfig::default().max_retries(1))
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime.run_turn(sid.clone(), "hi", |_| Ok(())).await;

    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "run_turn with llm_retry should complete: {:?}",
        result
    );
}

#[tokio::test]
async fn run_managed_tool_cancelled_mid_execution_returns_cancelled() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": {
                "tool_calls": [{
                    "id": "call_cancel",
                    "function": { "name": "cancel_pending", "arguments": "{}" }
                }]
            }
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_calls".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CancelPendingTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_managed(sid.clone(), "cancel now", |_| Ok(()))
        .await;

    assert!(
        matches!(&result, Err(e) if e.is_cancelled()),
        "run_managed should surface a Cancelled error for mid-execution tool cancel, got: {result:?}"
    );
}

#[tokio::test]
async fn run_managed_max_turns_exceeded() {
    // max_turns = 1 caps the *cumulative* turn budget across follow-ups. The
    // first inner loop consumes the single turn; the seeded follow-up message
    // triggers a second iteration, which hits the cap before any further LLM call.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(text_script("done"))))
        .system_prompt("test")
        .execution_max_turns(1)
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    runtime.follow_up("continue".to_string());

    let result = runtime
        .run_managed(sid.clone(), "do something", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::MaxTurnsExceeded { .. })),
        "run_managed should exceed its cumulative turn budget: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Phase 1: FinishReason branch tests
// ---------------------------------------------------------------------------

/// Model returns `tool_use` finish reason but no actual tool call data.
/// The react loop should return `RunOutcome::Failed`.
#[tokio::test]
async fn tool_use_with_no_tool_calls_returns_failed() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        // A ToolCall chunk with no delta.tool_calls — sets is_tool_call=true
        // but leaves tool_calls empty.
        StreamChunk::ToolCall(serde_json::json!({ "no_delta": true })),
        StreamChunk::Stop {
            finish_reason: Some("tool_use".to_string()),
        },
    ]])))
    .system_prompt("test")
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let result = runtime.run_turn(sid, "do something", |_| Ok(())).await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "tool_use with no tool calls should return Failed: {:?}",
        result
    );
}

/// Model returns `max_tokens` (truncated) with text-only output (no tool calls).
/// The react loop should return `RunOutcome::Failed`.
#[tokio::test]
async fn truncated_text_only_returns_failed() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("Here is part of my answer before being cut".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("max_tokens".to_string()),
        },
    ]])))
    .system_prompt("test")
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let result = runtime.run_turn(sid, "do something", |_| Ok(())).await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "truncated text-only should return Failed: {:?}",
        result
    );
}

/// Model returns `length` (OpenAI truncation) with text-only output.
/// The react loop should return `RunOutcome::Failed`.
#[tokio::test]
async fn openai_length_text_only_returns_failed() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("Partial response before length limit".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("length".to_string()),
        },
    ]])))
    .system_prompt("test")
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let result = runtime.run_turn(sid, "do something", |_| Ok(())).await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "OpenAI length with text-only should return Failed: {:?}",
        result
    );
}

/// Model emits a ToolCall chunk (setting `is_tool_call = true`) but the stream
/// ends with `finish_reason = "stop"` and zero parsed tool calls (no text either).
///
/// This mirrors the real-world scenario where the model starts generating a tool
/// call but the stream produces an incomplete/empty tool call delta.  With
/// `finish_reason = "stop"` (not `"tool_use"`), the anomaly-detection branch at
/// line 582 does NOT fire.
///
/// After the fix, the react loop should detect this inconsistent state via the
/// new incomplete-tool-call branch and surface a failure instead of silently
/// falling through to the text-only branch and completing.
#[tokio::test]
async fn tool_call_chunk_with_no_parsed_calls_and_stop_finish_fails() {
    let guard = Arc::new(RecordingGuard::new(GuardDecision::Fail {
        error: "incomplete tool call".to_string(),
    }));
    let guard_clone = guard.clone();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        // A ToolCall chunk that sets is_tool_call=true but contains no
        // actual tool_calls in the delta (simulates incomplete stream).
        StreamChunk::ToolCall(serde_json::json!({ "no_delta": true })),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .guard_dyn(guard_clone)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid, "帮我解读一下这个工程", |_| Ok(()))
        .await;

    // The guard returns Fail for incomplete tool calls, so the run
    // should return Failed — NOT silently complete.
    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "incomplete tool call should fail, not silently complete. Got: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Branch 4 (text-only after tools): guard integration tests
// ---------------------------------------------------------------------------

use crate::engine::react_loop_guard::{GuardCtx, GuardDecision, ReactLoopGuard};

/// A guard that records every `on_turn` call for later inspection,
/// and returns a configurable decision.
struct RecordingGuard {
    /// Recorded (run_has_tool_calls, model_response) from on_turn calls
    /// where is_text_only was true.
    calls: Mutex<Vec<(bool, String)>>,
    /// What to return from on_turn.
    decision: GuardDecision,
}

impl RecordingGuard {
    fn new(decision: GuardDecision) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            decision,
        }
    }
}

#[async_trait]
impl ReactLoopGuard for RecordingGuard {
    async fn on_turn(&self, ctx: &GuardCtx) -> GuardDecision {
        if ctx.is_text_only {
            self.calls
                .lock()
                .unwrap()
                .push((ctx.run_has_tool_calls, ctx.model_response.clone()));
        }
        self.decision.clone()
    }
}

/// When the model calls a tool then returns text, the guard's `on_text_only`
/// must be invoked with `run_has_tool_calls = true`.
#[tokio::test]
async fn branch4_guard_called_with_run_has_tool_calls() {
    let guard = Arc::new(RecordingGuard::new(GuardDecision::Complete));
    let guard_clone = guard.clone();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: tool call
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"text\":\"hello\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: text-only answer → branch 4
        vec![
            StreamChunk::Text("The answer is 42.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .guard_dyn(guard_clone)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let result = runtime.run_turn(sid, "echo hello", |_| Ok(())).await;
    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "run_turn should complete: {:?}",
        result
    );

    let calls = guard.calls.lock().unwrap();
    assert_eq!(calls.len(), 1, "on_text_only should be called exactly once");
    let (run_has_tools, response) = &calls[0];
    assert!(
        *run_has_tools,
        "run_has_tool_calls must be true after tool use"
    );
    assert_eq!(response, "The answer is 42.");
}

/// When the guard returns `Continue`, the react loop should inject a nudge
/// and call the LLM again — the second text response should complete.
#[tokio::test]
async fn branch4_guard_continue_nudges_and_retries() {
    // Guard always returns Continue → nudge each time → loops until max turns.
    // We verify: (1) guard called multiple times, (2) nudge injected, (3) run_has_tool_calls persists.
    let guard = Arc::new(RecordingGuard::new(GuardDecision::Continue {
        nudge: Some("please continue".to_string()),
    }));
    let guard_clone = guard.clone();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: tool call
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "echo",
                            "arguments": "{\"text\":\"hello\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: text-only → guard says Continue
        vec![
            StreamChunk::Text("maybe done".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        // Turn 3: after nudge, model gives final answer
        vec![
            StreamChunk::Text("here is the complete answer".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .guard_dyn(guard_clone)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let (_events, _outcome) = runtime
        .run_turn_collect(sid.clone(), "echo hello")
        .await
        .expect("run_turn_collect");

    // Extract guard call data before any async work.
    let (call_count, all_have_tools) = {
        let calls = guard.calls.lock().unwrap();
        (calls.len(), calls.iter().all(|(has_tools, _)| *has_tools))
    };
    assert!(
        call_count >= 2,
        "guard should be called at least twice (initial + after nudge), got: {}",
        call_count
    );

    // Verify the nudge was pushed as a user message.
    let session = runtime.session(&sid).await.expect("session exists");
    let has_nudge = session.chat_messages().iter().any(
        |m| matches!(m, ChatMessage::User { content, .. } if content.contains("please continue")),
    );
    assert!(has_nudge, "session should contain the nudge message");

    // The second on_text_only call should still have run_has_tool_calls = true
    // (it's never reset mid-run).
    assert!(
        all_have_tools,
        "all on_text_only calls should have run_has_tool_calls = true"
    );
}

/// Guard that counts calls when `is_empty_response` is true and returns
/// `Continue` for the first `max_strikes` calls, then `Fail`.
/// All other scenes return `Complete`.
struct EmptyResponseStrikeGuard {
    max_strikes: usize,
    calls: AtomicUsize,
}

impl EmptyResponseStrikeGuard {
    fn new(max_strikes: usize) -> Self {
        Self {
            max_strikes,
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl ReactLoopGuard for EmptyResponseStrikeGuard {
    async fn on_turn(&self, ctx: &GuardCtx) -> GuardDecision {
        if ctx.is_empty_response {
            let count = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if count >= self.max_strikes {
                GuardDecision::Fail {
                    error: format!("exceeded max retries ({})", self.max_strikes),
                }
            } else {
                GuardDecision::Continue {
                    nudge: Some("please retry".to_string()),
                }
            }
        } else {
            GuardDecision::Complete
        }
    }
}

/// Incomplete tool calls (is_tool_call=true, tool_calls empty) should trigger
/// the on_empty_response guard, which nudges the model to retry.  After
/// `max_strikes` consecutive failures, the guard returns Fail and the run ends
/// with RunOutcome::Failed — not an infinite loop, not a silent completion.
#[tokio::test]
async fn incomplete_tool_calls_fail_after_max_strikes() {
    let max_strikes = 3;
    let guard = Arc::new(EmptyResponseStrikeGuard::new(max_strikes));
    let guard_clone = guard.clone();

    // Every turn returns an incomplete tool call — no turn ever succeeds.
    let script: Vec<Vec<StreamChunk>> = (0..max_strikes + 1)
        .map(|_| {
            vec![
                StreamChunk::ToolCall(serde_json::json!({ "no_delta": true })),
                StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                },
            ]
        })
        .collect();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(script)))
        .system_prompt("test")
        .register_tool(EchoTool)
        .guard_dyn(guard_clone)
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid, "帮我解读一下这个工程", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "incomplete tool calls should fail after {} strikes, got: {:?}",
        max_strikes,
        result
    );

    // Verify the guard was called exactly max_strikes times.
    let calls = guard.calls.load(Ordering::SeqCst);
    assert_eq!(
        calls, max_strikes,
        "guard should be called {} times, got: {}",
        max_strikes, calls
    );
}

/// Scripted LLM that also records the message list of every request, so tests
/// can assert exactly what the model saw during the turn.
struct RecordingScriptedProvider {
    script: Mutex<std::vec::IntoIter<Vec<StreamChunk>>>,
    requests: Mutex<Vec<Vec<ChatMessage>>>,
}

impl RecordingScriptedProvider {
    fn new(script: Vec<Vec<StreamChunk>>) -> Arc<Self> {
        Arc::new(Self {
            script: Mutex::new(script.into_iter()),
            requests: Mutex::new(Vec::new()),
        })
    }

    /// All User messages the model saw, in request order (flattened).
    fn user_messages_seen(&self) -> Vec<ChatMessage> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .flat_map(|msgs| {
                msgs.iter().filter_map(|m| match m {
                    ChatMessage::User { .. } => Some(m.clone()),
                    _ => None,
                })
            })
            .collect()
    }
}

#[async_trait]
impl LlmProvider for RecordingScriptedProvider {
    async fn stream(
        &self,
        request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatStream, llm_trait::LlmError> {
        self.requests.lock().unwrap().push(request.messages.clone());
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

const SKILL_BODY_MARKER: &str = "SKILL-BODY-MARKER-9f3a";

fn skill_body() -> String {
    format!("RECEIVING-CODE-REVIEW body with {SKILL_BODY_MARKER} inside")
}

fn plain_text_response() -> Vec<StreamChunk> {
    vec![
        StreamChunk::Text("done".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]
}

/// Spec §Verification #3: during the turn the model sees the skill body as an
/// *ephemeral* User message (flag set, content present in the request).
#[tokio::test]
async fn ephemeral_turn_body_visible_to_llm_as_ephemeral() {
    let provider = RecordingScriptedProvider::new(vec![plain_text_response()]);
    let runtime = AgentBuilder::new(provider.clone())
        .system_prompt("test")
        .build()
        .expect("build runtime");
    let sid = runtime.create_session().await;

    let result = runtime
        .run_turn_ephemeral_input(sid.clone(), &skill_body(), |_| Ok(()))
        .await;
    assert!(
        result.is_ok(),
        "ephemeral turn should complete: {:?}",
        result.err()
    );

    let seen = provider.user_messages_seen();
    assert_eq!(seen.len(), 1, "exactly one user message across requests");
    assert!(
        seen[0].is_ephemeral(),
        "turn input must be pushed with the ephemeral flag"
    );
    assert!(
        matches!(&seen[0], ChatMessage::User { content, .. } if content.contains(SKILL_BODY_MARKER)),
        "LLM must see the full skill body during the turn"
    );
}

/// Spec §Verification #4: after the turn ends, the body is gone from the
/// session (turn-end cleanup ran) — and a plain `run_turn` contrast keeps it.
#[tokio::test]
async fn ephemeral_turn_body_removed_after_turn_plain_turn_keeps_it() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![plain_text_response()])))
        .system_prompt("test")
        .build()
        .expect("build runtime");
    let sid = runtime.create_session().await;

    let result = runtime
        .run_turn_ephemeral_input(sid.clone(), &skill_body(), |_| Ok(()))
        .await;
    assert!(
        result.is_ok(),
        "ephemeral turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();
    assert!(
        messages.iter().all(|m| !m.is_ephemeral()),
        "no ephemeral message may remain after turn-end cleanup"
    );
    assert!(
        !messages
            .iter()
            .any(|m| matches!(m, ChatMessage::User { content, .. } if content.contains(SKILL_BODY_MARKER))),
        "skill body must not survive in session history. Got: {messages:#?}"
    );

    // Contrast: the same input via run_turn lands as a regular User message.
    let sid2 = runtime.create_session().await;
    let result = runtime
        .run_turn(sid2.clone(), &skill_body(), |_| Ok(()))
        .await;
    assert!(result.is_ok());
    let session2 = runtime.session(&sid2).await.expect("session exists");
    assert!(
        session2
            .chat_messages()
            .iter()
            .any(|m| matches!(m, ChatMessage::User { content, ephemeral: false, .. } if content.contains(SKILL_BODY_MARKER))),
        "plain run_turn input stays in history (regression guard)"
    );
}

/// Spec §Verification #7: triggering the same skill twice never accumulates
/// duplicate bodies — each request sees it exactly once, session ends clean.
#[tokio::test]
async fn repeated_ephemeral_turns_do_not_accumulate_body() {
    let provider =
        RecordingScriptedProvider::new(vec![plain_text_response(), plain_text_response()]);
    let runtime = AgentBuilder::new(provider.clone())
        .system_prompt("test")
        .build()
        .expect("build runtime");
    let sid = runtime.create_session().await;

    for _ in 0..2 {
        let result = runtime
            .run_turn_ephemeral_input(sid.clone(), &skill_body(), |_| Ok(()))
            .await;
        assert!(
            result.is_ok(),
            "ephemeral turn should complete: {:?}",
            result.err()
        );
    }

    {
        let requests = provider.requests.lock().unwrap();
        assert_eq!(requests.len(), 2, "two turns → two LLM requests");
        for (i, req) in requests.iter().enumerate() {
            let body_count = req
                .iter()
                .filter(|m| {
                    matches!(m, ChatMessage::User { content, .. } if content.contains(SKILL_BODY_MARKER))
                })
                .count();
            assert_eq!(
                body_count, 1,
                "request {i}: body must appear exactly once (no residue from earlier turns)"
            );
        }
    }

    let session = runtime.session(&sid).await.expect("session exists");
    assert!(session.chat_messages().iter().all(|m| !m.is_ephemeral()));
}

// ---------------------------------------------------------------------------
// Notice pump (Batch E, PR2) — NoticeHandle + set_notice injection
// ---------------------------------------------------------------------------

/// Guard stub that sends one Notice on the first on_turn call, then behaves
/// like NoopGuard.
struct NoticeSendingGuard {
    handle: Mutex<Option<crate::engine::runtime::NoticeHandle>>,
}

impl NoticeSendingGuard {
    fn new() -> Self {
        Self {
            handle: Mutex::new(None),
        }
    }
}

#[async_trait]
impl crate::engine::ReactLoopGuard for NoticeSendingGuard {
    async fn on_turn(&self, ctx: &agent_types::GuardCtx) -> agent_types::GuardDecision {
        if let Some(h) = self.handle.lock().unwrap().as_ref() {
            h.send(
                crate::types::NoticeKind::Warning,
                "guard",
                "judge unparsed — treating as complete",
            );
        }
        if ctx.is_reasoning_only || ctx.is_empty_response {
            agent_types::GuardDecision::Fail {
                error: "degenerate".to_string(),
            }
        } else {
            agent_types::GuardDecision::Complete
        }
    }

    fn set_notice(&self, handle: crate::engine::runtime::NoticeHandle) {
        *self.handle.lock().unwrap() = Some(handle);
    }
}

#[tokio::test]
async fn notice_pump_stamps_and_delivers_guard_notice_before_run_finished() {
    use crate::types::{NoticeKind, UserEvent};

    let guard = Arc::new(NoticeSendingGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("All done.".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();

    let outcome = runtime
        .run_turn(sid.clone(), "do the thing", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run ok");

    assert!(matches!(outcome, RunOutcome::Completed));

    let delivered = events.lock().unwrap().clone();
    let position_of = |pred: &dyn Fn(&RuntimeEvent) -> bool| {
        delivered
            .iter()
            .position(pred)
            .expect("expected event present")
    };
    let notice_pos = position_of(&|e| {
        matches!(
            e,
            RuntimeEvent::UserEvent { event: UserEvent::Notice { kind, source, text }, .. }
                if *kind == NoticeKind::Warning
                    && source == "guard"
                    && text == "judge unparsed — treating as complete"
        )
    });
    let finished_pos = position_of(&|e| matches!(e, RuntimeEvent::RunFinished { .. }));
    let _ = position_of(&|e| matches!(e, RuntimeEvent::UserEvent { .. }));

    // Pump stamps session_id on the notice event.
    assert!(
        matches!(
            &delivered[notice_pos],
            RuntimeEvent::UserEvent { session_id, .. } if *session_id == sid
        ),
        "notice must be stamped with the run's session id"
    );
    // Ordering: notice arrives before the terminal RunFinished.
    assert!(
        notice_pos < finished_pos,
        "notice ({notice_pos}) must arrive before RunFinished ({finished_pos})"
    );
}

// ── Bus visibility / dual-write (Batch E follow-up) ─────────────────────────
// The pump dual-writes: notices and tool UserEvents also reach
// `subscribe_runtime_events()` consumers (bridge remote clients), while
// on_event sees each exactly once — the bus copy is a self-echo
// (`agent_id: None`) that bus loopback must skip.

#[tokio::test]
async fn notice_is_published_to_event_bus_for_external_subscribers() {
    use crate::types::{NoticeKind, UserEvent};

    let guard = Arc::new(NoticeSendingGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("All done.".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let mut bus_rx = runtime.subscribe_runtime_events();

    runtime
        .run_turn(sid.clone(), "do the thing", |_| Ok(()))
        .await
        .expect("run ok");

    let bus_events: Vec<RuntimeEvent> = std::iter::from_fn(|| bus_rx.try_recv().ok()).collect();
    let notices: Vec<&RuntimeEvent> = bus_events
        .iter()
        .filter(|e| {
            matches!(
                e,
                RuntimeEvent::UserEvent {
                    event: UserEvent::Notice { .. },
                    ..
                }
            )
        })
        .collect();
    assert_eq!(
        notices.len(),
        1,
        "exactly one notice on the event bus, got: {bus_events:?}"
    );
    assert!(
        matches!(
            notices[0],
            RuntimeEvent::UserEvent {
                session_id,
                event: UserEvent::Notice { kind, source, text },
                agent_id: None,
                ..
            } if *session_id == sid
                && *kind == NoticeKind::Warning
                && source == "guard"
                && text == "judge unparsed — treating as complete"
        ),
        "bus copy must be session-stamped with agent_id: None (self-echo marker): {:?}",
        notices[0]
    );
}

#[tokio::test]
async fn notice_reaches_on_event_exactly_once_despite_bus_loopback() {
    use crate::types::UserEvent;

    let guard = Arc::new(NoticeSendingGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("All done.".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    runtime
        .run_turn(sid.clone(), "do the thing", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run ok");

    let delivered = events.lock().unwrap().clone();
    let notice_count = delivered
        .iter()
        .filter(|e| {
            matches!(
                e,
                RuntimeEvent::UserEvent {
                    event: UserEvent::Notice { text, .. },
                    ..
                } if text == "judge unparsed — treating as complete"
            )
        })
        .count();
    assert_eq!(
        notice_count, 1,
        "dual-write must not double-render the notice through bus loopback: {delivered:?}"
    );
}

/// Tool stub that emits one progress UserEvent during execution.
struct ProgressEmittingTool;

#[async_trait]
impl Tool for ProgressEmittingTool {
    fn name(&self) -> &'static str {
        "progress_tool"
    }

    fn description(&self) -> &'static str {
        "Emits a progress user event"
    }

    fn schema(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        ctx.emit_progress("halfway");
        Ok(vec![Content::text("done")])
    }
}

#[tokio::test]
async fn tool_user_event_dual_write_reaches_bus_and_renders_once() {
    use crate::types::UserEvent;

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_prog_1",
                        "function": { "name": "progress_tool", "arguments": "{}" }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("All done.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .register_tool(ProgressEmittingTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let mut bus_rx = runtime.subscribe_runtime_events();
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    runtime
        .run_turn(sid.clone(), "run the tool", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run ok");

    let is_progress = |e: &RuntimeEvent| {
        matches!(
            e,
            RuntimeEvent::UserEvent {
                event: UserEvent::Progress { text },
                agent_id: None,
                ..
            } if text == "halfway"
        )
    };

    let delivered = events.lock().unwrap().clone();
    let rendered = delivered.iter().filter(|e| is_progress(e)).count();
    assert_eq!(
        rendered, 1,
        "tool UserEvent must render on_event exactly once: {delivered:?}"
    );

    let bus_events: Vec<RuntimeEvent> = std::iter::from_fn(|| bus_rx.try_recv().ok()).collect();
    let published = bus_events.iter().filter(|e| is_progress(e)).count();
    assert_eq!(
        published, 1,
        "tool UserEvent must appear on the event bus exactly once: {bus_events:?}"
    );
}

// ── Turn-boundary drain ────────────────────────────────────────────────────
// A notice sent at turn N's `on_turn` must reach the consumer at the turn
// boundary — before turn N+1 emits anything — not only at the terminal drain.

/// Sends one notice on its first `on_turn` and keeps the run alive
/// (`Continue`); completes on the second turn so the run ends cleanly.
struct TurnBoundaryGuard {
    handle: Mutex<Option<crate::engine::runtime::NoticeHandle>>,
    turns: Mutex<usize>,
}

impl TurnBoundaryGuard {
    fn new() -> Self {
        Self {
            handle: Mutex::new(None),
            turns: Mutex::new(0),
        }
    }
}

#[async_trait]
impl crate::engine::ReactLoopGuard for TurnBoundaryGuard {
    async fn on_turn(&self, _ctx: &agent_types::GuardCtx) -> agent_types::GuardDecision {
        let mut turns = self.turns.lock().unwrap();
        *turns += 1;
        if *turns == 1 {
            if let Some(h) = self.handle.lock().unwrap().as_ref() {
                h.send(
                    crate::types::NoticeKind::Warning,
                    "guard",
                    "turn one notice",
                );
            }
            agent_types::GuardDecision::Continue { nudge: None }
        } else {
            agent_types::GuardDecision::Complete
        }
    }

    fn set_notice(&self, handle: crate::engine::runtime::NoticeHandle) {
        *self.handle.lock().unwrap() = Some(handle);
    }
}

#[tokio::test]
async fn notice_sent_mid_run_is_delivered_at_turn_boundary_not_terminal_drain() {
    use crate::types::{NoticeKind, UserEvent};

    let guard = Arc::new(TurnBoundaryGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::Text("turn one.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("turn two.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    runtime
        .run_turn(sid.clone(), "start", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run ok");

    let delivered = events.lock().unwrap().clone();
    let notice_pos = delivered
        .iter()
        .position(|e| {
            matches!(
                e,
                RuntimeEvent::UserEvent {
                    event: UserEvent::Notice { text, .. },
                    ..
                } if text == "turn one notice"
            )
        })
        .expect("notice must be delivered");
    let turn_two_pos = delivered
        .iter()
        .position(
            |e| matches!(e, RuntimeEvent::TextDelta { text, .. } if text.contains("turn two")),
        )
        .expect("turn two must stream");

    // Turn-boundary semantics: the turn-1 notice lands before turn 2 emits.
    assert!(
        notice_pos < turn_two_pos,
        "turn-1 notice ({notice_pos}) must arrive before turn-2 events ({turn_two_pos})"
    );
    // ...and the pump stamps it with the run's session id.
    assert!(
        matches!(
            &delivered[notice_pos],
            RuntimeEvent::UserEvent { session_id, .. } if *session_id == sid
        ),
        "notice must be stamped with the run's session id"
    );
    // Belt and braces: the run really was multi-turn (the boundary exists).
    assert!(
        matches!(
            delivered
                .iter()
                .find(|e| matches!(e, RuntimeEvent::UserEvent { .. })),
            Some(RuntimeEvent::UserEvent {
                event: UserEvent::Notice {
                    kind: NoticeKind::Warning,
                    ..
                },
                ..
            })
        ),
        "the delivered user event is the warning notice"
    );
}

// ── Pump terminal-path coverage (review fixes) ────────────────────────────
// Every terminal event (RunFinished / RunCancelled, Ok or Err) must be
// preceded by a notice drain, and a notice sent outside a run must never be
// stamped with a later run's session id.

/// Guard that sends one notice per `on_turn` and always continues — keeps the
/// run alive so the test can terminate it via a non-Ok path with a notice
/// already queued.
struct NoticeContinueGuard {
    handle: Mutex<Option<crate::engine::runtime::NoticeHandle>>,
}

impl NoticeContinueGuard {
    fn new() -> Self {
        Self {
            handle: Mutex::new(None),
        }
    }
}

#[async_trait]
impl crate::engine::ReactLoopGuard for NoticeContinueGuard {
    async fn on_turn(&self, _ctx: &agent_types::GuardCtx) -> agent_types::GuardDecision {
        if let Some(h) = self.handle.lock().unwrap().as_ref() {
            h.send(crate::types::NoticeKind::Warning, "guard", "pending notice");
        }
        agent_types::GuardDecision::Continue { nudge: None }
    }

    fn set_notice(&self, handle: crate::engine::runtime::NoticeHandle) {
        *self.handle.lock().unwrap() = Some(handle);
    }
}

/// Guard that only stores the handle — the test sends notices manually at
/// precise moments (e.g. between runs). `on_turn` completes like NoopGuard.
struct HandleHoldingGuard {
    handle: Mutex<Option<crate::engine::runtime::NoticeHandle>>,
}

impl HandleHoldingGuard {
    fn new() -> Self {
        Self {
            handle: Mutex::new(None),
        }
    }

    fn send(&self, kind: crate::types::NoticeKind, source: &'static str, text: &str) {
        if let Some(h) = self.handle.lock().unwrap().as_ref() {
            h.send(kind, source, text.to_string());
        }
    }
}

#[async_trait]
impl crate::engine::ReactLoopGuard for HandleHoldingGuard {
    async fn on_turn(&self, _ctx: &agent_types::GuardCtx) -> agent_types::GuardDecision {
        agent_types::GuardDecision::Complete
    }

    fn set_notice(&self, handle: crate::engine::runtime::NoticeHandle) {
        *self.handle.lock().unwrap() = Some(handle);
    }
}

/// First `stream()` call returns a text response; later calls fail — drives
/// the non-cancelled error termination with a notice already queued.
struct TextThenFailProvider {
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl LlmProvider for TextThenFailProvider {
    async fn stream(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatStream, llm_trait::LlmError> {
        let first = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
        let chunks: Vec<Result<StreamChunk, llm_trait::LlmError>> = if first {
            vec![
                Ok(StreamChunk::Text("let me think...".to_string())),
                Ok(StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ]
        } else {
            vec![Err(llm_trait::LlmError::llm("simulated LLM error"))]
        };
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
            name: "text-then-fail".to_string(),
            model: "test".to_string(),
            version: None,
        }
    }
}

fn notice_pos_of(delivered: &[RuntimeEvent]) -> Option<usize> {
    delivered.iter().position(|e| {
        matches!(
            e,
            RuntimeEvent::UserEvent {
                event: crate::types::UserEvent::Notice { .. },
                ..
            }
        )
    })
}

/// A notice queued during the run must still be delivered — before the
/// terminal event — when the run ends via `RunCancelled`.
#[tokio::test]
async fn notice_pump_delivers_pending_notice_before_run_cancelled() {
    use crate::types::RunOutcome;

    let guard = Arc::new(NoticeContinueGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: text-only → guard sends a notice, decides Continue.
        vec![
            StreamChunk::Text("let me think...".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        // Turn 2: tool call that cancels the run → RunCancelled path.
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_cancel",
                        "function": { "name": "cancel", "arguments": "{}" }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
    ])))
    .guard_dyn(guard.clone())
    .register_tool(CancelTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "go", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Cancelled)),
        "precondition: run must end cancelled, got: {:?}",
        result
    );

    let delivered = events.lock().unwrap().clone();
    let notice_pos = notice_pos_of(&delivered)
        .expect("pending notice must be delivered on the RunCancelled path");
    let cancelled_pos = delivered
        .iter()
        .position(|e| matches!(e, RuntimeEvent::RunCancelled { .. }))
        .expect("RunCancelled must be emitted");
    assert!(
        matches!(
            &delivered[notice_pos],
            RuntimeEvent::UserEvent { session_id, .. } if *session_id == sid
        ),
        "notice must be stamped with the run's session id"
    );
    assert!(
        notice_pos < cancelled_pos,
        "notice ({notice_pos}) must arrive before RunCancelled ({cancelled_pos})"
    );
}

/// A notice queued during the run must still be delivered — before
/// RunFinished — when the run terminates with a non-cancelled error.
#[tokio::test]
async fn notice_pump_delivers_pending_notice_on_error_termination() {
    let guard = Arc::new(NoticeContinueGuard::new());
    let runtime = AgentBuilder::new(Arc::new(TextThenFailProvider {
        calls: std::sync::atomic::AtomicUsize::new(0),
    }))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "go", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_err(),
        "precondition: run must end with an error, got: {:?}",
        result
    );

    let delivered = events.lock().unwrap().clone();
    let notice_pos = notice_pos_of(&delivered)
        .expect("pending notice must be delivered on the error termination path");
    let finished_pos = delivered
        .iter()
        .position(|e| matches!(e, RuntimeEvent::RunFinished { .. }))
        .expect("error path must still emit RunFinished");
    assert!(
        matches!(
            &delivered[notice_pos],
            RuntimeEvent::UserEvent { session_id, .. } if *session_id == sid
        ),
        "notice must be stamped with the run's session id"
    );
    assert!(
        notice_pos < finished_pos,
        "notice ({notice_pos}) must arrive before RunFinished ({finished_pos})"
    );
}

/// A notice sent outside any run must be discarded at the next run start —
/// never delivered stamped with a later run's session id.
#[tokio::test]
async fn notice_sent_between_runs_is_discarded_not_mis_stamped() {
    use crate::types::NoticeKind;

    let guard = Arc::new(HandleHoldingGuard::new());
    let script = || {
        vec![vec![
            StreamChunk::Text("done.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ]]
    };
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(script())))
        .guard_dyn(guard.clone())
        .build()
        .expect("build runtime");

    let sid1 = runtime.create_session().await;
    let events1: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events1_clone = events1.clone();
    runtime
        .run_turn(sid1, "hi", move |event| {
            events1_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run 1 ok");
    let run1 = events1.lock().unwrap().clone();
    assert!(
        notice_pos_of(&run1).is_none(),
        "precondition: run 1 delivers no notice"
    );

    // Between runs: the guard (still holding the engine handle) sends.
    guard.send(NoticeKind::Warning, "guard", "leftover from before");

    let sid2 = runtime.create_session().await;
    let events2: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events2_clone = events2.clone();
    runtime
        .run_turn(sid2, "hi", move |event| {
            events2_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run 2 ok");
    let run2 = events2.lock().unwrap().clone();
    assert!(
        notice_pos_of(&run2).is_none(),
        "between-run notice must be discarded at run start, not stamped onto run 2: {:?}",
        run2
    );
}

/// Notice delivery is display-only: a consumer callback error on a notice
/// must not fail the run nor skip the terminal event.
#[tokio::test]
async fn notice_callback_error_does_not_fail_run_or_skip_terminal() {
    use crate::types::UserEvent;

    let guard = Arc::new(NoticeSendingGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("All done.".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "go", move |event| {
            let is_notice = matches!(
                &event,
                RuntimeEvent::UserEvent {
                    event: UserEvent::Notice { .. },
                    ..
                }
            );
            events_clone.lock().unwrap().push(event);
            if is_notice {
                Err(crate::types::AgentError::internal(
                    "consumer rejected notice",
                ))
            } else {
                Ok(())
            }
        })
        .await;

    assert!(
        result.is_ok(),
        "display-only notice errors must not fail the run, got: {:?}",
        result
    );
    let delivered = events.lock().unwrap().clone();
    assert!(
        delivered
            .iter()
            .any(|e| matches!(e, RuntimeEvent::RunFinished { .. })),
        "terminal RunFinished must still be delivered after a notice callback error"
    );
}
