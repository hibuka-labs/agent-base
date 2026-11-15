use super::*;
use crate::types::{AgentError, RetryConfig};
use async_trait::async_trait;
use llm_trait::{Capabilities, ChatRequest, ChatStream, LlmError, LlmProvider, ProviderInfo};
use std::pin::Pin;

struct StubProvider(&'static str);

#[async_trait]
impl LlmProvider for StubProvider {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Ok(ChatStream::new(Box::pin(futures_util::stream::empty())))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<llm_trait::ChatResponse, LlmError> {
        Err(LlmError::Llm("not implemented".into()))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "stub".to_string(),
            model: self.0.to_string(),
            version: None,
        }
    }
}

fn engine(model: &'static str) -> LlmEngine {
    LlmEngine::new(Arc::new(StubProvider(model)), EventBus::new(16))
}

#[test]
fn get_provider_and_set_provider_swap() {
    let engine = engine("model-a");
    assert_eq!(engine.get_provider().info().model, "model-a");
    engine.set_provider(Arc::new(StubProvider("model-b")));
    assert_eq!(engine.get_provider().info().model, "model-b");
}

#[test]
fn get_provider_returns_provider() {
    let engine = engine("model-a");
    let provider = engine.get_provider();
    assert_eq!(provider.info().model, "model-a");
}

#[tokio::test]
async fn process_stream_aggregates_text_usage_and_stop() {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![
        Ok(StreamChunk::Text("Hello".into())),
        Ok(StreamChunk::Text(" world".into())),
        Ok(StreamChunk::Usage(UsageInfo {
            prompt_tokens: Some(5),
            completion_tokens: Some(2),
            total_tokens: Some(7),
            reasoning_tokens: None,
            cache_read_tokens: None,
        })),
        Ok(StreamChunk::Stop {
            finish_reason: Some("stop".into()),
        }),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(move |e| {
                events_clone.lock().unwrap().push(e);
                Ok(())
            })),
            &cancel,
        )
        .await
        .unwrap();

    assert_eq!(result.full_text, "Hello world");
    assert!(!result.is_tool_call);
    assert_eq!(result.finish_reason.as_deref(), Some("stop"));
    let usage = result.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(5));
    assert_eq!(usage.completion_tokens, Some(2));
    assert_eq!(usage.total_tokens, Some(7));
    assert!(result.llm_duration_ms >= result.ttft_ms);

    let events = events.lock().unwrap();
    let texts: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            RuntimeEvent::TextDelta { text, .. } => Some(text.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["Hello".to_string(), " world".to_string()]);
}

#[tokio::test]
async fn process_stream_merges_partial_usage_events() {
    // Anthropic streams split usage across two events: message_start
    // carries prompt_tokens only, message_delta carries prompt (optional)
    // plus the final completion count. Replacing the struct wholesale
    // made every Anthropic session record input_tokens = 0.
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![
        Ok(StreamChunk::Usage(UsageInfo {
            prompt_tokens: Some(2),
            completion_tokens: Some(0),
            total_tokens: None,
            reasoning_tokens: None,
            cache_read_tokens: None,
        })),
        Ok(StreamChunk::Usage(UsageInfo {
            prompt_tokens: Some(63),
            completion_tokens: Some(26),
            total_tokens: None,
            reasoning_tokens: None,
            cache_read_tokens: None,
        })),
        Ok(StreamChunk::Stop {
            finish_reason: Some("end_turn".into()),
        }),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let noop = std::sync::Arc::new(std::sync::Mutex::new(|_e: RuntimeEvent| Ok(())));
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            noop,
            &cancel,
        )
        .await
        .unwrap();

    let usage = result.usage.unwrap();
    assert_eq!(usage.prompt_tokens, Some(63), "later Some wins");
    assert_eq!(usage.completion_tokens, Some(26));
}

#[tokio::test]
async fn process_stream_keeps_usage_when_delta_omits_prompt() {
    // Real Anthropic: message_delta has no input_tokens at all; the
    // message_start value must survive the merge rather than be zeroed.
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![
        Ok(StreamChunk::Usage(UsageInfo {
            prompt_tokens: Some(5),
            completion_tokens: Some(1),
            total_tokens: None,
            reasoning_tokens: None,
            cache_read_tokens: None,
        })),
        Ok(StreamChunk::Usage(UsageInfo {
            prompt_tokens: None,
            completion_tokens: Some(26),
            total_tokens: None,
            reasoning_tokens: None,
            cache_read_tokens: None,
        })),
        Ok(StreamChunk::Stop {
            finish_reason: Some("end_turn".into()),
        }),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let noop = std::sync::Arc::new(std::sync::Mutex::new(|_e: RuntimeEvent| Ok(())));
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            noop,
            &cancel,
        )
        .await
        .unwrap();

    let usage = result.usage.unwrap();
    assert_eq!(
        usage.prompt_tokens,
        Some(5),
        "prompt survives a partial delta"
    );
    assert_eq!(usage.completion_tokens, Some(26));
}

#[tokio::test]
async fn process_stream_aggregates_tool_calls() {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![
        Ok(StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [
                { "index": 0, "id": "call_1", "function": { "name": "shell", "arguments": "{\"cmd\":" } }
            ] }
        }))),
        Ok(StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": "\"ls\"" } }
            ] }
        }))),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_e| Ok(()))),
            &cancel,
        )
        .await
        .unwrap();

    assert!(result.is_tool_call);
    assert!(result.full_text.is_empty());
    assert_eq!(result.tool_calls.len(), 1);
    let tc = &result.tool_calls[0];
    assert_eq!(tc["id"].as_str(), Some("call_1"));
    assert_eq!(tc["type"].as_str(), Some("function"));
    assert_eq!(tc["function"]["name"].as_str(), Some("shell"));
    assert_eq!(
        tc["function"]["arguments"].as_str(),
        Some("{\"cmd\":\"ls\"")
    );
}

#[tokio::test]
async fn process_stream_emits_throttled_tool_call_drafts() {
    // Tool-call args arrive as fragments and are invisible to every
    // consumer until the stream drains — multi-KB briefs (spawn_agent)
    // were a tens-of-seconds black hole (session 20261006_9264ba3e).
    // The draft is the liveness signal: throttled to slot creation,
    // name arrival and each 512-char args bucket — never per fragment,
    // and never carrying the args themselves.
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let args_400 = "x".repeat(400);
    let args_200 = "y".repeat(200);
    let args_50 = "z".repeat(50);
    let chunks = vec![
        // Fresh slot + name → one draft.
        Ok(StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [
                { "index": 0, "id": "call_1", "function": { "name": "spawn_agent", "arguments": args_400 } }
            ] }
        }))),
        // +200 chars → crosses the 512 bucket → one draft.
        Ok(StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": args_200 } }
            ] }
        }))),
        // +50 chars → inside the same bucket → silent.
        Ok(StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [
                { "index": 0, "function": { "arguments": args_50 } }
            ] }
        }))),
        // Second call: fresh slot + name → one draft, count 2.
        Ok(StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [
                { "index": 1, "id": "call_2", "function": { "name": "shell", "arguments": "{}" } }
            ] }
        }))),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(move |e| {
                events_clone.lock().unwrap().push(e);
                Ok(())
            })),
            &cancel,
        )
        .await
        .unwrap();
    assert_eq!(result.tool_calls.len(), 2);

    let events = events.lock().unwrap();
    let drafts: Vec<(usize, String, usize, usize)> = events
        .iter()
        .filter_map(|e| match e {
            RuntimeEvent::ActionDraft {
                index,
                name,
                args_len,
                count,
                ..
            } => Some((*index, name.clone(), *args_len, *count)),
            _ => None,
        })
        .collect();
    assert_eq!(
        drafts,
        vec![
            (0, "spawn_agent".to_string(), 400, 1),
            (0, "spawn_agent".to_string(), 600, 1),
            (1, "shell".to_string(), 2, 2),
        ]
    );
}

#[tokio::test]
async fn process_stream_emits_text_after_tool_call_chunks() {
    // Regression: the old gate `!aggregator.is_tool_call` suppressed every
    // TextDelta/ThoughtDelta after the first tool_call chunk, but
    // aggregator.full_text still accumulated them — the user saw nothing
    // while the model context carried the text (user sees ≠ model sees,
    // session 20260904_efad759c: 2935 chars of prose rendered nowhere).
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![
        Ok(StreamChunk::Text("before".into())),
        Ok(StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [
                { "index": 0, "id": "call_1", "function": { "name": "shell", "arguments": "{}" } }
            ] }
        }))),
        Ok(StreamChunk::Text("interleaved".into())),
        Ok(StreamChunk::Thought("late thought".into())),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let events = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(move |e| {
                events_clone.lock().unwrap().push(e);
                Ok(())
            })),
            &cancel,
        )
        .await
        .unwrap();

    assert!(result.is_tool_call);
    assert_eq!(result.full_text, "beforeinterleaved");

    let events = events.lock().unwrap();
    let texts: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            RuntimeEvent::TextDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(texts, vec!["before", "interleaved"]);
    let thoughts: Vec<&str> = events
        .iter()
        .filter_map(|e| match e {
            RuntimeEvent::ThoughtDelta { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(thoughts, vec!["late thought"]);
}

#[tokio::test]
async fn process_stream_flags_reasoning_only_without_promoting() {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![Ok(StreamChunk::Thought("deep thought".into()))];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_e| Ok(()))),
            &cancel,
        )
        .await
        .unwrap();

    // Reasoning must NOT be promoted into the answer text.
    assert_eq!(result.full_text, "");
    assert_eq!(result.reasoning_text, "deep thought");
    assert!(!result.is_tool_call);
    assert!(result.reasoning_only, "reasoning-only should be flagged");
}

#[tokio::test]
async fn process_stream_keeps_reasoning_on_tool_call_turns() {
    // Issue #32 regression anchor at the stream-result boundary: a turn
    // that carries both Thought chunks and a tool call must keep the
    // reasoning text in the result. `thinking_bytes` is derived from this
    // string (same UTF-8 byte source as the `reasoning_len` log field),
    // so an empty reasoning_text here is what zeroed the metric on every
    // tool-call turn (session 20260927_4180845c: 27x undercount).
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![
        Ok(StreamChunk::Thought("why this tool, now".into())),
        Ok(StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [
                { "index": 0, "id": "call_1", "function": { "name": "shell", "arguments": "{}" } }
            ] }
        }))),
        Ok(StreamChunk::Thought("...and what it should do".into())),
        Ok(StreamChunk::Stop {
            finish_reason: Some("tool_calls".to_string()),
        }),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_e| Ok(()))),
            &cancel,
        )
        .await
        .unwrap();

    assert!(result.is_tool_call);
    assert!(!result.reasoning_only);
    assert_eq!(
        result.reasoning_text,
        "why this tool, now...and what it should do"
    );
    // UTF-8 bytes — the unit `thinking_bytes` reports downstream.
    assert_eq!(
        result.reasoning_text.len(),
        "why this tool, now".len() + "...and what it should do".len()
    );
}

#[tokio::test]
async fn process_stream_propagates_error() {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![
        Ok(StreamChunk::Text("partial".into())),
        Err(AgentError::internal("boom")),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let cancel = tokio_util::sync::CancellationToken::new();
    let err = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_e| Ok(()))),
            &cancel,
        )
        .await
        .err()
        .expect("stream should error");
    assert!(err.to_string().contains("boom"));
}

#[tokio::test]
async fn process_stream_cancelled() {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let stream: Pin<Box<dyn Stream<Item = AgentResult<StreamChunk>> + Send>> =
        Box::pin(futures_util::stream::pending());
    let mut rx = bus.subscribe();
    let cancel = tokio_util::sync::CancellationToken::new();
    cancel.cancel();
    let err = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_e| Ok(()))),
            &cancel,
        )
        .await
        .err()
        .expect("stream should error");
    assert!(matches!(err, AgentError::Cancelled));
}

// ── B5: chat_stream + retry + remaining process_stream branches ────────

/// A provider whose `stream()` always fails.
struct AlwaysFail;

#[async_trait]
impl LlmProvider for AlwaysFail {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        Err(LlmError::Llm("api down".into()))
    }

    async fn chat(&self, _request: ChatRequest) -> Result<llm_trait::ChatResponse, LlmError> {
        Err(LlmError::Llm("api down".into()))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "always-fail".to_string(),
            model: "always-fail".to_string(),
            version: None,
        }
    }
}

/// A provider that fails the first `n` `stream()` calls, then succeeds.
struct FailThenSucceed {
    remaining: std::sync::Mutex<usize>,
}

impl FailThenSucceed {
    fn new(failures: usize) -> Self {
        Self {
            remaining: std::sync::Mutex::new(failures),
        }
    }
}

#[async_trait]
impl LlmProvider for FailThenSucceed {
    async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
        let mut remaining = self.remaining.lock().unwrap();
        if *remaining > 0 {
            *remaining -= 1;
            Err(LlmError::Llm("transient".into()))
        } else {
            Ok(ChatStream::new(Box::pin(futures_util::stream::empty())))
        }
    }

    async fn chat(&self, _request: ChatRequest) -> Result<llm_trait::ChatResponse, LlmError> {
        Err(LlmError::Llm("not implemented".into()))
    }

    fn capabilities(&self) -> Capabilities {
        Capabilities::default()
    }

    fn info(&self) -> ProviderInfo {
        ProviderInfo {
            name: "fail-then-succeed".to_string(),
            model: "fail-then-succeed".to_string(),
            version: None,
        }
    }
}

#[tokio::test]
async fn chat_stream_returns_stream_on_ok() {
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), EventBus::new(16));
    let mut stream = engine
        .chat_stream(
            &[],
            &[],
            None,
            None,
            false,
            &tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
    let next = futures_util::StreamExt::next(&mut stream).await;
    assert!(next.is_none());
}

#[tokio::test]
async fn chat_stream_with_thinking_disabled() {
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), EventBus::new(16));
    // Test with thinking_disabled=true — should still work, just without reasoning
    let mut stream = engine
        .chat_stream(
            &[],
            &[],
            None,
            None,
            true,
            &tokio_util::sync::CancellationToken::new(),
        )
        .await
        .unwrap();
    let next = futures_util::StreamExt::next(&mut stream).await;
    assert!(next.is_none());
}

#[tokio::test]
async fn chat_stream_forwards_error() {
    let engine = LlmEngine::new(Arc::new(AlwaysFail), EventBus::new(16));
    let err = engine
        .chat_stream(
            &[],
            &[],
            None,
            None,
            false,
            &tokio_util::sync::CancellationToken::new(),
        )
        .await
        .err()
        .expect("should fail");
    assert!(err.to_string().contains("api down"));
}

#[tokio::test]
async fn run_llm_turn_retries_then_succeeds() {
    let engine = LlmEngine::new(Arc::new(FailThenSucceed::new(2)), EventBus::new(16));
    let cfg = RetryConfig {
        max_retries: 3,
        initial_backoff_ms: 1,
        max_backoff_ms: 1000,
        backoff_multiplier: 2.0,
        jitter: false,
    };
    let _stream = engine
        .run_llm_turn_with_retry(
            &SessionId::new(1),
            &[],
            &[],
            None,
            None,
            cfg,
            false,
            &tokio_util::sync::CancellationToken::new(),
        )
        .await
        .expect("should succeed after retries");
}

#[tokio::test]
async fn run_llm_turn_retries_exhausted() {
    let engine = LlmEngine::new(Arc::new(FailThenSucceed::new(100)), EventBus::new(16));
    let cfg = RetryConfig {
        max_retries: 2,
        initial_backoff_ms: 1,
        max_backoff_ms: 1,
        backoff_multiplier: 2.0,
        jitter: false,
    };
    let err = engine
        .run_llm_turn_with_retry(
            &SessionId::new(1),
            &[],
            &[],
            None,
            None,
            cfg,
            false,
            &tokio_util::sync::CancellationToken::new(),
        )
        .await
        .err()
        .expect("should be exhausted");
    assert!(err.to_string().contains("transient"));
}

#[tokio::test]
async fn chat_stream_cancelled_while_awaiting_response() {
    // A provider whose `stream()` never resolves — the shape of a silent
    // network drop while waiting for response headers. Cancelling must
    // abort the wait instead of hanging forever.
    struct NeverResolves;

    #[async_trait]
    impl LlmProvider for NeverResolves {
        async fn stream(&self, _request: ChatRequest) -> Result<ChatStream, LlmError> {
            std::future::pending().await
        }

        async fn chat(&self, _request: ChatRequest) -> Result<llm_trait::ChatResponse, LlmError> {
            Err(LlmError::Llm("not implemented".into()))
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities::default()
        }

        fn info(&self) -> ProviderInfo {
            ProviderInfo {
                name: "never-resolves".to_string(),
                model: "never-resolves".to_string(),
                version: None,
            }
        }
    }

    let engine = LlmEngine::new(Arc::new(NeverResolves), EventBus::new(16));
    let token = tokio_util::sync::CancellationToken::new();
    let fut = engine.chat_stream(&[], &[], None, None, false, &token);
    tokio::pin!(fut);
    token.cancel();
    let err = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .expect("chat_stream must resolve after cancel")
        .err()
        .expect("must be cancelled");
    assert!(err.is_cancelled(), "expected Cancelled, got: {err}");
}

#[tokio::test]
async fn process_stream_tool_call_missing_delta_continues() {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![
        Ok(StreamChunk::ToolCall(
            serde_json::json!({ "no_delta": true }),
        )),
        Ok(StreamChunk::Stop {
            finish_reason: Some("stop".into()),
        }),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_e| Ok(()))),
            &cancel,
        )
        .await
        .unwrap();
    assert!(result.is_tool_call);
    assert!(result.tool_calls.is_empty());
}

#[tokio::test]
async fn process_stream_tool_call_empty_id_and_name_ignored() {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![Ok(StreamChunk::ToolCall(serde_json::json!({
        "delta": { "tool_calls": [
            { "index": 0, "id": "", "function": { "name": "", "arguments": "{}" } }
        ] }
    })))];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_e| Ok(()))),
            &cancel,
        )
        .await
        .unwrap();
    // A bucket that carries an index but never a function name is a
    // phantom fragment some OpenAI-compatible providers emit. It must NOT
    // surface as a tool call — kept, it becomes a name-less assistant
    // tool_call and the NEXT request is rejected HTTP 400 "missing a
    // function name". The stream is fully drained here, so an empty name
    // at assembly time is permanent, not a name that arrives later.
    assert_eq!(result.tool_calls.len(), 0);
}

#[tokio::test]
async fn process_stream_stop_none_finish_reason_ignored() {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let chunks = vec![
        Ok(StreamChunk::Stop {
            finish_reason: Some("length".into()),
        }),
        Ok(StreamChunk::Stop {
            finish_reason: None,
        }),
    ];
    let stream = Box::pin(futures_util::stream::iter(chunks));
    let mut rx = bus.subscribe();
    let cancel = tokio_util::sync::CancellationToken::new();
    let result = engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_e| Ok(()))),
            &cancel,
        )
        .await
        .unwrap();
    assert_eq!(result.finish_reason.as_deref(), Some("length"));
}

#[test]
fn emit_text_delta_emits_event() {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let mut rx = bus.subscribe();
    engine.emit_text_delta(&SessionId::new(1), "hello".into());
    let ev = rx.try_recv().expect("event should be emitted");
    assert!(matches!(ev, RuntimeEvent::TextDelta { text, .. } if text == "hello"));
}

/// protects: a word-only history builds byte-identical — the image budget is
/// a no-op without pictures and the wire copy is not rewritten in passing.
/// fails_when: text messages gain, lose or reorder content during the build.
/// no_claim: picture handling — pinned in `image_tests.rs` and
/// `crate::image_ops`'s own tests.
#[test]
fn a_word_only_history_is_untouched() {
    let messages = vec![
        ChatMessage::user("hi"),
        ChatMessage::tool("t1", "done"),
        ChatMessage::assistant("ok"),
    ];
    let request = build_chat_request(&messages, &[], None, None, false, None)
        .expect("a word-only history builds");
    assert_eq!(
        format!("{:?}", request.messages),
        format!("{messages:?}"),
        "byte-identical pass-through"
    );
}

// ── Turn-safe provider swap (set_provider_when_idle) ───────────────────────

/// protects: a swap staged while a run executes lands only when the last
/// active run ends — one conversation is never split across two models (the
/// tool-continuation steps must keep the provider the turn started with).
/// fails_when: the swap lands mid-run, or never lands after the run ends.
/// no_claim: which provider a product stages or when it stages one.
#[test]
fn a_swap_staged_during_a_run_lands_when_the_run_ends() {
    let engine = engine("old");
    let run = engine.begin_run();
    assert!(engine.has_active_run());
    let deferred = engine.set_provider_when_idle(Arc::new(StubProvider("new")));
    assert!(deferred, "the swap defers while a run is active");
    assert_eq!(
        engine.get_provider().info().model,
        "old",
        "the live run keeps its provider"
    );
    drop(run);
    assert!(!engine.has_active_run());
    assert_eq!(
        engine.get_provider().info().model,
        "new",
        "the staged swap lands at run end"
    );
}

/// protects: an idle engine swaps at once — `true` back means deferred, and
/// a caller that says "applies after the current turn" reads that bit.
/// fails_when: an idle swap parks until some future run ends. no_claim:
/// the product's confirm wording.
#[test]
fn a_swap_staged_while_idle_lands_immediately() {
    let engine = engine("old");
    let deferred = engine.set_provider_when_idle(Arc::new(StubProvider("new")));
    assert!(!deferred, "an idle engine swaps at once");
    assert_eq!(engine.get_provider().info().model, "new");
}

/// protects: overlapping runs hold the swap until the LAST one ends.
/// fails_when: the swap lands while any run is still executing. no_claim:
/// why runs overlap (sub-agents, back-to-back turns).
#[test]
fn overlapping_runs_hold_the_swap_until_the_last_one_ends() {
    let engine = engine("old");
    let run_a = engine.begin_run();
    let run_b = engine.begin_run();
    assert!(engine.set_provider_when_idle(Arc::new(StubProvider("new"))));
    drop(run_a);
    assert_eq!(
        engine.get_provider().info().model,
        "old",
        "run_b is still active"
    );
    drop(run_b);
    assert_eq!(engine.get_provider().info().model, "new");
}

/// protects: two staged swaps resolve last-wins — a flip-flop of picks must
/// land on the final choice, not replay both. fails_when: the first staged
/// swap overwrites the second. no_claim: how a product stages two swaps.
#[test]
fn the_last_staged_swap_wins() {
    let engine = engine("old");
    let run = engine.begin_run();
    assert!(engine.set_provider_when_idle(Arc::new(StubProvider("first"))));
    assert!(engine.set_provider_when_idle(Arc::new(StubProvider("last"))));
    drop(run);
    assert_eq!(engine.get_provider().info().model, "last");
}

#[test]
fn provider_handle_follows_set_provider() {
    // A handle is the anti-snapshot seam (session 20261107_27dc727a): holders
    // that cloned the provider at build time kept judging on the boot model
    // after a `/model` swap. A handle must always resolve the current slot.
    let engine = engine("model-a");
    let handle = engine.provider_handle();
    assert_eq!(handle.current().info().model, "model-a");
    engine.set_provider(Arc::new(StubProvider("model-b")));
    assert_eq!(handle.current().info().model, "model-b");
    // Independent handles stay in sync — they alias one slot.
    let second = handle.clone();
    engine.set_provider(Arc::new(StubProvider("model-c")));
    assert_eq!(second.current().info().model, "model-c");
}

#[test]
fn standalone_provider_handle_resolves_its_initial() {
    let handle = ProviderHandle::new(Arc::new(StubProvider("model-x")));
    assert_eq!(handle.current().info().model, "model-x");
}
