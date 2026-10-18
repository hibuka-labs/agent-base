use agent_base::{AgentBuilder, StreamChunk};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

/// Mock LLM provider used to capture request parameters.
struct CaptureLlmProvider {
    /// Captured request bodies (JSON format)
    captured_bodies: Mutex<Vec<Value>>,
}

impl CaptureLlmProvider {
    fn new() -> Self {
        Self {
            captured_bodies: Mutex::new(Vec::new()),
        }
    }

    fn captured_bodies(&self) -> Vec<Value> {
        self.captured_bodies.lock().unwrap().clone()
    }
}

#[async_trait]
impl agent_base::llm_trait::LlmProvider for CaptureLlmProvider {
    async fn stream(
        &self,
        request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        // Construct a mock request body to verify parameter passing
        let mut body = json!({
            "model": "test-model",
            "messages": [],
            "tools": [],
            "stream": true,
        });

        if let Some(ref config) = request.reasoning {
            if let Some(enabled) = config.enabled
                && let Some(obj) = body.as_object_mut()
            {
                obj.insert("enable_thinking".to_string(), json!(enabled));
            }
            if let Some(budget) = config.budget_tokens
                && let Some(obj) = body.as_object_mut()
            {
                obj.insert("thinking_budget".to_string(), json!(budget));
            }
        }

        self.captured_bodies.lock().unwrap().push(body);

        // Return an empty stream
        let stream = futures_util::stream::iter(vec![Ok(StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        })]);
        Ok(agent_base::llm_trait::ChatStream::new(Box::pin(stream)))
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        unimplemented!()
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_thinking: true,
            max_context_tokens: Some(128_000),
            max_output_tokens: Some(16_384),
        }
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "capture".to_string(),
            model: "test-model".to_string(),
            version: None,
        }
    }
}

#[tokio::test]
async fn test_thinking_budget_parameter_passed() {
    let capture = Arc::new(CaptureLlmProvider::new());

    let runtime = AgentBuilder::new(capture.clone())
        .system_prompt("test")
        .enable_thinking(true)
        .thinking_budget(128)
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;
    let _ = runtime.run_turn_collect(session_id, "test").await;

    let bodies = capture.captured_bodies();
    assert!(
        !bodies.is_empty(),
        "Should have captured at least one request body"
    );

    let body = &bodies[0];
    println!(
        "Captured request body: {}",
        serde_json::to_string_pretty(body).unwrap()
    );

    // Verify enable_thinking is set correctly
    assert_eq!(
        body["enable_thinking"], true,
        "enable_thinking should be true"
    );

    // Verify thinking_budget is set correctly
    assert_eq!(
        body["thinking_budget"], 128,
        "thinking_budget should be 128"
    );
}

#[tokio::test]
async fn test_extra_body_format_for_thinking() {
    let capture = Arc::new(CaptureLlmProvider::new());

    let runtime = AgentBuilder::new(capture.clone())
        .system_prompt("test")
        .enable_thinking(true)
        .thinking_budget(128)
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;
    let _ = runtime.run_turn_collect(session_id, "test").await;

    let bodies = capture.captured_bodies();
    assert!(
        !bodies.is_empty(),
        "Should have captured at least one request body"
    );

    let body = &bodies[0];
    println!(
        "Captured request body: {}",
        serde_json::to_string_pretty(body).unwrap()
    );

    // Verify the parameters are present
    assert!(
        body.get("enable_thinking").is_some(),
        "enable_thinking should be present"
    );
    assert!(
        body.get("thinking_budget").is_some(),
        "thinking_budget should be present"
    );
}
