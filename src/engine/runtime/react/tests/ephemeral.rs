use super::*;

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
