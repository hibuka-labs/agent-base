use super::*;

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
