use super::*;

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
