use super::*;

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
            .any(|e| matches!(e, RuntimeEvent::ActionStarted { .. })),
        "run_turn should emit ActionStarted"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ActionFinished { .. })),
        "run_turn should emit ActionFinished"
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

/// protects: an oversized turn input is refused BEFORE the push, with the
/// size spelled out — never popped into silence at push time (session
/// 20261108_c525b6c7's empty-request 400 came from exactly that pop).
/// fails_when: the message vanishes silently and the turn runs on nothing,
/// or the refusal is invisible. no_claim: where any provider's true limit
/// sits.
#[tokio::test]
async fn an_oversized_turn_input_is_refused_before_the_push() {
    // DummyProvider panics if the LLM is ever reached — the gate must fire
    // first.
    let runtime = AgentBuilder::new(Arc::new(DummyProvider))
        .system_prompt("test")
        .max_message_tokens(500)
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let big = "lorem ipsum dolor sit amet ".repeat(200);
    let result = runtime.run_turn(sid.clone(), &big, |_| Ok(())).await;

    let err = result.expect_err("the turn must be refused");
    assert!(
        format!("{err}").contains("NOT sent"),
        "the refusal names itself: {err}"
    );

    // Nothing was pushed — not popped-and-forgotten, never pushed at all.
    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session
            .chat_messages()
            .iter()
            .all(|m| !matches!(m, ChatMessage::User { .. })),
        "the oversized input must not reach the session: {:?}",
        session.chat_messages()
    );
}

/// protects: the incident's own journey end to end — two pasted screenshots
/// and a question complete a turn at the production limit (120k tokens),
/// through the size gate AND the safety valve, with both pictures still on
/// the user message. fails_when: any layer of the size machinery eats a
/// normal screenshot message. no_claim: what a provider bills for them.
#[tokio::test]
async fn two_screenshots_complete_a_turn_at_the_production_limit() {
    let scripted = Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("I see two screenshots.".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]]));
    let runtime = AgentBuilder::new(scripted)
        .system_prompt("test")
        .max_message_tokens(120_000)
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let shots: Vec<crate::types::ImageAttachment> = (0..2)
        .map(|_| crate::types::ImageAttachment::Base64 {
            data: crate::image_ops::to_base64(&crate::image_ops::test_fixtures::noisy_png(
                1278, 1152,
            )),
            media_type: Some("image/png".to_string()),
            detail: None,
        })
        .collect();
    let result = runtime
        .run_turn_with_images(sid.clone(), "look at these", shots, |_| Ok(()))
        .await;
    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "the turn must complete: {result:?}"
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let user = session
        .chat_messages()
        .iter()
        .find(|m| matches!(m, ChatMessage::User { .. }))
        .expect("the user message survives");
    match user {
        ChatMessage::User { images, .. } => {
            assert_eq!(images.len(), 2, "both screenshots survive to the wire")
        }
        unreachable => panic!("not a user message: {unreachable:?}"),
    }
}
