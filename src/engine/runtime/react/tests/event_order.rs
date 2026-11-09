use super::*;

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
                CheckpointStep::BeforeActions { .. } => "BeforeActions",
                CheckpointStep::AfterActions { .. } => "AfterActions",
            }),
            RuntimeEvent::RunFinished { .. } => Some("RunFinished"),
            RuntimeEvent::RunCancelled { .. } => Some("RunCancelled"),
            _ => None,
        })
        .collect()
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
            "BeforeActions",
            "AfterActions",
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
            "BeforeActions",
            "AfterActions",
            "RunFinished"
        ]
    );
}
