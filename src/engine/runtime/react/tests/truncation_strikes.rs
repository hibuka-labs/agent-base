use super::*;

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
                error.contains("consecutive damaged responses"),
                "hard-stop copy must state the real cause (provider-side                  truncation, cross-tool streak): {error}"
            );
            assert!(
                !error.contains("did not follow redirect"),
                "hard-stop copy must not blame the model for the redirect: {error}"
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
