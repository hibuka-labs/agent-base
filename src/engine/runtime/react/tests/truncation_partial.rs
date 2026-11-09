use super::*;

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
