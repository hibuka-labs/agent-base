use super::*;

// ── the cap's salvage turn (the wall must not discard the run's work) ──
//
// 2026-11-21 incident: two reviewer children died at the 32-turn wall right
// after their last `read_file` — the loop tripped `turn_count > max_turns`
// before the next LLM call, so no final answer was ever written and the
// parent received a bare "max turns exceeded" with every finding discarded.
// The cap must run ONE text-only synthesis call first, then end honestly
// with `MaxTurnsExceeded` carrying that text in the transcript and events.

#[tokio::test]
async fn cap_out_salvages_a_final_answer() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: a tool call — work in progress when the wall hits.
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
        // The salvage turn: text only.
        vec![
            StreamChunk::Text("final findings: the widget leaks".to_string()),
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
        .run_turn_collect(sid.clone(), "go")
        .await
        .expect("run_turn_collect");

    // The cap is still reported honestly.
    assert!(
        matches!(outcome, RunOutcome::MaxTurnsExceeded { .. }),
        "the run must still end MaxTurnsExceeded, got: {outcome:?}"
    );

    // The salvage text landed in the transcript as the child's final answer.
    let session = runtime.session(&sid).await.expect("session exists");
    let salvaged = session.chat_messages().iter().rev().find_map(|m| match m {
        ChatMessage::Assistant { content, .. } => content.clone(),
        _ => None,
    });
    assert!(
        salvaged
            .as_deref()
            .is_some_and(|c| c.contains("final findings")),
        "the salvaged final answer must be the last assistant message, got: {salvaged:?}"
    );

    // …and in the event stream after the last tool action — the seam
    // `extract_last_assistant_message` walks (parent-facing report).
    let last_action = events
        .iter()
        .rposition(|e| matches!(e, RuntimeEvent::ActionStarted { .. }))
        .expect("turn 1 tool call produced an ActionStarted");
    let after: String = events[last_action..]
        .iter()
        .filter_map(|e| match e {
            RuntimeEvent::TextDelta {
                text,
                agent_id: None,
                ..
            } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        after.contains("final findings"),
        "salvage text must arrive as TextDelta after the last ActionStarted, got: {after:?}"
    );
}

#[tokio::test]
async fn cap_out_without_salvage_text_still_ends_max_turns() {
    // If the salvage call yields nothing (provider hiccup, empty response),
    // the run must still end cleanly with MaxTurnsExceeded — never hang and
    // never relabel the outcome.
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
        // The salvage call gets an empty stream.
        vec![],
    ])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .execution_max_turns(1)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let (_events, outcome) = runtime
        .run_turn_collect(sid.clone(), "go")
        .await
        .expect("run_turn_collect");

    assert!(
        matches!(outcome, RunOutcome::MaxTurnsExceeded { .. }),
        "an empty salvage must not change the outcome, got: {outcome:?}"
    );
}
