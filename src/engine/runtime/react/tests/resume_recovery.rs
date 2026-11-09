use super::*;

/// Tool double whose only behavior is counting executions — side-effect
/// evidence for the D3 "never re-execute" contract.
struct CountingTool {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl Tool for CountingTool {
    fn name(&self) -> &'static str {
        "count"
    }

    fn description(&self) -> &'static str {
        "Count invocations"
    }

    fn schema(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(vec![Content::text("counted")])
    }
}

// --- D12 recovery group — the three assertions, tool-call backend -----------
// (1) never re-execute: zero side-effect replay, dangling actions get
//     synthetic fates (resume_does_not_reexecute_dangling_actions,
//     resume_replays_recorded_observation_without_executing).
// (2) counting trusts the persisted values, crash window included
//     (resume_turn_count_trusts_the_persisted_value,
//     resume_counters_trust_the_persisted_run_state,
//     checkpoint_run_state_matches_the_recorded_batch).
// (3) history rebuilds correctly
//     (resume_rebuild_pairs_every_call_with_exactly_one_result,
//     resume_migrated_after_batch_replays_legacy_summaries).
// Cross-backend resume refusal is pinned below. The same group runs a second
// time on the CodeCellExecutor backend when it lands (T10).

#[tokio::test]
async fn resume_does_not_reexecute_dangling_actions() {
    // D3: side effects are one-shot. A dangling batch (crash window) is
    // synthesized as「已中断/未执行」— the tool never runs again.
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("resumed answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CountingTool {
        calls: calls.clone(),
    })
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let checkpoint = CheckpointData::new(
        sid.clone(),
        "resume",
        CheckpointStep::BeforeActions {
            actions: vec![crate::types::Action::from_tool_call(
                "call_resume",
                "count",
                "{}",
            )],
        },
        0,
        ActionFormat::ToolCall,
        RunState::default(),
    );

    let result = runtime
        .runner
        .resume_from_checkpoint(checkpoint, |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "resume_from_checkpoint should complete: {:?}",
        result.err()
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "dangling action must never execute"
    );

    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::Tool { content, .. } if content.contains("已中断/未执行"))
        ),
        "the model must be told plainly that nothing ran"
    );
    assert!(
        !session
            .chat_messages()
            .iter()
            .any(|m| matches!(m, ChatMessage::Tool { content, .. } if content.contains("counted"))),
        "no tool output may appear for a batch that never ran"
    );
}

#[tokio::test]
async fn resume_replays_recorded_observation_without_executing() {
    // D3: an AfterActions checkpoint carries its observation — resume
    // rebuilds history from it instead of re-running the batch.
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("resumed answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CountingTool {
        calls: calls.clone(),
    })
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let observation = Observation {
        outcomes: vec![ActionOutcome {
            action_id: "call_resume".to_string(),
            name: "count".to_string(),
            ok: true,
            denied: false,
            summary: "stored result".to_string(),
            details: None,
        }],
        events: vec![],
        history_messages: vec![ChatMessage::tool("call_resume", "stored result")],
        batch_status: BatchStatus::Completed,
    };
    let checkpoint = CheckpointData::new(
        sid.clone(),
        "resume",
        CheckpointStep::AfterActions {
            actions: vec![crate::types::Action::from_tool_call(
                "call_resume",
                "count",
                "{}",
            )],
            observation: Some(observation),
        },
        1,
        ActionFormat::ToolCall,
        RunState::default(),
    );

    let result = runtime
        .runner
        .resume_from_checkpoint(checkpoint, |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "resume_from_checkpoint should complete: {:?}",
        result.err()
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "recorded observation must be replayed, not re-executed"
    );

    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::Tool { content, .. } if content.contains("stored result"))
        ),
        "the stored observation's history is rebuilt verbatim"
    );
}

#[tokio::test]
async fn resume_rebuild_pairs_every_call_with_exactly_one_result() {
    // 历史重建正确: the replayed protocol pair keeps provider validation
    // sound — one assistant tool-call message, then one result per call.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("resumed answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CountingTool {
        calls: Arc::new(AtomicUsize::new(0)),
    })
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let checkpoint = CheckpointData::new(
        sid.clone(),
        "resume",
        CheckpointStep::BeforeActions {
            actions: vec![
                crate::types::Action::from_tool_call("call_a", "count", "{}"),
                crate::types::Action::from_tool_call("call_b", "count", "{}"),
            ],
        },
        0,
        ActionFormat::ToolCall,
        RunState::default(),
    );

    let result = runtime
        .runner
        .resume_from_checkpoint(checkpoint, |_| Ok(()))
        .await;
    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "resume_from_checkpoint should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages();
    let assistant_pos = messages
        .iter()
        .position(
            |m| matches!(m, ChatMessage::Assistant { tool_calls: Some(calls), .. } if calls.len() == 2),
        )
        .expect("assistant tool-call message present");
    for id in ["call_a", "call_b"] {
        let hits: Vec<usize> = messages
            .iter()
            .enumerate()
            .filter(
                |(_, m)| matches!(m, ChatMessage::Tool { tool_call_id, .. } if tool_call_id == id),
            )
            .map(|(i, _)| i)
            .collect();
        assert_eq!(hits.len(), 1, "exactly one result for {id}: {hits:?}");
        assert!(
            hits[0] > assistant_pos,
            "result for {id} must come after the assistant message"
        );
    }
}

#[tokio::test]
async fn resume_turn_count_trusts_the_persisted_value() {
    // D12 (2): counting continues from the checkpoint's persisted value,
    // never restarting at zero.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("resumed answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CountingTool {
        calls: Arc::new(AtomicUsize::new(0)),
    })
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let checkpoint = CheckpointData::new(
        sid.clone(),
        "resume",
        CheckpointStep::AfterUserInput,
        7,
        ActionFormat::ToolCall,
        RunState::default(),
    );

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .runner
        .resume_from_checkpoint(checkpoint, move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "resume_from_checkpoint should complete: {:?}",
        result.err()
    );

    let events = events.lock().unwrap();
    let llm_turns: Vec<u32> = events
        .iter()
        .filter_map(|e| match e {
            RuntimeEvent::Checkpoint { checkpoint, .. } => match &checkpoint.step {
                CheckpointStep::BeforeLlm { .. } => Some(checkpoint.turn_count),
                _ => None,
            },
            _ => None,
        })
        .collect();
    assert_eq!(
        llm_turns,
        vec![8],
        "the resumed loop continues at persisted turn + 1"
    );
}

#[tokio::test]
async fn resume_counters_trust_the_persisted_run_state() {
    // D12 (2) + D5+OV-11: counting restores the RunState co-written in the
    // checkpoint. The dangling batch (2 actions) is replayed as
    // 「已中断/未执行」 and must NOT move the counts — neither the replayed
    // assistant tool_calls (message lens → 2) nor a "count what was pending"
    // read (7 + 2 = 9) is the truth. The persisted value is.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("resumed answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CountingTool {
        calls: Arc::new(AtomicUsize::new(0)),
    })
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let run_state = RunState {
        turn_actions: 7,
        total_actions: 11,
        run_has_actions: true,
        ..RunState::default()
    };
    let checkpoint = CheckpointData::new(
        sid.clone(),
        "resume",
        CheckpointStep::BeforeActions {
            actions: vec![
                crate::types::Action::from_tool_call("call_a", "count", "{}"),
                crate::types::Action::from_tool_call("call_b", "count", "{}"),
            ],
        },
        7,
        ActionFormat::ToolCall,
        run_state,
    );

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .runner
        .resume_from_checkpoint(checkpoint, move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "resume_from_checkpoint should complete: {:?}",
        result.err()
    );

    let events = events.lock().unwrap();
    let first_llm = events
        .iter()
        .find_map(|e| match e {
            RuntimeEvent::Checkpoint { checkpoint, .. }
                if matches!(checkpoint.step, CheckpointStep::BeforeLlm { .. }) =>
            {
                Some(checkpoint)
            }
            _ => None,
        })
        .expect("the resumed loop reaches BeforeLlm");
    let rs = first_llm
        .run_state
        .as_ref()
        .expect("native checkpoints co-write RunState");
    assert_eq!(
        rs.turn_actions, 7,
        "counts come from the persisted value, not the replayed batch"
    );
    assert_eq!(
        rs.total_actions, 11,
        "the session total is restored, not recounted from messages"
    );
    assert!(rs.run_has_actions);
}

#[tokio::test]
async fn checkpoint_run_state_matches_the_recorded_batch() {
    // D5+OV-11 对账: the AfterActions line carries the batch (账) and the
    // post-bump counts (数) in one write — they can never disagree. The
    // BeforeActions line carries the pre-batch counts.
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_1",
                        "function": {
                            "name": "count",
                            "arguments": "{}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("done".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(CountingTool {
        calls: calls.clone(),
    })
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "count once", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "run should complete: {:?}",
        result.err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1, "the tool ran exactly once");

    let events = events.lock().unwrap();
    let mut before = None;
    let mut after = None;
    for e in events.iter() {
        if let RuntimeEvent::Checkpoint { checkpoint, .. } = e {
            match &checkpoint.step {
                CheckpointStep::BeforeActions { actions } => {
                    before = Some((actions.len(), checkpoint.run_state.clone()));
                }
                CheckpointStep::AfterActions { actions, .. } => {
                    after = Some((actions.len(), checkpoint.run_state.clone()));
                }
                _ => {}
            }
        }
    }
    let (before_actions, before_rs) = before.expect("BeforeActions checkpoint emitted");
    let (after_actions, after_rs) = after.expect("AfterActions checkpoint emitted");
    assert_eq!(before_actions, 1, "the line records the batch (账)");
    assert_eq!(after_actions, 1, "the line records the batch (账)");

    let before_rs = before_rs.expect("native checkpoints co-write RunState");
    let after_rs = after_rs.expect("native checkpoints co-write RunState");
    // Pre-batch: nothing counted yet. Post-batch: the batch is counted.
    assert_eq!(before_rs.turn_actions, 0, "pre-batch counts on the line");
    assert_eq!(before_rs.total_actions, 0, "pre-batch counts on the line");
    assert_eq!(
        after_rs.turn_actions, after_actions,
        "数 on the line matches the 账 on the line"
    );
    assert_eq!(
        after_rs.total_actions, after_actions,
        "数 on the line matches the 账 on the line"
    );
    assert!(after_rs.run_has_actions);
}

#[tokio::test]
async fn resume_migrated_after_batch_replays_legacy_summaries() {
    // T5 mark + T6: a pre-M3 `AfterToolCalls { results }` payload recorded
    // result summaries — those are evidence and are replayed as-is, not
    // rewritten into「已中断/未执行」.
    let calls = Arc::new(AtomicUsize::new(0));
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("resumed answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(CountingTool {
        calls: calls.clone(),
    })
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let checkpoint = CheckpointData {
        session_id: sid.clone(),
        user_input: "resume".to_string(),
        step: CheckpointStep::AfterActions {
            actions: vec![crate::types::Action::from_tool_call(
                "call_2", "count", "{}",
            )],
            observation: None,
        },
        turn_count: 1,
        version: 0,
        backend: ActionFormat::ToolCall,
        migrated_from: Some(MigrationSource {
            kind: MigrationKind::ToolCallTriplesV0,
            legacy_results: vec![ToolResultData {
                tool_call_id: "call_2".to_string(),
                tool_name: "count".to_string(),
                summary: "file list".to_string(),
            }],
        }),
        // A converted legacy payload carries no counting state (不猜不回填).
        run_state: None,
    };

    let result = runtime
        .runner
        .resume_from_checkpoint(checkpoint, |_| Ok(()))
        .await;
    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "resume_from_checkpoint should complete: {:?}",
        result.err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0, "replay never executes");

    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::Tool { content, .. } if content.contains("file list"))
        ),
        "legacy summaries are recorded evidence — replayed verbatim"
    );
    assert!(
        !session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::Tool { content, .. } if content.contains("已中断/未执行"))
        ),
        "a completed pre-M3 batch must not be reported as interrupted"
    );
}

#[tokio::test]
async fn resume_from_checkpoint_refuses_cross_backend_checkpoints() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("must not run".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    // A checkpoint pinned to the code_act backend. This runtime executes
    // tool-call actions — the resume must refuse before doing anything, and
    // the refusal must name both sides (跨后端恢复被拒且明说).
    let checkpoint = CheckpointData::new(
        sid.clone(),
        "resume",
        CheckpointStep::BeforeActions {
            actions: vec![crate::types::Action {
                id: "cell_1".to_string(),
                name: "code_cell".to_string(),
                args: Value::String(r#"{"code":"1+1"}"#.to_string()),
                identity: r#"code_cell::{"code":"1+1"}"#.to_string(),
                format: ActionFormat::CodeAct,
            }],
        },
        0,
        ActionFormat::CodeAct,
        RunState::default(),
    );

    let result = runtime
        .runner
        .resume_from_checkpoint(checkpoint, |_| Ok(()))
        .await;

    let err = match result {
        Ok(outcome) => panic!("cross-backend resume must be refused, got {outcome:?}"),
        Err(e) => e,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("CodeAct"),
        "refusal names the pinned backend: {msg}"
    );
    assert!(
        msg.contains("ToolCall"),
        "refusal names the resuming backend: {msg}"
    );
    assert!(msg.contains("refused"), "refusal is explicit: {msg}");
}
