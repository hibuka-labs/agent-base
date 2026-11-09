use super::*;

// --- D12 recovery group, second run — the mounted action backend (T10c) -----
// The same three assertions as the tool-call group above (never re-execute,
// counting trusts the persisted values, history rebuilds correctly), on a
// runtime with an action backend mounted. The double below proves the kernel
// ROUTING (parse → flow → 账+数 checkpoints → observation replay); the real
// CodeAct shape-A history pin lives in agent-works' adapter tests.

/// Mounted action backend double: claims `code:`-prefixed text turns as one
/// CodeAct cell each, counts executions, and renders its own synthesis shape.
/// Envelope turns never reach it — the kernel routes those to tools.
struct FakeCodeBackend {
    executed: Arc<AtomicUsize>,
}

#[async_trait]
impl ActionExecutor for FakeCodeBackend {
    fn parse(&self, turn: &LlmTurnResult) -> Result<ParseOutcome, ParseError> {
        match turn.full_text.strip_prefix("code:") {
            Some(code) => Ok(ParseOutcome::Actions(vec![Action::from_code_cell(
                "cell_1",
                "test",
                code.trim(),
            )])),
            None => Ok(ParseOutcome::Empty),
        }
    }

    async fn execute(
        &mut self,
        actions: Vec<Action>,
        _ctx: &ExecCtx,
    ) -> Result<Observation, ExecError> {
        self.executed.fetch_add(1, Ordering::SeqCst);
        Ok(Observation {
            outcomes: actions
                .iter()
                .map(|a| ActionOutcome {
                    action_id: a.id.clone(),
                    name: a.name.clone(),
                    ok: true,
                    denied: false,
                    summary: "ran".to_string(),
                    details: None,
                })
                .collect(),
            events: vec![],
            history_messages: vec![
                ChatMessage::assistant("[cell]"),
                ChatMessage::user("[observation]\nran"),
            ],
            batch_status: BatchStatus::Completed,
        })
    }
}

#[async_trait]
impl SessionHandle for FakeCodeBackend {
    fn is_alive(&self) -> bool {
        true
    }

    async fn rebuild(&mut self) -> Result<(), SessionError> {
        Ok(())
    }
}

impl ActionBackend for FakeCodeBackend {
    fn format(&self) -> ActionFormat {
        ActionFormat::CodeAct
    }

    fn render_fates(&self, actions: &[Action], fates: &[ActionOutcome]) -> Vec<ChatMessage> {
        // The double's own history shape (the shape is the backend's
        // protocol; the fate text inside it is the kernel's vocabulary).
        let blocks: Vec<String> = actions.iter().map(|a| format!("[cell {}]", a.id)).collect();
        let mut pane = String::from("[observation]\n");
        for fate in fates {
            pane.push_str(&fate.summary);
            pane.push('\n');
        }
        vec![
            ChatMessage::assistant(blocks.join("\n")),
            ChatMessage::user(pane),
        ]
    }
}

/// One runtime with the double mounted and a scripted single answer for the
/// resumed loop.
fn code_backend_runtime(executed: Arc<AtomicUsize>) -> AgentRuntime {
    AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("resumed answer".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .action_backend(Box::new(FakeCodeBackend { executed }))
    .build()
    .expect("build runtime")
}

#[tokio::test]
async fn resume_does_not_reexecute_dangling_code_actions() {
    // D3 twin: a dangling CodeAct batch (crash window) is synthesized in the
    // mounted backend's shape with the kernel's「已中断/未执行」fates — the
    // cell never runs again.
    let executed = Arc::new(AtomicUsize::new(0));
    let runtime = code_backend_runtime(executed.clone());
    let sid = runtime.create_session().await;

    let checkpoint = CheckpointData::new(
        sid.clone(),
        "resume",
        CheckpointStep::BeforeActions {
            actions: vec![Action::from_code_cell("cell_1", "task", "print(1)")],
        },
        0,
        ActionFormat::CodeAct,
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
        executed.load(Ordering::SeqCst),
        0,
        "dangling action must never execute"
    );

    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::User { content, .. } if content.contains("已中断/未执行"))
        ),
        "the model must be told plainly that nothing ran"
    );
    assert!(
        !session
            .chat_messages()
            .iter()
            .any(|m| matches!(m, ChatMessage::User { content, .. } if content.contains("ran\n"))),
        "no execution output may appear for a batch that never ran"
    );
}

#[tokio::test]
async fn resume_replays_recorded_code_observation_without_executing() {
    // D3 twin: an AfterActions checkpoint carries its observation — resume
    // rebuilds history from it (the backend's messages verbatim) instead of
    // re-running the batch.
    let executed = Arc::new(AtomicUsize::new(0));
    let runtime = code_backend_runtime(executed.clone());
    let sid = runtime.create_session().await;

    let observation = Observation {
        outcomes: vec![ActionOutcome {
            action_id: "cell_1".to_string(),
            name: "code_cell".to_string(),
            ok: true,
            denied: false,
            summary: "stored result".to_string(),
            details: None,
        }],
        events: vec![],
        history_messages: vec![
            ChatMessage::assistant("[cell cell_1]"),
            ChatMessage::user("[observation]\nstored result"),
        ],
        batch_status: BatchStatus::Completed,
    };
    let checkpoint = CheckpointData::new(
        sid.clone(),
        "resume",
        CheckpointStep::AfterActions {
            actions: vec![Action::from_code_cell("cell_1", "task", "print(1)")],
            observation: Some(observation),
        },
        1,
        ActionFormat::CodeAct,
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
        executed.load(Ordering::SeqCst),
        0,
        "recorded observation must be replayed, not re-executed"
    );

    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::User { content, .. } if content.contains("stored result"))
        ),
        "the stored observation's history is rebuilt verbatim"
    );
}

#[tokio::test]
async fn resume_counters_trust_the_persisted_run_state_for_code_batches() {
    // D12 (2) twin + D5+OV-11: the dangling code batch must NOT move the
    // counts — the persisted value is the truth.
    let executed = Arc::new(AtomicUsize::new(0));
    let runtime = code_backend_runtime(executed.clone());
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
                Action::from_code_cell("cell_1", "task", "print(1)"),
                Action::from_code_cell("cell_2", "task", "print(2)"),
            ],
        },
        7,
        ActionFormat::CodeAct,
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
async fn resume_rebuild_pairs_every_code_cell_with_its_fate() {
    // 历史重建正确 twin: the replayed shape-A pair keeps the transcript sound
    // — one assistant message carrying the batch, then one fate line per
    // cell, in submission order.
    let executed = Arc::new(AtomicUsize::new(0));
    let runtime = code_backend_runtime(executed.clone());
    let sid = runtime.create_session().await;

    let checkpoint = CheckpointData::new(
        sid.clone(),
        "resume",
        CheckpointStep::BeforeActions {
            actions: vec![
                Action::from_code_cell("cell_a", "task", "print(1)"),
                Action::from_code_cell("cell_b", "task", "print(2)"),
            ],
        },
        0,
        ActionFormat::CodeAct,
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
            |m| matches!(m, ChatMessage::Assistant { content: Some(c), .. } if c.contains("[cell cell_a]") && c.contains("[cell cell_b]")),
        )
        .expect("one assistant message carries the whole batch");
    let observation_pos = messages
        .iter()
        .position(
            |m| matches!(m, ChatMessage::User { content, .. } if content.starts_with("[observation]")),
        )
        .expect("the observation pane follows");
    assert!(
        observation_pos > assistant_pos,
        "shape-A order: assistant blocks first, observation second"
    );
    let pane = match &messages[observation_pos] {
        ChatMessage::User { content, .. } => content.clone(),
        _ => unreachable!(),
    };
    let fate_lines = pane.lines().filter(|l| *l != "[observation]").count();
    assert_eq!(
        fate_lines, 2,
        "one fate line per cell, nothing invented: {pane:?}"
    );
}

#[tokio::test]
async fn a_text_turn_with_cells_runs_through_the_mounted_backend() {
    // Live smoke of the mount (T10c): a text turn carrying cells claims to
    // the mounted backend's parse, executes through its flow, and lands the
    // 账+数 checkpoint pair with the backend's observation — envelope turns
    // would have taken the tool branch instead.
    let executed = Arc::new(AtomicUsize::new(0));
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::Text("code: print(1)".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
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
    .action_backend(Box::new(FakeCodeBackend {
        executed: executed.clone(),
    }))
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "run the cell", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "run should complete: {:?}",
        result.err()
    );
    assert_eq!(
        executed.load(Ordering::SeqCst),
        1,
        "the batch ran exactly once"
    );

    let mut before = None;
    let mut after = None;
    {
        // Scoped so the std MutexGuard is dropped before the awaits below.
        let events = events.lock().unwrap();
        for e in events.iter() {
            if let RuntimeEvent::Checkpoint { checkpoint, .. } = e {
                match &checkpoint.step {
                    CheckpointStep::BeforeActions { actions } => {
                        assert_eq!(
                            checkpoint.backend,
                            ActionFormat::CodeAct,
                            "the line is pinned to the backend that parsed the batch"
                        );
                        before = Some(actions.len());
                    }
                    CheckpointStep::AfterActions {
                        actions,
                        observation,
                    } => {
                        assert_eq!(checkpoint.backend, ActionFormat::CodeAct);
                        after = Some((actions.len(), observation.clone()));
                    }
                    _ => {}
                }
            }
        }
    }
    let before_actions = before.expect("BeforeActions checkpoint emitted");
    let (after_actions, observation) = after.expect("AfterActions checkpoint emitted");
    assert_eq!(before_actions, 1, "the line records the batch (账)");
    assert_eq!(after_actions, 1, "the line records the batch (账)");
    let observation = observation.expect("the observation rides the line");
    assert_eq!(observation.outcomes.len(), 1, "one fate per action");
    assert!(
        observation.batch_status == BatchStatus::Completed,
        "the batch ran to fates: {:?}",
        observation.batch_status
    );

    // The backend's protocol pair lands in history verbatim.
    let session = runtime.session(&sid).await.expect("session exists");
    assert!(
        session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::Assistant { content: Some(c), .. } if c.contains("[cell]"))
        ),
        "the assistant half of the pair is pushed"
    );
    assert!(
        session.chat_messages().iter().any(
            |m| matches!(m, ChatMessage::User { content, .. } if content.contains("[observation]"))
        ),
        "the observation half of the pair is pushed"
    );
}
