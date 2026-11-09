use super::*;

// ---------------------------------------------------------------------------
// Notice pump (Batch E, PR2) — NoticeHandle + set_notice injection
// ---------------------------------------------------------------------------

/// Guard stub that sends one Notice on the first on_turn call, then behaves
/// like NoopGuard.
struct NoticeSendingGuard {
    handle: Mutex<Option<crate::engine::runtime::NoticeHandle>>,
}

impl NoticeSendingGuard {
    fn new() -> Self {
        Self {
            handle: Mutex::new(None),
        }
    }
}

#[async_trait]
impl crate::engine::ReactLoopGuard for NoticeSendingGuard {
    async fn on_turn(&self, ctx: &agent_types::GuardCtx) -> agent_types::GuardDecision {
        if let Some(h) = self.handle.lock().unwrap().as_ref() {
            h.send(
                crate::types::NoticeKind::Warning,
                "guard",
                "judge unparsed — treating as complete",
            );
        }
        if ctx.is_reasoning_only || ctx.is_empty_response {
            agent_types::GuardDecision::Fail {
                error: "degenerate".to_string(),
            }
        } else {
            agent_types::GuardDecision::Complete
        }
    }

    fn set_notice(&self, handle: crate::engine::runtime::NoticeHandle) {
        *self.handle.lock().unwrap() = Some(handle);
    }
}

#[tokio::test]
async fn notice_pump_stamps_and_delivers_guard_notice_before_run_finished() {
    use crate::types::{NoticeKind, UserEvent};

    let guard = Arc::new(NoticeSendingGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("All done.".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();

    let outcome = runtime
        .run_turn(sid.clone(), "do the thing", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run ok");

    assert!(matches!(outcome, RunOutcome::Completed));

    let delivered = events.lock().unwrap().clone();
    let position_of = |pred: &dyn Fn(&RuntimeEvent) -> bool| {
        delivered
            .iter()
            .position(pred)
            .expect("expected event present")
    };
    let notice_pos = position_of(&|e| {
        matches!(
            e,
            RuntimeEvent::UserEvent { event: UserEvent::Notice { kind, source, text }, .. }
                if *kind == NoticeKind::Warning
                    && source == "guard"
                    && text == "judge unparsed — treating as complete"
        )
    });
    let finished_pos = position_of(&|e| matches!(e, RuntimeEvent::RunFinished { .. }));
    let _ = position_of(&|e| matches!(e, RuntimeEvent::UserEvent { .. }));

    // Pump stamps session_id on the notice event.
    assert!(
        matches!(
            &delivered[notice_pos],
            RuntimeEvent::UserEvent { session_id, .. } if *session_id == sid
        ),
        "notice must be stamped with the run's session id"
    );
    // Ordering: notice arrives before the terminal RunFinished.
    assert!(
        notice_pos < finished_pos,
        "notice ({notice_pos}) must arrive before RunFinished ({finished_pos})"
    );
}

// ── Bus visibility / dual-write (Batch E follow-up) ─────────────────────────
// The pump dual-writes: notices and tool UserEvents also reach
// `subscribe_runtime_events()` consumers (bridge remote clients), while
// on_event sees each exactly once — the bus copy is a self-echo
// (`agent_id: None`) that bus loopback must skip.

#[tokio::test]
async fn notice_is_published_to_event_bus_for_external_subscribers() {
    use crate::types::{NoticeKind, UserEvent};

    let guard = Arc::new(NoticeSendingGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("All done.".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let mut bus_rx = runtime.subscribe_runtime_events();

    runtime
        .run_turn(sid.clone(), "do the thing", |_| Ok(()))
        .await
        .expect("run ok");

    let bus_events: Vec<RuntimeEvent> = std::iter::from_fn(|| bus_rx.try_recv().ok()).collect();
    let notices: Vec<&RuntimeEvent> = bus_events
        .iter()
        .filter(|e| {
            matches!(
                e,
                RuntimeEvent::UserEvent {
                    event: UserEvent::Notice { .. },
                    ..
                }
            )
        })
        .collect();
    assert_eq!(
        notices.len(),
        1,
        "exactly one notice on the event bus, got: {bus_events:?}"
    );
    assert!(
        matches!(
            notices[0],
            RuntimeEvent::UserEvent {
                session_id,
                event: UserEvent::Notice { kind, source, text },
                agent_id: None,
                ..
            } if *session_id == sid
                && *kind == NoticeKind::Warning
                && source == "guard"
                && text == "judge unparsed — treating as complete"
        ),
        "bus copy must be session-stamped with agent_id: None (self-echo marker): {:?}",
        notices[0]
    );
}

#[tokio::test]
async fn notice_reaches_on_event_exactly_once_despite_bus_loopback() {
    use crate::types::UserEvent;

    let guard = Arc::new(NoticeSendingGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("All done.".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    runtime
        .run_turn(sid.clone(), "do the thing", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run ok");

    let delivered = events.lock().unwrap().clone();
    let notice_count = delivered
        .iter()
        .filter(|e| {
            matches!(
                e,
                RuntimeEvent::UserEvent {
                    event: UserEvent::Notice { text, .. },
                    ..
                } if text == "judge unparsed — treating as complete"
            )
        })
        .count();
    assert_eq!(
        notice_count, 1,
        "dual-write must not double-render the notice through bus loopback: {delivered:?}"
    );
}

/// Tool stub that emits one progress UserEvent during execution.
struct ProgressEmittingTool;

#[async_trait]
impl Tool for ProgressEmittingTool {
    fn name(&self) -> &'static str {
        "progress_tool"
    }

    fn description(&self) -> &'static str {
        "Emits a progress user event"
    }

    fn schema(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        ctx.emit_progress("halfway");
        Ok(vec![Content::text("done")])
    }
}

#[tokio::test]
async fn tool_user_event_dual_write_reaches_bus_and_renders_once() {
    use crate::types::UserEvent;

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_prog_1",
                        "function": { "name": "progress_tool", "arguments": "{}" }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("All done.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .register_tool(ProgressEmittingTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let mut bus_rx = runtime.subscribe_runtime_events();
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    runtime
        .run_turn(sid.clone(), "run the tool", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run ok");

    let is_progress = |e: &RuntimeEvent| {
        matches!(
            e,
            RuntimeEvent::UserEvent {
                event: UserEvent::Progress { text },
                agent_id: None,
                ..
            } if text == "halfway"
        )
    };

    let delivered = events.lock().unwrap().clone();
    let rendered = delivered.iter().filter(|e| is_progress(e)).count();
    assert_eq!(
        rendered, 1,
        "tool UserEvent must render on_event exactly once: {delivered:?}"
    );

    let bus_events: Vec<RuntimeEvent> = std::iter::from_fn(|| bus_rx.try_recv().ok()).collect();
    let published = bus_events.iter().filter(|e| is_progress(e)).count();
    assert_eq!(
        published, 1,
        "tool UserEvent must appear on the event bus exactly once: {bus_events:?}"
    );
}

// ── Turn-boundary drain ────────────────────────────────────────────────────
// A notice sent at turn N's `on_turn` must reach the consumer at the turn
// boundary — before turn N+1 emits anything — not only at the terminal drain.

/// Sends one notice on its first `on_turn` and keeps the run alive
/// (`Continue`); completes on the second turn so the run ends cleanly.
struct TurnBoundaryGuard {
    handle: Mutex<Option<crate::engine::runtime::NoticeHandle>>,
    turns: Mutex<usize>,
}

impl TurnBoundaryGuard {
    fn new() -> Self {
        Self {
            handle: Mutex::new(None),
            turns: Mutex::new(0),
        }
    }
}

#[async_trait]
impl crate::engine::ReactLoopGuard for TurnBoundaryGuard {
    async fn on_turn(&self, _ctx: &agent_types::GuardCtx) -> agent_types::GuardDecision {
        let mut turns = self.turns.lock().unwrap();
        *turns += 1;
        if *turns == 1 {
            if let Some(h) = self.handle.lock().unwrap().as_ref() {
                h.send(
                    crate::types::NoticeKind::Warning,
                    "guard",
                    "turn one notice",
                );
            }
            agent_types::GuardDecision::Continue { nudge: None }
        } else {
            agent_types::GuardDecision::Complete
        }
    }

    fn set_notice(&self, handle: crate::engine::runtime::NoticeHandle) {
        *self.handle.lock().unwrap() = Some(handle);
    }
}

#[tokio::test]
async fn notice_sent_mid_run_is_delivered_at_turn_boundary_not_terminal_drain() {
    use crate::types::{NoticeKind, UserEvent};

    let guard = Arc::new(TurnBoundaryGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::Text("turn one.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("turn two.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    runtime
        .run_turn(sid.clone(), "start", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run ok");

    let delivered = events.lock().unwrap().clone();
    let notice_pos = delivered
        .iter()
        .position(|e| {
            matches!(
                e,
                RuntimeEvent::UserEvent {
                    event: UserEvent::Notice { text, .. },
                    ..
                } if text == "turn one notice"
            )
        })
        .expect("notice must be delivered");
    let turn_two_pos = delivered
        .iter()
        .position(
            |e| matches!(e, RuntimeEvent::TextDelta { text, .. } if text.contains("turn two")),
        )
        .expect("turn two must stream");

    // Turn-boundary semantics: the turn-1 notice lands before turn 2 emits.
    assert!(
        notice_pos < turn_two_pos,
        "turn-1 notice ({notice_pos}) must arrive before turn-2 events ({turn_two_pos})"
    );
    // ...and the pump stamps it with the run's session id.
    assert!(
        matches!(
            &delivered[notice_pos],
            RuntimeEvent::UserEvent { session_id, .. } if *session_id == sid
        ),
        "notice must be stamped with the run's session id"
    );
    // Belt and braces: the run really was multi-turn (the boundary exists).
    assert!(
        matches!(
            delivered
                .iter()
                .find(|e| matches!(e, RuntimeEvent::UserEvent { .. })),
            Some(RuntimeEvent::UserEvent {
                event: UserEvent::Notice {
                    kind: NoticeKind::Warning,
                    ..
                },
                ..
            })
        ),
        "the delivered user event is the warning notice"
    );
}

// ── Pump terminal-path coverage (review fixes) ────────────────────────────
// Every terminal event (RunFinished / RunCancelled, Ok or Err) must be
// preceded by a notice drain, and a notice sent outside a run must never be
// stamped with a later run's session id.

/// Guard that sends one notice per `on_turn` and always continues — keeps the
/// run alive so the test can terminate it via a non-Ok path with a notice
/// already queued.
struct NoticeContinueGuard {
    handle: Mutex<Option<crate::engine::runtime::NoticeHandle>>,
}

impl NoticeContinueGuard {
    fn new() -> Self {
        Self {
            handle: Mutex::new(None),
        }
    }
}

#[async_trait]
impl crate::engine::ReactLoopGuard for NoticeContinueGuard {
    async fn on_turn(&self, _ctx: &agent_types::GuardCtx) -> agent_types::GuardDecision {
        if let Some(h) = self.handle.lock().unwrap().as_ref() {
            h.send(crate::types::NoticeKind::Warning, "guard", "pending notice");
        }
        agent_types::GuardDecision::Continue { nudge: None }
    }

    fn set_notice(&self, handle: crate::engine::runtime::NoticeHandle) {
        *self.handle.lock().unwrap() = Some(handle);
    }
}

/// Guard that only stores the handle — the test sends notices manually at
/// precise moments (e.g. between runs). `on_turn` completes like NoopGuard.
struct HandleHoldingGuard {
    handle: Mutex<Option<crate::engine::runtime::NoticeHandle>>,
}

impl HandleHoldingGuard {
    fn new() -> Self {
        Self {
            handle: Mutex::new(None),
        }
    }

    fn send(&self, kind: crate::types::NoticeKind, source: &'static str, text: &str) {
        if let Some(h) = self.handle.lock().unwrap().as_ref() {
            h.send(kind, source, text.to_string());
        }
    }
}

#[async_trait]
impl crate::engine::ReactLoopGuard for HandleHoldingGuard {
    async fn on_turn(&self, _ctx: &agent_types::GuardCtx) -> agent_types::GuardDecision {
        agent_types::GuardDecision::Complete
    }

    fn set_notice(&self, handle: crate::engine::runtime::NoticeHandle) {
        *self.handle.lock().unwrap() = Some(handle);
    }
}

/// First `stream()` call returns a text response; later calls fail — drives
/// the non-cancelled error termination with a notice already queued.
struct TextThenFailProvider {
    calls: std::sync::atomic::AtomicUsize,
}

#[async_trait]
impl LlmProvider for TextThenFailProvider {
    async fn stream(
        &self,
        _request: llm_trait::ChatRequest,
    ) -> Result<llm_trait::ChatStream, llm_trait::LlmError> {
        let first = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) == 0;
        let chunks: Vec<Result<StreamChunk, llm_trait::LlmError>> = if first {
            vec![
                Ok(StreamChunk::Text("let me think...".to_string())),
                Ok(StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                }),
            ]
        } else {
            vec![Err(llm_trait::LlmError::llm("simulated LLM error"))]
        };
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
            name: "text-then-fail".to_string(),
            model: "test".to_string(),
            version: None,
        }
    }
}

fn notice_pos_of(delivered: &[RuntimeEvent]) -> Option<usize> {
    delivered.iter().position(|e| {
        matches!(
            e,
            RuntimeEvent::UserEvent {
                event: crate::types::UserEvent::Notice { .. },
                ..
            }
        )
    })
}

/// A notice queued during the run must still be delivered — before the
/// terminal event — when the run ends via `RunCancelled`.
#[tokio::test]
async fn notice_pump_delivers_pending_notice_before_run_cancelled() {
    use crate::types::RunOutcome;

    let guard = Arc::new(NoticeContinueGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: text-only → guard sends a notice, decides Continue.
        vec![
            StreamChunk::Text("let me think...".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        // Turn 2: tool call that cancels the run → RunCancelled path.
        vec![
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
        ],
    ])))
    .guard_dyn(guard.clone())
    .register_tool(CancelTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "go", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Cancelled)),
        "precondition: run must end cancelled, got: {:?}",
        result
    );

    let delivered = events.lock().unwrap().clone();
    let notice_pos = notice_pos_of(&delivered)
        .expect("pending notice must be delivered on the RunCancelled path");
    let cancelled_pos = delivered
        .iter()
        .position(|e| matches!(e, RuntimeEvent::RunCancelled { .. }))
        .expect("RunCancelled must be emitted");
    assert!(
        matches!(
            &delivered[notice_pos],
            RuntimeEvent::UserEvent { session_id, .. } if *session_id == sid
        ),
        "notice must be stamped with the run's session id"
    );
    assert!(
        notice_pos < cancelled_pos,
        "notice ({notice_pos}) must arrive before RunCancelled ({cancelled_pos})"
    );
}

/// A notice queued during the run must still be delivered — before
/// RunFinished — when the run terminates with a non-cancelled error.
#[tokio::test]
async fn notice_pump_delivers_pending_notice_on_error_termination() {
    let guard = Arc::new(NoticeContinueGuard::new());
    let runtime = AgentBuilder::new(Arc::new(TextThenFailProvider {
        calls: std::sync::atomic::AtomicUsize::new(0),
    }))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "go", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_err(),
        "precondition: run must end with an error, got: {:?}",
        result
    );

    let delivered = events.lock().unwrap().clone();
    let notice_pos = notice_pos_of(&delivered)
        .expect("pending notice must be delivered on the error termination path");
    let finished_pos = delivered
        .iter()
        .position(|e| matches!(e, RuntimeEvent::RunFinished { .. }))
        .expect("error path must still emit RunFinished");
    assert!(
        matches!(
            &delivered[notice_pos],
            RuntimeEvent::UserEvent { session_id, .. } if *session_id == sid
        ),
        "notice must be stamped with the run's session id"
    );
    assert!(
        notice_pos < finished_pos,
        "notice ({notice_pos}) must arrive before RunFinished ({finished_pos})"
    );
}

/// A notice sent outside any run must be discarded at the next run start —
/// never delivered stamped with a later run's session id.
#[tokio::test]
async fn notice_sent_between_runs_is_discarded_not_mis_stamped() {
    use crate::types::NoticeKind;

    let guard = Arc::new(HandleHoldingGuard::new());
    let script = || {
        vec![vec![
            StreamChunk::Text("done.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ]]
    };
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(script())))
        .guard_dyn(guard.clone())
        .build()
        .expect("build runtime");

    let sid1 = runtime.create_session().await;
    let events1: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events1_clone = events1.clone();
    runtime
        .run_turn(sid1, "hi", move |event| {
            events1_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run 1 ok");
    let run1 = events1.lock().unwrap().clone();
    assert!(
        notice_pos_of(&run1).is_none(),
        "precondition: run 1 delivers no notice"
    );

    // Between runs: the guard (still holding the engine handle) sends.
    guard.send(NoticeKind::Warning, "guard", "leftover from before");

    let sid2 = runtime.create_session().await;
    let events2: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events2_clone = events2.clone();
    runtime
        .run_turn(sid2, "hi", move |event| {
            events2_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await
        .expect("run 2 ok");
    let run2 = events2.lock().unwrap().clone();
    assert!(
        notice_pos_of(&run2).is_none(),
        "between-run notice must be discarded at run start, not stamped onto run 2: {:?}",
        run2
    );
}

/// Notice delivery is display-only: a consumer callback error on a notice
/// must not fail the run nor skip the terminal event.
#[tokio::test]
async fn notice_callback_error_does_not_fail_run_or_skip_terminal() {
    use crate::types::UserEvent;

    let guard = Arc::new(NoticeSendingGuard::new());
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("All done.".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .guard_dyn(guard.clone())
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events: Arc<Mutex<Vec<RuntimeEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "go", move |event| {
            let is_notice = matches!(
                &event,
                RuntimeEvent::UserEvent {
                    event: UserEvent::Notice { .. },
                    ..
                }
            );
            events_clone.lock().unwrap().push(event);
            if is_notice {
                Err(crate::types::AgentError::internal(
                    "consumer rejected notice",
                ))
            } else {
                Ok(())
            }
        })
        .await;

    assert!(
        result.is_ok(),
        "display-only notice errors must not fail the run, got: {:?}",
        result
    );
    let delivered = events.lock().unwrap().clone();
    assert!(
        delivered
            .iter()
            .any(|e| matches!(e, RuntimeEvent::RunFinished { .. })),
        "terminal RunFinished must still be delivered after a notice callback error"
    );
}
