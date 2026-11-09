use super::*;

#[tokio::test]
async fn orchestrate_executes_multiple_tool_calls() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);

    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    let outcome = engine
        .orchestrate(
            &SessionId::new(1),
            &[
                (
                    "call_a".to_string(),
                    "echo".to_string(),
                    r#"{"text":"a"}"#.to_string(),
                ),
                (
                    "call_b".to_string(),
                    "echo".to_string(),
                    r#"{"text":"b"}"#.to_string(),
                ),
            ],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("orchestrate should succeed");

    assert_eq!(outcome.results.len(), 2);
    assert!(outcome.failures.is_empty());
    assert_eq!(outcome.results[0].id, "call_a");
    assert_eq!(content_text(&outcome.results[0].output), "echo: a");
    assert_eq!(outcome.results[1].id, "call_b");
    assert_eq!(content_text(&outcome.results[1].output), "echo: b");
}

#[tokio::test]
async fn orchestrate_collects_invalid_json_args_as_failure() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);

    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    let outcome = engine
        .orchestrate(
            &SessionId::new(1),
            &[(
                "call_x".to_string(),
                "echo".to_string(),
                "not-json".to_string(),
            )],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("orchestrate should not hard-fail on bad args");

    assert!(
        outcome.results.is_empty(),
        "no tool should execute for bad args"
    );
    assert_eq!(outcome.failures.len(), 1);
    assert!(matches!(
        &outcome.failures[0].error,
        AgentError::ToolArgsInvalid { .. }
    ));
}

#[tokio::test]
async fn orchestrate_continues_past_bad_args_call() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);

    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    // [good, bad-args, good] — the bad call in the middle must not stop the
    // trailing good call from executing.
    let outcome = engine
        .orchestrate(
            &SessionId::new(1),
            &[
                (
                    "call_a".to_string(),
                    "echo".to_string(),
                    r#"{"text":"a"}"#.to_string(),
                ),
                (
                    "call_b".to_string(),
                    "echo".to_string(),
                    "not-json".to_string(),
                ),
                (
                    "call_c".to_string(),
                    "echo".to_string(),
                    r#"{"text":"c"}"#.to_string(),
                ),
            ],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("orchestrate should succeed even with a bad call in the middle");

    assert_eq!(outcome.results.len(), 2, "both good calls should execute");
    assert_eq!(outcome.results[0].id, "call_a");
    assert_eq!(content_text(&outcome.results[0].output), "echo: a");
    assert_eq!(outcome.results[1].id, "call_c");
    assert_eq!(content_text(&outcome.results[1].output), "echo: c");

    assert_eq!(outcome.failures.len(), 1, "only the bad call should fail");
    assert_eq!(outcome.failures[0].id, "call_b");
    assert!(matches!(
        &outcome.failures[0].error,
        AgentError::ToolArgsInvalid { .. }
    ));
}

#[tokio::test]
async fn orchestrate_continues_past_execution_failure() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);
    registry.register(FailingTool);

    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    let outcome = engine
        .orchestrate(
            &SessionId::new(1),
            &[
                (
                    "call_a".to_string(),
                    "echo".to_string(),
                    r#"{"text":"a"}"#.to_string(),
                ),
                (
                    "call_fail".to_string(),
                    "failing".to_string(),
                    "{}".to_string(),
                ),
                (
                    "call_b".to_string(),
                    "echo".to_string(),
                    r#"{"text":"b"}"#.to_string(),
                ),
            ],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("orchestrate should succeed even with a tool execution failure");

    assert_eq!(outcome.results.len(), 2);
    assert_eq!(outcome.results[0].id, "call_a");
    assert_eq!(outcome.results[1].id, "call_b");
    assert_eq!(outcome.failures.len(), 1);
    assert!(matches!(
        &outcome.failures[0].error,
        AgentError::ToolExecution { .. }
    ));
}

#[tokio::test]
async fn orchestrate_delivers_events_exactly_once() {
    // Regression: parallel orchestrate must not double-deliver events.
    // Previously, events were delivered via local_rx (resubscribe) AND
    // again via the original event_rx drain, causing duplicates.
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);

    let event_bus = EventBus::new(64);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();

    let outcome = engine
        .orchestrate(
            &SessionId::new(1),
            &[
                (
                    "call_a".to_string(),
                    "echo".to_string(),
                    r#"{"text":"a"}"#.to_string(),
                ),
                (
                    "call_b".to_string(),
                    "echo".to_string(),
                    r#"{"text":"b"}"#.to_string(),
                ),
            ],
            &c,
            &mut event_rx,
            Arc::new(Mutex::new(move |ev| -> AgentResult<()> {
                let tag = match &ev {
                    RuntimeEvent::ActionStarted { tool_name, .. } => {
                        format!("started:{tool_name}")
                    }
                    RuntimeEvent::ActionFinished { tool_name, .. } => {
                        format!("finished:{tool_name}")
                    }
                    _ => return Ok(()),
                };
                events_clone.lock().unwrap().push(tag);
                Ok(())
            })),
        )
        .await
        .expect("orchestrate should succeed");

    assert_eq!(outcome.results.len(), 2);

    let ev = events.lock().unwrap();
    // Each tool should produce exactly one started + one finished = 4 total.
    let started: Vec<_> = ev.iter().filter(|e| e.starts_with("started:")).collect();
    let finished: Vec<_> = ev.iter().filter(|e| e.starts_with("finished:")).collect();
    assert_eq!(
        started.len(),
        2,
        "expected 2 ActionStarted, got {started:?}"
    );
    assert_eq!(
        finished.len(),
        2,
        "expected 2 ActionFinished, got {finished:?}"
    );
}

#[tokio::test]
async fn orchestrate_invalid_args_error_includes_serde_details() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);

    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    // Malformed JSON — serde should tell us what went wrong
    let outcome = engine
        .orchestrate(
            &SessionId::new(1),
            &[(
                "call_x".to_string(),
                "echo".to_string(),
                "{invalid".to_string(),
            )],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("orchestrate should not hard-fail on bad args");

    assert_eq!(outcome.failures.len(), 1);
    match &outcome.failures[0].error {
        AgentError::ToolArgsInvalid { raw, .. } => {
            // The raw field should now contain the serde error message,
            // not just the original args string
            assert!(
                raw.contains("args: {invalid"),
                "should include original args, got: {}",
                raw
            );
            // Serde error should be present (e.g., "key must be a string")
            assert!(
                raw.len() > "{invalid".to_string().len() + 10,
                "raw should contain serde error in addition to args, got: {}",
                raw
            );
        }
        other => panic!("expected ToolArgsInvalid, got {:?}", other),
    }
}

#[tokio::test]
async fn orchestrate_non_json_args_error_includes_parse_details() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);

    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    let outcome = engine
        .orchestrate(
            &SessionId::new(1),
            &[(
                "call_x".to_string(),
                "echo".to_string(),
                "not-valid-json".to_string(),
            )],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("orchestrate should not hard-fail");

    assert_eq!(outcome.failures.len(), 1);
    match &outcome.failures[0].error {
        AgentError::ToolArgsInvalid { raw, .. } => {
            // Should contain the serde error description (e.g., "expected value at line 1 column 1")
            assert!(
                raw.contains("args: not-valid-json"),
                "should include original args, got: {}",
                raw
            );
            assert!(
                raw.len() > "not-valid-json".to_string().len(),
                "raw should contain serde error in addition to args, got: {}",
                raw
            );
        }
        other => panic!("expected ToolArgsInvalid, got {:?}", other),
    }
}

// ── Parallel orchestrate tests ──

struct SlowEchoTool;
#[async_trait]
impl Tool for SlowEchoTool {
    fn name(&self) -> &'static str {
        "slow_echo"
    }
    fn description(&self) -> &'static str {
        ""
    }
    fn schema(&self) -> Value {
        serde_json::json!({})
    }
    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        // Simulate a slow tool (50ms)
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let text = args.get("text").and_then(Value::as_str).unwrap_or("");
        Ok(vec![Content::text(format!("slow: {text}"))])
    }
}

#[tokio::test]
async fn orchestrate_executes_in_parallel() {
    let mut registry = ToolRegistry::default();
    registry.register(SlowEchoTool);

    let event_bus = EventBus::new(64);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    let start = std::time::Instant::now();
    let outcome = engine
        .orchestrate(
            &SessionId::new(1),
            &[
                ("c1".into(), "slow_echo".into(), r#"{"text":"a"}"#.into()),
                ("c2".into(), "slow_echo".into(), r#"{"text":"b"}"#.into()),
                ("c3".into(), "slow_echo".into(), r#"{"text":"c"}"#.into()),
            ],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("orchestrate should succeed");

    let elapsed = start.elapsed();

    assert_eq!(outcome.results.len(), 3);
    assert!(outcome.failures.is_empty());

    // If sequential, 3 × 50ms = 150ms+. With parallel, should be ~50ms.
    // Use 120ms as a generous threshold (sequential would be 150ms+).
    assert!(
        elapsed < std::time::Duration::from_millis(120),
        "expected parallel execution < 120ms, took {elapsed:?}"
    );
}

#[tokio::test]
async fn orchestrate_parallel_failure_does_not_abort_others() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);
    registry.register(FailingTool);

    let event_bus = EventBus::new(64);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    let outcome = engine
        .orchestrate(
            &SessionId::new(1),
            &[
                ("c1".into(), "echo".into(), r#"{"text":"ok"}"#.into()),
                ("c2".into(), "failing".into(), "{}".into()),
                ("c3".into(), "echo".into(), r#"{"text":"also ok"}"#.into()),
            ],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("orchestrate should succeed even with a failure");

    assert_eq!(outcome.results.len(), 2);
    assert_eq!(content_text(&outcome.results[0].output), "echo: ok");
    assert_eq!(content_text(&outcome.results[1].output), "echo: also ok");

    assert_eq!(outcome.failures.len(), 1);
    assert!(matches!(
        &outcome.failures[0].error,
        AgentError::ToolExecution { .. }
    ));
}

/// Tool that does synchronous file I/O — mimics history_list_items.
/// This exercises the real-world path where a blocking `fs::read_to_string`
/// runs inside an async `call` on a multi-thread tokio runtime.
struct SyncFileReadTool {
    dir: tempfile::TempDir,
}

impl SyncFileReadTool {
    fn new() -> Self {
        let dir = tempfile::TempDir::new().unwrap();
        // Write a few small files to simulate session history
        for i in 0..5 {
            let path = dir.path().join(format!("file_{i}.jsonl"));
            let content = format!("line1 for file {i}\nline2 for file {i}\n");
            std::fs::write(&path, content).unwrap();
        }
        Self { dir }
    }
}

#[async_trait]
impl Tool for SyncFileReadTool {
    fn name(&self) -> &'static str {
        "sync_file_read"
    }
    fn description(&self) -> &'static str {
        ""
    }
    fn schema(&self) -> Value {
        serde_json::json!({})
    }
    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        // Synchronous file I/O — same pattern as history_list_items
        let mut output = String::new();
        for i in 0..5 {
            let path = self.dir.path().join(format!("file_{i}.jsonl"));
            let content = std::fs::read_to_string(&path)
                .map_err(|e| AgentError::internal(format!("sync_file_read: {e}")))?;
            output.push_str(&content);
        }
        Ok(vec![Content::text(output)])
    }
}

#[tokio::test]
async fn orchestrate_sync_io_tool_does_not_hang() {
    let tool = SyncFileReadTool::new();
    let mut registry = ToolRegistry::default();
    registry.register(tool);

    let event_bus = EventBus::new(64);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);
    let sm = session_manager();
    let c = ctx(&sm);

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        engine.orchestrate(
            &SessionId::new(1),
            &[("c1".into(), "sync_file_read".into(), "{}".into())],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        ),
    )
    .await;

    assert!(result.is_ok(), "sync I/O tool timed out (hang!)");
    let outcome = result.unwrap().unwrap();
    assert_eq!(outcome.results.len(), 1);
    assert!(outcome.failures.is_empty());
}

#[tokio::test]
async fn orchestrate_parallel_sync_io_tools_do_not_hang() {
    let tool = SyncFileReadTool::new();
    let mut registry = ToolRegistry::default();
    registry.register(tool);
    registry.register(EchoTool);

    let event_bus = EventBus::new(64);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);
    let sm = session_manager();
    let c = ctx(&sm);

    // Mix sync I/O tools with fast tools in parallel
    let result = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        engine.orchestrate(
            &SessionId::new(1),
            &[
                ("c1".into(), "sync_file_read".into(), "{}".into()),
                ("c2".into(), "echo".into(), r#"{"text":"hello"}"#.into()),
                ("c3".into(), "sync_file_read".into(), "{}".into()),
                ("c4".into(), "echo".into(), r#"{"text":"world"}"#.into()),
            ],
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        ),
    )
    .await;

    assert!(result.is_ok(), "parallel sync I/O tools timed out (hang!)");
    let outcome = result.unwrap().unwrap();
    assert_eq!(outcome.results.len(), 4, "expected 4 results");
    assert!(
        outcome.failures.is_empty(),
        "failures: {:?}",
        outcome.failures
    );
}
