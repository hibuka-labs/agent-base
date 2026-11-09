use super::*;

#[tokio::test]
async fn execute_tool_returns_not_found_for_unknown_tool() {
    // Empty registry + minimal runtime plumbing so the tool-not-found
    // branch is the only code path exercised.
    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(
        ToolRegistry::default(),
        None,
        None,
        Arc::new(StopOnError),
        event_bus,
    );

    let session_manager = SessionManager::new(
        Arc::new(AtomicU64SessionIdGenerator::default()),
        Arc::new(InMemorySessionStore::new()),
        SessionConfig::default(),
    );

    for (language, expected) in [
        (Language::En, "Tool no_such_tool not found"),
        (Language::Zh, "工具 no_such_tool 未找到"),
    ] {
        let ctx = ExecutionContext {
            session_manager: session_manager.clone(),
            llm_client: None,
            language,
            tool_timeout_ms: None,
            max_output_chars: None,
            cancel_token: CancellationToken::new(),
        };

        let session_id = SessionId::new(1);
        let result = engine
            .execute_tool(
                &session_id,
                "call_1",
                "no_such_tool",
                &Value::Null,
                "{}",
                &ctx,
                &mut event_rx,
                std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
            )
            .await
            .expect("unknown tool should not error");

        assert_eq!(result.id, "call_1");
        assert_eq!(result.name, "no_such_tool");
        assert_eq!(content_text(&result.output), expected);
    }
}

/// Registers `EchoTool` into the shared registry from inside `call` —
/// mirrors phimint's `add_tool`, which mounts a new tool while executing.
/// With the registry read lock held across the call this write deadlocked.
struct SelfRegisterTool {
    registry: Arc<RwLock<ToolRegistry>>,
}

#[async_trait]
impl Tool for SelfRegisterTool {
    fn name(&self) -> &'static str {
        "self_register"
    }

    fn description(&self) -> &'static str {
        "registers another tool while running"
    }

    fn schema(&self) -> Value {
        serde_json::json!({})
    }

    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        // Must complete promptly — the regression is a deadlock, so the
        // write itself gets a watchdog too (clear failure, not a hung test).
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            self.registry.write().await.register(EchoTool);
        })
        .await
        .map_err(|_| AgentError::internal("registry.write() deadlocked"))?;
        Ok(vec![Content::text("registered")])
    }
}

/// Self-registering tools must run to completion on the serial path and
/// the new tool must show up in the schema without a rebuild/restart.
#[tokio::test]
async fn self_registering_tool_does_not_deadlock_and_hot_refreshes() {
    let engine = ToolEngine::new(
        ToolRegistry::default(),
        None,
        None,
        Arc::new(StopOnError),
        EventBus::new(16),
    );
    let handle = engine.tools_handle();
    handle.write().await.register(SelfRegisterTool {
        registry: handle.clone(),
    });

    let mut event_rx = engine.event_bus.subscribe();
    let sm = session_manager();
    let c = ctx(&sm);
    let sid = SessionId::new(1);

    let call = engine.execute_tool(
        &sid,
        "call_1",
        "self_register",
        &Value::Null,
        "{}",
        &c,
        &mut event_rx,
        std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
    );
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), call)
        .await
        .expect("self-registering tool must not deadlock on registry.write()");
    let result = result.expect("self_register should execute");
    assert_eq!(content_text(&result.output), "registered");

    // Hot refresh: the mounted tool is in the schema for the next turn.
    let defs = engine.definitions().await;
    let names: Vec<&str> = defs
        .iter()
        .filter_map(|d| d["function"]["name"].as_str())
        .collect();
    assert!(
        names.contains(&"echo"),
        "registered tool missing: {names:?}"
    );
}

/// Same on the parallel orchestration path (`execute_tool_static`).
#[tokio::test]
async fn self_registering_tool_does_not_deadlock_on_the_parallel_path() {
    let engine = ToolEngine::new(
        ToolRegistry::default(),
        None,
        None,
        Arc::new(StopOnError),
        EventBus::new(16),
    );
    let handle = engine.tools_handle();
    handle.write().await.register(SelfRegisterTool {
        registry: handle.clone(),
    });

    let mut event_rx = engine.event_bus.subscribe();
    let pipeline = DefaultPipeline::new(None, None, None);
    let error_recovery: Arc<dyn ToolErrorRecovery> = Arc::new(StopOnError);
    let sm = session_manager();
    let c = ctx(&sm);
    let sid = SessionId::new(1);

    let call = ToolEngine::execute_tool_static(
        &handle,
        &pipeline,
        &error_recovery,
        &engine.event_bus,
        &sid,
        "call_1",
        "self_register",
        &Value::Null,
        "{}",
        &c,
        &mut event_rx,
        std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
    );
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), call)
        .await
        .expect("parallel path must not deadlock on registry.write()");
    let result = result.expect("self_register should execute");
    assert_eq!(content_text(&result.output), "registered");
}

#[tokio::test]
async fn execute_tool_found_runs_and_emits_events() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);

    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = engine
        .execute_tool(
            &SessionId::new(1),
            "call_1",
            "echo",
            &serde_json::json!({"text": "hi"}),
            r#"{"text":"hi"}"#,
            &c,
            &mut event_rx,
            Arc::new(Mutex::new(move |e| -> AgentResult<()> {
                events_clone.lock().unwrap().push(e);
                Ok(())
            })),
        )
        .await
        .expect("echo tool should execute");

    let events = events.lock().unwrap();

    assert_eq!(result.id, "call_1");
    assert_eq!(result.name, "echo");
    assert_eq!(content_text(&result.output), "echo: hi");

    let started = events.iter().find_map(|e| match e {
        RuntimeEvent::ActionStarted {
            tool_name,
            args_json,
            ..
        } => Some((tool_name.as_str(), args_json.as_str())),
        _ => None,
    });
    assert_eq!(started, Some(("echo", r#"{"text":"hi"}"#)));

    assert!(events.iter().any(
        |e| matches!(e, RuntimeEvent::ActionFinished { summary, .. } if summary == "echo: hi")
    ));
}

#[tokio::test]
async fn definitions_returns_registered_tools() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);
    let engine = ToolEngine::new(
        registry,
        None,
        None,
        Arc::new(StopOnError),
        EventBus::new(4),
    );

    let defs = engine.definitions().await;
    assert_eq!(defs.len(), 1); // echo
    assert_eq!(defs[0]["function"]["name"], "echo");
}

#[tokio::test]
async fn getters_expose_engine_state() {
    let mut registry = ToolRegistry::default();
    registry.register(EchoTool);

    let approval = Arc::new(AllowAllApprovalHandler);
    let recovery = Arc::new(StopOnError);
    let event_bus = EventBus::new(4);
    let engine = ToolEngine::new(
        registry,
        Some(approval.clone()),
        None,
        recovery.clone(),
        event_bus,
    );

    assert!(engine.approval_handler().is_some());
    assert!(Arc::ptr_eq(
        engine.error_recovery(),
        &(recovery.clone() as Arc<dyn ToolErrorRecovery>)
    ));
    assert_eq!(engine.tools_arc().read().await.len(), 1);

    let defs = engine.definitions().await;
    assert_eq!(defs.len(), 1); // echo
    assert_eq!(defs[0]["function"]["name"], "echo");
}

// ── B5: tool-failure / user-event forwarding / approval edges ────────

struct ProgressTool;
#[async_trait]
impl Tool for ProgressTool {
    fn name(&self) -> &'static str {
        "progress"
    }
    fn description(&self) -> &'static str {
        ""
    }
    fn schema(&self) -> Value {
        serde_json::json!({})
    }
    async fn call(&self, _args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        ctx.emit_progress("working");
        Ok(vec![Content::text("done")])
    }
}

#[tokio::test]
async fn execute_tool_returns_tool_execution_error_on_failure() {
    let mut registry = ToolRegistry::default();
    registry.register(FailingTool);

    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();

    for language in [Language::En, Language::Zh] {
        let c = ExecutionContext {
            session_manager: sm.clone(),
            llm_client: None,
            language,
            tool_timeout_ms: None,
            max_output_chars: None,
            cancel_token: CancellationToken::new(),
        };
        let events = Arc::new(Mutex::new(Vec::new()));
        let events_clone = events.clone();
        let err = engine
            .execute_tool(
                &SessionId::new(1),
                "call_1",
                "failing",
                &Value::Null,
                "{}",
                &c,
                &mut event_rx,
                Arc::new(Mutex::new(move |e| -> AgentResult<()> {
                    events_clone.lock().unwrap().push(e);
                    Ok(())
                })),
            )
            .await
            .expect_err("failing tool should error");

        let events = events.lock().unwrap();

        assert!(
            matches!(err, AgentError::ToolExecution { .. }),
            "expected ToolExecution, got {err:?}"
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, RuntimeEvent::ActionFinished { summary, .. }
                    if summary.contains("failed") || summary.contains("失败"))),
            "should emit error ActionFinished"
        );
    }
}

#[tokio::test]
async fn execute_tool_forwards_user_events() {
    let mut registry = ToolRegistry::default();
    registry.register(ProgressTool);

    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(registry, None, None, Arc::new(StopOnError), event_bus);

    let sm = session_manager();
    let c = ctx(&sm);

    let forwarded = Arc::new(Mutex::new(Vec::new()));
    let forwarded_clone = forwarded.clone();
    let result = engine
        .execute_tool(
            &SessionId::new(1),
            "call_1",
            "progress",
            &Value::Null,
            "{}",
            &c,
            &mut event_rx,
            Arc::new(Mutex::new(move |e| -> AgentResult<()> {
                forwarded_clone.lock().unwrap().push(e);
                Ok(())
            })),
        )
        .await
        .expect("progress tool should execute");

    let forwarded = forwarded.lock().unwrap();

    assert_eq!(content_text(&result.output), "done");
    assert!(
        forwarded.iter().any(|e| matches!(
            e,
            RuntimeEvent::UserEvent {
                event: UserEvent::Progress { .. },
                ..
            }
        )),
        "should forward progress as a UserEvent"
    );
}
