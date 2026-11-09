use super::*;

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
async fn process_approval_without_policy_is_noop() {
    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    // No policy → auto-approve → returns Ok even with a deny-all handler.
    let engine = ToolEngine::new(
        ToolRegistry::default(),
        Some(Arc::new(DenyAllApprovalHandler)),
        None,
        Arc::new(StopOnError),
        event_bus,
    );
    let sm = session_manager();
    let c = ctx(&sm);

    engine
        .process_approval(
            &SessionId::new(1),
            "echo",
            &Value::Null,
            "{}",
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("no policy should skip approval");
}

#[tokio::test]
async fn process_approval_denies_when_handler_denies() {
    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(
        ToolRegistry::default(),
        Some(Arc::new(DenyAllApprovalHandler)),
        Some(Arc::new(RequireApproval)),
        Arc::new(StopOnError),
        event_bus,
    );
    let sm = session_manager();
    let c = ctx(&sm);
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();

    let err = engine
        .process_approval(
            &SessionId::new(1),
            "echo",
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
        .unwrap_err();

    let events = events.lock().unwrap();

    assert!(matches!(err, AgentError::ApprovalDenied { .. }));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::AwaitingApproval { .. })),
        "should emit AwaitingApproval before denial"
    );
    assert!(
        events
            .iter()
            .any(|e| matches!(e, RuntimeEvent::ActionFinished { denied: true, .. })),
        "should emit a denied ActionFinished"
    );
}

#[tokio::test]
async fn process_approval_allows_when_handler_allows() {
    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(
        ToolRegistry::default(),
        Some(Arc::new(AllowAllApprovalHandler)),
        Some(Arc::new(RequireApproval)),
        Arc::new(StopOnError),
        event_bus,
    );
    let sm = session_manager();
    let c = ctx(&sm);

    engine
        .process_approval(
            &SessionId::new(1),
            "echo",
            &Value::Null,
            "{}",
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("allow-all handler should grant approval");
}

#[tokio::test]
async fn process_approval_skips_when_cached() {
    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(
        ToolRegistry::default(),
        Some(Arc::new(DenyAllApprovalHandler)),
        Some(Arc::new(RequireApproval)),
        Arc::new(StopOnError),
        event_bus,
    );
    let sm = session_manager();
    let c = ctx(&sm);
    let sid = sm.create_session(None).await;

    // Pre-cache the approval with the same action_key the policy emits.
    sm.cache_approval(&sid, "approve:echo".to_string()).await;

    // Cached approval short-circuits even though the handler would deny.
    engine
        .process_approval(
            &sid,
            "echo",
            &Value::Null,
            "{}",
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .expect("cached approval should skip");
}

#[tokio::test]
async fn process_approval_denies_when_no_handler() {
    let event_bus = EventBus::new(16);
    let mut event_rx = event_bus.subscribe();
    let engine = ToolEngine::new(
        ToolRegistry::default(),
        None, // no approval handler
        Some(Arc::new(RequireApproval)),
        Arc::new(StopOnError),
        event_bus,
    );
    let sm = session_manager();
    let c = ctx(&sm);

    let err = engine
        .process_approval(
            &SessionId::new(1),
            "echo",
            &Value::Null,
            "{}",
            &c,
            &mut event_rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_| -> AgentResult<()> { Ok(()) })),
        )
        .await
        .unwrap_err();

    assert!(matches!(err, AgentError::ApprovalDenied { .. }));
}
