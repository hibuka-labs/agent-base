//! The delivery contract for the terminal `RunCancelled`: it lands on the
//! event bus (external subscribers — the TUI bridge — see the cancellation)
//! AND loops back to the run's own `on_event` callback (the turn recorder).
//! The bus half used to be missing: the event was emitted cb-direct, like a
//! private note, so a mid-run Ctrl+C never reached the UI and the turn
//! settled as "done".

use agent_base::{AgentBuilder, AgentError, RunOutcome, RuntimeEvent};
use async_trait::async_trait;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// A provider whose stream never yields: the turn hangs in flight until the
/// test cancels it — the only deterministic way to exercise the cancel path.
struct HangingLlmProvider {
    /// Fires once the turn has reached the model (stream is being polled).
    started: tokio::sync::Notify,
}

#[async_trait]
impl agent_base::llm_trait::LlmProvider for HangingLlmProvider {
    async fn stream(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatStream, agent_base::llm_trait::LlmError> {
        // Signal the test that the turn reached the model, then hang forever.
        self.started.notify_one();
        std::future::pending::<()>().await;
        unreachable!()
    }

    async fn chat(
        &self,
        _request: agent_base::llm_trait::ChatRequest,
    ) -> Result<agent_base::llm_trait::ChatResponse, agent_base::llm_trait::LlmError> {
        unimplemented!()
    }

    fn capabilities(&self) -> agent_base::llm_trait::Capabilities {
        agent_base::llm_trait::Capabilities {
            supports_streaming: true,
            supports_tools: true,
            supports_vision: false,
            supports_thinking: false,
            max_context_tokens: Some(128_000),
            max_output_tokens: Some(16_384),
        }
    }

    fn info(&self) -> agent_base::llm_trait::ProviderInfo {
        agent_base::llm_trait::ProviderInfo {
            name: "hanging".to_string(),
            model: "test-model".to_string(),
            version: None,
        }
    }
}

#[tokio::test]
async fn a_cancel_reaches_both_the_bus_and_the_run_callback() {
    let provider = Arc::new(HangingLlmProvider {
        started: tokio::sync::Notify::new(),
    });
    let runtime = AgentBuilder::new(provider.clone())
        .system_prompt("test")
        .build()
        .unwrap();
    let session_id = runtime.create_session().await;

    // The TUI bridge subscribes here — this is the delivery path that was
    // missing for RunCancelled.
    let mut bus_rx = runtime.subscribe_runtime_events();

    let cb_saw_cancel = Arc::new(AtomicBool::new(false));
    let cb_flag = cb_saw_cancel.clone();
    let run = {
        let runtime = runtime.clone();
        let session_id = session_id.clone();
        tokio::spawn(async move {
            runtime
                .run_turn(session_id, "hello", move |ev| {
                    if matches!(ev, RuntimeEvent::RunCancelled { .. }) {
                        cb_flag.store(true, Ordering::SeqCst);
                    }
                    Ok(())
                })
                .await
        })
    };

    // Wait for the turn to reach the hanging model, then cancel it.
    tokio::time::timeout(Duration::from_secs(10), provider.started.notified())
        .await
        .expect("the turn must reach the model");
    runtime.cancel();

    let result = tokio::time::timeout(Duration::from_secs(10), run)
        .await
        .expect("a cancelled turn must return promptly")
        .unwrap();
    assert!(
        matches!(result, Err(AgentError::Cancelled)) || matches!(result, Ok(RunOutcome::Cancelled)),
        "the run must settle as a cancel"
    );

    // THE regression: the bus carries RunCancelled. Before the fix it was
    // emitted cb-direct only, so bus subscribers (the TUI) never saw it.
    let mut saw_bus_cancel = false;
    while let Ok(ev) = bus_rx.try_recv() {
        if matches!(ev, RuntimeEvent::RunCancelled { .. }) {
            saw_bus_cancel = true;
        }
    }
    assert!(saw_bus_cancel, "RunCancelled must land on the event bus");

    // The run's own recorder still gets its copy from the loopback.
    assert!(
        cb_saw_cancel.load(Ordering::SeqCst),
        "the on_event callback must still record RunCancelled"
    );
}
