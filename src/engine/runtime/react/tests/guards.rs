use super::*;

// ---------------------------------------------------------------------------
// Phase 1: FinishReason branch tests
// ---------------------------------------------------------------------------

/// Model returns `tool_use` finish reason but no actual tool call data.
/// The react loop should return `RunOutcome::Failed`.
#[tokio::test]
async fn tool_use_with_no_tool_calls_returns_failed() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        // A ToolCall chunk with no delta.tool_calls — sets is_tool_call=true
        // but leaves tool_calls empty.
        StreamChunk::ToolCall(serde_json::json!({ "no_delta": true })),
        StreamChunk::Stop {
            finish_reason: Some("tool_use".to_string()),
        },
    ]])))
    .system_prompt("test")
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let result = runtime.run_turn(sid, "do something", |_| Ok(())).await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "tool_use with no tool calls should return Failed: {:?}",
        result
    );
}

/// Model returns `max_tokens` (truncated) with text-only output (no tool calls).
/// The react loop should return `RunOutcome::Failed`.
#[tokio::test]
async fn truncated_text_only_returns_failed() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("Here is part of my answer before being cut".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("max_tokens".to_string()),
        },
    ]])))
    .system_prompt("test")
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let result = runtime.run_turn(sid, "do something", |_| Ok(())).await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "truncated text-only should return Failed: {:?}",
        result
    );
}

/// Model returns `length` (OpenAI truncation) with text-only output.
/// The react loop should return `RunOutcome::Failed`.
#[tokio::test]
async fn openai_length_text_only_returns_failed() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        StreamChunk::Text("Partial response before length limit".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("length".to_string()),
        },
    ]])))
    .system_prompt("test")
    .build()
    .expect("build runtime");
    let sid = runtime.create_session().await;

    let result = runtime.run_turn(sid, "do something", |_| Ok(())).await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "OpenAI length with text-only should return Failed: {:?}",
        result
    );
}

/// Model emits a ToolCall chunk (setting `is_tool_call = true`) but the stream
/// ends with `finish_reason = "stop"` and zero parsed tool calls (no text either).
///
/// This mirrors the real-world scenario where the model starts generating a tool
/// call but the stream produces an incomplete/empty tool call delta.  With
/// `finish_reason = "stop"` (not `"tool_use"`), the anomaly-detection branch at
/// line 582 does NOT fire.
///
/// After the fix, the react loop should detect this inconsistent state via the
/// new incomplete-tool-call branch and surface a failure instead of silently
/// falling through to the text-only branch and completing.
#[tokio::test]
async fn tool_call_chunk_with_no_parsed_calls_and_stop_finish_fails() {
    let guard = Arc::new(RecordingGuard::new(GuardDecision::Fail {
        error: "incomplete tool call".to_string(),
    }));
    let guard_clone = guard.clone();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![vec![
        // A ToolCall chunk that sets is_tool_call=true but contains no
        // actual tool_calls in the delta (simulates incomplete stream).
        StreamChunk::ToolCall(serde_json::json!({ "no_delta": true })),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .guard_dyn(guard_clone)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid, "帮我解读一下这个工程", |_| Ok(()))
        .await;

    // The guard returns Fail for incomplete tool calls, so the run
    // should return Failed — NOT silently complete.
    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "incomplete tool call should fail, not silently complete. Got: {:?}",
        result
    );
}

// ---------------------------------------------------------------------------
// Branch 4 (text-only after tools): guard integration tests
// ---------------------------------------------------------------------------

use crate::engine::react_loop_guard::{GuardCtx, GuardDecision, ReactLoopGuard};

/// A guard that records every `on_turn` call for later inspection,
/// and returns a configurable decision.
struct RecordingGuard {
    /// Recorded (run_has_actions, model_response) from on_turn calls
    /// where is_text_only was true.
    calls: Mutex<Vec<(bool, String)>>,
    /// What to return from on_turn.
    decision: GuardDecision,
}

impl RecordingGuard {
    fn new(decision: GuardDecision) -> Self {
        Self {
            calls: Mutex::new(Vec::new()),
            decision,
        }
    }
}

#[async_trait]
impl ReactLoopGuard for RecordingGuard {
    async fn on_turn(&self, ctx: &GuardCtx) -> GuardDecision {
        if ctx.is_text_only {
            self.calls
                .lock()
                .unwrap()
                .push((ctx.run_has_actions, ctx.model_response.clone()));
        }
        self.decision.clone()
    }
}

/// When the model calls a tool then returns text, the guard's `on_text_only`
/// must be invoked with `run_has_actions = true`.
#[tokio::test]
async fn branch4_guard_called_with_run_has_actions() {
    let guard = Arc::new(RecordingGuard::new(GuardDecision::Complete));
    let guard_clone = guard.clone();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: tool call
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
        // Turn 2: text-only answer → branch 4
        vec![
            StreamChunk::Text("The answer is 42.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .guard_dyn(guard_clone)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let result = runtime.run_turn(sid, "echo hello", |_| Ok(())).await;
    assert!(
        matches!(result, Ok(RunOutcome::Completed)),
        "run_turn should complete: {:?}",
        result
    );

    let calls = guard.calls.lock().unwrap();
    assert_eq!(calls.len(), 1, "on_text_only should be called exactly once");
    let (run_has_tools, response) = &calls[0];
    assert!(
        *run_has_tools,
        "run_has_actions must be true after tool use"
    );
    assert_eq!(response, "The answer is 42.");
}

/// When the guard returns `Continue`, the react loop should inject a nudge
/// and call the LLM again — the second text response should complete.
#[tokio::test]
async fn branch4_guard_continue_nudges_and_retries() {
    // Guard always returns Continue → nudge each time → loops until max turns.
    // We verify: (1) guard called multiple times, (2) nudge injected, (3) run_has_actions persists.
    let guard = Arc::new(RecordingGuard::new(GuardDecision::Continue {
        nudge: Some("please continue".to_string()),
    }));
    let guard_clone = guard.clone();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: tool call
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
        // Turn 2: text-only → guard says Continue
        vec![
            StreamChunk::Text("maybe done".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        // Turn 3: after nudge, model gives final answer
        vec![
            StreamChunk::Text("here is the complete answer".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(EchoTool)
    .guard_dyn(guard_clone)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let (_events, _outcome) = runtime
        .run_turn_collect(sid.clone(), "echo hello")
        .await
        .expect("run_turn_collect");

    // Extract guard call data before any async work.
    let (call_count, all_have_tools) = {
        let calls = guard.calls.lock().unwrap();
        (calls.len(), calls.iter().all(|(has_tools, _)| *has_tools))
    };
    assert!(
        call_count >= 2,
        "guard should be called at least twice (initial + after nudge), got: {}",
        call_count
    );

    // Verify the nudge was pushed as a user message.
    let session = runtime.session(&sid).await.expect("session exists");
    let has_nudge = session.chat_messages().iter().any(
        |m| matches!(m, ChatMessage::User { content, .. } if content.contains("please continue")),
    );
    assert!(has_nudge, "session should contain the nudge message");

    // The second on_text_only call should still have run_has_actions = true
    // (it's never reset mid-run).
    assert!(
        all_have_tools,
        "all on_text_only calls should have run_has_actions = true"
    );
}

/// Guard that counts calls when `is_empty_response` is true and returns
/// `Continue` for the first `max_strikes` calls, then `Fail`.
/// All other scenes return `Complete`.
struct EmptyResponseStrikeGuard {
    max_strikes: usize,
    calls: AtomicUsize,
}

impl EmptyResponseStrikeGuard {
    fn new(max_strikes: usize) -> Self {
        Self {
            max_strikes,
            calls: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl ReactLoopGuard for EmptyResponseStrikeGuard {
    async fn on_turn(&self, ctx: &GuardCtx) -> GuardDecision {
        if ctx.is_empty_response {
            let count = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            if count >= self.max_strikes {
                GuardDecision::Fail {
                    error: format!("exceeded max retries ({})", self.max_strikes),
                }
            } else {
                GuardDecision::Continue {
                    nudge: Some("please retry".to_string()),
                }
            }
        } else {
            GuardDecision::Complete
        }
    }
}

/// Incomplete tool calls (is_tool_call=true, tool_calls empty) should trigger
/// the on_empty_response guard, which nudges the model to retry.  After
/// `max_strikes` consecutive failures, the guard returns Fail and the run ends
/// with RunOutcome::Failed — not an infinite loop, not a silent completion.
#[tokio::test]
async fn incomplete_tool_calls_fail_after_max_strikes() {
    let max_strikes = 3;
    let guard = Arc::new(EmptyResponseStrikeGuard::new(max_strikes));
    let guard_clone = guard.clone();

    // Every turn returns an incomplete tool call — no turn ever succeeds.
    let script: Vec<Vec<StreamChunk>> = (0..max_strikes + 1)
        .map(|_| {
            vec![
                StreamChunk::ToolCall(serde_json::json!({ "no_delta": true })),
                StreamChunk::Stop {
                    finish_reason: Some("stop".to_string()),
                },
            ]
        })
        .collect();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(script)))
        .system_prompt("test")
        .register_tool(EchoTool)
        .guard_dyn(guard_clone)
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let result = runtime
        .run_turn(sid, "帮我解读一下这个工程", |_| Ok(()))
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "incomplete tool calls should fail after {} strikes, got: {:?}",
        max_strikes,
        result
    );

    // Verify the guard was called exactly max_strikes times.
    let calls = guard.calls.load(Ordering::SeqCst);
    assert_eq!(
        calls, max_strikes,
        "guard should be called {} times, got: {}",
        max_strikes, calls
    );
}
