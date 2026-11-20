//! Turn-as-transaction: a corrupt silent round is discarded whole and the
//! LLM is re-asked from the untouched history (design: notes/
//! turn-transaction-design.md in phimint). The guard (policy layer) decides
//! RetryTurn; the kernel executes it — no history commit, no tool execution,
//! budget bookkeeping in run_state.
//!
//! Wire provenance: the corrupt streams below mirror the captured MiniMax
//! anatomy (named call with unparseable args + a misplaced tail fragment on a
//! name-less bucket) — see llm_engine/tests/integrity.rs for the byte-level
//! fixture tests.

use super::*;
use crate::engine::react_loop_guard::{GuardCtx, GuardDecision, ReactLoopGuard};

/// Policy stub mirroring the works-layer judgment: silent + integrity dirty +
/// one retry per round → RetryTurn.
struct IntegrityRetryGuard;

#[async_trait]
impl ReactLoopGuard for IntegrityRetryGuard {
    async fn on_turn(&self, _ctx: &GuardCtx) -> GuardDecision {
        GuardDecision::Complete
    }

    async fn on_tool_call(&self, ctx: &GuardCtx) -> GuardDecision {
        if !ctx.integrity.is_clean() && !ctx.has_visible_text && ctx.silent_retries_in_round == 0 {
            GuardDecision::RetryTurn
        } else {
            GuardDecision::Complete
        }
    }
}

/// The corrupted batch: named `echo` call with args cut mid-JSON, plus the
/// lost tail arriving on a name-less bucket (index 1). Silent (no text).
fn corrupt_silent_round(id: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [{
                "id": id,
                "index": 0,
                "function": { "name": "echo", "arguments": "{\"text\":\"hel" }
            }]}
        })),
        // Misplaced tail: same batch's leftover fragment under a fresh index,
        // no name ever arrives → orphan bucket.
        StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [{
                "index": 1,
                "function": { "arguments": "lo\"}" }
            }]}
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_use".to_string()),
        },
    ]
}

fn clean_round(id: &str) -> Vec<StreamChunk> {
    vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": { "tool_calls": [{
                "id": id,
                "index": 0,
                "function": { "name": "echo", "arguments": "{\"text\":\"hi\"}" }
            }]}
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_use".to_string()),
        },
    ]
}

fn done_round() -> Vec<StreamChunk> {
    vec![
        StreamChunk::Text("done".to_string()),
        StreamChunk::Stop {
            finish_reason: Some("stop".to_string()),
        },
    ]
}

fn assistant_calls(messages: &[ChatMessage]) -> Vec<(String, String)> {
    messages
        .iter()
        .filter_map(|m| match m {
            ChatMessage::Assistant {
                tool_calls: Some(tcs),
                ..
            } => Some(tcs),
            _ => None,
        })
        .flatten()
        .map(|tc| (tc.name.clone(), tc.arguments.clone()))
        .collect()
}

fn tool_results(messages: &[ChatMessage]) -> Vec<String> {
    messages
        .iter()
        .filter_map(|m| match m {
            ChatMessage::Tool { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

/// A silent corrupt round leaves ZERO trace: the response is discarded whole,
/// the retry's clean call is the only thing history remembers, and the tool
/// executes exactly once.
#[tokio::test]
async fn silent_corrupt_round_is_discarded_and_retried_once() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        corrupt_silent_round("c_bad"),
        clean_round("c_good"),
        done_round(),
    ])))
    .system_prompt("test")
    .guard(IntegrityRetryGuard)
    .register_tool(EchoTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let outcome = runtime
        .run_turn(sid.clone(), "echo something", |_| Ok(()))
        .await
        .expect("run_turn completes");

    assert!(
        matches!(outcome, RunOutcome::Completed),
        "expected Completed: {outcome:?}"
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    // Exactly one committed call — the clean retry, not the corrupt original.
    let calls = assistant_calls(&messages);
    assert_eq!(
        calls,
        vec![("echo".to_string(), "{\"text\":\"hi\"}".to_string())],
        "history must contain only the clean call"
    );

    // Exactly one tool execution, with the clean args' output.
    let results = tool_results(&messages);
    assert_eq!(results, vec!["echo: hi".to_string()]);

    // Zero ladder residue — the corrupt round never reached the guard path.
    assert!(
        !results.iter().any(|r| r.contains("was not executed")),
        "silent retry must not leave re-issue residue: {results:?}"
    );
}

/// A corrupt round WITH visible text is not silently discarded (the user has
/// already seen the text) — it falls through to the existing ladder, which
/// leaves its re-issue trace in history.
#[tokio::test]
async fn corrupt_round_with_text_is_not_silently_retried() {
    let mut corrupt_with_text = vec![StreamChunk::Text("let me think...".to_string())];
    corrupt_with_text.extend(corrupt_silent_round("c_bad"));
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        corrupt_with_text,
        done_round(),
    ])))
    .system_prompt("test")
    .guard(IntegrityRetryGuard)
    .register_tool(EchoTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let outcome = runtime
        .run_turn(sid.clone(), "echo something", |_| Ok(()))
        .await
        .expect("run_turn completes");
    assert!(matches!(outcome, RunOutcome::Completed));

    let session = runtime.session(&sid).await.expect("session exists");
    let results = tool_results(session.chat_messages());
    assert!(
        results.iter().any(|r| r.contains("was not executed")),
        "non-silent corrupt round must fall to the ladder (re-issue residue): {results:?}"
    );
}

/// The discard is observable three ways: session counters, the turn-end
/// context (metrics seam), and a runtime event carrying the discarded
/// response (forensics — the session's event log keeps the raw calls).
#[tokio::test]
async fn discard_is_observable_in_counters_and_event() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        corrupt_silent_round("c_bad"),
        clean_round("c_good"),
        done_round(),
    ])))
    .system_prompt("test")
    .guard(IntegrityRetryGuard)
    .register_tool(EchoTool)
    .build()
    .expect("build runtime");

    let turns = Arc::new(Mutex::new(Vec::new()));
    let tc = turns.clone();
    runtime.on_turn_end(move |ctx| {
        tc.lock()
            .unwrap()
            .push((ctx.silent_retries, ctx.integrity_failures));
    });

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let ec = events.clone();
    runtime
        .run_turn(sid.clone(), "echo something", move |e| {
            ec.lock().unwrap().push(e);
            Ok(())
        })
        .await
        .expect("run_turn completes");

    // Session counters (metrics source).
    let session = runtime.session(&sid).await.expect("session");
    assert_eq!(session.run_state.silent_retries, 1);
    assert_eq!(session.run_state.integrity_failures, 1);

    // The turn-end context carries them (telemetry seam).
    let turns = turns.lock().unwrap();
    let last = turns.last().expect("a turn-end fired");
    assert_eq!(*last, (1, 1));

    // The discarded response is published as an event — the raw (still broken)
    // arguments are preserved verbatim for forensics.
    let events = events.lock().unwrap();
    let discarded: Vec<&RuntimeEvent> = events
        .iter()
        .filter(|e| matches!(e, RuntimeEvent::IntegrityDiscard { .. }))
        .collect();
    assert_eq!(discarded.len(), 1, "exactly one discard published");
    let RuntimeEvent::IntegrityDiscard {
        turn,
        invalid_args,
        orphan_args_bytes,
        tool_calls,
        text,
        ..
    } = discarded[0]
    else {
        panic!("expected IntegrityDiscard");
    };
    assert_eq!(*turn, 1);
    assert_eq!(*invalid_args, 1);
    // the synthetic round's misplaced tail is `lo"}` (4 bytes)
    assert_eq!(*orphan_args_bytes, 4);
    assert!(text.is_empty(), "a silent round carries no text");
    assert_eq!(
        tool_calls.len(),
        1,
        "the orphan bucket stays out of tool_calls"
    );
    let args = tool_calls[0]["function"]["arguments"].as_str().unwrap();
    assert!(
        serde_json::from_str::<serde_json::Value>(args).is_err(),
        "the discarded args stay broken — that is the evidence: {args}"
    );
}

/// Retry budget is one per round: a second consecutive corrupt round falls to
/// the ladder instead of retrying forever.
#[tokio::test]
async fn retry_budget_stops_after_one_discard() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        corrupt_silent_round("c_bad_1"),
        corrupt_silent_round("c_bad_2"),
        done_round(),
    ])))
    .system_prompt("test")
    .guard(IntegrityRetryGuard)
    .register_tool(EchoTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let outcome = runtime
        .run_turn(sid.clone(), "echo something", |_| Ok(()))
        .await
        .expect("run_turn completes");
    assert!(matches!(outcome, RunOutcome::Completed));

    let session = runtime.session(&sid).await.expect("session exists");
    let results = tool_results(session.chat_messages());
    // Exactly one ladder round: the second corrupt response (budget exhausted).
    let residue = results
        .iter()
        .filter(|r| r.contains("was not executed"))
        .count();
    assert_eq!(
        residue, 1,
        "second corrupt round must hit the ladder exactly once: {results:?}"
    );
}
