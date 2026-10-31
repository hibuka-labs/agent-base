//! Parse four-state fixture set (T9, D4+OV-7): pinned provider envelopes →
//! the state the classifier must return. Unit-only — no network, no model.
//!
//! The four states and their loop branch (P2 dispatch mapping):
//! - **Actions**    — materialized calls; the tool branch runs them.
//! - **Incomplete** — tool intent, nothing materialized; light nudge (retry).
//! - **Empty**      — nothing actionable; the old empty/text branch.
//! - **Malformed**  — structural anomaly; the finish-anomaly branch.
//!
//! Two traps are pinned on purpose: empty text beats Malformed (the old chain
//! checks the empty branch first), and invalid-JSON args still materialize —
//! parse keeps the raw bytes (`Action.args` is the argument text verbatim);
//! validity and truncation salvage are the execute side's job (P2).

use crate::engine::executor::ParseOutcome;
use crate::engine::runtime::LlmTurnResult;
use crate::types::Action;

/// Envelope entry: `{"id", "function": {"name", "arguments"}}`.
fn call(id: &str, name: &str, arguments: &str) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "function": {"name": name, "arguments": arguments},
    })
}

fn turn(
    is_tool_call: bool,
    calls: Vec<serde_json::Value>,
    text: &str,
    finish: Option<&str>,
) -> LlmTurnResult {
    LlmTurnResult {
        full_text: text.into(),
        reasoning_text: String::new(),
        is_tool_call,
        tool_calls: calls,
        usage: None,
        finish_reason: finish.map(String::from),
        ttft_ms: 1,
        llm_duration_ms: 2,
        reasoning_only: false,
        thinking_signature: None,
    }
}

/// A reasoning-only turn: thought, no output.
fn reasoning_only_turn() -> LlmTurnResult {
    let mut turn = turn(false, vec![], "", None);
    turn.reasoning_only = true;
    turn.reasoning_text = "thinking…".into();
    turn
}

#[test]
fn actions_fixtures_materialize_every_call() {
    // The plain case: one call with valid JSON args.
    let out = super::classify_turn(&turn(
        true,
        vec![call("call_1", "write_file", r#"{"path":"a.rs"}"#)],
        "",
        Some("tool_calls"),
    ));
    assert_eq!(
        out,
        ParseOutcome::Actions(vec![Action::from_tool_call(
            "call_1",
            "write_file",
            r#"{"path":"a.rs"}"#
        )]),
        "one envelope entry = one action, id and raw args preserved"
    );

    // A batch keeps envelope order and pairing ids.
    let out = super::classify_turn(&turn(
        true,
        vec![
            call("call_1", "read_file", r#"{"path":"a.rs"}"#),
            call("call_2", "execute_command", r#"{"command":"ls"}"#),
        ],
        "",
        Some("tool_calls"),
    ));
    let ParseOutcome::Actions(actions) = &out else {
        panic!("expected Actions, got {out:?}");
    };
    assert_eq!(
        actions.iter().map(|a| a.id.as_str()).collect::<Vec<_>>(),
        ["call_1", "call_2"]
    );
    assert_eq!(
        actions.iter().map(|a| a.name.as_str()).collect::<Vec<_>>(),
        ["read_file", "execute_command"]
    );

    // Invalid-JSON args still materialize: parse keeps the bytes verbatim
    // (P2) — the raw text is what history echoes back; validity/salvage is
    // the execute side's job.
    let out = super::classify_turn(&turn(
        true,
        vec![call("call_1", "write_file", r#"{"path":"a.rs"#)],
        "",
        Some("tool_calls"),
    ));
    let ParseOutcome::Actions(actions) = &out else {
        panic!("expected Actions, got {out:?}");
    };
    assert_eq!(actions[0].raw_args(), r#"{"path":"a.rs"#);
    assert_eq!(actions[0].identity, r#"write_file::{"path":"a.rs"#);
}

#[test]
fn incomplete_fixtures_are_tool_intent_without_materialization() {
    // The flag arrived but the calls were lost (truncation window).
    let out = super::classify_turn(&turn(true, vec![], "", Some("tool_calls")));
    assert_eq!(out, ParseOutcome::Incomplete, "flag without calls");

    // Truncated finish changes nothing: intent without materialization is
    // Incomplete (the light-retry nudge), not Malformed.
    let out = super::classify_turn(&turn(true, vec![], "", Some("length")));
    assert_eq!(out, ParseOutcome::Incomplete, "flag without calls, length");
}

#[test]
fn empty_fixtures_have_no_action_intent() {
    // The normal text answer — content, no action intent.
    let out = super::classify_turn(&turn(false, vec![], "all done.", Some("stop")));
    assert_eq!(out, ParseOutcome::Empty, "plain text answer");

    // A reasoning-only turn produced no output at all.
    let out = super::classify_turn(&reasoning_only_turn());
    assert_eq!(out, ParseOutcome::Empty, "reasoning-only turn");

    // Trap: empty text beats Malformed. A truncated finish with no text at
    // all is the empty-response branch (the old chain checks it first) —
    // not a structural anomaly.
    let out = super::classify_turn(&turn(false, vec![], "", Some("length")));
    assert_eq!(out, ParseOutcome::Empty, "empty text first, even truncated");
}

#[test]
fn malformed_fixtures_are_structural_anomalies() {
    // Tool intent without a call, and text that says so mid-stream: the
    // finish-anomaly branch.
    let out = super::classify_turn(&turn(false, vec![], "partial…", Some("length")));
    assert_eq!(out, ParseOutcome::Malformed, "truncated finish with text");

    // ToolUse mislabel: the provider said `"tool_use"` but delivered text
    // instead of tool_use blocks. (OpenAI's "tool_calls" never reaches the
    // loop — it is consumed client-side, `from_raw` says so.)
    let out = super::classify_turn(&turn(false, vec![], "some text", Some("tool_use")));
    assert_eq!(out, ParseOutcome::Malformed, "ToolUse mislabel with text");
}
