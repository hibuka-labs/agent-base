use super::*;

use crate::engine::executor::ParseOutcome;
use crate::engine::runtime::LlmTurnResult;
use crate::types::AgentError;

#[test]
fn failing_tool_names_isolates_single_failing_tool() {
    let tool_calls = vec![
        (
            "id1".to_string(),
            "write_file".to_string(),
            "{}".to_string(),
        ),
        (
            "id2".to_string(),
            "write_file".to_string(),
            "{}".to_string(),
        ),
        (
            "id3".to_string(),
            "write_file".to_string(),
            "{}".to_string(),
        ),
    ];

    // One bad call in a batch of three must yield one name, not three.
    let err = AgentError::ToolArgsInvalid {
        name: "write_file".to_string(),
        raw: "truncated-json".to_string(),
    };
    assert_eq!(
        failing_tool_names(&tool_calls, &err),
        vec!["write_file".to_string()]
    );
}

#[test]
fn failing_tool_names_extracts_execution_failure_name() {
    let tool_calls = vec![
        ("id1".to_string(), "bash".to_string(), "{}".to_string()),
        ("id2".to_string(), "bash".to_string(), "{}".to_string()),
    ];
    let err = AgentError::ToolExecution {
        name: "bash".to_string(),
        source: Box::new(AgentError::internal("boom")),
    };
    assert_eq!(
        failing_tool_names(&tool_calls, &err),
        vec!["bash".to_string()]
    );
}

#[test]
fn failing_tool_names_falls_back_to_full_batch_for_unknown_error() {
    let tool_calls = vec![
        ("id1".to_string(), "a".to_string(), "{}".to_string()),
        ("id2".to_string(), "b".to_string(), "{}".to_string()),
    ];
    let err = AgentError::internal("unexpected");
    assert_eq!(
        failing_tool_names(&tool_calls, &err),
        vec!["a".to_string(), "b".to_string()]
    );
}

/// Four-state verdict of the envelope classifier (D4+OV-7 mapping).
#[test]
fn classify_maps_envelope_states() {
    let mut turn = LlmTurnResult {
        full_text: String::new(),
        reasoning_text: String::new(),
        is_tool_call: true,
        tool_calls: vec![serde_json::json!({
            "id": "call_1", "name": "echo", "arguments": "{\"x\":1}"
        })],
        usage: None,
        finish_reason: Some("tool_calls".into()),
        ttft_ms: 1,
        llm_duration_ms: 2,
        reasoning_only: false,
        thinking_signature: None,
    };

    // Calls materialized → Actions.
    let out = classify_turn(&turn);
    assert!(matches!(out, ParseOutcome::Actions(ref a) if a.len() == 1 && a[0].id == "call_1"));

    // Tool intent, nothing parsed → Incomplete.
    turn.tool_calls = vec![];
    assert_eq!(classify_turn(&turn), ParseOutcome::Incomplete);

    // Truncated text without call flag → Malformed.
    turn.is_tool_call = false;
    turn.full_text = "partial...".into();
    turn.finish_reason = Some("length".into());
    assert_eq!(classify_turn(&turn), ParseOutcome::Malformed);

    // ToolUse mislabel with text → Malformed.
    turn.finish_reason = Some("tool_use".into());
    assert_eq!(classify_turn(&turn), ParseOutcome::Malformed);

    // Clean text → Empty (the loop's text path).
    turn.finish_reason = Some("stop".into());
    assert_eq!(classify_turn(&turn), ParseOutcome::Empty);

    // Empty text wins over the finish reason → Empty (old empty branch).
    turn.full_text = String::new();
    turn.finish_reason = Some("length".into());
    assert_eq!(classify_turn(&turn), ParseOutcome::Empty);
}
