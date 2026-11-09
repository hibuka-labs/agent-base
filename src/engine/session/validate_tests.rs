use super::*;

#[test]
fn test_valid_simple_sequence() {
    let msgs = vec![ChatMessage::user("hello"), ChatMessage::assistant("hi")];
    assert!(validate_message_sequence(&msgs).is_ok());
}

#[test]
fn test_valid_tool_call_sequence() {
    let msgs = vec![
        ChatMessage::user("run command"),
        ChatMessage::assistant_tool_call("call_1", "bash", r#"{"cmd":"ls"}"#),
        ChatMessage::tool("call_1", "file1 file2"),
        ChatMessage::assistant("done"),
    ];
    assert!(validate_message_sequence(&msgs).is_ok());
}

#[test]
fn test_valid_multi_tool_call_sequence() {
    let msgs = vec![
        ChatMessage::user("run commands"),
        ChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![
                crate::types::ToolCallMessage {
                    id: "call_1".into(),
                    name: "bash".into(),
                    arguments: "{}".into(),
                },
                crate::types::ToolCallMessage {
                    id: "call_2".into(),
                    name: "read".into(),
                    arguments: "{}".into(),
                },
            ]),
            thinking_signature: None,
        },
        ChatMessage::tool("call_1", "result1"),
        ChatMessage::tool("call_2", "result2"),
        ChatMessage::assistant("done"),
    ];
    assert!(validate_message_sequence(&msgs).is_ok());
}

#[test]
fn test_orphaned_tool_result() {
    let msgs = vec![
        ChatMessage::user("hello"),
        ChatMessage::tool("call_1", "orphaned result"),
    ];
    let err = validate_message_sequence(&msgs).unwrap_err();
    assert!(err.contains("no preceding tool_call"));
}

#[test]
fn test_system_only_sequence_rejected() {
    // A compaction that leaves only System/Custom messages would send an
    // empty `messages` array (System maps to the `system` param) — the
    // contract rejects it.
    let msgs = vec![
        ChatMessage::system("prompt"),
        ChatMessage::system_ephemeral("reminder"),
        ChatMessage::Custom {
            role: "artifact".into(),
            data: serde_json::json!({"id": "x"}),
        },
    ];
    let err = validate_message_sequence(&msgs).unwrap_err();
    assert!(err.contains("no sendable message"));
}

#[test]
fn test_system_plus_user_ok() {
    let msgs = vec![ChatMessage::system("prompt"), ChatMessage::user("hi")];
    assert!(validate_message_sequence(&msgs).is_ok());
}

#[test]
fn test_mismatched_tool_call_id() {
    let msgs = vec![
        ChatMessage::user("run"),
        ChatMessage::assistant_tool_call("call_1", "bash", "{}"),
        ChatMessage::tool("call_2", "wrong id"),
    ];
    let err = validate_message_sequence(&msgs).unwrap_err();
    assert!(err.contains("does not match"));
}

#[test]
fn test_set_chat_messages_valid() {
    let mut s = make_session();
    let msgs = vec![ChatMessage::user("hello"), ChatMessage::assistant("hi")];
    assert!(s.set_chat_messages(msgs.clone()).is_ok());
    assert_eq!(s.chat_messages().len(), 2);
}

#[test]
fn test_set_chat_messages_invalid() {
    let mut s = make_session();
    let msgs = vec![ChatMessage::tool("call_1", "orphaned")];
    assert!(s.set_chat_messages(msgs).is_err());
}

// ── Backward-compatible deserialization ────────────────────────────────

#[test]
fn deserialize_legacy_flat_fields() {
    // Old format: nudge_count, turn_tool_calls, etc. as flat fields
    let json = r#"{
            "id": null,
            "chat_messages": [],
            "always_allowed_actions": [],
            "total_tool_calls": 5,
            "nudge_count": 3,
            "turn_tool_calls": 2,
            "reasoning_only_strikes": 1,
            "empty_response_strikes": 0
        }"#;
    let session: AgentSession = serde_json::from_str(json).unwrap();
    assert_eq!(session.run_state.nudge_count, 3);
    assert_eq!(session.run_state.turn_actions, 2);
    assert_eq!(session.run_state.reasoning_only_strikes, 1);
    assert_eq!(session.run_state.empty_response_strikes, 0);
    assert!(!session.run_state.run_has_actions); // default
}

#[test]
fn deserialize_new_run_state_format() {
    // New format: nested run_state
    let json = r#"{
            "id": null,
            "chat_messages": [],
            "always_allowed_actions": [],
            "total_tool_calls": 5,
            "run_state": {
                "turn_tool_calls": 4,
                "run_has_tool_calls": true,
                "reasoning_only_strikes": 0,
                "empty_response_strikes": 1,
                "nudge_count": 2
            }
        }"#;
    let session: AgentSession = serde_json::from_str(json).unwrap();
    assert_eq!(session.run_state.turn_actions, 4);
    assert!(session.run_state.run_has_actions);
    assert_eq!(session.run_state.empty_response_strikes, 1);
    assert_eq!(session.run_state.nudge_count, 2);
}

#[test]
fn deserialize_run_state_takes_precedence_over_flat() {
    // When both are present, run_state wins
    let json = r#"{
            "id": null,
            "chat_messages": [],
            "always_allowed_actions": [],
            "total_tool_calls": 0,
            "run_state": {
                "turn_tool_calls": 10,
                "run_has_tool_calls": true,
                "reasoning_only_strikes": 0,
                "empty_response_strikes": 0,
                "nudge_count": 0
            },
            "nudge_count": 99,
            "turn_tool_calls": 99
        }"#;
    let session: AgentSession = serde_json::from_str(json).unwrap();
    assert_eq!(session.run_state.turn_actions, 10); // run_state wins
    assert_eq!(session.run_state.nudge_count, 0); // run_state wins
}

#[test]
fn deserialize_legacy_missing_optional_fields() {
    // Old format with some fields missing (defaults to 0)
    let json = r#"{
            "id": null,
            "chat_messages": [],
            "always_allowed_actions": [],
            "total_tool_calls": 0,
            "nudge_count": 1
        }"#;
    let session: AgentSession = serde_json::from_str(json).unwrap();
    assert_eq!(session.run_state.nudge_count, 1);
    assert_eq!(session.run_state.turn_actions, 0); // missing → 0
    assert_eq!(session.run_state.reasoning_only_strikes, 0);
    assert_eq!(session.run_state.empty_response_strikes, 0);
}

#[test]
fn roundtrip_preserves_run_state() {
    let mut session = AgentSession::new(SessionId::new(1));
    session.run_state.nudge_count = 5;
    session.run_state.turn_actions = 3;
    session.run_state.run_has_actions = true;
    session.run_state.reasoning_only_strikes = 2;

    let json = serde_json::to_string(&session).unwrap();
    let restored: AgentSession = serde_json::from_str(&json).unwrap();
    assert_eq!(restored.run_state.nudge_count, 5);
    assert_eq!(restored.run_state.turn_actions, 3);
    assert!(restored.run_state.run_has_actions);
    assert_eq!(restored.run_state.reasoning_only_strikes, 2);
}

#[test]
fn deserialize_migrates_top_level_total_into_run_state() {
    // Pre-T7 wire: the session total lived at the top level. It migrates
    // into the one counting home (D5), and the old top-level field is gone
    // from new writes.
    let json = r#"{
            "id": null,
            "chat_messages": [],
            "always_allowed_actions": [],
            "total_tool_calls": 7,
            "run_state": {
                "turn_tool_calls": 4,
                "run_has_tool_calls": true,
                "reasoning_only_strikes": 0,
                "empty_response_strikes": 0,
                "nudge_count": 0
            }
        }"#;
    let session: AgentSession = serde_json::from_str(json).unwrap();
    assert_eq!(session.run_state.total_actions, 7);
    assert_eq!(session.run_state.turn_actions, 4);

    let rewritten = serde_json::to_string(&session).unwrap();
    assert!(
        !rewritten.contains("total_tool_calls"),
        "new writes keep the total inside run_state: {rewritten}"
    );
}

#[test]
fn push_assistant_tool_calls_validates_json_args() {
    let mut s = make_session();

    // Valid JSON args should pass through unchanged
    let valid_args = r#"{"path": "src/main.rs", "content": "fn main() {}"}"#;
    s.push_assistant_tool_calls(
        &[("id1".into(), "write_file".into(), valid_args.into())],
        None,
        None,
    );
    if let ChatMessage::Assistant {
        tool_calls: Some(ref tc),
        ..
    } = s.chat_messages[0]
    {
        assert_eq!(tc[0].arguments, valid_args);
    } else {
        panic!("expected Assistant message with tool_calls");
    }

    // Truncated (invalid JSON) args must be sanitized to an empty object,
    // NOT wrapped in a descriptive error object. Two reasons:
    //   1. "{}" is valid JSON, so the next request is never rejected 400.
    //   2. a rich `{error, original_args_preview, message}` object in
    //      assistant history is an imitation vector — the model replays it
    //      verbatim as the next call's arguments, and because it parses as
    //      valid JSON it slips past the react truncation guard and resurfaces
    //      as a downstream ToolArgsInvalid. The truncation explanation
    //      belongs in the tool_result, not the assistant arguments.
    let truncated_args = r#"{"path": "src/ui/markdown.rs", "content": "#;
    s.push_assistant_tool_calls(
        &[("id2".into(), "write_file".into(), truncated_args.into())],
        None,
        None,
    );
    if let ChatMessage::Assistant {
        tool_calls: Some(ref tc),
        ..
    } = s.chat_messages[1]
    {
        assert_eq!(tc[0].arguments, "{}");
        assert!(
            !tc[0].arguments.contains("tool_call_arguments_truncated"),
            "assistant arguments must not carry the poison wrapper object"
        );
    } else {
        panic!("expected Assistant message with tool_calls");
    }
}

#[test]
fn push_assistant_tool_calls_truncated_multibyte_no_panic() {
    // Bug-2: Invalid JSON args with multi-byte UTF-8 chars near byte 200
    // cause a panic at char boundary when slicing &args[..args.len().min(200)].
    let mut s = make_session();

    // Build invalid JSON with CJK chars that straddle the 200-byte boundary.
    // "あ" = 3 bytes in UTF-8. Repeating ~70 times = ~210 bytes, then add invalid suffix.
    let mut bad_args = "あ".repeat(70); // 70 * 3 = 210 bytes
    bad_args.push_str("truncated"); // makes it invalid JSON

    // This must NOT panic. Invalid args are sanitized to "{}" (a fixed
    // literal), so no slicing of the multibyte string happens at all.
    s.push_assistant_tool_calls(&[("id1".into(), "tool".into(), bad_args)], None, None);

    if let ChatMessage::Assistant {
        tool_calls: Some(ref tc),
        ..
    } = s.chat_messages[0]
    {
        assert_eq!(tc[0].arguments, "{}");
    } else {
        panic!("expected Assistant message with tool_calls");
    }
}

#[test]
fn push_assistant_tool_calls_then_tool_result_matches_anthropic_protocol() {
    // Regression test for: when LLM response is truncated (finish_reason=max_tokens),
    // the code must push assistant message WITH tool_use blocks (not plain text)
    // so that subsequent tool_result messages can match the tool_use_id.
    // This is required by Anthropic protocol.
    let mut s = make_session();

    // Simulate truncated tool call response
    let tool_calls = vec![
        (
            "call_00_VJtlnKha0ZZ2Yo8t5ysQ8883".to_string(),
            "write_file".to_string(),
            "{}".to_string(),
        ),
        (
            "call_01_abc123".to_string(),
            "bash".to_string(),
            r#"{"command": "ls"}"#.to_string(),
        ),
    ];

    // Push assistant message with tool_calls (as the fix does)
    s.push_assistant_tool_calls(
        &tool_calls,
        Some("thinking...".to_string()),
        Some("I'll help you".to_string()),
    );

    // Push tool results for each tool call
    for (tc_id, _, _) in &tool_calls {
        s.push_tool_result(
            tc_id,
            "Tool call was not executed: the response hit the output token limit.",
        );
    }

    // Verify the message sequence is valid
    // 1. Assistant message should have tool_calls
    if let ChatMessage::Assistant {
        tool_calls: Some(ref tc),
        ..
    } = s.chat_messages[0]
    {
        assert_eq!(tc.len(), 2);
        assert_eq!(tc[0].id, "call_00_VJtlnKha0ZZ2Yo8t5ysQ8883");
        assert_eq!(tc[0].name, "write_file");
        assert_eq!(tc[1].id, "call_01_abc123");
        assert_eq!(tc[1].name, "bash");
    } else {
        panic!("expected Assistant message with tool_calls");
    }

    // 2. Tool result messages should have matching tool_use_id
    if let ChatMessage::Tool { tool_call_id, .. } = &s.chat_messages[1] {
        assert_eq!(tool_call_id, "call_00_VJtlnKha0ZZ2Yo8t5ysQ8883");
    } else {
        panic!("expected Tool message for first tool call");
    }

    if let ChatMessage::Tool { tool_call_id, .. } = &s.chat_messages[2] {
        assert_eq!(tool_call_id, "call_01_abc123");
    } else {
        panic!("expected Tool message for second tool call");
    }

    // 3. Validate message sequence (this would catch the bug)
    assert!(
        validate_message_sequence(&s.chat_messages).is_ok(),
        "message sequence should be valid with matching tool_use and tool_result"
    );
}
