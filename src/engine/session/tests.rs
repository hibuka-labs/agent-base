use super::*;

#[test]
fn test_turn_count_empty() {
    let s = make_session();
    assert_eq!(s.turn_count(), 0);
}

#[test]
fn test_turn_count_with_system_and_user() {
    let mut s = make_session();
    s.push_message(MessageRole::System, "system");
    assert_eq!(s.turn_count(), 0);
    s.push_message(MessageRole::User, "hello");
    assert_eq!(s.turn_count(), 1);
    s.push_message(MessageRole::Assistant, "hi");
    assert_eq!(s.turn_count(), 1);
    s.push_message(MessageRole::User, "bye");
    assert_eq!(s.turn_count(), 2);
}

#[test]
fn test_turn_count_with_tool_calls() {
    let mut s = make_session();
    s.push_message(MessageRole::User, "do something");
    s.push_assistant_tool_calls(&[("id1".into(), "tool".into(), "{}".into())], None, None);
    s.push_tool_result("id1", "result");
    s.push_message(MessageRole::Assistant, "done");
    // One user turn: User -> Assistant(tool_calls) -> Tool -> Assistant(text)
    assert_eq!(s.turn_count(), 1);
}

#[test]
fn test_trim_oldest_turns_noop() {
    let mut s = make_session();
    s.push_message(MessageRole::User, "hello");
    s.push_message(MessageRole::Assistant, "hi");
    s.trim_oldest_turns(5);
    assert_eq!(s.turn_count(), 1);
    assert_eq!(s.chat_messages().len(), 2);
}

#[test]
fn test_trim_oldest_turns_removes_old() {
    let mut s = make_session();
    s.push_message(MessageRole::System, "sys");
    // Turn 1
    s.push_message(MessageRole::User, "u1");
    s.push_message(MessageRole::Assistant, "a1");
    // Turn 2
    s.push_message(MessageRole::User, "u2");
    s.push_message(MessageRole::Assistant, "a2");
    // Turn 3
    s.push_message(MessageRole::User, "u3");
    s.push_message(MessageRole::Assistant, "a3");

    s.trim_oldest_turns(2);
    assert_eq!(s.turn_count(), 2);
    // System message preserved
    assert!(matches!(s.chat_messages()[0], ChatMessage::System { .. }));
    // Oldest user message is u2
    assert!(
        matches!(s.chat_messages()[1], ChatMessage::User { ref content, .. } if content == "u2")
    );
}

#[test]
fn test_trim_oldest_turns_with_tool_calls() {
    let mut s = make_session();
    // Turn 1 with tool call
    s.push_message(MessageRole::User, "u1");
    s.push_assistant_tool_calls(&[("id1".into(), "t".into(), "{}".into())], None, None);
    s.push_tool_result("id1", "r1");
    s.push_message(MessageRole::Assistant, "a1");
    // Turn 2
    s.push_message(MessageRole::User, "u2");
    s.push_message(MessageRole::Assistant, "a2");

    let msg_before = s.simple_messages().len();
    let chat_before = s.chat_messages().len();
    s.trim_oldest_turns(1);
    assert_eq!(s.turn_count(), 1);
    // chat_messages should have lost 4 entries (User, Assistant(tool), Tool, Assistant(text))
    assert_eq!(s.chat_messages().len(), chat_before - 4);
    // simple_messages (derived from chat_messages, tool_calls-only filtered) loses 3 entries
    assert_eq!(s.simple_messages().len(), msg_before - 3);
}

#[test]
fn test_pop_last_message_text() {
    let mut s = make_session();
    s.push_message(MessageRole::User, "hello");
    s.push_message(MessageRole::Assistant, "hi");
    assert_eq!(s.chat_messages().len(), 2);
    s.pop_last_message();
    assert_eq!(s.chat_messages().len(), 1);
    assert_eq!(s.simple_messages().len(), 1);
}

#[test]
fn test_pop_last_message_tool_calls_only() {
    let mut s = make_session();
    s.push_message(MessageRole::User, "do it");
    s.push_assistant_tool_calls(&[("id1".into(), "t".into(), "{}".into())], None, None);
    assert_eq!(s.chat_messages().len(), 2);
    assert_eq!(s.simple_messages().len(), 1); // only User in simple_messages (tool_calls-only filtered)
    s.pop_last_message();
    assert_eq!(s.chat_messages().len(), 1);
    assert_eq!(s.simple_messages().len(), 1); // simple_messages unchanged (still just User)
}

#[test]
fn test_pop_last_message_empty_session() {
    let mut s = make_session();
    s.pop_last_message(); // should not panic
    assert_eq!(s.chat_messages().len(), 0);
}

// ── B5: remaining session lifecycle paths ──────────────────────────────

#[test]
fn test_id_and_action_allowlist() {
    let mut s = make_session();
    assert_eq!(s.id(), Some(SessionId::new(1)));
    assert!(!s.is_action_allowed("approve:rm"));
    s.allow_action("approve:rm");
    assert!(s.is_action_allowed("approve:rm"));
    assert!(!s.is_action_allowed("approve:shell"));
}

#[test]
fn test_chat_messages_mut() {
    let mut s = make_session();
    s.chat_messages_mut().push(ChatMessage::user("direct"));
    assert_eq!(s.chat_messages().len(), 1);
}

#[test]
fn test_push_message_tool_role() {
    let mut s = make_session();
    s.push_message(MessageRole::Tool, "result");
    assert!(matches!(s.chat_messages()[0], ChatMessage::Tool { .. }));
}

#[test]
fn test_push_assistant_with_reasoning() {
    let mut s = make_session();
    s.push_assistant_with_reasoning("answer", "thinking");
    match &s.chat_messages()[0] {
        ChatMessage::Assistant {
            content,
            reasoning_content,
            ..
        } => {
            assert_eq!(content.as_deref(), Some("answer"));
            assert_eq!(reasoning_content.as_deref(), Some("thinking"));
        }
        other => panic!("unexpected message: {other:?}"),
    }
}

#[test]
fn test_push_user_message_with_images() {
    let mut s = make_session();
    s.push_user_message_with_images(
        "look",
        vec![ImageAttachment::Url {
            url: "http://x".into(),
            detail: None,
        }],
    );
    match &s.chat_messages()[0] {
        ChatMessage::User { images, .. } => assert_eq!(images.len(), 1),
        other => panic!("unexpected message: {other:?}"),
    }
}

/// The paste pipeline's chain pin: a user message with pictures keeps BOTH
/// legs through `chat_messages` — the text and the image payloads (base64
/// byte-identical) — so the wire request can carry them. fails_when: an
/// image is dropped, re-encoded, or detached from its user message.
#[test]
fn user_images_ride_the_user_message_intact() {
    let mut s = make_session();
    let png = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 42];
    s.push_user_message_with_images(
        "what is in this screenshot?",
        vec![
            ImageAttachment::Base64 {
                data: crate::image_ops::to_base64(&png),
                media_type: Some("image/png".to_string()),
                detail: None,
            },
            ImageAttachment::Url {
                url: "http://x".into(),
                detail: None,
            },
        ],
    );
    let msgs = s.chat_messages();
    assert_eq!(msgs.len(), 1);
    let ChatMessage::User {
        content, images, ..
    } = &msgs[0]
    else {
        panic!("unexpected message: {:?}", msgs[0]);
    };
    assert_eq!(content, "what is in this screenshot?");
    assert_eq!(images.len(), 2);
    let ImageAttachment::Base64 {
        data, media_type, ..
    } = &images[0]
    else {
        panic!("unexpected attachment: {:?}", images[0]);
    };
    assert_eq!(media_type.as_deref(), Some("image/png"));
    assert_eq!(
        crate::image_ops::from_base64(data).expect("valid base64"),
        png
    );
}

/// Machine-composed notices land on the user role with the synthetic
/// flag: the rotation trail must never carry them as the user's words.
#[test]
fn test_push_user_synthetic_flags_the_message() {
    let mut s = make_session();
    s.push_user_synthetic("Tool calls failed: exec_command");
    s.push_message(MessageRole::User, "actually do the thing");
    let msgs = s.chat_messages();
    assert!(msgs[0].is_synthetic());
    assert!(!msgs[1].is_synthetic());
}

#[test]
fn test_push_assistant_tool_call_singular() {
    let mut s = make_session();
    s.push_assistant_tool_call("call_1", "bash", "{}");
    match &s.chat_messages()[0] {
        ChatMessage::Assistant {
            tool_calls: Some(tc),
            ..
        } => {
            assert_eq!(tc.len(), 1);
            assert_eq!(tc[0].id, "call_1");
            assert_eq!(tc[0].name, "bash");
        }
        other => panic!("unexpected message: {other:?}"),
    }
}

#[test]
fn test_simple_messages_filters_empty_content_tool_calls() {
    let mut s = make_session();
    s.chat_messages_mut().push(ChatMessage::Assistant {
        content: Some(String::new()),
        reasoning_content: None,
        tool_calls: Some(vec![ToolCallMessage {
            id: "c".into(),
            name: "t".into(),
            arguments: "{}".into(),
        }]),
        thinking_signature: None,
    });
    assert!(s.simple_messages().is_empty());
}

#[test]
fn test_remove_ephemeral_messages() {
    let mut s = make_session();
    s.push_message(MessageRole::System, "keep");
    s.chat_messages_mut()
        .push(ChatMessage::user_ephemeral("temp"));
    s.chat_messages_mut()
        .push(ChatMessage::system_ephemeral("temp2"));
    s.push_message(MessageRole::User, "keep2");
    assert_eq!(s.chat_messages().len(), 4);
    s.remove_ephemeral_messages();
    assert_eq!(s.chat_messages().len(), 2);
    assert!(s.chat_messages().iter().all(|m| !m.is_ephemeral()));
}

#[test]
fn test_set_system_prompt_replaces_first_non_ephemeral_system() {
    let mut s = make_session();
    s.push_message(MessageRole::System, "old prompt");
    s.push_message(MessageRole::User, "hi");
    s.push_message(MessageRole::Assistant, "hello");

    s.set_system_prompt("new prompt");

    let msgs = s.chat_messages();
    assert_eq!(msgs.len(), 3, "history length unchanged");
    assert!(
        matches!(&msgs[0], ChatMessage::System { content, ephemeral: false } if content == "new prompt")
    );
    assert!(matches!(&msgs[1], ChatMessage::User { content, .. } if content == "hi"));
    assert!(matches!(&msgs[2], ChatMessage::Assistant { content: Some(c), .. } if c == "hello"));
}

#[test]
fn test_set_system_prompt_inserts_when_absent() {
    let mut s = make_session();
    s.push_message(MessageRole::User, "hi");

    s.set_system_prompt("fresh prompt");

    let msgs = s.chat_messages();
    assert_eq!(msgs.len(), 2);
    assert!(
        matches!(&msgs[0], ChatMessage::System { content, ephemeral: false } if content == "fresh prompt")
    );
    assert!(matches!(&msgs[1], ChatMessage::User { .. }));
}

#[test]
fn test_set_system_prompt_skips_ephemeral_system_and_inserts() {
    let mut s = make_session();
    s.chat_messages_mut()
        .push(ChatMessage::system_ephemeral("ephemeral nudge"));
    s.push_message(MessageRole::User, "hi");

    s.set_system_prompt("real prompt");

    // No non-ephemeral System exists → insert at the very front; the ephemeral nudge stays where it is.
    let msgs = s.chat_messages();
    assert_eq!(msgs.len(), 3);
    assert!(
        matches!(&msgs[0], ChatMessage::System { content, ephemeral: false } if content == "real prompt")
    );
    assert!(matches!(
        &msgs[1],
        ChatMessage::System {
            ephemeral: true,
            ..
        }
    ));
    assert!(matches!(&msgs[2], ChatMessage::User { .. }));
}

#[test]
fn test_close_dangling_tool_calls_noop_without_tool_call() {
    let mut s = make_session();
    s.push_message(MessageRole::User, "hi");
    s.push_message(MessageRole::Assistant, "hi");
    s.close_dangling_tool_calls("failed");
    assert_eq!(s.chat_messages().len(), 2);
}

#[test]
fn test_close_dangling_tool_calls_adds_missing_results() {
    let mut s = make_session();
    s.push_message(MessageRole::User, "do");
    s.push_assistant_tool_calls(
        &[
            ("c1".into(), "t".into(), "{}".into()),
            ("c2".into(), "t".into(), "{}".into()),
        ],
        None,
        None,
    );
    s.push_tool_result("c1", "ok"); // only c1 answered
    s.close_dangling_tool_calls("failed");

    let tool_results: Vec<(String, String)> = s
        .chat_messages()
        .iter()
        .filter_map(|m| match m {
            ChatMessage::Tool {
                tool_call_id,
                content,
                ..
            } => Some((tool_call_id.clone(), content.clone())),
            _ => None,
        })
        .collect();
    assert_eq!(tool_results.len(), 2);
    assert!(
        tool_results
            .iter()
            .any(|(id, c)| id == "c2" && c == "failed")
    );
}

#[test]
fn set_chat_messages_leaves_counters_alone() {
    // D5: the counting source is explicit — a messages replace must not
    // recount. (Replaces `test_set_chat_messages_recalculates_total_tool_calls`,
    // a named D13 exception: the demotion is the designed behavior change.)
    let mut s = make_session();
    s.run_state.record_actions(2);
    let msgs = vec![
        ChatMessage::user("do"),
        ChatMessage::assistant_tool_call("c1", "t", "{}"),
        ChatMessage::tool("c1", "result"),
    ];
    s.set_chat_messages(msgs).unwrap();
    assert_eq!(s.run_state.total_actions, 2); // untouched, not 1
    assert_eq!(s.run_state.turn_actions, 2);
}

#[test]
fn recount_actions_from_messages_counts_the_assistant_lens() {
    // The old-format entry point keeps the previous rebuild behavior.
    let mut s = make_session();
    let msgs = vec![
        ChatMessage::user("do"),
        ChatMessage::assistant_tool_call("c1", "t", "{}"),
        ChatMessage::tool("c1", "result"),
    ];
    s.set_chat_messages(msgs).unwrap();
    s.recount_actions_from_messages();
    assert_eq!(s.run_state.total_actions, 1);
}
