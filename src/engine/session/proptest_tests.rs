use super::*;
use proptest::prelude::*;

proptest! {
    // ── push_assistant_tool_calls property tests ────────────────────────

    #[test]
    fn push_assistant_tool_calls_valid_json_unchanged(args in r"\{[^{}]{0,200}\}") {
        // Only test strings that are actually valid JSON objects
        if serde_json::from_str::<serde_json::Value>(&args).is_err() {
            return Ok(());
        }
        let mut s = make_session();
        s.push_assistant_tool_calls(
            &[("id".into(), "tool".into(), args.clone())],
            None,
            None,
        );
        if let ChatMessage::Assistant { tool_calls: Some(ref tc), .. } = s.chat_messages()[0] {
            assert_eq!(tc[0].arguments, args);
        } else {
            panic!("expected Assistant with tool_calls");
        }
    }

    #[test]
    fn push_assistant_tool_calls_invalid_json_sanitized_to_empty(
        bad_args in "[a-z\u{4e00}-\u{9fff}]{0,300}"
    ) {
        // Skip if it happens to be valid JSON
        if serde_json::from_str::<serde_json::Value>(&bad_args).is_ok() {
            return Ok(());
        }
        let mut s = make_session();
        s.push_assistant_tool_calls(
            &[("id".into(), "tool".into(), bad_args)],
            None,
            None,
        );
        if let ChatMessage::Assistant { tool_calls: Some(ref tc), .. } = s.chat_messages()[0] {
            // Must be valid JSON (no 400 on replay) AND carry no poison
            // wrapper the model could echo back as arguments.
            serde_json::from_str::<serde_json::Value>(&tc[0].arguments)
                .expect("sanitized args must be valid JSON");
            assert_eq!(tc[0].arguments, "{}");
        } else {
            panic!("expected Assistant with tool_calls");
        }
    }

    #[test]
    fn preview_args_escapes_and_caps_at_max_chars(payload in "[a-z\u{4e00}-\u{9fff}\n\t]{0,400}") {
        // args_preview for the truncation WARN: truncate by characters (not bytes),
        // escape control characters in debug form (each input character expands to
        // at most 2 characters), with length capped at min(80, len) * 2 + the paired quotes.
        let preview = AgentSession::preview_args(&payload, 80);
        let expected_cap = 2 * payload.chars().count().min(80) + 2;
        assert!(
            preview.chars().count() <= expected_cap,
            "preview must cap at escaped length plus quotes: {preview}"
        );
        if payload.contains('\n') || payload.contains('\t') {
            assert!(
                preview.contains("\\n") || preview.contains("\\t"),
                "control chars must be escaped: {preview}"
            );
        }
        if payload.chars().count() <= 80 && !payload.contains('\n') && !payload.contains('\t') {
            assert_eq!(preview, format!("{payload:?}"));
        }
    }

    // ── trim_oldest_turns property tests ────────────────────────────────

    #[test]
    fn trim_oldest_turns_never_exceeds_max(turns in 1usize..20, max in 1usize..20) {
        let mut s = make_session();
        for i in 0..turns {
            s.push_message(MessageRole::User, format!("u{}", i));
            s.push_message(MessageRole::Assistant, format!("a{}", i));
        }
        s.trim_oldest_turns(max);
        assert!(s.turn_count() <= max || turns <= max);
    }

    // ── validate_message_sequence property tests ────────────────────────

    #[test]
    fn validate_simple_user_assistant_always_passes(count in 1usize..20) {
        let mut msgs = Vec::new();
        for i in 0..count {
            msgs.push(ChatMessage::user(format!("msg{}", i)));
            msgs.push(ChatMessage::assistant(format!("reply{}", i)));
        }
        assert!(validate_message_sequence(&msgs).is_ok());
    }
}
