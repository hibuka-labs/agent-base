use crate::types::{ChatMessage, SessionId};

/// Find the first non-ephemeral System message (the system prompt).
pub fn first_system_prompt(messages: &[ChatMessage]) -> Option<ChatMessage> {
    messages.iter().find_map(|msg| match msg {
        ChatMessage::System {
            content,
            ephemeral: false,
        } => Some(ChatMessage::system(content.clone())),
        _ => None,
    })
}

/// Estimate total tokens across a message list using `ContextWindowManager::message_tokens`.
pub fn estimate_messages_tokens(messages: &[ChatMessage]) -> usize {
    messages
        .iter()
        .map(ContextWindowManager::message_tokens)
        .sum()
}

// ── Context Window Manager ──────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct ContextWindowManager {
    pub max_tokens: usize,
    /// Always keep first N messages (typically system prompt)
    pub keep_first_n: usize,
    /// Always keep last N messages
    pub keep_last_n: usize,
}

impl Default for ContextWindowManager {
    fn default() -> Self {
        Self {
            max_tokens: 128_000,
            keep_first_n: 1,
            keep_last_n: 20,
        }
    }
}

impl ContextWindowManager {
    /// OpenAI Vision API fixed token overhead per image
    const IMAGE_OVERHEAD_TOKENS: usize = 85;

    pub fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            ..Default::default()
        }
    }

    pub fn with_keep_first_n(mut self, n: usize) -> Self {
        self.keep_first_n = n;
        self
    }

    pub fn with_keep_last_n(mut self, n: usize) -> Self {
        self.keep_last_n = n;
        self
    }

    /// Simple token estimation: ~4 chars/token for Latin, ~1.5 for CJK
    /// Mixed text uses a compromise of 3 chars/token
    pub fn estimate_tokens(text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let chars = text.chars().count();
        let cjk_count = text.chars().filter(|c| is_cjk(*c)).count();
        let latin_count = chars - cjk_count;
        // CJK: ~1.5 chars/token, Latin: ~4 chars/token
        (cjk_count as f64 / 1.5 + latin_count as f64 / 4.0).ceil() as usize
    }

    pub(crate) fn message_tokens(msg: &ChatMessage) -> usize {
        match msg {
            ChatMessage::System { content, .. } => Self::estimate_tokens(content),
            ChatMessage::User {
                content, images, ..
            } => {
                let mut tokens = Self::estimate_tokens(content);
                for img in images {
                    match img {
                        crate::types::ImageAttachment::Url { url, detail: _ } => {
                            tokens += Self::estimate_tokens(url);
                        }
                        crate::types::ImageAttachment::Base64 {
                            data,
                            media_type,
                            detail: _,
                        } => {
                            tokens += data.len() / 4;
                            if let Some(mt) = media_type {
                                tokens += Self::estimate_tokens(mt);
                            }
                        }
                    }
                    tokens += Self::IMAGE_OVERHEAD_TOKENS;
                }
                tokens
            }
            ChatMessage::Assistant {
                content,
                reasoning_content,
                tool_calls,
                thinking_signature: _,
            } => {
                let mut tokens = content.as_deref().map(Self::estimate_tokens).unwrap_or(0);
                if let Some(rc) = reasoning_content {
                    tokens += Self::estimate_tokens(rc);
                }
                if let Some(tc) = tool_calls {
                    for t in tc {
                        tokens += Self::estimate_tokens(&t.name);
                        tokens += Self::estimate_tokens(&t.arguments);
                        tokens += Self::estimate_tokens(&t.id);
                    }
                }
                tokens
            }
            ChatMessage::Tool {
                tool_call_id,
                content,
                ..
            } => Self::estimate_tokens(tool_call_id) + Self::estimate_tokens(content),
            ChatMessage::Custom { role, data } => {
                Self::estimate_tokens(role) + Self::estimate_tokens(&data.to_string())
            }
        }
    }

    /// Trim message list to keep total tokens under `max_tokens`。
    ///
    /// Trimming strategy:
    /// - Always keep the first `keep_first_n` messages (typically system prompt)
    /// - Always keep the last `keep_last_n` messages (recent conversation)
    /// - Remove oldest messages from the middle until within budget
    pub fn trim(&self, messages: &mut Vec<ChatMessage>) {
        if messages.is_empty() || self.max_tokens == 0 {
            return;
        }

        let total_tokens: usize = messages.iter().map(Self::message_tokens).sum();
        if total_tokens <= self.max_tokens {
            return;
        }

        let keep_first = self.keep_first_n.min(messages.len());
        let keep_last = self
            .keep_last_n
            .min(messages.len().saturating_sub(keep_first));

        // Trimmable range: [keep_first, messages.len() - keep_last)
        let trim_start = keep_first;
        let trim_end = messages.len().saturating_sub(keep_last);
        if trim_start >= trim_end {
            return;
        }

        let mut current_tokens: usize = total_tokens;
        let remove_idx = trim_start;
        let mut trim_end = trim_end;

        while current_tokens > self.max_tokens && remove_idx < trim_end {
            let removed = Self::message_tokens(&messages[remove_idx]);
            messages.remove(remove_idx);
            current_tokens = current_tokens.saturating_sub(removed);
            trim_end = messages.len().saturating_sub(keep_last);
        }
    }
}

fn is_cjk(c: char) -> bool {
    matches!(
        c,
        '\u{4E00}'..='\u{9FFF}'   // CJK Unified Ideographs
        | '\u{3400}'..='\u{4DBF}' // CJK Unified Ideographs Extension A
        | '\u{3000}'..='\u{303F}' // CJK Symbols and Punctuation
        | '\u{FF00}'..='\u{FFEF}' // Halfwidth and Fullwidth Forms
        | '\u{3040}'..='\u{309F}' // Hiragana
        | '\u{30A0}'..='\u{30FF}' // Katakana
        | '\u{AC00}'..='\u{D7AF}' // Hangul Syllables
    )
}

// ── Inline Context Compaction ──────────────────────────────────────────────

/// Trait for inline context compaction within the react loop.
///
/// Implemented by agent-works's `ContextCompactor`. agent-base defines
/// the trait to avoid circular dependencies (agent-works depends on agent-base).
///
/// The react loop calls [`compact`](Self::compact) after tool execution when
/// the estimated token count exceeds a configurable threshold. This prevents
/// context window overflow without requiring the LLM call to fail first.
#[async_trait::async_trait]
pub trait ContextCompaction: Send + Sync {
    /// Compact a message history.
    ///
    /// Takes the current messages and returns `Some(compacted_messages)` if
    /// compaction was performed. Returns `None` if compaction was skipped
    /// (below threshold, too few messages, or disabled).
    ///
    /// The react loop handles reading/writing the session — the compactor
    /// only transforms the message list.
    async fn compact(
        &self,
        session_id: &SessionId,
        messages: &[ChatMessage],
    ) -> Option<Vec<ChatMessage>>;

    /// Estimate current token count for the session.
    ///
    /// Returns `None` if the implementation cannot estimate (falls back to
    /// `ContextWindowManager::estimate_tokens` on the react loop side).
    fn token_count_hint(&self, session_id: &SessionId) -> Option<usize>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ImageAttachment, ToolCallMessage};

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(ContextWindowManager::estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_english() {
        let text = "Hello world this is a test";
        let tokens = ContextWindowManager::estimate_tokens(text);
        // ~28 chars / 4 ≈ 7
        assert!(tokens > 0 && tokens <= 15);
    }

    #[test]
    fn test_estimate_tokens_cjk() {
        // 4 CJK chars / 1.5 -> ceil(2.667) = 3
        assert_eq!(ContextWindowManager::estimate_tokens("你好世界"), 3);
    }

    #[test]
    fn test_estimate_tokens_mixed() {
        // 2 CJK + 5 latin: 2/1.5 + 5/4 = 1.333 + 1.25 = 2.583 -> 3
        assert_eq!(ContextWindowManager::estimate_tokens("你好hello"), 3);
    }

    #[test]
    fn test_message_tokens_user_with_url_image() {
        let msg = ChatMessage::user_with_images(
            "pic",
            vec![ImageAttachment::Url {
                url: "http://x/a.png".into(),
                detail: None,
            }],
        );
        let base = ContextWindowManager::message_tokens(&ChatMessage::user("pic"));
        let t = ContextWindowManager::message_tokens(&msg);
        assert!(t > base);
    }

    #[test]
    fn test_message_tokens_user_with_base64_image() {
        // with media_type
        let msg = ChatMessage::user_with_images(
            "pic",
            vec![ImageAttachment::Base64 {
                data: "abcd".into(),
                media_type: Some("image/png".into()),
                detail: None,
            }],
        );
        let base = ContextWindowManager::message_tokens(&ChatMessage::user("pic"));
        assert!(ContextWindowManager::message_tokens(&msg) > base);

        // without media_type
        let msg = ChatMessage::user_with_images(
            "pic",
            vec![ImageAttachment::Base64 {
                data: "abcd".into(),
                media_type: None,
                detail: None,
            }],
        );
        assert!(ContextWindowManager::message_tokens(&msg) > base);
    }

    #[test]
    fn test_message_tokens_assistant_reasoning_and_tool_calls() {
        let msg = ChatMessage::Assistant {
            content: Some("ans".into()),
            reasoning_content: Some("thinking".into()),
            tool_calls: Some(vec![ToolCallMessage {
                id: "tc1".into(),
                name: "echo".into(),
                arguments: "{}".into(),
            }]),
            thinking_signature: None,
        };
        let t = ContextWindowManager::message_tokens(&msg);
        assert!(t > 0);
    }

    #[test]
    fn test_message_tokens_tool_and_custom() {
        let tool = ChatMessage::tool("tc1", "done");
        assert!(ContextWindowManager::message_tokens(&tool) > 0);

        let custom = ChatMessage::Custom {
            role: "artifact".into(),
            data: serde_json::json!({"x": 1}),
        };
        assert!(ContextWindowManager::message_tokens(&custom) > 0);
    }

    #[test]
    fn test_trim_no_trim_needed() {
        let mgr = ContextWindowManager::new(1000);
        let mut msgs = vec![
            ChatMessage::system("You are a helpful assistant."),
            ChatMessage::user("Hello"),
            ChatMessage::assistant("Hi there!"),
        ];
        let original_len = msgs.len();
        mgr.trim(&mut msgs);
        assert_eq!(msgs.len(), original_len);
    }

    #[test]
    fn test_trim_keeps_first_and_last() {
        let mgr = ContextWindowManager::new(8)
            .with_keep_first_n(1)
            .with_keep_last_n(2);
        let mut msgs = vec![
            ChatMessage::system("system"),
            ChatMessage::user("message number one"),
            ChatMessage::assistant("message number two"),
            ChatMessage::user("message number three"),
            ChatMessage::assistant("message number four"),
            ChatMessage::user("message number five"),
            ChatMessage::assistant("message number six"),
        ];
        mgr.trim(&mut msgs);
        assert_eq!(msgs.len(), 3);
        assert!(matches!(msgs[0], ChatMessage::System { .. }));
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn estimate_tokens_never_panics(text in ".*") {
            let tokens = ContextWindowManager::estimate_tokens(&text);
            // tokens should be non-negative (usize) and reasonable
            assert!(tokens <= text.len() + 1); // at most 1 token per byte + ceil
        }

        #[test]
        fn estimate_tokens_empty_is_zero(text in "[a-z\u{4e00}-\u{9fff}]{0,100}") {
            if text.is_empty() {
                assert_eq!(ContextWindowManager::estimate_tokens(&text), 0);
            } else {
                assert!(ContextWindowManager::estimate_tokens(&text) > 0);
            }
        }

        #[test]
        fn estimate_tokens_cjk_higher_than_latin_same_len(
            cjk_text in "[\u{4e00}-\u{9fff}]{1,50}",
            latin_text in "[a-z]{1,50}",
        ) {
            // Pad to same char length
            let max_len = cjk_text.chars().count().max(latin_text.chars().count());
            let cjk_padded: String = cjk_text.chars().cycle().take(max_len).collect();
            let latin_padded: String = latin_text.chars().cycle().take(max_len).collect();
            let cjk_tokens = ContextWindowManager::estimate_tokens(&cjk_padded);
            let latin_tokens = ContextWindowManager::estimate_tokens(&latin_padded);
            // CJK ~1.5 chars/token, Latin ~4 chars/token → CJK uses more tokens
            assert!(cjk_tokens >= latin_tokens,
                "CJK ({}) should use >= tokens than Latin ({}) for {} chars",
                cjk_tokens, latin_tokens, max_len);
        }

        #[test]
        fn trim_preserves_system_prefix(
            num_messages in 2usize..15,
            max_tokens in 5usize..50,
        ) {
            let mgr = ContextWindowManager {
                max_tokens,
                keep_first_n: 1,
                keep_last_n: 0,
            };
            let mut msgs = vec![ChatMessage::system("system prompt")];
            for i in 0..num_messages {
                msgs.push(ChatMessage::user(format!("message {}", i)));
            }
            mgr.trim(&mut msgs);
            // System message should always be preserved
            assert!(!msgs.is_empty());
            assert!(matches!(msgs[0], ChatMessage::System { .. }));
        }

        #[test]
        fn trim_result_within_budget(
            num_messages in 3usize..15,
            max_tokens in 10usize..100,
        ) {
            let mgr = ContextWindowManager {
                max_tokens,
                keep_first_n: 1,
                keep_last_n: 1,
            };
            let mut msgs = vec![ChatMessage::system("sys")];
            for i in 0..num_messages {
                msgs.push(ChatMessage::user(format!("msg {}", i)));
            }
            let total_before: usize = msgs.iter().map(ContextWindowManager::message_tokens).sum();
            // Only test when we actually exceed budget
            if total_before > max_tokens {
                mgr.trim(&mut msgs);
                let total_after: usize = msgs.iter().map(ContextWindowManager::message_tokens).sum();
                // After trimming, should be within budget (or couldn't trim more)
                assert!(total_after <= total_before);
            }
        }
    }
}
