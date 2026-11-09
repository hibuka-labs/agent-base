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

    /// Vision pricing: ~750 pixels per token at high detail — the magnitude
    /// every provider's image bill lives in. The estimator prices the
    /// picture as the WIRE sends it (shrunk to `MAX_WIRE_LONG_EDGE`), never
    /// the session's stored original.
    ///
    /// The old estimator priced the base64 payload as text (`data.len() / 4`),
    /// which billed one 1278x1152 screenshot at ~219k tokens instead of ~2k.
    /// Two pasted screenshots then tripped the `max_message_tokens` safety
    /// valve, the valve popped the whole user message, and the turn went out
    /// with an empty message list (session 20261108_c525b6c7).
    const IMAGE_PIXELS_PER_TOKEN: u64 = 750;

    /// Price for an attachment whose header cannot be read (exotic format,
    /// corrupt payload, not an image at all): the worst case the wire budget
    /// lets through — a downscaled 1568x1568 picture is ~3.3k tokens — so an
    /// unreadable payload is never underpriced into the request.
    const IMAGE_UNREADABLE_TOKENS: usize = 3_500;

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

    /// What one picture costs — fixed overhead plus either its wire-scale
    /// pixel count (a base64 payload whose header reads), the URL text (a
    /// reference the provider fetches), or the capped fallback.
    fn image_tokens(image: &crate::types::ImageAttachment) -> usize {
        use crate::types::ImageAttachment;
        let body = match image {
            ImageAttachment::Url { url, detail: _ } => Self::estimate_tokens(url),
            ImageAttachment::Base64 {
                data,
                media_type,
                detail: _,
            } => {
                let mut tokens = match crate::image_ops::base64_prefix_dims(data) {
                    Some((w, h)) => Self::pixel_tokens(w, h),
                    None => Self::IMAGE_UNREADABLE_TOKENS,
                };
                if let Some(mt) = media_type {
                    tokens += Self::estimate_tokens(mt);
                }
                tokens
            }
        };
        body + Self::IMAGE_OVERHEAD_TOKENS
    }

    /// Pixel price of a picture as it travels: the wire shrinks any long
    /// edge above the cap, so price the scaled pixels — never more than a
    /// full-cap picture.
    fn pixel_tokens(w: u32, h: u32) -> usize {
        let long = u64::from(w.max(h)).max(1);
        let cap = u64::from(crate::image_ops::MAX_WIRE_LONG_EDGE);
        let (w, h) = if long > cap {
            (u64::from(w) * cap / long, u64::from(h) * cap / long)
        } else {
            (u64::from(w), u64::from(h))
        };
        let pixels = (w * h).max(1);
        (pixels.div_ceil(Self::IMAGE_PIXELS_PER_TOKEN)) as usize
    }

    /// Estimate a user message's tokens from its parts — the turn-entry size
    /// gate uses it before the message exists as a [`ChatMessage`].
    pub(crate) fn user_message_tokens(
        content: &str,
        images: &[crate::types::ImageAttachment],
    ) -> usize {
        Self::estimate_tokens(content) + images.iter().map(Self::image_tokens).sum::<usize>()
    }

    pub(crate) fn message_tokens(msg: &ChatMessage) -> usize {
        match msg {
            ChatMessage::System { content, .. } => Self::estimate_tokens(content),
            ChatMessage::User {
                content, images, ..
            } => Self::user_message_tokens(content, images),
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
                images,
                ..
            } => {
                // A tool result's pictures count like any other — before this
                // they rode the context budget for free while user pictures
                // were overcharged.
                let mut tokens =
                    Self::estimate_tokens(tool_call_id) + Self::estimate_tokens(content);
                for img in images {
                    tokens += Self::image_tokens(img);
                }
                tokens
            }
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

/// What a successful [`ContextCompaction::compact`] actually did to the
/// history. agent-base stays strategy-free: it only relays the kind into
/// the log line, so a nudge append never reads as a compaction and a real
/// replacement always does (issue #33).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionKind {
    /// One budget nudge appended at the end; prior messages untouched.
    Reminder,
    /// Last-turn nudge appended; prior messages untouched.
    Fallback,
    /// Messages replaced wholesale (window rotation or compression).
    Reset,
}

impl CompactionKind {
    /// The grep-able log line emitted when the compactor returns this kind.
    /// The three strings are deliberately distinct: a Reminder/Fallback append
    /// must NOT contain "compaction" (nothing was compacted — one message was
    /// appended), while a Reset must (the history was replaced).
    pub fn trigger_log(self) -> &'static str {
        match self {
            Self::Reminder => "context reminder appended",
            Self::Fallback => "context fallback appended",
            Self::Reset => "inline compaction triggered",
        }
    }

    /// The completion log line, emitted after the compactor's messages are
    /// installed. `None` for appends — they are atomic, there is nothing to
    /// complete, and a second line would only invite misreading.
    pub fn completion_log(self) -> Option<&'static str> {
        match self {
            Self::Reminder | Self::Fallback => None,
            Self::Reset => Some("inline compaction completed (window reset)"),
        }
    }
}

/// Successful compaction outcome: what happened plus the messages to install.
pub struct CompactionOutcome {
    pub kind: CompactionKind,
    pub messages: Vec<ChatMessage>,
}

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
    /// Takes the current messages and returns `Some(outcome)` if an action
    /// was performed — the outcome carries the action kind (for logging) and
    /// the messages to install. Returns `None` if compaction was skipped
    /// (below threshold, too few messages, or disabled).
    ///
    /// The react loop handles reading/writing the session — the compactor
    /// only transforms the message list.
    async fn compact(
        &self,
        session_id: &SessionId,
        messages: &[ChatMessage],
    ) -> Option<CompactionOutcome>;

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

    /// protects: a screenshot is priced by its wire-scale PIXELS (~750 px per
    /// token) — never by the length of its base64 payload, which billed one
    /// 1278x1152 screenshot at ~219k tokens and fed the max_message_tokens
    /// safety valve its victims (session 20261108_c525b6c7: two pastes
    /// estimated at 439k, the valve popped the user message, the turn 400ed
    /// on an empty message list). fails_when: the estimate scales with
    /// payload bytes. no_claim: any provider's exact bill.
    #[test]
    fn screenshot_images_price_by_pixels_not_base64_length() {
        let png = crate::image_ops::test_fixtures::noisy_png(1278, 1152);
        let b64 = crate::image_ops::to_base64(&png);
        let msg = ChatMessage::user_with_images(
            "what is in this screenshot?",
            vec![ImageAttachment::Base64 {
                data: b64.clone(),
                media_type: Some("image/png".into()),
                detail: None,
            }],
        );
        let tokens = ContextWindowManager::message_tokens(&msg);
        // The old formula's verdict on the very same payload:
        let old_formula = b64.len() / 4;
        assert!(
            old_formula > 100_000,
            "fixture must carry the incident's payload scale, got {old_formula}"
        );
        assert!(
            tokens < 5_000,
            "one screenshot must cost thousands of tokens, not {tokens} (payload was {old_formula} under the old formula)"
        );
        assert!(tokens > 1_500, "the pixel price is real, got {tokens}");
    }

    /// protects: a tool result's pictures enter the estimate like any other
    /// (they used to ride the context budget for free). fails_when: tool
    /// images are uncounted and trim starves. no_claim: that a tool's images
    /// and a user's images cost the same at any given provider.
    #[test]
    fn tool_result_images_are_counted_too() {
        let png =
            crate::image_ops::to_base64(&crate::image_ops::test_fixtures::noisy_png(200, 200));
        let text_only = ChatMessage::tool("tc1", "done");
        let with_image = ChatMessage::Tool {
            tool_call_id: "tc1".into(),
            name: Some("read_image".into()),
            content: "done".into(),
            images: vec![ImageAttachment::Base64 {
                data: png,
                media_type: Some("image/png".into()),
                detail: None,
            }],
        };
        let text_tokens = ContextWindowManager::message_tokens(&text_only);
        let image_tokens = ContextWindowManager::message_tokens(&with_image);
        assert!(
            image_tokens > text_tokens + 100,
            "tool image must cost its pixels: {text_tokens} vs {image_tokens}"
        );
    }

    /// protects: a payload whose header cannot be read (exotic format, junk)
    /// gets the capped fallback — bounded and independent of payload length,
    /// so garbage can never inflate the estimate into the safety valve's
    /// range. fails_when: unreadable payloads are priced by byte length or
    /// crash the estimator. no_claim: what such a payload would really cost.
    #[test]
    fn unreadable_image_payloads_get_the_capped_fallback() {
        let small: Vec<u8> = b"not an image".to_vec();
        let mut big = vec![b'x'; 512_000];
        big.extend_from_slice(b"also not an image");
        let estimate = |payload: Vec<u8>| {
            ContextWindowManager::user_message_tokens(
                "",
                &[ImageAttachment::Base64 {
                    data: crate::image_ops::to_base64(&payload),
                    media_type: None,
                    detail: None,
                }],
            )
        };
        let small_est = estimate(small);
        let big_est = estimate(big);
        assert_eq!(small_est, big_est, "fallback is size-independent");
        assert!(
            (1_000..5_000).contains(&small_est),
            "fallback is capped: {small_est}"
        );
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

    // ── CompactionKind log dispatch (issue #33) ──

    #[test]
    fn compaction_kind_log_lines_are_grep_able() {
        use super::{CompactionKind, CompactionOutcome};

        // A nudge append must never read as a compaction in the log...
        for kind in [CompactionKind::Reminder, CompactionKind::Fallback] {
            let trigger = kind.trigger_log();
            assert!(
                !trigger.contains("compaction"),
                "{kind:?} trigger log must not claim compaction: {trigger}"
            );
            assert!(
                kind.completion_log().is_none(),
                "{kind:?} append is atomic — no completion line"
            );
        }
        // ...while a real replacement must say compaction.
        let reset = CompactionKind::Reset;
        assert!(reset.trigger_log().contains("compaction"));
        let completion = reset.completion_log().expect("reset has a completion line");
        assert!(completion.contains("compaction"));

        // The three kinds are distinguishable by grep in session.log.
        let all = [
            CompactionKind::Reminder.trigger_log(),
            CompactionKind::Fallback.trigger_log(),
            reset.trigger_log(),
        ];
        for (i, a) in all.iter().enumerate() {
            for b in all.iter().skip(i + 1) {
                assert_ne!(a, b, "trigger log lines must be distinct");
            }
        }

        // Outcome struct: kind + messages travel together.
        let outcome = CompactionOutcome {
            kind: CompactionKind::Reminder,
            messages: vec![ChatMessage::user("nudge")],
        };
        assert_eq!(outcome.kind, CompactionKind::Reminder);
        assert_eq!(outcome.messages.len(), 1);
    }
}
