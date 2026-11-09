use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::types::{ChatMessage, ImageAttachment, Message, MessageRole, ToolCallMessage};

use crate::types::SessionId;

// RunState moved to agent-types (D5+OV-11): the counting state rides in
// `CheckpointData`, so it must live where the checkpoint payload lives.
// Re-exported here so existing `engine::session::RunState` imports keep working.
pub use crate::types::RunState;

/// Deserialization shim for backward compatibility.
///
/// Before `RunState` was introduced, `nudge_count`, `turn_tool_calls`,
/// `reasoning_only_strikes`, and `empty_response_strikes` were flat fields
/// on `AgentSession` — and the session-lifetime total lived at the top level
/// as `total_tool_calls`. This struct accepts **both** the old flat format and
/// the new nested `run_state` format, migrating legacy data on the fly.
#[derive(Deserialize)]
struct RawAgentSession {
    id: Option<SessionId>,
    chat_messages: Vec<ChatMessage>,
    always_allowed_actions: HashSet<String>,
    /// Pre-T7 session total (D5 moved it into `RunState::total_actions`).
    /// Optional so new-format payloads omit it; when present it migrates into
    /// the counting state — the one counting home.
    #[serde(default)]
    total_tool_calls: Option<usize>,

    // ── new format (preferred) ──
    run_state: Option<RunState>,

    // ── legacy flat fields (fallback) ──
    nudge_count: Option<usize>,
    turn_tool_calls: Option<usize>,
    reasoning_only_strikes: Option<usize>,
    empty_response_strikes: Option<usize>,
}

impl From<RawAgentSession> for AgentSession {
    fn from(raw: RawAgentSession) -> Self {
        let mut run_state = raw.run_state.unwrap_or_else(|| RunState {
            nudge_count: raw.nudge_count.unwrap_or(0),
            turn_actions: raw.turn_tool_calls.unwrap_or(0),
            reasoning_only_strikes: raw.reasoning_only_strikes.unwrap_or(0),
            empty_response_strikes: raw.empty_response_strikes.unwrap_or(0),
            ..RunState::default()
        });
        // Pre-T7 wire kept the session total at the top level; migrate it into
        // the counting state (D5: one counting home). New-format payloads omit
        // the top-level key and keep the total already inside `run_state`.
        if let Some(total) = raw.total_tool_calls {
            run_state.total_actions = total;
        }
        Self {
            id: raw.id,
            chat_messages: raw.chat_messages,
            always_allowed_actions: raw.always_allowed_actions,
            run_state,
        }
    }
}

/// Stable identity + rolling state of a single chat thread.
///
/// `Default` returns a fresh session; load persisted state via serde.
/// Backward-compatible with the old flat-field format (pre-`RunState` migration).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(from = "RawAgentSession")]
pub struct AgentSession {
    id: Option<SessionId>,
    /// LLM API format messages, sent directly to the provider.
    /// This is the single source of truth for the conversation state.
    chat_messages: Vec<ChatMessage>,
    always_allowed_actions: HashSet<String>,
    /// Run-level state tracking for the react loop. Holds ALL counting state
    /// (D5): `run_state.total_actions` is the session-lifetime total middleware
    /// decisions like "first_turn_only" read.
    pub run_state: RunState,
}

impl AgentSession {
    pub fn new(id: SessionId) -> Self {
        Self {
            id: Some(id),
            chat_messages: Vec::new(),
            always_allowed_actions: HashSet::new(),
            run_state: RunState::default(),
        }
    }

    pub fn id(&self) -> Option<SessionId> {
        self.id.clone()
    }

    /// Derive a simplified `Vec<Message>` view from the canonical `chat_messages`.
    /// Assistant messages that contain only tool_calls (no text content) are
    /// skipped, since they have no corresponding simplified representation.
    pub fn simple_messages(&self) -> Vec<Message> {
        self.chat_messages
            .iter()
            .filter_map(|cm| match cm {
                ChatMessage::Assistant { content: None, .. } => None,
                ChatMessage::Assistant {
                    content: Some(c),
                    tool_calls: Some(tc),
                    ..
                } if c.is_empty() && !tc.is_empty() => None,
                _ => Some(Message::from(cm)),
            })
            .collect()
    }

    pub fn chat_messages(&self) -> &[ChatMessage] {
        &self.chat_messages
    }

    /// Mutable reference, for advanced scenarios that need to operate on messages directly.
    pub fn chat_messages_mut(&mut self) -> &mut Vec<ChatMessage> {
        &mut self.chat_messages
    }

    pub fn is_action_allowed(&self, action_key: &str) -> bool {
        self.always_allowed_actions.contains(action_key)
    }

    pub fn allow_action(&mut self, action_key: impl Into<String>) {
        self.always_allowed_actions.insert(action_key.into());
    }

    pub fn push_message(&mut self, role: MessageRole, content: impl Into<String>) {
        let content = content.into();
        let chat_msg = match role {
            MessageRole::System => ChatMessage::system(content),
            MessageRole::User => ChatMessage::user(content),
            MessageRole::Assistant => ChatMessage::assistant(content),
            MessageRole::Tool => ChatMessage::tool(String::new(), content),
        };
        self.chat_messages.push(chat_msg);
    }

    /// Push a message as ephemeral (user/system only): the LLM sees it this
    /// turn, then `remove_ephemeral_messages` strips it at turn end — from
    /// both memory and persistence.
    pub fn push_message_ephemeral(&mut self, role: MessageRole, content: impl Into<String>) {
        let content = content.into();
        let chat_msg = match role {
            MessageRole::System => ChatMessage::system_ephemeral(content),
            MessageRole::User => ChatMessage::user_ephemeral(content),
            MessageRole::Assistant | MessageRole::Tool => {
                // Ephemeral assistant/tool messages are not a thing; fall back
                // to a regular push rather than silently mis-marking.
                self.push_message(role, content);
                return;
            }
        };
        self.chat_messages.push(chat_msg);
    }

    /// Push machine-composed input on the user role (a tool-failure notice, a
    /// background-task wake): the model treats it like a user turn, but the
    /// synthetic flag keeps it out of the task trail context rotation carries
    /// into a fresh window.
    pub fn push_user_synthetic(&mut self, content: impl Into<String>) {
        self.chat_messages
            .push(ChatMessage::user_synthetic(content));
    }

    /// Replace the system prompt — the first **non-ephemeral** System message
    /// (wherever it sits, matching how `context` locates it); inserted at
    /// index 0 if none exists. All other messages are untouched. Hosts that
    /// own prompt composition (e.g. skill activation) re-bake through this.
    pub fn set_system_prompt(&mut self, content: impl Into<String>) {
        let content = content.into();
        if let Some(idx) = self.chat_messages.iter().position(|m| {
            matches!(
                m,
                ChatMessage::System {
                    ephemeral: false,
                    ..
                }
            )
        }) {
            self.chat_messages[idx] = ChatMessage::system(content);
        } else {
            self.chat_messages.insert(0, ChatMessage::system(content));
        }
    }

    /// Push an assistant message with reasoning/thinking content preserved.
    /// This allows the LLM to see its own prior reasoning in subsequent turns,
    /// preventing it from re-deriving the same conclusions every turn.
    pub fn push_assistant_with_reasoning(
        &mut self,
        content: impl Into<String>,
        reasoning: impl Into<String>,
    ) {
        self.chat_messages
            .push(ChatMessage::assistant_with_reasoning(content, reasoning));
    }

    pub fn push_user_message_with_images(
        &mut self,
        content: impl Into<String>,
        images: Vec<ImageAttachment>,
    ) {
        self.chat_messages
            .push(ChatMessage::user_with_images(content, images));
    }

    pub fn push_assistant_tool_call(
        &mut self,
        tool_call_id: &str,
        tool_name: &str,
        arguments_json: &str,
    ) {
        self.chat_messages.push(ChatMessage::assistant_tool_call(
            tool_call_id,
            tool_name,
            arguments_json,
        ));
    }

    /// Escape and truncate the arguments to their first `max_chars` characters, so the
    /// truncation WARN can display the raw content directly. Rust debug escaping keeps
    /// unclosed JSON, control characters, and CJK boundaries clearly distinguishable, so
    /// troubleshooting provider truncation in production no longer requires indirectly
    /// reconstructing the content.
    fn preview_args(args: &str, max_chars: usize) -> String {
        format!("{:?}", args.chars().take(max_chars).collect::<String>())
    }

    /// Render one assistant message carrying tool calls — the tool backend's
    /// history protocol (invalid-JSON sanitization included). Split out of
    /// [`Self::push_assistant_tool_calls`] so the backend can both push the
    /// message and carry it in `Observation::history_messages` for checkpoint
    /// replay (M3 seam).
    pub fn render_assistant_tool_calls(
        tool_calls: &[(String, String, String)],
        reasoning: Option<String>,
        content: Option<String>,
    ) -> ChatMessage {
        let calls: Vec<ToolCallMessage> = tool_calls
            .iter()
            .map(|(id, name, args)| {
                // Invalid JSON args (provider truncated the stream mid-generation)
                // are sanitized to an empty object. We deliberately do NOT wrap
                // them in a descriptive `{error, original_args_preview, message}`
                // object: "{}" is valid JSON so the next request is never rejected
                // with 400, and — critically — a rich error object here becomes an
                // imitation vector: the model replays it verbatim as the next
                // call's arguments and, because it parses, it slips past the react
                // truncation guard. The truncation explanation lives in the
                // paired tool_result, which is where the model reads feedback.
                let valid_args = if serde_json::from_str::<serde_json::Value>(args).is_ok() {
                    args.clone()
                } else {
                    tracing::warn!(
                        tool_name = %name,
                        args_len = args.len(),
                        args_preview = %Self::preview_args(args, 80),
                        "tool call arguments are not valid JSON (provider truncated them), \
                         sanitizing to empty object; re-issue instruction goes to the tool_result"
                    );
                    "{}".to_string()
                };
                ToolCallMessage {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: valid_args,
                }
            })
            .collect();
        ChatMessage::Assistant {
            content,
            reasoning_content: reasoning,
            tool_calls: Some(calls),
            thinking_signature: None,
        }
    }

    pub fn push_assistant_tool_calls(
        &mut self,
        tool_calls: &[(String, String, String)],
        reasoning: Option<String>,
        content: Option<String>,
    ) {
        self.push_chat_message(Self::render_assistant_tool_calls(
            tool_calls, reasoning, content,
        ));
    }

    /// Push one already-built message — the kernel's verbatim push for
    /// backend-rendered history ([`crate::types::Observation::history_messages`]).
    pub fn push_chat_message(&mut self, message: ChatMessage) {
        self.chat_messages.push(message);
    }

    pub fn push_tool_result(&mut self, tool_call_id: &str, content: impl Into<String>) {
        self.chat_messages
            .push(ChatMessage::tool(tool_call_id, content));
    }

    /// Remove all ephemeral messages (ephemeral=true).
    ///
    /// Called at the end of a turn to ensure injected ephemeral content does not linger into the next turn.
    pub fn remove_ephemeral_messages(&mut self) {
        let before = self.chat_messages.len();
        self.chat_messages.retain(|m| !m.is_ephemeral());
        let removed = before - self.chat_messages.len();
        if removed > 0 {
            tracing::debug!(
                removed,
                remaining = self.chat_messages.len(),
                "ephemeral messages cleaned up"
            );
        }
    }

    /// Count the number of conversation turns.
    /// A turn starts with a User message and includes subsequent Assistant/Tool messages.
    pub fn turn_count(&self) -> usize {
        self.chat_messages
            .iter()
            .filter(|m| matches!(m, ChatMessage::User { .. }))
            .count()
    }

    /// Remove the oldest turns from the front until turn count ≤ max_turns.
    /// Preserves the System message at index 0 if present.
    pub fn trim_oldest_turns(&mut self, max_turns: usize) {
        let current_turns = self.turn_count();
        if current_turns <= max_turns {
            return;
        }
        let turns_to_remove = current_turns - max_turns;

        // Find User message positions (turn boundaries) in chat_messages
        let user_positions: Vec<usize> = self
            .chat_messages
            .iter()
            .enumerate()
            .filter_map(|(i, m)| {
                if matches!(m, ChatMessage::User { .. }) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if user_positions.len() <= turns_to_remove {
            return;
        }

        // Preserve system prefix: count leading System messages
        let system_prefix = self
            .chat_messages
            .iter()
            .take_while(|m| matches!(m, ChatMessage::System { .. }))
            .count();

        // Drain from system_prefix up to the start of the (turns_to_remove + 1)-th turn
        let drain_end = user_positions[turns_to_remove];
        if system_prefix >= drain_end {
            return; // nothing to drain after system messages
        }

        self.chat_messages.drain(system_prefix..drain_end);
    }

    /// Remove the last message from `chat_messages`.
    /// Used by the max_message_tokens safety valve to discard oversized messages.
    pub fn pop_last_message(&mut self) {
        self.chat_messages.pop();
    }

    pub fn close_dangling_tool_calls(&mut self, error_summary: &str) {
        let assistant_idx = self.chat_messages.iter().rposition(
            |m| matches!(m, ChatMessage::Assistant { tool_calls: Some(tc), .. } if !tc.is_empty()),
        );

        let Some(assistant_idx) = assistant_idx else {
            return;
        };

        let ChatMessage::Assistant {
            tool_calls: Some(tc),
            ..
        } = &self.chat_messages[assistant_idx]
        else {
            return;
        };

        let all_ids: Vec<String> = tc.iter().map(|t| t.id.clone()).collect();

        let answered_ids: Vec<String> = self.chat_messages[assistant_idx + 1..]
            .iter()
            .filter_map(|m| match m {
                ChatMessage::Tool { tool_call_id, .. } => Some(tool_call_id.clone()),
                _ => None,
            })
            .collect();

        for id in &all_ids {
            if !answered_ids.iter().any(|a| a == id) {
                self.push_tool_result(id, error_summary);
            }
        }
    }

    /// Replace chat messages — for persistence restore and window compaction.
    /// Validates message sequence before replacing.
    ///
    /// Counters are NOT touched (D5): the counting source is explicit. A
    /// checkpoint restore trusts the persisted `RunState` (co-written with the
    /// checkpoint/journal — OV-11); only a messages-only restore — the
    /// old-format corpus — recounts via [`Self::recount_actions_from_messages`].
    /// Recounting here would also deflate the total after compaction and lie
    /// for restored histories that carry synthesized "已中断/未执行" results.
    pub fn set_chat_messages(&mut self, messages: Vec<ChatMessage>) -> Result<(), String> {
        validate_message_sequence(&messages)?;
        self.chat_messages = messages;
        Ok(())
    }

    /// Old-format compatibility (D5: 消息重算只做旧格式兼容): rebuild the
    /// session-lifetime action total from the history's assistant tool-call
    /// lens. Call ONLY on restores that carry no persisted counting state —
    /// the lens counts call attempts, not executed actions, so it is a
    /// fallback, never the primary source.
    pub fn recount_actions_from_messages(&mut self) {
        self.run_state.total_actions = self
            .chat_messages
            .iter()
            .filter_map(|m| match m {
                ChatMessage::Assistant {
                    tool_calls: Some(tc),
                    ..
                } => Some(tc.len()),
                _ => None,
            })
            .sum();
    }
}

/// Validate that a chat message sequence is well-formed for LLM API consumption.
///
/// Checks:
/// - At least one non-System/non-Custom message (System maps to the
///   top-level `system` parameter, Custom is stripped — a sequence of only
///   those leaves an empty `messages` array, which providers reject with
///   HTTP 400). Guards compaction outputs against producing an unsendable
///   window.
/// - No Tool message without a preceding Assistant with matching tool_call
/// - No duplicate Tool messages for the same tool_call_id
/// - All tool_calls in an Assistant batch must be answered before the next Assistant batch
/// - No unanswered tool calls at the end of the sequence
pub fn validate_message_sequence(messages: &[ChatMessage]) -> Result<(), String> {
    if !messages
        .iter()
        .any(|m| !matches!(m, ChatMessage::System { .. } | ChatMessage::Custom { .. }))
    {
        return Err(
            "sequence contains no sendable message: System/Custom alone leave the \
             provider `messages` array empty"
                .to_string(),
        );
    }

    let mut pending_tool_call_ids: HashSet<String> = HashSet::new();

    for (i, msg) in messages.iter().enumerate() {
        match msg {
            ChatMessage::Tool { tool_call_id, .. } => {
                if pending_tool_call_ids.is_empty() {
                    return Err(format!(
                        "message[{}]: Tool message with call_id '{}' has no preceding tool_call",
                        i, tool_call_id
                    ));
                }
                // Remove the ID on match — also detects duplicates (second remove returns false)
                if !pending_tool_call_ids.remove(tool_call_id) {
                    return Err(format!(
                        "message[{}]: Tool message with call_id '{}' does not match any pending tool_call (already answered or unknown)",
                        i, tool_call_id
                    ));
                }
            }
            ChatMessage::Assistant {
                tool_calls: Some(tc),
                ..
            } => {
                // Previous batch must be fully answered before a new batch starts
                if !pending_tool_call_ids.is_empty() {
                    return Err(format!(
                        "message[{}]: Assistant message with new tool_calls appears before pending calls were answered: {:?}",
                        i, pending_tool_call_ids
                    ));
                }
                pending_tool_call_ids = tc.iter().map(|t| t.id.clone()).collect();
            }
            _ => {}
        }
    }

    // All tool calls must be answered by the end of the sequence
    if !pending_tool_call_ids.is_empty() {
        return Err(format!(
            "message sequence ends with unanswered tool calls: {:?}",
            pending_tool_call_ids
        ));
    }

    Ok(())
}

#[cfg(test)]
fn make_session() -> AgentSession {
    AgentSession::new(SessionId::new(1))
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod validate_tests;

#[cfg(test)]
mod proptest_tests;
