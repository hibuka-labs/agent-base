//! Action-currency pure types (M3 ActionExecutor seam): Action, ActionFormat,
//! ActionOutcome, Observation, ObservationEvent, BatchStatus.
//!
//! One set of vocabulary for every action backend — tool calls today, CodeAct
//! cells tomorrow. These types are the persistable currency of checkpoints
//! (`BeforeActions`/`AfterActions` payloads) and the runtime currency of the
//! `ActionExecutor` seam, so they live here with zero runtime dependencies.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use llm_trait::message::ChatMessage;

/// Which protocol an action came from. The format label travels with every
/// [`Action`] so mixed backends stay distinguishable in persisted payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionFormat {
    /// A tool call in the OpenAI-style envelope (id, name, args).
    ToolCall,
    /// A CodeAct step (intent + code cell).
    CodeAct,
}

/// One unit of "something the model asked the runtime to do".
///
/// Replaces the bare `(id, name, args)` triple as the kernel currency; the
/// triple stays internal to the tool-call backend's executor. `identity` is
/// the counting fingerprint assigned at parse time (tool = name + canonical
/// args, code = canonical code) so limit middlewares count by semantic unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Action {
    /// Pairing id (tool_call_id / cell id).
    pub id: String,
    /// Tool name, or the action kind label for non-tool backends.
    pub name: String,
    /// Action arguments. Tool calls carry the provider envelope's argument
    /// text **byte-for-byte** as a JSON string (`Value::String`) — it may be
    /// truncated non-JSON, and the history protocol must echo it exactly
    /// (sanitized downstream). Structured backends use an object, e.g.
    /// `{"code": …}` for cells. Use [`Action::raw_args`] to read the text.
    pub args: Value,
    /// Counting fingerprint computed at parse time by the backend.
    pub identity: String,
    /// Backend format label.
    pub format: ActionFormat,
}

impl Action {
    /// Build a tool-call action from the provider envelope. The argument text is
    /// kept byte-for-byte (`Value::String`); `identity` is the counting
    /// fingerprint ([`Action::tool_call_identity`]).
    pub fn from_tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        args: impl Into<String>,
    ) -> Self {
        let name = name.into();
        let args = args.into();
        Self {
            id: id.into(),
            identity: Self::tool_call_identity(&name, &args),
            name,
            args: Value::String(args),
            format: ActionFormat::ToolCall,
        }
    }

    /// Counting fingerprint for a tool call: name + canonical (key-sorted) args.
    /// The single source of the identity format — limit middlewares count by
    /// this same unit.
    pub fn tool_call_identity(name: &str, args: &str) -> String {
        let canonical = serde_json::from_str::<Value>(args)
            .map(|v| v.to_string())
            .unwrap_or_else(|_| args.to_string());
        format!("{name}::{canonical}")
    }

    /// Build a CodeAct cell action (intent + code). The code is the counting
    /// fingerprint verbatim (design D5: 代码=canonical code — a cell has no
    /// arguments to canonicalize, the source text *is* the semantic unit).
    /// `args` carries the engine's own block shape `{"intent", "code"}` — one
    /// vocabulary with the codexec host protocol.
    pub fn from_code_cell(
        id: impl Into<String>,
        intent: impl Into<String>,
        code: impl Into<String>,
    ) -> Self {
        let intent = intent.into();
        let code = code.into();
        Self {
            id: id.into(),
            name: "code_cell".into(),
            identity: code.clone(),
            args: serde_json::json!({ "intent": intent, "code": code }),
            format: ActionFormat::CodeAct,
        }
    }

    /// The action's argument text: string payloads verbatim (tool calls),
    /// anything else rendered as compact JSON (structured payloads).
    pub fn raw_args(&self) -> std::borrow::Cow<'_, str> {
        match &self.args {
            Value::String(s) => std::borrow::Cow::Borrowed(s.as_str()),
            other => std::borrow::Cow::Owned(other.to_string()),
        }
    }
}

/// Per-action result — the model-visible fate of one [`Action`] (D9: a policy
/// refusal lands here, never as a silent drop). Batch-level failure is
/// `ExecError`; "something went wrong along the way" is an outcome, not an
/// error.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ActionOutcome {
    /// Matches [`Action::id`] for pairing.
    pub action_id: String,
    /// Action name (tool name / kind label).
    pub name: String,
    /// Whether the action completed as intended.
    pub ok: bool,
    /// `true` when the failure was an approval denial (not an execution error).
    #[serde(default)]
    pub denied: bool,
    /// Model-visible summary (result text, or why it was refused).
    pub summary: String,
    /// Machine-readable metadata for UI consumers (never sent to the LLM).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub details: Option<Value>,
}

/// The tax-3 channel: observations become events, big payloads stay behind a
/// reference (D9: no `Error` variant — refusals and interceptions live in
/// [`ActionOutcome`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ObservationEvent {
    /// Inline text.
    Text { text: String },
    /// A captured artifact referenced by id; body is fetched on demand.
    Artifact {
        capture_id: String,
        kind: String,
        summary: String,
    },
}

/// Batch-level health of an execution round (D15).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    /// Every action ran to a fate (success, failure, or denial).
    Completed,
    /// The execution environment died mid-batch; completed work is preserved.
    EnvDied,
    /// The batch was cut off (crash window) and its fates were reconstructed
    /// at resume (D3 observation replay) — synthetic, not observed.
    Interrupted,
}

/// What the runtime observed after executing a batch of actions (D15: always
/// returned once anything ran — `ExecError` is reserved for "nothing ran").
///
/// `history_messages` is the BACKEND's protocol rendering; the kernel pushes
/// them verbatim and never interprets them.
// ChatMessage (llm-trait) has no PartialEq — compare observations via
// canonical JSON, the family convention (see checkpoint tests).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    /// One entry per action, in submission order (ids pair with the batch).
    pub outcomes: Vec<ActionOutcome>,
    /// Tax-3 event channel (typed empty until M3c fills it).
    pub events: Vec<ObservationEvent>,
    /// Backend-rendered history messages to push as-is.
    pub history_messages: Vec<ChatMessage>,
    /// Batch health.
    pub batch_status: BatchStatus,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_actions() -> Vec<Action> {
        vec![
            Action {
                id: "call_1".into(),
                name: "weather".into(),
                // Tool calls carry the envelope's raw argument text.
                args: Value::String(r#"{"city": "Tokyo"}"#.into()),
                identity: r#"weather::{"city":"Tokyo"}"#.into(),
                format: ActionFormat::ToolCall,
            },
            Action {
                id: "cell_1".into(),
                name: "code_cell".into(),
                args: serde_json::json!({"code": "print(1)"}),
                identity: "print(1)".into(),
                format: ActionFormat::CodeAct,
            },
        ]
    }

    #[test]
    fn action_raw_args_is_byte_faithful_for_tool_calls() {
        let actions = sample_actions();
        // The tool-call payload is echoed exactly (whitespace included).
        assert_eq!(actions[0].raw_args(), r#"{"city": "Tokyo"}"#);
        // Structured payloads render as compact JSON.
        assert_eq!(actions[1].raw_args(), r#"{"code":"print(1)"}"#);
    }

    #[test]
    fn from_tool_call_builds_byte_faithful_action_with_identity() {
        let a = Action::from_tool_call("call_1", "weather", r#"{"city": "Tokyo"}"#);
        assert_eq!(a.id, "call_1");
        assert_eq!(a.name, "weather");
        assert_eq!(a.args, Value::String(r#"{"city": "Tokyo"}"#.into()));
        // Identity canonicalizes (compacts) the args.
        assert_eq!(a.identity, r#"weather::{"city":"Tokyo"}"#);
        assert_eq!(a.format, ActionFormat::ToolCall);
    }

    #[test]
    fn from_code_cell_carries_the_engine_block_shape() {
        let a = Action::from_code_cell("cell_0", "bind x", "x = 1");
        assert_eq!(a.id, "cell_0");
        assert_eq!(a.name, "code_cell");
        // The block shape is the codexec host vocabulary: {"intent", "code"}.
        assert_eq!(
            a.args,
            serde_json::json!({"intent": "bind x", "code": "x = 1"})
        );
        // The code is the counting fingerprint, verbatim.
        assert_eq!(a.identity, "x = 1");
        assert_eq!(a.format, ActionFormat::CodeAct);
    }

    #[test]
    fn tool_call_identity_canonicalizes_key_order_only() {
        assert_eq!(
            Action::tool_call_identity("t", r#"{"a":1,"b":2}"#),
            Action::tool_call_identity("t", r#"{"b":2,"a":1}"#),
        );
        // Unparseable args fall back to the raw text.
        assert_eq!(Action::tool_call_identity("t", "not-json"), "t::not-json");
    }

    #[test]
    fn action_round_trips() {
        for action in sample_actions() {
            let json = serde_json::to_string(&action).unwrap();
            let decoded: Action = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded, action);
        }
    }

    #[test]
    fn action_format_serializes_snake_case() {
        let json = serde_json::to_string(&ActionFormat::ToolCall).unwrap();
        assert_eq!(json, "\"tool_call\"");
        let json = serde_json::to_string(&ActionFormat::CodeAct).unwrap();
        assert_eq!(json, "\"code_act\"");
    }

    #[test]
    fn observation_round_trips_with_all_parts() {
        let obs = Observation {
            outcomes: vec![ActionOutcome {
                action_id: "call_1".into(),
                name: "weather".into(),
                ok: true,
                denied: false,
                summary: "20C, clear".into(),
                details: Some(serde_json::json!({"lines": 3})),
            }],
            events: vec![
                ObservationEvent::Text {
                    text: "ran fine".into(),
                },
                ObservationEvent::Artifact {
                    capture_id: "cap_9".into(),
                    kind: "stdout".into(),
                    summary: "1.2 KB stdout".into(),
                },
            ],
            history_messages: vec![ChatMessage::tool("call_1", "20C, clear")],
            batch_status: BatchStatus::Completed,
        };
        let json = serde_json::to_string(&obs).unwrap();
        let decoded: Observation = serde_json::from_str(&json).unwrap();
        // ChatMessage has no PartialEq, so compare canonical JSON.
        assert_eq!(
            serde_json::to_string(&decoded).unwrap(),
            serde_json::to_string(&obs).unwrap()
        );
    }

    #[test]
    fn outcome_details_optional_field_is_backward_compatible() {
        // Old payload without `details` must still deserialize (serde default).
        let old =
            r#"{"action_id":"call_1","name":"weather","ok":true,"denied":false,"summary":"20C"}"#;
        let outcome: ActionOutcome = serde_json::from_str(old).unwrap();
        assert_eq!(outcome.details, None);
    }

    #[test]
    fn batch_status_serializes_snake_case() {
        let json = serde_json::to_string(&BatchStatus::EnvDied).unwrap();
        assert_eq!(json, "\"env_died\"");
    }
}
