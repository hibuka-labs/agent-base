//! Checkpoint-related pure types: CheckpointData, CheckpointStep, ToolResultData.
//!
//! M3 seam: `Before/AfterActions` carry the action vocabulary (`Vec<Action>`)
//! and, on the way out, the backend's [`Observation`] (D3 observation replay —
//! resume rebuilds history from it and never re-executes). Pre-M3 payloads
//! (`Before/AfterToolCalls` with `(id, name, args)` triples) are read by the
//! T5 migration layer, not by serde aliases: the field shapes differ.
//!
//! Persistence contract (OV-5+11):
//! - **Versioned** — native writes carry [`CHECKPOINT_VERSION`]; a payload
//!   without a `version` field is the pre-versioning corpus and reads as 0.
//!   A payload newer than this build is refused explicitly (拒绝时明说).
//! - **Backend affinity** — `backend` pins the action protocol the payload
//!   speaks; [`CheckpointData::verify_for_resume`] refuses cross-backend
//!   resumes and label mismatches, spelling the mismatch out.
//! - **Source marks** — data converted on read keeps a [`MigrationSource`]
//!   mark: read what is on the wire, never guess, never backfill (不猜不回填).

use llm_trait::message::ChatMessage;
use serde::{Deserialize, Deserializer, Serialize, de};
use serde_json::Value;

use crate::action::{Action, ActionFormat, Observation};
use crate::run_state::RunState;
use crate::session::SessionId;

/// The checkpoint payload version this build writes and understands. Payloads
/// without a `version` field are pre-versioning and read as 0.
pub const CHECKPOINT_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize)]
pub struct CheckpointData {
    pub session_id: SessionId,
    pub user_input: String,
    pub step: CheckpointStep,
    pub turn_count: u32,
    /// Payload version ([`CHECKPOINT_VERSION`] on native writes; 0 for the
    /// pre-versioning corpus, which had no such field). Never backfilled.
    pub version: u32,
    /// Backend affinity pin (OV-5+11): the action protocol this payload
    /// speaks. Absent on the wire means the pre-labeling tool-call corpus —
    /// the step shape itself is the evidence.
    pub backend: ActionFormat,
    /// Source mark set when the payload was converted on read (T5 migration
    /// layer). Absent for natively written data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub migrated_from: Option<MigrationSource>,
    /// The counting evidence (D5+OV-11): co-written with the step in the one
    /// journal line, so the recorded batch and the counts can never disagree
    /// across a crash window. Native writes ALWAYS carry it (`new` stores
    /// `Some`); on the wire it is optional so the pre-T7 corpus (and converted
    /// legacy payloads) stay readable with the field genuinely absent —
    /// restores of those fall back to the old-format paths, never guessed
    /// (不猜不回填).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_state: Option<RunState>,
}

impl CheckpointData {
    /// Native checkpoint of [`CHECKPOINT_VERSION`], pinned to `backend`, with
    /// the counting state co-written (D5+OV-11 事务同写).
    pub fn new(
        session_id: SessionId,
        user_input: impl Into<String>,
        step: CheckpointStep,
        turn_count: u32,
        backend: ActionFormat,
        run_state: RunState,
    ) -> Self {
        Self {
            session_id,
            user_input: user_input.into(),
            step,
            turn_count,
            version: CHECKPOINT_VERSION,
            backend,
            migrated_from: None,
            run_state: Some(run_state),
        }
    }

    /// The T5 persistence contract, checked before resume: the checkpoint must
    /// speak the resuming backend's protocol, and every action payload must
    /// carry a matching format label. Refusals spell out the mismatch.
    pub fn verify_for_resume(&self, expected: ActionFormat) -> Result<(), String> {
        if self.version > CHECKPOINT_VERSION {
            return Err(format!(
                "resume refused: checkpoint version {} is newer than this build supports ({CHECKPOINT_VERSION})",
                self.version
            ));
        }
        if self.backend != expected {
            return Err(format!(
                "resume refused: checkpoint is pinned to the {:?} backend, this runtime resumes {:?} checkpoints",
                self.backend, expected
            ));
        }
        self.verify_action_labels()
    }

    /// The same contract over the set of formats one runtime can resume
    /// (T10c): a session may interleave built-in tool-call turns with a
    /// mounted backend's turns, so the pin check accepts every format the
    /// runtime is assembled with — and refuses a pin naming a backend it does
    /// not carry ("mount it to resume this session"), never a silent replay
    /// through the wrong protocol.
    pub fn verify_for_resume_any(&self, accepted: &[ActionFormat]) -> Result<(), String> {
        if self.version > CHECKPOINT_VERSION {
            return Err(format!(
                "resume refused: checkpoint version {} is newer than this build supports ({CHECKPOINT_VERSION})",
                self.version
            ));
        }
        if !accepted.contains(&self.backend) {
            return Err(format!(
                "resume refused: checkpoint is pinned to the {:?} backend, this runtime resumes {:?} checkpoints",
                self.backend, accepted
            ));
        }
        self.verify_action_labels()
    }

    /// Every action payload must carry the pin's format label.
    fn verify_action_labels(&self) -> Result<(), String> {
        for action in self.step.actions() {
            if action.format != self.backend {
                return Err(format!(
                    "resume refused: checkpoint action '{}' carries format label {:?} but the checkpoint is pinned to {:?}",
                    action.id, action.format, self.backend
                ));
            }
        }
        Ok(())
    }
}

/// `CheckpointData` is read through the T5 migration layer: native shapes
/// first, then the pre-M3 shapes (converted, marked). Deserialize is manual so
/// the version/backend/source-mark contract lives in one place — Serialize
/// stays derived and always writes the native shape.
impl<'de> Deserialize<'de> for CheckpointData {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            session_id: SessionId,
            user_input: String,
            step: Value,
            turn_count: u32,
            #[serde(default)]
            version: Option<u32>,
            #[serde(default)]
            backend: Option<ActionFormat>,
            #[serde(default)]
            migrated_from: Option<MigrationSource>,
            #[serde(default)]
            run_state: Option<RunState>,
        }

        let wire = Wire::deserialize(deserializer)?;
        // Absent version = pre-versioning corpus (0). Never backfilled upward.
        let version = wire.version.unwrap_or(0);
        if version > CHECKPOINT_VERSION {
            return Err(de::Error::custom(format!(
                "checkpoint version {version} is newer than this build supports ({CHECKPOINT_VERSION}); refusing to read"
            )));
        }

        let (step, migrated_from) = match serde_json::from_value::<CheckpointStep>(
            wire.step.clone(),
        ) {
            Ok(step) => (step, wire.migrated_from),
            Err(native_err) => match serde_json::from_value::<LegacyStep>(wire.step) {
                Ok(legacy) => {
                    let (step, legacy_results) = legacy.into_step();
                    let mark = wire.migrated_from.unwrap_or(MigrationSource {
                        kind: MigrationKind::ToolCallTriplesV0,
                        legacy_results,
                    });
                    (step, Some(mark))
                }
                Err(_) => {
                    return Err(de::Error::custom(format!(
                        "unrecognized checkpoint step payload (not a native or pre-M3 shape): {native_err}"
                    )));
                }
            },
        };

        Ok(CheckpointData {
            session_id: wire.session_id,
            user_input: wire.user_input,
            step,
            turn_count: wire.turn_count,
            version,
            // Absent label = the pre-labeling tool-call corpus (the triples on
            // the wire are the evidence).
            backend: wire.backend.unwrap_or(ActionFormat::ToolCall),
            migrated_from,
            // Absent counting state = pre-T7 corpus (or a converted legacy
            // payload). Restores of those fall back to the old-format paths.
            run_state: wire.run_state,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CheckpointStep {
    AfterUserInput,
    BeforeLlm {
        messages: Vec<ChatMessage>,
        tools: Vec<Value>,
    },
    /// About to execute a batch of actions (id-paired to the observation).
    BeforeActions {
        actions: Vec<Action>,
    },
    /// The batch ran; `observation` is what the backend observed (D3).
    AfterActions {
        actions: Vec<Action>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        observation: Option<Observation>,
    },
}

impl CheckpointStep {
    /// The batch this step pins (empty for non-action steps).
    pub fn actions(&self) -> &[Action] {
        match self {
            CheckpointStep::AfterUserInput | CheckpointStep::BeforeLlm { .. } => &[],
            CheckpointStep::BeforeActions { actions }
            | CheckpointStep::AfterActions { actions, .. } => actions,
        }
    }
}

/// Provenance mark for checkpoint data converted on read (T5 migration layer):
/// the payload was written by a pre-M3 binary in an older wire shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MigrationSource {
    /// Which pre-M3 wire shape was converted.
    pub kind: MigrationKind,
    /// `AfterToolCalls { results }` records kept verbatim. The pre-M3 payload
    /// recorded no per-action fates, so nothing is guessed from them and no
    /// observation is backfilled (不猜不回填).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub legacy_results: Vec<ToolResultData>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationKind {
    /// Pre-M3 `Before/AfterToolCalls` steps: `(id, name, args)` triples.
    ToolCallTriplesV0,
}

/// The pre-M3 checkpoint step wire shape (tool-call triples). Read-only
/// migration input — never constructed, never serialized.
#[derive(Deserialize)]
enum LegacyStep {
    BeforeToolCalls {
        tool_calls: Vec<(String, String, String)>,
    },
    AfterToolCalls {
        tool_calls: Vec<(String, String, String)>,
        #[serde(default)]
        results: Vec<ToolResultData>,
    },
}

impl LegacyStep {
    /// Convert to the action vocabulary. Nothing is guessed: triples become
    /// tool-call actions (labels and identity computed by the one constructor),
    /// and the pre-M3 `results` ride along as-is instead of being pressed into
    /// an `Observation` (the payload recorded no fates).
    fn into_step(self) -> (CheckpointStep, Vec<ToolResultData>) {
        fn actions(triples: Vec<(String, String, String)>) -> Vec<Action> {
            triples
                .into_iter()
                .map(|(id, name, args)| Action::from_tool_call(id, name, args))
                .collect()
        }
        match self {
            LegacyStep::BeforeToolCalls { tool_calls } => (
                CheckpointStep::BeforeActions {
                    actions: actions(tool_calls),
                },
                Vec::new(),
            ),
            LegacyStep::AfterToolCalls {
                tool_calls,
                results,
            } => (
                CheckpointStep::AfterActions {
                    actions: actions(tool_calls),
                    observation: None,
                },
                results,
            ),
        }
    }
}

/// Legacy per-result record from the pre-M3 checkpoint payload. Kept so the
/// T5 migration layer can read old `AfterToolCalls { results }` data.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolResultData {
    pub tool_call_id: String,
    pub tool_name: String,
    pub summary: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::{ActionFormat, ActionOutcome, BatchStatus};

    fn sample_messages() -> Vec<ChatMessage> {
        vec![
            ChatMessage::system("You are a helpful agent."),
            ChatMessage::user("What is the weather in Tokyo?"),
        ]
    }

    fn sample_tools() -> Vec<Value> {
        vec![serde_json::json!({"name": "weather", "args": {"city": "Tokyo"}})]
    }

    fn sample_actions() -> Vec<Action> {
        vec![
            Action {
                id: "call_1".to_string(),
                name: "weather".to_string(),
                args: Value::String(r#"{"city": "Tokyo"}"#.to_string()),
                identity: r#"weather::{"city":"Tokyo"}"#.to_string(),
                format: ActionFormat::ToolCall,
            },
            Action {
                id: "call_2".to_string(),
                name: "time".to_string(),
                args: Value::String(r#"{"tz": "Asia/Tokyo"}"#.to_string()),
                identity: r#"time::{"tz":"Asia/Tokyo"}"#.to_string(),
                format: ActionFormat::ToolCall,
            },
        ]
    }

    fn sample_observation() -> Observation {
        Observation {
            outcomes: vec![
                ActionOutcome {
                    action_id: "call_1".to_string(),
                    name: "weather".to_string(),
                    ok: true,
                    denied: false,
                    summary: "20C, clear".to_string(),
                    details: None,
                },
                ActionOutcome {
                    action_id: "call_2".to_string(),
                    name: "time".to_string(),
                    ok: true,
                    denied: false,
                    summary: "09:41 JST".to_string(),
                    details: None,
                },
            ],
            events: vec![],
            history_messages: vec![ChatMessage::tool("call_1", "20C, clear")],
            batch_status: BatchStatus::Completed,
        }
    }

    fn checkpoint_data(step: CheckpointStep) -> CheckpointData {
        CheckpointData::new(
            SessionId::with_external_id(42, "test-session"),
            "What is the weather in Tokyo?",
            step,
            3,
            ActionFormat::ToolCall,
            RunState::default(),
        )
    }

    #[test]
    fn checkpoint_step_after_user_input_round_trips() {
        let step = CheckpointStep::AfterUserInput;
        let json = serde_json::to_string(&step).unwrap();
        let decoded: CheckpointStep = serde_json::from_str(&json).unwrap();
        assert!(matches!(decoded, CheckpointStep::AfterUserInput));
    }

    #[test]
    fn checkpoint_step_before_llm_round_trips() {
        let step = CheckpointStep::BeforeLlm {
            messages: sample_messages(),
            tools: sample_tools(),
        };
        let json = serde_json::to_string(&step).unwrap();
        let decoded: CheckpointStep = serde_json::from_str(&json).unwrap();
        match decoded {
            CheckpointStep::BeforeLlm { messages, tools } => {
                // ChatMessage has no PartialEq, so compare canonical JSON.
                assert_eq!(
                    serde_json::to_string(&messages).unwrap(),
                    serde_json::to_string(&sample_messages()).unwrap()
                );
                assert_eq!(tools, sample_tools());
            }
            other => panic!("expected BeforeLlm, got {other:?}"),
        }
    }

    #[test]
    fn checkpoint_step_before_actions_round_trips() {
        let step = CheckpointStep::BeforeActions {
            actions: sample_actions(),
        };
        let json = serde_json::to_string(&step).unwrap();
        let decoded: CheckpointStep = serde_json::from_str(&json).unwrap();
        match decoded {
            CheckpointStep::BeforeActions { actions } => {
                assert_eq!(actions, sample_actions());
            }
            other => panic!("expected BeforeActions, got {other:?}"),
        }
    }

    #[test]
    fn checkpoint_step_after_actions_round_trips() {
        let step = CheckpointStep::AfterActions {
            actions: sample_actions(),
            observation: Some(sample_observation()),
        };
        let json = serde_json::to_string(&step).unwrap();
        let decoded: CheckpointStep = serde_json::from_str(&json).unwrap();
        match decoded {
            CheckpointStep::AfterActions {
                actions,
                observation,
            } => {
                assert_eq!(actions, sample_actions());
                let observation = observation.expect("observation must survive the round trip");
                assert_eq!(observation.outcomes.len(), 2);
                assert_eq!(observation.outcomes[0].action_id, "call_1");
                assert_eq!(observation.outcomes[0].name, "weather");
                assert_eq!(observation.outcomes[0].summary, "20C, clear");
                assert_eq!(observation.outcomes[1].action_id, "call_2");
                assert_eq!(observation.outcomes[1].name, "time");
                assert_eq!(observation.outcomes[1].summary, "09:41 JST");
                assert_eq!(observation.batch_status, BatchStatus::Completed);
                // ChatMessage has no PartialEq — canonical JSON comparison.
                assert_eq!(
                    serde_json::to_string(&observation.history_messages).unwrap(),
                    serde_json::to_string(&sample_observation().history_messages).unwrap()
                );
            }
            other => panic!("expected AfterActions, got {other:?}"),
        }
    }

    #[test]
    fn checkpoint_step_after_actions_observation_is_optional() {
        // A batch checkpoint written before the observation was available
        // still deserializes (T5 migration fills what it can).
        let step = CheckpointStep::AfterActions {
            actions: sample_actions(),
            observation: None,
        };
        let json = serde_json::to_string(&step).unwrap();
        assert!(!json.contains("observation"));
        let decoded: CheckpointStep = serde_json::from_str(&json).unwrap();
        match decoded {
            CheckpointStep::AfterActions { observation, .. } => {
                assert!(observation.is_none());
            }
            other => panic!("expected AfterActions, got {other:?}"),
        }
    }

    #[test]
    fn checkpoint_data_round_trips_for_every_step_variant() {
        let variants = [
            CheckpointStep::AfterUserInput,
            CheckpointStep::BeforeLlm {
                messages: sample_messages(),
                tools: sample_tools(),
            },
            CheckpointStep::BeforeActions {
                actions: sample_actions(),
            },
            CheckpointStep::AfterActions {
                actions: sample_actions(),
                observation: Some(sample_observation()),
            },
        ];

        for step in variants {
            let data = checkpoint_data(step);
            let json = serde_json::to_string(&data).unwrap();
            let decoded: CheckpointData = serde_json::from_str(&json).unwrap();
            assert_eq!(decoded.session_id, data.session_id);
            assert_eq!(decoded.user_input, data.user_input);
            assert_eq!(decoded.turn_count, data.turn_count);
            // CheckpointStep does not derive PartialEq, so compare variant
            // discriminants for the step round-trip.
            assert_eq!(
                std::mem::discriminant(&data.step),
                std::mem::discriminant(&decoded.step)
            );
        }
    }

    #[test]
    fn checkpoint_step_deserialization_fails_gracefully_on_malformed_input() {
        // Unknown variant tag.
        assert!(serde_json::from_str::<CheckpointStep>(r#"{"NotAStep":{}}"#).is_err());
        // Wrong field type on a known variant.
        assert!(
            serde_json::from_str::<CheckpointStep>(
                r#"{"BeforeLlm":{"messages":"not-a-list","tools":[]}}"#,
            )
            .is_err()
        );
        // Truncated JSON.
        assert!(serde_json::from_str::<CheckpointStep>(r#"{"AfterUserInput""#).is_err());
        // Empty input.
        assert!(serde_json::from_str::<CheckpointStep>("").is_err());
    }

    #[test]
    fn checkpoint_data_deserialization_fails_gracefully_on_malformed_input() {
        // Missing required fields.
        assert!(
            serde_json::from_str::<CheckpointData>(r#"{"session_id":{"id":1},"user_input":"hi"}"#,)
                .is_err()
        );
        // Wrong session_id shape.
        assert!(
            serde_json::from_str::<CheckpointData>(
                r#"{"session_id":"oops","user_input":"hi","step":"AfterUserInput","turn_count":1}"#,
            )
            .is_err()
        );
        // turn_count is not a number.
        assert!(
            serde_json::from_str::<CheckpointData>(
                r#"{"session_id":{"id":1},"user_input":"hi","step":"AfterUserInput","turn_count":"three"}"#,
            )
            .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // T5 persistence contract: versioning, source marks, backend affinity.
    // -----------------------------------------------------------------------

    #[test]
    fn native_checkpoint_pins_version_and_backend() {
        let data = checkpoint_data(CheckpointStep::AfterUserInput);
        assert_eq!(data.version, CHECKPOINT_VERSION);
        assert_eq!(data.backend, ActionFormat::ToolCall);
        assert!(data.migrated_from.is_none());

        let json = serde_json::to_string(&data).unwrap();
        assert!(
            json.contains("\"version\":1"),
            "native write carries version: {json}"
        );
        assert!(
            json.contains("\"backend\":\"tool_call\""),
            "native write carries the format label: {json}"
        );
        let decoded: CheckpointData = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.version, CHECKPOINT_VERSION);
        assert_eq!(decoded.backend, ActionFormat::ToolCall);
        assert!(decoded.migrated_from.is_none());
    }

    #[test]
    fn pre_versioning_payload_reads_as_version_zero_without_a_mark() {
        // A pre-M3 `AfterUserInput`/`BeforeLlm` payload is shape-identical to
        // the native one — nothing migrated, so nothing is marked (不猜).
        let json = r#"{"session_id":{"id":1,"external_id":"s1"},"user_input":"hi","step":"AfterUserInput","turn_count":1}"#;
        let decoded: CheckpointData = serde_json::from_str(json).unwrap();
        assert_eq!(decoded.version, 0, "absent version must not be backfilled");
        assert_eq!(decoded.backend, ActionFormat::ToolCall);
        assert!(decoded.migrated_from.is_none());
    }

    #[test]
    fn legacy_before_tool_calls_migrates_with_source_mark() {
        // A checkpoint line written by a pre-M3 binary: tool-call triples, no
        // version/backend fields.
        let json = r#"{"session_id":{"id":7,"external_id":"s7"},"user_input":"resume me","step":{"BeforeToolCalls":{"tool_calls":[["call_1","echo","{\"text\":\"hi\"}"]]}},"turn_count":2}"#;
        let decoded: CheckpointData = serde_json::from_str(json).unwrap();
        assert_eq!(decoded.version, 0);
        assert_eq!(decoded.backend, ActionFormat::ToolCall);
        let mark = decoded
            .migrated_from
            .as_ref()
            .expect("legacy shape must carry its source mark");
        assert_eq!(mark.kind, MigrationKind::ToolCallTriplesV0);
        assert!(mark.legacy_results.is_empty());
        match &decoded.step {
            CheckpointStep::BeforeActions { actions } => {
                assert_eq!(actions.len(), 1);
                assert_eq!(actions[0].id, "call_1");
                assert_eq!(actions[0].name, "echo");
                assert_eq!(actions[0].raw_args(), r#"{"text":"hi"}"#);
                assert_eq!(actions[0].format, ActionFormat::ToolCall);
                assert_eq!(
                    actions[0].identity,
                    Action::tool_call_identity("echo", r#"{"text":"hi"}"#)
                );
            }
            other => panic!("expected BeforeActions, got {other:?}"),
        }
    }

    #[test]
    fn legacy_after_tool_calls_migrates_without_backfilling_an_observation() {
        let json = r#"{"session_id":{"id":8,"external_id":"s8"},"user_input":"go","step":{"AfterToolCalls":{"tool_calls":[["call_2","ls","{}"]],"results":[]}},"turn_count":3}"#;
        let decoded: CheckpointData = serde_json::from_str(json).unwrap();
        let mark = decoded
            .migrated_from
            .as_ref()
            .expect("legacy shape must carry its source mark");
        assert_eq!(mark.kind, MigrationKind::ToolCallTriplesV0);
        assert!(mark.legacy_results.is_empty());
        match &decoded.step {
            CheckpointStep::AfterActions {
                actions,
                observation,
            } => {
                assert_eq!(actions.len(), 1);
                assert_eq!(actions[0].id, "call_2");
                assert!(
                    observation.is_none(),
                    "nothing observed was recorded — no backfill"
                );
            }
            other => panic!("expected AfterActions, got {other:?}"),
        }
    }

    #[test]
    fn legacy_after_tool_calls_results_ride_along_verbatim() {
        let json = r#"{"session_id":{"id":8,"external_id":"s8"},"user_input":"go","step":{"AfterToolCalls":{"tool_calls":[["call_2","ls","{}"]],"results":[{"tool_call_id":"call_2","tool_name":"ls","summary":"file list"}]}},"turn_count":3}"#;
        let decoded: CheckpointData = serde_json::from_str(json).unwrap();
        let mark = decoded
            .migrated_from
            .as_ref()
            .expect("legacy shape must carry its source mark");
        assert_eq!(
            mark.legacy_results,
            vec![ToolResultData {
                tool_call_id: "call_2".to_string(),
                tool_name: "ls".to_string(),
                summary: "file list".to_string(),
            }],
            "legacy results are preserved as-is, not guessed into fates"
        );
        match &decoded.step {
            CheckpointStep::AfterActions { observation, .. } => {
                assert!(
                    observation.is_none(),
                    "results ≠ observation (no fates recorded)"
                );
            }
            other => panic!("expected AfterActions, got {other:?}"),
        }
    }

    #[test]
    fn legacy_payload_unknown_fields_are_tolerated() {
        let json = r#"{"session_id":{"id":9,"external_id":"s9"},"user_input":"hi","step":{"BeforeToolCalls":{"tool_calls":[["call_1","echo","{}"]],"future_field":true}},"turn_count":1,"some_new_top_level_field":"ignored"}"#;
        let decoded: CheckpointData = serde_json::from_str(json).unwrap();
        assert_eq!(decoded.version, 0);
        assert!(decoded.migrated_from.is_some());
        assert!(matches!(decoded.step, CheckpointStep::BeforeActions { .. }));
    }

    #[test]
    fn migrated_checkpoint_re_serializes_in_the_native_shape_and_keeps_its_mark() {
        let json = r#"{"session_id":{"id":8,"external_id":"s8"},"user_input":"go","step":{"AfterToolCalls":{"tool_calls":[["call_2","ls","{}"]],"results":[{"tool_call_id":"call_2","tool_name":"ls","summary":"file list"}]}},"turn_count":3}"#;
        let decoded: CheckpointData = serde_json::from_str(json).unwrap();
        let rewritten = serde_json::to_string(&decoded).unwrap();
        assert!(
            rewritten.contains("\"AfterActions\""),
            "rewrite is native-shape: {rewritten}"
        );
        assert!(!rewritten.contains("AfterToolCalls"));
        assert!(
            rewritten.contains("\"migrated_from\""),
            "the source mark survives persistence: {rewritten}"
        );
        assert!(rewritten.contains("\"legacy_results\""));

        let again: CheckpointData = serde_json::from_str(&rewritten).unwrap();
        assert_eq!(
            again.version, 0,
            "the mark and version keep agreeing across rounds"
        );
        let mark = again.migrated_from.expect("mark survives the round trip");
        assert_eq!(mark.kind, MigrationKind::ToolCallTriplesV0);
        assert_eq!(mark.legacy_results.len(), 1);
        assert_eq!(mark.legacy_results[0].summary, "file list");
    }

    #[test]
    fn future_version_payload_is_refused_explicitly() {
        let json = r#"{"session_id":{"id":1,"external_id":"s1"},"user_input":"hi","step":"AfterUserInput","turn_count":1,"version":2}"#;
        let err = serde_json::from_str::<CheckpointData>(json).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("version 2"),
            "refusal names the version: {msg}"
        );
        assert!(msg.contains("refusing"), "refusal is explicit: {msg}");
    }

    #[test]
    fn unrecognized_step_payload_is_refused_explicitly() {
        let json = r#"{"session_id":{"id":1,"external_id":"s1"},"user_input":"hi","step":{"NotAStep":{}},"turn_count":1}"#;
        let err = serde_json::from_str::<CheckpointData>(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("NotAStep"), "refusal names the payload: {msg}");
        assert!(msg.contains("unrecognized"), "refusal is explicit: {msg}");
    }

    #[test]
    fn verify_for_resume_accepts_a_native_tool_call_checkpoint() {
        let data = checkpoint_data(CheckpointStep::BeforeActions {
            actions: sample_actions(),
        });
        assert!(data.verify_for_resume(ActionFormat::ToolCall).is_ok());
    }

    #[test]
    fn verify_for_resume_refuses_cross_backend_pinning() {
        let data = CheckpointData::new(
            SessionId::with_external_id(1, "s1"),
            "go",
            CheckpointStep::AfterUserInput,
            0,
            ActionFormat::CodeAct,
            RunState::default(),
        );
        let msg = data.verify_for_resume(ActionFormat::ToolCall).unwrap_err();
        assert!(
            msg.contains("CodeAct"),
            "refusal names the pinned backend: {msg}"
        );
        assert!(
            msg.contains("ToolCall"),
            "refusal names the resuming backend: {msg}"
        );
        assert!(msg.contains("refused"), "refusal is explicit: {msg}");
    }

    #[test]
    fn verify_for_resume_refuses_mismatched_format_labels() {
        // Hand-built contradiction: the pin says tool_call, the payload says
        // code_act. The label is the evidence — no guessing.
        let data = CheckpointData::new(
            SessionId::with_external_id(1, "s1"),
            "go",
            CheckpointStep::BeforeActions {
                actions: vec![Action {
                    id: "cell_1".to_string(),
                    name: "code_cell".to_string(),
                    args: Value::String(r#"{"code":"1+1"}"#.to_string()),
                    identity: r#"code_cell::{"code":"1+1"}"#.to_string(),
                    format: ActionFormat::CodeAct,
                }],
            },
            0,
            ActionFormat::ToolCall,
            RunState::default(),
        );
        let msg = data.verify_for_resume(ActionFormat::ToolCall).unwrap_err();
        assert!(msg.contains("cell_1"), "refusal names the action: {msg}");
        assert!(
            msg.contains("format label"),
            "refusal explains the label mismatch: {msg}"
        );
    }

    #[test]
    fn verify_for_resume_refuses_a_future_version_value() {
        // Hand-built value — the read-side check is the real gate; this covers
        // callers that assemble CheckpointData in memory.
        let mut data = checkpoint_data(CheckpointStep::AfterUserInput);
        data.version = CHECKPOINT_VERSION + 1;
        let msg = data.verify_for_resume(ActionFormat::ToolCall).unwrap_err();
        assert!(msg.contains("newer"), "refusal explains the version: {msg}");
    }

    #[test]
    fn native_checkpoint_co_writes_run_state() {
        // D5+OV-11 事务同写: a native write always carries the counting state
        // alongside the step in one payload.
        let mut run_state = RunState::default();
        run_state.record_actions(2);
        let data = CheckpointData::new(
            SessionId::with_external_id(42, "test-session"),
            "go",
            CheckpointStep::BeforeActions {
                actions: sample_actions(),
            },
            3,
            ActionFormat::ToolCall,
            run_state,
        );
        let carried = data
            .run_state
            .clone()
            .expect("native write co-writes RunState");
        assert_eq!(carried.turn_actions, 2);
        assert_eq!(carried.total_actions, 2);

        let json = serde_json::to_string(&data).unwrap();
        assert!(
            json.contains("\"run_state\""),
            "wire carries the state: {json}"
        );
        let decoded: CheckpointData = serde_json::from_str(&json).unwrap();
        let restored = decoded.run_state.expect("state survives the round trip");
        assert_eq!(restored.turn_actions, 2);
        assert_eq!(restored.total_actions, 2);
        assert!(restored.run_has_actions);
    }

    #[test]
    fn pre_t7_checkpoint_reads_without_run_state() {
        // The pre-T7 corpus has no counting state on the wire. It must read
        // with the field genuinely absent — never guessed, never backfilled
        // (不猜不回填) — so restores fall back to the old-format paths.
        let json = r#"{"session_id":{"id":42,"external_id":"test-session"},"user_input":"go","step":{"BeforeActions":{"actions":[]}},"turn_count":3,"version":1,"backend":"tool_call"}"#;
        let decoded: CheckpointData = serde_json::from_str(json).unwrap();
        assert!(
            decoded.run_state.is_none(),
            "absent stays absent: {:?}",
            decoded.run_state
        );
    }

    #[test]
    fn converted_legacy_checkpoint_keeps_run_state_absent() {
        // A migrated pre-M3 payload gains the step migration and source mark,
        // but not a counting state — conversion must not invent one.
        let json = r#"{"session_id":{"id":1,"external_id":"s1"},"user_input":"go","step":{"BeforeToolCalls":{"tool_calls":[["call_1","weather","{\"city\":\"Tokyo\"}"]]}},"turn_count":1}"#;
        let decoded: CheckpointData = serde_json::from_str(json).unwrap();
        assert!(decoded.migrated_from.is_some(), "step migration is marked");
        assert!(
            decoded.run_state.is_none(),
            "conversion does not invent counting state: {:?}",
            decoded.run_state
        );
    }
}
