use serde::{Deserialize, Serialize};
use serde_json::Value;

use agent_types::NoticeKind;

use super::approval::ApprovalRequest;
use super::checkpoint::CheckpointData;
use super::plan_update::PlanItem;
use super::session::SessionId;

// ---------------------------------------------------------------------------
// UserEvent — user-space events produced by tools
// ---------------------------------------------------------------------------

/// User-space events produced by tools during execution.
///
/// Tools send these through `ToolContext::emit_user_event()`. The framework
/// wraps them in [`RuntimeEvent::UserEvent`] before delivering to external
/// consumers.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "userEventType", rename_all = "camelCase")]
pub enum UserEvent {
    /// Tool progress notification.
    Progress { text: String },
    /// User-defined structured event for custom business semantics.
    Structured { event_type: String, data: Value },
    /// Tool partial result — emitted during long-running tool execution.
    /// `is_partial: true` means more output is coming; `false` means this is the final chunk.
    ToolPartialResult {
        tool_call_id: String,
        content: String,
        is_partial: bool,
    },
    /// Display-only notice from an engine mechanism (guard degradation,
    /// reaper, ...). `text` is a mechanism-level English fact; product layers
    /// may rewrite it per `source`. `source` is a `String` (not `&'static str`)
    /// because `UserEvent` must deserialize across process boundaries.
    Notice {
        kind: NoticeKind,
        source: String,
        text: String,
    },
}

// ---------------------------------------------------------------------------
// RuntimeEvent — unified event stream for all consumers
// ---------------------------------------------------------------------------

/// Unified runtime event — the single event type for both internal and external
/// consumers (frontends, CLIs, tests).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "runtimeEventType", rename_all = "camelCase")]
pub enum RuntimeEvent {
    // --- Framework system events ---
    TextDelta {
        session_id: SessionId,
        text: String,
        /// Identifies the agent that produced this event (root / sub-agent path).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        /// Distributed tracing context carried across systems (e.g. MCP caller → phi-agent).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
    ThoughtDelta {
        session_id: SessionId,
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
    /// A tool call that is still being streamed into existence.
    ///
    /// Tool-call arguments arrive as fragments and are only materialized into
    /// a [`RuntimeEvent::ActionStarted`] once the whole turn's stream is
    /// drained — multi-KB arguments (a `spawn_agent` task brief) are invisible
    /// to every consumer until then. This event makes that window observable:
    /// coarse progress on the call being drafted (`index`), enough for a UI to
    /// say "writing tool calls" instead of looking hung. It carries no
    /// argument text, only a size, and is throttled at the source.
    #[serde(alias = "toolCallDraft")]
    ActionDraft {
        session_id: SessionId,
        /// The call's stream index within this turn (its eventual tool-call
        /// id is not known yet).
        index: usize,
        /// Function name, empty until the fragment carrying it arrives.
        name: String,
        /// Argument characters accumulated for this call so far.
        args_len: usize,
        /// How many calls this turn has produced so far (drafts included).
        count: usize,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
    #[serde(alias = "toolCallStarted")]
    ActionStarted {
        session_id: SessionId,
        tool_name: String,
        args_json: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
    #[serde(alias = "toolCallFinished")]
    ActionFinished {
        session_id: SessionId,
        tool_name: String,
        summary: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
        /// `true` when this finish is the result of an approval denial (not a
        /// normal tool result or execution error).
        #[serde(default)]
        denied: bool,
        /// Structured metadata from the tool result (e.g. edit line numbers).
        /// Carried through to UI consumers; never sent to the LLM.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        details: Option<Value>,
    },
    AwaitingApproval {
        session_id: SessionId,
        request: ApprovalRequest,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
    Checkpoint {
        session_id: SessionId,
        /// Boxed so the announcement event stays small on the hot path (every
        /// RuntimeEvent copy is as big as its largest variant).
        checkpoint: Box<CheckpointData>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
    RunFinished {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
    RunCancelled {
        session_id: SessionId,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
    // --- Lightweight plan update (display-only, no execution semantics) ---
    PlanUpdated {
        session_id: SessionId,
        objective: String,
        explanation: Option<String>,
        plan: Vec<PlanItem>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
    // --- User-space events ---
    /// A user-space event produced by a tool.
    UserEvent {
        session_id: SessionId,
        event: UserEvent,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        agent_id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        trace_id: Option<String>,
    },
}

impl RuntimeEvent {
    /// Get the session ID associated with this event.
    pub fn session_id(&self) -> &SessionId {
        match self {
            RuntimeEvent::TextDelta { session_id, .. } => session_id,
            RuntimeEvent::ThoughtDelta { session_id, .. } => session_id,
            RuntimeEvent::ActionDraft { session_id, .. } => session_id,
            RuntimeEvent::ActionStarted { session_id, .. } => session_id,
            RuntimeEvent::ActionFinished { session_id, .. } => session_id,
            RuntimeEvent::AwaitingApproval { session_id, .. } => session_id,
            RuntimeEvent::Checkpoint { session_id, .. } => session_id,
            RuntimeEvent::RunFinished { session_id, .. } => session_id,
            RuntimeEvent::RunCancelled { session_id, .. } => session_id,
            RuntimeEvent::PlanUpdated { session_id, .. } => session_id,
            RuntimeEvent::UserEvent { session_id, .. } => session_id,
        }
    }

    /// Get the agent ID associated with this event, if any.
    pub fn agent_id(&self) -> Option<&str> {
        match self {
            RuntimeEvent::TextDelta { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::ThoughtDelta { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::ActionDraft { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::ActionStarted { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::ActionFinished { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::AwaitingApproval { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::Checkpoint { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::RunFinished { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::RunCancelled { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::PlanUpdated { agent_id, .. } => agent_id.as_deref(),
            RuntimeEvent::UserEvent { agent_id, .. } => agent_id.as_deref(),
        }
    }

    /// Get the trace ID associated with this event, if any.
    pub fn trace_id(&self) -> Option<&str> {
        match self {
            RuntimeEvent::TextDelta { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::ThoughtDelta { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::ActionDraft { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::ActionStarted { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::ActionFinished { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::AwaitingApproval { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::Checkpoint { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::RunFinished { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::RunCancelled { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::PlanUpdated { trace_id, .. } => trace_id.as_deref(),
            RuntimeEvent::UserEvent { trace_id, .. } => trace_id.as_deref(),
        }
    }

    /// Set the agent ID (sub-agent path) on this event.
    pub fn with_agent_id(mut self, id: impl Into<String>) -> Self {
        let id = id.into();
        match &mut self {
            RuntimeEvent::TextDelta { agent_id, .. }
            | RuntimeEvent::ThoughtDelta { agent_id, .. }
            | RuntimeEvent::ActionDraft { agent_id, .. }
            | RuntimeEvent::ActionStarted { agent_id, .. }
            | RuntimeEvent::ActionFinished { agent_id, .. }
            | RuntimeEvent::AwaitingApproval { agent_id, .. }
            | RuntimeEvent::Checkpoint { agent_id, .. }
            | RuntimeEvent::RunFinished { agent_id, .. }
            | RuntimeEvent::RunCancelled { agent_id, .. }
            | RuntimeEvent::PlanUpdated { agent_id, .. }
            | RuntimeEvent::UserEvent { agent_id, .. } => *agent_id = Some(id),
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{
        ActionFormat, CheckpointStep, MigrationKind, PlanStepStatus, RiskLevel, RunState,
    };

    fn sid(id: u64) -> SessionId {
        SessionId::new(id)
    }

    fn approval_request() -> ApprovalRequest {
        ApprovalRequest {
            title: "title".to_string(),
            message: "message".to_string(),
            action_key: None,
            risk_level: RiskLevel::Safe,
            raw: None,
            source: None,
        }
    }

    fn checkpoint() -> CheckpointData {
        CheckpointData::new(
            sid(42),
            "input",
            CheckpointStep::AfterUserInput,
            0,
            ActionFormat::ToolCall,
            RunState::default(),
        )
    }

    #[test]
    fn checkpoint_event_reads_pre_m3_jsonl_fixture() {
        // A journal line written by a pre-M3 binary: tool-call triples, no
        // version/backend fields. Must decode with its migration mark intact
        // (旧 checkpoint/JSONL fixture → 新二进制可读).
        let line = r#"{"runtimeEventType":"checkpoint","session_id":{"id":7,"external_id":"s7"},"checkpoint":{"session_id":{"id":7,"external_id":"s7"},"user_input":"resume me","step":{"BeforeToolCalls":{"tool_calls":[["call_1","echo","{\"text\":\"hi\"}"]]}},"turn_count":2}}"#;
        let event: RuntimeEvent = serde_json::from_str(line).unwrap();
        match event {
            RuntimeEvent::Checkpoint { checkpoint, .. } => {
                assert_eq!(checkpoint.version, 0);
                assert_eq!(checkpoint.backend, ActionFormat::ToolCall);
                let mark = checkpoint
                    .migrated_from
                    .as_ref()
                    .expect("legacy payload must carry its source mark");
                assert_eq!(mark.kind, MigrationKind::ToolCallTriplesV0);
                match &checkpoint.step {
                    CheckpointStep::BeforeActions { actions } => {
                        assert_eq!(actions.len(), 1);
                        assert_eq!(actions[0].id, "call_1");
                        assert_eq!(actions[0].name, "echo");
                        assert_eq!(actions[0].raw_args(), r#"{"text":"hi"}"#);
                    }
                    other => panic!("expected BeforeActions, got {other:?}"),
                }
            }
            other => panic!("expected Checkpoint event, got {other:?}"),
        }
    }

    #[test]
    fn accessors_return_embedded_ids_for_all_variants() {
        let events: Vec<RuntimeEvent> = vec![
            RuntimeEvent::TextDelta {
                session_id: sid(1),
                text: "hi".into(),
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
            RuntimeEvent::ThoughtDelta {
                session_id: sid(2),
                text: "hmm".into(),
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
            RuntimeEvent::ActionStarted {
                session_id: sid(3),
                tool_name: "read".into(),
                args_json: "{}".into(),
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
            RuntimeEvent::ActionFinished {
                session_id: sid(4),
                tool_name: "read".into(),
                summary: "ok".into(),
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
                denied: false,
                details: None,
            },
            RuntimeEvent::AwaitingApproval {
                session_id: sid(5),
                request: approval_request(),
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
            RuntimeEvent::Checkpoint {
                session_id: sid(6),
                checkpoint: Box::new(checkpoint()),
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
            RuntimeEvent::RunFinished {
                session_id: sid(7),
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
            RuntimeEvent::RunCancelled {
                session_id: sid(8),
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
            RuntimeEvent::PlanUpdated {
                session_id: sid(9),
                objective: "goal".into(),
                explanation: None,
                plan: vec![PlanItem {
                    step: "s".into(),
                    status: PlanStepStatus::Pending,
                }],
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
            RuntimeEvent::UserEvent {
                session_id: sid(10),
                event: UserEvent::Progress { text: "p".into() },
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
            RuntimeEvent::ActionDraft {
                session_id: sid(11),
                index: 0,
                name: "spawn_agent".into(),
                args_len: 512,
                count: 1,
                agent_id: Some("a".into()),
                trace_id: Some("t".into()),
            },
        ];

        for (i, ev) in events.iter().enumerate() {
            let expected = i as u64 + 1;
            assert_eq!(ev.session_id(), &sid(expected), "variant {i}");
            assert_eq!(ev.agent_id(), Some("a"), "variant {i}");
            assert_eq!(ev.trace_id(), Some("t"), "variant {i}");
        }
    }

    #[test]
    fn with_agent_id_sets_id_on_all_variants() {
        let events: Vec<RuntimeEvent> = vec![
            RuntimeEvent::TextDelta {
                session_id: sid(1),
                text: "hi".into(),
                agent_id: None,
                trace_id: None,
            },
            RuntimeEvent::ThoughtDelta {
                session_id: sid(2),
                text: "hmm".into(),
                agent_id: None,
                trace_id: None,
            },
            RuntimeEvent::ActionStarted {
                session_id: sid(3),
                tool_name: "read".into(),
                args_json: "{}".into(),
                agent_id: None,
                trace_id: None,
            },
            RuntimeEvent::ActionFinished {
                session_id: sid(4),
                tool_name: "read".into(),
                summary: "ok".into(),
                agent_id: None,
                trace_id: None,
                denied: false,
                details: None,
            },
            RuntimeEvent::AwaitingApproval {
                session_id: sid(5),
                request: approval_request(),
                agent_id: None,
                trace_id: None,
            },
            RuntimeEvent::Checkpoint {
                session_id: sid(6),
                checkpoint: Box::new(checkpoint()),
                agent_id: None,
                trace_id: None,
            },
            RuntimeEvent::RunFinished {
                session_id: sid(7),
                agent_id: None,
                trace_id: None,
            },
            RuntimeEvent::RunCancelled {
                session_id: sid(8),
                agent_id: None,
                trace_id: None,
            },
            RuntimeEvent::PlanUpdated {
                session_id: sid(9),
                objective: "goal".into(),
                explanation: None,
                plan: vec![PlanItem {
                    step: "s".into(),
                    status: PlanStepStatus::Pending,
                }],
                agent_id: None,
                trace_id: None,
            },
            RuntimeEvent::UserEvent {
                session_id: sid(10),
                event: UserEvent::Progress { text: "p".into() },
                agent_id: None,
                trace_id: None,
            },
            RuntimeEvent::ActionDraft {
                session_id: sid(11),
                index: 0,
                name: "spawn_agent".into(),
                args_len: 512,
                count: 1,
                agent_id: None,
                trace_id: None,
            },
        ];

        for (i, ev) in events.into_iter().enumerate() {
            let tagged = ev.with_agent_id("sub/1");
            assert_eq!(tagged.agent_id(), Some("sub/1"), "variant {i}");
            assert_eq!(tagged.session_id(), &sid(i as u64 + 1), "variant {i}");
        }
    }

    #[test]
    fn accessors_return_none_when_ids_absent() {
        let ev = RuntimeEvent::RunFinished {
            session_id: sid(1),
            agent_id: None,
            trace_id: None,
        };
        assert_eq!(ev.session_id(), &sid(1));
        assert_eq!(ev.agent_id(), None);
        assert_eq!(ev.trace_id(), None);
    }

    #[test]
    fn runtime_event_serde_uses_camel_case_tag() {
        let ev = RuntimeEvent::ActionStarted {
            session_id: SessionId::with_external_id(7, "ext"),
            tool_name: "read".into(),
            args_json: "{}".into(),
            agent_id: Some("a".into()),
            trace_id: None,
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["runtimeEventType"], "actionStarted");
        assert_eq!(v["session_id"]["id"], serde_json::json!(7));
        assert_eq!(v["session_id"]["external_id"], "ext");
        // trace_id is None → field skipped
        assert!(v.get("trace_id").is_none());

        let back: RuntimeEvent = serde_json::from_value(v).unwrap();
        assert_eq!(back.session_id(), &SessionId::with_external_id(7, "ext"));
        assert_eq!(back.agent_id(), Some("a"));
        assert_eq!(back.trace_id(), None);
    }

    #[test]
    fn tool_call_draft_serde_round_trip() {
        let ev = RuntimeEvent::ActionDraft {
            session_id: sid(3),
            index: 2,
            name: "spawn_agent".into(),
            args_len: 640,
            count: 4,
            agent_id: None,
            trace_id: None,
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["runtimeEventType"], "actionDraft");
        assert_eq!(v["args_len"], 640);
        assert!(v.get("agent_id").is_none(), "None ids stay skipped");

        // Old wire names (pre-M3 rename) still deserialize via serde alias.
        let mut old = v.clone();
        old["runtimeEventType"] = serde_json::json!("toolCallDraft");
        let legacy: RuntimeEvent = serde_json::from_value(old).unwrap();
        assert!(matches!(legacy, RuntimeEvent::ActionDraft { .. }));

        let back: RuntimeEvent = serde_json::from_value(v).unwrap();
        match back {
            RuntimeEvent::ActionDraft {
                index,
                name,
                args_len,
                count,
                ..
            } => {
                assert_eq!(index, 2);
                assert_eq!(name, "spawn_agent");
                assert_eq!(args_len, 640);
                assert_eq!(count, 4);
            }
            other => panic!("expected ActionDraft, got {other:?}"),
        }
    }

    #[test]
    fn user_event_serde_tag() {
        let v = serde_json::to_value(UserEvent::Progress {
            text: "working".into(),
        })
        .unwrap();
        assert_eq!(v["userEventType"], "progress");
        assert_eq!(v["text"], "working");

        let v = serde_json::to_value(UserEvent::Structured {
            event_type: "custom".into(),
            data: serde_json::json!({"k": 1}),
        })
        .unwrap();
        assert_eq!(v["userEventType"], "structured");
        assert_eq!(v["data"]["k"], serde_json::json!(1));
    }

    // Design doc §9: UserEvent::Notice serialization tag + serde round-trip
    // (source must be String — &'static str cannot deserialize).
    #[test]
    fn user_event_notice_serde_round_trip() {
        let ev = UserEvent::Notice {
            kind: NoticeKind::Warning,
            source: "guard".into(),
            text: "guard judge unparsed — treating as complete".into(),
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["userEventType"], "notice");
        assert_eq!(v["kind"], serde_json::json!("warning"));
        assert_eq!(v["source"], serde_json::json!("guard"));
        assert_eq!(
            v["text"],
            serde_json::json!("guard judge unparsed — treating as complete")
        );

        let back: UserEvent = serde_json::from_value(v).unwrap();
        match back {
            UserEvent::Notice { kind, source, text } => {
                assert_eq!(kind, NoticeKind::Warning);
                assert_eq!(source, "guard");
                assert_eq!(text, "guard judge unparsed — treating as complete");
            }
            other => panic!("expected Notice, got {other:?}"),
        }
    }

    // Design doc §11: old JSONL replay — a Notice serialized elsewhere must
    // deserialize cleanly alongside existing variants (serde is add-only).
    #[test]
    fn notice_coexists_with_legacy_variants_on_replay() {
        let json = r#"{"userEventType":"progress","text":"legacy line"}"#;
        let back: UserEvent = serde_json::from_str(json).unwrap();
        assert!(matches!(back, UserEvent::Progress { .. }));
    }
}
