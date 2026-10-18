//! Approval-related pure types: RiskLevel, ApprovalDecision, ApprovalRequest.

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum RiskLevel {
    Safe,
    Sensitive,
    Destructive,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ApprovalRequest {
    pub title: String,
    pub message: String,
    pub action_key: Option<String>,
    pub risk_level: RiskLevel,
    pub raw: Option<Value>,
    /// Identifier of the initiator (the sub-agent's agent_path, e.g. "root/coder-1").
    /// None when initiated by the main agent (serde default ensures old JSON remains deserializable).
    #[serde(default)]
    pub source: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum ApprovalDecision {
    AllowOnce,
    AllowAlways,
    Deny,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Old JSON (without the source field) must deserialize as before (serde default for
    /// backward compatibility, design doc D4). Old logs / old callers are unaffected.
    #[test]
    fn deserializes_legacy_json_without_source() {
        let old = r#"{"title":"t","message":"m","action_key":null,"risk_level":"Safe","raw":null}"#;
        let req: ApprovalRequest = serde_json::from_str(old).unwrap();
        assert_eq!(req.source, None);
    }

    /// New JSON with source carries it through normally.
    #[test]
    fn roundtrips_source_field() {
        let req = ApprovalRequest {
            title: "write_file".into(),
            message: "Write file: src/x.rs".into(),
            action_key: Some("write_file:src/x.rs".into()),
            risk_level: RiskLevel::Sensitive,
            raw: None,
            source: Some("root/coder-1".into()),
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: ApprovalRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.source.as_deref(), Some("root/coder-1"));
    }
}
