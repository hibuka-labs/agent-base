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
    /// 发起方标识（子 agent 的 agent_path，如 "root/coder-1"）。
    /// 主 agent 发起时为 None（serde default 保证旧 JSON 可反序列化）。
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

    /// 旧 JSON（无 source 字段）必须照常反序列化（serde default 向后兼容，
    /// 设计文档 D4）。旧日志 / 旧调用方不受影响。
    #[test]
    fn deserializes_legacy_json_without_source() {
        let old = r#"{"title":"t","message":"m","action_key":null,"risk_level":"Safe","raw":null}"#;
        let req: ApprovalRequest = serde_json::from_str(old).unwrap();
        assert_eq!(req.source, None);
    }

    /// 新 JSON 带 source 时正常携带。
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
