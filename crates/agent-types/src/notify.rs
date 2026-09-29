//! Notify pure types: NoticeKind, Notice.

use serde::{Deserialize, Serialize};

/// Notice category — decides how consumers render the notification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NoticeKind {
    /// Persistent warning line (red), does not interrupt the turn.
    Warning,
    /// Transient progress (status bar), next one replaces the previous.
    Progress,
    /// Plain information line.
    Info,
}

/// Display-only notice from an engine mechanism.
///
/// `text` is a mechanism-level English fact; product layers (e.g. a TUI)
/// may rewrite it per `source`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Notice {
    pub kind: NoticeKind,
    /// Emitting mechanism: "guard" / "judge" / "tool:<name>" / "reaper" ...
    pub source: String,
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // Design doc §9: Notice/NoticeKind serde round-trip + serialization tag
    #[test]
    fn notice_kind_serde_snake_case_round_trip() {
        for (kind, tag) in [
            (NoticeKind::Warning, "warning"),
            (NoticeKind::Progress, "progress"),
            (NoticeKind::Info, "info"),
        ] {
            let v = serde_json::to_value(&kind).unwrap();
            assert_eq!(v, json!(tag));
            let back: NoticeKind = serde_json::from_value(v).unwrap();
            assert_eq!(&back, &kind);
        }
    }

    #[test]
    fn notice_serde_round_trip() {
        let notice = Notice {
            kind: NoticeKind::Warning,
            source: "guard".to_string(),
            text: "judge unparsed — treating as complete".to_string(),
        };
        let v = serde_json::to_value(&notice).unwrap();
        assert_eq!(v["kind"], json!("warning"));
        assert_eq!(v["source"], json!("guard"));
        assert_eq!(v["text"], json!("judge unparsed — treating as complete"));

        let back: Notice = serde_json::from_value(v).unwrap();
        assert_eq!(back, notice);
    }
}
