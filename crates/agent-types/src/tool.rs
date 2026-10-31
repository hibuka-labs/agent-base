//! Tool-related pure types: Content, ToolMetadata, ToolExposure, ActivationContext.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use crate::session::SessionId;
use llm_trait::message::ImageAttachment;

/// Structured content returned by a tool, aligned with the MCP `content`
/// array shape (no envelope, no orchestration/failure/truncation semantics).
///
/// `Text` is the textual face and `Image` the pictorial one: a tool that
/// hands the model something to *look at* returns an `Image`, and
/// [`content_images`] turns those into the attachments that ride the tool
/// result to the provider (`content_text` keeps only the words). `Detail`
/// carries machine-readable metadata (e.g. edit line numbers) that the UI
/// can consume but the LLM never sees.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Content {
    Text {
        text: String,
    },
    /// Base64-encoded image payload.
    Image {
        data: String,
        mime_type: String,
    },
    /// Machine-readable metadata for UI consumption (not sent to the LLM).
    Detail {
        data: serde_json::Value,
    },
    /// A stored observation artifact referenced by id (M3c,
    /// action-executor-design §6): the body is fetched on demand via
    /// `obs.fetch`; the model face is the summary+reference line carried in
    /// `Text` — so this typed channel marks the artifact for UI/logs without
    /// duplicating the line.
    Artifact {
        /// Stable store id (`art_<seq>`).
        capture_id: String,
        /// What the body is (`"stdout"` / `"stderr"` / `"observation"` / the
        /// emitter's own label).
        kind: String,
        /// The truncation-heuristic glimpse of the body.
        summary: String,
    },
}

impl Content {
    pub fn text(s: impl Into<String>) -> Self {
        Content::Text { text: s.into() }
    }

    pub fn image(data: impl Into<String>, mime_type: impl Into<String>) -> Self {
        Content::Image {
            data: data.into(),
            mime_type: mime_type.into(),
        }
    }

    pub fn detail(data: serde_json::Value) -> Self {
        Content::Detail { data }
    }

    pub fn artifact(
        capture_id: impl Into<String>,
        kind: impl Into<String>,
        summary: impl Into<String>,
    ) -> Self {
        Content::Artifact {
            capture_id: capture_id.into(),
            kind: kind.into(),
            summary: summary.into(),
        }
    }
}

impl From<Content> for Vec<Content> {
    fn from(c: Content) -> Self {
        vec![c]
    }
}

/// Join the textual portion of tool output into a single string for display
/// and session history. Non-text variants (e.g. `Image`, `Detail`) are skipped;
/// `Artifact` is skipped too — its model face is the summary+reference line
/// already carried in `Text` (the typed channel marks the artifact for UI).
pub fn content_text(contents: &[Content]) -> String {
    contents
        .iter()
        .filter_map(|c| match c {
            Content::Text { text } => Some(text.as_str()),
            Content::Image { .. } | Content::Detail { .. } | Content::Artifact { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Extract the pictures a tool's output hands the model to look at, in the
/// order they were produced — the image face of a tool result, exactly as
/// [`content_text`] is its textual face. Empty when the output carried none.
pub fn content_images(contents: &[Content]) -> Vec<ImageAttachment> {
    contents
        .iter()
        .filter_map(|c| match c {
            Content::Image { data, mime_type } => Some(ImageAttachment::Base64 {
                data: data.clone(),
                media_type: Some(mime_type.clone()),
                detail: None,
            }),
            _ => None,
        })
        .collect()
}

/// Extract structured detail metadata from tool output. When multiple `Detail`
/// entries exist, they are merged into a single object. Returns `None` if no
/// `Detail` entries are present.
pub fn content_details(contents: &[Content]) -> Option<serde_json::Value> {
    let details: Vec<&serde_json::Value> = contents
        .iter()
        .filter_map(|c| match c {
            Content::Detail { data } => Some(data),
            _ => None,
        })
        .collect();
    match details.len() {
        0 => None,
        1 => Some(details[0].clone()),
        _ => {
            // Merge multiple detail objects into one
            let mut merged = serde_json::Map::new();
            for d in details {
                if let Some(obj) = d.as_object() {
                    for (k, v) in obj {
                        merged.insert(k.clone(), v.clone());
                    }
                }
            }
            Some(serde_json::Value::Object(merged))
        }
    }
}

/// Machine-readable metadata for a registered tool — origin, version, and
/// runtime requirements in a stable shape consumers can inspect without
/// parsing the LLM-facing definition JSON.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ToolMetadata {
    /// Tool name (matches `Tool::name`).
    pub name: String,
    /// Human-readable description.
    pub description: String,
    /// Where this tool comes from: a crate name (e.g. `"phi-tools"`), a
    /// framework identifier (`"agent-base"`, `"agent-works"`), or
    /// `"custom"` for user-defined tools.
    pub origin: String,
    /// Crate / package version, or `"unknown"` when built outside a crate.
    pub version: String,
    /// Optional runtime requirements or capabilities this tool depends on.
    pub requirements: Vec<String>,
}

/// Visibility level of a tool to the LLM model.
///
/// Controls whether a tool appears in the tool definitions sent to the model
/// each turn. Defaults to `Direct` for backward compatibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolExposure {
    /// Always visible to the model. This is the default.
    Direct,
    /// Conditionally visible — the tool decides via `Tool::should_activate`.
    Deferred,
    /// Never visible to the model (internal/framework tools).
    Hidden,
}

/// Context passed to `Tool::should_activate` for Deferred tools.
///
/// Built once per react-loop iteration; tools inspect it to decide
/// whether they should be exposed to the model this turn.
#[derive(Clone, Debug)]
pub struct ActivationContext {
    /// Current session ID.
    pub session_id: SessionId,
    /// Names of tools already activated (Direct + activated Deferred) this turn.
    pub current_tools: Vec<String>,
    /// Workspace / working directory path.
    pub workspace: PathBuf,
}
