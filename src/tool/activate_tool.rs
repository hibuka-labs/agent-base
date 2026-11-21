//! `activate_tool`: the model opens a tool that was named in the roster but not
//! mounted.
//!
//! One call to open, and it stays open for the session (see [`ToolActivation`]).
//! The tool hands back the full wire definition — description and schema — so
//! the next call is made against the real shape and not a guess from the name.
//!
//! **Why it holds its own definitions instead of the registry.** Looking the
//! tool up live would mean taking the registry lock from inside a call, which
//! is the pattern that deadlocked before (`ToolEngine::tools_handle`'s docs).
//! The candidate list is fixed at build time and is the same list the roster is
//! rendered from, so the two cannot drift.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};

use crate::types::{AgentError, AgentResult};

use super::{Content, Tool, ToolActivation, ToolContext, ToolExposure};

/// Opens one of the tools the roster names but does not mount.
pub struct ActivateTool {
    activation: ToolActivation,
    /// Tool name -> the definition the model will get once it is open.
    candidates: HashMap<String, (ToolExposure, Value)>,
    /// The registry's shared deferred-definition catalog ([`ToolRegistry::deferred_defs`](super::ToolRegistry::deferred_defs)).
    ///
    /// Tools can register long after this opener is built (a factory inside
    /// `build`, a plugin, a product's late mount). The catalog fills itself as
    /// they land, so a roster name stays openable no matter when its tool
    /// arrived. Set once after build; `None` in tests with no registry.
    late: std::sync::OnceLock<Arc<std::sync::RwLock<HashMap<String, Value>>>>,
}

impl ActivateTool {
    /// `candidates` is every tool this can open — the same list the roster is
    /// rendered from. Keeping one list is what stops the two from drifting.
    pub fn new(
        activation: ToolActivation,
        candidates: Vec<(String, ToolExposure, Value)>,
    ) -> Arc<Self> {
        Arc::new(Self {
            activation,
            candidates: candidates
                .into_iter()
                .map(|(name, exposure, def)| (name, (exposure, def)))
                .collect(),
            late: std::sync::OnceLock::new(),
        })
    }

    /// Hand this opener the registry's shared deferred-definition catalog so
    /// tools registered after [`Self::new`] become openable too. Idempotent;
    /// the first call wins.
    pub fn set_deferred_defs(&self, defs: Arc<std::sync::RwLock<HashMap<String, Value>>>) {
        let _ = self.late.set(defs);
    }

    /// Candidates first (the build-time truth), then the late catalog (always
    /// `Deferred`: a late tool only exists because its name was deferred).
    fn lookup(&self, name: &str) -> Option<(ToolExposure, Value)> {
        if let Some(hit) = self.candidates.get(name) {
            return Some(hit.clone());
        }
        let defs = self.late.get()?;
        let def = defs
            .read()
            .expect("deferred-defs lock poisoned")
            .get(name)
            .cloned()?;
        Some((ToolExposure::Deferred, def))
    }

    /// Names in sorted order, for the error message a wrong guess deserves —
    /// the union of the build-time candidates and whatever has registered late.
    fn candidate_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.candidates.keys().cloned().collect();
        if let Some(defs) = self.late.get() {
            let defs = defs.read().expect("deferred-defs lock poisoned");
            for name in defs.keys() {
                if !names.contains(name) {
                    names.push(name.clone());
                }
            }
        }
        names.sort_unstable();
        names
    }
}

#[async_trait]
impl Tool for ActivateTool {
    fn name(&self) -> &'static str {
        "activate_tool"
    }

    fn description(&self) -> &'static str {
        "Mount a tool whose schema is not loaded yet.\n\n\
         Some tools are named in the system message but their schemas are NOT loaded — until\n\
         they are, only the name is known, and calling one fails: there is no parameter\n\
         schema to call it with. This tool loads it.\n\
         Pass the tool's name exactly as the roster spells it. The result is that tool's full\n\
         description and argument schema, and from your next turn on it is callable exactly\n\
         like any tool already in your list — and it stays mounted for the rest of the\n\
         session, so open it once and use it as often as you need.\n\
         A name that is not mountable comes back with the ones that are. A name that is\n\
         already mounted comes back saying so. Do not open a capability on the chance it\n\
         comes in handy: every mounted tool is sent on every later request."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The tool name exactly as the roster spells it."
                }
            },
            "required": ["name"],
            "additionalProperties": false,
        })
    }

    /// Exposure is `Direct`: this one must always be on the wire, or nothing can
    /// ever be opened.
    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let name = args
            .get("name")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| AgentError::ToolArgsInvalid {
                name: self.name().to_string(),
                raw: "missing field `name` — pass the tool name exactly as the roster spells it"
                    .to_string(),
            })?;

        let Some((exposure, definition)) = self.lookup(name) else {
            return Ok(vec![Content::text(format!(
                "No tool named `{name}` is mountable. Mountable names: {}.",
                self.candidate_names().join(", ")
            ))]);
        };

        match exposure {
            ToolExposure::Hidden => Ok(vec![Content::text(format!(
                "`{name}` is internal and is not mountable. It is not in the roster."
            ))]),
            ToolExposure::Direct => Ok(vec![Content::text(format!(
                "`{name}` is already mounted — call it directly."
            ))]),
            ToolExposure::Deferred => {
                let first_open = self.activation.activate(name);
                let status = if first_open {
                    "mounted"
                } else {
                    "already mounted"
                };
                Ok(vec![Content::text(format!(
                    "`{name}` {status}. Call it from the next turn on, with these arguments:\n{}",
                    serde_json::to_string_pretty(&definition).unwrap_or_default()
                ))])
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::engine::EventBus;
    use crate::tool::ExecClock;
    use crate::types::SessionId;
    use serde_json::json;

    fn test_context() -> ToolContext {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        ToolContext {
            session_id: SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: crate::types::Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            max_output_chars: None,
            exec_clock: ExecClock::default(),
            waiting_for_human: Arc::new(AtomicBool::new(false)),
            event_bus: EventBus::new(1),
        }
    }

    fn registry_free_tool() -> (ActivateTool, ToolActivation) {
        let activation = ToolActivation::new();
        let tool = ActivateTool::new(
            activation.clone(),
            vec![
                (
                    "browser_navigate".to_string(),
                    ToolExposure::Deferred,
                    json!({
                        "name": "browser_navigate",
                        "description": "Open a URL in the browser.",
                        "parameters": {"type": "object", "properties": {"url": {"type": "string"}}},
                    }),
                ),
                (
                    "always_here".to_string(),
                    ToolExposure::Direct,
                    json!({"name": "always_here"}),
                ),
                (
                    "internal".to_string(),
                    ToolExposure::Hidden,
                    json!({"name": "internal"}),
                ),
            ],
        );
        // `ActivateTool::new` returns Arc; unwrap for the tests to own the value.
        let tool = Arc::try_unwrap(tool).ok().expect("single owner in tests");
        (tool, activation)
    }

    async fn call(tool: &ActivateTool, args: Value) -> String {
        let ctx = test_context();
        let out = tool.call(&args, &ctx).await.expect("call succeeds");
        out.iter()
            .map(|c| match c {
                Content::Text { text } => text.clone(),
                _ => String::new(),
            })
            .collect()
    }

    /// Value: protects=the one thing the tool exists for — an opened tool must
    /// be callable afterwards and its shape must come from us, not from the
    /// model's reading of its name; fails_when=opening changes nothing, or
    /// returns something other than the real definition. why_new=the tool is
    /// new. seam=roster -> request; no_claim=that the model will read it.
    #[tokio::test]
    async fn opening_a_tool_mounts_it_and_hands_back_its_shape() {
        let (tool, activation) = registry_free_tool();
        let out = call(&tool, json!({"name": "browser_navigate"})).await;
        assert!(
            activation.is_active("browser_navigate"),
            "opening must stick for the session"
        );
        assert!(out.contains("mounted"), "say what happened: {out}");
        assert!(
            out.contains("\"url\""),
            "and hand back the real arguments, not just the name: {out}"
        );
    }

    /// Value: protects=the model against a confidently wrong guess — a name
    /// that is not mountable must come back with the list, because a silent
    /// success leaves it calling a tool that never arrived; fails_when=an
    /// unknown name is accepted. why_new=the tool is new. seam=roster; no_claim=
    /// that the list is the only way to discover names.
    #[tokio::test]
    async fn an_unknown_name_is_refused_with_the_real_ones() {
        let (tool, _activation) = registry_free_tool();
        let out = call(&tool, json!({"name": "browser_nav"})).await;
        assert!(out.contains("No tool named `browser_nav`"), "{out}");
        assert!(
            out.contains("browser_navigate"),
            "the answer must name the tool it meant: {out}"
        );
    }

    /// Value: protects=the reason `Hidden` exists — internal plumbing must stay
    /// internal, and a roster-shaped hole must not become a way in;
    /// fails_when=an internal tool can be mounted. why_new=the tool is new.
    /// seam=exposure policy; no_claim=that no other path reaches a hidden tool.
    #[tokio::test]
    async fn an_internal_tool_cannot_be_mounted() {
        let (tool, activation) = registry_free_tool();
        let out = call(&tool, json!({"name": "internal"})).await;
        assert!(out.contains("not mountable"), "{out}");
        assert!(!activation.is_active("internal"), "and nothing was opened");
    }

    /// Value: protects=a turn the model could have spent working — a mounted
    /// tool is already on the wire, so asking again must not read as success
    /// and cost another round; fails_when=re-opening reports a fresh mount.
    /// why_new=the tool is new. seam=session state; no_claim=about a session
    /// where the tool was unmounted by other means.
    #[tokio::test]
    async fn opening_twice_is_reported_as_already_mounted() {
        let (tool, _activation) = registry_free_tool();
        call(&tool, json!({"name": "browser_navigate"})).await;
        let out = call(&tool, json!({"name": "browser_navigate"})).await;
        assert!(out.contains("already mounted"), "{out}");
    }

    /// Value: protects=the budget the roster exists to save — `Direct` tools
    /// are already on every request, so "opening" one must not read as a win
    /// and must not change the request; fails_when=a Direct tool is reported
    /// mounted. why_new=the tool is new. seam=exposure policy.
    #[tokio::test]
    async fn an_already_mounted_tool_is_said_to_be() {
        let (tool, activation) = registry_free_tool();
        let out = call(&tool, json!({"name": "always_here"})).await;
        assert!(out.contains("already mounted"), "{out}");
        assert!(
            !activation.is_active("always_here"),
            "a Direct tool needs no activation entry"
        );
    }

    /// Value: protects=the roster's promise for tools that register late (a
    /// factory inside build, a plugin, a product's late mount) — the opener
    /// must serve them from the registry's shared catalog instead of answering
    /// "not mountable"; fails_when=a late-registered deferred tool cannot be
    /// opened. why_new=session 20261121_f984398e: `close_agent` came back
    /// "not mountable" because its tool registered after the candidates were
    /// built. seam=registry catalog -> opener; no_claim=about which tools
    /// should be late.
    #[tokio::test]
    async fn a_late_registered_deferred_tool_is_openable() {
        let (tool, activation) = registry_free_tool();
        let catalog: Arc<std::sync::RwLock<HashMap<String, Value>>> = Default::default();
        catalog.write().unwrap().insert(
            "close_agent".to_string(),
            json!({"name": "close_agent", "description": "Close a sub-agent."}),
        );
        tool.set_deferred_defs(catalog);
        let out = call(&tool, json!({"name": "close_agent"})).await;
        assert!(
            activation.is_active("close_agent"),
            "opening must stick for the session: {out}"
        );
        assert!(out.contains("mounted"), "{out}");
        assert!(
            out.contains("Close a sub-agent"),
            "hand back the real definition, not just the name: {out}"
        );
    }

    /// Value: protects=the arg contract — a call with no name is a mistake the
    /// model can correct if we say so, and an empty name is not a name;
    /// fails_when=both are accepted and quietly do nothing.
    /// why_new=the tool is new. seam=args.
    #[tokio::test]
    async fn a_missing_name_says_what_is_missing() {
        let (tool, _activation) = registry_free_tool();
        let err = tool.call(&json!({}), &test_context()).await;
        let msg = format!("{:?}", err.expect_err("no name is an error"));
        assert!(
            msg.contains("`name`"),
            "the error must name the field: {msg}"
        );
    }
}
