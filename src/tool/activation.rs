//! Session-scoped tool activation: which `Deferred` tools the model has opened.
//!
//! A `Deferred` tool is named in the roster the model sees but costs no schema
//! until it is opened. Opening one is one call; keeping it open is free. See
//! `notes/tool-face-design.md` for why the roster is names-plus-a-line rather
//! than a search index.
//!
//! **Sticky for the session.** The alternative — closing again after each use —
//! would charge a turn every time the model reaches for it, which is exactly
//! the cost the roster exists to avoid.

use std::collections::BTreeSet;
use std::sync::{Arc, RwLock};

/// Names of the `Deferred` tools the model has opened this session.
///
/// Cheap to clone ([`Arc`]) so an [`crate::tool::ActivateTool`] can hold one
/// while the engine holds another.
#[derive(Clone, Default)]
pub struct ToolActivation {
    inner: Arc<RwLock<BTreeSet<String>>>,
}

impl ToolActivation {
    pub fn new() -> Self {
        Self::default()
    }

    /// Open a tool for the rest of the session. Idempotent; returns whether
    /// this call is what opened it.
    pub fn activate(&self, name: &str) -> bool {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .insert(name.to_string())
    }

    /// Has the model opened this tool?
    pub fn is_active(&self, name: &str) -> bool {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .contains(name)
    }

    /// Opened names in sorted order — what
    /// [`agent_types::ActivationContext::current_tools`] carries.
    pub fn snapshot(&self) -> Vec<String> {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .cloned()
            .collect()
    }

    /// Forget everything. A new session starts closed.
    pub fn clear(&self) {
        self.inner
            .write()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Value: protects=the cost model of the whole roster — a tool opened once
    /// must stay open, or the model pays a turn each time it reaches for it and
    /// the roster saves nothing; fails_when=re-activation is required.
    /// why_new=activation is new. seam=session state; no_claim=that stickiness
    /// is right across a `/resume`, only within one running session.
    #[test]
    fn opening_a_tool_once_keeps_it_open() {
        let act = ToolActivation::new();
        assert!(
            act.activate("browser_navigate"),
            "the first open reports it"
        );
        assert!(
            !act.activate("browser_navigate"),
            "and the second is a no-op"
        );
        assert!(act.is_active("browser_navigate"), "still open");
    }

    /// Value: protects=the roster's promise that unopened tools cost nothing —
    /// a tool the model never asked for must never appear; fails_when=a closed
    /// tool reports as open. why_new=activation is new. seam=session state;
    /// no_claim=about tools activated by other means (a predicate override).
    #[test]
    fn a_tool_that_was_never_opened_stays_closed() {
        let act = ToolActivation::new();
        act.activate("a");
        assert!(!act.is_active("b"), "opening one tool opens one tool");
    }

    /// Value: protects=determinism of the tool list — the snapshot drives which
    /// schemas go on the wire, so it must be a stable order or the request
    /// prefix churns and prompt caching misses every turn; fails_when=the
    /// iteration order follows the hash map. why_new=activation is new.
    /// seam=session state -> request; no_claim=that sorted order is required by
    /// any provider, only that it is stable.
    #[test]
    fn the_snapshot_is_stable_so_the_request_prefix_is() {
        let act = ToolActivation::new();
        act.activate("zeta");
        act.activate("alpha");
        act.activate("mid");
        assert_eq!(act.snapshot(), vec!["alpha", "mid", "zeta"]);
    }

    /// Value: protects=session boundaries — one session's openings must not
    /// leak into the next, or a tool from a past task shows up in this one;
    /// fails_when=clear leaves names behind. why_new=activation is new.
    /// seam=session state; no_claim=about process-level reuse across sessions.
    #[test]
    fn clear_forgets_every_opening() {
        let act = ToolActivation::new();
        act.activate("a");
        act.clear();
        assert!(act.snapshot().is_empty());
        assert!(!act.is_active("a"));
    }
}
