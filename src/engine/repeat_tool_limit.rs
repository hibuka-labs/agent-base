use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::engine::middleware::{Middleware, PostLlmCtx, UserMessageCtx};
use crate::types::{AgentResult, SessionId};

/// Default nudge threshold: identical call count at which the model gets a
/// one-time "stop polling" follow-up.
pub const DEFAULT_NUDGE_AFTER: usize = 5;
/// Default block threshold: identical call count at which pending calls are
/// discarded outright (same hard block as `TurnToolLimitMiddleware`).
pub const DEFAULT_BLOCK_AFTER: usize = 10;

/// Configuration for [`RepeatToolLimitMiddleware`].
#[derive(Clone, Debug)]
pub struct RepeatToolLimitConfig {
    /// Identical-call count that triggers the one-time nudge. `0` disables
    /// nudging (blocking still applies at `block_after`).
    pub nudge_after: usize,
    /// Identical-call count that hard-blocks the pending calls.
    pub block_after: usize,
    pub nudge_message: String,
    pub block_message: String,
}

impl Default for RepeatToolLimitConfig {
    fn default() -> Self {
        Self {
            nudge_after: DEFAULT_NUDGE_AFTER,
            block_after: DEFAULT_BLOCK_AFTER,
            nudge_message: "You have repeatedly issued the same tool call with the same arguments \
                (e.g. polling list_agents for progress). Sub-agent reports are pushed to you \
                automatically the moment your turn ends — polling cannot speed them up and only \
                burns tokens. End your turn now with a brief progress note and no further tool calls."
                .to_string(),
            block_message: "You have issued this identical tool call far too many times. The \
                pending calls were discarded. Based on everything you already know, write your \
                response and end the turn. Do not call this tool again."
                .to_string(),
        }
    }
}

/// Break poll loops — repeated identical tool calls — hard.
///
/// Session 20260914_50cf809d: a root spawned two sub-agents, then polled
/// `list_agents` 124 times over 15 minutes (57% of wall clock, interleaved
/// with real work) instead of ending its turn to receive the pushed reports.
/// The system prompt forbade polling; the model did it anyway, and no
/// mechanism stopped it. This middleware is that mechanism.
///
/// **Counting is cumulative per run, not consecutive**: the incident's polls
/// were interleaved with `read_file` etc., so a consecutive-counter would
/// reset forever and never fire. The fingerprint is tool name + canonical
/// (key-sorted) arguments — a model legitimately reading 51 *different*
/// files never accumulates a fingerprint.
///
/// Two-stage response, mirroring existing middleware precedents:
/// - at `nudge_after` (once per fingerprint): `follow_up_message` nudge
///   (soft, like `MaxTurnsNudgeMiddleware`);
/// - at `block_after` (and every attempt beyond): pending `tool_calls` are
///   discarded and a follow-up forces a summary (hard, like
///   `TurnToolLimitMiddleware`).
///
/// State is per-session and reset by `on_user_message` — a run's counts must
/// not leak into the next run (the middleware instance is process-long in
/// phimint). Clearing all pending calls on block (not just the offending
/// one) follows the `TurnToolLimitMiddleware` precedent: the breaker only
/// trips after `block_after` identical calls, so collateral is negligible.
pub struct RepeatToolLimitMiddleware {
    config: RepeatToolLimitConfig,
    counts: Mutex<HashMap<SessionId, HashMap<String, usize>>>,
}

impl RepeatToolLimitMiddleware {
    pub fn new(config: RepeatToolLimitConfig) -> Self {
        Self {
            config,
            counts: Mutex::new(HashMap::new()),
        }
    }

    /// Fingerprint of one tool call: name + canonical (key-sorted) args.
    fn fingerprint(name: &str, args: &str) -> String {
        let canonical = serde_json::from_str::<serde_json::Value>(args)
            .map(|v| v.to_string())
            .unwrap_or_else(|_| args.to_string());
        format!("{name}::{canonical}")
    }

    /// Bump the run counter for a fingerprint; returns the new count.
    fn bump(&self, session_id: &SessionId, fingerprint: &str) -> usize {
        let mut counts = self.counts.lock().unwrap();
        let session_counts = counts.entry(session_id.clone()).or_default();
        let entry = session_counts.entry(fingerprint.to_string()).or_insert(0);
        *entry += 1;
        *entry
    }
}

#[async_trait]
impl Middleware for RepeatToolLimitMiddleware {
    async fn on_user_message(&self, ctx: &mut UserMessageCtx) -> AgentResult<()> {
        // New run: last run's call counts are stale by definition.
        self.counts.lock().unwrap().remove(&ctx.session_id);
        Ok(())
    }

    async fn on_post_llm(&self, ctx: &mut PostLlmCtx) -> AgentResult<()> {
        if !ctx.is_tool_call || ctx.tool_calls.is_empty() {
            return Ok(());
        }

        let mut max_count = 0usize;
        let mut offending_tool = String::new();
        for (_id, name, args) in &ctx.tool_calls {
            let fingerprint = Self::fingerprint(name, args);
            let count = self.bump(&ctx.session_id, &fingerprint);
            if count > max_count {
                max_count = count;
                offending_tool = name.clone();
            }
        }

        if max_count >= self.config.block_after {
            tracing::warn!(
                session_id = ctx.session_id.id,
                tool = %offending_tool,
                count = max_count,
                pending_calls = ctx.tool_calls.len(),
                "RepeatToolLimit: blocking tool calls — identical-call limit reached"
            );
            ctx.tool_calls.clear();
            ctx.is_tool_call = false;
            ctx.follow_up_message = Some(self.config.block_message.clone());
        } else if self.config.nudge_after > 0 && max_count == self.config.nudge_after {
            // Fire exactly once per fingerprint: the count equals the
            // threshold only on the crossing call.
            tracing::info!(
                session_id = ctx.session_id.id,
                tool = %offending_tool,
                count = max_count,
                "RepeatToolLimit: nudging — identical-call threshold reached"
            );
            ctx.follow_up_message = Some(self.config.nudge_message.clone());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FinishReason;

    fn mw() -> RepeatToolLimitMiddleware {
        RepeatToolLimitMiddleware::new(RepeatToolLimitConfig::default())
    }

    fn ctx(session: u64, calls: Vec<(&str, &str)>) -> PostLlmCtx {
        PostLlmCtx {
            session_id: SessionId::new(session),
            full_text: String::new(),
            is_tool_call: !calls.is_empty(),
            tool_calls: calls
                .into_iter()
                .enumerate()
                .map(|(i, (name, args))| (format!("call_{i}"), name.to_string(), args.to_string()))
                .collect(),
            available_tools: vec![],
            turn_count: 1,
            total_tool_calls: 0,
            nudge_count: 0,
            turn_tool_calls: 0,
            skip_push: false,
            follow_up_message: None,
            finish_reason: FinishReason::ToolUse,
        }
    }

    async fn call(mw: &RepeatToolLimitMiddleware, session: u64, calls: Vec<(&str, &str)>) {
        let mut c = ctx(session, calls);
        mw.on_post_llm(&mut c).await.unwrap();
    }

    #[tokio::test]
    async fn interleaved_identical_calls_still_count() {
        // The incident shape: polls interleaved with other work. A
        // consecutive counter would reset on every read_file; cumulative
        // counting must still reach the nudge at 5 identical calls.
        let mw = mw();
        for _ in 0..4 {
            call(
                &mw,
                1,
                vec![("list_agents", "{}"), ("read_file", r#"{"path":"a.rs"}"#)],
            )
            .await;
        }
        // 5th identical list_agents → nudge.
        let mut c = ctx(
            1,
            vec![("list_agents", "{}"), ("read_file", r#"{"path":"b.rs"}"#)],
        );
        mw.on_post_llm(&mut c).await.unwrap();
        assert!(c.follow_up_message.is_some(), "5th identical call nudges");
        assert!(
            !c.tool_calls.is_empty() && c.is_tool_call,
            "nudge is soft — calls still execute"
        );
    }

    #[tokio::test]
    async fn nudge_fires_only_once_per_fingerprint() {
        let mw = mw();
        let mut fired = 0;
        for _ in 0..8 {
            let mut c = ctx(1, vec![("list_agents", "{}")]);
            mw.on_post_llm(&mut c).await.unwrap();
            if c.follow_up_message.is_some() {
                fired += 1;
            }
        }
        assert_eq!(fired, 1, "nudge must fire exactly once, got {fired}");
    }

    #[tokio::test]
    async fn blocks_at_threshold_and_discards_pending_calls() {
        let mw = mw();
        for _ in 0..9 {
            call(&mw, 1, vec![("list_agents", "{}")]).await;
        }
        let mut c = ctx(
            1,
            vec![("list_agents", "{}"), ("read_file", r#"{"path":"x"}"#)],
        );
        mw.on_post_llm(&mut c).await.unwrap();
        assert!(c.tool_calls.is_empty(), "pending calls discarded");
        assert!(!c.is_tool_call, "response demoted to text-only");
        assert!(c.follow_up_message.is_some(), "block carries a follow-up");
    }

    #[tokio::test]
    async fn different_args_are_different_fingerprints() {
        // A model legitimately reading many different files must never trip.
        // Each probe uses a unique path so the probe itself does not
        // accumulate to the same fingerprint (a repeated probe would count
        // as an identical call on its own).
        let mw = mw();
        for i in 0..20 {
            call(
                &mw,
                1,
                vec![("read_file", &format!(r#"{{"path":"f{i}.rs"}}"#))],
            )
            .await;
            let mut probe = ctx(
                1,
                vec![("read_file", &format!(r#"{{"path":"probe{i}.rs"}}"#))],
            );
            mw.on_post_llm(&mut probe).await.unwrap();
            assert!(
                probe.follow_up_message.is_none() && probe.is_tool_call,
                "legitimate distinct calls must never be limited"
            );
        }
    }

    #[tokio::test]
    async fn same_args_reordered_keys_share_fingerprint() {
        let mw = mw();
        for _ in 0..4 {
            call(&mw, 1, vec![("wait_agent", r#"{"name":"a","timeout":5}"#)]).await;
        }
        // Same call, keys in the other order → still the 5th identical call.
        let mut c = ctx(1, vec![("wait_agent", r#"{"timeout":5,"name":"a"}"#)]);
        mw.on_post_llm(&mut c).await.unwrap();
        assert!(c.follow_up_message.is_some(), "canonical args must unify");
    }

    #[tokio::test]
    async fn on_user_message_resets_counts() {
        let mw = mw();
        for _ in 0..4 {
            call(&mw, 1, vec![("list_agents", "{}")]).await;
        }
        mw.on_user_message(&mut UserMessageCtx {
            session_id: SessionId::new(1),
            user_input: "next run".to_string(),
        })
        .await
        .unwrap();
        // Fresh run: 4 more calls are below the threshold again.
        for _ in 0..4 {
            let mut c = ctx(1, vec![("list_agents", "{}")]);
            mw.on_post_llm(&mut c).await.unwrap();
            assert!(c.follow_up_message.is_none(), "counts must reset per run");
        }
    }

    #[tokio::test]
    async fn sessions_are_isolated() {
        let mw = mw();
        for _ in 0..4 {
            call(&mw, 1, vec![("list_agents", "{}")]).await;
        }
        let mut other = ctx(2, vec![("list_agents", "{}")]);
        mw.on_post_llm(&mut other).await.unwrap();
        assert!(
            other.follow_up_message.is_none(),
            "session 2's first call must not see session 1's count"
        );
    }
}
