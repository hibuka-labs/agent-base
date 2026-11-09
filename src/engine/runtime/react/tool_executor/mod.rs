//! Tool-call backend for the M3 `ActionExecutor` seam: the `ToolCallExecutor`
//! owns everything the tool protocol needs — envelope parse, truncation rescue,
//! batch execution, and history rendering. The kernel sees only
//! [`ParseOutcome`] / [`Observation`] / [`ExecError`].
//!
//! The `(id, name, args)` triple is internal to this backend; the kernel
//! currency is [`Action`]. `run_actions` keeps the react loop's turn-level
//! flow (breaker strikes, turn-end metrics) — full flow-face unification on
//! the trait is T10 debt (see the `ActionExecutor` impl below).

mod batch;
mod errors;
mod run_actions;
mod seam;

use serde_json::Value;

use crate::engine::executor::ParseOutcome;
use crate::engine::runtime::LlmTurnResult;
use crate::engine::runtime::plan_runner::RuntimeCore;
use crate::types::{Action, AgentError, FinishReason, Observation, SessionId};

/// A finished batch: the observation (per-action fates + rendered history) and,
/// for the flow path, the first per-call failure to feed recovery.
pub(super) struct ToolBatchReport {
    pub observation: Observation,
    /// First per-call failure (an outcome, not an error — D9); `None` when
    /// every call had a success fate.
    pub first_failure: Option<AgentError>,
}

/// The tool-call action backend (M3 seam).
pub(super) struct ToolCallExecutor<'c> {
    core: &'c RuntimeCore,
    session_id: SessionId,
}

/// Internal triple currency: `Action` → backend triple.
pub(super) fn actions_to_triples(actions: &[Action]) -> Vec<(String, String, String)> {
    actions
        .iter()
        .map(|a| (a.id.clone(), a.name.clone(), a.raw_args().into_owned()))
        .collect()
}

/// Internal triple currency: backend triple → `Action` (identity recomputed
/// from the current — possibly salvaged — argument text).
pub(super) fn triples_to_actions(tool_calls: &[(String, String, String)]) -> Vec<Action> {
    tool_calls
        .iter()
        .map(|(id, name, args)| Action::from_tool_call(id.clone(), name.clone(), args.clone()))
        .collect()
}
impl<'c> ToolCallExecutor<'c> {
    pub(super) fn new(core: &'c RuntimeCore, session_id: SessionId) -> Self {
        Self { core, session_id }
    }

    /// Turn output → four-state verdict (D4+OV-7). Envelope flattening +
    /// classification only — no schema-dependent salvage here (that runs on
    /// the execute side, where schema lookup is async).
    pub(super) fn classify(&self, turn: &LlmTurnResult) -> ParseOutcome {
        classify_turn(turn)
    }
}

/// Four-state classification of one turn (D4+OV-7). State-free: pure over the
/// provider envelope.
fn classify_turn(turn: &LlmTurnResult) -> ParseOutcome {
    let actions = triples_to_actions(&flatten_envelope(turn));
    if turn.is_tool_call {
        if actions.is_empty() {
            // Tool intent signalled but nothing materialized.
            ParseOutcome::Incomplete
        } else {
            ParseOutcome::Actions(actions)
        }
    } else if !turn.full_text.is_empty() {
        let finish = FinishReason::from_raw(turn.finish_reason.as_deref());
        // Intent without a call: truncated finish or a ToolUse mislabel —
        // the old finish-anomaly branch.
        if finish.is_truncated() || matches!(finish, FinishReason::ToolUse) {
            ParseOutcome::Malformed
        } else {
            ParseOutcome::Empty
        }
    } else {
        ParseOutcome::Empty
    }
}

/// Flatten the provider envelope's tool calls into backend triples.
fn flatten_envelope(turn: &LlmTurnResult) -> Vec<(String, String, String)> {
    turn.tool_calls
        .iter()
        .map(|tc| {
            let id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let name = tc
                .get("function")
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("")
                .to_string();
            let args = tc
                .get("function")
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .unwrap_or("")
                .to_string();
            (id, name, args)
        })
        .collect()
}

/// Names of the tools that actually failed in a batch of tool calls.
///
/// Tool-level errors carry the failing tool's name on the error itself, so a single
/// bad call in a large batch yields exactly one name (not every tool in the batch,
/// which would inflate the consecutive-failure counter and trip the Stop threshold).
/// Unknown error types conservatively fall back to the full batch.
fn failing_tool_names(tool_calls: &[(String, String, String)], error: &AgentError) -> Vec<String> {
    match error {
        AgentError::ToolArgsInvalid { name, .. } => vec![name.clone()],
        AgentError::ToolExecution { name, .. } => vec![name.clone()],
        _ => tool_calls.iter().map(|(_, n, _)| n.clone()).collect(),
    }
}

/// True when a tool call's (valid) JSON arguments are actually a truncation
/// error wrapper the model echoed back from history, rather than a real call.
/// Such an object parses fine, so the structural (invalid-JSON) guard can't see
/// it, but executing it only reproduces a ToolArgsInvalid failure. Detected by
/// its distinctive marker key (plus the preview field it always carried).
fn is_truncation_wrapper(args: &Value) -> bool {
    let obj = match args.as_object() {
        Some(o) => o,
        None => return false,
    };
    let marker = obj
        .get("error")
        .and_then(Value::as_str)
        .is_some_and(|s| s == "tool_call_arguments_truncated");
    let preview = obj.contains_key("original_args_preview");
    marker || (preview && obj.contains_key("message"))
}

#[cfg(test)]
mod parse_fixtures;
#[cfg(test)]
mod tests;
