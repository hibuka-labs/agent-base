use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use tokio::sync::broadcast;

use crate::engine::executor::{ActionExecutor, ExecCtx, ExecError, ParseError, ParseOutcome};
use crate::engine::runtime::LlmTurnResult;
use crate::types::{Action, AgentResult, Observation, RuntimeEvent};

use super::{ToolCallExecutor, actions_to_triples};

fn noop_event(_event: RuntimeEvent) -> AgentResult<()> {
    Ok(())
}

/// The seam face (M3). `parse` is the live four-state classifier; `execute` is
/// the observation-shaped batch core. Flow-face debt: the react loop still
/// drives turns through `run_actions` (truncation rescue, breaker strikes, and
/// run-level flow signals are turn plumbing `ExecCtx` does not yet carry —
/// T8 wired the interaction channel and its guards, T10 switches the loop to
/// this trait).
#[async_trait]
impl<'c> ActionExecutor for ToolCallExecutor<'c> {
    fn parse(&self, turn: &LlmTurnResult) -> Result<ParseOutcome, ParseError> {
        Ok(self.classify(turn))
    }

    async fn execute(
        &mut self,
        actions: Vec<Action>,
        ctx: &ExecCtx,
    ) -> Result<Observation, ExecError> {
        if actions.is_empty() {
            return Err(ExecError::NothingRan {
                reason: "empty batch".into(),
            });
        }
        // Per-action fates are outcomes (D9); ExecError is reserved for "not a
        // single action got off the ground" (D15). Interactions from inside
        // the batch ride `ctx` (T8: consent/ask_user suspend there, on the
        // run's cancel token); the batch's own cancel wiring and live event
        // streaming need run plumbing (T10) — events ride a sinkless channel
        // here.
        let _ = ctx;
        let triples = actions_to_triples(&actions);
        let (_event_tx, mut event_rx) = broadcast::channel(8);
        let on_event: Arc<Mutex<fn(RuntimeEvent) -> AgentResult<()>>> =
            Arc::new(Mutex::new(noop_event));
        match self
            .handle_tool_calls(
                &self.session_id,
                &triples,
                &[],
                &mut event_rx,
                on_event,
                "",
                "",
            )
            .await
        {
            Ok(batch) => Ok(batch.observation),
            Err(e) => Err(ExecError::NothingRan {
                reason: e.to_string(),
            }),
        }
    }
}
