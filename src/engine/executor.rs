//! ActionExecutor seam (M3 design §3) — the kernel's one hole for "how a turn's
//! output becomes actions, and how actions become observations".
//!
//! Domain-neutral by construction: the kernel knows nothing about tool calls or
//! CodeAct. Backends implement [`ActionExecutor`] (and own their history
//! protocol — the kernel pushes [`Observation::history_messages`] verbatim).
//! Session life-and-death is a separate axis ([`SessionHandle`]) so neither
//! trait becomes a god-face (OV-12).
//!
//! Contracts encoded here:
//! - **Four-state parse** (D4+OV-7): [`ParseOutcome::Actions`] /
//!   [`ParseOutcome::Empty`] / [`ParseOutcome::Incomplete`] /
//!   [`ParseOutcome::Malformed`] map onto the loop's existing branches
//!   (execute / nudge / nudge-retry / salvage-breaker).
//! - **Observation always returns** (D15): once anything ran, `execute`
//!   returns `Ok(Observation)` carrying `batch_status` (even `EnvDied`);
//!   [`ExecError`] means not a single action got off the ground.
//! - **Two-layer error face** (D9): per-action refusals and failures are
//!   [`ActionOutcome`]s (model-visible), never error variants.

use std::future::Future;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

use crate::engine::runtime::LlmTurnResult;
use crate::types::{
    Action, ActionFormat, ActionOutcome, ApprovalDecision, ApprovalRequest, AskAnswer, AskOption,
    ChatMessage, Observation,
};

/// The four-state verdict of `parse` (D4+OV-7).
#[derive(Clone, Debug, PartialEq)]
pub enum ParseOutcome {
    /// This turn's output is one or more backend-format actions.
    Actions(Vec<Action>),
    /// Nothing actionable — the loop's old empty branch (nudge).
    Empty,
    /// Truncation salvage ran and the turn is still incomplete —
    /// the loop's nudge-retry branch.
    Incomplete,
    /// Structural anomaly after salvage — the loop's old anomaly branch
    /// (existing breaker / salvage semantics).
    Malformed,
}

impl ParseOutcome {
    /// True when this turn's output is action work (materialized or not):
    /// such turns are never text-nudged or text-pushed.
    pub fn has_action_intent(&self) -> bool {
        matches!(self, ParseOutcome::Actions(_) | ParseOutcome::Incomplete)
    }
}

/// `parse` failed to classify at all (internal invariant broken).
#[derive(thiserror::Error, Debug)]
pub enum ParseError {
    #[error("turn output could not be classified: {0}")]
    Unparseable(String),
}

/// Batch-level execution failure — reserved for "not one action ran" (D15).
#[derive(thiserror::Error, Debug)]
pub enum ExecError {
    #[error("batch did not run: {reason}")]
    NothingRan { reason: String },
}

/// Session lifecycle failure ([`SessionHandle::rebuild`]).
#[derive(thiserror::Error, Debug)]
pub enum SessionError {
    #[error("session rebuild failed: {0}")]
    RebuildFailed(String),
}

/// The product layer failed to carry an interaction to completion.
#[derive(thiserror::Error, Debug)]
pub enum InteractionError {
    #[error("interaction timed out: {0}")]
    Timeout(String),
    #[error("interaction cancelled")]
    Cancelled,
    #[error("interaction channel failed: {0}")]
    Channel(String),
}

/// Default per-interaction watchdog ([`ExecCtx::interact_timeout`]) — an
/// unanswered suspension must not deadlock the run (M3 failure table: an
/// unanswered consent hangs forever). Generous, because this is human reading
/// time; a product that owns the human relationship (phimint's TUI) clears it
/// to `None` and leans on the run's cancel token alone.
pub const DEFAULT_INTERACT_TIMEOUT: Duration = Duration::from_secs(600);

/// Product-layer interaction surface hung off [`ExecCtx`] (OV-6): consent and
/// ask_user suspend the executing action and wait for the product to answer.
/// Routed through the host's single throat — the same seam policy and journal
/// already use. `cancel` is the run's one token (interrupt / timeout /
/// watchdog); implementations must release promptly on it.
#[async_trait]
pub trait InteractionChannel: Send + Sync {
    /// Consent / approval request; suspends until the product layer answers.
    async fn request_consent(
        &self,
        request: ApprovalRequest,
        cancel: &CancellationToken,
    ) -> Result<ApprovalDecision, InteractionError>;

    /// Ask the user to pick between candidates; suspends until answered.
    /// A skip is an answer too ([`AskAnswer::Skipped`]) — never re-ask.
    async fn ask_user(
        &self,
        question: &str,
        options: &[AskOption],
        cancel: &CancellationToken,
    ) -> Result<AskAnswer, InteractionError>;
}

/// Execution context threaded through `execute` (OV-6): one cancel token for
/// user interrupt / timeout / watchdog, one interaction channel for consent
/// and ask_user. Reach the channel only through [`ExecCtx::request_consent`] /
/// [`ExecCtx::ask_user`] — those carry the suspension discipline (answer /
/// cancel / watchdog deadline); calling the trait directly drops it.
pub struct ExecCtx {
    /// User interrupt / timeout / watchdog share this token.
    pub cancel: CancellationToken,
    /// Consent / ask_user suspension points.
    pub interact: Arc<dyn InteractionChannel>,
    /// Per-interaction deadline: an unanswered suspension cancels the run's
    /// token and fails with [`InteractionError::Timeout`] instead of hanging
    /// forever. `None` = the run's cancel token is the only guard (human time
    /// is unbounded — phimint's TUI sets this).
    pub interact_timeout: Option<Duration>,
}

impl ExecCtx {
    /// One run's execution context. Interactions carry the
    /// [`DEFAULT_INTERACT_TIMEOUT`] watchdog; the construction site overrides
    /// (or clears) `interact_timeout` when it knows better.
    pub fn new(cancel: CancellationToken, interact: Arc<dyn InteractionChannel>) -> Self {
        Self {
            cancel,
            interact,
            interact_timeout: Some(DEFAULT_INTERACT_TIMEOUT),
        }
    }

    /// Set the per-interaction deadline (`None` = wait as long as the run lives).
    pub fn with_interact_timeout(mut self, interact_timeout: Option<Duration>) -> Self {
        self.interact_timeout = interact_timeout;
        self
    }

    /// Consent request: suspends on the product layer, released by the answer,
    /// the cancel token, or the interaction watchdog.
    pub async fn request_consent(
        &self,
        request: ApprovalRequest,
    ) -> Result<ApprovalDecision, InteractionError> {
        let call = self.interact.request_consent(request, &self.cancel);
        self.guard(call).await
    }

    /// Ask the user to pick between candidates: suspends like
    /// [`ExecCtx::request_consent`] and shares its guards.
    pub async fn ask_user(
        &self,
        question: &str,
        options: &[AskOption],
    ) -> Result<AskAnswer, InteractionError> {
        let call = self.interact.ask_user(question, options, &self.cancel);
        self.guard(call).await
    }

    /// The suspension discipline (failure table: an unanswered interaction
    /// must not deadlock the run): answer first, then cancel, then the
    /// watchdog — which fires through cancel too, so every other suspension
    /// point and the batch unwind on the same token.
    async fn guard<T>(
        &self,
        call: impl Future<Output = Result<T, InteractionError>>,
    ) -> Result<T, InteractionError> {
        tokio::select! {
            biased;
            answer = call => answer,
            _ = self.cancel.cancelled() => Err(InteractionError::Cancelled),
            waited = async {
                match self.interact_timeout {
                    Some(d) => {
                        tokio::time::sleep(d).await;
                        d
                    }
                    None => std::future::pending().await,
                }
            } => {
                self.cancel.cancel();
                Err(InteractionError::Timeout(format!(
                    "no answer within {waited:?}; the run was cancelled"
                )))
            }
        }
    }
}

/// The seam: one turn's output in, one observation out.
#[async_trait]
pub trait ActionExecutor: Send + Sync {
    /// Turn output → this backend's action format. Truncation salvage (structural
    /// verdict + text-channel harvest) is a parse-end move; stream aggregation
    /// stays in the provider layer. CodeAct implementations must validate the
    /// full batch before executing anything (D2).
    fn parse(&self, turn: &LlmTurnResult) -> Result<ParseOutcome, ParseError>;

    /// Execute a batch and return the observation. Never dropped because the
    /// environment died (D15): if anything ran, it's `Ok(Observation)` with
    /// `batch_status` set; `Err(ExecError)` only when nothing ran.
    async fn execute(
        &mut self,
        actions: Vec<Action>,
        ctx: &ExecCtx,
    ) -> Result<Observation, ExecError>;
}

/// Session life-and-death (OV-12): a different axis from the action face.
#[async_trait]
pub trait SessionHandle: Send + Sync {
    /// Whether the execution environment is still usable.
    fn is_alive(&self) -> bool;

    /// Rebuild after `is_alive()` goes false. Failure fails the whole turn —
    /// no retry loop on a dead environment (D10).
    async fn rebuild(&mut self) -> Result<(), SessionError>;
}

/// Model-visible fate for an action the crash window cut off (OV-4). The
/// kernel owns this vocabulary and hands it to the backend inside
/// [`ActionOutcome::summary`]; a backend only renders the fate in its own
/// history shape ([`ActionBackend::render_fates`]).
pub const INTERRUPTED_SUMMARY: &str = "已中断/未执行";

/// The kernel's mount face for one runtime's injected action backend (T10c).
/// The built-in tool-call backend keeps its own flow; this face serves any
/// other action format one session can interleave with it (envelope turns go
/// to tools, text turns go here).
///
/// Supertraits are the M3a seam contracts. The extra methods are the two
/// things the kernel needs to route a session around a backend — its format
/// (checkpoint pin / resume affinity, T5) and its history shape for a batch
/// that produced no observation (dangling-batch synthesis at resume, OV-4;
/// `Err(ExecError)` fates, D15). The kernel pushes the rendered messages
/// verbatim, same as [`Observation::history_messages`].
#[async_trait]
pub trait ActionBackend: ActionExecutor + SessionHandle {
    /// This backend's action format. Must not be [`ActionFormat::ToolCall`] —
    /// the built-in tool-call backend owns that format and its flow.
    fn format(&self) -> ActionFormat;

    /// Render one batch's history from per-action fates in this backend's own
    /// protocol shape — the messages ride [`ActionOutcome`]s' summaries
    /// (including [`INTERRUPTED_SUMMARY`]) and are pushed as-is.
    fn render_fates(&self, actions: &[Action], fates: &[ActionOutcome]) -> Vec<ChatMessage>;
}

/// The fail-closed interaction surface for a runtime that mounted an action
/// backend but no product channel: every human request refuses with the gap
/// named. An unanswered interaction must fail closed — never hang, never
/// silently allow.
pub(crate) struct FailClosedInteract;

#[async_trait]
impl InteractionChannel for FailClosedInteract {
    async fn request_consent(
        &self,
        _request: ApprovalRequest,
        _cancel: &CancellationToken,
    ) -> Result<ApprovalDecision, InteractionError> {
        Err(InteractionError::Channel(
            "no interaction channel mounted on this runtime".into(),
        ))
    }

    async fn ask_user(
        &self,
        _question: &str,
        _options: &[AskOption],
        _cancel: &CancellationToken,
    ) -> Result<AskAnswer, InteractionError> {
        Err(InteractionError::Channel(
            "no interaction channel mounted on this runtime".into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ActionFormat, ActionOutcome, BatchStatus};
    use std::sync::Mutex;
    use tokio::sync::{mpsc, oneshot};

    fn turn() -> LlmTurnResult {
        LlmTurnResult {
            full_text: String::new(),
            reasoning_text: String::new(),
            is_tool_call: true,
            tool_calls: vec![serde_json::json!({
                "id": "call_1", "name": "echo", "arguments": "{\"x\":1}"
            })],
            usage: None,
            finish_reason: Some("tool_calls".into()),
            ttft_ms: 1,
            llm_duration_ms: 2,
            reasoning_only: false,
            thinking_signature: None,
        }
    }

    fn action(id: &str) -> Action {
        Action {
            id: id.into(),
            name: "echo".into(),
            args: serde_json::Value::String(r#"{"x":1}"#.into()),
            identity: r#"echo::{"x":1}"#.into(),
            format: ActionFormat::ToolCall,
        }
    }

    /// Test double standing in for a backend (the tool-call executor lands in
    /// T4; this pins the contract shape in the meantime).
    struct DummyExecutor;

    #[async_trait]
    impl ActionExecutor for DummyExecutor {
        fn parse(&self, turn: &LlmTurnResult) -> Result<ParseOutcome, ParseError> {
            if turn.is_tool_call && !turn.tool_calls.is_empty() {
                Ok(ParseOutcome::Actions(vec![action("call_1")]))
            } else {
                Ok(ParseOutcome::Empty)
            }
        }

        async fn execute(
            &mut self,
            actions: Vec<Action>,
            _ctx: &ExecCtx,
        ) -> Result<Observation, ExecError> {
            if actions.is_empty() {
                return Err(ExecError::NothingRan {
                    reason: "empty batch".into(),
                });
            }
            Ok(Observation {
                outcomes: actions
                    .iter()
                    .map(|a| ActionOutcome {
                        action_id: a.id.clone(),
                        name: a.name.clone(),
                        ok: true,
                        denied: false,
                        summary: "1".into(),
                        details: None,
                    })
                    .collect(),
                events: vec![],
                history_messages: vec![],
                batch_status: BatchStatus::Completed,
            })
        }
    }

    struct NoopInteract;

    #[async_trait]
    impl InteractionChannel for NoopInteract {
        async fn request_consent(
            &self,
            _request: ApprovalRequest,
            _cancel: &CancellationToken,
        ) -> Result<ApprovalDecision, InteractionError> {
            Ok(ApprovalDecision::AllowOnce)
        }

        async fn ask_user(
            &self,
            _question: &str,
            _options: &[AskOption],
            _cancel: &CancellationToken,
        ) -> Result<AskAnswer, InteractionError> {
            Ok(AskAnswer::Picked {
                id: "a".into(),
                label: "ok".into(),
            })
        }
    }

    fn ctx() -> ExecCtx {
        ExecCtx::new(CancellationToken::new(), Arc::new(NoopInteract))
    }

    fn approval_request() -> ApprovalRequest {
        ApprovalRequest {
            title: "t".into(),
            message: "m".into(),
            action_key: None,
            risk_level: crate::types::RiskLevel::Safe,
            raw: None,
            source: None,
        }
    }

    #[tokio::test]
    async fn parse_four_states_are_distinct() {
        let exec = DummyExecutor;
        assert!(matches!(
            exec.parse(&turn()).unwrap(),
            ParseOutcome::Actions(_)
        ));
        let mut empty_turn = turn();
        empty_turn.is_tool_call = false;
        empty_turn.tool_calls = vec![];
        assert_eq!(exec.parse(&empty_turn).unwrap(), ParseOutcome::Empty);
        // Incomplete / Malformed are backend verdicts; the contract is that all
        // four states are expressible and distinct.
        assert_ne!(
            std::mem::discriminant(&ParseOutcome::Incomplete),
            std::mem::discriminant(&ParseOutcome::Malformed)
        );
    }

    #[tokio::test]
    async fn execute_returns_observation_when_anything_ran() {
        let mut exec = DummyExecutor;
        let obs = exec.execute(vec![action("call_1")], &ctx()).await.unwrap();
        assert_eq!(obs.batch_status, BatchStatus::Completed);
        assert_eq!(obs.outcomes.len(), 1);
        assert!(obs.outcomes[0].ok);
    }

    #[tokio::test]
    async fn execute_errors_only_when_nothing_ran() {
        let mut exec = DummyExecutor;
        let err = exec.execute(vec![], &ctx()).await.unwrap_err();
        assert!(matches!(err, ExecError::NothingRan { .. }));
    }

    #[tokio::test]
    async fn interaction_channel_round_trips() {
        let ctx = ctx();
        let decision = ctx.request_consent(approval_request()).await.unwrap();
        assert_eq!(decision, ApprovalDecision::AllowOnce);
        assert_eq!(
            ctx.ask_user("?", &[]).await.unwrap(),
            AskAnswer::Picked {
                id: "a".into(),
                label: "ok".into(),
            }
        );
    }

    /// A channel whose answers arrive on a oneshot the test holds — the
    /// product side of a suspension (in a cell, the host-call round trip).
    /// An uncompleted oneshot is "nobody answered".
    struct ScriptedInteract {
        arrived: mpsc::UnboundedSender<&'static str>,
        consent: Mutex<Option<oneshot::Receiver<ApprovalDecision>>>,
        ask: Mutex<Option<oneshot::Receiver<AskAnswer>>>,
    }

    #[async_trait]
    impl InteractionChannel for ScriptedInteract {
        async fn request_consent(
            &self,
            _request: ApprovalRequest,
            _cancel: &CancellationToken,
        ) -> Result<ApprovalDecision, InteractionError> {
            let _ = self.arrived.send("consent");
            let rx = self
                .consent
                .lock()
                .unwrap()
                .take()
                .expect("one consent answer per test");
            rx.await
                .map_err(|_| InteractionError::Channel("answer dropped".into()))
        }

        async fn ask_user(
            &self,
            _question: &str,
            _options: &[AskOption],
            _cancel: &CancellationToken,
        ) -> Result<AskAnswer, InteractionError> {
            let _ = self.arrived.send("ask");
            let rx = self
                .ask
                .lock()
                .unwrap()
                .take()
                .expect("one ask answer per test");
            rx.await
                .map_err(|_| InteractionError::Channel("answer dropped".into()))
        }
    }

    fn scripted(
        consent: Option<oneshot::Receiver<ApprovalDecision>>,
        ask: Option<oneshot::Receiver<AskAnswer>>,
    ) -> (Arc<ScriptedInteract>, mpsc::UnboundedReceiver<&'static str>) {
        let (arrived_tx, arrived_rx) = mpsc::unbounded_channel();
        (
            Arc::new(ScriptedInteract {
                arrived: arrived_tx,
                consent: Mutex::new(consent),
                ask: Mutex::new(ask),
            }),
            arrived_rx,
        )
    }

    /// Scenario 1 (T8): a consent from inside the batch suspends on the
    /// product layer and resumes with the answer.
    #[tokio::test]
    async fn consent_suspends_until_the_product_answers() {
        let (answer_tx, answer_rx) = oneshot::channel();
        let (interact, mut arrived) = scripted(Some(answer_rx), None);
        let ctx = ExecCtx::new(CancellationToken::new(), interact);
        let call = tokio::spawn(async move { ctx.request_consent(approval_request()).await });
        assert_eq!(
            arrived.recv().await,
            Some("consent"),
            "the suspension is parked"
        );
        answer_tx.send(ApprovalDecision::AllowAlways).unwrap();
        assert_eq!(call.await.unwrap().unwrap(), ApprovalDecision::AllowAlways);
    }

    /// Scenario 2 (T8): an ask suspends and resumes with the user's pick.
    #[tokio::test]
    async fn ask_user_suspends_until_the_user_picks() {
        let (answer_tx, answer_rx) = oneshot::channel();
        let (interact, mut arrived) = scripted(None, Some(answer_rx));
        let ctx = ExecCtx::new(CancellationToken::new(), interact);
        let call = tokio::spawn(async move {
            ctx.ask_user(
                "which first?",
                &[AskOption {
                    id: "a".into(),
                    label: "search".into(),
                    note: None,
                }],
            )
            .await
        });
        assert_eq!(arrived.recv().await, Some("ask"));
        answer_tx
            .send(AskAnswer::Picked {
                id: "a".into(),
                label: "search".into(),
            })
            .unwrap();
        assert_eq!(
            call.await.unwrap().unwrap(),
            AskAnswer::Picked {
                id: "a".into(),
                label: "search".into(),
            }
        );
    }

    /// Scenario 3 (T8): interrupt — the run's cancel token releases a
    /// suspended interaction instead of leaving it hanging.
    #[tokio::test]
    async fn interrupt_cancels_a_suspended_interaction() {
        let (answer_tx, answer_rx) = oneshot::channel();
        let (interact, mut arrived) = scripted(Some(answer_rx), None);
        let token = CancellationToken::new();
        let ctx = ExecCtx::new(token.clone(), interact);
        let call = tokio::spawn(async move { ctx.request_consent(approval_request()).await });
        assert_eq!(arrived.recv().await, Some("consent"));
        token.cancel();
        assert!(matches!(
            call.await.unwrap(),
            Err(InteractionError::Cancelled)
        ));
        drop(answer_tx);
    }

    /// Failure-table guard: an unanswered interaction must not deadlock —
    /// the watchdog fires, the run is cancelled through the same token
    /// (timeout goes through cancel), and the error says so in words.
    #[tokio::test]
    async fn an_unanswered_interaction_fails_the_run_through_cancel() {
        let (answer_tx, answer_rx) = oneshot::channel();
        let (interact, mut arrived) = scripted(Some(answer_rx), None);
        let token = CancellationToken::new();
        let ctx = ExecCtx::new(token.clone(), interact)
            .with_interact_timeout(Some(Duration::from_millis(50)));
        let call = tokio::spawn(async move { ctx.request_consent(approval_request()).await });
        assert_eq!(arrived.recv().await, Some("consent"));
        let err = call.await.unwrap().unwrap_err();
        let msg = err.to_string();
        assert!(matches!(&err, InteractionError::Timeout(_)), "{msg}");
        assert!(
            msg.contains("no answer within 50ms"),
            "timeout wording: {msg}"
        );
        assert!(
            msg.contains("the run was cancelled"),
            "timeout wording: {msg}"
        );
        assert!(
            token.is_cancelled(),
            "timeout must fire through the run's cancel token"
        );
        drop(answer_tx);
    }

    struct DeadSession;

    #[async_trait]
    impl SessionHandle for DeadSession {
        fn is_alive(&self) -> bool {
            false
        }

        async fn rebuild(&mut self) -> Result<(), SessionError> {
            Err(SessionError::RebuildFailed("worker gone".into()))
        }
    }

    #[tokio::test]
    async fn session_handle_rebuild_failure_is_explicit() {
        let mut handle = DeadSession;
        assert!(!handle.is_alive());
        assert!(matches!(
            handle.rebuild().await,
            Err(SessionError::RebuildFailed(_))
        ));
    }
}
