use std::collections::HashMap;
use std::sync::Mutex;

use crate::types::{AgentError, AgentResult, SessionId};

/// Action taken by the runtime after a tool execution failure
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolErrorAction {
    /// Stop the current run with a failed outcome
    Stop,
    /// Feed the error back to the LLM and continue reasoning
    Retry,
    /// Feed the full error history back to the LLM and let it evaluate.
    ///
    /// Emitted by [`ConsecutiveFailureRecovery`] when the same tool fails
    /// `max_consecutive_failures` times in a row. Gives the LLM one grace
    /// round to read the full error stack and either switch strategy or
    /// explain the failure to the user.
    RetryWithHistory {
        /// Collected error messages from the consecutive failure streak.
        errors: Vec<String>,
    },
}

/// Recovery strategy after a tool execution failure
///
/// Defaults to [`StopOnError`], following the lightweight kernel design of
/// conservative defaults and strategy injection.
/// Upper-layer agents can inject custom strategies such as [`RetryOnError`].
pub trait ToolErrorRecovery: Send + Sync {
    fn on_error(
        &self,
        _session_id: &SessionId,
        _tool_names: &[String],
        _error: &AgentError,
    ) -> AgentResult<ToolErrorAction>;

    /// Called when a tool executes successfully.
    ///
    /// Default no-op. Strategies that track consecutive failures (e.g.
    /// [`ConsecutiveFailureRecovery`]) override this to reset that tool's counter so
    /// the "consecutive" semantics hold: a success breaks the failure streak.
    fn on_success(&self, _session_id: &SessionId, _tool_name: &str) {}
}

/// Default strategy: stop on tool failure.
///
/// This is the most conservative strategy. The kernel only reports the fact
/// without making business recovery decisions for the upper layer.
pub struct StopOnError;

impl ToolErrorRecovery for StopOnError {
    fn on_error(
        &self,
        _session_id: &SessionId,
        _tool_names: &[String],
        _error: &AgentError,
    ) -> AgentResult<ToolErrorAction> {
        Ok(ToolErrorAction::Stop)
    }
}

/// Continue on tool failure, feeding the error back to the model
///
/// Suitable for scenarios where model self-healing is desired (e.g. code-agent, browser-agent).
pub struct RetryOnError;

impl ToolErrorRecovery for RetryOnError {
    fn on_error(
        &self,
        _session_id: &SessionId,
        _tool_names: &[String],
        _error: &AgentError,
    ) -> AgentResult<ToolErrorAction> {
        Ok(ToolErrorAction::Retry)
    }
}

/// Recovery strategy that tracks consecutive failures per tool and stops after a limit.
///
/// This prevents runaway retry loops where the model keeps calling the same failing
/// tool (e.g., `execute_ssh_command` failing due to auth, but the model retries
/// indefinitely). After `max_consecutive_failures` for the same tool name, the
/// strategy switches from `Retry` to `Stop` with a summary message.
///
/// Failure counters are tracked per session and reset when a different tool
/// succeeds or a new session starts.
pub struct ConsecutiveFailureRecovery {
    max_consecutive_failures: usize,
    /// session_id -> (tool_name -> consecutive_failures)
    failure_counts: Mutex<HashMap<u64, HashMap<String, usize>>>,
    /// session_id -> (tool_name -> collected error messages for the current streak)
    error_messages: Mutex<HashMap<u64, HashMap<String, Vec<String>>>>,
    /// session_id -> (tool_name -> whether RetryWithHistory grace was already used)
    grace_used: Mutex<HashMap<u64, HashMap<String, bool>>>,
}

impl ConsecutiveFailureRecovery {
    pub fn new(max_consecutive_failures: usize) -> Self {
        Self {
            max_consecutive_failures,
            failure_counts: Mutex::new(HashMap::new()),
            error_messages: Mutex::new(HashMap::new()),
            grace_used: Mutex::new(HashMap::new()),
        }
    }

    /// Reset failure count, error messages, and grace flag for a specific tool in a session.
    /// Call this when a tool succeeds to avoid false positives.
    pub fn reset_failures(&self, session_id: &SessionId, tool_name: &str) {
        if let Ok(mut counts) = self.failure_counts.lock()
            && let Some(session_counts) = counts.get_mut(&session_id.id)
        {
            session_counts.remove(tool_name);
        }
        if let Ok(mut msgs) = self.error_messages.lock()
            && let Some(session_msgs) = msgs.get_mut(&session_id.id)
        {
            session_msgs.remove(tool_name);
        }
        if let Ok(mut grace) = self.grace_used.lock()
            && let Some(session_grace) = grace.get_mut(&session_id.id)
        {
            session_grace.remove(tool_name);
        }
    }

    /// Reset all failure counts, error messages, and grace flags for a session.
    /// Call this when a new turn starts.
    pub fn reset_session(&self, session_id: &SessionId) {
        if let Ok(mut counts) = self.failure_counts.lock() {
            counts.remove(&session_id.id);
        }
        if let Ok(mut msgs) = self.error_messages.lock() {
            msgs.remove(&session_id.id);
        }
        if let Ok(mut grace) = self.grace_used.lock() {
            grace.remove(&session_id.id);
        }
    }
}

impl ToolErrorRecovery for ConsecutiveFailureRecovery {
    fn on_error(
        &self,
        session_id: &SessionId,
        tool_names: &[String],
        error: &AgentError,
    ) -> AgentResult<ToolErrorAction> {
        let mut counts = self
            .failure_counts
            .lock()
            .map_err(|e| AgentError::internal(format!("Failed to lock failure counts: {}", e)))?;
        let mut msgs = self
            .error_messages
            .lock()
            .map_err(|e| AgentError::internal(format!("Failed to lock error messages: {}", e)))?;
        let mut grace = self
            .grace_used
            .lock()
            .map_err(|e| AgentError::internal(format!("Failed to lock grace_used: {}", e)))?;

        let session_counts = counts.entry(session_id.id).or_insert_with(HashMap::new);
        let session_msgs = msgs.entry(session_id.id).or_insert_with(HashMap::new);
        let session_grace = grace.entry(session_id.id).or_insert_with(HashMap::new);

        // Record error message and increment failure count for each failing tool
        let error_text = error.to_string();
        let mut max_failures = 0;
        let mut threshold_tool: Option<String> = None;
        for name in tool_names {
            let count = session_counts.entry(name.clone()).or_insert(0);
            *count += 1;
            session_msgs
                .entry(name.clone())
                .or_insert_with(Vec::new)
                .push(error_text.clone());
            if *count > max_failures {
                max_failures = *count;
            }
            if *count >= self.max_consecutive_failures {
                threshold_tool = Some(name.clone());
            }
        }

        if max_failures >= self.max_consecutive_failures {
            let tool_name = threshold_tool.unwrap_or_else(|| tool_names[0].clone());

            tracing::warn!(
                session_id = session_id.id,
                tool = %tool_name,
                failures = max_failures,
                max_consecutive_failures = self.max_consecutive_failures,
                "ConsecutiveFailureRecovery: threshold reached"
            );

            // Check if grace was already used for this tool
            let already_used = session_grace.get(&tool_name).copied().unwrap_or(false);

            if already_used {
                // Grace exhausted → hard stop. Clear all state for this session.
                counts.remove(&session_id.id);
                msgs.remove(&session_id.id);
                grace.remove(&session_id.id);
                return Ok(ToolErrorAction::Stop);
            }

            // First time at threshold → give LLM one grace round
            session_grace.insert(tool_name.clone(), true);

            let errors = session_msgs.get(&tool_name).cloned().unwrap_or_default();

            // Clear messages but keep counts and grace flag so that the next
            // failure for the same tool triggers Stop (not another RetryWithHistory).
            msgs.remove(&session_id.id);

            return Ok(ToolErrorAction::RetryWithHistory { errors });
        }

        Ok(ToolErrorAction::Retry)
    }

    fn on_success(&self, session_id: &SessionId, tool_name: &str) {
        self.reset_failures(session_id, tool_name);
    }
}

// ---------------------------------------------------------------------------
// Invalid tool arguments (schema-level, pre-execution)
// ---------------------------------------------------------------------------

/// A tool call whose JSON arguments parse but omit fields the tool's schema
/// declares as required.
///
/// Distinct from [`ToolErrorAction`]: this describes a **malformed call**, not
/// a failed execution. The tool is never invoked, so `ToolErrorRecovery` is
/// not involved and no side effect happens.
#[derive(Debug, Clone)]
pub struct InvalidArgsCtx<'a> {
    /// Name of the tool the model tried to call.
    pub tool_name: &'a str,
    /// Required fields absent from the arguments object.
    pub missing_fields: Vec<String>,
    /// The raw arguments JSON produced by the model.
    pub raw_args: &'a str,
    /// The tool's input schema as declared to the model.
    pub schema: Option<&'a serde_json::Value>,
}

/// What the runtime should do after it rejects an argument-invalid call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidArgsAction {
    /// Feed the message back as the tool result and keep the run going so the
    /// model can re-issue the call with complete arguments.
    Reissue {
        /// Guidance shown to the model in place of the tool result.
        message: String,
    },
    /// Feed the message back as the tool result and end the run.
    ///
    /// For applications that prefer invalid calls to fail loudly instead of
    /// silently retrying.
    Stop {
        /// Guidance shown to the model in place of the tool result.
        message: String,
    },
}

/// Strategy for calls that omit schema-required arguments.
///
/// Called before execution, once per rejected call, with the structured
/// details. Implementations may perform side effects (load docs, trigger
/// workflows) before returning an action.
///
/// The hook is `async` so applications can do I/O; the kernel only decides
/// *when* to ask, not *how* to respond.
#[async_trait::async_trait]
pub trait InvalidArgsHandler: Send + Sync {
    async fn on_invalid_args(&self, ctx: InvalidArgsCtx<'_>) -> AgentResult<InvalidArgsAction>;
}

/// Default handler: tell the model which required fields are missing and let it
/// re-issue. Never surfaced to the user.
pub struct ReissueMissingFields;

#[async_trait::async_trait]
impl InvalidArgsHandler for ReissueMissingFields {
    async fn on_invalid_args(&self, ctx: InvalidArgsCtx<'_>) -> AgentResult<InvalidArgsAction> {
        let missing = ctx.missing_fields.join("`, `");
        Ok(InvalidArgsAction::Reissue {
            message: format!(
                "Tool call was not executed: it omitted required field(s) `{missing}`. \
                 Re-issue the call with every required field filled in. If you need several \
                 calls, emit each one with complete arguments."
            ),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_on_error_always_stops() {
        let recovery = StopOnError;
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");
        assert_eq!(
            recovery.on_error(&session_id, &names, &error).unwrap(),
            ToolErrorAction::Stop
        );
    }

    #[test]
    fn retry_on_error_always_retries() {
        let recovery = RetryOnError;
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");
        assert_eq!(
            recovery.on_error(&session_id, &names, &error).unwrap(),
            ToolErrorAction::Retry
        );
    }

    // ── ConsecutiveFailureRecovery: basic flow ───────────────────────

    #[test]
    fn consecutive_failure_retries_then_retry_with_history() {
        let recovery = ConsecutiveFailureRecovery::new(3);
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");

        // First two failures: retry
        assert_eq!(
            recovery.on_error(&session_id, &names, &error).unwrap(),
            ToolErrorAction::Retry
        );
        assert_eq!(
            recovery.on_error(&session_id, &names, &error).unwrap(),
            ToolErrorAction::Retry
        );

        // Third failure: RetryWithHistory (grace round)
        let action = recovery.on_error(&session_id, &names, &error).unwrap();
        assert!(matches!(action, ToolErrorAction::RetryWithHistory { .. }));
    }

    #[test]
    fn consecutive_failure_stop_after_grace() {
        let recovery = ConsecutiveFailureRecovery::new(3);
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");

        // 3 failures → RetryWithHistory (grace)
        for _ in 0..2 {
            recovery.on_error(&session_id, &names, &error).unwrap();
        }
        let action = recovery.on_error(&session_id, &names, &error).unwrap();
        assert!(matches!(action, ToolErrorAction::RetryWithHistory { .. }));

        // Counts NOT cleared — tool_a is still at count=3, grace_used=true.
        // Next failure (count=4) hits threshold → grace already used → Stop.
        let action = recovery.on_error(&session_id, &names, &error).unwrap();
        assert!(
            matches!(action, ToolErrorAction::Stop),
            "should Stop after grace exhausted, got {:?}",
            action
        );
    }

    #[test]
    fn consecutive_failure_retry_with_history_collects_errors() {
        let recovery = ConsecutiveFailureRecovery::new(2);
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];

        // Two failures with different error messages
        let error1 = AgentError::internal("first error");
        let error2 = AgentError::internal("second error");

        recovery.on_error(&session_id, &names, &error1).unwrap();
        let action = recovery.on_error(&session_id, &names, &error2).unwrap();

        match action {
            ToolErrorAction::RetryWithHistory { errors } => {
                assert_eq!(errors.len(), 2);
                assert!(errors[0].contains("first error"));
                assert!(errors[1].contains("second error"));
            }
            _ => panic!("expected RetryWithHistory, got {:?}", action),
        }
    }

    // ── ConsecutiveFailureRecovery: reset behavior ───────────────────

    #[test]
    fn consecutive_failure_resets_on_success() {
        let recovery = ConsecutiveFailureRecovery::new(3);
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");

        // Two failures
        recovery.on_error(&session_id, &names, &error).unwrap();
        recovery.on_error(&session_id, &names, &error).unwrap();

        // Reset on success
        recovery.reset_failures(&session_id, "tool_a");

        // Should retry again (counter reset)
        assert_eq!(
            recovery.on_error(&session_id, &names, &error).unwrap(),
            ToolErrorAction::Retry
        );
    }

    #[test]
    fn consecutive_failure_on_success_resets_via_trait() {
        let recovery = ConsecutiveFailureRecovery::new(2);
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");

        // One failure (below the threshold of 2)
        assert_eq!(
            recovery.on_error(&session_id, &names, &error).unwrap(),
            ToolErrorAction::Retry
        );

        // A successful execution breaks the streak (this is what the react loop calls).
        recovery.on_success(&session_id, "tool_a");

        // The next failure starts a fresh streak: retry again, not stop.
        assert_eq!(
            recovery.on_error(&session_id, &names, &error).unwrap(),
            ToolErrorAction::Retry
        );
    }

    #[test]
    fn consecutive_failure_on_success_resets_grace() {
        let recovery = ConsecutiveFailureRecovery::new(2);
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");

        // 2 failures → RetryWithHistory
        recovery.on_error(&session_id, &names, &error).unwrap();
        let action = recovery.on_error(&session_id, &names, &error).unwrap();
        assert!(matches!(action, ToolErrorAction::RetryWithHistory { .. }));

        // Tool succeeds → reset
        recovery.on_success(&session_id, "tool_a");

        // Next 2 failures → should get RetryWithHistory again (grace reset)
        recovery.on_error(&session_id, &names, &error).unwrap();
        let action = recovery.on_error(&session_id, &names, &error).unwrap();
        assert!(
            matches!(action, ToolErrorAction::RetryWithHistory { .. }),
            "grace should be reusable after success, got {:?}",
            action
        );
    }

    #[test]
    fn consecutive_failure_resets_clears_error_messages() {
        let recovery = ConsecutiveFailureRecovery::new(3);
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");

        // Two failures → error messages accumulated
        recovery.on_error(&session_id, &names, &error).unwrap();
        recovery.on_error(&session_id, &names, &error).unwrap();

        // Reset
        recovery.reset_failures(&session_id, "tool_a");

        // Next failure → if we reach threshold, error messages should only
        // contain the new ones (not the old ones before reset)
        let error_new = AgentError::internal("new error");
        recovery.on_error(&session_id, &names, &error_new).unwrap();
        recovery.on_error(&session_id, &names, &error_new).unwrap();
        let action = recovery.on_error(&session_id, &names, &error_new).unwrap();
        match action {
            ToolErrorAction::RetryWithHistory { errors } => {
                assert_eq!(errors.len(), 3);
                for e in &errors {
                    assert!(e.contains("new error"), "old errors should be cleared");
                }
            }
            _ => panic!("expected RetryWithHistory, got {:?}", action),
        }
    }

    #[test]
    fn consecutive_failure_resets_session_clears_everything() {
        let recovery = ConsecutiveFailureRecovery::new(2);
        let session_id = SessionId::new(1);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");

        // 2 failures → RetryWithHistory
        recovery.on_error(&session_id, &names, &error).unwrap();
        let action = recovery.on_error(&session_id, &names, &error).unwrap();
        assert!(matches!(action, ToolErrorAction::RetryWithHistory { .. }));

        // Reset session
        recovery.reset_session(&session_id);

        // Should retry again (fresh session)
        assert_eq!(
            recovery.on_error(&session_id, &names, &error).unwrap(),
            ToolErrorAction::Retry
        );
    }

    // ── ConsecutiveFailureRecovery: session isolation ─────────────────

    #[test]
    fn consecutive_failure_per_session_isolation() {
        let recovery = ConsecutiveFailureRecovery::new(2);
        let session1 = SessionId::new(1);
        let session2 = SessionId::new(2);
        let names = vec!["tool_a".to_string()];
        let error = AgentError::internal("test error");

        // Session 1: two failures → RetryWithHistory
        recovery.on_error(&session1, &names, &error).unwrap();
        let action = recovery.on_error(&session1, &names, &error).unwrap();
        assert!(matches!(action, ToolErrorAction::RetryWithHistory { .. }));

        // Session 1: one more failure → Stop (grace already used)
        let action = recovery.on_error(&session1, &names, &error).unwrap();
        assert!(
            matches!(action, ToolErrorAction::Stop),
            "session 1 should Stop after grace exhausted, got {:?}",
            action
        );

        // Session 2: should still retry (isolated, no grace used)
        assert_eq!(
            recovery.on_error(&session2, &names, &error).unwrap(),
            ToolErrorAction::Retry
        );
    }

    // ── ConsecutiveFailureRecovery: different tools independent ───────

    #[test]
    fn consecutive_failure_different_tools_independent() {
        let recovery = ConsecutiveFailureRecovery::new(2);
        let session_id = SessionId::new(1);
        let error = AgentError::internal("test error");

        // tool_a: 1 failure
        recovery
            .on_error(&session_id, &["tool_a".to_string()], &error)
            .unwrap();

        // tool_b: 1 failure (independent counter)
        assert_eq!(
            recovery
                .on_error(&session_id, &["tool_b".to_string()], &error,)
                .unwrap(),
            ToolErrorAction::Retry
        );

        // tool_a: 2nd failure → RetryWithHistory (not affected by tool_b)
        let action = recovery
            .on_error(&session_id, &["tool_a".to_string()], &error)
            .unwrap();
        assert!(
            matches!(action, ToolErrorAction::RetryWithHistory { .. }),
            "tool_a should trigger RetryWithHistory independently of tool_b"
        );
    }

    // ── ConsecutiveFailureRecovery: ToolArgsInvalid error messages ────

    #[test]
    fn consecutive_failure_records_serde_error_details() {
        let recovery = ConsecutiveFailureRecovery::new(2);
        let session_id = SessionId::new(1);
        let names = vec!["write_file".to_string()];

        // Simulate the improved serde error message
        let error = AgentError::ToolArgsInvalid {
            name: "write_file".to_string(),
            raw: "missing field `path` at line 1 column 2 (args: {})".to_string(),
        };

        recovery.on_error(&session_id, &names, &error).unwrap();
        let action = recovery.on_error(&session_id, &names, &error).unwrap();

        match action {
            ToolErrorAction::RetryWithHistory { errors } => {
                assert_eq!(errors.len(), 2);
                assert!(
                    errors[0].contains("missing field"),
                    "should contain serde error details"
                );
            }
            _ => panic!("expected RetryWithHistory, got {:?}", action),
        }
    }
}
