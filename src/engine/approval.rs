use async_trait::async_trait;

use crate::types::{AgentResult, ApprovalDecision, ApprovalRequest};

/// Trait for handling tool approval requests.
///
/// # Cancellation Contract
///
/// Implementors MUST handle cancellation via the provided `cancel_token`.
/// The token is cancelled when the user interrupts the agent (e.g., Ctrl+C).
///
/// - **Async handlers** (channels, timers): race against `cancel_token.cancelled()` using `tokio::select!`
/// - **Blocking handlers** (stdin, file IO): use `tokio::task::spawn_blocking` + `tokio::select!` to avoid blocking the tokio runtime
///
/// # Timeout
///
/// The caller (`process_approval`) wraps this method with a 300-second timeout.
/// If the handler does not return within 300s, the approval is denied automatically.
///
/// # Example (blocking stdin)
///
/// ```ignore
/// async fn approve(&self, request: ApprovalRequest, cancel_token: CancellationToken) -> AgentResult<ApprovalDecision> {
///     tokio::select! {
///         _ = cancel_token.cancelled() => Err(AgentError::Cancelled),
///         result = tokio::task::spawn_blocking(|| read_stdin()) => result?,
///     }
/// }
/// ```
#[async_trait]
pub trait ApprovalHandler: Send + Sync {
    async fn approve(
        &self,
        request: ApprovalRequest,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> AgentResult<ApprovalDecision>;
}

#[derive(Clone, Debug, Default)]
pub struct DenyAllApprovalHandler;

#[async_trait]
impl ApprovalHandler for DenyAllApprovalHandler {
    async fn approve(
        &self,
        _request: ApprovalRequest,
        _cancel_token: tokio_util::sync::CancellationToken,
    ) -> AgentResult<ApprovalDecision> {
        Ok(ApprovalDecision::Deny)
    }
}

#[derive(Clone, Debug, Default)]
pub struct AllowAllApprovalHandler;

#[async_trait]
impl ApprovalHandler for AllowAllApprovalHandler {
    async fn approve(
        &self,
        _request: ApprovalRequest,
        _cancel_token: tokio_util::sync::CancellationToken,
    ) -> AgentResult<ApprovalDecision> {
        Ok(ApprovalDecision::AllowAlways)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::RiskLevel;
    use tokio_util::sync::CancellationToken;

    fn request() -> ApprovalRequest {
        ApprovalRequest {
            title: "Delete file".to_string(),
            message: "Really delete?".to_string(),
            action_key: Some("delete".to_string()),
            risk_level: RiskLevel::Destructive,
            raw: None,
            source: None,
        }
    }

    #[tokio::test]
    async fn deny_all_returns_deny() {
        let decision = DenyAllApprovalHandler
            .approve(request(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(decision, ApprovalDecision::Deny);
    }

    #[tokio::test]
    async fn allow_all_returns_allow_always() {
        let decision = AllowAllApprovalHandler
            .approve(request(), CancellationToken::new())
            .await
            .unwrap();
        assert_eq!(decision, ApprovalDecision::AllowAlways);
    }
}
