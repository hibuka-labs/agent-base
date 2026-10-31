use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("LLM call failed: {0}")]
    Llm(String),

    #[error("LLM API error: {message}")]
    LlmApi { message: String },

    #[error("LLM rate limit exceeded")]
    RateLimitExceeded,

    #[error("LLM service unavailable: {0}")]
    ServiceUnavailable(String),

    #[error("SSE stream error: {0}")]
    LlmStream(String),

    #[error("JSON parse error: {0}")]
    Json(String),

    #[error("Tool '{name}' not registered")]
    ToolNotFound { name: String },

    #[error("Tool '{name}' argument parsing failed: {raw}")]
    ToolArgsInvalid { name: String, raw: String },

    #[error("Tool '{name}' execution failed: {source}")]
    ToolExecution {
        name: String,
        #[source]
        source: Box<AgentError>,
    },

    #[error("Tool timeout exceeded")]
    ToolTimeout,

    #[error("Tool '{name}' output exceeds the {max_chars}-char limit")]
    ToolOutputTooLarge { name: String, max_chars: usize },

    #[error("Tool call rejected by approval: {tool_name}")]
    ApprovalDenied { tool_name: String },

    #[error("Tool call blocked: {0}")]
    ToolBlocked(String),

    #[error("Session {0} not found")]
    SessionNotFound(u64),

    #[error("Max turns ({limit}) reached, stopping forcibly")]
    MaxTurnsExceeded { limit: u32 },

    #[error("Operation cancelled")]
    Cancelled,

    #[error("Resource unavailable: {0}")]
    ResourceUnavailable(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),

    /// Checkpoint persistence contract refused (T5): a payload this build
    /// will not read, or a cross-backend resume. The message spells out why.
    #[error("{0}")]
    Checkpoint(String),

    #[error("Internal error: {0}")]
    Internal(String),
}

impl AgentError {
    pub fn llm(message: impl Into<String>) -> Self {
        Self::Llm(message.into())
    }

    pub fn json(message: impl Into<String>) -> Self {
        Self::Json(message.into())
    }

    pub fn internal(message: impl Into<String>) -> Self {
        Self::Internal(message.into())
    }

    pub fn tool_not_found(name: impl Into<String>) -> Self {
        Self::ToolNotFound { name: name.into() }
    }

    pub fn session_not_found(id: u64) -> Self {
        Self::SessionNotFound(id)
    }

    pub fn tool_timeout() -> Self {
        Self::ToolTimeout
    }

    pub fn rate_limit_exceeded() -> Self {
        Self::RateLimitExceeded
    }

    pub fn service_unavailable(message: impl Into<String>) -> Self {
        Self::ServiceUnavailable(message.into())
    }

    pub fn resource_unavailable(message: impl Into<String>) -> Self {
        Self::ResourceUnavailable(message.into())
    }

    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigError(message.into())
    }

    pub fn is_cancelled(&self) -> bool {
        matches!(self, Self::Cancelled)
    }

    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::Llm(_)
                | Self::LlmApi { .. }
                | Self::LlmStream(_)
                | Self::ServiceUnavailable(_)
                | Self::RateLimitExceeded
        )
    }

    pub fn is_rate_limited(&self) -> bool {
        matches!(self, Self::RateLimitExceeded)
    }

    pub fn is_resource_unavailable(&self) -> bool {
        matches!(self, Self::ResourceUnavailable(_))
    }

    /// Classify this error into an `ErrorKind` for recovery decisions.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Self::ToolExecution { name, .. } => ErrorKind::ToolCallFailed {
                tool_name: name.clone(),
            },
            Self::ToolNotFound { .. } => ErrorKind::ToolNotFound,
            Self::ToolArgsInvalid { .. } => ErrorKind::ToolArgsInvalid,
            Self::ToolTimeout => ErrorKind::ToolTimeout,
            Self::ToolOutputTooLarge { name, .. } => ErrorKind::ToolCallFailed {
                tool_name: name.clone(),
            },
            Self::ServiceUnavailable(_) => ErrorKind::ModelOverloaded,
            Self::RateLimitExceeded => ErrorKind::RateLimited,
            Self::Cancelled => ErrorKind::Cancelled,
            // Llm, LlmApi, LlmStream → overloaded (transient LLM failures)
            Self::Llm(_) | Self::LlmApi { .. } | Self::LlmStream(_) => ErrorKind::ModelOverloaded,
            // Everything else → internal
            _ => ErrorKind::Internal,
        }
    }
}

// ── Convenience From impls for common error types ──

impl From<std::io::Error> for AgentError {
    fn from(e: std::io::Error) -> Self {
        AgentError::internal(e.to_string())
    }
}

impl From<serde_json::Error> for AgentError {
    fn from(e: serde_json::Error) -> Self {
        AgentError::json(e.to_string())
    }
}

impl From<llm_trait::LlmError> for AgentError {
    fn from(e: llm_trait::LlmError) -> Self {
        match e {
            llm_trait::LlmError::Config(msg) => AgentError::ConfigError(msg),
            llm_trait::LlmError::LlmApi { status, message } => {
                if status == 429 {
                    AgentError::RateLimitExceeded
                } else if status >= 500 {
                    AgentError::ServiceUnavailable(format!("HTTP {status}: {message}"))
                } else {
                    AgentError::LlmApi {
                        message: format!("HTTP {status}: {message}"),
                    }
                }
            }
            llm_trait::LlmError::Llm(msg) => AgentError::Llm(msg),
            llm_trait::LlmError::Stream(msg) => AgentError::LlmStream(msg),
        }
    }
}

/// Classifies an `AgentError` into a broad category for recovery decisions.
///
/// Unlike `AgentError` which carries full context (messages, nested errors, etc.),
/// `ErrorKind` is a lightweight discriminant that lets `RecoveryPolicy` and other
/// decision-makers branch without pattern-matching on every `AgentError` variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// A tool call failed during execution.
    ToolCallFailed { tool_name: String },
    /// The requested tool was not found in the registry.
    ToolNotFound,
    /// Tool arguments were invalid.
    ToolArgsInvalid,
    /// Tool execution timed out.
    ToolTimeout,
    /// The model/LLM service is overloaded (e.g. 529, 503).
    ModelOverloaded,
    /// Rate limit was exceeded.
    RateLimited,
    /// The operation was cancelled by the user or a cancellation signal.
    Cancelled,
    /// Catch-all for errors that don't fit a specific category.
    Internal,
}

impl std::fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ToolCallFailed { tool_name } => write!(f, "tool call failed: {tool_name}"),
            Self::ToolNotFound => write!(f, "tool not found"),
            Self::ToolArgsInvalid => write!(f, "tool args invalid"),
            Self::ToolTimeout => write!(f, "tool timeout"),
            Self::ModelOverloaded => write!(f, "model overloaded"),
            Self::RateLimited => write!(f, "rate limited"),
            Self::Cancelled => write!(f, "cancelled"),
            Self::Internal => write!(f, "internal error"),
        }
    }
}

pub type AgentResult<T> = Result<T, AgentError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_tool_execution() {
        let err = AgentError::ToolExecution {
            name: "my_tool".to_string(),
            source: Box::new(AgentError::internal("boom")),
        };
        assert_eq!(
            err.kind(),
            ErrorKind::ToolCallFailed {
                tool_name: "my_tool".to_string()
            }
        );
    }

    #[test]
    fn kind_tool_not_found() {
        let err = AgentError::tool_not_found("missing");
        assert_eq!(err.kind(), ErrorKind::ToolNotFound);
    }

    #[test]
    fn kind_tool_args_invalid() {
        let err = AgentError::ToolArgsInvalid {
            name: "t".to_string(),
            raw: "bad".to_string(),
        };
        assert_eq!(err.kind(), ErrorKind::ToolArgsInvalid);
    }

    #[test]
    fn kind_tool_timeout() {
        let err = AgentError::tool_timeout();
        assert_eq!(err.kind(), ErrorKind::ToolTimeout);
    }

    #[test]
    fn kind_service_unavailable() {
        let err = AgentError::service_unavailable("overloaded");
        assert_eq!(err.kind(), ErrorKind::ModelOverloaded);
    }

    #[test]
    fn kind_rate_limit() {
        let err = AgentError::rate_limit_exceeded();
        assert_eq!(err.kind(), ErrorKind::RateLimited);
    }

    #[test]
    fn kind_llm_maps_to_overloaded() {
        let err = AgentError::llm("connection refused");
        assert_eq!(err.kind(), ErrorKind::ModelOverloaded);
    }

    #[test]
    fn kind_llm_api_maps_to_overloaded() {
        let err = AgentError::LlmApi {
            message: "529".to_string(),
        };
        assert_eq!(err.kind(), ErrorKind::ModelOverloaded);
    }

    #[test]
    fn kind_llm_stream_maps_to_overloaded() {
        let err = AgentError::LlmStream("stream broken".to_string());
        assert_eq!(err.kind(), ErrorKind::ModelOverloaded);
    }

    #[test]
    fn kind_cancelled() {
        assert_eq!(AgentError::Cancelled.kind(), ErrorKind::Cancelled);
    }

    #[test]
    fn kind_internal_fallback() {
        let err = AgentError::internal("something");
        assert_eq!(err.kind(), ErrorKind::Internal);

        let err = AgentError::config_error("bad config");
        assert_eq!(err.kind(), ErrorKind::Internal);
    }

    #[test]
    fn error_kind_display() {
        assert_eq!(
            ErrorKind::ToolCallFailed {
                tool_name: "t".to_string()
            }
            .to_string(),
            "tool call failed: t"
        );
        assert_eq!(ErrorKind::ToolNotFound.to_string(), "tool not found");
        assert_eq!(ErrorKind::ModelOverloaded.to_string(), "model overloaded");
        assert_eq!(ErrorKind::RateLimited.to_string(), "rate limited");
    }

    // ── From impls ──

    #[test]
    fn from_io_error_maps_to_internal() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file missing");
        let agent_err: AgentError = io_err.into();
        assert!(matches!(agent_err, AgentError::Internal(_)));
        assert!(agent_err.to_string().contains("file missing"));
    }

    #[test]
    fn from_serde_json_error_maps_to_json() {
        let json_err = serde_json::from_str::<serde_json::Value>("not json").unwrap_err();
        let agent_err: AgentError = json_err.into();
        assert!(matches!(agent_err, AgentError::Json(_)));
    }

    /// Verify `?` works: a function returning AgentResult<T> can use `?`
    /// on io::Error and serde_json::Error through the From impls.
    #[test]
    fn from_impls_work_with_try_operator() -> AgentResult<()> {
        // io::Error via ?
        fn read_file() -> AgentResult<String> {
            let _ = std::fs::read_to_string("/nonexistent/path")?;
            unreachable!()
        }
        assert!(read_file().is_err());

        // serde_json::Error via ?
        fn parse_json() -> AgentResult<serde_json::Value> {
            let v: serde_json::Value = serde_json::from_str("bad json")?;
            Ok(v)
        }
        assert!(parse_json().is_err());

        Ok(())
    }
}
