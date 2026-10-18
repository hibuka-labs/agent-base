use serde::{Deserialize, Serialize};

// Re-export ResponseFormat from llm-trait.
pub use llm_trait::request::ResponseFormat;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Language {
    #[default]
    En,
    Zh,
}

impl std::fmt::Display for Language {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Language::En => write!(f, "en"),
            Language::Zh => write!(f, "zh"),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
    pub backoff_multiplier: f64,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 500,
            max_backoff_ms: 10_000,
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    pub fn initial_backoff_ms(mut self, ms: u64) -> Self {
        self.initial_backoff_ms = ms;
        self
    }

    pub fn max_backoff_ms(mut self, ms: u64) -> Self {
        self.max_backoff_ms = ms;
        self
    }

    pub fn no_jitter(mut self) -> Self {
        self.jitter = false;
        self
    }
}

use crate::llm::ReasoningConfig;

/// Safety configuration for agent runtime guardrails.
///
/// These limits are hard constraints enforced by code, not prompt —
/// the model cannot bypass them regardless of its capability.
#[derive(Clone, Debug)]
pub struct SafetyConfig {
    /// Maximum number of tool calls allowed per turn.
    /// When exceeded, tool calls are discarded and the LLM is forced to summarize.
    /// Default: 128.
    pub max_tool_calls_per_turn: usize,

    /// Maximum consecutive failures for the same tool before stopping retries.
    /// Default: 3.
    pub max_consecutive_failures: usize,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            max_tool_calls_per_turn: 128,
            max_consecutive_failures: 3,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct AgentConfig {
    pub system_prompt: Option<String>,
    /// Controls whether to include the reasoning content in LLM responses.
    ///
    /// Distinction from `reasoning.enabled`:
    /// - `enable_thought`: controls whether the `reasoning_content` field is forwarded
    ///   to consumers (i.e., "show the thinking process")
    /// - `reasoning.enabled`: controls whether the model's extended thinking / reasoning
    ///   mode is enabled (i.e., "let the model think deeply")
    ///
    /// Both are usually kept in sync, but can be controlled independently. For example,
    /// to enable deep thinking without showing the process, set `enable_thought = false`
    /// and `reasoning.enabled = true`.
    pub enable_thought: bool,
    /// Reasoning/thinking configuration that controls LLM reasoning behavior.
    ///
    /// - `enabled`: whether to enable extended thinking mode (equivalent to builder's `enable_thinking`)
    /// - `budget_tokens`: thinking token budget cap
    /// - `effort`: reasoning intensity/depth (semantics vary by provider)
    pub reasoning: Option<ReasoningConfig>,
    pub language: Language,
    pub execution: ExecutionConfig,
    pub llm: LlmConfig,
    pub tool: ToolConfig,
    pub session: SessionConfig,
    pub safety: SafetyConfig,
}

impl AgentConfig {
    /// Validate the configuration, returning an error for invalid values.
    pub fn validate(&self) -> crate::types::AgentResult<()> {
        use crate::types::AgentError;

        if let Some(max_turns) = self.execution.max_turns
            && max_turns == 0
        {
            return Err(AgentError::config_error(
                "execution.max_turns must be > 0".to_string(),
            ));
        }

        if let Some(max_sessions) = self.session.max_sessions
            && max_sessions == 0
        {
            return Err(AgentError::config_error(
                "session.max_sessions must be > 0".to_string(),
            ));
        }

        if let Some(tool_timeout_ms) = self.tool.tool_timeout_ms
            && tool_timeout_ms == 0
        {
            return Err(AgentError::config_error(
                "tool.tool_timeout_ms must be > 0".to_string(),
            ));
        }

        if self.safety.max_tool_calls_per_turn == 0 {
            return Err(AgentError::config_error(
                "safety.max_tool_calls_per_turn must be > 0".to_string(),
            ));
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct ExecutionConfig {
    pub max_turns: Option<u32>,
    pub approval_timeout_ms: Option<u64>,
    pub fail_on_persist_error: bool,
}

#[derive(Clone, Debug, Default)]
pub struct LlmConfig {
    pub response_format: Option<ResponseFormat>,
    pub llm_retry: Option<RetryConfig>,
}

/// Default tool timeout: 10 minutes.
/// This is a safety net against hung tools; tools should override
/// `timeout_ms()` with a value appropriate for their workload.
pub const DEFAULT_TOOL_TIMEOUT_MS: u64 = 600_000;

#[derive(Clone, Debug)]
pub struct ToolConfig {
    /// Default timeout for tools that don't override `timeout_ms()`.
    /// Defaults to `DEFAULT_TOOL_TIMEOUT_MS` (10 minutes).
    pub default_tool_timeout_ms: u64,
    pub tool_timeout_ms: Option<u64>,
    pub max_tool_output_chars: Option<usize>,
    pub tool_error_retry_prompt: Option<String>,
}

impl Default for ToolConfig {
    fn default() -> Self {
        Self {
            default_tool_timeout_ms: DEFAULT_TOOL_TIMEOUT_MS,
            tool_timeout_ms: None,
            max_tool_output_chars: None,
            tool_error_retry_prompt: None,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct SessionConfig {
    /// Maximum number of sessions; when exceeded, LRU evicts whole sessions (unloaded from memory, data preserved)
    /// None = no limit
    pub max_sessions: Option<usize>,

    /// Maximum turns retained per session; when exceeded, the oldest turns are truncated from the front
    /// None = no limit
    pub max_turns_per_session: Option<usize>,

    /// Per-message token cap (safety valve); messages exceeding it are not stored in the session history
    /// The threshold should be set very high (e.g. 100k) to only catch abnormal cases
    /// None = no limit
    pub max_message_tokens: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::AgentError;

    #[test]
    fn language_display_and_default() {
        assert_eq!(Language::default(), Language::En);
        assert_eq!(Language::En.to_string(), "en");
        assert_eq!(Language::Zh.to_string(), "zh");
    }

    #[test]
    fn retry_config_defaults_and_builders() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.initial_backoff_ms, 500);
        assert_eq!(cfg.max_backoff_ms, 10_000);
        assert_eq!(cfg.backoff_multiplier, 2.0);
        assert!(cfg.jitter);

        let cfg = RetryConfig::new()
            .max_retries(7)
            .initial_backoff_ms(100)
            .max_backoff_ms(20_000)
            .no_jitter();
        assert_eq!(cfg.max_retries, 7);
        assert_eq!(cfg.initial_backoff_ms, 100);
        assert_eq!(cfg.max_backoff_ms, 20_000);
        assert!(!cfg.jitter);
    }

    #[test]
    fn response_format_to_api_value() {
        let v = ResponseFormat::JsonObject.to_api_value();
        assert_eq!(v["type"], "json_object");

        let v = ResponseFormat::JsonSchema {
            name: "event".to_string(),
            schema: serde_json::json!({"type": "object"}),
        }
        .to_api_value();
        assert_eq!(v["type"], "json_schema");
        assert_eq!(v["json_schema"]["name"], "event");
        assert_eq!(v["json_schema"]["schema"]["type"], "object");
    }

    #[test]
    fn safety_config_defaults() {
        let cfg = SafetyConfig::default();
        assert_eq!(cfg.max_tool_calls_per_turn, 128);
        assert_eq!(cfg.max_consecutive_failures, 3);
    }

    #[test]
    fn validate_accepts_default() {
        assert!(AgentConfig::default().validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_max_turns() {
        let mut cfg = AgentConfig::default();
        cfg.execution.max_turns = Some(0);
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, AgentError::ConfigError(_)));
        assert!(err.to_string().contains("max_turns"));
    }

    #[test]
    fn validate_rejects_zero_max_sessions() {
        let mut cfg = AgentConfig::default();
        cfg.session.max_sessions = Some(0);
        assert!(matches!(cfg.validate(), Err(AgentError::ConfigError(_))));
    }

    #[test]
    fn validate_rejects_zero_tool_timeout() {
        let mut cfg = AgentConfig::default();
        cfg.tool.tool_timeout_ms = Some(0);
        assert!(matches!(cfg.validate(), Err(AgentError::ConfigError(_))));
    }

    #[test]
    fn validate_rejects_zero_max_tool_calls() {
        let mut cfg = AgentConfig::default();
        cfg.safety.max_tool_calls_per_turn = 0;
        assert!(matches!(cfg.validate(), Err(AgentError::ConfigError(_))));
    }
}
