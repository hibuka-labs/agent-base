# agent-base

[![crates.io](https://img.shields.io/crates/v/agent-base.svg)](https://crates.io/crates/agent-base)
[![Documentation](https://docs.rs/agent-base/badge.svg)](https://docs.rs/agent-base)
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![codecov](https://codecov.io/gh/hibuka-labs/agent-base/branch/master/graph/badge.svg)](https://codecov.io/gh/hibuka-labs/agent-base)

[English](README.md) | [中文](README_CN.md)

A lightweight **Agent Runtime Kernel** for building AI agents in Rust.

`agent-base` provides the minimal orchestration layer needed to build custom AI agents — LLM integration, tool dispatch, multi-turn conversation, approval flows, event streaming, and error recovery — all with zero business assumptions.

## Installation

```toml
[dependencies]
agent-base = "0.2.1"
```

## Design Principles

- **Clear semantics** — `RunOutcome` explicitly distinguishes `Completed` from `Failed`; events capture the process, the return value captures the final result.
- **Simple state model** — Runtime memory is the source of truth for live sessions; `SessionStore` is an optional persistence adapter.
- **Conservative by default** — On tool failure, the runtime stops by default (`StopOnError`) rather than guessing how to recover.
- **Strategy injection** — All variable behaviors are injected via traits (`ToolErrorRecovery`, `ToolPolicy`, `ApprovalHandler`, `Middleware`), not hardcoded.

## Features

- **LLM Abstraction** — `LlmProvider` trait via `llm-trait`; `llm-unified` provides concrete implementations for OpenAI-compatible, Anthropic, and custom backends
- **LLM Retry** — Configurable retry with exponential backoff via `RetryConfig`
- **Tool System** — `Tool` trait + `ToolRegistry` for registration and dispatch; configurable `tool_timeout`
- **Approval Flow** — `ApprovalHandler` trait with `AllowOnce` / `AllowAlways` / `Deny` decisions + cancellation support
- **Error Recovery** — `ToolErrorRecovery` trait; defaults to `StopOnError`, opt-in `RetryOnError` + custom retry prompts
- **Event Streaming** — Structured `RuntimeEvent` stream with configurable `EventBus` capacity
- **Multi-turn Sessions** — `AgentSession` manages message history; `SessionStore` for optional persistence; `max_sessions` / `max_turns_per_session` limits
- **SQLite Session Store** — `SqliteSessionStore` behind `sqlite-session` feature flag for persistent session storage
- **Context Management** — Configurable `ContextWindowManager` for token budget control; `max_message_tokens` cap
- **Middleware** — Hooks at `on_user_message`, `on_pre_llm`, and `on_post_llm` for extensions
- **Ephemeral Messages** — Messages can be marked ephemeral; visible to LLM during the current turn, automatically cleaned from memory after turn ends, excluded from persistence
- **Custom Messages** — `ChatMessage::Custom` variant with `convert_to_llm` callback for domain-specific message types
- **Plan Checklist** — Built-in `UpdatePlanTool` for multi-step task tracking with `PlanItem` / `PlanStepStatus`
- **Checkpoints** — Structured `CheckpointData` / `CheckpointStep` events enable replay, debugging, and resume
- **Tool Enforcement** — `ToolEnforcementMiddleware` nudges the LLM to call tools instead of just describing actions
- **Turn Tool Limit** — `TurnToolLimitMiddleware` caps tool calls per turn
- **Circuit Breaker** — `ConsecutiveFailureRecovery` stops the run after N consecutive failures
- **Thinking / Reasoning** — Per-model thinking budget and effort level configuration
- **Response Format** — Structured output via `ResponseFormat` (JSON Schema / JSON Object)
- **Session ID Generator** — Pluggable `SessionIdGenerator` for custom ID strategies
- **Tool Output Truncation** — Configurable `max_tool_output_chars` with structured `TruncationInfo`
- **Tool Partial Results** — `ToolContext::emit_partial_result()` for streaming intermediate output during long-running tool execution
- **Truncation Guard** — Automatically detects truncated tool calls when LLM hits the token limit, forcing re-issue with complete arguments
- **Message Queue** — `MessageQueue` with steering/follow-up queues and configurable `QueueMode` for ordered or one-at-a-time draining

## Feature Flags

| Flag | Description | Default |
|------|-------------|---------|
| `sqlite-session` | Enable `SqliteSessionStore` for SQLite-backed session persistence | off |
| `telemetry` | Enable OpenTelemetry integration for distributed tracing | off |

```toml
[dependencies]
agent-base = { version = "0.8.0", features = ["sqlite-session"] }
```

## Quick Start

### 1. Define a Tool

Any capability you want your agent to have is expressed as a `Tool`:

```rust
use agent_base::{Tool, ToolContext, Content, AgentResult};
use async_trait::async_trait;
use serde_json::{json, Value};

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &'static str { "get_weather" }

    fn description(&self) -> &'static str {
        "Get current weather for a city"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "City name" }
            },
            "required": ["city"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let city = args["city"].as_str().unwrap_or("unknown");
        Ok(vec![Content::text(format!("Weather in {}: 22°C, sunny", city))])
    }
}
```

### 2. Build the Agent

```rust
use agent_base::{
    AgentBuilder, AgentResult, RuntimeEvent, RunOutcome,
};
use llm_trait::LlmConfig;
use llm_unified::create_provider;

#[tokio::main]
async fn main() -> AgentResult<()> {
    let llm = create_provider(&LlmConfig {
        backend: "custom".to_string(),
        protocol: Some("openai".to_string()),
        api_key: std::env::var("OPENAI_API_KEY").unwrap(),
        model: "gpt-4o".into(),
        base_url: None,
        options: Default::default(),
    }).unwrap();

    let runtime = AgentBuilder::new(llm)
        .system_prompt("You are a helpful weather assistant.")
        .register_tool(WeatherTool)
        .build()?;

    let session_id = runtime.create_session().await;

    runtime
        .run_turn(session_id, "What's the weather in Tokyo?", |event| {
            match event {
                RuntimeEvent::TextDelta { text, .. } => print!("{}", text),
                RuntimeEvent::ToolCallStarted { tool_name, .. } => {
                    println!("
[Calling tool: {}]", tool_name);
                }
                RuntimeEvent::ToolCallFinished { summary, .. } => {
                    println!("[Tool result: {}]", summary);
                }
                RuntimeEvent::RunFinished { .. } => println!("
[Done]"),
                _ => {}
            }
            Ok(())
        })
        .await?;

    Ok(())
}
```

The callback approach gives you full control over event handling. For simpler cases, `run_turn_collect` returns `(Vec<RuntimeEvent>, RunOutcome)` directly.

### 3. Handle Tool Errors

By default, tool failures stop the run. For self-healing agents (e.g. code agents that retry compilation), inject `RetryOnError`:

```rust
use agent_base::RetryOnError;

let runtime = AgentBuilder::new(llm)
    .register_tool(MyTool)
    .error_recovery(Arc::new(RetryOnError))  // ← retry on failure
    .build()?;
```

### 4. Add Approval for Sensitive Tools

```rust
use agent_base::{
    ApprovalHandler, ApprovalRequest, ApprovalDecision,
    ToolPolicy, RiskLevel,
};
use tokio_util::sync::CancellationToken;

struct MyApprovalHandler;
#[async_trait::async_trait]
impl ApprovalHandler for MyApprovalHandler {
    async fn approve(
        &self,
        _req: ApprovalRequest,
        _cancel_token: CancellationToken,
    ) -> AgentResult<ApprovalDecision> {
        // Ask user via UI, CLI, etc.
        Ok(ApprovalDecision::AllowOnce)
    }
}

struct MyToolPolicy;
#[async_trait::async_trait]
impl ToolPolicy for MyToolPolicy {
    async fn evaluate_approval(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Option<ApprovalRequest> {
        if tool_name == "dangerous_tool" {
            Some(ApprovalRequest {
                title: "Confirm action".into(),
                message: format!("Execute `{}`?", tool_name),
                risk_level: RiskLevel::Sensitive,
                ..Default::default()
            })
        } else {
            None  // auto-allow
        }
    }

    // before_call / after_call hooks available for logging, auditing, etc.
}

let runtime = AgentBuilder::new(llm)
    .register_tool(DangerousTool)
    .tool_policy(Arc::new(MyToolPolicy))
    .approval_handler(Arc::new(MyApprovalHandler))
    .build()?;
```

## Examples

```bash
# Configure API key
cp .env.example .env
# Edit .env with your OPENAI_API_KEY or ANTHROPIC_API_KEY

# Interactive REPL
cargo run --example repl

# Full quickstart demo (tools + approval + middleware)
cargo run --example quickstart_demo

# SubAgent demo
cargo run --example subagent_demo

# Middleware demo
cargo run --example middleware_demo

# Approval & policy demo
cargo run --example approval_policy_demo

# Tool context demo
cargo run --example tool_context_demo

# Thinking / reasoning test
cargo run --example thinking_test
```

## What agent-base Does NOT Do

- Built-in SSH, filesystem, or database tools
- Workflow DAG or multi-agent orchestration engine
- Memory or RAG (Retrieval-Augmented Generation) framework
- Terminal UI or built-in approval dialog
- Production-grade persistence or transaction system

Business-specific tools and strategies belong in **upper layers** (e.g. `phi-agent`, `agent-works`, `phi-tools`).

## Typical Layering

```
phi-agent / agent-works / ...              ← Framework / Enhanced toolkits
    └── agent-base                          ← Lightweight Runtime Kernel
```

## v1 Semantics

| Convention | Meaning |
|---|---|
| `run_turn` → callback `FnMut(RuntimeEvent)` | Process events as they arrive; `run_turn_collect` batches them |
| `RunOutcome` | `Completed` / `Failed` / `MaxTurnsExceeded` / `Cancelled` |
| `RuntimeEvent::RunFinished` | Process ended — final status is in the `run_turn` return value |
| Tool failure → defaults to `StopOnError` | Inject `RetryOnError` for self-healing agents |
| SubAgent → defaults to `Ephemeral` | Use `with_persistent()` for shared context |
| Session → memory is source of truth | `SessionStore` is an optional persistence adapter |


## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/aryansk"><img src="https://avatars.githubusercontent.com/u/70511529?v=4?s=100" width="100px;" alt="Aryan Singh K."/><br /><sub><b>Aryan Singh K.</b></sub></a><br /><a href="https://github.com/hibuka-labs/agent-base/commits?author=aryansk" title="Tests">✅</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Acknowledgments

This project draws inspiration from the [OpenAI Codex CLI](https://github.com/openai/codex) project — particularly its approach to tool orchestration and task planning.

## Stability

This project is in early development (v0.2.1). The core abstractions are settling but not yet frozen. Expect minor API changes as the ecosystem evolves.

## License

MIT
