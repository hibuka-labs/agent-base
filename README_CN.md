# agent-base

[![crates.io](https://img.shields.io/crates/v/agent-base.svg)](https://crates.io/crates/agent-base)
[![Documentation](https://docs.rs/agent-base/badge.svg)](https://docs.rs/agent-base)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/License-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE-MIT)

[English](README.md) | [中文](README_CN.md)

用 Rust 构建 AI Agent 的轻量级 **Agent 运行时内核**。

`agent-base` 提供了构建自定义 AI Agent 所需的最小编排层——LLM 集成、工具调度、多轮对话、审批流程、事件流和错误恢复——零业务假设。

## 安装

```toml
[dependencies]
agent-base = "0.2.1"
```

## 设计原则

- **清晰的语义** — `RunOutcome` 明确区分 `Completed`（成功）和 `Failed`（失败）；事件捕获过程，返回值捕获最终结果。
- **简洁的状态模型** — 运行时内存是活跃会话的唯一数据源；`SessionStore` 是可选的持久化适配器。
- **默认保守** — 工具失败时，运行时默认停止（`StopOnError`），而不是猜测如何恢复。
- **策略注入** — 所有可变行为通过 trait 注入（`ToolErrorRecovery`、`ToolPolicy`、`ApprovalHandler`、`Middleware`），而非硬编码。

## 特性

- **LLM 抽象** — 通过 `llm-trait` 定义 `LlmProvider` trait；`llm-unified` 提供 OpenAI 兼容、Anthropic 和自定义后端的具体实现
- **LLM 重试** — 可配置的指数退避重试策略 `RetryConfig`
- **工具系统** — `Tool` trait + `ToolRegistry` 注册和调度；可配置 `tool_timeout`
- **审批流程** — `ApprovalHandler` trait，支持 `AllowOnce` / `AllowAlways` / `Deny` 决策，内置取消支持
- **错误恢复** — `ToolErrorRecovery` trait；默认 `StopOnError`，可选 `RetryOnError` + 自定义重试提示
- **事件流** — 结构化 `RuntimeEvent` 流，可配置 `EventBus` 容量
- **多轮会话** — `AgentSession` 管理消息历史；`SessionStore` 可选持久化；支持 `max_sessions` / `max_turns_per_session` 限制
- **SQLite 会话存储** — `SqliteSessionStore`，通过 `sqlite-session` feature flag 开启持久化会话存储
- **上下文管理** — 可配置的 `ContextWindowManager` 控制 token 预算；`max_message_tokens` 上限
- **中间件** — `on_user_message`、`on_pre_llm`、`on_post_llm` 三个钩子用于扩展
- **临时消息（Ephemeral Messages）** — 消息可标记为临时，LLM 本轮可见，turn 结束后自动从内存清理且不持久化
- **自定义消息** — `ChatMessage::Custom` 变体，配合 `convert_to_llm` 回调支持领域特定的消息类型
- **Plan 检查清单** — `UpdatePlanTool` 内置工具，支持 `PlanItem` / `PlanStepStatus` 多步骤任务追踪
- **检查点** — 结构化 `CheckpointData` / `CheckpointStep` 事件支持重放、调试和恢复
- **工具执行强制** — `ToolEnforcementMiddleware` 促使 LLM 调用工具而非仅描述操作
- **Turn 工具限制** — `TurnToolLimitMiddleware` 可按 turn 限制工具调用次数
- **熔断器** — `ConsecutiveFailureRecovery` 连续失败 N 次后自动停止
- **Thinking / Reasoning** — 支持按模型配置思考预算和思考强度
- **结构化输出** — 通过 `ResponseFormat` 指定输出格式（JSON Schema / JSON Object）
- **会话 ID 生成器** — 可插拔的 `SessionIdGenerator` 支持自定义 ID 策略
- **工具输出截断** — 可配置 `max_tool_output_chars`，附带结构化 `TruncationInfo`
- **工具部分结果** — `ToolContext::emit_partial_result()` 支持长时间工具执行期间流式输出中间结果
- **截断保护** — 自动检测 LLM 达到 token 上限时产生的截断工具调用，强制重新发出完整参数
- **消息队列** — `MessageQueue` 支持 steering/follow-up 双队列，可配置 `QueueMode` 顺序或逐一消费

## Feature Flags

| Flag | 说明 | 默认 |
|------|------|------|
| `sqlite-session` | 启用 `SqliteSessionStore`（SQLite 持久化会话存储） | 关闭 |
| `telemetry` | 启用 OpenTelemetry 集成，支持分布式追踪 | 关闭 |

```toml
[dependencies]
agent-base = { version = "0.2.1", features = ["sqlite-session"] }
```

## 快速上手

### 1. 定义工具

Agent 的任何能力都以 `Tool` 的形式表达：

```rust
use agent_base::{Tool, ToolContext, Content, AgentResult};
use async_trait::async_trait;
use serde_json::{json, Value};

struct WeatherTool;

#[async_trait]
impl Tool for WeatherTool {
    fn name(&self) -> &'static str { "get_weather" }

    fn description(&self) -> &'static str {
        "查询指定城市的天气"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "city": { "type": "string", "description": "城市名称" }
            },
            "required": ["city"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let city = args["city"].as_str().unwrap_or("未知");
        Ok(vec![Content::text(format!("{}天气：22°C，晴", city))])
    }
}
```

### 2. 构建 Agent

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
        .system_prompt("你是一个有用的天气助手。")
        .register_tool(WeatherTool)
        .build()?;

    let session_id = runtime.create_session().await;

    runtime
        .run_turn(session_id, "东京今天天气怎么样？", |event| {
            match event {
                RuntimeEvent::TextDelta { text, .. } => print!("{}", text),
                RuntimeEvent::ToolCallStarted { tool_name, .. } => {
                    println!("\n[调用工具: {}]", tool_name);
                }
                RuntimeEvent::ToolCallFinished { summary, .. } => {
                    println!("[工具结果: {}]", summary);
                }
                RuntimeEvent::RunFinished { .. } => println!("\n[完成]"),
                _ => {}
            }
            Ok(())
        })
        .await?;

    Ok(())
}
```

回调模式让你完全掌控事件处理。简单场景可改用 `run_turn_collect`，它直接返回 `(Vec<RuntimeEvent>, RunOutcome)`。

### 3. 处理工具错误

默认情况下，工具失败会停止运行。对于需要自愈的 Agent（如代码 Agent 需要重试编译），注入 `RetryOnError`：

```rust
use agent_base::RetryOnError;

let runtime = AgentBuilder::new(llm)
    .register_tool(MyTool)
    .error_recovery(Arc::new(RetryOnError))  // ← 失败时重试
    .build()?;
```

### 4. 为敏感工具添加审批

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
        // 通过 UI、CLI 等方式询问用户
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
                title: "确认操作".into(),
                message: format!("是否执行 `{}`？", tool_name),
                risk_level: RiskLevel::Sensitive,
                ..Default::default()
            })
        } else {
            None  // 自动放行
        }
    }

    // before_call / after_call 钩子可用于日志记录、审计等
}

let runtime = AgentBuilder::new(llm)
    .register_tool(DangerousTool)
    .tool_policy(Arc::new(MyToolPolicy))
    .approval_handler(Arc::new(MyApprovalHandler))
    .build()?;
```

## 示例

```bash
# 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 OPENAI_API_KEY 或 ANTHROPIC_API_KEY

# 交互式 REPL
cargo run --example repl

# 完整快速上手示例（工具 + 审批 + 中间件）
cargo run --example quickstart_demo

# 子 Agent 示例
cargo run --example subagent_demo

# 中间件示例
cargo run --example middleware_demo

# 审批策略示例
cargo run --example approval_policy_demo

# 工具上下文示例
cargo run --example tool_context_demo

# Thinking / Reasoning 测试
cargo run --example thinking_test
```

## agent-base 不做什么

- 内置 SSH、文件系统或数据库工具
- 工作流 DAG 或多 Agent 编排引擎
- 记忆或 RAG（检索增强生成）框架
- 终端 UI 或内置审批对话框
- 生产级持久化或事务系统

业务特定的工具和策略属于**上层**（如 `phi-agent`、`agent-works`、`phi-tools`）。

## 典型分层

```
phi-agent / agent-works / ...              ← 框架 / 增强工具集
    └── agent-base                          ← 轻量级运行时内核
```

## v1 语义

| 约定 | 含义 |
|---|---|
| `run_turn` → 回调 `FnMut(RuntimeEvent)` | 实时处理事件流；`run_turn_collect` 则批量返回 |
| `RunOutcome` | `Completed` / `Failed` / `MaxTurnsExceeded` / `Cancelled` |
| `RuntimeEvent::RunFinished` | 流程结束 — 最终状态见 `run_turn` 返回值 |
| 工具失败 → 默认 `StopOnError` | 注入 `RetryOnError` 获得自愈能力 |
| SubAgent → 默认 `Ephemeral` | 使用 `with_persistent()` 共享上下文 |
| Session → 内存是唯一数据源 | `SessionStore` 是可选的持久化适配器 |

## 致谢

本项目在设计过程中参考了 [OpenAI Codex CLI](https://github.com/openai/codex) 项目，尤其在工具编排和任务规划方面有所借鉴。

## 稳定性

本项目处于早期开发阶段（v0.2.1）。核心抽象已趋于稳定但尚未冻结。生态演进过程中可能会有小幅 API 变更。

## 许可证

MIT OR Apache-2.0（见 [LICENSE-MIT](LICENSE-MIT) 与 [LICENSE-APACHE](LICENSE-APACHE)）。
