//! Quickstart Demo — Corresponding to the QUICKSTART.md tutorial
//!
//! A complete server health check Agent demonstrating:
//!   - Tool definitions (disk check, memory check, service restart)
//!   - Approval flow (ToolPolicy + ApprovalHandler)
//!   - Middleware (anti-hallucination nudge)
//!   - Real-time event stream
//!   - Multi-turn REPL conversation
//!
//! How to run:
//!   cp .env.example .env
//!   # Edit .env and fill in your API Key
//!   cargo run --example quickstart_demo

use std::io::{self, Write};
use std::sync::Arc;

use agent_base::{
    AgentBuilder, AgentError, AgentResult, ApprovalDecision, ApprovalHandler, ApprovalRequest,
    Content, RiskLevel, RuntimeEvent, Tool, ToolContext, ToolDecision, ToolPolicy,
};
use async_trait::async_trait;
use dotenvy::dotenv;
use serde_json::{Value, json};

// ============================================================================
// Tool definitions
// ============================================================================

/// Disk usage check tool
struct DiskCheckTool;

#[async_trait]
impl Tool for DiskCheckTool {
    fn name(&self) -> &'static str {
        "check_disk"
    }

    fn description(&self) -> &'static str {
        "Check server disk usage. Returns used/total space and usage percentage."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Filesystem path to check (e.g. '/', '/home', '/var')"
                }
            },
            "required": ["path"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let path = args["path"].as_str().unwrap_or("/");
        let output = format!(
            "Filesystem: {}\nTotal: 50G  Used: 32G  Available: 18G  Usage: 64%",
            path
        );
        Ok(vec![Content::text(output)])
    }
}

/// Memory usage check tool
struct MemCheckTool;

#[async_trait]
impl Tool for MemCheckTool {
    fn name(&self) -> &'static str {
        "check_memory"
    }

    fn description(&self) -> &'static str {
        "Check server memory usage. Returns used/total memory and usage percentage."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {}
        })
    }

    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        Ok(vec![Content::text(
            "Total: 16G  Used: 12G  Available: 4G  Usage: 75%\nSwap: Total 4G  Used 512M",
        )])
    }
}

/// Service restart tool (sensitive operation, requires approval)
struct RestartServiceTool;

#[async_trait]
impl Tool for RestartServiceTool {
    fn name(&self) -> &'static str {
        "restart_service"
    }

    fn description(&self) -> &'static str {
        "Restart a specified system service. This operation causes a brief service interruption and requires manual approval."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Service name (e.g. nginx, mysql, redis)"
                }
            },
            "required": ["service"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let service = args["service"].as_str().unwrap_or("unknown");
        Ok(vec![Content::text(format!(
            "Service '{}' has been successfully restarted. Status: active (running)",
            service
        ))])
    }
}

// ============================================================================
// Approval: ToolPolicy + ApprovalHandler
// ============================================================================

/// Approval policy: restart_service requires manual approval, others are auto-approved
struct HealthCheckPolicy;

#[async_trait]
impl ToolPolicy for HealthCheckPolicy {
    async fn evaluate_approval(&self, tool_name: &str, args: &Value) -> Option<ApprovalRequest> {
        match tool_name {
            "restart_service" => {
                let service = args
                    .get("service")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                Some(ApprovalRequest {
                    title: "Restart Service".into(),
                    message: format!(
                        "Allow restart of service '{}'? This will cause a brief interruption.",
                        service
                    ),
                    risk_level: RiskLevel::Sensitive,
                    action_key: Some(format!("restart:{}", service)),
                    raw: None,
                    source: None,
                })
            }
            _ => None,
        }
    }

    fn before_call(
        &self,
        _tool_name: &str,
        _args: &Value,
        _ctx: &ToolContext,
    ) -> AgentResult<ToolDecision> {
        Ok(ToolDecision::Proceed)
    }

    fn after_call(
        &self,
        _tool_name: &str,
        _args: &Value,
        _result: &[Content],
        _ctx: &ToolContext,
    ) -> AgentResult<()> {
        Ok(())
    }
}

/// CLI-based approval interaction
struct CliApproval;

#[async_trait]
impl ApprovalHandler for CliApproval {
    async fn approve(
        &self,
        request: ApprovalRequest,
        _cancel_token: tokio_util::sync::CancellationToken,
    ) -> AgentResult<ApprovalDecision> {
        println!();
        println!("⚠️  Approval Request: {}", request.title);
        println!("   Risk Level: {:?}", request.risk_level);
        println!("   Details: {}", request.message);

        loop {
            print!("   Choose [y=allow once / a=allow always / n=deny]: ");
            io::stdout()
                .flush()
                .map_err(|e| AgentError::internal(format!("flush stdout failed: {e}")))?;

            let mut input = String::new();
            match io::stdin().read_line(&mut input) {
                Ok(0) => {
                    println!("   [stdin EOF, defaulting to deny]");
                    return Ok(ApprovalDecision::Deny);
                }
                Ok(_) => {}
                Err(e) => {
                    return Err(AgentError::internal(format!("read stdin failed: {e}")));
                }
            }
            match input.trim().to_ascii_lowercase().as_str() {
                "y" | "yes" => return Ok(ApprovalDecision::AllowOnce),
                "a" | "always" => return Ok(ApprovalDecision::AllowAlways),
                "n" | "no" => return Ok(ApprovalDecision::Deny),
                "" => {
                    return Ok(ApprovalDecision::Deny);
                }
                _ => println!("   Invalid input, enter y / a / n"),
            }
        }
    }
}

// ============================================================================
// Event printing
// ============================================================================

struct CliEventPrinter {
    assistant_prefix_printed: bool,
}

impl CliEventPrinter {
    fn new() -> Self {
        Self {
            assistant_prefix_printed: false,
        }
    }

    fn handle(&mut self, event: RuntimeEvent) -> AgentResult<()> {
        match event {
            RuntimeEvent::TextDelta { text, .. } => {
                if !self.assistant_prefix_printed {
                    print!("Assistant > ");
                    self.assistant_prefix_printed = true;
                }
                print!("{}", text);
                io::stdout()
                    .flush()
                    .map_err(|e| AgentError::internal(format!("flush failed: {e}")))?;
            }
            RuntimeEvent::ThoughtDelta { text, .. } => {
                print!("\x1b[90m[Thought] {} \x1b[0m", text);
                io::stdout()
                    .flush()
                    .map_err(|e| AgentError::internal(format!("flush failed: {e}")))?;
            }
            RuntimeEvent::ToolCallStarted {
                tool_name,
                args_json,
                ..
            } => {
                self.finish();
                println!("[Tool Call] {} ({})", tool_name, args_json);
            }
            RuntimeEvent::ToolCallFinished {
                tool_name, summary, ..
            } => {
                self.finish();
                let display = if summary.len() > 300 {
                    format!("{}...", &summary[..300])
                } else {
                    summary.clone()
                };
                println!("[Tool Finish] {}", tool_name);
                println!("  -> {}", display);
            }
            RuntimeEvent::AwaitingApproval { .. } => {
                self.finish();
            }
            RuntimeEvent::RunFinished { .. } => {
                self.finish();
            }
            _ => {}
        }
        Ok(())
    }

    fn finish(&mut self) {
        if self.assistant_prefix_printed {
            println!();
            self.assistant_prefix_printed = false;
        }
    }
}

// ============================================================================
// System Prompt
// ============================================================================

const SYSTEM_PROMPT: &str = r#"You are a server health check assistant.

You have the following tools available:
- check_disk: Check disk usage at a specified path
- check_memory: Check memory usage
- restart_service: Restart a system service (requires approval)

When the user asks about server health, **you must call the tools** to retrieve data — do not fabricate numbers.
Keep your answers concise and report results in bullet points.
If usage is high, proactively alert the user and offer recommendations."#;

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> AgentResult<()> {
    dotenv().ok();

    let api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("DASHSCOPE_API_KEY"))
        .map_err(|_| {
            AgentError::internal("Please set OPENAI_API_KEY or DASHSCOPE_API_KEY in your .env file")
        })?;

    let model = std::env::var("OPENAI_MODEL")
        .or_else(|_| std::env::var("DASHSCOPE_MODEL"))
        .unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let base_url = std::env::var("OPENAI_BASE_URL")
        .or_else(|_| std::env::var("DASHSCOPE_BASE_URL"))
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let llm = llm_unified::create_provider(&llm_trait::LlmConfig {
        protocol: Some(llm_trait::Protocol::OpenAi),
        api_key,
        model: model.clone(),
        base_url,
        options: std::collections::HashMap::new(),
    })
    .map_err(|e| AgentError::internal(e.to_string()))?;

    let runtime = AgentBuilder::new(llm)
        .system_prompt(SYSTEM_PROMPT)
        .enable_thought(false)
        .enable_thinking(false)
        .register_tool(DiskCheckTool)
        .register_tool(MemCheckTool)
        .register_tool(RestartServiceTool)
        .tool_policy(Arc::new(HealthCheckPolicy))
        .approval_handler(Arc::new(CliApproval))
        .build()?;

    let mut session_id = runtime.create_session().await;

    println!("╔══════════════════════════════════════════════════════╗");
    println!("║     agent-base Quickstart Demo (Health Check)       ║");
    println!("╠══════════════════════════════════════════════════════╣");
    println!("║  Model: {:<46} ║", model);
    println!("║                                                      ║");
    println!("║  Available tools:                                    ║");
    println!("║    · check_disk      Check disk usage                ║");
    println!("║    · check_memory    Check memory usage              ║");
    println!("║    · restart_service Restart service (needs approval)║");
    println!("║                                                      ║");
    println!("║  Try saying:                                         ║");
    println!("║    \"Check the disk\"                                  ║");
    println!("║    \"How's the memory?\"                               ║");
    println!("║    \"Restart nginx\"                                   ║");
    println!("║    \"Full server health check\"                       ║");
    println!("║                                                      ║");
    println!("║  Commands: exit=quit  reset=reset session            ║");
    println!("║             session=view chat history                ║");
    println!("╚══════════════════════════════════════════════════════╝");
    println!();

    loop {
        print!("User > ");
        io::stdout()
            .flush()
            .map_err(|e| AgentError::internal(format!("flush failed: {e}")))?;

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                println!("Goodbye!");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                return Err(AgentError::internal(format!("read stdin failed: {e}")));
            }
        }
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }
        if matches!(input.as_str(), "exit" | "quit") {
            println!("Goodbye!");
            break;
        }
        if input == "reset" {
            session_id = runtime.create_session().await;
            println!("Session reset\n");
            continue;
        }
        if input == "session" {
            if let Some(session) = runtime.session(&session_id).await {
                println!("\n--- Chat History ---");
                for msg in session.chat_messages() {
                    match msg {
                        agent_base::ChatMessage::System { content, .. } => {
                            println!("[System] {}...", &content[..content.len().min(80)]);
                        }
                        agent_base::ChatMessage::User { content, .. } => {
                            println!("[User] {}", content);
                        }
                        agent_base::ChatMessage::Assistant {
                            content,
                            tool_calls,
                            ..
                        } => {
                            if let Some(tc) = tool_calls {
                                println!(
                                    "[Assistant] Tool calls: {:?}",
                                    tc.iter()
                                        .map(|t| format!("{}({})", t.name, t.arguments))
                                        .collect::<Vec<_>>()
                                );
                            } else if let Some(c) = content {
                                let display = if c.len() > 120 {
                                    format!("{}...", &c[..120])
                                } else {
                                    c.clone()
                                };
                                println!("[Assistant] {}", display);
                            }
                        }
                        agent_base::ChatMessage::Tool {
                            tool_call_id,
                            content,
                            ..
                        } => {
                            let display = if content.len() > 120 {
                                format!("{}...", &content[..120])
                            } else {
                                content.clone()
                            };
                            println!("[Tool:{}] {}", tool_call_id, display);
                        }
                        agent_base::ChatMessage::Custom { role, data } => {
                            println!("[Custom:{}] {}", role, data);
                        }
                    }
                }
                println!("---------------\n");
            }
            continue;
        }

        let printer = std::sync::Arc::new(std::sync::Mutex::new(CliEventPrinter::new()));
        let printer_clone = printer.clone();
        match runtime
            .run_turn(session_id.clone(), &input, move |event| {
                printer_clone.lock().unwrap().handle(event)
            })
            .await
        {
            Ok(_outcome) => {
                printer.lock().unwrap().finish();
            }
            Err(e) => {
                printer.lock().unwrap().finish();
                if e.is_cancelled() {
                    println!("[Cancelled]");
                } else {
                    eprintln!("[Error] {}", e);
                }
            }
        }
    }

    Ok(())
}
