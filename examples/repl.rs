use std::io::{self, Write};
use std::sync::Arc;

use agent_base::{
    AgentBuilder, AgentError, AgentResult, ApprovalDecision, ApprovalHandler, ApprovalRequest,
    Content, RiskLevel, RuntimeEvent, Tool, ToolContext, ToolDecision, ToolPolicy,
};
use async_trait::async_trait;
use dotenvy::dotenv;
use serde_json::{Value, json};

// ---------------------------------------------------------------------------
// Arithmetic tools
// ---------------------------------------------------------------------------

struct AddTool;

#[async_trait]
impl Tool for AddTool {
    fn name(&self) -> &'static str {
        "add"
    }

    fn description(&self) -> &'static str {
        "Calculate the sum of two integers"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "First addend" },
                "b": { "type": "integer", "description": "Second addend" }
            },
            "required": ["a", "b"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        let result = a + b;
        Ok(vec![Content::text(format!("{} + {} = {}", a, b, result))])
    }
}

struct SubtractTool;

#[async_trait]
impl Tool for SubtractTool {
    fn name(&self) -> &'static str {
        "subtract"
    }

    fn description(&self) -> &'static str {
        "Calculate the difference of two integers（a - b）"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "Minuend" },
                "b": { "type": "integer", "description": "Subtrahend" }
            },
            "required": ["a", "b"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        let result = a - b;
        Ok(vec![Content::text(format!("{} - {} = {}", a, b, result))])
    }
}

struct MultiplyTool;

#[async_trait]
impl Tool for MultiplyTool {
    fn name(&self) -> &'static str {
        "multiply"
    }

    fn description(&self) -> &'static str {
        "Calculate the product of two integers"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "Multiplier" },
                "b": { "type": "integer", "description": "Multiplier" }
            },
            "required": ["a", "b"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        let result = a * b;
        Ok(vec![Content::text(format!("{} × {} = {}", a, b, result))])
    }
}

struct DivideTool;

#[async_trait]
impl Tool for DivideTool {
    fn name(&self) -> &'static str {
        "divide"
    }

    fn description(&self) -> &'static str {
        "Calculate the quotient of two integers（a ÷ b），Returns quotient and remainder"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "integer", "description": "Dividend" },
                "b": { "type": "integer", "description": "Divisor" }
            },
            "required": ["a", "b"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let a = args["a"].as_i64().unwrap_or(0);
        let b = args["b"].as_i64().unwrap_or(0);
        if b == 0 {
            return Ok(vec![Content::text("Error：Divisor cannot be zero")]);
        }
        let quotient = a / b;
        let remainder = a % b;
        Ok(vec![Content::text(format!(
            "{} ÷ {} = {}（remainder {}）",
            a, b, quotient, remainder
        ))])
    }
}

// ---------------------------------------------------------------------------
// CLI Approval handler
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct CliApprovalHandler;

#[async_trait]
impl ApprovalHandler for CliApprovalHandler {
    async fn approve(
        &self,
        request: ApprovalRequest,
        _cancel_token: tokio_util::sync::CancellationToken,
    ) -> AgentResult<ApprovalDecision> {
        println!();
        println!("[Approval request] {}", request.title);
        println!("  Risk level: {:?}", request.risk_level);
        println!("  Content: {}", request.message);

        loop {
            print!("  Choice [y=Allow / a=AlwaysAllow / n=Deny]: ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin()
                .read_line(&mut input)
                .map_err(|e| AgentError::internal(format!("Failed to read input: {e}")))?;
            match input.trim().to_ascii_lowercase().as_str() {
                "y" | "yes" => return Ok(ApprovalDecision::AllowOnce),
                "a" | "always" => return Ok(ApprovalDecision::AllowAlways),
                "n" | "no" => return Ok(ApprovalDecision::Deny),
                _ => println!("  Invalid input，Please enter y / a / n"),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Event printer
// ---------------------------------------------------------------------------

struct EventPrinter;

impl EventPrinter {
    fn handle(event: RuntimeEvent) -> AgentResult<()> {
        match event {
            RuntimeEvent::TextDelta { text, .. } => {
                print!("{}", text);
                io::stdout().flush().unwrap();
            }
            RuntimeEvent::ThoughtDelta { text, .. } => {
                print!("[Thinking]：\x1b[90m{} \x1b[0m", text);
                println!();
                io::stdout().flush().unwrap();
            }
            RuntimeEvent::ToolCallStarted {
                tool_name,
                args_json,
                ..
            } => {
                println!();
                println!("[Tool call] {} (with args: {})", tool_name, args_json);
            }
            RuntimeEvent::ToolCallFinished {
                tool_name, summary, ..
            } => {
                println!("[Tool finished] {} -> {}", tool_name, summary);
            }
            RuntimeEvent::AwaitingApproval { request, .. } => {
                println!(
                    "[Waiting for approval] {} (Risk: {:?})",
                    request.title, request.risk_level
                );
            }
            RuntimeEvent::RunFinished { .. } => {
                println!();
                println!("[Run finished]");
            }
            RuntimeEvent::Checkpoint { .. } => {}
            _ => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tool approval policy
// ---------------------------------------------------------------------------

struct ArithmeticToolPolicy;

#[async_trait]
impl ToolPolicy for ArithmeticToolPolicy {
    async fn evaluate_approval(&self, tool_name: &str, _args: &Value) -> Option<ApprovalRequest> {
        if tool_name == "divide" {
            return Some(ApprovalRequest {
                title: "Division operation".to_string(),
                message: "Allow division execution?".to_string(),
                action_key: Some("divide".to_string()),
                risk_level: RiskLevel::Safe,
                raw: None,
                source: None,
            });
        }
        None
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

// ---------------------------------------------------------------------------
// Main - REPL Entry
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = r#"You are an arithmetic assistant，You can help users perform various arithmetic operations。

The tools available to you include：
- add: Calculate the sum of two integers
- subtract: Calculate the difference of two integers
- multiply: Calculate the product of two integers
- divide: Calculate the quotient of two integers

Please select the appropriate tool based on the user's request. If the request involves multiple steps,
you may call tools step by step。After each calculation，explain the result to the user。"#;

#[tokio::main]
async fn main() -> AgentResult<()> {
    dotenv().ok();

    let api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("DASHSCOPE_API_KEY"))
        .map_err(|_| {
            AgentError::internal(
                "Please set OPENAI_API_KEY or DASHSCOPE_API_KEY environment variable",
            )
        })?;

    let model = std::env::var("OPENAI_MODEL")
        .or_else(|_| std::env::var("DASHSCOPE_MODEL"))
        .unwrap_or_else(|_| "gpt-4o-mini".to_string());

    let base_url = std::env::var("OPENAI_BASE_URL")
        .or_else(|_| std::env::var("DASHSCOPE_BASE_URL"))
        .unwrap_or_else(|_| "https://api.openai.com/v1".to_string());

    let llm_client = llm_unified::create_provider(&llm_trait::LlmConfig {
        protocol: Some(llm_trait::Protocol::OpenAi),
        api_key,
        model: model.clone(),
        base_url,
        options: std::collections::HashMap::new(),
    })
    .map_err(|e| AgentError::internal(e.to_string()))?;

    let runtime = AgentBuilder::new(llm_client)
        .system_prompt(SYSTEM_PROMPT)
        .enable_thought(false)
        .enable_thinking(false)
        .register_tool(AddTool)
        .register_tool(SubtractTool)
        .register_tool(MultiplyTool)
        .register_tool(DivideTool)
        .tool_policy(Arc::new(ArithmeticToolPolicy))
        .approval_handler(Arc::new(CliApprovalHandler))
        .build()
        .unwrap();

    let mut session_id = runtime.create_session().await;

    println!("=== agent-base REPL (arithmetic Demo) ===");
    println!("model: {}", model);
    println!("Input 'exit' quit, 'reset' recreate session");
    println!();

    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| AgentError::internal(format!("Failed to read input: {e}")))?;
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }
        if matches!(input.as_str(), "exit" | "quit") {
            break;
        }
        if input == "reset" {
            session_id = runtime.create_session().await;
            println!("Created new session");
            continue;
        }

        match runtime
            .run_turn(session_id.clone(), &input, |event| {
                EventPrinter::handle(event)
            })
            .await
        {
            Ok(_outcome) => {}
            Err(e) => {
                if e.is_cancelled() {
                    println!("Cancelled");
                } else {
                    println!("Error: {}", e);
                }
            }
        }
    }

    Ok(())
}
