use std::sync::atomic::{AtomicUsize, Ordering};

use agent_base::{
    AgentBuilder, AgentError, AgentResult, ChatMessage, Content, Middleware, PostLlmCtx,
    RuntimeEvent, SessionId, Tool, ToolContext,
};
use async_trait::async_trait;
use dotenvy::dotenv;
use serde_json::{Value, json};

struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn name(&self) -> &'static str {
        "echo"
    }

    fn description(&self) -> &'static str {
        "echo back the message"
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "message": { "type": "string" }
            },
            "required": ["message"]
        })
    }

    async fn call(&self, args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let msg = args["message"].as_str().unwrap_or("");
        Ok(vec![Content::text(format!("echo: {msg}"))])
    }
}

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

// ---------------------------------------------------------------------------
// Anti-Hallucination Middleware (implemented at the business layer)
// ---------------------------------------------------------------------------

struct AntiHallucinationMiddleware {
    max_nudges: usize,
    nudge_count: AtomicUsize,
}

impl AntiHallucinationMiddleware {
    fn new(max_nudges: usize) -> Self {
        Self {
            max_nudges,
            nudge_count: AtomicUsize::new(0),
        }
    }
}

#[async_trait]
impl Middleware for AntiHallucinationMiddleware {
    async fn on_post_llm(&self, ctx: &mut PostLlmCtx) -> AgentResult<()> {
        // Only triggers when "tools are available but no tool has been executed yet"
        if ctx.available_tools.is_empty() || ctx.total_tool_calls > 0 {
            return Ok(());
        }

        // If the LLM has already called a tool, nothing to do
        if ctx.is_tool_call {
            return Ok(());
        }

        // If the LLM returned no text, nothing to do either
        if ctx.full_text.is_empty() {
            return Ok(());
        }

        let count = self.nudge_count.fetch_add(1, Ordering::SeqCst);

        if count >= self.max_nudges {
            // Exceeded the maximum nudge count: do not suppress, let the LLM text output normally
            return Ok(());
        }

        println!(
            "[Middleware] Nudge #{count}: suppressing hallucination text, injecting follow-up message"
        );

        // Silently drop the current assistant message
        ctx.skip_push = true;

        // Set the nudge follow-up message; react will inject it and continue
        ctx.follow_up_message = Some(
            "CRITICAL: You have tools available but did not call any. \
             Do NOT describe what you would do — actually DO it by calling tools. \
             Call the appropriate tool NOW."
                .to_string(),
        );

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Sensitive content retraction Middleware (another skip_push use case)
// ---------------------------------------------------------------------------

struct SensitiveContentFilter {
    blocked_words: Vec<String>,
}

impl SensitiveContentFilter {
    fn new(words: Vec<String>) -> Self {
        Self {
            blocked_words: words,
        }
    }
}

#[async_trait]
impl Middleware for SensitiveContentFilter {
    async fn on_post_llm(&self, ctx: &mut PostLlmCtx) -> AgentResult<()> {
        let lower = ctx.full_text.to_lowercase();
        for word in &self.blocked_words {
            if lower.contains(&word.to_lowercase()) {
                println!("[Middleware] Blocked sensitive content containing '{word}', suppressing");
                ctx.skip_push = true;
                ctx.full_text.clear();
                return Ok(());
            }
        }
        Ok(())
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
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
            RuntimeEvent::ToolCallStarted {
                tool_name,
                args_json,
                ..
            } => {
                println!();
                println!("[Tool call] {} (args: {})", tool_name, args_json);
            }
            RuntimeEvent::ToolCallFinished {
                tool_name, summary, ..
            } => {
                println!("[Tool finished] {} -> {}", tool_name, summary);
            }
            RuntimeEvent::RunFinished { .. } => {
                println!();
                println!("[Run finished]");
            }
            _ => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Session inspector
// ---------------------------------------------------------------------------

async fn print_session_messages(runtime: &agent_base::AgentRuntime, session_id: &SessionId) {
    match runtime.session(session_id).await {
        Some(session) => {
            println!("\n--- Session Messages ---");
            for msg in session.chat_messages() {
                match msg {
                    ChatMessage::System { content, .. } => {
                        println!("[SYSTEM] {}", content);
                    }
                    ChatMessage::User { content, .. } => {
                        if content.len() > 120 {
                            println!("[USER] {}...", &content[..120]);
                        } else {
                            println!("[USER] {}", content);
                        }
                    }
                    ChatMessage::Assistant {
                        content,
                        tool_calls,
                        ..
                    } => {
                        if let Some(tc) = tool_calls {
                            println!(
                                "[ASSISTANT] tool_calls: {:?}",
                                tc.iter()
                                    .map(|t| format!("{}({})", t.name, t.arguments))
                                    .collect::<Vec<_>>()
                            );
                        } else if let Some(c) = content {
                            println!("[ASSISTANT] {}", c);
                        }
                    }
                    ChatMessage::Tool {
                        tool_call_id,
                        content,
                        ..
                    } => {
                        if content.len() > 120 {
                            println!("[TOOL:{}] {}...", tool_call_id, &content[..120]);
                        } else {
                            println!("[TOOL:{}] {}", tool_call_id, content);
                        }
                    }
                    ChatMessage::Custom { role, data } => {
                        println!("[CUSTOM:{}] {}", role, data);
                    }
                }
            }
            println!("-------------------------");
        }
        None => {
            println!("Session not found");
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const SYSTEM_PROMPT: &str = r#"You are a helpful assistant with calculation tools.

Available tools:
- add: Calculate the sum of two integers
- echo: echo back a message

When the user asks for a calculation, you MUST use the appropriate tool.
Do NOT describe what you would do — actually call the tool."#;

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
        .register_tool(AddTool)
        .register_tool(EchoTool)
        .middleware(AntiHallucinationMiddleware::new(3))
        .middleware(SensitiveContentFilter::new(vec![
            "password".to_string(),
            "api_key".to_string(),
        ]))
        .build()
        .unwrap();

    let session_id = runtime.create_session().await;

    println!("=== Middleware Demo ===");
    println!("Model: {model}");
    println!();
    println!("This demo shows two middleware use cases:");
    println!(
        "  1. AntiHallucinationMiddleware — if LLM text-replies without calling tools, suppress + nudge"
    );
    println!(
        "  2. SensitiveContentFilter — if LLM output contains blocked words, suppress the message"
    );
    println!();
    println!("Try asking: 'calculate 3 + 5'");
    println!("Type 'exit' to quit, 'session' to view session history");
    println!();

    loop {
        use std::io::{self, Write};

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
        if input == "session" {
            print_session_messages(&runtime, &session_id).await;
            continue;
        }

        match runtime
            .run_turn(session_id.clone(), &input, |event| {
                EventPrinter::handle(event)
            })
            .await
        {
            Ok(outcome) => {
                println!("\nOutcome: {:?}", outcome);
            }
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
