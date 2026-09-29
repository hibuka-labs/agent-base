//! ToolContext Demo — Demonstrates `user_event_tx` and `llm_client` usage
//!
//! Shows three tools that exercise `ToolContext`:
//!   1. `analyze_text`  — uses `ctx.emit_progress()` to report progress
//!   2. `summarize`     — uses `ctx.llm_client` to make a nested LLM call
//!   3. `notify`        — uses `ctx.emit_user_event()` with `UserEvent::Structured`
//!
//! How to run:
//!   cp .env.example .env
//!   # Edit .env and fill in your API Key
//!   cargo run --example tool_context_demo
//!
//! # Testing Guide
//!
//! After starting, you'll enter an interactive REPL. Try these inputs:
//!
//! ## 1. Trigger `analyze_text` (progress events via `ctx.emit_progress`)
//!
//!   Chinese:
//!     帮我分析一下这段文字：今天天气真好，我感觉非常开心，生活充满了美好和希望。
//!
//!   English:
//!     Analyze this text: The weather is wonderful today. I feel happy and great!
//!
//!   Expected output:
//!     [Tool Call] analyze_text ({"text": "..."})
//!       ⏳ [Progress] Analyzing text: tokenizing...
//!       ⏳ [Progress] Analyzing text: counting sentences...
//!       ⏳ [Progress] Analyzing text: evaluating sentiment...
//!       ⏳ [Progress] Analyzing text: complete!
//!     [Tool Result] analyze_text: Char count: ..., Sentence count: ..., Sentiment: positive
//!
//! ## 2. Trigger `summarize` (nested LLM call via `ctx.llm_client`)
//!
//!   Chinese:
//!     帮我总结一下：Rust 是一门系统编程语言，运行速度极快，能防止段错误并保证线程安全。
//!     它通过独特的所有权系统在编译期强制保证内存安全，无需垃圾回收器。
//!
//!   English:
//!     Summarize: Rust is a systems programming language that runs blazingly fast, prevents
//!     segfaults, and guarantees thread safety. It achieves these goals through its unique
//!     ownership system, which enforces memory safety at compile time.
//!
//!   Expected output:
//!     [Tool Call] summarize ({"text": "..."})
//!       ⏳ [Progress] Summarize: calling LLM for one-sentence summary...
//!       ⏳ [Progress] Summarize: done!
//!     [Tool Result] summarize: Summary: ...
//!
//! ## 3. Trigger `notify` (structured event via `ctx.emit_user_event`)
//!
//!   Chinese:
//!     发一条 Slack 消息，内容是"部署完成"，严重级别为 error
//!
//!   English:
//!     Send a Slack notification: deploy complete, severity error
//!
//!   Expected output:
//!     [Tool Call] notify ({"channel": "slack", "message": "...", "severity": "error"})
//!       📦 [Structured] type=notification data={ ... }
//!     [Tool Result] notify: Notification sent to slack (severity: error): ...
//!
//! ## 4. Multi-tool (LLM calls multiple tools in one turn)
//!
//!   Chinese:
//!     先分析一下这段话，然后总结一下：人工智能正在改变世界，从自动驾驶到医疗诊断，
//!     从语音识别到自然语言处理，AI 的应用无处不在，未来充满无限可能。
//!
//!   English:
//!     First analyze, then summarize: AI is changing the world. From self-driving cars
//!     to medical diagnosis, from speech recognition to NLP, AI is everywhere.
//!
//! ## 5. No-tool (greeting, should NOT trigger any tool)
//!
//!   你好 / Hello
//!
//!   Expected: direct reply from LLM, no tool call.

use std::io::Write;

use agent_base::{
    AgentBuilder, AgentResult, ChatMessage, Content, RuntimeEvent, Tool, ToolContext, UserEvent,
};
use async_trait::async_trait;
use dotenvy::dotenv;
use serde_json::{Value, json};

// ============================================================================
// Tool 1: analyze_text — uses ctx.emit_progress()
// ============================================================================

/// Text analysis tool that supports both Chinese and English.
///
/// Demonstrates `ctx.emit_progress()` to report incremental progress
/// back to the host application through the `user_event_tx` channel.
struct AnalyzeTextTool;

#[async_trait]
impl Tool for AnalyzeTextTool {
    fn name(&self) -> &'static str {
        "analyze_text"
    }

    fn description(&self) -> &'static str {
        "Analyze text and return character/word count, sentence count, and sentiment (positive/negative/neutral). Supports both Chinese and English. Emits progress events during execution. Only call when the user explicitly provides text to analyze."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to analyze (Chinese or English)"
                }
            },
            "required": ["text"]
        })
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let text = args["text"].as_str().unwrap_or("");

        // Step 1: Count characters / words
        ctx.emit_progress("Analyzing text: tokenizing...");

        let is_chinese = text.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));

        let char_or_word_count = if is_chinese {
            // For CJK text, count non-whitespace characters (each char ≈ one word)
            text.chars().filter(|c| !c.is_whitespace()).count()
        } else {
            // For English, split by whitespace
            text.split_whitespace().count()
        };

        // Step 2: Count sentences (supports both Chinese and English punctuation)
        ctx.emit_progress("Analyzing text: counting sentences...");
        let sentence_count = text
            .matches(|c: char| {
                c == '.' || c == '!' || c == '?' || c == '。' || c == '！' || c == '？'
            })
            .count()
            .max(if text.trim().is_empty() { 0 } else { 1 });

        // Step 3: Sentiment analysis with bilingual keyword lists
        ctx.emit_progress("Analyzing text: evaluating sentiment...");

        let positive_en = [
            "good",
            "great",
            "excellent",
            "happy",
            "love",
            "wonderful",
            "amazing",
            "fantastic",
        ];
        let negative_en = [
            "bad",
            "terrible",
            "awful",
            "hate",
            "sad",
            "horrible",
            "worst",
            "disgusting",
        ];
        let positive_zh = ["开心", "快乐", "美好", "希望", "喜欢", "棒", "优秀", "幸福"];
        let negative_zh = [
            "难过", "糟糕", "讨厌", "悲伤", "可怕", "失望", "痛苦", "愤怒",
        ];

        let lower = text.to_lowercase();
        let pos = positive_en.iter().filter(|w| lower.contains(*w)).count()
            + positive_zh.iter().filter(|w| text.contains(*w)).count();
        let neg = negative_en.iter().filter(|w| lower.contains(*w)).count()
            + negative_zh.iter().filter(|w| text.contains(*w)).count();

        let sentiment = if pos > neg {
            "positive"
        } else if neg > pos {
            "negative"
        } else {
            "neutral"
        };

        // Done
        ctx.emit_progress("Analyzing text: complete!");

        let count_label = if is_chinese { "chars" } else { "words" };
        Ok(vec![Content::text(format!(
            "{} {}: {}, Sentence count: {}, Sentiment: {}",
            if is_chinese { "Char" } else { "Word" },
            count_label,
            char_or_word_count,
            sentence_count,
            sentiment
        ))])
    }
}

// ============================================================================
// Tool 2: summarize — uses ctx.llm_client to make a nested LLM call
// ============================================================================

/// Summarization tool that delegates to the LLM via `ctx.llm_client`.
///
/// Demonstrates how a tool can make its own LLM call using the client
/// provided in `ToolContext`, enabling tool-level AI orchestration.
struct SummarizeTool;

#[async_trait]
impl Tool for SummarizeTool {
    fn name(&self) -> &'static str {
        "summarize"
    }

    fn description(&self) -> &'static str {
        "Summarize the given text into one sentence using the LLM. Only call when the user explicitly asks to summarize a longer passage."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to summarize"
                }
            },
            "required": ["text"]
        })
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let text = args["text"].as_str().unwrap_or("");

        // Get the LLM client from ToolContext — returns error if not configured.
        let llm = ctx.llm_client.as_ref().ok_or_else(|| {
            agent_base::AgentError::internal("No LLM client available in ToolContext")
        })?;

        ctx.emit_progress("Summarize: calling LLM for one-sentence summary...");

        // Make a nested LLM call from within the tool.
        let messages = vec![
            ChatMessage::system(
                "You are a concise summarizer. Respond with exactly one sentence. Match the language of the input text.",
            ),
            ChatMessage::user(format!("Summarize this:\n\n{}", text)),
        ];

        let raw = llm.chat(llm_trait::ChatRequest::new(messages)).await?;

        // LlmProvider::chat() returns a ChatResponse; check content.
        let summary = if raw.content.is_empty() {
            "(no summary)"
        } else {
            &raw.content
        };

        ctx.emit_progress("Summarize: done!");

        Ok(vec![Content::text(format!("Summary: {}", summary))])
    }
}

// ============================================================================
// Tool 3: notify — uses ctx.emit_user_event() with UserEvent::Structured
// ============================================================================

/// Notification tool that emits a `UserEvent::Structured` event.
///
/// Demonstrates how a tool can send custom structured events back to the
/// host application for routing (e.g. Slack, email, webhook).
struct NotifyTool;

#[async_trait]
impl Tool for NotifyTool {
    fn name(&self) -> &'static str {
        "notify"
    }

    fn description(&self) -> &'static str {
        "Send a notification to an external channel (Slack, email, webhook). Emits a structured event for the host to route. Only call when the user explicitly asks to send a notification."
    }

    fn schema(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "channel": {
                    "type": "string",
                    "description": "Notification channel: slack, email, webhook"
                },
                "message": {
                    "type": "string",
                    "description": "The notification message body"
                },
                "severity": {
                    "type": "string",
                    "description": "info, warning, or error"
                }
            },
            "required": ["channel", "message"]
        })
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let channel = args["channel"].as_str().unwrap_or("slack");
        let message = args["message"].as_str().unwrap_or("");
        let severity = args["severity"].as_str().unwrap_or("info");

        // Emit a structured event so the host application can route the notification.
        // This goes through `ctx.user_event_tx` and arrives as
        // `RuntimeEvent::UserEvent(UserEvent::Structured { .. })` in the event stream.
        ctx.emit_user_event(UserEvent::Structured {
            event_type: "notification".to_string(),
            data: json!({
                "channel": channel,
                "message": message,
                "severity": severity,
                "timestamp": "2026-06-05T12:00:00Z"
            }),
        });

        Ok(vec![Content::text(format!(
            "Notification sent to {} (severity: {}): {}",
            channel, severity, message
        ))])
    }
}

// ============================================================================
// System prompt — bilingual to support both Chinese and English input
// ============================================================================

const SYSTEM_PROMPT: &str = r#"You are a helpful assistant with the following tools:
- analyze_text: Analyze text for character/word count, sentence count, and sentiment.
- summarize: Summarize a long passage into one sentence.
- notify: Send a notification to an external channel (Slack, email, webhook).

Rules:
- Simple greetings and chitchat do NOT require any tool — just reply directly.
- Only call a tool when the user's request clearly matches its purpose.
- If unsure, ask the user to clarify first."#;

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> AgentResult<()> {
    dotenv().ok();

    let api_key = std::env::var("OPENAI_API_KEY")
        .or_else(|_| std::env::var("DASHSCOPE_API_KEY"))
        .map_err(|_| {
            agent_base::AgentError::internal(
                "Please set OPENAI_API_KEY or DASHSCOPE_API_KEY in your .env file",
            )
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
    .map_err(|e| agent_base::AgentError::internal(e.to_string()))?;

    let runtime = AgentBuilder::new(llm)
        .system_prompt(SYSTEM_PROMPT)
        .enable_thought(false)
        .enable_thinking(false)
        .register_tool(AnalyzeTextTool)
        .register_tool(SummarizeTool)
        .register_tool(NotifyTool)
        .build()?;

    let session_id = runtime.create_session().await;

    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║         ToolContext Demo (user_event_tx + llm_client)    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    println!("║  Model: {:<50} ║", model);
    println!("║                                                          ║");
    println!("║  Tools:                                                  ║");
    println!("║    · analyze_text  — emit_progress() for each step       ║");
    println!("║    · summarize     — ctx.llm_client nested LLM call      ║");
    println!("║    · notify        — emit_user_event(Structured)          ║");
    println!("║                                                          ║");
    println!("║  Try (Chinese or English):                               ║");
    println!("║    \"帮我分析一下这段文字：...\"                              ║");
    println!("║    \"Analyze this text: ...\"                               ║");
    println!("║    \"帮我总结一下：...\"                                      ║");
    println!("║    \"Summarize: ...\"                                       ║");
    println!("║    \"发一条 Slack 消息：部署完成\"                             ║");
    println!("║    \"Send a Slack notification: deploy complete\"           ║");
    println!("║                                                          ║");
    println!("║  Commands: exit=quit  reset=reset session                ║");
    println!("╚══════════════════════════════════════════════════════════╝");
    println!();

    // Event loop using `run_turn` — events arrive in real-time via callback.
    loop {
        print!("User > ");
        std::io::stdout()
            .flush()
            .map_err(|e| agent_base::AgentError::internal(format!("flush failed: {e}")))?;

        let mut input = String::new();
        std::io::stdin()
            .read_line(&mut input)
            .map_err(|e| agent_base::AgentError::internal(format!("read stdin failed: {e}")))?;
        let input = input.trim().to_string();

        if input.is_empty() {
            continue;
        }
        if matches!(input.as_str(), "exit" | "quit") {
            println!("Goodbye!");
            break;
        }
        if input == "reset" {
            println!("Session reset is not supported in this demo. Restart the binary.");
            continue;
        }

        let assistant_started = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
        let assistant_started_clone = assistant_started.clone();
        runtime
            .run_turn(session_id.clone(), &input, move |event| {
                match event {
                    RuntimeEvent::TextDelta { text, .. } => {
                        if !assistant_started_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            print!("Assistant > ");
                            assistant_started_clone
                                .store(true, std::sync::atomic::Ordering::Relaxed);
                        }
                        print!("{}", text);
                    }
                    RuntimeEvent::ToolCallStarted {
                        tool_name,
                        args_json,
                        ..
                    } => {
                        if assistant_started_clone.load(std::sync::atomic::Ordering::Relaxed) {
                            println!();
                            assistant_started_clone
                                .store(false, std::sync::atomic::Ordering::Relaxed);
                        }
                        println!("[Tool Call] {} ({})", tool_name, args_json);
                    }
                    RuntimeEvent::ToolCallFinished {
                        tool_name, summary, ..
                    } => {
                        println!("[Tool Result] {}: {}", tool_name, summary);
                    }
                    RuntimeEvent::UserEvent { event, .. } => match event {
                        UserEvent::Progress { text } => {
                            println!("  ⏳ [Progress] {}", text);
                        }
                        UserEvent::ToolPartialResult {
                            tool_call_id,
                            content,
                            is_partial,
                        } => {
                            let marker = if is_partial { "⏳" } else { "✅" };
                            println!(
                                "  {} [Partial:{}] {}",
                                marker,
                                tool_call_id,
                                &content[..content.len().min(100)]
                            );
                        }
                        UserEvent::Structured { event_type, data } => {
                            println!(
                                "  📦 [Structured] type={} data={}",
                                event_type,
                                serde_json::to_string_pretty(&data).unwrap_or_default()
                            );
                        }
                        UserEvent::Notice { kind, source, text } => {
                            println!("  🔔 [Notice] {:?}/{source}: {text}", kind);
                        }
                    },
                    RuntimeEvent::RunFinished { .. }
                        if assistant_started_clone.load(std::sync::atomic::Ordering::Relaxed) =>
                    {
                        println!();
                    }
                    _ => {}
                }
                Ok(())
            })
            .await?;
        println!();
    }

    Ok(())
}
