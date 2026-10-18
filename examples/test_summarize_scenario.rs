/// Simulate the real ops-omni scenario: the model has already called a tool, got the
/// df -h result, and now needs to write a text summary.
///
/// Test whether qwen3.7-max produces content in the "summarize the reply" scenario.
use agent_base::{AgentResult, ChatMessage};
use llm_trait::{ChatRequest, LlmProvider, ReasoningConfig, StreamChunk};
use llm_unified::create_provider;

async fn test_summarize(
    provider: &dyn LlmProvider,
    label: &str,
    enable_thinking: bool,
) -> AgentResult<()> {
    println!("\n{}", "=".repeat(70));
    println!("{} — thinking={}", label, enable_thinking);
    println!("{}", "=".repeat(70));

    // Simulate ops-omni's multi-turn conversation: user asks -> model calls a tool -> tool returns a result -> model needs to summarize
    let messages = vec![
        ChatMessage::system("你是运维工程师助手，回复简洁直接。"),
        ChatMessage::user("看下磁盘空间"),
        // The model's previous reply (it called a tool)
        ChatMessage::Assistant {
            content: None,
            reasoning_content: None,
            tool_calls: Some(vec![agent_base::ToolCallMessage {
                id: "call_001".to_string(),
                name: "execute_command".to_string(),
                arguments: r#"{"command":"df -h","target_host":"10.0.0.1"}"#.to_string(),
            }]),
            thinking_signature: None,
        },
        // Result returned by the tool
        ChatMessage::Tool {
            tool_call_id: "call_001".to_string(),
            name: None,
            content: "Filesystem      Size  Used Avail Use% Mounted on\n/dev/vda3        40G  8.0G   30G  22% /\ntmpfs           1.8G     0  1.8G   0% /dev/shm\n/dev/vda2       197M  6.3M  191M   4% /boot/efi".to_string(),
        },
    ];

    let reasoning = ReasoningConfig {
        enabled: Some(enable_thinking),
        budget_tokens: None,
        effort: None,
    };

    let request = ChatRequest::new(messages).with_reasoning(reasoning);
    let mut stream = provider
        .stream(request)
        .await
        .map_err(|e| agent_base::AgentError::internal(e.to_string()))?;

    let mut thought = String::new();
    let mut text = String::new();
    let mut tool_calls = 0;
    let mut has_stop = false;

    while let Some(chunk) = stream.next().await {
        match chunk.map_err(|e| agent_base::AgentError::internal(e.to_string()))? {
            StreamChunk::Thought(t) => {
                thought.push_str(&t);
            }
            StreamChunk::Text(t) => {
                text.push_str(&t);
            }
            StreamChunk::ToolCall(_) => {
                tool_calls += 1;
            }
            StreamChunk::Stop { .. } => {
                has_stop = true;
            }
            StreamChunk::Usage(u) => {
                println!(
                    "  📊 prompt={}, completion={}",
                    u.prompt_tokens.unwrap_or(0),
                    u.completion_tokens.unwrap_or(0)
                );
            }
            StreamChunk::ThinkingSignature(_) => {
                // Thinking signature — not needed for this test
            }
            StreamChunk::Error(e) => {
                eprintln!("  ❌ Stream error: {}", e);
            }
        }
    }

    println!("  💭 Thought: {} chars", thought.len());
    if !thought.is_empty() {
        let preview: String = thought.chars().take(200).collect();
        println!("     {}", preview);
    }
    println!("  📝 Reply: {} chars", text.len());
    if !text.is_empty() {
        let preview: String = text.chars().take(300).collect();
        println!("     {}", preview);
    }
    println!("  🔧 Tool calls: {} | 🏁 Stop: {}", tool_calls, has_stop);

    if text.is_empty() && tool_calls == 0 {
        println!("  ⚠️  Empty response! react will retry");
    } else if text.is_empty() && !thought.is_empty() {
        println!(
            "  ⚠️  Thought but no body text (the thought content should serve as fallback output)"
        );
    } else {
        println!("  ✅ OK");
    }

    Ok(())
}

#[tokio::main]
async fn main() -> AgentResult<()> {
    dotenvy::dotenv().ok();

    let api_key = std::env::var("DASHSCOPE_API_KEY")
        .or_else(|_| std::env::var("OPENAI_API_KEY"))
        .map_err(|_| agent_base::AgentError::internal("API key is not set"))?;

    let base_url = std::env::var("DASHSCOPE_BASE_URL")
        .or_else(|_| std::env::var("OPENAI_BASE_URL"))
        .unwrap_or_else(|_| "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string());

    let models = vec![
        ("qwen-flash", "千问 Flash"),
        ("qwen-plus", "千问 Plus"),
        ("qwen3.7-max", "千问 Max"),
    ];

    for (model_id, model_label) in &models {
        let provider = create_provider(&llm_trait::LlmConfig {
            protocol: Some(llm_trait::Protocol::OpenAi),
            api_key: api_key.clone(),
            model: model_id.to_string(),
            base_url: base_url.clone(),
            options: std::collections::HashMap::new(),
        })
        .map_err(|e| agent_base::AgentError::internal(e.to_string()))?;

        test_summarize(provider.as_ref(), model_label, false).await?;
        test_summarize(provider.as_ref(), model_label, true).await?;
    }

    Ok(())
}
