//! Stream-integrity detection tests, driven by the wire anatomy of a real
//! provider failure.
//!
//! What happened on 2026-11-20 (session `20261120_7dc71c91`): the provider
//! (MiniMax-M3.1-Flash via api.minimax.cn/anthropic) cut a tool call's
//! arguments mid-JSON and leaked the tail into the NEXT content block's index
//! (a thinking block), while reporting `stop_reason: tool_use`. The tail
//! lands in a name-less bucket ("phantom") that assembly drops, so the tool
//! call's args stay unclosed.
//!
//! The fixtures below reproduce that anatomy byte-for-byte in SHAPE (the
//! captured stream itself stays out of the repository — it carries a local
//! path prefix; the analysis lives in phimint's notes and in the project
//! memory, both outside git).

use super::*;

/// Corrupted SSE stream in the captured shape: the `execute_command` args are
/// cut mid-JSON (`{"command":"cat /workspace/app/RE`), and the 9-byte tail
/// `ADME.md"}` arrives as an `input_json_delta` under the NEXT block index
/// (a thinking block, so no tool name ever reaches it).
const CORRUPT_FIXTURE: &str = include_str!("../testdata/corrupt-args-tail.sse.txt");

/// The same stream without the damage: two complete tool calls.
const CLEAN_FIXTURE: &str = include_str!("../testdata/clean-args.sse.txt");

/// Decode an Anthropic-protocol SSE body into engine stream chunks, mirroring
/// the adapter semantics in llm-providers (`convert_event`): tool_use starts
/// and `input_json_delta` fragments become ToolCall chunks keyed by block
/// index; thinking/text starts emit nothing but their deltas map to
/// Thought/Text. This is what the engine actually sees for the fixture bytes.
fn chunks_from_anthropic_sse(sse: &str) -> Vec<Result<StreamChunk, AgentError>> {
    let mut out: Vec<Result<StreamChunk, AgentError>> = Vec::new();
    let (mut ev_type, mut ev_data) = (String::new(), String::new());
    let flush = |ev_type: &str, ev_data: &str, out: &mut Vec<Result<StreamChunk, AgentError>>| {
        if ev_data.is_empty() {
            return;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(ev_data) else {
            return;
        };
        let t = v.get("type").and_then(|x| x.as_str()).unwrap_or(ev_type);
        match t {
            "content_block_start" => {
                let idx = v.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                if v["content_block"]["type"] == "tool_use" {
                    out.push(Ok(StreamChunk::ToolCall(serde_json::json!({
                        "delta": { "tool_calls": [{
                            "index": idx,
                            "id": v["content_block"]["id"],
                            "function": {
                                "name": v["content_block"]["name"],
                                "arguments": "",
                            }
                        }]}
                    }))));
                }
            }
            "content_block_delta" => {
                let idx = v.get("index").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
                let d = &v["delta"];
                match d["type"].as_str().unwrap_or("") {
                    "input_json_delta" => out.push(Ok(StreamChunk::ToolCall(serde_json::json!({
                        "delta": { "tool_calls": [{
                            "index": idx,
                            "function": { "arguments": d["partial_json"] }
                        }]}
                    })))),
                    "thinking_delta" => out.push(Ok(StreamChunk::Thought(
                        d["thinking"].as_str().unwrap_or("").to_string(),
                    ))),
                    "text_delta" => out.push(Ok(StreamChunk::Text(
                        d["text"].as_str().unwrap_or("").to_string(),
                    ))),
                    _ => {}
                }
            }
            "message_delta" => {
                let stop = v["delta"]["stop_reason"].as_str().map(|s| s.to_string());
                out.push(Ok(StreamChunk::Stop {
                    finish_reason: stop,
                }));
            }
            _ => {}
        }
    };
    for raw in sse.lines() {
        if let Some(rest) = raw.strip_prefix("event: ") {
            if !ev_data.is_empty() {
                flush(&ev_type, &ev_data, &mut out);
                ev_data.clear();
            }
            ev_type = rest.trim().to_string();
        } else if let Some(rest) = raw.strip_prefix("data: ") {
            if !ev_data.is_empty() {
                ev_data.push('\n');
            }
            ev_data.push_str(rest.trim());
        } else if raw.is_empty() {
            flush(&ev_type, &ev_data, &mut out);
            ev_data.clear();
            ev_type.clear();
        }
    }
    flush(&ev_type, &ev_data, &mut out);
    out
}

async fn run(sse: &str) -> LlmTurnResult {
    let bus = EventBus::new(16);
    let engine = LlmEngine::new(Arc::new(StubProvider("x")), bus.clone());
    let stream = Box::pin(futures_util::stream::iter(chunks_from_anthropic_sse(sse)));
    let mut rx = bus.subscribe();
    let cancel = tokio_util::sync::CancellationToken::new();
    engine
        .process_stream(
            &SessionId::new(1),
            stream,
            tracing::Span::none(),
            &mut rx,
            std::sync::Arc::new(std::sync::Mutex::new(|_e| Ok(()))),
            &cancel,
        )
        .await
        .unwrap()
}

/// The corruption must be reported as integrity facts: 1 named call with
/// unparseable args + 9 orphan bytes whose fragments landed on a name-less
/// bucket. The orphan bucket itself must still be dropped from `tool_calls`
/// (existing anti-400 behavior).
#[tokio::test]
async fn integrity_flags_misplaced_tail_and_truncated_args() {
    let result = run(CORRUPT_FIXTURE).await;
    assert_eq!(result.integrity.invalid_args, 1);
    assert_eq!(result.integrity.orphan_args_bytes, 9);
    assert!(!result.integrity.is_clean());
    // read_file (valid) + execute_command (truncated) survive; the phantom is dropped.
    assert_eq!(result.tool_calls.len(), 2);
}

/// A healthy stream reports a clean integrity record.
#[tokio::test]
async fn integrity_clean_stream_reports_clean() {
    let result = run(CLEAN_FIXTURE).await;
    assert!(result.integrity.is_clean(), "{:?}", result.integrity);
    assert_eq!(result.tool_calls.len(), 2);
}
