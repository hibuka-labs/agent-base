use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use crate::engine::runtime::llm_engine::LlmTurnResult;
use crate::engine::runtime::plan_runner::RuntimeCore;
use crate::types::{AgentResult, RuntimeEvent, SessionId};

/// Mid-stream retry: if the SSE stream breaks after the connection is
/// established (e.g. server drops the connection mid-response), retry with a
/// fresh stream.  Safe because `process_stream` fails before any tool is
/// executed, so the message history is unchanged.
const STREAM_RETRY_MAX: u32 = 3;
const STREAM_RETRY_INITIAL_MS: u64 = 1_000;
const STREAM_RETRY_MAX_MS: u64 = 10_000;
const STREAM_RETRY_BACKOFF: f64 = 2.0;

impl RuntimeCore {
    /// Call the LLM and process the stream, retrying on mid-stream errors.
    ///
    /// Wraps both the initial LLM call (with optional connection-level retry
    /// via `config.llm.llm_retry`) and the mid-stream retry loop that handles
    /// SSE connection drops after the stream is established.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn llm_call_with_retry<F>(
        &self,
        session_id: &SessionId,
        messages: &[crate::types::ChatMessage],
        tools: &[serde_json::Value],
        config: &crate::types::AgentConfig,
        thinking_disabled: bool,
        turn_count: u32,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<LlmTurnResult>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        let mut retry_left: u32 = STREAM_RETRY_MAX;
        let mut retry_delay_ms: u64 = STREAM_RETRY_INITIAL_MS;
        let cancel_token = self.cancel_token();

        loop {
            let stream = match config.llm.llm_retry.as_ref() {
                Some(retry) => {
                    tracing::debug!(
                        session_id = session_id.id,
                        turn = turn_count,
                        "LLM: using retry mode"
                    );
                    self.llm_engine
                        .run_llm_turn_with_retry(
                            session_id,
                            messages,
                            tools,
                            config.reasoning.as_ref(),
                            config.llm.response_format.as_ref(),
                            retry.clone(),
                            thinking_disabled,
                            &cancel_token,
                        )
                        .await?
                }
                None => {
                    tracing::debug!(
                        session_id = session_id.id,
                        turn = turn_count,
                        "LLM: calling chat_stream"
                    );
                    self.llm_engine
                        .chat_stream(
                            messages,
                            tools,
                            config.reasoning.as_ref(),
                            config.llm.response_format.as_ref(),
                            thinking_disabled,
                            &cancel_token,
                        )
                        .await?
                }
            };
            tracing::info!(
                session_id = session_id.id,
                turn = turn_count,
                "LLM stream obtained, processing"
            );

            let span =
                tracing::info_span!("llm_turn", session_id = session_id.id, turn = turn_count);
            let cancel_token = self.cancel_token();
            let result = self
                .llm_engine
                .process_stream(
                    session_id,
                    stream,
                    span,
                    event_rx,
                    on_event.clone(),
                    &cancel_token,
                )
                .await;
            tracing::info!(
                session_id = session_id.id,
                turn = turn_count,
                is_err = result.is_err(),
                "LLM stream processed"
            );

            // Retry on mid-stream errors (not cancellations).
            // process_stream fails before tool execution, so the
            // message history is unchanged and we can safely re-send.
            if let Err(ref e) = result
                && !e.is_cancelled()
                && retry_left > 0
            {
                retry_left -= 1;
                tracing::warn!(
                    session_id = session_id.id,
                    turn = turn_count,
                    error = %e,
                    retries_left = retry_left,
                    delay_ms = retry_delay_ms,
                    "SSE stream interrupted, retrying..."
                );
                tokio::time::sleep(std::time::Duration::from_millis(retry_delay_ms)).await;
                retry_delay_ms = (retry_delay_ms as f64 * STREAM_RETRY_BACKOFF) as u64;
                retry_delay_ms = retry_delay_ms.min(STREAM_RETRY_MAX_MS);
                continue;
            }
            return result;
        }
    }
}
