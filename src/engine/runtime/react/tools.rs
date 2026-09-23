use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;

use serde_json::Value;

use crate::engine::recovery::ToolErrorAction;
use crate::engine::runtime::plan_runner::RuntimeCore;
use crate::engine::runtime::tool_engine::ExecutionContext;
use crate::tool::content_text;
use crate::types::{
    AgentError, AgentResult, CheckpointData, CheckpointStep, FinishReason, MessageRole, RunOutcome,
    RuntimeEvent, SessionId,
};

use super::entry::drain_locked;
use super::turn_end::TurnEndCtx;
use super::turn_guard::TurnMetrics;
use super::turn_loop::TurnFlow;

impl RuntimeCore {
    /// Execute tool calls for one LLM turn: truncation guard, execution,
    /// checkpoint emission, and turn-end metrics.
    #[allow(clippy::too_many_arguments)]
    pub(super) async fn run_tool_turn<F>(
        &self,
        session_id: &SessionId,
        user_input: &str,
        turn_count: u32,
        turn_start: std::time::Instant,
        model: &str,
        mut tool_calls: Vec<(String, String, String)>,
        finish_reason: &FinishReason,
        reasoning_text: String,
        full_text: String,
        metrics: &TurnMetrics<'_>,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<TurnFlow>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        // Truncation guard — a well-formed provider stream always yields
        // complete JSON for each tool call's arguments. When that breaks, the
        // call must NOT be executed: re-issue it instead. Three ways it breaks:
        //   1. finish_reason=length — the model hit the output token limit
        //      mid-arguments (provider reports this honestly).
        //   2. finish_reason=tool_calls (or anything else) but a *named* call's
        //      arguments don't parse as JSON — the provider truncated the
        //      argument stream and then *mislabelled* the finish as a normal
        //      tool_calls stop. mimo-v2.5-pro does exactly this under load
        //      (emits a dangling `{"agent_path": ` with finish_reason=tool_calls
        //      and never sets length). Checking the structural symptom makes
        //      the guard independent of a finish_reason we can't trust.
        //   3. arguments parse as JSON but are a `{error:"tool_call_arguments_truncated",...}`
        //      wrapper — the model echoed back truncation residue it saw earlier.
        //      Valid JSON, so case 2's check misses it; recognise it by shape.
        //   4. arguments parse as a valid *empty object* `{}` for a tool that
        //      declares required parameters. This is not provider truncation — it
        //      is the model *mimicking* our own sanitizer: truncated spawn_agent
        //      args get stored as `{}` (see session.rs, to avoid a 400 on the next
        //      request), the model sees a `spawn_agent {}` in its history and emits
        //      a stray empty twin next to a real call. It parses fine, so case 2
        //      skips it, and it fails typed schema at execution
        //      ("argument parsing failed: {}"). Detect it via the tool schema.
        // Left to fall through, cases (2)/(3)/(4) become a ToolArgsInvalid
        // execution failure that feeds ConsecutiveFailureRecovery and aborts the
        // run ("failed 3 consecutive times"). Routing them here degrades to a
        // benign retry that never increments the failure counter.
        // ── Per-call validity check ──
        // Check each tool_call individually so we can execute the valid ones
        // and only re-issue the broken ones (instead of discarding the whole batch).
        let truncated_by_limit = finish_reason.is_truncated();
        let mut invalid_indices: Vec<usize> = Vec::new();
        for (i, (_id, name, args)) in tool_calls.iter().enumerate() {
            if name.is_empty() {
                continue;
            }
            // Empty args for a named tool = truncation (provider cut off
            // before generating any arguments). Treat as invalid so the
            // truncation guard / circuit breaker handles it.
            if args.trim().is_empty() {
                invalid_indices.push(i);
                continue;
            }
            let dominated = serde_json::from_str::<Value>(args)
                .map(|v| is_truncation_wrapper(&v))
                .unwrap_or(true); // parse failure = truncated
            if dominated {
                invalid_indices.push(i);
                continue;
            }
            // Case 4: empty `{}` for a tool with required fields
            let empty_object = serde_json::from_str::<Value>(args)
                .map(|v| v.as_object().is_some_and(|o| o.is_empty()))
                .unwrap_or(false);
            if empty_object && self.tool_engine.tool_requires_params(name).await {
                invalid_indices.push(i);
                continue;
            }
            // Valid — no issue found
        }
        // finish_reason=length means the whole stream was cut — mark all as invalid
        if truncated_by_limit {
            invalid_indices = (0..tool_calls.len()).collect();
        }

        // ── Markup salvage：从文本通道泄漏的模板 markup 中救回参数 ──
        // mimo 家族 provider 偶发把同一个调用写在两个通道：native tool_call
        // 流的 arguments 中途截断（finish 谎报 Stop），而 content 文本里泄漏
        // 出聊天模板 markup `<tool_call><function=…><parameter=…>…`（实测
        // session 20260923_263149bf：native 死在 15 字节 `{"fork_turns":"`，
        // 泄漏文本却带着完整 task 串）。泄漏副本是模型的真实意图，优先于
        // 烧掉一次 re-issue strike（重发大概率再截断）。finish_reason=length
        // 时跳过：那里生成整体被预算砍断，markup 与 args 同残，救回无意义。
        // schema 校验（required 齐全 + 类型转换）不过关的仍走 re-issue，
        // 避免把死螺旋入口从"截断"换成"缺参执行失败"。
        if !truncated_by_limit && !invalid_indices.is_empty() {
            let leaked = super::text_toolcall::harvest(&full_text);
            if !leaked.is_empty() {
                let mut salvaged: Vec<usize> = Vec::new();
                for &i in &invalid_indices {
                    let name = tool_calls[i].1.clone();
                    let Some(candidate) = leaked.iter().find(|c| c.name == name) else {
                        continue;
                    };
                    let schema = self.tool_engine.tool_schema(&name).await;
                    let Some(args_json) =
                        super::text_toolcall::build_arguments(candidate, schema.as_ref())
                    else {
                        tracing::debug!(
                            session_id = session_id.id,
                            turn = turn_count,
                            tool = %name,
                            "leaked markup found but arguments unusable (missing required \
                             fields) — keeping re-issue path"
                        );
                        continue;
                    };
                    if serde_json::from_str::<Value>(&args_json)
                        .map(|v| is_truncation_wrapper(&v))
                        .unwrap_or(true)
                    {
                        continue;
                    }
                    tool_calls[i].2 = args_json;
                    salvaged.push(i);
                    tracing::info!(
                        session_id = session_id.id,
                        turn = turn_count,
                        tool = %name,
                        recovered_args_len = tool_calls[i].2.len(),
                        "salvaged truncated tool call from leaked text-channel markup — \
                         executing recovered arguments instead of re-issuing"
                    );
                }
                if !salvaged.is_empty() {
                    invalid_indices.retain(|i| !salvaged.contains(i));
                }
            }
        }

        if !invalid_indices.is_empty() {
            let has_valid = invalid_indices.len() < tool_calls.len();
            let is_truncated_call = |args: &str| -> bool {
                serde_json::from_str::<Value>(args)
                    .map(|v| is_truncation_wrapper(&v))
                    .unwrap_or(true)
            };
            let has_truncated = invalid_indices
                .iter()
                .any(|&i| is_truncated_call(&tool_calls[i].2));

            tracing::warn!(
                session_id = session_id.id,
                turn = turn_count,
                tool_count = tool_calls.len(),
                invalid_count = invalid_indices.len(),
                finish_reason = ?finish_reason,
                has_valid_calls = has_valid,
                "some tool-call arguments incomplete — re-issuing invalid, executing valid"
            );

            // ── Circuit breaker (only when ALL calls are invalid) ──
            const TRUNCATION_STRIKE_LIMIT: usize = 3;
            const TRUNCATION_HARD_LIMIT: usize = 5;
            let strikes_after = if has_valid {
                0 // valid calls will reset the counter
            } else {
                let current = self
                    .with_session_mut(session_id, |s| s.run_state.truncation_strikes)
                    .await?;
                current + 1
            };

            if !has_valid && strikes_after >= TRUNCATION_HARD_LIMIT {
                // Model kept calling the same tool even after redirect — hard stop.
                let failed_tools: Vec<String> = tool_calls
                    .iter()
                    .map(|(_, name, _)| name.clone())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                let error_msg = format!(
                    "Tool [{}] repeatedly truncated ({} attempts). Model did not \
                     follow redirect instruction. Stopping run.",
                    failed_tools.join(", "),
                    strikes_after,
                );
                tracing::error!(
                    session_id = session_id.id,
                    turn = turn_count,
                    strikes = strikes_after,
                    tools = ?failed_tools,
                    "truncation hard limit reached, stopping run"
                );
                // Record the final strike and close dangling tool calls.
                self.with_session_mut(session_id, |session| {
                    session.push_assistant_tool_calls(
                        &tool_calls,
                        Some(reasoning_text),
                        Some(full_text),
                    );
                    for &i in &invalid_indices {
                        session.push_tool_result(&tool_calls[i].0, &error_msg);
                    }
                    session.run_state.record_truncation();
                })
                .await?;
                return Ok(TurnFlow::Done(RunOutcome::Failed { error: error_msg }));
            }

            if !has_valid && strikes_after >= TRUNCATION_STRIKE_LIMIT {
                let failed_tools: Vec<String> = tool_calls
                    .iter()
                    .map(|(_, name, _)| name.clone())
                    .collect::<std::collections::HashSet<_>>()
                    .into_iter()
                    .collect();
                tracing::warn!(
                    session_id = session_id.id,
                    turn = turn_count,
                    strikes = strikes_after,
                    tools = ?failed_tools,
                    "truncation circuit breaker tripped — redirecting model away from truncated tool"
                );
                let redirect_msg = format!(
                    "CRITICAL: The tool [{}] has had its arguments truncated {} \
                     consecutive times. Its parameter schema is too complex for \
                     this model to generate. You MUST NOT call [{}] again in this \
                     session. Instead, perform the task directly using other \
                     available tools (bash, read_file, write_file, etc.). \
                     Do not attempt [{}] again.",
                    failed_tools.join(", "),
                    strikes_after,
                    failed_tools.join(", "),
                    failed_tools.join(", "),
                );
                self.with_session_mut(session_id, |session| {
                    session.push_assistant_tool_calls(
                        &tool_calls,
                        Some(reasoning_text),
                        Some(full_text),
                    );
                    for &i in &invalid_indices {
                        session.push_tool_result(&tool_calls[i].0, &redirect_msg);
                    }
                    session.run_state.record_truncation();
                })
                .await?;
                return Ok(TurnFlow::Continue);
            }

            // ── Choose guidance text ──
            let guidance: String = if strikes_after >= 2 {
                // Stronger guidance on second+ attempt
                if has_truncated || truncated_by_limit {
                    format!(
                        "Tool call was not executed (attempt {}): arguments are still \
                         truncated. STOP trying this tool — its argument schema is too \
                         complex for your output budget. Use a simpler alternative tool \
                         or explain to the user what you were trying to do.",
                        strikes_after,
                    )
                } else {
                    format!(
                        "Tool call was not executed (attempt {}): empty {{}} arguments. \
                         STOP trying this tool. Use a simpler alternative or explain \
                         the failure to the user.",
                        strikes_after,
                    )
                }
            } else if truncated_by_limit {
                "Tool call was not executed: the response hit the output token limit, \
                 so its arguments may be truncated. Re-issue the tool call with complete arguments."
                    .to_string()
            } else if has_truncated {
                "Tool call was not executed: the provider truncated the argument stream \
                 mid-generation, so its arguments are incomplete. Re-issue the tool call — \
                 emit ONE tool call per turn, or shorten long string fields (e.g. message, \
                 system_prompt) so the full arguments fit in a single response."
                    .to_string()
            } else {
                "Tool call was not executed: it carried an empty argument object `{}`, but \
                 this tool requires fields. Re-issue it with every required field filled in \
                 — never send a bare `{}` placeholder. If you need several calls, emit each \
                 one with complete arguments."
                    .to_string()
            };

            // Push the assistant message WITH all tool_calls (protocol requirement),
            // and tool_results with guidance for the invalid ones.
            self.with_session_mut(session_id, |session| {
                session.push_assistant_tool_calls(
                    &tool_calls,
                    Some(reasoning_text),
                    Some(full_text),
                );
                for &i in &invalid_indices {
                    session.push_tool_result(&tool_calls[i].0, &guidance);
                }
                // Record the strike if all calls were invalid.
                if !has_valid {
                    session.run_state.record_truncation();
                }
            })
            .await?;

            if has_valid {
                // Execute only the valid tool_calls through the normal path.
                let valid_calls: Vec<(String, String, String)> = tool_calls
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !invalid_indices.contains(i))
                    .map(|(_, tc)| tc.clone())
                    .collect();
                self.handle_tool_calls(session_id, &valid_calls, event_rx, on_event, "", "")
                    .await?;
                // Valid calls executed — reset truncation counter.
                self.with_session_mut(session_id, |session| {
                    session.run_state.truncation_strikes = 0;
                })
                .await?;
            }
            return Ok(TurnFlow::Continue);
        }

        tracing::info!(
            session_id = session_id.id,
            turn = turn_count,
            tool_count = tool_calls.len(),
            "handling tool calls"
        );
        self.event_bus.emit(RuntimeEvent::Checkpoint {
            session_id: session_id.clone(),
            checkpoint: CheckpointData {
                session_id: session_id.clone(),
                user_input: user_input.to_string(),
                step: CheckpointStep::BeforeToolCalls {
                    tool_calls: tool_calls.clone(),
                },
                turn_count,
            },
            agent_id: None,
            trace_id: None,
        });

        let tool_start = std::time::Instant::now();
        let tool_call_count = tool_calls.len() as u32;
        let tool_names: Vec<String> = tool_calls.iter().map(|(_, name, _)| name.clone()).collect();

        match self
            .handle_tool_calls(
                session_id,
                &tool_calls,
                event_rx,
                on_event.clone(),
                &reasoning_text,
                &full_text,
            )
            .await
        {
            Ok(()) => {
                let tool_duration_ms = tool_start.elapsed().as_millis() as u64;
                self.fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    tool_duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    tool_call_count,
                    tools_used: &tool_names,
                    tool_success: tool_call_count,
                    llm_calls: 1,
                    ..TurnEndCtx::new(
                        session_id,
                        turn_count,
                        turn_start,
                        model,
                        user_input,
                        RunOutcome::Completed,
                    )
                })
                .await;
                tracing::info!(
                    session_id = session_id.id,
                    turn = turn_count,
                    "tool calls done, continuing loop"
                );
                let n = tool_calls.len();
                self.with_session_mut(session_id, |session| {
                    session.total_tool_calls += n;
                    session.run_state.record_tool_calls(n);
                })
                .await?;
                self.event_bus.emit(RuntimeEvent::Checkpoint {
                    session_id: session_id.clone(),
                    checkpoint: CheckpointData {
                        session_id: session_id.clone(),
                        user_input: user_input.to_string(),
                        step: CheckpointStep::AfterToolCalls {
                            tool_calls,
                            results: Vec::new(),
                        },
                        turn_count,
                    },
                    agent_id: None,
                    trace_id: None,
                });
                Ok(TurnFlow::Continue)
            }
            Err(e) => {
                let tool_duration_ms = tool_start.elapsed().as_millis() as u64;
                let error_msg = e.to_string();
                if let Some(outcome) = self
                    .handle_tool_error(session_id, &tool_calls, e, event_rx, on_event)
                    .await?
                {
                    self.fire_turn_end(TurnEndCtx {
                        ttft_ms: metrics.ttft_ms,
                        llm_duration_ms: metrics.llm_duration_ms,
                        tool_duration_ms,
                        usage: metrics.usage,
                        text_length: metrics.text_len,
                        has_thinking: metrics.has_thinking,
                        tool_call_count,
                        tools_used: &tool_names,
                        tool_failed: tool_call_count,
                        error_message: Some(&error_msg),
                        llm_calls: 1,
                        ..TurnEndCtx::new(
                            session_id,
                            turn_count,
                            turn_start,
                            model,
                            user_input,
                            RunOutcome::Failed {
                                error: error_msg.clone(),
                            },
                        )
                    })
                    .await;
                    return Ok(TurnFlow::Done(outcome));
                }
                // Retry: record metrics for the failed attempt
                self.fire_turn_end(TurnEndCtx {
                    ttft_ms: metrics.ttft_ms,
                    llm_duration_ms: metrics.llm_duration_ms,
                    tool_duration_ms,
                    usage: metrics.usage,
                    text_length: metrics.text_len,
                    has_thinking: metrics.has_thinking,
                    tool_call_count,
                    tools_used: &tool_names,
                    tool_failed: tool_call_count,
                    error_message: Some(&error_msg),
                    llm_calls: 1,
                    ..TurnEndCtx::new(
                        session_id,
                        turn_count,
                        turn_start,
                        model,
                        user_input,
                        RunOutcome::Failed {
                            error: error_msg.clone(),
                        },
                    )
                })
                .await;
                Ok(TurnFlow::Continue)
            }
        }
    }

    pub(super) async fn handle_tool_error<F>(
        &self,
        session_id: &SessionId,
        tool_calls: &[(String, String, String)],
        e: AgentError,
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
    ) -> AgentResult<Option<RunOutcome>>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send,
    {
        let config = self.config_snapshot_async().await;

        if e.is_cancelled() {
            // Don't emit RunFinished when cancelled — RunCancelled will be emitted by run_turn
            // But still drain pending events and persist session
            drain_locked(event_rx, &on_event)?;
            let session = self.session_manager.session_or_err(session_id).await?;
            if let Err(e) = self.session_manager.session_store().save(&session).await {
                tracing::warn!(session_id = session_id.id, error = %e, "Failed to persist session");
                if config.execution.fail_on_persist_error {
                    return Err(AgentError::internal(format!(
                        "Session persistence failed: {e}"
                    )));
                }
            }
            return Err(e);
        }

        let names = failing_tool_names(tool_calls, &e);
        let error_text = e.to_string();
        let retry_prompt_template: Option<String> = config.tool.tool_error_retry_prompt.clone();

        // 用户主动拒绝（输入 n）→ 立即停止，不走 recovery 重试逻辑
        if matches!(e, AgentError::ApprovalDenied { .. }) {
            tracing::info!(
                session_id = session_id.id,
                "user rejected tool call, stopping immediately"
            );
            let error_summary = if config.language == crate::types::Language::Zh {
                format!("❌ 用户拒绝执行: {}", e)
            } else {
                format!("❌ User rejected: {}", e)
            };
            self.with_session_mut(session_id, |session| {
                session.close_dangling_tool_calls(&error_summary);
                session.remove_ephemeral_messages();
            })
            .await?;
            self.event_bus.emit(RuntimeEvent::RunFinished {
                session_id: session_id.clone(),
                agent_id: None,
                trace_id: None,
            });
            drain_locked(event_rx, &on_event)?;
            let session = self.session_manager.session_or_err(session_id).await?;
            if let Err(e) = self.session_manager.session_store().save(&session).await {
                tracing::warn!(session_id = session_id.id, error = %e, "Failed to persist session");
            }
            return Ok(Some(RunOutcome::Completed));
        }

        let action = self
            .tool_engine
            .error_recovery()
            .on_error(session_id, &names, &e)?;
        match action {
            ToolErrorAction::Stop => {
                // Close any dangling tool calls in the session before stopping
                let error_summary = if config.language == crate::types::Language::Zh {
                    format!("❌ 执行失败: {}", e)
                } else {
                    format!("❌ Tool execution failed: {}", e)
                };
                // Emit ToolCallFinished so TUI shows the failure before RunFinished
                self.event_bus.emit(RuntimeEvent::ToolCallFinished {
                    session_id: session_id.clone(),
                    tool_name: names.join(", "),
                    summary: error_summary.clone(),
                    agent_id: None,
                    trace_id: None,
                    denied: false,
                    details: None,
                });
                self.with_session_mut(session_id, |session| {
                    session.close_dangling_tool_calls(&error_summary);
                    session.remove_ephemeral_messages();
                })
                .await?;
                self.event_bus.emit(RuntimeEvent::RunFinished {
                    session_id: session_id.clone(),
                    agent_id: None,
                    trace_id: None,
                });
                drain_locked(event_rx, &on_event)?;
                let session = self.session_manager.session_or_err(session_id).await?;
                if let Err(e) = self.session_manager.session_store().save(&session).await {
                    tracing::warn!(session_id = session_id.id, error = %e, "Failed to persist session");
                    if config.execution.fail_on_persist_error {
                        return Err(AgentError::internal(format!(
                            "Session persistence failed: {e}"
                        )));
                    }
                }
                Ok(Some(RunOutcome::Failed {
                    error: format!("Tool execution failed: {}", e),
                }))
            }
            ToolErrorAction::RetryWithHistory { errors } => {
                let tool_name = names.join(", ");
                let error_history = format!(
                    "The tool `{}` has failed {} consecutive times. \
                     Error history:\n\n{}\n\n\
                     Please analyze these errors and either switch to a different \
                     approach or explain the failure to the user.",
                    tool_name,
                    errors.len(),
                    errors
                        .iter()
                        .enumerate()
                        .map(|(i, e)| format!("{}. {}", i + 1, e))
                        .collect::<Vec<_>>()
                        .join("\n\n"),
                );

                let summary = if config.language == crate::types::Language::Zh {
                    format!(
                        "⚠️ {} 连续失败 {} 次，已将错误历史发给 LLM 评估",
                        tool_name,
                        errors.len()
                    )
                } else {
                    format!(
                        "⚠️ {} failed {} consecutive times, error history sent to LLM",
                        tool_name,
                        errors.len()
                    )
                };

                // Emit ToolCallFinished so TUI shows the recovery action
                self.event_bus.emit(RuntimeEvent::ToolCallFinished {
                    session_id: session_id.clone(),
                    tool_name: tool_name.clone(),
                    summary: summary.clone(),
                    agent_id: None,
                    trace_id: None,
                    denied: false,
                    details: None,
                });

                self.with_session_mut(session_id, |session| {
                    session.close_dangling_tool_calls(&summary);
                    session.push_message(MessageRole::User, error_history);
                })
                .await?;
                Ok(None)
            }
            ToolErrorAction::Retry => {
                let retry_prompt = match &retry_prompt_template {
                    Some(template) => template
                        .replace("{tool_names}", &names.join(", "))
                        .replace("{error}", &error_text),
                    None => format!(
                        "Tool calls failed: {}\nError: {}\nPlease analyze the error and adjust your approach.",
                        names.join(", "),
                        error_text,
                    ),
                };

                let error_summary = if config.language == crate::types::Language::Zh {
                    format!("❌ 执行失败: {}", error_text)
                } else {
                    format!("❌ Tool execution failed: {}", error_text)
                };
                self.with_session_mut(session_id, |session| {
                    session.close_dangling_tool_calls(&error_summary);
                    session.push_message(MessageRole::User, retry_prompt);
                })
                .await?;
                Ok(None)
            }
        }
    }

    pub(super) async fn handle_tool_calls<F>(
        &self,
        session_id: &SessionId,
        tool_calls: &[(String, String, String)],
        event_rx: &mut broadcast::Receiver<RuntimeEvent>,
        on_event: Arc<Mutex<F>>,
        reasoning: &str,
        full_text: &str,
    ) -> AgentResult<()>
    where
        F: FnMut(RuntimeEvent) -> AgentResult<()> + Send + 'static,
    {
        let tool_names: Vec<&str> = tool_calls
            .iter()
            .map(|(_, name, _)| name.as_str())
            .collect();
        tracing::debug!(
            session_id = session_id.id,
            ?tool_names,
            "handle tool calls start"
        );

        let config = self.config_snapshot_async().await;

        let ctx = ExecutionContext {
            session_manager: self.session_manager.clone(),
            llm_client: Some(self.llm_engine.get_provider()),
            language: config.language.clone(),
            tool_timeout_ms: config.tool.tool_timeout_ms,
            max_output_chars: config.tool.max_tool_output_chars,
            cancel_token: self.cancel_token(),
        };

        // Orchestrate: approval check + execution for all tool calls.
        // Approval denial is handled inside process_approval (pushes fake tool state
        // and returns Err). We only push real tool calls on success.
        let outcome = self
            .tool_engine
            .orchestrate(session_id, tool_calls, &ctx, event_rx, on_event.clone())
            .await?;

        // Push assistant tool calls to session after all approvals pass
        {
            let tc: Vec<(String, String, String)> = tool_calls.to_vec();
            let ft = if full_text.is_empty() {
                None
            } else {
                Some(full_text.to_string())
            };
            let rt = if reasoning.is_empty() {
                None
            } else {
                Some(reasoning.to_string())
            };
            self.with_session_mut(session_id, |session| {
                session.push_assistant_tool_calls(&tc, rt, ft);
            })
            .await?;
        }

        // Process successful results: push to session.
        for result in &outcome.results {
            self.with_session_mut(session_id, |session| {
                session.push_tool_result(&result.id, content_text(&result.output));
            })
            .await?;
        }

        // Push each failure's own error text so the model sees exactly what broke
        // (not just the first failure's message) and every tool call has a result.
        for failure in &outcome.failures {
            let summary = failure.error.to_string();
            self.with_session_mut(session_id, |session| {
                session.push_tool_result(&failure.id, summary);
            })
            .await?;
        }

        // Surface the first per-call failure back to the caller so the existing
        // `handle_tool_error` path can run recovery and feed the model a retry prompt.
        // Recovery only counts this first failing tool here — an undercount in the
        // rare multi-failure batch, which errs on the lenient (never false-Stop) side.
        if let Some(first) = outcome.failures.into_iter().next() {
            return Err(first.error);
        }

        Ok(())
    }
}

/// Names of the tools that actually failed in a batch of tool calls.
///
/// Tool-level errors carry the failing tool's name on the error itself, so a single
/// bad call in a large batch yields exactly one name (not every tool in the batch,
/// which would inflate the consecutive-failure counter and trip the Stop threshold).
/// Unknown error types conservatively fall back to the full batch.
fn failing_tool_names(tool_calls: &[(String, String, String)], error: &AgentError) -> Vec<String> {
    match error {
        AgentError::ToolArgsInvalid { name, .. } => vec![name.clone()],
        AgentError::ToolExecution { name, .. } => vec![name.clone()],
        _ => tool_calls.iter().map(|(_, n, _)| n.clone()).collect(),
    }
}

/// True when a tool call's (valid) JSON arguments are actually a truncation
/// error wrapper the model echoed back from history, rather than a real call.
/// Such an object parses fine, so the structural (invalid-JSON) guard can't see
/// it, but executing it only reproduces a ToolArgsInvalid failure. Detected by
/// its distinctive marker key (plus the preview field it always carried).
fn is_truncation_wrapper(args: &Value) -> bool {
    let obj = match args.as_object() {
        Some(o) => o,
        None => return false,
    };
    let marker = obj
        .get("error")
        .and_then(Value::as_str)
        .is_some_and(|s| s == "tool_call_arguments_truncated");
    let preview = obj.contains_key("original_args_preview");
    marker || (preview && obj.contains_key("message"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failing_tool_names_isolates_single_failing_tool() {
        let tool_calls = vec![
            (
                "id1".to_string(),
                "write_file".to_string(),
                "{}".to_string(),
            ),
            (
                "id2".to_string(),
                "write_file".to_string(),
                "{}".to_string(),
            ),
            (
                "id3".to_string(),
                "write_file".to_string(),
                "{}".to_string(),
            ),
        ];

        // One bad call in a batch of three must yield one name, not three.
        let err = AgentError::ToolArgsInvalid {
            name: "write_file".to_string(),
            raw: "truncated-json".to_string(),
        };
        assert_eq!(
            failing_tool_names(&tool_calls, &err),
            vec!["write_file".to_string()]
        );
    }

    #[test]
    fn failing_tool_names_extracts_execution_failure_name() {
        let tool_calls = vec![
            ("id1".to_string(), "bash".to_string(), "{}".to_string()),
            ("id2".to_string(), "bash".to_string(), "{}".to_string()),
        ];
        let err = AgentError::ToolExecution {
            name: "bash".to_string(),
            source: Box::new(AgentError::internal("boom")),
        };
        assert_eq!(
            failing_tool_names(&tool_calls, &err),
            vec!["bash".to_string()]
        );
    }

    #[test]
    fn failing_tool_names_falls_back_to_full_batch_for_unknown_error() {
        let tool_calls = vec![
            ("id1".to_string(), "a".to_string(), "{}".to_string()),
            ("id2".to_string(), "b".to_string(), "{}".to_string()),
        ];
        let err = AgentError::internal("unexpected");
        assert_eq!(
            failing_tool_names(&tool_calls, &err),
            vec!["a".to_string(), "b".to_string()]
        );
    }
}
