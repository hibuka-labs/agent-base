use super::*;

#[tokio::test]
async fn truncation_guard_blocks_tool_calls_on_length_finish_reason() {
    // First call: tool call with finish_reason="length" — should be blocked by guard.
    // Second call: model retries with corrected approach (text response).
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: truncated tool call
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_trunc",
                        "function": {
                            "name": "shell",
                            "arguments": "{\"cmd\": \"rm -rf /inco"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("length".to_string()),
            },
        ],
        // Turn 2: model sees the error and retries
        vec![
            StreamChunk::Text(
                "I see the previous call was truncated. Let me re-issue it.".to_string(),
            ),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "run a command", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    // Verify the session messages contain the truncation error, not the
    // partial argument that would have been executed.
    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    let has_truncation_error = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed") && content.contains("output token limit")
        } else {
            false
        }
    });
    assert!(
        has_truncation_error,
        "session should contain truncation error tool result. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn truncation_guard_blocks_tool_calls_on_finish_tool_calls_mimo() {
    // mimo-v2.5-pro truncates the tool-call argument stream mid-generation
    // under load but reports `finish_reason=tool_calls` (NOT `length`), so the
    // length-only guard lets these fall through to tool execution as
    // ToolArgsInvalid — which feeds ConsecutiveFailureRecovery and aborts the
    // run ("failed 3 consecutive times"). The widened guard must detect the
    // structurally-incomplete JSON args regardless of the claimed finish reason
    // and route through the clean re-issue path instead.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: named tool call with a dangling, non-JSON argument string,
        // provider claims a normal tool_calls finish.
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_trunc_1",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": "{\"agent_path\": "
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: model sees the re-issue instruction and replies in text.
        vec![
            StreamChunk::Text("Re-issuing the call with complete arguments.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;

    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn a sub-agent", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    // The truncated call must NOT have been executed / recorded as a parse
    // failure; it must have been routed to the re-issue guard.
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
                && content.contains("provider truncated the argument stream")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "session should contain a re-issue tool result for the truncated call. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // And crucially the raw malformed args must never appear as an executed
    // ToolArgsInvalid failure (the death-spiral trigger).
    let has_args_invalid = messages.iter().any(|m| {
        let s = format!("{:?}", m);
        s.contains("argument parsing failed") || s.contains("EOF while parsing")
    });
    assert!(
        !has_args_invalid,
        "truncated args must not surface as a ToolArgsInvalid failure. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn truncation_guard_salvages_call_from_leaked_text_markup() {
    // mimo-v2.6-flash (session 20260923_263149bf) emits the SAME tool call on
    // two channels: the native tool_call argument stream dies mid-JSON with a
    // bogus "stop" finish, while the content channel leaks the chat-template
    // markup carrying the full parameter values. The guard must harvest the
    // leaked copy (name match + schema-valid) and EXECUTE it, not burn a
    // re-issue strike — the re-issue truncated again on the real session.
    let leaked = "我来派一个子代理。<tool_call><function=spawn_agent>\
<parameter=task_name>research<parameter=message>只读调研 codex，不改文件。";
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::Text(leaked.to_string()),
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_leak_1",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": "{\"task_name\":\"res"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("Should never be reached.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn a sub-agent", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    // The salvaged call must have executed exactly once.
    let spawned_count = messages
        .iter()
        .filter(|m| {
            if let ChatMessage::Tool { content, .. } = m {
                content.contains("spawned")
            } else {
                false
            }
        })
        .count();
    assert_eq!(
        spawned_count,
        1,
        "leaked call must be salvaged and executed. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // No re-issue guidance, and no ToolArgsInvalid death-spiral residue.
    let bad = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
                || content.contains("argument parsing failed")
                || content.contains("EOF while parsing")
        } else {
            false
        }
    });
    assert!(
        !bad,
        "salvaged call must not surface re-issue guidance or parse failure. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn salvage_rejects_leak_missing_required_field() {
    // A leak that omits a schema-required field must NOT be salvaged: executing
    // it would fail typed parsing (ToolArgsInvalid — the death-spiral trigger).
    // The call stays on the re-issue path instead.
    let leaked = "<tool_call><function=spawn_agent><parameter=task_name>research</tool_call>";
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::Text(leaked.to_string()),
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_leak_2",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": "{\"task_name\":\"res"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("Re-issuing with complete arguments.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn a sub-agent", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    let executed = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("spawned")
        } else {
            false
        }
    });
    assert!(
        !executed,
        "incomplete leak must not execute. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "incomplete leak must fall back to re-issue guidance. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn truncation_guard_recognizes_wrapper_echo() {
    // mimo, having seen a `{error:"tool_call_arguments_truncated",...}` object
    // in its history, replays it VERBATIM as the next call's arguments. That
    // payload is valid JSON, so the raw-args structural guard can't catch it —
    // but it is unmistakably truncation residue. The guard must recognise the
    // wrapper marker and re-issue instead of executing it (which would surface
    // as a downstream ToolArgsInvalid, the send_message slip seen in the field).
    let wrapper = serde_json::json!({
        "error": "tool_call_arguments_truncated",
        "message": "The tool call arguments were truncated or invalid. Please retry with complete arguments.",
        "original_args_preview": "{\"agent_path\": "
    })
    .to_string();

    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "id": "call_echo",
                        "function": {
                            "name": "send_message",
                            "arguments": wrapper
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("Now sending a real message.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "continue", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "echoed wrapper args must be routed to the re-issue guard. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    let has_args_invalid = messages.iter().any(|m| {
        let s = format!("{:?}", m);
        s.contains("argument parsing failed")
    });
    assert!(
        !has_args_invalid,
        "echoed wrapper must not reach tool execution. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

/// Minimal tool whose schema has no required fields — mirrors `list_agents`.
struct NoArgTool;

#[async_trait]
impl Tool for NoArgTool {
    fn name(&self) -> &'static str {
        "list_agents"
    }

    fn description(&self) -> &'static str {
        "List running agents"
    }

    fn schema(&self) -> Value {
        serde_json::json!({ "type": "object", "properties": {} })
    }

    async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        Ok(vec![Content::text("no agents")])
    }
}

#[tokio::test]
async fn truncation_guard_blocks_empty_args_for_required_field_tool() {
    // The model, having seen sanitized `spawn_agent {}` entries in its history,
    // echoes the shape: one real spawn + one empty `{}` in the same turn.
    // `{}` is valid JSON (case 2 skips it) but the tool's schema requires
    // task_name + message — case 4 must detect this as degenerate and re-issue
    // the whole turn, avoiding a ToolArgsInvalid execution failure.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: two spawn_agent buckets — one complete, one empty `{}`.
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_real",
                            "function": {
                                "name": "spawn_agent",
                                "arguments": "{\"task_name\":\"foo\",\"message\":\"do stuff\"}"
                            }
                        },
                        {
                            "index": 1,
                            "id": "call_empty",
                            "function": {
                                "name": "spawn_agent",
                                "arguments": "{}"
                            }
                        }
                    ]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: model sees the re-issue instruction and replies.
        vec![
            StreamChunk::Text("Re-issuing with complete arguments.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn two agents", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    // Must contain a re-issue tool result for the empty-required case.
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
                && content.contains("empty argument object")
                && content.contains("requires fields")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "empty {{}} for a required-field tool must be caught by case 4 guard. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // Crucially the empty args must NOT have reached tool execution as a
    // ToolArgsInvalid failure.
    let has_args_invalid = messages.iter().any(|m| {
        let s = format!("{:?}", m);
        s.contains("argument parsing failed") || s.contains("EOF while parsing")
    });
    assert!(
        !has_args_invalid,
        "empty {{}} args must not surface as a ToolArgsInvalid failure. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn truncation_guard_reissues_args_missing_required_field() {
    // Session 20261121_a62ca27d: a parallel spawn_agent call arrived as valid
    // JSON with the required task text dropped ({"task_name":…,"model":…}).
    // Case 2 cannot see it (the object parses) and case 4 cannot either (not
    // empty), so without case 5 it falls through to a typed-arg execution
    // failure. It must instead be re-issued with the missing field named.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_missing_task",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": "{\"task_name\":\"mod-agent\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: model sees the re-issue instruction and replies.
        vec![
            StreamChunk::Text("Re-issuing with the task filled in.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("You are a careful assistant.")
    .register_tool(SpawnLikeTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn an agent", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
                && content.contains("omitted required field(s)")
                && content.contains("message")
        } else {
            false
        }
    });
    assert!(
        has_reissue,
        "args missing a required field must be caught by case 5 guard. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    let has_args_invalid = messages.iter().any(|m| {
        let s = format!("{:?}", m);
        s.contains("argument parsing failed") || s.contains("EOF while parsing")
    });
    assert!(
        !has_args_invalid,
        "missing-field args must not surface as a ToolArgsInvalid failure. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn no_arg_tool_with_empty_object_args_executes_normally() {
    // A tool with no required fields (like `list_agents`) legitimately receives
    // `{}` as arguments. The case 4 guard must NOT fire — the tool should
    // execute normally.
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        // Turn 1: list_agents with `{}` — should execute.
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_list",
                        "function": {
                            "name": "list_agents",
                            "arguments": "{}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        // Turn 2: model reports the result.
        vec![
            StreamChunk::Text("No agents running.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(NoArgTool)
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "list agents", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();

    // Must NOT contain a re-issue guidance — no required fields, so {} is valid.
    let has_reissue = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("Tool call was not executed")
                && content.contains("empty argument object")
        } else {
            false
        }
    });
    assert!(
        !has_reissue,
        "no-arg tool with {{}} must execute, not be re-issued. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );

    // And must contain the tool's actual output (execution succeeded).
    let has_output = messages.iter().any(|m| {
        if let ChatMessage::Tool { content, .. } = m {
            content.contains("no agents")
        } else {
            false
        }
    });
    assert!(
        has_output,
        "no-arg tool must have executed and returned output. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

// ── InvalidArgsHandler: application-defined responses ─────────────────

/// Custom handler that re-issues with product-specific guidance.
struct DocsHintHandler;

#[async_trait]
impl InvalidArgsHandler for DocsHintHandler {
    async fn on_invalid_args(&self, ctx: InvalidArgsCtx<'_>) -> AgentResult<InvalidArgsAction> {
        Ok(InvalidArgsAction::Reissue {
            message: format!(
                "custom hint: missing `{}` — read docs/agents.md before retrying",
                ctx.missing_fields.join("`, `")
            ),
        })
    }
}

/// Strict handler: invalid arguments end the run instead of retrying.
struct StopOnMissingArgs;

#[async_trait]
impl InvalidArgsHandler for StopOnMissingArgs {
    async fn on_invalid_args(&self, ctx: InvalidArgsCtx<'_>) -> AgentResult<InvalidArgsAction> {
        Ok(InvalidArgsAction::Stop {
            message: format!("policy stop: missing `{}`", ctx.missing_fields.join("`, `")),
        })
    }
}

#[tokio::test]
async fn invalid_args_handler_can_reissue_with_custom_message() {
    let runtime = AgentBuilder::new(Arc::new(ScriptedProvider::new(vec![
        vec![
            StreamChunk::ToolCall(serde_json::json!({
                "delta": {
                    "tool_calls": [{
                        "index": 0,
                        "id": "call_missing",
                        "function": {
                            "name": "spawn_agent",
                            "arguments": "{\"task_name\":\"mod-agent\"}"
                        }
                    }]
                }
            })),
            StreamChunk::Stop {
                finish_reason: Some("tool_calls".to_string()),
            },
        ],
        vec![
            StreamChunk::Text("Retrying after reading the docs.".to_string()),
            StreamChunk::Stop {
                finish_reason: Some("stop".to_string()),
            },
        ],
    ])))
    .system_prompt("test")
    .register_tool(SpawnLikeTool)
    .invalid_args_handler(Arc::new(DocsHintHandler))
    .build()
    .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn an agent", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;
    assert!(
        result.is_ok(),
        "run_turn should complete: {:?}",
        result.err()
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();
    let has_custom = messages.iter().any(|m| {
        matches!(m, ChatMessage::Tool { content, .. } if content.contains("custom hint: missing `message`"))
    });
    assert!(
        has_custom,
        "the injected handler's message must reach the model. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn invalid_args_handler_can_stop_the_run() {
    let provider = ScriptedProvider::new(vec![vec![
        StreamChunk::ToolCall(serde_json::json!({
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_missing",
                    "function": {
                        "name": "spawn_agent",
                        "arguments": "{\"task_name\":\"mod-agent\"}"
                    }
                }]
            }
        })),
        StreamChunk::Stop {
            finish_reason: Some("tool_calls".to_string()),
        },
    ]]);
    let runtime = AgentBuilder::new(Arc::new(provider))
        .system_prompt("test")
        .register_tool(SpawnLikeTool)
        .invalid_args_handler(Arc::new(StopOnMissingArgs))
        .build()
        .expect("build runtime");

    let sid = runtime.create_session().await;
    let events = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let result = runtime
        .run_turn(sid.clone(), "spawn an agent", move |event| {
            events_clone.lock().unwrap().push(event);
            Ok(())
        })
        .await;

    assert!(
        matches!(result, Ok(RunOutcome::Failed { .. })),
        "a Stop handler must end the run as Failed, got {:?}",
        result
    );

    let session = runtime.session(&sid).await.expect("session exists");
    let messages = session.chat_messages().to_vec();
    let has_policy = messages.iter().any(|m| {
        matches!(m, ChatMessage::Tool { content, .. } if content.contains("policy stop: missing `message`"))
    });
    assert!(
        has_policy,
        "the stop message must still be recorded as the tool result. Messages: {:#?}",
        messages
            .iter()
            .map(|m| format!("{:?}", m))
            .collect::<Vec<_>>()
    );
}
