# Changelog

All notable changes to `agent-base` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-09-27

### Added
- **Truncated tool-call salvage** (`react/text_toolcall.rs`): recovers tool calls
  leaked into the text channel as template markup (parameter-style, closing tag
  optional, and Hermes JSON blocks) when native tool-call args are truncated
  mid-stream. `build_arguments()` validates schema-required fields and coerces
  scalar types; the truncation guard salvages per-call by name before strike
  accounting — recovered args execute instead of burning a re-issue strike.
  Skipped under `finish_reason=length`; schema-invalid leaks keep the re-issue
  path. `tool_engine::tool_schema()` read accessor; truncation WARN logs now
  carry an `args_preview` (first 80 chars, debug-escaped).
- **`ApprovalRequest.source`** (`agent-types` 0.2.0): optional requesting-agent
  path (e.g. `root/coder-1`); legacy callers leave it `None` and old JSON
  deserializes unchanged (`#[serde(default)]`). `SourcedApprovalHandler` fills it
  from `ctx.agent_path`.
- **Per-session approval cache isolation**: the allow-always cache is scoped per
  session — a child's "always" no longer leaks to the parent or siblings.

### Changed
- Bump `agent-types` to 0.2.0 (adds `ApprovalRequest.source`; struct-literal
  construction sites must add the field).

## [0.7.0] - 2026-09-18

### Added
- **`RepeatToolLimitMiddleware`** + `RepeatToolLimitConfig`: breaks repeated
  identical tool-call loops (e.g. polling) with a configurable nudge→block
  escalation (default nudge at 5, block at 10).
- **Ephemeral-input turns** + session prompt surgery for skill injection:
  `ChatMessage::User` gains `ephemeral: bool` for one-shot context that is
  stripped after the LLM turn; `AgentRuntime::inject_prompt()` splices content
  into the session at Continue points.

### Fixed
- Scope `MutexGuard` in tests to avoid holding across await points (clippy).
- Exclude `fuzz/` from published crate.

## [0.6.0] - 2026-09-11

### Added
- **Cancellable LLM streaming + TTFB timeout**: `stream_cancellable` selects over
  cancel token and a 60 s TTFB timeout; silently-hung connections surface as
  retryable `AgentError::Llm` instead of hanging forever.
- **`model_override`** on `LlmEngine` and `AgentRuntime` (`set_model_override()`
  / `get_model_override()`): sub-agents can specify their model tier (e.g.
  `"lite"`); the override is applied to every `ChatRequest` before dispatch.
- **`ContextCompaction` trait** + `ContextWindowManager`: trait refinement with
  `plan_runner` integration and a compact hook at Continue points in `turn_loop`.
  Session checkpoint support in `engine::session`.

### Changed
- `turn_loop` and `tool_engine` refactored for cleaner control flow and
  orchestration outcome handling.
- Clippy: removed redundant closure in `turn_loop.rs`.

## [0.6.0] - 2026-09-11

### Added
- **Cancellable LLM streaming + TTFB timeout**: `stream_cancellable` selects over
  cancel token and a 60 s TTFB timeout; silently-hung connections surface as
  retryable `AgentError::Llm` instead of hanging forever. Cancel token is
  threaded through `chat_stream` and `run_llm_turn_with_retry`.
- **`model_override`** on `LlmEngine` and `AgentRuntime`: `set_model_override()`
  / `get_model_override()` let sub-agents specify their model tier (e.g.
  `"lite"`); the override is applied to every `ChatRequest` before dispatch.
- **`ContextCompaction` trait** + `ContextWindowManager`: trait refinement with
  `plan_runner` integration and a compact hook at Continue points in `turn_loop`.
  `tool_engine` orchestration outcome and error handling improved.
- Session checkpoint support in `engine::session`.

### Changed
- `turn_loop` and `tool_engine` refactored for cleaner control flow.
- Clippy: removed redundant closure in `turn_loop.rs`.

## [0.5.0] - 2026-09-06

### Added
- Truncation circuit breaker: when a tool-call loop hits the token-truncation
  guard repeatedly, the run is redirected with guidance instead of being killed,
  preventing the re-issue death spiral.
- Partial execution of valid tool_calls when truncation strikes: tool calls
  whose arguments fit are still executed; only the oversized ones are rejected.
- `Content::Detail` — structured tool-result metadata (e.g. multi-agent
  tool-result details) alongside the plain text payload.
- `thinking_bytes` / total thinking byte tracking on `TurnContext` for
  reasoning-token accounting.
- `GuardCtx` now exposes the rejected-tool-call state so downstream guards can
  react to rejections instead of re-judging completion.

### Fixed
- Truncation guard now catches empty (`{}`) arguments for tools whose schema
  requires fields (`args_len = 0` no longer bypasses the check).
- Streaming no longer suppresses text/thought events that arrive after a
  `tool_call` chunk.
- Deduplicated event delivery in parallel orchestration (child-agent events
  could previously be emitted twice).
