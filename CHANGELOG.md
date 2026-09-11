# Changelog

All notable changes to `agent-base` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
