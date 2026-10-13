use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Value, json};
use tokio::sync::mpsc;

use crate::engine::SessionStore;
use crate::types::{AgentError, AgentResult, RuntimeEvent, SessionId, UserEvent};

pub mod auto_continue;
pub mod policy;
pub mod update_plan;

pub use auto_continue::AutoContinueTool;
pub use update_plan::UpdatePlanTool;

pub use policy::{DenyAllToolPolicy, ToolDecision, ToolPolicy};

// Re-export pure types from agent-types
pub use agent_types::{
    ActivationContext, Content, ToolExposure, ToolMetadata, content_details, content_text,
};

#[derive(Clone)]
pub struct ToolContext {
    pub session_id: SessionId,
    /// Channel for sending user-space events (progress, sub-agent, structured).
    /// Tools should use `emit_user_event()` or `emit_progress()`.
    pub user_event_tx: mpsc::UnboundedSender<UserEvent>,
    pub llm_client: Option<Arc<dyn llm_trait::LlmProvider>>,
    pub session_store: Option<Arc<dyn SessionStore>>,
    /// Language preference for tool output.
    /// Defaults to `Language::En` if not set.
    pub language: crate::types::Language,
    /// Cancellation token for checking if the operation should be cancelled.
    pub cancel_token: tokio_util::sync::CancellationToken,
    /// Output budget for this call, in characters (set from the engine's
    /// `max_tool_output_chars`). Tools that can return large results (e.g.
    /// `read_file`) should self-truncate to this bound and mark the cut, so
    /// the engine's hard reject (§6.5) never fires for a paginated read.
    pub max_output_chars: Option<usize>,
    /// Internal runtime event bus (framework tools emit `RuntimeEvent`s here).
    /// `pub(crate)` — engine-internal; user tools should use `emit_user_event()`.
    pub(crate) event_bus: crate::engine::EventBus,
}

impl ToolContext {
    /// Send a user-space event (progress, sub-agent forwarding, structured data).
    ///
    /// Dual-write: the event goes to `user_event_tx` (the tool select arm
    /// renders it in real time) and to the event bus for external subscribers
    /// (bridge remote clients, persistence). The bus copy carries `agent_id:
    /// None` — the self-echo marker — so bus loopback drops it instead of
    /// re-rendering (see `crate::engine::runtime::is_self_echo_user_event`).
    pub fn emit_user_event(&self, event: UserEvent) {
        let _ = self.user_event_tx.send(event.clone());
        self.event_bus.emit(RuntimeEvent::UserEvent {
            session_id: self.session_id.clone(),
            event,
            agent_id: None,
            trace_id: None,
        });
    }

    /// Convenience: send a progress event with text.
    pub fn emit_progress(&self, text: impl Into<String>) {
        self.emit_user_event(UserEvent::Progress { text: text.into() });
    }

    /// Convenience: emit a partial result during long-running tool execution.
    /// `is_partial: true` means more output is coming; `false` means final.
    pub fn emit_partial_result(
        &self,
        tool_call_id: &str,
        content: impl Into<String>,
        is_partial: bool,
    ) {
        self.emit_user_event(UserEvent::ToolPartialResult {
            tool_call_id: tool_call_id.to_string(),
            content: content.into(),
            is_partial,
        });
    }

    /// Test-only constructor: a `ToolContext` with a disconnected event
    /// channel and no LLM/session backends. Lets downstream tests drop their
    /// bespoke `dummy_ctx()` helpers.
    pub fn for_test() -> Self {
        let (tx, _rx) = mpsc::unbounded_channel();
        ToolContext {
            session_id: SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: crate::types::Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            max_output_chars: None,
            event_bus: crate::engine::EventBus::new(1),
        }
    }

    /// Host-side constructor for host-initiated tool execution outside the
    /// engine loop (e.g. a REPL driving a tool through the same approval path
    /// as the engine). User events are accepted but dropped — there is no
    /// engine event loop to render them.
    pub fn new(
        session_id: SessionId,
        llm_client: Option<Arc<dyn crate::llm_trait::LlmProvider>>,
        session_store: Option<Arc<dyn SessionStore>>,
        cancel_token: tokio_util::sync::CancellationToken,
    ) -> Self {
        let (tx, _rx) = mpsc::unbounded_channel();
        ToolContext {
            session_id,
            user_event_tx: tx,
            llm_client,
            session_store,
            language: crate::types::Language::En,
            cancel_token,
            max_output_chars: None,
            event_bus: crate::engine::EventBus::new(1),
        }
    }
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    /// Human-readable description of what this tool does and when to use it.
    fn description(&self) -> &'static str;
    /// JSON Schema for the tool's input arguments (MCP `inputSchema` shape,
    /// without the provider envelope).
    fn schema(&self) -> Value;
    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>>;

    /// Timeout for this tool in milliseconds.
    ///
    /// Returns `Some(ms)` to enforce a timeout, or `None` to use the
    /// framework's default timeout from `ToolConfig.default_tool_timeout_ms`.
    /// Tools that need more or less time should override this method.
    fn timeout_ms(&self) -> Option<u64> {
        None // Use framework default
    }

    /// Machine-readable metadata for tool introspection.
    ///
    /// The default implementation derives `name` and `description` from
    /// [`Tool::name`] and [`Tool::description`], sets `origin` to `"custom"`,
    /// and leaves `requirements` empty. Tool authors are encouraged to
    /// override this to provide an accurate `origin` and `version`.
    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name().to_string(),
            description: self.description().to_string(),
            origin: "custom".to_string(),
            version: "unknown".to_string(),
            requirements: vec![],
        }
    }

    /// Visibility level for this tool. Default: `Direct` (always visible).
    ///
    /// Override to return `Deferred` (conditionally visible) or `Hidden`
    /// (never visible to the model). `Deferred` tools are only included
    /// when [`Tool::should_activate`] returns `true`.
    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }

    /// Activation condition for `Deferred` tools.
    ///
    /// Called once per turn for each `Deferred` tool. Return `true` to
    /// include this tool in the model's tool list this turn. Ignored for
    /// `Direct` and `Hidden` tools.
    fn should_activate(&self, _ctx: &ActivationContext) -> bool {
        true
    }
}

#[async_trait]
pub trait TypedTool: Send + Sync {
    type Args: serde::de::DeserializeOwned + schemars::JsonSchema;
    type Output: serde::Serialize;

    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    async fn call_typed(&self, args: Self::Args, ctx: &ToolContext) -> AgentResult<Self::Output>;

    fn format_output(&self, output: Self::Output) -> Content {
        // A `String` output is emitted verbatim; any other serializable type is
        // rendered as JSON. This avoids `serde_json::to_string` wrapping a plain
        // string in literal double quotes (`hello` → `"hello"`), which would leak
        // quotes into the LLM-visible tool result.
        match serde_json::to_value(&output) {
            Ok(serde_json::Value::String(s)) => Content::text(s),
            Ok(other) => Content::text(other.to_string()),
            Err(_) => Content::text(String::new()),
        }
    }

    /// Machine-readable origin of this tool (crate name, `"agent-base"`, or `"custom"`).
    fn origin(&self) -> &'static str {
        "custom"
    }

    /// Crate/package version, or `"unknown"` when built outside a crate.
    fn version(&self) -> &'static str {
        "unknown"
    }

    /// Visibility level for this tool. Default: `Direct` (always visible).
    fn exposure(&self) -> ToolExposure {
        ToolExposure::Direct
    }

    /// Activation condition for `Deferred` tools. Ignored for `Direct`/`Hidden`.
    fn should_activate(&self, _ctx: &ActivationContext) -> bool {
        true
    }
}

#[async_trait]
impl<T: TypedTool + Send + Sync + 'static> Tool for T {
    fn name(&self) -> &'static str {
        TypedTool::name(self)
    }

    fn description(&self) -> &'static str {
        TypedTool::description(self)
    }

    fn schema(&self) -> Value {
        // Generate a provider-safe JSON Schema: Draft 7 (not 2020-12), with
        // nested subschemas inlined (no `$ref`/`$defs`/`/definitions`) and no
        // root `$schema`/meta-schema key. OpenAI-compatible function-calling
        // rejects `$ref` and 2020-12's `$defs`, so the default 2020-12 output
        // would break any `Args` containing a nested enum or struct.
        let settings = schemars::generate::SchemaSettings::draft07().with(|s| {
            s.inline_subschemas = true;
            s.meta_schema = None;
        });
        let generator = schemars::SchemaGenerator::new(settings);
        let schema = generator.into_root_schema_for::<T::Args>();
        serde_json::to_value(schema).unwrap_or(Value::Null)
    }

    fn metadata(&self) -> ToolMetadata {
        ToolMetadata {
            name: self.name().to_string(),
            description: self.description().to_string(),
            origin: self.origin().to_string(),
            version: self.version().to_string(),
            requirements: vec![],
        }
    }

    fn exposure(&self) -> ToolExposure {
        TypedTool::exposure(self)
    }

    fn should_activate(&self, ctx: &ActivationContext) -> bool {
        TypedTool::should_activate(self, ctx)
    }

    async fn call(&self, args: &Value, ctx: &ToolContext) -> AgentResult<Vec<Content>> {
        let typed_args: T::Args =
            serde_json::from_value(args.clone()).map_err(|_| AgentError::ToolArgsInvalid {
                name: self.name().to_string(),
                raw: args.to_string(),
            })?;
        let output = self.call_typed(typed_args, ctx).await?;
        Ok(vec![self.format_output(output)])
    }
}

/// Render a single tool's definition into the OpenAI function-calling
/// envelope. Tools only provide `name`/`description`/`schema`; the envelope is
/// assembled here at the LLM boundary (Anthropic gets its own renderer, and
/// tool authors stay protocol-agnostic).
pub fn render_tool_definition(tool: &dyn Tool) -> Value {
    json!({
        "type": "function",
        "function": {
            "name": tool.name(),
            "description": tool.description(),
            "parameters": tool.schema(),
        }
    })
}

pub(crate) type ToolRef = Arc<dyn Tool>;

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: HashMap<String, ToolRef>,
}

impl ToolRegistry {
    pub fn register(&mut self, tool: impl Tool + 'static) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    pub fn register_arc(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.name().to_string(), tool);
    }

    /// Remove a tool from the registry by name.
    pub fn remove(&mut self, name: &str) {
        self.tools.remove(name);
    }

    pub fn get(&self, name: &str) -> Option<ToolRef> {
        self.tools.get(name).cloned()
    }

    pub fn definitions(&self) -> Vec<Value> {
        let mut tools: Vec<_> = self.tools.values().collect();
        tools.sort_by_key(|t| t.name());
        tools
            .into_iter()
            .map(|t| render_tool_definition(t.as_ref()))
            .collect()
    }

    /// Return tool definitions filtered by [`ToolExposure`].
    ///
    /// - `Direct` tools are always included.
    /// - `Deferred` tools are included only when `should_activate(ctx)` returns `true`.
    /// - `Hidden` tools are never included.
    pub fn definitions_filtered(&self, ctx: &ActivationContext) -> Vec<Value> {
        let mut tools: Vec<_> = self.tools.values().collect();
        tools.sort_by_key(|t| t.name());

        // Single sequential pass: collect Direct names first, then evaluate
        // Deferred tools in sorted order. Each Deferred tool sees the names of
        // all Direct + previously-activated Deferred tools in ctx.current_tools.
        // Evaluation order matters: a later Deferred tool can see an earlier one,
        // but not vice versa.
        let direct_names: Vec<String> = tools
            .iter()
            .filter(|t| t.exposure() == ToolExposure::Direct)
            .map(|t| t.name().to_string())
            .collect();

        let mut activated_names = direct_names.clone();
        for t in &tools {
            if t.exposure() == ToolExposure::Deferred {
                let mut ctx_with_tools = ctx.clone();
                ctx_with_tools.current_tools = activated_names.clone();
                if t.should_activate(&ctx_with_tools) {
                    activated_names.push(t.name().to_string());
                }
            }
        }

        tools
            .into_iter()
            .filter(|t| match t.exposure() {
                ToolExposure::Direct => true,
                ToolExposure::Deferred => activated_names.contains(&t.name().to_string()),
                ToolExposure::Hidden => false,
            })
            .map(|t| render_tool_definition(t.as_ref()))
            .collect()
    }

    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Collect metadata for every registered tool, sorted by name.
    ///
    /// This is the preferred introspection API for consumers — it returns a
    /// stable `ToolMetadata` struct per tool instead of having callers parse
    /// the LLM-facing JSON definitions.
    pub fn metadatas(&self) -> Vec<ToolMetadata> {
        let mut list: Vec<_> = self.tools.values().map(|tool| tool.metadata()).collect();
        list.sort_by(|a, b| a.name.cmp(&b.name));
        list
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_text_ctor_and_into_vec() {
        let c = Content::text("hello");
        let v: Vec<Content> = c.clone().into();
        assert_eq!(v.len(), 1);
        assert!(matches!(v[0], Content::Text { .. }));
        assert!(matches!(&c, Content::Text { text } if text == "hello"));
    }

    #[test]
    fn content_serializes_with_type_tag() {
        let c = Content::text("hi");
        let j = serde_json::to_value(&c).unwrap();
        assert_eq!(j["type"], "text");
        assert_eq!(j["text"], "hi");
    }

    #[test]
    fn tool_context_for_test_constructs() {
        let ctx = ToolContext::for_test();
        assert!(ctx.llm_client.is_none());
        assert!(ctx.session_store.is_none());
        assert!(!ctx.cancel_token.is_cancelled());
        ctx.emit_progress("hello");
    }

    #[test]
    fn typed_tool_schema_is_derived_from_args() {
        #[derive(schemars::JsonSchema, serde::Deserialize, serde::Serialize)]
        struct GreetArgs {
            name: String,
            #[serde(default)]
            times: u32,
        }

        let schema = schemars::schema_for!(GreetArgs);
        let j = serde_json::to_value(&schema).unwrap();
        // The derived schema exposes the struct's fields as object properties.
        assert!(j["properties"]["name"].is_object());
        assert!(j["properties"]["times"].is_object());
    }

    #[test]
    fn typed_tool_schema_is_provider_safe_for_nested_enum() {
        #[derive(schemars::JsonSchema, serde::Deserialize, serde::Serialize)]
        enum Status {
            Active,
            Paused,
        }

        #[derive(schemars::JsonSchema, serde::Deserialize, serde::Serialize)]
        struct Args {
            name: String,
            status: Status,
        }

        #[derive(Default)]
        struct NestedTool;
        #[async_trait]
        impl TypedTool for NestedTool {
            type Args = Args;
            type Output = String;
            fn name(&self) -> &'static str {
                "nested"
            }
            fn description(&self) -> &'static str {
                ""
            }
            async fn call_typed(
                &self,
                _args: Args,
                _ctx: &ToolContext,
            ) -> crate::types::AgentResult<String> {
                Ok(String::new())
            }
        }

        let schema = Tool::schema(&NestedTool);
        let raw = schema.to_string();
        // OpenAI-compatible function-calling rejects $ref / $defs; the nested
        // enum must be inlined rather than referenced.
        assert!(!raw.contains("$ref"), "schema contains $ref: {raw}");
        assert!(!raw.contains("$defs"), "schema contains $defs: {raw}");
        assert!(
            !raw.contains("definitions"),
            "schema has definitions: {raw}"
        );
        assert!(schema.get("$schema").is_none(), "schema has $schema key");

        // The enum variants are inlined directly under properties.status.
        let variants: Vec<&str> = schema["properties"]["status"]["enum"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        assert!(variants.contains(&"Active"), "missing Active: {variants:?}");
        assert!(variants.contains(&"Paused"), "missing Paused: {variants:?}");
    }

    #[test]
    fn definitions_are_sorted_by_name() {
        struct NamedTool(&'static str);
        #[async_trait::async_trait]
        impl Tool for NamedTool {
            fn name(&self) -> &'static str {
                self.0
            }
            fn description(&self) -> &'static str {
                ""
            }
            fn schema(&self) -> serde_json::Value {
                serde_json::Value::Null
            }
            async fn call(
                &self,
                _args: &serde_json::Value,
                _ctx: &ToolContext,
            ) -> crate::types::AgentResult<Vec<Content>> {
                Ok(vec![])
            }
        }

        let mut registry = ToolRegistry::default();
        registry.register(NamedTool("zeta"));
        registry.register(NamedTool("alpha"));
        registry.register(NamedTool("mike"));

        let defs = registry.definitions();
        let names: Vec<&str> = defs
            .iter()
            .map(|d| d["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["alpha", "mike", "zeta"]);
    }

    // ── B4: content image / partial result / typed-tool blanket / registry ──

    #[test]
    fn content_image_and_content_text_skips_images() {
        let img = Content::image("base64data", "image/png");
        assert!(
            matches!(&img, Content::Image { data, mime_type } if data == "base64data" && mime_type == "image/png")
        );

        let text = content_text(&[
            Content::text("a"),
            Content::image("b", "image/png"),
            Content::text("c"),
        ]);
        assert_eq!(text, "a\nc");
    }

    #[test]
    fn emit_partial_result_sends_event() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let ctx = ToolContext {
            session_id: SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: crate::types::Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            max_output_chars: None,
            event_bus: crate::engine::EventBus::new(1),
        };
        ctx.emit_partial_result("tc1", "partial", true);
        match rx.try_recv().unwrap() {
            UserEvent::ToolPartialResult {
                tool_call_id,
                content,
                is_partial,
            } => {
                assert_eq!(tool_call_id, "tc1");
                assert_eq!(content, "partial");
                assert!(is_partial);
            }
            other => panic!("unexpected event: {other:?}"),
        }
    }

    // Dual-write contract (Batch E follow-up): a tool UserEvent must reach
    // event_bus subscribers (bridge remote clients, persistence) as well as the
    // per-call user_event channel — and the bus copy must carry `agent_id: None`,
    // the self-echo marker that tells bus loopback not to re-render it (see
    // `engine::event_bus::is_self_echo_user_event`).
    #[test]
    fn emit_user_event_dual_writes_to_event_bus() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let ctx = ToolContext {
            session_id: SessionId::new(7),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: crate::types::Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            max_output_chars: None,
            event_bus: crate::engine::EventBus::new(4),
        };
        let mut bus_rx = ctx.event_bus.subscribe();
        ctx.emit_progress("working");

        // Direct channel still gets the event (real-time tool select arm).
        assert!(
            matches!(rx.try_recv().unwrap(), UserEvent::Progress { text } if text == "working")
        );

        // Bus copy for external subscribers, stamped and marked self-echo.
        match bus_rx.try_recv().unwrap() {
            crate::types::RuntimeEvent::UserEvent {
                session_id,
                event,
                agent_id,
                ..
            } => {
                assert_eq!(session_id, SessionId::new(7));
                assert!(matches!(event, UserEvent::Progress { text } if text == "working"));
                assert!(agent_id.is_none());
            }
            other => panic!("unexpected bus event: {other:?}"),
        }
    }

    // A concrete TypedTool exercising the blanket `impl Tool for T`.
    #[derive(schemars::JsonSchema, serde::Deserialize, serde::Serialize)]
    struct GreetArgs {
        name: String,
    }

    struct GreetTool;
    #[async_trait]
    impl TypedTool for GreetTool {
        type Args = GreetArgs;
        type Output = String;
        fn name(&self) -> &'static str {
            "greet"
        }
        fn description(&self) -> &'static str {
            "greets a name"
        }
        fn origin(&self) -> &'static str {
            "test-crate"
        }
        fn version(&self) -> &'static str {
            "1.0.0"
        }
        async fn call_typed(&self, args: GreetArgs, _ctx: &ToolContext) -> AgentResult<String> {
            Ok(format!("Hello, {}!", args.name))
        }
    }

    #[test]
    fn typed_tool_blanket_delegates_name_description() {
        let t = GreetTool;
        assert_eq!(Tool::name(&t), "greet");
        assert_eq!(Tool::description(&t), "greets a name");
    }

    #[test]
    fn typed_tool_metadata_uses_origin_and_version() {
        let m = Tool::metadata(&GreetTool);
        assert_eq!(m.name, "greet");
        assert_eq!(m.description, "greets a name");
        assert_eq!(m.origin, "test-crate");
        assert_eq!(m.version, "1.0.0");
        assert!(m.requirements.is_empty());
    }

    #[tokio::test]
    async fn typed_tool_call_deserializes_and_formats() {
        let ctx = ToolContext::for_test();
        let out = Tool::call(&GreetTool, &json!({"name": "world"}), &ctx)
            .await
            .unwrap();
        // format_output emits a String verbatim (not JSON-quoted).
        assert_eq!(content_text(&out), "Hello, world!");
    }

    // A struct-typed output confirms non-String outputs still serialize as JSON.
    #[derive(serde::Serialize)]
    struct GreetResult {
        message: String,
    }

    struct GreetStructTool;
    #[async_trait]
    impl TypedTool for GreetStructTool {
        type Args = GreetArgs;
        type Output = GreetResult;
        fn name(&self) -> &'static str {
            "greet_struct"
        }
        fn description(&self) -> &'static str {
            "greets as json"
        }
        async fn call_typed(
            &self,
            args: GreetArgs,
            _ctx: &ToolContext,
        ) -> AgentResult<GreetResult> {
            Ok(GreetResult {
                message: format!("Hello, {}!", args.name),
            })
        }
    }

    #[test]
    fn format_output_json_serializes_struct() {
        let out = GreetStructTool.format_output(GreetResult {
            message: "hi".into(),
        });
        assert_eq!(content_text(&[out]), r#"{"message":"hi"}"#);
    }

    #[tokio::test]
    async fn typed_tool_call_invalid_args_is_tool_args_invalid() {
        let ctx = ToolContext::for_test();
        let err = Tool::call(&GreetTool, &json!({"nope": 1}), &ctx)
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::ToolArgsInvalid { .. }));
    }

    struct NamedTool(&'static str);
    #[async_trait]
    impl Tool for NamedTool {
        fn name(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            ""
        }
        fn schema(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        async fn call(
            &self,
            _args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> AgentResult<Vec<Content>> {
            Ok(vec![])
        }
    }

    #[test]
    fn registry_register_arc_get_remove_len_is_empty() {
        let mut r = ToolRegistry::default();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);

        let t: Arc<dyn Tool> = Arc::new(NamedTool("x"));
        r.register_arc(t);
        assert!(!r.is_empty());
        assert_eq!(r.len(), 1);
        assert!(r.get("x").is_some());
        assert!(r.get("missing").is_none());

        r.remove("x");
        assert!(r.is_empty());
    }

    #[test]
    fn metadatas_are_sorted_by_name() {
        let mut r = ToolRegistry::default();
        r.register(NamedTool("zeta"));
        r.register(NamedTool("alpha"));

        let metas = r.metadatas();
        let names: Vec<&str> = metas.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "zeta"]);
        assert_eq!(metas[0].origin, "custom");
        assert_eq!(metas[0].version, "unknown");
    }

    // ── Phase 2: ToolExposure + definitions_filtered ────────────────────

    struct ExposureTool(&'static str, ToolExposure);
    #[async_trait]
    impl Tool for ExposureTool {
        fn name(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            ""
        }
        fn schema(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        async fn call(
            &self,
            _args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> AgentResult<Vec<Content>> {
            Ok(vec![])
        }
        fn exposure(&self) -> ToolExposure {
            self.1.clone()
        }
    }

    struct ConditionalTool(&'static str, bool);
    #[async_trait]
    impl Tool for ConditionalTool {
        fn name(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            ""
        }
        fn schema(&self) -> serde_json::Value {
            serde_json::Value::Null
        }
        async fn call(
            &self,
            _args: &serde_json::Value,
            _ctx: &ToolContext,
        ) -> AgentResult<Vec<Content>> {
            Ok(vec![])
        }
        fn exposure(&self) -> ToolExposure {
            ToolExposure::Deferred
        }
        fn should_activate(&self, _ctx: &ActivationContext) -> bool {
            self.1
        }
    }

    fn default_ctx() -> ActivationContext {
        ActivationContext {
            session_id: crate::types::SessionId::new(0),
            current_tools: vec![],
            workspace: std::path::PathBuf::from("/tmp"),
        }
    }

    #[test]
    fn tool_exposure_default_is_direct() {
        // NamedTool doesn't override exposure() → default Direct
        let t = NamedTool("x");
        assert_eq!(t.exposure(), ToolExposure::Direct);
    }

    #[test]
    fn tool_should_activate_default_is_true() {
        let t = NamedTool("x");
        assert!(t.should_activate(&default_ctx()));
    }

    #[test]
    fn definitions_filtered_includes_direct_excludes_hidden() {
        let mut r = ToolRegistry::default();
        r.register(ExposureTool("direct_a", ToolExposure::Direct));
        r.register(ExposureTool("hidden_a", ToolExposure::Hidden));
        r.register(ExposureTool("direct_b", ToolExposure::Direct));

        let defs = r.definitions_filtered(&default_ctx());
        let names: Vec<&str> = defs
            .iter()
            .map(|d| d["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["direct_a", "direct_b"]);
    }

    #[test]
    fn definitions_filtered_includes_deferred_when_activated() {
        let mut r = ToolRegistry::default();
        r.register(ExposureTool("always", ToolExposure::Direct));
        r.register(ConditionalTool("maybe", true)); // should_activate = true
        r.register(ConditionalTool("never", false)); // should_activate = false

        let defs = r.definitions_filtered(&default_ctx());
        let names: Vec<&str> = defs
            .iter()
            .map(|d| d["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["always", "maybe"]);
    }

    #[test]
    fn definitions_filtered_all_hidden_returns_empty() {
        let mut r = ToolRegistry::default();
        r.register(ExposureTool("h1", ToolExposure::Hidden));
        r.register(ExposureTool("h2", ToolExposure::Hidden));

        let defs = r.definitions_filtered(&default_ctx());
        assert!(defs.is_empty());
    }

    #[test]
    fn definitions_filtered_empty_registry() {
        let r = ToolRegistry::default();
        let defs = r.definitions_filtered(&default_ctx());
        assert!(defs.is_empty());
    }

    #[test]
    fn definitions_unfiltered_includes_hidden() {
        // The original definitions() method does NOT filter — all tools appear.
        let mut r = ToolRegistry::default();
        r.register(ExposureTool("visible", ToolExposure::Direct));
        r.register(ExposureTool("secret", ToolExposure::Hidden));

        let defs = r.definitions();
        assert_eq!(defs.len(), 2);
    }
}
