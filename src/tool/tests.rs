use super::*;
use llm_trait::ImageAttachment;

#[test]
fn exec_clock_pause_accounting() {
    let clock = ExecClock::default();
    assert_eq!(clock.paused_total(), std::time::Duration::ZERO);

    let guard = clock.pause();
    std::thread::sleep(std::time::Duration::from_millis(30));
    // An open pause counts in-flight — the pipeline's deadline moves live.
    assert!(clock.paused_total() >= std::time::Duration::from_millis(30));
    drop(guard);
    let banked = clock.paused_total();
    assert!(banked >= std::time::Duration::from_millis(30));

    // Closed pauses stay banked; a later pause adds to them.
    std::thread::sleep(std::time::Duration::from_millis(10));
    assert_eq!(clock.paused_total(), banked);
    let g2 = clock.pause();
    std::thread::sleep(std::time::Duration::from_millis(20));
    assert!(clock.paused_total() >= banked + std::time::Duration::from_millis(20));
    drop(g2);
}

#[test]
fn exec_clock_nested_guards_resume_once() {
    let clock = ExecClock::default();
    let outer = clock.pause();
    let inner = clock.pause();
    std::thread::sleep(std::time::Duration::from_millis(15));
    drop(inner);
    std::thread::sleep(std::time::Duration::from_millis(15));
    // Still paused via `outer` — the in-flight pause keeps growing.
    let during = clock.paused_total();
    assert!(during >= std::time::Duration::from_millis(30));
    drop(outer);
    let after = clock.paused_total();
    assert!(after >= during);
    std::thread::sleep(std::time::Duration::from_millis(10));
    assert_eq!(clock.paused_total(), after);
}

#[test]
fn exec_clock_guard_drop_unwinds_from_panic() {
    // A panic while holding the guard must not poison the clock.
    let clock = ExecClock::default();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _guard = clock.pause();
        panic!("boom");
    }));
    assert!(caught.is_err());
    std::thread::sleep(std::time::Duration::from_millis(5));
    let banked = clock.paused_total();
    std::thread::sleep(std::time::Duration::from_millis(10));
    assert_eq!(clock.paused_total(), banked, "clock must be resumed");
}

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

#[tokio::test]
async fn typed_tool_arg_error_names_the_missing_field() {
    // Session 20261104_23c198dc: 4 spawn_agent calls omitted the required
    // `task` field and the model-visible error echoed the raw args with no
    // serde detail — the model had to guess what was wrong. The error must
    // name the missing field and keep the args echo (recovery.rs contract).
    #[derive(schemars::JsonSchema, serde::Deserialize)]
    struct GreetArgs {
        #[allow(dead_code)]
        name: String,
    }

    struct Greet;

    #[async_trait]
    impl TypedTool for Greet {
        type Args = GreetArgs;
        type Output = String;

        fn name(&self) -> &'static str {
            "greet"
        }
        fn description(&self) -> &'static str {
            "greet"
        }
        async fn call_typed(
            &self,
            _args: Self::Args,
            _ctx: &ToolContext,
        ) -> AgentResult<Self::Output> {
            Ok(String::new())
        }
    }

    let err = Greet
        .call(&serde_json::json!({}), &ToolContext::for_test())
        .await
        .expect_err("missing required field must fail");
    let msg = err.to_string();
    assert!(
        msg.contains("missing field `name`"),
        "serde detail must reach the model: {msg}"
    );
    assert!(msg.contains("(args: {}"), "raw args echo kept: {msg}");
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

/// The two faces of a tool result partition it: `content_text` keeps the
/// words, `content_images` the pictures — in the order the tool produced
/// them, so a screenshot the model must compare against the one before
/// it arrives in the same order it was taken.
#[test]
fn content_images_takes_the_pictures_content_text_drops() {
    let out = [
        Content::text("words"),
        Content::image("Zmlyc3Q=", "image/png"),
        Content::text("more words"),
        Content::image("c2Vjb25k", "image/png"),
        Content::detail(serde_json::json!({"line": 3})),
    ];
    let images = content_images(&out);
    assert_eq!(images.len(), 2);
    for (img, want) in images.iter().zip(["Zmlyc3Q=", "c2Vjb25k"]) {
        match img {
            ImageAttachment::Base64 {
                data, media_type, ..
            } => {
                assert_eq!(data, want);
                assert_eq!(media_type.as_deref(), Some("image/png"));
            }
            other => panic!("expected a base64 attachment, got {other:?}"),
        }
    }
    assert!(content_images(&[Content::text("no pictures")]).is_empty());
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
        exec_clock: ExecClock::default(),
        waiting_for_human: Arc::new(AtomicBool::new(false)),
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
        exec_clock: ExecClock::default(),
        waiting_for_human: Arc::new(AtomicBool::new(false)),
        event_bus: crate::engine::EventBus::new(4),
    };
    let mut bus_rx = ctx.event_bus.subscribe();
    ctx.emit_progress("working");

    // Direct channel still gets the event (real-time tool select arm).
    assert!(matches!(rx.try_recv().unwrap(), UserEvent::Progress { text } if text == "working"));

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
    async fn call_typed(&self, args: GreetArgs, _ctx: &ToolContext) -> AgentResult<GreetResult> {
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

/// Value: protects=what `Deferred` means — opt-in, not opt-out. A tool that is
/// named in the roster but unopened must cost the request nothing, and the
/// default must not hand it back anyway; fails_when=an unopened tool is
/// included, or an opened one is not. why_new=the default used to return `true`,
/// which made `Deferred` a no-op label. seam=Tool::should_activate; no_claim=
/// about overrides that auto-open on a workspace shape.
#[test]
fn a_deferred_tool_is_closed_until_the_model_opens_it() {
    let t = NamedTool("browser_navigate");
    let closed = default_ctx();
    assert!(
        !t.should_activate(&closed),
        "unopened means absent from the request"
    );

    let mut open = default_ctx();
    open.current_tools = vec!["browser_navigate".to_string()];
    assert!(t.should_activate(&open), "opening it is what puts it back");
}

/// Value: protects=the end-to-end promise of the roster — what the model opens
/// is exactly what it gets, no more and no less; fails_when=the activation
/// names never reach the filter, so every Deferred tool is stuck shut.
/// why_new=the seeding is new. seam=ToolActivation -> definitions_filtered;
/// no_claim=that the roster is the only way a tool can open.
#[test]
fn definitions_filtered_honours_the_session_openings() {
    let mut r = ToolRegistry::default();
    r.register(ExposureTool("direct_a", ToolExposure::Direct));
    r.register(ExposureTool("deferred_a", ToolExposure::Deferred));
    r.register(ExposureTool("deferred_b", ToolExposure::Deferred));

    let mut ctx = default_ctx();
    ctx.current_tools = vec!["deferred_b".to_string()];
    let defs = r.definitions_filtered(&ctx);
    let names: Vec<&str> = defs
        .iter()
        .map(|d| d["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(
        names,
        vec!["deferred_b", "direct_a"],
        "the opened one comes back and the unopened one does not"
    );
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
    /// Value: protects=order-independence — some tools are registered early,
    /// some arrive from a factory at build time, some come from plugins. A
    /// mark recorded before the tool exists must still defer it when it lands;
    /// session 20261116_b97461af left 9 tools mounted because they registered
    /// after the call that was supposed to defer them. fails_when=deferring
    /// before registration is a no-op. why_new=the first cut wrapped at call
    /// time and so depended on order. seam=ToolRegistry::defer -> register_arc.
    #[test]
    fn a_tool_deferred_before_registration_is_still_deferred() {
        let mut r = ToolRegistry::default();
        assert!(
            r.defer("late").is_none(),
            "not registered yet, so no definition"
        );
        r.register(NamedTool("late"));

        let ctx = default_ctx();
        let defs = r.definitions_filtered(&ctx);
        let names: Vec<&str> = defs
            .iter()
            .map(|d| d["function"]["name"].as_str().unwrap())
            .collect();
        assert!(
            names.is_empty(),
            "a deferred-before-registration tool must not mount: {names:?}"
        );

        let mut open = ctx.clone();
        open.current_tools = vec!["late".to_string()];
        let defs = r.definitions_filtered(&open);
        let names: Vec<&str> = defs
            .iter()
            .map(|d| d["function"]["name"].as_str().unwrap())
            .collect();
        assert_eq!(names, vec!["late"], "and it mounts once opened");
    }
}
