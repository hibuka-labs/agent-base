//! [`DeferredTool`]: wrap a tool so it is named in the roster but costs the
//! request nothing until the model opens it.
//!
//! The wrapper is the whole point. The alternative — overriding `exposure()` on
//! each tool — means touching every crate a tool lives in, and the ones worth
//! deferring (browser, canvas, image) are exactly the ones phimint does not
//! own. One adapter at the registration site keeps the decision where it
//! belongs: with the product, which knows what its current shape of work needs.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::{Content, Tool, ToolContext, ToolExposure};

/// A tool that is on the roster but not on the wire until it is opened.
///
/// Everything except [`Tool::exposure`] is delegated — same name, same
/// description, same schema, same behaviour. The only thing that changes is
/// when the model is told it exists.
pub struct DeferredTool {
    inner: Arc<dyn Tool>,
}

impl DeferredTool {
    pub fn new(inner: Arc<dyn Tool>) -> Self {
        Self { inner }
    }

    /// The wrapped tool, for a caller that needs the real one (the roster and
    /// `activate_tool`'s candidates are both rendered from it).
    pub fn inner(&self) -> Arc<dyn Tool> {
        Arc::clone(&self.inner)
    }
}

#[async_trait]
impl Tool for DeferredTool {
    fn name(&self) -> &'static str {
        self.inner.name()
    }

    fn description(&self) -> &'static str {
        self.inner.description()
    }

    fn schema(&self) -> Value {
        self.inner.schema()
    }

    fn exposure(&self) -> ToolExposure {
        ToolExposure::Deferred
    }

    async fn call(
        &self,
        args: &Value,
        ctx: &ToolContext,
    ) -> crate::types::AgentResult<Vec<Content>> {
        self.inner.call(args, ctx).await
    }

    fn timeout_ms(&self) -> Option<u64> {
        self.inner.timeout_ms()
    }

    fn call_timeout_ms(&self, args: &Value) -> Option<u64> {
        self.inner.call_timeout_ms(args)
    }

    fn metadata(&self) -> super::ToolMetadata {
        self.inner.metadata()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::engine::EventBus;
    use crate::tool::{ActivationContext, ExecClock, ToolActivation, render_tool_definition};
    use crate::types::{AgentResult, SessionId};
    use serde_json::json;

    struct Named(&'static str, ToolExposure);

    #[async_trait]
    impl Tool for Named {
        fn name(&self) -> &'static str {
            self.0
        }
        fn description(&self) -> &'static str {
            "inner description"
        }
        fn schema(&self) -> Value {
            json!({"type": "object", "properties": {"a": {"type": "string"}}})
        }
        fn exposure(&self) -> ToolExposure {
            self.1.clone()
        }
        async fn call(&self, _args: &Value, _ctx: &ToolContext) -> AgentResult<Vec<Content>> {
            Ok(vec![Content::text("inner ran")])
        }
    }

    fn ctx() -> ToolContext {
        let (tx, _rx) = tokio::sync::mpsc::unbounded_channel();
        ToolContext {
            session_id: SessionId::new(0),
            user_event_tx: tx,
            llm_client: None,
            session_store: None,
            language: crate::types::Language::En,
            cancel_token: tokio_util::sync::CancellationToken::new(),
            max_output_chars: None,
            exec_clock: ExecClock::default(),
            waiting_for_human: Arc::new(AtomicBool::new(false)),
            event_bus: EventBus::new(1),
        }
    }

    /// Value: protects=the decision the wrapper exists for — deferring is a
    /// packaging call, so it must not change what the tool is or does; the
    /// model gets the same name, description, schema and behaviour either way.
    /// fails_when=the wrapper alters any of them. why_new=the wrapper is new.
    /// seam=registration -> request; no_claim=that the inner tool is unchanged
    /// in ways not covered here (its own config).
    #[tokio::test]
    async fn wrapping_changes_when_a_tool_is_announced_and_nothing_else() {
        let inner = Arc::new(Named("browser_navigate", ToolExposure::Direct));
        let deferred = DeferredTool::new(Arc::clone(&inner) as Arc<dyn Tool>);

        assert_eq!(deferred.name(), "browser_navigate");
        assert_eq!(deferred.description(), "inner description");
        assert_eq!(deferred.schema(), inner.schema());
        assert_eq!(deferred.timeout_ms(), inner.timeout_ms());

        assert_eq!(deferred.exposure(), ToolExposure::Deferred);
        assert_eq!(
            inner.exposure(),
            ToolExposure::Direct,
            "the inner is untouched"
        );

        let out = deferred.call(&json!({}), &ctx()).await.expect("delegates");
        assert!(
            matches!(&out[0], Content::Text { text } if text == "inner ran"),
            "and it still does the work"
        );
    }

    /// Value: protects=the roster's promise — a wrapped tool must cost the
    /// request nothing until it is opened, which is the entire reason to wrap
    /// it; fails_when=a wrapped tool shows up unopened. why_new=the wrapper is
    /// new. seam=ToolExposure -> definitions_filtered; no_claim=that the
    /// product wraps the right tools.
    #[test]
    fn a_wrapped_tool_costs_nothing_until_it_is_opened() {
        use crate::tool::ToolRegistry;

        let mut r = ToolRegistry::default();
        r.register(Named("plain", ToolExposure::Direct));
        r.register(DeferredTool::new(Arc::new(Named(
            "late",
            ToolExposure::Direct,
        ))));

        let closed = ActivationContext {
            session_id: SessionId::new(0),
            current_tools: vec![],
            workspace: std::path::PathBuf::from("/tmp"),
        };
        let names = |ctx: &ActivationContext| -> Vec<String> {
            r.definitions_filtered(ctx)
                .iter()
                .map(|d| d["function"]["name"].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(names(&closed), vec!["plain".to_string()], "not yet");

        let _ = ToolActivation::new();
        let mut open = closed.clone();
        open.current_tools = vec!["late".to_string()];
        assert_eq!(
            names(&open),
            vec!["late".to_string(), "plain".to_string()],
            "and once opened it is on the request like any other"
        );
    }

    /// Value: protects=what `activate_tool` hands back — it renders from the
    /// wrapper, so the wrapper must reach the inner tool's real definition or
    /// the model gets a blank schema and calls blind; fails_when=the rendered
    /// definition loses the inner's arguments. why_new=the wrapper is new.
    /// seam=wrapper -> wire definition.
    #[test]
    fn the_rendered_definition_is_the_inner_tools() {
        let deferred = DeferredTool::new(Arc::new(Named("late", ToolExposure::Direct)));
        let def = render_tool_definition(&deferred);
        assert_eq!(def["function"]["name"], "late");
        assert_eq!(def["function"]["description"], "inner description");
        assert_eq!(
            def["function"]["parameters"]["properties"]["a"]["type"],
            "string"
        );
    }
}
