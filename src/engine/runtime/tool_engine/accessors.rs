use std::sync::Arc;

use serde_json::Value;
use tokio::sync::RwLock;

use crate::engine::approval::ApprovalHandler;
use crate::engine::pipeline::DefaultPipeline;
use crate::engine::recovery::ToolErrorRecovery;
use crate::engine::runtime::event_bus::EventBus;
use crate::tool::{ActivationContext, ToolPolicy, ToolRegistry};

use super::ToolEngine;

impl ToolEngine {
    pub fn new(
        tools: ToolRegistry,
        approval_handler: Option<Arc<dyn ApprovalHandler>>,
        tool_policy: Option<Arc<dyn ToolPolicy>>,
        error_recovery: Arc<dyn ToolErrorRecovery>,
        event_bus: EventBus,
    ) -> Self {
        let pipeline = DefaultPipeline::new(tool_policy.clone(), None, None);
        Self {
            tools: Arc::new(RwLock::new(tools)),
            approval_handler,
            tool_policy,
            error_recovery,
            event_bus,
            pipeline,
        }
    }

    #[allow(dead_code)]
    pub async fn definitions(&self) -> Vec<Value> {
        self.tools.read().await.definitions()
    }

    /// Return tool definitions filtered by [`ToolExposure`].
    ///
    /// Respects `Direct` / `Deferred` / `Hidden` visibility levels.
    pub async fn definitions_filtered(&self, ctx: &ActivationContext) -> Vec<Value> {
        self.tools.read().await.definitions_filtered(ctx)
    }

    /// Shared handle to the tool registry (test-only).
    ///
    /// Lets a test fixture hold the same `Arc` the engine executes against, so
    /// it can register tools from inside `call` — the self-registration pattern
    /// that deadlocked when the engine held the registry lock across a call.
    #[cfg(test)]
    pub(crate) fn tools_handle(&self) -> Arc<RwLock<ToolRegistry>> {
        Arc::clone(&self.tools)
    }

    /// Does the named tool's JSON schema declare at least one required
    /// parameter?
    ///
    /// Used by the react truncation guard to tell a *legitimate* empty-arguments
    /// call (`{}` for a no-arg tool such as `list_agents`) apart from a
    /// *degenerate* one: an empty `spawn_agent {}` the model echoed back from
    /// sanitized history. A call to a required-field tool with `{}` args can
    /// never succeed, so the guard re-issues it instead of letting it fall
    /// through to a `ToolArgsInvalid` execution failure. Unknown tools return
    /// `false` so they take the normal schema-validation path.
    pub async fn tool_requires_params(&self, name: &str) -> bool {
        let guard = self.tools.read().await;
        guard
            .get(name)
            .map(|t| {
                t.schema()
                    .get("required")
                    .and_then(Value::as_array)
                    .is_some_and(|a| !a.is_empty())
            })
            .unwrap_or(false)
    }

    /// The named tool's input schema, or `None` when the tool is unknown.
    /// Read-only sibling of [`Self::tool_requires_params`] for callers that
    /// need the full schema — e.g. the react truncation guard's markup
    /// salvage, which validates required fields and coerces recovered
    /// argument values by declared type.
    pub async fn tool_schema(&self, name: &str) -> Option<Value> {
        let guard = self.tools.read().await;
        guard.get(name).map(|t| t.schema())
    }

    pub fn error_recovery(&self) -> &Arc<dyn ToolErrorRecovery> {
        &self.error_recovery
    }

    pub fn approval_handler(&self) -> Option<&Arc<dyn ApprovalHandler>> {
        self.approval_handler.as_ref()
    }

    pub fn tool_policy(&self) -> Option<&Arc<dyn ToolPolicy>> {
        self.tool_policy.as_ref()
    }

    pub fn tools_arc(&self) -> Arc<RwLock<ToolRegistry>> {
        self.tools.clone()
    }
}
