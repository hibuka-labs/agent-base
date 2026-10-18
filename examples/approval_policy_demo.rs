use serde_json::{Value, json};
use std::collections::HashMap;
/// Demonstrates how ToolPolicy decides whether approval is needed based on the plan status
///
/// Problem scenario:
/// - After create_plan succeeds, react automatically runs execute_plan
/// - But ToolPolicy doesn't know the user has confirmed, and still requests approval
///
/// Solution:
/// - ToolPolicy checks the plan status
/// - If the plan status is "awaiting_confirmation", skip approval
///
/// How to run:
/// cargo run --example approval_policy_demo
use std::sync::Arc;
use tokio::sync::RwLock;

/// Mock PlanStore
struct MockPlanStore {
    plans: RwLock<HashMap<String, PlanData>>,
}

#[derive(Clone)]
struct PlanData {
    plan_id: String,
    status: String, // "awaiting_confirmation", "approved", "executing", etc.
    steps: Vec<StepData>,
}

#[derive(Clone)]
#[allow(dead_code)]
struct StepData {
    id: String,
    command: String,
    risk_level: String, // "safe", "sensitive", "destructive"
}

impl MockPlanStore {
    fn new() -> Self {
        Self {
            plans: RwLock::new(HashMap::new()),
        }
    }

    async fn save_plan(&self, plan: PlanData) {
        let mut plans = self.plans.write().await;
        plans.insert(plan.plan_id.clone(), plan);
    }

    async fn load_plan(&self, plan_id: &str) -> Option<PlanData> {
        let plans = self.plans.read().await;
        plans.get(plan_id).cloned()
    }
}

/// Mock ToolPolicy
struct TestToolPolicy {
    plan_store: Arc<MockPlanStore>,
}

impl TestToolPolicy {
    fn new(plan_store: Arc<MockPlanStore>) -> Self {
        Self { plan_store }
    }

    /// Evaluate whether approval is needed
    async fn evaluate_approval(&self, tool_name: &str, args: &Value) -> Option<String> {
        if tool_name != "execute_plan" {
            return None;
        }

        let plan_id = args.get("plan_id")?.as_str()?;

        // Load the plan
        let plan_data = self.plan_store.load_plan(plan_id).await?;

        // Check the plan status
        if plan_data.status == "approved" {
            // The user has confirmed the plan; skip approval
            println!(
                "[ToolPolicy] Plan status is approved, skipping approval (user has confirmed)"
            );
            return None;
        }

        // Check the command risk levels
        let has_sensitive = plan_data.steps.iter().any(|s| s.risk_level == "sensitive");
        let has_destructive = plan_data
            .steps
            .iter()
            .any(|s| s.risk_level == "destructive");

        if has_destructive {
            println!("[ToolPolicy] Destructive command detected, approval required");
            return Some("destructive".to_string());
        }

        if has_sensitive {
            println!("[ToolPolicy] Sensitive command detected, approval required");
            return Some("sensitive".to_string());
        }

        println!("[ToolPolicy] All commands are safe, skipping approval");
        None
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToolPolicy Approval Decision Demo ===\n");

    let plan_store = Arc::new(MockPlanStore::new());
    let policy = TestToolPolicy::new(plan_store.clone());

    // Scenario 1: plan status is approved (user has confirmed)
    println!("--- Scenario 1: user has confirmed the plan (status is approved) ---");
    plan_store
        .save_plan(PlanData {
            plan_id: "plan-1".to_string(),
            status: "approved".to_string(),
            steps: vec![StepData {
                id: "step-1".to_string(),
                command: "df -h".to_string(),
                risk_level: "safe".to_string(),
            }],
        })
        .await;

    let args = json!({"plan_id": "plan-1"});
    let result = policy.evaluate_approval("execute_plan", &args).await;
    println!("Result: {:?}\n", result);

    // Scenario 2: plan status is awaiting_confirmation (waiting for user confirmation), with a sensitive command
    println!(
        "--- Scenario 2: waiting for user confirmation, plan contains a sensitive command ---"
    );
    plan_store
        .save_plan(PlanData {
            plan_id: "plan-2".to_string(),
            status: "awaiting_confirmation".to_string(),
            steps: vec![StepData {
                id: "step-1".to_string(),
                command: "systemctl restart nginx".to_string(),
                risk_level: "sensitive".to_string(),
            }],
        })
        .await;

    let args = json!({"plan_id": "plan-2"});
    let result = policy.evaluate_approval("execute_plan", &args).await;
    println!("Result: {:?}\n", result);

    // Scenario 3: plan status is awaiting_confirmation (waiting for user confirmation), all commands are safe
    println!("--- Scenario 3: waiting for user confirmation, all commands are safe ---");
    plan_store
        .save_plan(PlanData {
            plan_id: "plan-3".to_string(),
            status: "awaiting_confirmation".to_string(),
            steps: vec![
                StepData {
                    id: "step-1".to_string(),
                    command: "df -h".to_string(),
                    risk_level: "safe".to_string(),
                },
                StepData {
                    id: "step-2".to_string(),
                    command: "free -m".to_string(),
                    risk_level: "safe".to_string(),
                },
            ],
        })
        .await;

    let args = json!({"plan_id": "plan-3"});
    let result = policy.evaluate_approval("execute_plan", &args).await;
    println!("Result: {:?}\n", result);

    // Scenario 4: plan status is awaiting_confirmation (waiting for user confirmation), with a destructive command
    println!("--- Scenario 4: waiting for user confirmation, contains a destructive command ---");
    plan_store
        .save_plan(PlanData {
            plan_id: "plan-4".to_string(),
            status: "awaiting_confirmation".to_string(),
            steps: vec![StepData {
                id: "step-1".to_string(),
                command: "rm -rf /tmp/*".to_string(),
                risk_level: "destructive".to_string(),
            }],
        })
        .await;

    let args = json!({"plan_id": "plan-4"});
    let result = policy.evaluate_approval("execute_plan", &args).await;
    println!("Result: {:?}\n", result);

    println!("=== Demo complete ===");
    println!("\nConclusion:");
    println!(
        "- ToolPolicy decides whether approval is needed based on the plan status and command risk levels"
    );
    println!("- If the plan status is approved (user has confirmed), skip approval");
    println!(
        "- If the plan status is awaiting_confirmation (waiting for confirmation), approval is required"
    );
    println!("- If the commands are safe, skip approval");
    println!("- If the commands are sensitive or destructive, approval is required");

    Ok(())
}
