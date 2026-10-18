/// Simulates the `block_on` deadlock scenario and its solution
///
/// Problem: `ToolPolicy.evaluate_approval` is a synchronous method, but it needs to call async operations
///
/// How to run:
/// cargo run --example block_on_deadlock_demo
///
/// Conclusion:
/// - Using `handle.block_on()` will cause a deadlock or panic
/// - Solution: change `ToolPolicy.evaluate_approval` to async
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;

/// Mock PlanStore
struct MockPlanStore {
    plans: RwLock<std::collections::HashMap<String, String>>,
}

impl MockPlanStore {
    fn new() -> Self {
        let mut plans = std::collections::HashMap::new();
        plans.insert("plan-1".to_string(), "plan_data_1".to_string());

        Self {
            plans: RwLock::new(plans),
        }
    }

    async fn load_plan(&self, plan_id: &str) -> Option<String> {
        // Simulate async IO
        tokio::time::sleep(Duration::from_millis(100)).await;
        let plans = self.plans.read().await;
        plans.get(plan_id).cloned()
    }
}

/// The current ToolPolicy trait (synchronous)
trait SyncToolPolicy: Send + Sync {
    fn evaluate_approval(&self, tool_name: &str, plan_id: &str) -> Option<String>;
}

/// The improved ToolPolicy trait (asynchronous)
#[async_trait::async_trait]
trait AsyncToolPolicy: Send + Sync {
    async fn evaluate_approval(&self, tool_name: &str, plan_id: &str) -> Option<String>;
}

/// Approach 1: using block_on (problematic)
struct BlockOnPolicy {
    plan_store: Arc<MockPlanStore>,
}

impl SyncToolPolicy for BlockOnPolicy {
    fn evaluate_approval(&self, tool_name: &str, plan_id: &str) -> Option<String> {
        println!("[block_on] evaluate_approval: {} - {}", tool_name, plan_id);

        // Try to get the runtime handle
        let handle = match tokio::runtime::Handle::try_current() {
            Ok(h) => h,
            Err(_) => {
                println!("[block_on] Unable to get runtime handle");
                return None;
            }
        };

        // This will cause a panic: "Cannot start a runtime from within a runtime"
        println!("[block_on] Calling block_on...");
        let result = handle.block_on(self.plan_store.load_plan(plan_id));
        println!("[block_on] Done: {:?}", result);
        result
    }
}

/// Approach 2: using async trait (recommended)
struct AsyncPolicy {
    plan_store: Arc<MockPlanStore>,
}

#[async_trait::async_trait]
impl AsyncToolPolicy for AsyncPolicy {
    async fn evaluate_approval(&self, tool_name: &str, plan_id: &str) -> Option<String> {
        println!("[async] evaluate_approval: {} - {}", tool_name, plan_id);

        // Await directly; no deadlock risk
        let result = self.plan_store.load_plan(plan_id).await;
        println!("[async] Done: {:?}", result);
        result
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ToolPolicy Deadlock Problem Demo ===\n");

    let plan_store = Arc::new(MockPlanStore::new());

    // Test approach 2: async trait (recommended)
    println!("--- Approach 2: async trait (recommended) ---");
    let policy2 = AsyncPolicy {
        plan_store: plan_store.clone(),
    };
    match tokio::time::timeout(
        Duration::from_secs(2),
        policy2.evaluate_approval("execute_plan", "plan-1"),
    )
    .await
    {
        Ok(result) => println!("Approach 2 result: {:?}\n", result),
        Err(_) => println!("Approach 2 timed out!\n"),
    }

    // Test approach 1: block_on (will panic)
    println!("--- Approach 1: block_on (will panic) ---");
    println!("Note: will panic with 'Cannot start a runtime from within a runtime'\n");

    let policy1 = BlockOnPolicy {
        plan_store: plan_store.clone(),
    };

    // Use catch_unwind to catch the panic
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        policy1.evaluate_approval("execute_plan", "plan-1")
    }));

    match result {
        Ok(val) => println!("Approach 1 result: {:?}\n", val),
        Err(e) => {
            if let Some(msg) = e.downcast_ref::<String>() {
                println!("Approach 1 panic: {}\n", msg);
            } else if let Some(msg) = e.downcast_ref::<&str>() {
                println!("Approach 1 panic: {}\n", msg);
            } else {
                println!("Approach 1 panic: (unknown)\n");
            }
        }
    }

    println!("=== Conclusion ===");
    println!(
        "1. Using block_on to call async operations inside a synchronous function causes a panic"
    );
    println!("2. Error message: 'Cannot start a runtime from within a runtime'");
    println!("3. Solution: change ToolPolicy.evaluate_approval to async");
    println!();
    println!("=== Suggested trait improvement ===");
    println!("// Current (problematic):");
    println!("pub trait ToolPolicy: Send + Sync {{");
    println!(
        "    fn evaluate_approval(&self, tool_name: &str, args: &Value) -> Option<ApprovalRequest>;"
    );
    println!("}}");
    println!();
    println!("// Improved:");
    println!("#[async_trait]");
    println!("pub trait ToolPolicy: Send + Sync {{");
    println!(
        "    async fn evaluate_approval(&self, tool_name: &str, args: &Value) -> Option<ApprovalRequest>;"
    );
    println!("}}");

    Ok(())
}
