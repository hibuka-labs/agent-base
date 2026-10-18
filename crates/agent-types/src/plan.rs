//! Plan-related pure types: PlanItem, PlanStepStatus, UpdatePlanArgs.

use serde::{Deserialize, Serialize};

/// Lightweight plan step status — display-only, no execution semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlanStepStatus {
    Pending,
    InProgress,
    Completed,
}

/// A single step in a lightweight plan checklist.
///
/// Contains only human-readable display text and a status.
/// Does NOT carry tool names, command payloads, host info, or dependency graphs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanItem {
    pub step: String,
    pub status: PlanStepStatus,
}

/// Arguments for the `update_plan` tool.
///
/// This is a lightweight progress-display protocol — the tool broadcasts a
/// structured snapshot to the UI and does nothing else.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdatePlanArgs {
    /// A one-sentence summary of the user's goal.
    /// Optional on subsequent calls — the tool remembers the last objective.
    /// Example: "Install the Casdoor authentication system"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub objective: Option<String>,
    /// Optional explanation of why the plan changed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
    /// The full plan checklist (replaces any previous display).
    pub plan: Vec<PlanItem>,
}

impl UpdatePlanArgs {
    /// Validate the plan arguments.
    ///
    /// Rules:
    /// - `objective`, if provided, must be non-empty after trimming and at most 200 chars.
    /// - `plan` must contain 1–50 steps.
    /// - Each `step` must be non-empty.
    /// - At most one step may be `InProgress`.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(ref objective) = self.objective {
            let objective_trimmed = objective.trim();
            if objective_trimmed.is_empty() {
                return Err("objective must not be empty when provided".to_string());
            }
            if objective_trimmed.chars().count() > 200 {
                return Err(format!(
                    "objective must be at most 200 characters, got {}",
                    objective_trimmed.chars().count()
                ));
            }
        }
        if self.plan.is_empty() {
            return Err("plan must contain at least one step".to_string());
        }
        if self.plan.len() > 50 {
            return Err(format!(
                "plan must contain at most 50 steps, got {}",
                self.plan.len()
            ));
        }

        let in_progress_count = self
            .plan
            .iter()
            .filter(|item| item.status == PlanStepStatus::InProgress)
            .count();

        if in_progress_count > 1 {
            return Err(format!(
                "at most one step may be in_progress, found {}",
                in_progress_count
            ));
        }

        for (i, item) in self.plan.iter().enumerate() {
            if item.step.trim().is_empty() {
                return Err(format!("step[{}] must not be empty", i));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optional_objective_deserialization() {
        // objective present — should parse fine
        let json =
            r#"{"objective": "Install Docker", "plan": [{"step": "Step 1", "status": "pending"}]}"#;
        let args: UpdatePlanArgs = serde_json::from_str(json).unwrap();
        assert_eq!(args.objective.as_deref(), Some("Install Docker"));
        assert!(args.validate().is_ok());

        // objective absent — should parse with None
        let json = r#"{"plan": [{"step": "Step 1", "status": "pending"}]}"#;
        let args: UpdatePlanArgs = serde_json::from_str(json).unwrap();
        assert!(args.objective.is_none());
        assert!(args.validate().is_ok());

        // objective present but empty — should fail validation
        let json = r#"{"objective": "  ", "plan": [{"step": "Step 1", "status": "pending"}]}"#;
        let args: UpdatePlanArgs = serde_json::from_str(json).unwrap();
        assert!(args.validate().is_err());
    }
}

#[cfg(test)]
mod proptest_tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn validate_rejects_empty_plan() {
        let args = UpdatePlanArgs {
            objective: None,
            explanation: None,
            plan: vec![],
        };
        assert!(args.validate().is_err());
    }

    proptest! {
        #[test]
        fn validate_accepts_1_to_50_steps(count in 1usize..=50) {
            let plan: Vec<PlanItem> = (0..count)
                .map(|i| PlanItem {
                    step: format!("step {}", i),
                    status: PlanStepStatus::Pending,
                })
                .collect();
            let args = UpdatePlanArgs {
                objective: None,
                explanation: None,
                plan,
            };
            assert!(args.validate().is_ok());
        }

        #[test]
        fn validate_rejects_more_than_50_steps(count in 51usize..100) {
            let plan: Vec<PlanItem> = (0..count)
                .map(|i| PlanItem {
                    step: format!("step {}", i),
                    status: PlanStepStatus::Pending,
                })
                .collect();
            let args = UpdatePlanArgs {
                objective: None,
                explanation: None,
                plan,
            };
            assert!(args.validate().is_err());
        }

        #[test]
        fn validate_rejects_empty_step_text(step_text in r"[ \t\r\n]{0,20}") {
            let args = UpdatePlanArgs {
                objective: None,
                explanation: None,
                plan: vec![PlanItem {
                    step: step_text,
                    status: PlanStepStatus::Pending,
                }],
            };
            assert!(args.validate().is_err());
        }

        #[test]
        fn validate_objective_length_boundary(len in 0usize..300) {
            let objective = "x".repeat(len);
            let args = UpdatePlanArgs {
                objective: Some(objective),
                explanation: None,
                plan: vec![PlanItem {
                    step: "do something".into(),
                    status: PlanStepStatus::Pending,
                }],
            };
            if len == 0 {
                // Empty after trim → fails
                assert!(args.validate().is_err());
            } else if len <= 200 {
                assert!(args.validate().is_ok());
            } else {
                // Over 200 chars → fails
                assert!(args.validate().is_err());
            }
        }
    }
}
