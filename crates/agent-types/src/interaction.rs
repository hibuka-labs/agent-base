//! Interaction-channel pure types: the ask half of the `ExecCtx` suspension
//! vocabulary (the consent half is [`crate::approval`]).
//!
//! One vocabulary across the seam: the product's `ask_user` picker (question +
//! candidates → the user's pick) and a code cell's host-call ask ride these
//! shapes, so the channel face and the product surface cannot drift apart.

use serde::{Deserialize, Serialize};

/// One candidate the user can pick. `id` is the caller's stable key (echoed
/// back on pick); `label` is what the picker shows and what comes back with
/// the answer.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub struct AskOption {
    pub id: String,
    pub label: String,
    pub note: Option<String>,
}

/// What the user did with an ask.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
pub enum AskAnswer {
    /// Picked a candidate — `id` echoed, `label` as shown.
    Picked { id: String, label: String },
    /// Declined to pick (Esc). A skip is an answer too; never re-ask.
    Skipped,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The ask vocabulary is wire data (a cell's host-call answer is JSON) —
    /// it must round-trip without loss.
    #[test]
    fn ask_vocabulary_round_trips_through_json() {
        let option = AskOption {
            id: "login".into(),
            label: "登录".into(),
            note: Some("先打通会话".into()),
        };
        let back: AskOption =
            serde_json::from_str(&serde_json::to_string(&option).unwrap()).unwrap();
        assert_eq!(back, option);

        for answer in [
            AskAnswer::Picked {
                id: "login".into(),
                label: "登录".into(),
            },
            AskAnswer::Skipped,
        ] {
            let json = serde_json::to_string(&answer).unwrap();
            let back: AskAnswer = serde_json::from_str(&json).unwrap();
            assert_eq!(back, answer);
        }
    }
}
