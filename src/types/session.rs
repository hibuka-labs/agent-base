use std::sync::{Arc, OnceLock};

pub use agent_types::SessionId;

pub trait SessionIdGenerator: Send + Sync {
    fn generate(&self) -> SessionId;
}

pub struct AtomicU64SessionIdGenerator {
    counter: std::sync::atomic::AtomicU64,
}

impl AtomicU64SessionIdGenerator {
    pub fn new(start: u64) -> Self {
        Self {
            counter: std::sync::atomic::AtomicU64::new(start),
        }
    }
}

impl Default for AtomicU64SessionIdGenerator {
    fn default() -> Self {
        Self::new(1)
    }
}

impl SessionIdGenerator for AtomicU64SessionIdGenerator {
    fn generate(&self) -> SessionId {
        SessionId::new(
            self.counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed),
        )
    }
}

/// Process-global numbering for engines that bring no generator of their own.
///
/// One process routinely runs several engines — a host and its sub-agents —
/// and each engine that minted its own counter started over at 1, so all of
/// them held a "session 1" and tool calls / metrics could not tell the engines
/// apart. Engines built without a generator draw from this one counter, so
/// ids are unique within the process.
pub fn shared_session_ids() -> Arc<dyn SessionIdGenerator> {
    static SHARED: OnceLock<Arc<AtomicU64SessionIdGenerator>> = OnceLock::new();
    SHARED
        .get_or_init(|| Arc::new(AtomicU64SessionIdGenerator::default()))
        .clone()
}

pub struct UuidSessionIdGenerator;

impl SessionIdGenerator for UuidSessionIdGenerator {
    fn generate(&self) -> SessionId {
        SessionId::with_external_id(0, uuid::Uuid::new_v4().to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_id_display_with_and_without_external_id() {
        assert_eq!(SessionId::new(42).to_string(), "42");
        assert_eq!(
            SessionId::with_external_id(42, "ext").to_string(),
            "42(ext)"
        );
    }

    #[test]
    fn atomic_generator_increments() {
        let g = AtomicU64SessionIdGenerator::new(10);
        assert_eq!(g.generate().id, 10);
        assert_eq!(g.generate().id, 11);

        let g = AtomicU64SessionIdGenerator::default();
        assert_eq!(g.generate().id, 1);
    }

    #[test]
    fn uuid_generator_sets_external_id() {
        let sid = UuidSessionIdGenerator.generate();
        assert_eq!(sid.id, 0);
        assert!(sid.external_id.is_some());
    }

    #[test]
    fn shared_numbering_distinguishes_engines_in_one_process() {
        // Two engines, no generator of their own. Per-engine counters handed
        // both a "1"; the shared counter must not collide.
        let a = shared_session_ids();
        let b = shared_session_ids();
        assert_ne!(a.generate().id, b.generate().id);
    }
}
