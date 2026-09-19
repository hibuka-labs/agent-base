use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{Mutex, RwLock};

use crate::engine::AgentSession;
use crate::engine::context::ContextWindowManager;
use crate::engine::session_store::SessionStore;
use crate::types::{
    AgentError, AgentResult, MessageRole, SessionConfig, SessionId, SessionIdGenerator,
};

#[derive(Clone)]
pub struct SessionManager {
    session_id_generator: Arc<dyn SessionIdGenerator>,
    sessions: Arc<RwLock<HashMap<SessionId, AgentSession>>>,
    /// Separate LRU timestamp map — protected by a lightweight Mutex,
    /// so session reads (RwLock::read) don't need exclusive access just to
    /// update the last-active time.
    lru_times: Arc<Mutex<HashMap<SessionId, Instant>>>,
    session_store: Arc<dyn SessionStore>,
    config: SessionConfig,
}

impl SessionManager {
    pub fn new(
        session_id_generator: Arc<dyn SessionIdGenerator>,
        session_store: Arc<dyn SessionStore>,
        config: SessionConfig,
    ) -> Self {
        Self {
            session_id_generator,
            sessions: Arc::new(RwLock::new(HashMap::new())),
            lru_times: Arc::new(Mutex::new(HashMap::new())),
            session_store,
            config,
        }
    }

    pub async fn create_session(&self, system_prompt: Option<&str>) -> SessionId {
        // Eviction is best-effort — if it fails, we still create the session
        if let Err(e) = self.evict_if_needed().await {
            tracing::warn!(error = %e, "session eviction failed, proceeding with creation");
        }

        let id = self.session_id_generator.generate();
        let mut session = AgentSession::new(id.clone());
        if let Some(prompt) = system_prompt {
            session.push_message(MessageRole::System, prompt);
        }
        {
            let mut sessions = self.sessions.write().await;
            sessions.insert(id.clone(), session);
        }
        {
            let mut lru = self.lru_times.lock().await;
            lru.insert(id.clone(), Instant::now());
        }
        tracing::debug!(session_id = id.id, "session created");
        id
    }

    pub async fn restore_session(&self, session_id: &SessionId) -> Option<AgentSession> {
        {
            let sessions = self.sessions.read().await;
            if sessions.contains_key(session_id) {
                let mut lru = self.lru_times.lock().await;
                lru.insert(session_id.clone(), Instant::now());
                tracing::debug!(session_id = session_id.id, "session restore cache hit");
                return sessions.get(session_id).cloned();
            }
        }
        match self.session_store.load(session_id).await {
            Ok(Some(session)) => {
                let msg_count = session.chat_messages().len();
                // Validate the restored message sequence — corrupt persisted data
                // would cause LLM API errors downstream, so warn early.
                if let Err(e) =
                    crate::engine::session::validate_message_sequence(session.chat_messages())
                {
                    tracing::warn!(session_id = session_id.id, error = %e, "restored session has invalid message sequence");
                }
                // Evict before inserting to make room if needed
                self.evict_if_needed().await.ok();
                {
                    let mut sessions = self.sessions.write().await;
                    sessions.insert(session_id.clone(), session.clone());
                }
                {
                    let mut lru = self.lru_times.lock().await;
                    lru.insert(session_id.clone(), Instant::now());
                }
                tracing::debug!(
                    session_id = session_id.id,
                    msg_count,
                    "session restored from store"
                );
                Some(session)
            }
            Ok(None) => {
                tracing::debug!(session_id = session_id.id, "session not found in store");
                None
            }
            Err(e) => {
                tracing::warn!(session_id = session_id.id, error = %e, "session restore failed");
                None
            }
        }
    }

    pub async fn session(&self, session_id: &SessionId) -> Option<AgentSession> {
        let sessions = self.sessions.read().await;
        let result = sessions.get(session_id).cloned();
        if result.is_some() {
            let mut lru = self.lru_times.lock().await;
            lru.insert(session_id.clone(), Instant::now());
        }
        result
    }

    pub async fn session_or_err(&self, session_id: &SessionId) -> AgentResult<AgentSession> {
        let sessions = self.sessions.read().await;
        let result = sessions
            .get(session_id)
            .cloned()
            .ok_or_else(|| AgentError::session_not_found(session_id.id));
        if result.is_ok() {
            let mut lru = self.lru_times.lock().await;
            lru.insert(session_id.clone(), Instant::now());
        }
        result
    }

    pub async fn with_session_mut<F, R>(&self, session_id: &SessionId, f: F) -> AgentResult<R>
    where
        F: FnOnce(&mut AgentSession) -> R,
    {
        // The write lock on `sessions` is held only for the closure execution.
        // It MUST be dropped before `enforce_session_limits` — that method may
        // call `save_session` → `session_or_err` which re-acquires a (read) lock
        // on the same `sessions` map. Holding the write lock across that call
        // would deadlock.
        let result = {
            let mut sessions = self.sessions.write().await;
            let session = sessions
                .get_mut(session_id)
                .ok_or_else(|| AgentError::session_not_found(session_id.id))?;
            f(session)
        };
        // Update LRU timestamp (lightweight Mutex, not RwLock)
        {
            let mut lru = self.lru_times.lock().await;
            lru.insert(session_id.clone(), Instant::now());
        }
        self.enforce_session_limits(session_id).await;
        Ok(result)
    }

    pub async fn cached_approval(&self, session_id: &SessionId, action_key: &str) -> bool {
        let sessions = self.sessions.read().await;
        sessions
            .get(session_id)
            .is_some_and(|session| session.is_action_allowed(action_key))
    }

    pub async fn cache_approval(&self, session_id: &SessionId, action_key: String) {
        let mut sessions = self.sessions.write().await;
        if let Some(session) = sessions.get_mut(session_id) {
            session.allow_action(action_key);
        } else {
            // `get_mut` returns None when the session doesn't exist yet — the
            // approval cache is silently a no-op in that case. Surface it so a
            // caller caching before session creation isn't left wondering why
            // approval was re-prompted.
            tracing::warn!(
                session_id = session_id.id,
                action = %action_key,
                "cache_approval ignored: session not found"
            );
        }
    }

    pub async fn save_session(&self, session_id: &SessionId) -> AgentResult<()> {
        let session = self.session_or_err(session_id).await?;
        let msg_count = session.chat_messages().len();
        tracing::debug!(session_id = session_id.id, msg_count, "saving session");
        self.session_store
            .save(&session)
            .await
            .map_err(|e| AgentError::internal(format!("Session persistence failed: {e}")))
    }

    pub fn session_store(&self) -> &Arc<dyn SessionStore> {
        &self.session_store
    }

    /// Evict the least recently used session if at capacity.
    ///
    /// Note: there is a benign TOCTOU window between the read-lock capacity
    /// check and the write-lock removal — another task may concurrently create
    /// a session and also trigger eviction. The worst case is one extra eviction,
    /// which is harmless since `restore_session` transparently reloads from store.
    async fn evict_if_needed(&self) -> AgentResult<()> {
        let max = match self.config.max_sessions {
            Some(m) => m,
            None => return Ok(()),
        };

        // Find the LRU victim (read lock on sessions, Mutex on lru_times)
        let victim = {
            let sessions = self.sessions.read().await;
            if sessions.len() < max {
                return Ok(());
            }
            let lru = self.lru_times.lock().await;
            sessions
                .keys()
                .min_by_key(|id| lru.get(*id).copied().unwrap_or(Instant::now()))
                .cloned()
        };

        let Some(victim_id) = victim else {
            return Ok(());
        };

        // Persist before evicting
        if let Err(e) = self.save_session(&victim_id).await {
            tracing::warn!(session_id = victim_id.id, error = %e, "failed to persist session before eviction");
        }

        // Remove from both maps
        {
            let mut sessions = self.sessions.write().await;
            sessions.remove(&victim_id);
        }
        {
            let mut lru = self.lru_times.lock().await;
            lru.remove(&victim_id);
        }
        tracing::info!(session_id = victim_id.id, "session evicted (LRU)");

        Ok(())
    }

    /// Enforce turn count and message size limits after a session mutation.
    async fn enforce_session_limits(&self, session_id: &SessionId) {
        // Layer 2: Turn count trimming
        if let Some(max_turns) = self.config.max_turns_per_session {
            let needs_trim = {
                let sessions = self.sessions.read().await;
                sessions
                    .get(session_id)
                    .is_some_and(|session| session.turn_count() > max_turns)
            };

            if needs_trim {
                // Persist full history before trimming
                if let Err(e) = self.save_session(session_id).await {
                    tracing::warn!(session_id = session_id.id, error = %e, "failed to persist before turn trim");
                }
                // Trim under write lock
                let mut sessions = self.sessions.write().await;
                if let Some(session) = sessions.get_mut(session_id) {
                    let before = session.turn_count();
                    session.trim_oldest_turns(max_turns);
                    tracing::info!(
                        session_id = session_id.id,
                        before,
                        after = session.turn_count(),
                        max_turns,
                        "session turns trimmed"
                    );
                }
            }
        }

        // Layer 3: Oversized message safety valve
        if let Some(max_tokens) = self.config.max_message_tokens {
            let mut sessions = self.sessions.write().await;
            if let Some(session) = sessions.get_mut(session_id)
                && let Some(last) = session.chat_messages().last()
            {
                let tokens = ContextWindowManager::message_tokens(last);
                if tokens > max_tokens {
                    session.pop_last_message();
                    tracing::warn!(
                        session_id = session_id.id,
                        tokens,
                        max_tokens,
                        "oversized message removed from session (safety valve)"
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::InMemorySessionStore;
    use crate::types::{AtomicU64SessionIdGenerator, ChatMessage};
    use std::sync::Arc;

    fn manager() -> SessionManager {
        SessionManager::new(
            Arc::new(AtomicU64SessionIdGenerator::default()),
            Arc::new(InMemorySessionStore::new()),
            SessionConfig::default(),
        )
    }

    fn manager_with_config(config: SessionConfig) -> SessionManager {
        SessionManager::new(
            Arc::new(AtomicU64SessionIdGenerator::default()),
            Arc::new(InMemorySessionStore::new()),
            config,
        )
    }

    struct FailingStore;

    #[async_trait::async_trait]
    impl SessionStore for FailingStore {
        async fn save(&self, _session: &AgentSession) -> AgentResult<()> {
            Ok(())
        }
        async fn load(&self, _session_id: &SessionId) -> AgentResult<Option<AgentSession>> {
            Err(AgentError::internal("boom"))
        }
        async fn list(&self) -> AgentResult<Vec<SessionId>> {
            Ok(vec![])
        }
        async fn delete(&self, _session_id: &SessionId) -> AgentResult<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn create_session_with_system_prompt() {
        let m = manager();
        let id = m.create_session(Some("be helpful")).await;
        assert_eq!(id.id, 1);

        let session = m.session(&id).await.unwrap();
        assert_eq!(session.chat_messages().len(), 1);
        assert!(matches!(
            session.chat_messages()[0],
            ChatMessage::System { .. }
        ));
    }

    #[tokio::test]
    async fn create_session_without_prompt_is_empty() {
        let m = manager();
        let id = m.create_session(None).await;
        let session = m.session(&id).await.unwrap();
        assert!(session.chat_messages().is_empty());
    }

    #[tokio::test]
    async fn session_returns_none_for_unknown() {
        let m = manager();
        assert!(m.session(&SessionId::new(999)).await.is_none());
    }

    #[tokio::test]
    async fn session_or_err_returns_error_for_unknown() {
        let m = manager();
        let err = m.session_or_err(&SessionId::new(999)).await.unwrap_err();
        assert!(matches!(err, AgentError::SessionNotFound(_)));
    }

    #[tokio::test]
    async fn restore_session_cache_hit() {
        let m = manager();
        let id = m.create_session(Some("sys")).await;
        let restored = m.restore_session(&id).await;
        assert!(restored.is_some());
        assert_eq!(restored.unwrap().chat_messages().len(), 1);
    }

    #[tokio::test]
    async fn restore_session_from_store() {
        let store = Arc::new(InMemorySessionStore::new());
        let mut session = AgentSession::new(SessionId::new(42));
        session.push_message(MessageRole::User, "persisted");
        store.save(&session).await.unwrap();

        let m = SessionManager::new(
            Arc::new(AtomicU64SessionIdGenerator::default()),
            store.clone(),
            SessionConfig::default(),
        );
        let restored = m.restore_session(&SessionId::new(42)).await;
        assert!(restored.is_some());
        assert_eq!(restored.unwrap().chat_messages().len(), 1);
        // now cached in memory
        assert!(m.session(&SessionId::new(42)).await.is_some());
    }

    #[tokio::test]
    async fn restore_session_returns_none_when_not_found() {
        let m = manager();
        assert!(m.restore_session(&SessionId::new(999)).await.is_none());
    }

    #[tokio::test]
    async fn restore_session_returns_none_on_store_error() {
        let m = SessionManager::new(
            Arc::new(AtomicU64SessionIdGenerator::default()),
            Arc::new(FailingStore),
            SessionConfig::default(),
        );
        assert!(m.restore_session(&SessionId::new(1)).await.is_none());
    }

    #[tokio::test]
    async fn with_session_mut_applies_closure() {
        let m = manager();
        let id = m.create_session(None).await;
        m.with_session_mut(&id, |s| {
            s.push_message(MessageRole::User, "hello");
            s.push_message(MessageRole::Assistant, "hi");
        })
        .await
        .unwrap();

        let session = m.session(&id).await.unwrap();
        assert_eq!(session.chat_messages().len(), 2);
    }

    #[tokio::test]
    async fn with_session_mut_errors_for_unknown() {
        let m = manager();
        let err = m
            .with_session_mut(&SessionId::new(999), |_s| ())
            .await
            .unwrap_err();
        assert!(matches!(err, AgentError::SessionNotFound(_)));
    }

    #[tokio::test]
    async fn approval_cache_roundtrip() {
        let m = manager();
        let id = m.create_session(None).await;
        assert!(!m.cached_approval(&id, "read_file").await);

        m.cache_approval(&id, "read_file".into()).await;
        assert!(m.cached_approval(&id, "read_file").await);
    }

    /// D4 不变式：AllowAlways 缓存键控在 (session_id, action_key)——子 agent
    /// 的 "always" 授权绝不泄漏到父/兄弟 session（子 agent 持独立 session_id）。
    #[tokio::test]
    async fn approval_cache_is_scoped_per_session() {
        let m = manager();
        let parent = m.create_session(None).await;
        let child = m.create_session(None).await;
        m.cache_approval(&child, "write_file:src/x.rs".to_string())
            .await;
        assert!(
            m.cached_approval(&child, "write_file:src/x.rs").await,
            "同 session 命中"
        );
        assert!(
            !m.cached_approval(&parent, "write_file:src/x.rs").await,
            "跨 session 必须未命中（子 always 不影响父）"
        );
        let sibling = m.create_session(None).await;
        assert!(
            !m.cached_approval(&sibling, "write_file:src/x.rs").await,
            "跨 session 必须未命中（子 always 不影响兄弟）"
        );
    }

    #[tokio::test]
    async fn save_session_persists_to_store() {
        let store = Arc::new(InMemorySessionStore::new());
        let m = SessionManager::new(
            Arc::new(AtomicU64SessionIdGenerator::default()),
            store.clone(),
            SessionConfig::default(),
        );
        let id = m.create_session(Some("sys")).await;
        m.with_session_mut(&id, |s| s.push_message(MessageRole::User, "hello"))
            .await
            .unwrap();
        m.save_session(&id).await.unwrap();

        let loaded = store.load(&id).await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().chat_messages().len(), 2);
    }

    #[tokio::test]
    async fn save_session_errors_for_unknown() {
        let m = manager();
        let err = m.save_session(&SessionId::new(999)).await.unwrap_err();
        assert!(matches!(err, AgentError::SessionNotFound(_)));
    }

    #[tokio::test]
    async fn session_store_getter_returns_store() {
        let m = manager();
        assert!(m.session_store().list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn eviction_evicts_lru_when_at_capacity() {
        let cfg = SessionConfig {
            max_sessions: Some(2),
            ..Default::default()
        };
        let m = manager_with_config(cfg);
        let id1 = m.create_session(Some("s1")).await;
        let id2 = m.create_session(Some("s2")).await;
        let id3 = m.create_session(Some("s3")).await;

        assert!(m.session(&id1).await.is_none()); // LRU evicted
        assert!(m.session(&id2).await.is_some());
        assert!(m.session(&id3).await.is_some());
    }

    #[tokio::test]
    async fn turn_trimming_enforced_after_mutation() {
        let cfg = SessionConfig {
            max_turns_per_session: Some(1),
            ..Default::default()
        };
        let m = manager_with_config(cfg);
        let id = m.create_session(None).await;
        m.with_session_mut(&id, |s| {
            s.push_message(MessageRole::User, "u1");
            s.push_message(MessageRole::Assistant, "a1");
            s.push_message(MessageRole::User, "u2");
            s.push_message(MessageRole::Assistant, "a2");
        })
        .await
        .unwrap();

        let session = m.session(&id).await.unwrap();
        assert_eq!(session.turn_count(), 1);
    }

    #[tokio::test]
    async fn oversized_message_removed_after_mutation() {
        let cfg = SessionConfig {
            max_message_tokens: Some(10),
            ..Default::default()
        };
        let m = manager_with_config(cfg);
        let id = m.create_session(None).await;
        let big = "x".repeat(200);
        m.with_session_mut(&id, |s| s.push_message(MessageRole::User, big))
            .await
            .unwrap();

        let session = m.session(&id).await.unwrap();
        assert!(session.chat_messages().is_empty());
    }
}
