use std::collections::HashMap;

use async_trait::async_trait;
use tokio::sync::Mutex;

use super::AgentSession;
use crate::types::{AgentError, AgentResult, RuntimeEvent, SessionId};

/// Session Persistence Adapter
///
/// `SessionStore` is an optional persistence interface for agent sessions.
/// Under the lightweight kernel design:
/// - `AgentRuntime.sessions` is the authoritative live state during execution
/// - `SessionStore` is a persistence adapter for save/load/list/delete
/// - Does not participate in the execution control flow
///
/// Replace the default [`InMemorySessionStore`] with a custom implementation
/// to persist sessions to a database, filesystem, or other storage.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Save a session snapshot to the persistence layer
    async fn save(&self, session: &AgentSession) -> AgentResult<()>;

    /// Load a session from the persistence layer
    async fn load(&self, session_id: &SessionId) -> AgentResult<Option<AgentSession>>;

    /// List all saved session IDs
    async fn list(&self) -> AgentResult<Vec<SessionId>>;

    /// Delete a specific session
    async fn delete(&self, session_id: &SessionId) -> AgentResult<()>;

    /// Optionally persist a runtime event for audit/replay.
    /// Default implementation is a no-op — override to implement
    /// append-log style persistence (e.g. JSONL file, database event log).
    async fn append_event(
        &self,
        _session_id: &SessionId,
        _event: &RuntimeEvent,
    ) -> AgentResult<()> {
        Ok(())
    }
}

pub struct InMemorySessionStore {
    sessions: Mutex<HashMap<SessionId, AgentSession>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for InMemorySessionStore {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn save(&self, session: &AgentSession) -> AgentResult<()> {
        let session_id = session
            .id()
            .ok_or_else(|| AgentError::internal("session has no id"))?;
        self.sessions
            .lock()
            .await
            .insert(session_id, session.clone());
        Ok(())
    }

    async fn load(&self, session_id: &SessionId) -> AgentResult<Option<AgentSession>> {
        Ok(self.sessions.lock().await.get(session_id).cloned())
    }

    async fn list(&self) -> AgentResult<Vec<SessionId>> {
        Ok(self.sessions.lock().await.keys().cloned().collect())
    }

    async fn delete(&self, session_id: &SessionId) -> AgentResult<()> {
        self.sessions.lock().await.remove(session_id);
        Ok(())
    }
}

// ── SqliteSessionStore ──

#[cfg(feature = "sqlite-session")]
use std::sync::Mutex as StdMutex;

/// SQLite-backed session persistence.
///
/// Stores each [`AgentSession`] as a JSON blob in a single `sessions` table.
/// Enable with `features = ["sqlite-session"]` in your `Cargo.toml`.
///
/// # Example
///
/// ```ignore
/// use agent_base::engine::SqliteSessionStore;
/// let store = SqliteSessionStore::open("sessions.db").unwrap();
/// ```
#[cfg(feature = "sqlite-session")]
pub struct SqliteSessionStore {
    db: StdMutex<rusqlite::Connection>,
}

#[cfg(feature = "sqlite-session")]
impl SqliteSessionStore {
    /// Open (or create) a SQLite database at `path` and initialize the schema.
    pub fn open(path: impl AsRef<std::path::Path>) -> AgentResult<Self> {
        let conn = rusqlite::Connection::open(path)
            .map_err(|e| AgentError::internal(format!("sqlite open: {e}")))?;
        Self::init_tables(&conn)?;
        Ok(Self {
            db: StdMutex::new(conn),
        })
    }

    /// Build from an already-open [`rusqlite::Connection`].
    ///
    /// The connection must outlive the store. Caller is responsible for closing it.
    pub fn from_connection(conn: rusqlite::Connection) -> AgentResult<Self> {
        Self::init_tables(&conn)?;
        Ok(Self {
            db: StdMutex::new(conn),
        })
    }

    fn init_tables(conn: &rusqlite::Connection) -> AgentResult<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id   TEXT PRIMARY KEY,
                data TEXT NOT NULL
            );",
        )
        .map_err(|e| AgentError::internal(format!("sqlite init: {e}")))
    }

    fn session_key(id: &SessionId) -> String {
        // Use serde_json for robust round-tripping — never depend on Display format.
        serde_json::to_string(id).unwrap_or_else(|_| id.to_string())
    }
}

#[cfg(feature = "sqlite-session")]
#[async_trait]
impl SessionStore for SqliteSessionStore {
    async fn save(&self, session: &AgentSession) -> AgentResult<()> {
        let session_id = session
            .id()
            .ok_or_else(|| AgentError::internal("session has no id"))?;
        let key = Self::session_key(&session_id);
        let data = serde_json::to_string(session).map_err(|e| AgentError::json(e.to_string()))?;
        let db = self.db.lock().unwrap_or_else(|e| e.into_inner());
        db.execute(
            "INSERT OR REPLACE INTO sessions (id, data) VALUES (?1, ?2)",
            rusqlite::params![key, data],
        )
        .map_err(|e| AgentError::internal(format!("sqlite save: {e}")))?;
        Ok(())
    }

    async fn load(&self, session_id: &SessionId) -> AgentResult<Option<AgentSession>> {
        let key = Self::session_key(session_id);
        let db = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = db
            .prepare("SELECT data FROM sessions WHERE id = ?1")
            .map_err(|e| AgentError::internal(format!("sqlite prepare: {e}")))?;
        let result: Result<String, rusqlite::Error> =
            stmt.query_row(rusqlite::params![key], |row| row.get(0));
        match result {
            Ok(data) => {
                let session: AgentSession = serde_json::from_str(&data)
                    .map_err(|e| AgentError::json(format!("deserialize session: {e}")))?;
                Ok(Some(session))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(AgentError::internal(format!("sqlite load: {e}"))),
        }
    }

    async fn list(&self) -> AgentResult<Vec<SessionId>> {
        let db = self.db.lock().unwrap_or_else(|e| e.into_inner());
        let mut stmt = db
            .prepare("SELECT id FROM sessions ORDER BY id")
            .map_err(|e| AgentError::internal(format!("sqlite prepare: {e}")))?;
        let rows = stmt
            .query_map([], |row| row.get::<_, String>(0))
            .map_err(|e| AgentError::internal(format!("sqlite list: {e}")))?;
        let mut ids = Vec::new();
        for row in rows {
            let id_str = row.map_err(|e| AgentError::internal(format!("sqlite list row: {e}")))?;
            // Keys are serialized with serde_json — deserialize for robust round-tripping.
            match serde_json::from_str::<SessionId>(&id_str) {
                Ok(sid) => ids.push(sid),
                Err(e) => {
                    tracing::warn!(key = id_str, error = %e, "failed to deserialize session key, skipping");
                }
            }
        }
        Ok(ids)
    }

    async fn delete(&self, session_id: &SessionId) -> AgentResult<()> {
        let key = Self::session_key(session_id);
        let db = self.db.lock().unwrap_or_else(|e| e.into_inner());
        db.execute("DELETE FROM sessions WHERE id = ?1", rusqlite::params![key])
            .map_err(|e| AgentError::internal(format!("sqlite delete: {e}")))?;
        Ok(())
    }
}

#[cfg(feature = "sqlite-session")]
impl std::fmt::Debug for SqliteSessionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteSessionStore").finish_non_exhaustive()
    }
}

// ── Tests ──

#[cfg(test)]
#[cfg(feature = "sqlite-session")]
mod sqlite_tests {
    use super::*;

    fn make_store() -> SqliteSessionStore {
        let conn = rusqlite::Connection::open_in_memory().expect("open in-memory sqlite");
        SqliteSessionStore::from_connection(conn).expect("init tables")
    }

    fn make_session(id: u64) -> AgentSession {
        let mut s = AgentSession::new(SessionId::new(id));
        s.push_message(crate::types::MessageRole::User, "hello");
        s.push_message(crate::types::MessageRole::Assistant, "hi");
        s
    }

    #[tokio::test]
    async fn save_and_load() {
        let store = make_store();
        let session = make_session(1);
        store.save(&session).await.unwrap();

        let loaded = store.load(&SessionId::new(1)).await.unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.chat_messages().len(), 2);
    }

    #[tokio::test]
    async fn load_nonexistent_returns_none() {
        let store = make_store();
        let result = store.load(&SessionId::new(999)).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn save_overwrites() {
        let store = make_store();
        let mut session = make_session(1);
        store.save(&session).await.unwrap();

        session.push_message(crate::types::MessageRole::User, "another");
        store.save(&session).await.unwrap();

        let loaded = store.load(&SessionId::new(1)).await.unwrap().unwrap();
        assert_eq!(loaded.chat_messages().len(), 3);
    }

    #[tokio::test]
    async fn list_sessions() {
        let store = make_store();
        store.save(&make_session(1)).await.unwrap();
        store.save(&make_session(2)).await.unwrap();
        store.save(&make_session(3)).await.unwrap();

        let ids = store.list().await.unwrap();
        assert_eq!(ids.len(), 3);
    }

    #[tokio::test]
    async fn list_empty() {
        let store = make_store();
        let ids = store.list().await.unwrap();
        assert!(ids.is_empty());
    }

    #[tokio::test]
    async fn delete_session() {
        let store = make_store();
        store.save(&make_session(1)).await.unwrap();
        assert!(store.load(&SessionId::new(1)).await.unwrap().is_some());

        store.delete(&SessionId::new(1)).await.unwrap();
        assert!(store.load(&SessionId::new(1)).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_nonexistent_is_noop() {
        let store = make_store();
        // Should not error
        store.delete(&SessionId::new(999)).await.unwrap();
    }

    #[tokio::test]
    async fn save_session_with_external_id() {
        let store = make_store();
        let mut s = AgentSession::new(SessionId::with_external_id(42, "my-ext-id"));
        s.push_message(crate::types::MessageRole::User, "test");
        store.save(&s).await.unwrap();

        let loaded = store
            .load(&SessionId::with_external_id(42, "my-ext-id"))
            .await
            .unwrap();
        assert!(loaded.is_some());

        let ids = store.list().await.unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0].to_string(), "42(my-ext-id)");
    }

    #[tokio::test]
    async fn save_without_id_errors() {
        let store = make_store();
        let session = AgentSession::default(); // id is None
        let err = store.save(&session).await.unwrap_err();
        assert!(err.to_string().contains("no id"));
    }

    #[tokio::test]
    async fn append_event_default_noop() {
        let store = make_store();
        // Default append_event should not error
        store
            .append_event(
                &SessionId::new(1),
                &RuntimeEvent::UserEvent {
                    session_id: SessionId::new(1),
                    event: crate::types::UserEvent::Progress {
                        text: "test".into(),
                    },
                    agent_id: None,
                    trace_id: None,
                },
            )
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn roundtrip_preserves_fields() {
        let store = make_store();
        let mut session = AgentSession::new(SessionId::new(7));
        session.push_message(crate::types::MessageRole::System, "system prompt");
        session.push_message(crate::types::MessageRole::User, "question");
        session.push_message(crate::types::MessageRole::Assistant, "answer");
        session.allow_action("read_file");
        session.run_state.total_actions = 3;

        store.save(&session).await.unwrap();
        let loaded = store.load(&SessionId::new(7)).await.unwrap().unwrap();

        assert_eq!(loaded.chat_messages().len(), 3);
        assert!(loaded.is_action_allowed("read_file"));
        assert_eq!(loaded.run_state.total_actions, 3);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_session(id: u64) -> AgentSession {
        let mut s = AgentSession::new(SessionId::new(id));
        s.push_message(crate::types::MessageRole::User, "hello");
        s.push_message(crate::types::MessageRole::Assistant, "hi");
        s
    }

    #[tokio::test]
    async fn save_and_load_roundtrips_fields() {
        let store = InMemorySessionStore::new();
        let mut session = make_session(1);
        session.allow_action("read_file");
        session.run_state.total_actions = 2;
        store.save(&session).await.unwrap();

        let loaded = store.load(&SessionId::new(1)).await.unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.id(), Some(SessionId::new(1)));
        assert_eq!(loaded.chat_messages().len(), 2);
        assert!(loaded.is_action_allowed("read_file"));
        assert_eq!(loaded.run_state.total_actions, 2);
    }

    #[tokio::test]
    async fn load_nonexistent_returns_none() {
        let store = InMemorySessionStore::new();
        let result = store.load(&SessionId::new(999)).await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn list_returns_all_saved_ids() {
        let store = InMemorySessionStore::new();
        store.save(&make_session(1)).await.unwrap();
        store.save(&make_session(2)).await.unwrap();
        store.save(&make_session(3)).await.unwrap();

        let ids = store.list().await.unwrap();
        assert_eq!(ids.len(), 3);
        let mut sorted = ids.clone();
        sorted.sort_by_key(|id| id.id);
        assert_eq!(
            sorted.iter().map(|id| id.id).collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[tokio::test]
    async fn delete_removes_session() {
        let store = InMemorySessionStore::new();
        store.save(&make_session(1)).await.unwrap();
        assert!(store.load(&SessionId::new(1)).await.unwrap().is_some());

        store.delete(&SessionId::new(1)).await.unwrap();
        assert!(store.load(&SessionId::new(1)).await.unwrap().is_none());
        assert!(store.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn delete_nonexistent_is_noop() {
        let store = InMemorySessionStore::new();
        // Deleting an unknown session must succeed without error.
        store.delete(&SessionId::new(999)).await.unwrap();
        assert!(store.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn save_overwrites_existing_session() {
        let store = InMemorySessionStore::new();
        let mut session = make_session(1);
        store.save(&session).await.unwrap();

        session.push_message(crate::types::MessageRole::User, "another");
        store.save(&session).await.unwrap();

        let loaded = store.load(&SessionId::new(1)).await.unwrap().unwrap();
        assert_eq!(loaded.chat_messages().len(), 3);
    }

    #[tokio::test]
    async fn save_without_id_errors() {
        let store = InMemorySessionStore::new();
        let session = AgentSession::default(); // id is None
        let err = store.save(&session).await.unwrap_err();
        assert!(err.to_string().contains("no id"));
    }

    #[tokio::test]
    async fn save_with_external_id_roundtrips() {
        let store = InMemorySessionStore::new();
        let mut s = AgentSession::new(SessionId::with_external_id(42, "my-ext-id"));
        s.push_message(crate::types::MessageRole::User, "test");
        store.save(&s).await.unwrap();

        let loaded = store
            .load(&SessionId::with_external_id(42, "my-ext-id"))
            .await
            .unwrap();
        assert!(loaded.is_some());
    }

    #[tokio::test]
    async fn list_empty_store() {
        let store = InMemorySessionStore::new();
        assert!(store.list().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn append_event_default_noop() {
        let store = InMemorySessionStore::new();
        store
            .append_event(
                &SessionId::new(1),
                &RuntimeEvent::UserEvent {
                    session_id: SessionId::new(1),
                    event: crate::types::UserEvent::Progress {
                        text: "test".into(),
                    },
                    agent_id: None,
                    trace_id: None,
                },
            )
            .await
            .unwrap();
    }
}
