//! SQLite structural layer (ADR-0001).
//!
//! The structural source of truth is one SQLite database in WAL mode;
//! segment content lives in the content-addressed object store and rows
//! reference it by hash. Structure is append-only inserts — no row is ever
//! updated or deleted. Every DB call goes through the tokio-rusqlite
//! dedicated-thread connection; never a blocking call on a runtime worker.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use rusqlite::OptionalExtension;
use tokio_rusqlite::Connection;

use crate::error::{Result, StoreError};
use crate::object_store::ObjectStore;
use crate::segment::Segment;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sessions(
    id              TEXT PRIMARY KEY,
    parent_id       TEXT,
    parent_position INTEGER,
    kind            TEXT NOT NULL,
    created_at      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS entries(
    session_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    asset_hash TEXT NOT NULL,
    kind       TEXT NOT NULL,
    PRIMARY KEY(session_id, seq)
);
CREATE TABLE IF NOT EXISTS compactions(
    session_id   TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    start        INTEGER NOT NULL,
    end          INTEGER NOT NULL,
    summary_hash TEXT NOT NULL,
    PRIMARY KEY(session_id, seq)
);
CREATE TABLE IF NOT EXISTS side_effects(
    session_id  TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    tool_call_id TEXT NOT NULL,
    before_hash TEXT,
    after_hash  TEXT,
    path        TEXT NOT NULL,
    PRIMARY KEY(session_id, seq)
);
CREATE TABLE IF NOT EXISTS snapshots(
    session_id TEXT NOT NULL,
    position   INTEGER NOT NULL,
    label      TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY(session_id, position, label)
);
";

/// A session store: content-addressed objects + SQLite structure, one handle.
#[derive(Debug, Clone)]
pub struct SessionStore {
    root: PathBuf,
    objects: ObjectStore,
    db: Connection,
}

impl SessionStore {
    /// Open (or create) a store under `root`: `objects/` for content,
    /// `store.db` for structure. WAL mode makes multi-process readers safe
    /// while the agent writes.
    pub async fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        tokio::fs::create_dir_all(&root).await?;
        let db = Connection::open(root.join("store.db")).await?;
        db.call(|conn| -> rusqlite::Result<()> {
            let _: String = conn.query_row("PRAGMA journal_mode = WAL;", [], |r| r.get(0))?;
            conn.execute_batch(SCHEMA)?;
            Ok(())
        })
        .await?;
        Ok(Self {
            objects: ObjectStore::new(root.clone()),
            root,
            db,
        })
    }

    /// Store root directory.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Content-addressed object store half.
    pub fn objects(&self) -> &ObjectStore {
        &self.objects
    }

    /// Create a root session (no parent). Returns its id.
    pub async fn create_root(&self) -> Result<String> {
        let id = crate::new_id();
        let row_id = id.clone();
        self.db
            .call(move |conn| -> rusqlite::Result<()> {
                conn.execute(
                    "INSERT INTO sessions(id, parent_id, parent_position, kind, created_at)
                     VALUES (?1, NULL, NULL, 'root', ?2)",
                    (&row_id, crate::now_ms() as i64),
                )?;
                Ok(())
            })
            .await?;
        Ok(id)
    }

    /// Append a segment at the tail of a session: content into the object
    /// store, one row into `entries`. Returns the content hash.
    pub async fn append_segment(&self, session_id: &str, segment: &Segment) -> Result<String> {
        self.require_session(session_id).await?;
        let hash = self.objects.put(segment).await?;
        let (sid, row_hash, kind) = (session_id.to_string(), hash.clone(), segment.kind().to_string());
        self.db
            .call(move |conn| -> rusqlite::Result<()> {
                let seq: i64 = conn.query_row(
                    "SELECT COALESCE(MAX(seq) + 1, 0) FROM entries WHERE session_id = ?1",
                    (&sid,),
                    |r| r.get(0),
                )?;
                conn.execute(
                    "INSERT INTO entries(session_id, seq, asset_hash, kind)
                     VALUES (?1, ?2, ?3, ?4)",
                    (&sid, seq, &row_hash, &kind),
                )?;
                Ok(())
            })
            .await?;
        Ok(hash)
    }

    /// Derive a new session from `parent` at its current tail (a 分叉:
    /// pure-append derivation sharing the parent prefix). Returns the new id.
    pub async fn fork(&self, parent_id: &str) -> Result<String> {
        let row = self.session_row(parent_id).await?;
        let position = self.view_len(&row).await?;
        self.insert_derivation(&row.id, position, "fork").await
    }

    /// Derive from `parent` at an explicit prefix length — the bookmark /
    /// rollback entry point. `position` is half-open: the child inherits
    /// parent view items `[0, position)`.
    pub async fn fork_at(&self, parent_id: &str, position: u64) -> Result<String> {
        let row = self.session_row(parent_id).await?;
        let len = self.view_len(&row).await?;
        if position as i64 > len {
            return Err(StoreError::InvalidLog {
                context: "derivation".into(),
                reason: format!("position {position} beyond parent view length {len}"),
            });
        }
        self.insert_derivation(&row.id, position as i64, "fork").await
    }

    async fn insert_derivation(&self, parent_id: &str, position: i64, kind: &str) -> Result<String> {
        let id = crate::new_id();
        let (pid, kind, row_id) = (parent_id.to_string(), kind.to_string(), id.clone());
        self.db
            .call(move |conn| -> rusqlite::Result<()> {
                conn.execute(
                    "INSERT INTO sessions(id, parent_id, parent_position, kind, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    (&row_id, &pid, position, &kind, crate::now_ms() as i64),
                )?;
                Ok(())
            })
            .await?;
        Ok(id)
    }

    /// Materialize a session's view: walk the derivation chain to the root,
    /// then fold — at each derivation edge the inherited prefix truncates to
    /// `parent_position` and the session's own entries extend it. Chain order
    /// is priority; there are no per-index conflict rules.
    pub async fn materialize(&self, session_id: &str) -> Result<Vec<(Segment, String)>> {
        let mut chain = Vec::new();
        let mut visited = HashSet::new();
        let mut current = session_id.to_string();
        loop {
            if !visited.insert(current.clone()) {
                return Err(StoreError::CyclicChain(current));
            }
            let row = self.session_row(&current).await?;
            let next = row.parent_id.clone();
            chain.push(row);
            match next {
                Some(parent) => current = parent,
                None => break,
            }
        }

        let mut hashes: Vec<String> = Vec::new();
        for row in chain.iter().rev() {
            if let Some(pos) = row.parent_position {
                hashes.truncate(pos as usize);
            }
            hashes.extend(self.entry_hashes(&row.id).await?);
        }

        let mut view = Vec::with_capacity(hashes.len());
        for hash in hashes {
            view.push((self.objects.get(&hash).await?, hash));
        }
        Ok(view)
    }

    /// [`materialize`](Self::materialize) without the hashes.
    pub async fn materialize_plain(&self, session_id: &str) -> Result<Vec<Segment>> {
        Ok(self
            .materialize(session_id)
            .await?
            .into_iter()
            .map(|(seg, _)| seg)
            .collect())
    }

    /// The most recently created session (any kind), or `None` on an empty
    /// store. There is no HEAD pointer — "current" is a query.
    pub async fn latest_session(&self) -> Result<Option<String>> {
        Ok(self
            .db
            .call(|conn| -> rusqlite::Result<Option<String>> {
                conn.query_row(
                    "SELECT id FROM sessions ORDER BY created_at DESC, rowid DESC LIMIT 1",
                    [],
                    |r| r.get(0),
                )
                .optional()
            })
            .await?)
    }

    async fn require_session(&self, session_id: &str) -> Result<()> {
        self.session_row(session_id).await.map(|_| ())
    }

    async fn session_row(&self, session_id: &str) -> Result<SessionRow> {
        let sid = session_id.to_string();
        let row = self
            .db
            .call(move |conn| -> rusqlite::Result<Option<SessionRow>> {
                conn.query_row(
                    "SELECT id, parent_id, parent_position FROM sessions WHERE id = ?1",
                    (&sid,),
                    |r| {
                        Ok(SessionRow {
                            id: r.get(0)?,
                            parent_id: r.get(1)?,
                            parent_position: r.get(2)?,
                        })
                    },
                )
                .optional()
            })
            .await?;
        row.ok_or_else(|| StoreError::NotFound(format!("session {session_id}")))
    }

    /// Length of a session's own view: inherited prefix + own entries.
    async fn view_len(&self, row: &SessionRow) -> Result<i64> {
        let inherited = row.parent_position.unwrap_or(0);
        let sid = row.id.clone();
        let own: i64 = self
            .db
            .call(move |conn| -> rusqlite::Result<i64> {
                conn.query_row(
                    "SELECT COUNT(*) FROM entries WHERE session_id = ?1",
                    (&sid,),
                    |r| r.get(0),
                )
            })
            .await?;
        Ok(inherited + own)
    }

    async fn entry_hashes(&self, session_id: &str) -> Result<Vec<String>> {
        let sid = session_id.to_string();
        Ok(self
            .db
            .call(move |conn| -> rusqlite::Result<Vec<String>> {
                let mut stmt = conn
                    .prepare("SELECT asset_hash FROM entries WHERE session_id = ?1 ORDER BY seq")?;
                let hashes = stmt
                    .query_map([&sid], |r| r.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(hashes)
            })
            .await?)
    }
}

struct SessionRow {
    id: String,
    parent_id: Option<String>,
    parent_position: Option<i64>,
}
