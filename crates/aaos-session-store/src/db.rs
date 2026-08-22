//! SQLite structural layer (ADR-0001).
//!
//! The structural source of truth is one SQLite database in WAL mode;
//! segment content lives in the content-addressed object store and rows
//! reference it by hash. Structure is append-only inserts — no row is ever
//! updated or deleted. Every DB call goes through the tokio-rusqlite
//! dedicated-thread connection; never a blocking call on a runtime worker.

use std::path::{Path, PathBuf};

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

    /// Materialize a session's view: entries in seq order, each segment
    /// fetched from the object store, paired with its hash.
    pub async fn materialize(&self, session_id: &str) -> Result<Vec<(Segment, String)>> {
        let sid = session_id.to_string();
        let rows: Vec<String> = self
            .db
            .call(move |conn| -> rusqlite::Result<Vec<String>> {
                let mut stmt = conn
                    .prepare("SELECT asset_hash FROM entries WHERE session_id = ?1 ORDER BY seq")?;
                let hashes = stmt
                    .query_map([&sid], |r| r.get::<_, String>(0))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(hashes)
            })
            .await?;
        let mut view = Vec::with_capacity(rows.len());
        for hash in rows {
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

    async fn require_session(&self, session_id: &str) -> Result<()> {
        let sid = session_id.to_string();
        let found: Option<()> = self
            .db
            .call(move |conn| -> rusqlite::Result<Option<()>> {
                match conn.query_row(
                    "SELECT 1 FROM sessions WHERE id = ?1",
                    (&sid,),
                    |_| Ok(()),
                ) {
                    Ok(()) => Ok(Some(())),
                    Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
                    Err(e) => Err(e),
                }
            })
            .await?;
        found.ok_or_else(|| StoreError::NotFound(format!("session {session_id}")))
    }
}
