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

    /// Materialize a session's view: the derivation chain folded root-first
    /// — at each edge the inherited prefix truncates to `parent_position`,
    /// compact sessions apply their map records, own entries extend. Chain
    /// order is priority; there are no per-index conflict rules.
    pub async fn materialize(&self, session_id: &str) -> Result<Vec<(Segment, String)>> {
        let hashes = self.view_hashes(session_id, None).await?;
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

    /// Record a tool side effect: before/after payloads go into the object
    /// store (content-addressed, deduplicated), one row into
    /// `side_effects`. Seq is session-level monotonic and inherited along
    /// the derivation chain — fork and compaction never reset it.
    pub async fn append_side_effect(
        &self,
        session_id: &str,
        tool_call_id: &str,
        before: Option<&[u8]>,
        after: Option<&[u8]>,
        path: &str,
    ) -> Result<SideEffectRecord> {
        self.require_session(session_id).await?;
        let before_hash = match before {
            Some(bytes) => Some(self.objects.put_bytes(bytes).await?),
            None => None,
        };
        let after_hash = match after {
            Some(bytes) => Some(self.objects.put_bytes(bytes).await?),
            None => None,
        };
        let seq = self.next_side_effect_seq(session_id).await?;
        let record = SideEffectRecord {
            session_id: session_id.to_string(),
            seq,
            tool_call_id: tool_call_id.to_string(),
            before_hash,
            after_hash,
            path: path.to_string(),
        };
        let row = (
            record.session_id.clone(),
            record.seq as i64,
            record.tool_call_id.clone(),
            record.before_hash.clone(),
            record.after_hash.clone(),
            record.path.clone(),
        );
        self.db
            .call(move |conn| -> rusqlite::Result<()> {
                conn.execute(
                    "INSERT INTO side_effects(session_id, seq, tool_call_id, before_hash, after_hash, path)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    row,
                )?;
                Ok(())
            })
            .await?;
        Ok(record)
    }

    /// This session's own side-effect records in seq order. The lineage's
    /// full WAL is each derivation's own rows; seq continuity encodes the
    /// ordering across the chain.
    pub async fn side_effects(&self, session_id: &str) -> Result<Vec<SideEffectRecord>> {
        self.require_session(session_id).await?;
        let sid = session_id.to_string();
        Ok(self
            .db
            .call(move |conn| -> rusqlite::Result<Vec<SideEffectRecord>> {
                let mut stmt = conn.prepare(
                    "SELECT session_id, seq, tool_call_id, before_hash, after_hash, path
                     FROM side_effects WHERE session_id = ?1 ORDER BY seq",
                )?;
                let records = stmt
                    .query_map([&sid], |r| {
                        Ok(SideEffectRecord {
                            session_id: r.get(0)?,
                            seq: r.get::<_, i64>(1)? as u64,
                            tool_call_id: r.get(2)?,
                            before_hash: r.get(3)?,
                            after_hash: r.get(4)?,
                            path: r.get(5)?,
                        })
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(records)
            })
            .await?)
    }

    /// Next side-effect seq for a session: max over the derivation chain of
    /// (own max + 1) — seq is monotonic across derivation edges.
    async fn next_side_effect_seq(&self, session_id: &str) -> Result<u64> {
        let mut ids = Vec::new();
        let mut visited = HashSet::new();
        let mut current = session_id.to_string();
        loop {
            if !visited.insert(current.clone()) {
                return Err(StoreError::CyclicChain(current));
            }
            let row = self.session_row(&current).await?;
            let next = row.parent_id.clone();
            ids.push(row.id);
            match next {
                Some(parent) => current = parent,
                None => break,
            }
        }
        let mut next: i64 = 0;
        for sid in ids {
            let next_for_sid: i64 = self
                .db
                .call(move |conn| -> rusqlite::Result<i64> {
                    conn.query_row(
                        "SELECT COALESCE(MAX(seq) + 1, 0) FROM side_effects WHERE session_id = ?1",
                        (&sid,),
                        |r| r.get(0),
                    )
                })
                .await?;
            next = next.max(next_for_sid);
        }
        Ok(next as u64)
    }

    /// Compact `parent`: a derivation (kind `compact`) whose map records
    /// replace half-open ranges `[start, end)` of the parent's view — indices
    /// into the parent view at derivation time — with `summary`. Consecutive
    /// covered slots sharing one summary collapse into a single view item.
    /// The parent stays intact and re-derivable (undo = derive from it).
    pub async fn compact(
        &self,
        parent_id: &str,
        mappings: &[(u64, u64)],
        summary: &Segment,
    ) -> Result<String> {
        let row = self.session_row(parent_id).await?;
        let len = self.view_len(&row).await?;
        for &(start, end) in mappings {
            if start >= end {
                return Err(StoreError::InvalidLog {
                    context: "compaction".into(),
                    reason: format!("range [{start},{end}) is empty"),
                });
            }
            if end as i64 > len {
                return Err(StoreError::InvalidLog {
                    context: "compaction".into(),
                    reason: format!("range [{start},{end}) beyond parent view length {len}"),
                });
            }
        }
        let summary_hash = self.objects.put(summary).await?;
        let id = self
            .insert_derivation(&row.id, len, "compact")
            .await?;
        for (seq, &(start, end)) in mappings.iter().enumerate() {
            let (sid, hash) = (id.clone(), summary_hash.clone());
            self.db
                .call(move |conn| -> rusqlite::Result<()> {
                    conn.execute(
                        "INSERT INTO compactions(session_id, seq, start, end, summary_hash)
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                        (&sid, seq as i64, start as i64, end as i64, &hash),
                    )?;
                    Ok(())
                })
                .await?;
        }
        Ok(id)
    }

    /// Originals covered by a compaction's maps: the parent view at
    /// derivation time, sliced per map. The structural route — the content
    /// route is `SummarySegment.sources`.
    pub async fn fetch_originals(
        &self,
        session_id: &str,
    ) -> Result<Vec<(u64, u64, Vec<Segment>)>> {
        let row = self.session_row(session_id).await?;
        if row.parent_id.is_none() {
            return Ok(Vec::new());
        }
        let maps = self.compaction_maps(&row.id).await?;
        // Parent view truncated to the inherited prefix = what the maps index.
        let mut hashes = self.view_hashes(&row.parent_id.unwrap(), row.parent_position).await?;
        let mut out = Vec::with_capacity(maps.len());
        for (start, end, _) in &maps {
            let slice = hashes[*start as usize..*end as usize].to_vec();
            let mut segs = Vec::with_capacity(slice.len());
            for hash in slice {
                segs.push(self.objects.get(&hash).await?);
            }
            out.push((*start as u64, *end as u64, segs));
        }
        hashes.clear();
        Ok(out)
    }

    async fn compaction_maps(&self, session_id: &str) -> Result<Vec<(i64, i64, String)>> {
        let sid = session_id.to_string();
        Ok(self
            .db
            .call(move |conn| -> rusqlite::Result<Vec<(i64, i64, String)>> {
                let mut stmt = conn.prepare(
                    "SELECT start, end, summary_hash FROM compactions
                     WHERE session_id = ?1 ORDER BY seq",
                )?;
                let maps = stmt
                    .query_map([&sid], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok(maps)
            })
            .await?)
    }

    /// View hashes of `session_id` truncated to `limit` (None = full view).
    /// Same chain fold as [`materialize`](Self::materialize), hashes only.
    async fn view_hashes(&self, session_id: &str, limit: Option<i64>) -> Result<Vec<String>> {
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
            if row.kind == "compact" {
                self.apply_compaction_maps(&mut hashes, &row.id).await?;
            }
            hashes.extend(self.entry_hashes(&row.id).await?);
        }
        if let Some(limit) = limit {
            hashes.truncate(limit as usize);
        }
        Ok(hashes)
    }

    /// Apply a compact session's map records to its inherited prefix:
    /// per-slot assignment in seq order, consecutive slots sharing a
    /// summary hash collapse to one item.
    async fn apply_compaction_maps(&self, hashes: &mut Vec<String>, session_id: &str) -> Result<()> {
        let maps = self.compaction_maps(session_id).await?;
        if maps.is_empty() {
            return Ok(());
        }
        let mut slots = vec![None::<String>; hashes.len()];
        for (start, end, summary_hash) in &maps {
            for slot in slots.iter_mut().take(*end as usize).skip(*start as usize) {
                *slot = Some(summary_hash.clone());
            }
        }
        let mut folded: Vec<String> = Vec::with_capacity(hashes.len());
        let mut last_summary: Option<String> = None;
        for (hash, slot) in hashes.drain(..).zip(slots) {
            match slot {
                Some(summary_hash) => {
                    if last_summary.as_ref() != Some(&summary_hash) {
                        folded.push(summary_hash.clone());
                        last_summary = Some(summary_hash);
                    }
                }
                None => {
                    folded.push(hash);
                    last_summary = None;
                }
            }
        }
        *hashes = folded;
        Ok(())
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
                    "SELECT id, parent_id, parent_position, kind FROM sessions WHERE id = ?1",
                    (&sid,),
                    |r| {
                        Ok(SessionRow {
                            id: r.get(0)?,
                            parent_id: r.get(1)?,
                            parent_position: r.get(2)?,
                            kind: r.get(3)?,
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
    kind: String,
}

/// A recorded side effect of a tool call — the structural row; before/after
/// payloads are content-addressed objects referenced by hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SideEffectRecord {
    pub session_id: String,
    pub seq: u64,
    pub tool_call_id: String,
    pub before_hash: Option<String>,
    pub after_hash: Option<String>,
    pub path: String,
}
