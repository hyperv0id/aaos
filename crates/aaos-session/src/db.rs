//! SQLite structural layer (ADR-0001, ADR-0003).
//!
//! The structural source of truth is one SQLite database in WAL mode;
//! segment content lives in the content-addressed object store and rows
//! reference it by hash. Structure is append-only inserts — no row is ever
//! updated or deleted, with one deliberate exception: the `meta` table
//! holds the mutable head pointer (ADR-0003). Every DB call goes through
//! the tokio-rusqlite dedicated-thread connection; never a blocking call
//! on a runtime worker.

use std::collections::HashSet;
use std::path::PathBuf;

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
CREATE TABLE IF NOT EXISTS meta(
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
";

/// meta key holding the head pointer (ADR-0003) — the session node last
/// appended to. Shared by the append-path upsert and the `head` query so
/// the two can never drift.
const HEAD_KEY: &str = "head";

/// A session store: content-addressed objects + SQLite structure, one handle.
#[derive(Debug, Clone)]
pub struct SessionStore {
    objects: ObjectStore,
    db: Connection,
}

impl SessionStore {
    /// Open (or create) a store under `root`: `objects/` for content,
    /// `store.db` for structure. WAL mode makes multi-process readers safe
    /// while the agent writes; the busy timeout makes multi-process
    /// writers queue instead of failing on lock contention.
    pub async fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        tokio::fs::create_dir_all(&root).await?;
        let db = Connection::open(root.join("store.db")).await?;
        db.call(|conn| -> rusqlite::Result<()> {
            conn.busy_timeout(std::time::Duration::from_millis(5_000))?;
            let _: String = conn.query_row("PRAGMA journal_mode = WAL;", [], |r| r.get(0))?;
            conn.execute_batch(SCHEMA)?;
            Ok(())
        })
        .await?;
        Ok(Self {
            objects: ObjectStore::new(root),
            db,
        })
    }

    /// Content-addressed object store half.
    pub fn objects(&self) -> &ObjectStore {
        &self.objects
    }

    /// Create a root session (no parent). Returns its id.
    pub async fn create_root(&self) -> Result<String> {
        let id = crate::new_id();
        let (row_id, kind) = (id.clone(), SessionKind::Root.as_str().to_string());
        self.db
            .call(move |conn| -> rusqlite::Result<()> {
                conn.execute(
                    "INSERT INTO sessions(id, parent_id, parent_position, kind, created_at)
                     VALUES (?1, NULL, NULL, ?2, ?3)",
                    (&row_id, &kind, crate::now_ms() as i64),
                )?;
                Ok(())
            })
            .await?;
        Ok(id)
    }

    /// Append a segment at the tail of a session: content into the object
    /// store, one row into `entries`. Moves the head pointer to
    /// `session_id` — HEAD is the line last written. Seq assignment, the
    /// insert, and the head move share one IMMEDIATE transaction: the write
    /// lock is taken up front, so concurrent processes serialize instead of
    /// racing the read-then-write (a deferred transaction would die with
    /// BUSY_SNAPSHOT, which the busy timeout never retries).
    pub async fn append_segment(&self, session_id: &str, segment: &Segment) -> Result<String> {
        self.require_session(session_id).await?;
        let hash = self.objects.put(segment).await?;
        let (sid, row_hash, kind) = (
            session_id.to_string(),
            hash.clone(),
            segment.kind().to_string(),
        );
        self.db
            .call(move |conn| -> rusqlite::Result<()> {
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let seq: i64 = tx.query_row(
                    "SELECT COALESCE(MAX(seq) + 1, 0) FROM entries WHERE session_id = ?1",
                    (&sid,),
                    |r| r.get(0),
                )?;
                tx.execute(
                    "INSERT INTO entries(session_id, seq, asset_hash, kind)
                     VALUES (?1, ?2, ?3, ?4)",
                    (&sid, seq, &row_hash, &kind),
                )?;
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (HEAD_KEY, &sid),
                )?;
                tx.commit()?;
                Ok(())
            })
            .await?;
        Ok(hash)
    }

    /// Derive a new session from `parent` at its current tail (a 分叉:
    /// pure-append derivation sharing the parent prefix). Returns the new id.
    pub async fn fork(&self, parent_id: &str) -> Result<String> {
        let row = self.session_row(parent_id).await?;
        let position = self.view_len(&row.id).await?;
        self.insert_derivation(&row.id, position, SessionKind::Fork)
            .await
    }

    /// Derive from `parent` at an explicit prefix length — the bookmark /
    /// rollback entry point. `position` is half-open: the child inherits
    /// parent view items `[0, position)`.
    pub async fn fork_at(&self, parent_id: &str, position: u64) -> Result<String> {
        let row = self.session_row(parent_id).await?;
        let len = self.view_len(&row.id).await?;
        if position as i64 > len {
            return Err(StoreError::InvalidDerivation {
                context: "fork".into(),
                reason: format!("position {position} beyond parent view length {len}"),
            });
        }
        self.insert_derivation(&row.id, position as i64, SessionKind::Fork)
            .await
    }

    async fn insert_derivation(
        &self,
        parent_id: &str,
        position: i64,
        kind: SessionKind,
    ) -> Result<String> {
        let id = crate::new_id();
        let (pid, kind, row_id) = (parent_id.to_string(), kind.as_str().to_string(), id.clone());
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

    /// Record a bookmark: (session, current view length, label). Pure marker —
    /// nothing ever auto-restores to it; rollback = `fork_at` from it, which
    /// is a derivation like any other.
    pub async fn bookmark(&self, session_id: &str, label: &str) -> Result<Bookmark> {
        let row = self.session_row(session_id).await?;
        let position = self.view_len(&row.id).await?;
        let created_at = crate::now_ms();
        let (sid, lbl) = (row.id.clone(), label.to_string());
        self.db
            .call(move |conn| -> rusqlite::Result<()> {
                conn.execute(
                    "INSERT OR IGNORE INTO snapshots(session_id, position, label, created_at)
                     VALUES (?1, ?2, ?3, ?4)",
                    (&sid, position, &lbl, created_at as i64),
                )?;
                Ok(())
            })
            .await?;
        Ok(Bookmark {
            session_id: session_id.to_string(),
            position: position as u64,
            label: label.to_string(),
            created_at,
        })
    }

    /// A session's bookmarks, oldest first. Same label may appear at several
    /// positions (its history); callers pick what they mean.
    pub async fn bookmarks(&self, session_id: &str) -> Result<Vec<Bookmark>> {
        self.require_session(session_id).await?;
        self.select_session(
            "SELECT session_id, position, label, created_at
                 FROM snapshots WHERE session_id = ?1 ORDER BY created_at, rowid",
            session_id,
            |r| {
                Ok(Bookmark {
                    session_id: r.get(0)?,
                    position: r.get::<_, i64>(1)? as u64,
                    label: r.get(2)?,
                    created_at: r.get::<_, i64>(3)? as u64,
                })
            },
        )
        .await
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
        // Seq computation and the insert go through one call on the
        // dedicated-thread connection: calls are serialized there, so no
        // other append can interleave between read and write.
        let ids: Vec<String> = self
            .chain_rows(session_id)
            .await?
            .into_iter()
            .map(|row| row.id)
            .collect();
        let sid0 = session_id.to_string();
        let (tool_call_id, path) = (tool_call_id.to_string(), path.to_string());
        let record = self
            .db
            .call(move |conn| -> rusqlite::Result<SideEffectRecord> {
                let mut next: i64 = 0;
                for sid in &ids {
                    let next_for_sid: i64 = conn.query_row(
                        "SELECT COALESCE(MAX(seq) + 1, 0) FROM side_effects WHERE session_id = ?1",
                        (sid,),
                        |r| r.get(0),
                    )?;
                    next = next.max(next_for_sid);
                }
                let record = SideEffectRecord {
                    session_id: sid0,
                    seq: next as u64,
                    tool_call_id,
                    before_hash,
                    after_hash,
                    path,
                };
                conn.execute(
                    "INSERT INTO side_effects(session_id, seq, tool_call_id, before_hash, after_hash, path)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    (
                        &record.session_id,
                        record.seq as i64,
                        &record.tool_call_id,
                        &record.before_hash,
                        &record.after_hash,
                        &record.path,
                    ),
                )?;
                Ok(record)
            })
            .await?;
        Ok(record)
    }

    /// This session's own side-effect records in seq order. The lineage's
    /// full record set is each derivation's own rows; seq continuity encodes
    /// the ordering across the chain.
    pub async fn side_effects(&self, session_id: &str) -> Result<Vec<SideEffectRecord>> {
        self.require_session(session_id).await?;
        self.select_session(
            "SELECT session_id, seq, tool_call_id, before_hash, after_hash, path
                 FROM side_effects WHERE session_id = ?1 ORDER BY seq",
            session_id,
            |r| {
                Ok(SideEffectRecord {
                    session_id: r.get(0)?,
                    seq: r.get::<_, i64>(1)? as u64,
                    tool_call_id: r.get(2)?,
                    before_hash: r.get(3)?,
                    after_hash: r.get(4)?,
                    path: r.get(5)?,
                })
            },
        )
        .await
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
        let len = self.view_len(&row.id).await?;
        for &(start, end) in mappings {
            if start >= end {
                return Err(StoreError::InvalidDerivation {
                    context: "compaction".into(),
                    reason: format!("range [{start},{end}) is empty"),
                });
            }
            if end as i64 > len {
                return Err(StoreError::InvalidDerivation {
                    context: "compaction".into(),
                    reason: format!("range [{start},{end}) beyond parent view length {len}"),
                });
            }
        }
        let summary_hash = self.objects.put(summary).await?;
        let id = crate::new_id();
        let (pid, kind) = (row.id.clone(), SessionKind::Compact);
        let maps = mappings.to_vec();
        let row_id = id.clone();
        // Session row and map rows land in one transaction: a crash never
        // leaves a half-applied compaction.
        self.db
            .call(move |conn| -> rusqlite::Result<()> {
                let tx = conn.transaction()?;
                tx.execute(
                    "INSERT INTO sessions(id, parent_id, parent_position, kind, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    (&row_id, &pid, len, kind.as_str(), crate::now_ms() as i64),
                )?;
                for (seq, (start, end)) in maps.iter().enumerate() {
                    tx.execute(
                        "INSERT INTO compactions(session_id, seq, start, end, summary_hash)
                         VALUES (?1, ?2, ?3, ?4, ?5)",
                        (
                            &row_id,
                            seq as i64,
                            *start as i64,
                            *end as i64,
                            &summary_hash,
                        ),
                    )?;
                }
                tx.commit()?;
                Ok(())
            })
            .await?;
        Ok(id)
    }

    /// Originals covered by a compaction's maps: the parent view at
    /// derivation time, sliced per map. The structural route — the content
    /// route is `SummarySegment.sources`.
    pub async fn fetch_originals(&self, session_id: &str) -> Result<Vec<CoveredRange>> {
        let row = self.session_row(session_id).await?;
        let Some(parent_id) = &row.parent_id else {
            return Ok(Vec::new());
        };
        let maps = self.compaction_maps(&row.id).await?;
        // Parent view truncated to the inherited prefix = what the maps index.
        let hashes = self.view_hashes(parent_id, row.parent_position).await?;
        let mut out = Vec::with_capacity(maps.len());
        for (start, end, _) in &maps {
            let mut originals = Vec::with_capacity((*end - *start) as usize);
            for hash in &hashes[*start as usize..*end as usize] {
                originals.push(self.objects.get(hash).await?);
            }
            out.push(CoveredRange {
                start: *start as u64,
                end: *end as u64,
                originals,
            });
        }
        Ok(out)
    }

    async fn compaction_maps(&self, session_id: &str) -> Result<Vec<(i64, i64, String)>> {
        self.select_session(
            "SELECT start, end, summary_hash FROM compactions
                 WHERE session_id = ?1 ORDER BY seq",
            session_id,
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .await
    }

    /// View hashes of `session_id` truncated to `limit` (None = full view).
    /// Same chain fold as [`materialize`](Self::materialize), hashes only.
    async fn view_hashes(&self, session_id: &str, limit: Option<i64>) -> Result<Vec<String>> {
        let chain = self.chain_rows(session_id).await?;
        let mut hashes: Vec<String> = Vec::new();
        for row in chain.iter().rev() {
            if let Some(pos) = row.parent_position {
                hashes.truncate(pos as usize);
            }
            if row.kind == SessionKind::Compact {
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
    async fn apply_compaction_maps(
        &self,
        hashes: &mut Vec<String>,
        session_id: &str,
    ) -> Result<()> {
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

    /// The head pointer (ADR-0003): the session node last appended to, or
    /// `None` on a store that has never seen an append (fresh, or created
    /// before the pointer existed). This is the default resume target —
    /// "the line the user was last writing". Under concurrent processes the
    /// pointer is last-writer-wins; each process still appends to its own
    /// node, so the flip never moves anyone else's writes.
    pub async fn head(&self) -> Result<Option<String>> {
        Ok(self
            .db
            .call(|conn| -> rusqlite::Result<Option<String>> {
                conn.query_row("SELECT value FROM meta WHERE key = ?1", [HEAD_KEY], |r| {
                    r.get(0)
                })
                .optional()
            })
            .await?)
    }

    /// Whether `session_id` names an existing session row. `false` only for
    /// the not-found case; every other error propagates.
    pub async fn session_exists(&self, session_id: &str) -> Result<bool> {
        match self.session_row(session_id).await {
            Ok(_) => Ok(true),
            Err(StoreError::NotFound(_)) => Ok(false),
            Err(err) => Err(err),
        }
    }

    /// The most recently created session (any kind), or `None` on an empty
    /// store. Fallback for stores predating the head pointer (ADR-0003) —
    /// "current" is [`head`](Self::head), not this query.
    pub async fn latest_created_session(&self) -> Result<Option<String>> {
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

    /// The derivation chain of `session_id`, leaf first, with a cyclic guard.
    async fn chain_rows(&self, session_id: &str) -> Result<Vec<SessionRow>> {
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
                None => return Ok(chain),
            }
        }
    }

    /// Run a SELECT keyed on `session_id` and map every returned row.
    async fn select_session<T, F>(&self, sql: &str, session_id: &str, map: F) -> Result<Vec<T>>
    where
        F: Fn(&rusqlite::Row) -> rusqlite::Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let (sid, sql) = (session_id.to_string(), sql.to_string());
        Ok(self
            .db
            .call(move |conn| -> rusqlite::Result<Vec<T>> {
                let mut stmt = conn.prepare(&sql)?;
                let rows = stmt
                    .query_map([&sid], |r| map(r))?
                    .collect::<rusqlite::Result<Vec<T>>>()?;
                Ok(rows)
            })
            .await?)
    }

    async fn session_row(&self, session_id: &str) -> Result<SessionRow> {
        let sid = session_id.to_string();
        let row = self
            .db
            .call(move |conn| -> rusqlite::Result<Option<RawSessionRow>> {
                conn.query_row(
                    "SELECT id, parent_id, parent_position, kind FROM sessions WHERE id = ?1",
                    (&sid,),
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                )
                .optional()
            })
            .await?;
        let (id, parent_id, parent_position, kind) =
            row.ok_or_else(|| StoreError::NotFound(format!("session {session_id}")))?;
        Ok(SessionRow {
            id,
            parent_id,
            parent_position,
            kind: SessionKind::from_db(&kind)?,
        })
    }

    /// Length of a session's own view, in folded coordinates: after the
    /// chain fold applies truncations and compaction maps. Positions that
    /// address a view (fork_at, bookmark) live in these coordinates.
    async fn view_len(&self, session_id: &str) -> Result<i64> {
        Ok(self.view_hashes(session_id, None).await?.len() as i64)
    }

    async fn entry_hashes(&self, session_id: &str) -> Result<Vec<String>> {
        self.select_session(
            "SELECT asset_hash FROM entries WHERE session_id = ?1 ORDER BY seq",
            session_id,
            |r| r.get::<_, String>(0),
        )
        .await
    }
}

struct SessionRow {
    id: String,
    parent_id: Option<String>,
    parent_position: Option<i64>,
    kind: SessionKind,
}

/// (id, parent_id, parent_position, kind) as read off the row, kind raw.
type RawSessionRow = (String, Option<String>, Option<i64>, String);

/// The kind of a session row — which record shape its derivation carries
/// (CONTEXT.md: 派生 is the operation; 分叉 and 压缩 are its two shapes).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionKind {
    Root,
    Fork,
    Compact,
}

impl SessionKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SessionKind::Root => "root",
            SessionKind::Fork => "fork",
            SessionKind::Compact => "compact",
        }
    }

    fn from_db(kind: &str) -> Result<Self> {
        match kind {
            "root" => Ok(SessionKind::Root),
            "fork" => Ok(SessionKind::Fork),
            "compact" => Ok(SessionKind::Compact),
            other => Err(StoreError::Decode(format!(
                "unknown session kind {other:?}"
            ))),
        }
    }
}

/// One compaction map's slice of the covered parent view: the half-open
/// range `[start, end)` and the original segments it replaced.
#[derive(Debug, Clone, PartialEq)]
pub struct CoveredRange {
    pub start: u64,
    pub end: u64,
    pub originals: Vec<Segment>,
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

/// A bookmark on a session — (session, position, label). Pure marker: it
/// never restores anything by itself; it anchors a rollback derivation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bookmark {
    pub session_id: String,
    pub position: u64,
    pub label: String,
    pub created_at: u64,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// Ticket 02 structural invariant: derivation copies nothing — the
    /// child's own entries rows cover only its own appends, and the parent's
    /// rows are untouched. Checked at the internal seam on purpose: the
    /// public seam can only see the view.
    #[tokio::test]
    async fn fork_adds_entries_rows_only_for_own_segments() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        for text in ["a", "b", "c"] {
            store
                .append_segment(&root, &Segment::user_text(text))
                .await
                .unwrap();
        }
        let child = store.fork(&root).await.unwrap();
        store
            .append_segment(&child, &Segment::user_text("own"))
            .await
            .unwrap();

        let count = |sid: String| {
            store.db.call(move |conn| -> rusqlite::Result<i64> {
                conn.query_row(
                    "SELECT COUNT(*) FROM entries WHERE session_id = ?1",
                    (&sid,),
                    |r| r.get(0),
                )
            })
        };
        assert_eq!(count(root.clone()).await.unwrap(), 3);
        assert_eq!(count(child.clone()).await.unwrap(), 1);
        assert_eq!(store.materialize_plain(&child).await.unwrap().len(), 4);
    }
}
