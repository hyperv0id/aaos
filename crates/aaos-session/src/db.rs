//! SQLite structural layer (ADR-0001, ADR-0003, ADR-0006).
//!
//! The structural source of truth is one SQLite database in WAL mode;
//! block content lives in the content-addressed object store and rows
//! reference it by hash. Structure is append-only inserts — no row is ever
//! updated or deleted, with one deliberate exception: the `meta` table
//! holds the mutable head pointer (ADR-0003). Every DB call goes through
//! the tokio-rusqlite dedicated-thread connection; never a blocking call
//! on a runtime worker.
//!
//! Schema is v2 (ADR-0006): a segment's content is decomposed into
//! per-block objects — raw text bytes, image payloads, canonical JSON —
//! that are self-readable outside the DB, with an `entry_blocks` row list
//! per entry carrying what the bytes cannot (kind, mime type, tool
//! attribution); message metadata that is not content lives in `entries`
//! columns; compaction summaries are raw text content objects. The view
//! fold shuffles coordinates `(session_id, seq)` instead of hashes.
//!
//! There is no migration script and no version machinery (the schema has
//! always been plain `CREATE TABLE IF NOT EXISTS`): a v1 store — JSON
//! envelope objects, no new columns — is incompatible by design with v2 and
//! must be wiped (`~/.config/aaos`), the migration decided by ADR-0006.

use std::collections::HashSet;
use std::path::PathBuf;

use rusqlite::OptionalExtension;
use serde_json::Value;
use tokio_rusqlite::Connection;

use crate::error::{Result, StoreError};
use crate::object_store::ObjectStore;
use crate::segment::{
    AssistantSegment, ContentBlock, ImageSource, Segment, SummarySegment, ToolCall,
    ToolResultSegment, UserSegment,
};

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sessions(
    id              TEXT PRIMARY KEY,
    parent_id       TEXT,
    parent_position INTEGER,
    kind            TEXT NOT NULL,
    created_at      INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS entries(
    session_id       TEXT NOT NULL,
    seq              INTEGER NOT NULL,
    kind             TEXT NOT NULL,
    created_at       INTEGER NOT NULL,
    stop_reason      TEXT,
    model            TEXT,
    provider         TEXT,
    api              TEXT,
    usage            TEXT,
    error_message    TEXT,
    is_error         INTEGER,
    added_tool_names TEXT,
    PRIMARY KEY(session_id, seq)
);
CREATE TABLE IF NOT EXISTS entry_blocks(
    session_id   TEXT NOT NULL,
    seq          INTEGER NOT NULL,
    idx          INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    hash         TEXT NOT NULL,
    mime_type    TEXT,
    tool_call_id TEXT,
    tool_name    TEXT,
    PRIMARY KEY(session_id, seq, idx)
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

/// Block kinds stored in `entry_blocks` (ADR-0006). Text/thinking are raw
/// UTF-8, image is the payload itself, tool_call/details are canonical JSON
/// of `arguments` / `details`, summary is the raw summary text; the DB
/// columns carry what the bytes cannot (mime type, tool attribution).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockKind {
    Text,
    Thinking,
    Image,
    ToolCall,
    /// A tool result's `details` — the block row also carries the
    /// message-level tool attribution (id/name), which has no other home.
    Details,
    Summary,
}

impl BlockKind {
    fn as_str(self) -> &'static str {
        match self {
            BlockKind::Text => "text",
            BlockKind::Thinking => "thinking",
            BlockKind::Image => "image",
            BlockKind::ToolCall => "tool_call",
            BlockKind::Details => "details",
            BlockKind::Summary => "summary",
        }
    }

    fn from_str(kind: &str) -> Result<Self> {
        match kind {
            "text" => Ok(BlockKind::Text),
            "thinking" => Ok(BlockKind::Thinking),
            "image" => Ok(BlockKind::Image),
            "tool_call" => Ok(BlockKind::ToolCall),
            "details" => Ok(BlockKind::Details),
            "summary" => Ok(BlockKind::Summary),
            other => Err(StoreError::Decode(format!(
                "unknown entry_blocks kind {other:?}"
            ))),
        }
    }
}

/// Canonical JSON encoding for `Value`-typed content (tool_call arguments,
/// tool_result details). The hash is stable because serde_json is
/// deterministic, not because it sorts: struct fields serialize in
/// declaration order (fixed at compile time), and `Value::Object` keys
/// serialize in sorted order via `BTreeMap`. Identity = BLAKE3-256 hex.
///
/// NOTE: the sorted-key property for `Value::Object` holds only while
/// `serde_json`'s `preserve_order` feature is not enabled anywhere in the
/// dependency graph. Enabling it switches `Value::Object` to `IndexMap`
/// (insertion order), breaking content-addressing for the `details` and
/// `arguments` fields. Never add `preserve_order` to a crate that links
/// `aaos-session`.
pub(crate) fn canonical_json(value: &Value) -> Result<Vec<u8>> {
    serde_json::to_vec(value).map_err(|e| StoreError::Encode(e.to_string()))
}

/// The raw object bytes of one content block (ADR-0006) — UTF-8 text,
/// image payload, or canonical JSON of tool-call arguments. Content
/// addressing makes `hash_hex(block_bytes(b))` the block's object identity,
/// so the compaction transcript resolves a block's object path by
/// recomputation, with no DB round-trip.
pub(crate) fn block_bytes(block: &ContentBlock) -> Result<Vec<u8>> {
    match block {
        ContentBlock::Text { text } => Ok(text.as_bytes().to_vec()),
        ContentBlock::Thinking { text } => Ok(text.as_bytes().to_vec()),
        ContentBlock::Image { source } => Ok(source.bytes.clone()),
        ContentBlock::ToolCall(call) => canonical_json(&call.arguments),
    }
}

/// One encoded content block: the raw object bytes plus DB-bound
/// attributes. `bytes` are written to the object store as-is; the rest
/// land on the `entry_blocks` row.
struct EncodedBlock {
    kind: BlockKind,
    bytes: Vec<u8>,
    mime_type: Option<String>,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
}

/// Encode one content block per ADR-0006: the bytes are self-readable
/// outside the DB ([`block_bytes`]), everything the bytes cannot carry is
/// attributed here.
fn encode_content_block(block: &ContentBlock) -> Result<EncodedBlock> {
    let bytes = block_bytes(block)?;
    Ok(match block {
        ContentBlock::Text { .. } => EncodedBlock {
            kind: BlockKind::Text,
            bytes,
            mime_type: None,
            tool_call_id: None,
            tool_name: None,
        },
        ContentBlock::Thinking { .. } => EncodedBlock {
            kind: BlockKind::Thinking,
            bytes,
            mime_type: None,
            tool_call_id: None,
            tool_name: None,
        },
        ContentBlock::Image { source } => EncodedBlock {
            kind: BlockKind::Image,
            bytes,
            mime_type: Some(source.mime_type.clone()),
            tool_call_id: None,
            tool_name: None,
        },
        ContentBlock::ToolCall(call) => EncodedBlock {
            kind: BlockKind::ToolCall,
            bytes,
            mime_type: None,
            tool_call_id: Some(call.id.clone()),
            tool_name: Some(call.name.clone()),
        },
    })
}

/// A block row ready for insertion: content hash + `entry_blocks` columns.
struct StoredBlock {
    kind: BlockKind,
    hash: String,
    mime_type: Option<String>,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
}

/// Message metadata that is not content (ADR-0006): stored as `entries`
/// columns, `None` when the segment kind does not carry the field.
/// `usage`, `added_tool_names` and `stop_reason` are JSON-encoded.
#[derive(Debug, Default)]
struct EntryMeta {
    stop_reason: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    api: Option<String>,
    usage: Option<String>,
    error_message: Option<String>,
    is_error: Option<bool>,
    added_tool_names: Option<String>,
}

fn json_string(v: &impl serde::Serialize) -> Result<String> {
    serde_json::to_string(v).map_err(|e| StoreError::Encode(e.to_string()))
}

fn entry_meta(segment: &Segment) -> Result<EntryMeta> {
    Ok(match segment {
        Segment::User(_) | Segment::Summary(_) => EntryMeta::default(),
        Segment::Assistant(a) => EntryMeta {
            stop_reason: Some(json_string(&a.stop_reason)?),
            model: Some(a.model.clone()),
            provider: Some(a.provider.clone()),
            api: Some(a.api.clone()),
            usage: Some(json_string(&a.usage)?),
            error_message: a.error_message.clone(),
            ..EntryMeta::default()
        },
        Segment::ToolResult(t) => EntryMeta {
            usage: t.usage.as_ref().map(json_string).transpose()?,
            is_error: Some(t.is_error),
            added_tool_names: t.added_tool_names.as_ref().map(json_string).transpose()?,
            ..EntryMeta::default()
        },
    })
}

/// Raw `entry_blocks` row as read off the DB (kind parse happens outside the
/// dedicated-thread closure, where crate errors can propagate).
type RawBlockRow = (
    String,
    String,
    Option<String>,
    Option<String>,
    Option<String>,
);

/// One `entry_blocks` row as read back.
struct BlockRow {
    kind: BlockKind,
    hash: String,
    mime_type: Option<String>,
    tool_call_id: Option<String>,
    tool_name: Option<String>,
}

/// One `entries` row as read back (kind + created_at + metadata columns).
struct EntryRow {
    kind: String,
    created_at: i64,
    stop_reason: Option<String>,
    model: Option<String>,
    provider: Option<String>,
    api: Option<String>,
    usage: Option<String>,
    error_message: Option<String>,
    is_error: Option<bool>,
    added_tool_names: Option<String>,
}

/// A view item in folded coordinates (ADR-0006): an `entries` row or a
/// summary produced by a `compactions` map row, each identified by its own
/// `(session_id, seq)`. The fold only shuffles these coordinates.
#[derive(Debug, Clone)]
enum ViewItem {
    Entry { session_id: String, seq: i64 },
    Summary { session_id: String, seq: i64 },
}

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

    /// Append a segment at the tail of a session: content blocks into the
    /// object store (raw, self-readable bytes), one `entries` row holding
    /// the non-content metadata, one `entry_blocks` row per block
    /// (ADR-0006). Moves the head pointer to `session_id` — HEAD is the
    /// session last written. Returns the entry coordinate
    /// `"<session_id>:<seq>"` — the segment's store identity now that
    /// block granularity leaves no per-message hash. Seq assignment, the
    /// inserts, and the head move share one IMMEDIATE transaction: the
    /// write lock is taken up front, so concurrent processes serialize
    /// instead of racing the read-then-write (a deferred transaction would
    /// die with BUSY_SNAPSHOT, which the busy timeout never retries).
    pub async fn append_segment(&self, session_id: &str, segment: &Segment) -> Result<String> {
        self.require_session(session_id).await?;
        let blocks = self.store_segment_blocks(segment).await?;
        let meta = entry_meta(segment)?;
        let created_at = crate::now_ms();
        let (sid, kind) = (session_id.to_string(), segment.kind().to_string());
        let seq = self
            .db
            .call(move |conn| -> rusqlite::Result<i64> {
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let seq: i64 = tx.query_row(
                    "SELECT COALESCE(MAX(seq) + 1, 0) FROM entries WHERE session_id = ?1",
                    (&sid,),
                    |r| r.get(0),
                )?;
                tx.execute(
                    "INSERT INTO entries(session_id, seq, kind, created_at, stop_reason, model, provider, api, usage, error_message, is_error, added_tool_names)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                    (
                        &sid,
                        seq,
                        &kind,
                        created_at as i64,
                        &meta.stop_reason,
                        &meta.model,
                        &meta.provider,
                        &meta.api,
                        &meta.usage,
                        &meta.error_message,
                        meta.is_error.map(i64::from),
                        &meta.added_tool_names,
                    ),
                )?;
                for (idx, block) in blocks.iter().enumerate() {
                    tx.execute(
                        "INSERT INTO entry_blocks(session_id, seq, idx, kind, hash, mime_type, tool_call_id, tool_name)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                        (
                            &sid,
                            seq,
                            idx as i64,
                            block.kind.as_str(),
                            &block.hash,
                            &block.mime_type,
                            &block.tool_call_id,
                            &block.tool_name,
                        ),
                    )?;
                }
                tx.execute(
                    "INSERT INTO meta(key, value) VALUES (?1, ?2)
                     ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (HEAD_KEY, &sid),
                )?;
                tx.commit()?;
                Ok(seq)
            })
            .await?;
        Ok(format!("{session_id}:{seq}"))
    }

    /// Store a segment's blocks into the object store (write half of ADR-0006):
    /// encode each block to raw bytes, deduplicated by content hash, and
    /// write the bytes to the object store. The block rows themselves are
    /// inserted by the caller inside the append transaction.
    async fn store_segment_blocks(&self, segment: &Segment) -> Result<Vec<StoredBlock>> {
        let blocks: Vec<EncodedBlock> = match segment {
            Segment::User(user) => user
                .content
                .iter()
                .map(encode_content_block)
                .collect::<Result<_>>()?,
            Segment::Assistant(assistant) => assistant
                .content
                .iter()
                .map(encode_content_block)
                .collect::<Result<_>>()?,
            Segment::ToolResult(result) => {
                let mut blocks = result
                    .content
                    .iter()
                    .map(encode_content_block)
                    .collect::<Result<Vec<_>>>()?;
                // The details object is the canonical JSON of `details`;
                // its block row doubles as the message-level tool
                // attribution (id/name), which has no other column.
                blocks.push(EncodedBlock {
                    kind: BlockKind::Details,
                    bytes: canonical_json(&result.details)?,
                    mime_type: None,
                    tool_call_id: Some(result.tool_call_id.clone()),
                    tool_name: Some(result.tool_name.clone()),
                });
                blocks
            }
            Segment::Summary(summary) => vec![EncodedBlock {
                kind: BlockKind::Summary,
                bytes: summary.content.as_bytes().to_vec(),
                mime_type: None,
                tool_call_id: None,
                tool_name: None,
            }],
        };
        let mut out = Vec::with_capacity(blocks.len());
        for block in blocks {
            out.push(StoredBlock {
                kind: block.kind,
                hash: self.objects.put_bytes(&block.bytes).await?,
                mime_type: block.mime_type,
                tool_call_id: block.tool_call_id,
                tool_name: block.tool_name,
            });
        }
        Ok(out)
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

    /// [`materialize_view`](Self::materialize_view) without the coordinates
    /// or write timestamps.
    pub async fn materialize_plain(&self, session_id: &str) -> Result<Vec<Segment>> {
        Ok(self
            .materialize_view(session_id)
            .await?
            .into_iter()
            .map(|(seg, _, _)| seg)
            .collect())
    }

    /// Materialize with each segment's true write time — `entries.created_at`
    /// for entries, the compaction derivation's `sessions.created_at` for
    /// fold summaries. The resume path stamps agent messages with these
    /// instead of `now()` (ADR-0006 timestamp fix).
    pub(crate) async fn materialize_view(
        &self,
        session_id: &str,
    ) -> Result<Vec<(Segment, String, u64)>> {
        let items = self.view_items(session_id, None).await?;
        let mut view = Vec::with_capacity(items.len());
        for item in &items {
            let (segment, created_at) = match item {
                ViewItem::Entry { session_id, seq } => {
                    let (segment, created_at) = self.segment_from_entry(session_id, *seq).await?;
                    (segment, created_at as u64)
                }
                ViewItem::Summary { session_id, seq } => (
                    self.segment_from_summary(session_id, *seq).await?,
                    self.session_created_at(session_id).await? as u64,
                ),
            };
            let coordinate = item.coordinate();
            view.push((segment, coordinate, created_at));
        }
        Ok(view)
    }

    /// Rebuild the segment a view item points at: an entry from its row +
    /// block rows, a fold summary from its compaction row.
    async fn segment_from_item(&self, item: &ViewItem) -> Result<Segment> {
        match item {
            ViewItem::Entry { session_id, seq } => {
                self.segment_from_entry(session_id, *seq).await.map(|s| s.0)
            }
            ViewItem::Summary { session_id, seq } => {
                self.segment_from_summary(session_id, *seq).await
            }
        }
    }

    /// Decode the content blocks (Text, Thinking, Image, ToolCall) of an
    /// entry, in block order, into `ContentBlock`s. The special block kinds
    /// (Details, Summary) belong to their entry kind's own decode path.
    async fn decode_content_blocks(&self, rows: &[BlockRow]) -> Result<Vec<ContentBlock>> {
        let mut content = Vec::with_capacity(rows.len());
        for row in rows {
            match row.kind {
                BlockKind::Text => {
                    content.push(ContentBlock::Text {
                        text: utf8(self.objects.get_bytes(&row.hash).await?)?,
                    });
                }
                BlockKind::Thinking => {
                    content.push(ContentBlock::Thinking {
                        text: utf8(self.objects.get_bytes(&row.hash).await?)?,
                    });
                }
                BlockKind::Image => {
                    content.push(ContentBlock::Image {
                        source: ImageSource {
                            mime_type: row.mime_type.clone().ok_or_else(|| {
                                StoreError::Decode("image block without mime_type".into())
                            })?,
                            bytes: self.objects.get_bytes(&row.hash).await?,
                        },
                    });
                }
                BlockKind::ToolCall => {
                    content.push(ContentBlock::ToolCall(ToolCall {
                        id: row.tool_call_id.clone().ok_or_else(|| {
                            StoreError::Decode("tool_call block without id".into())
                        })?,
                        name: row.tool_name.clone().ok_or_else(|| {
                            StoreError::Decode("tool_call block without name".into())
                        })?,
                        arguments: from_canonical_json(&self.objects.get_bytes(&row.hash).await?)?,
                    }));
                }
                // Not a content block; the entry kind that stores it
                // decodes it on its own path.
                BlockKind::Details | BlockKind::Summary => {}
            }
        }
        Ok(content)
    }

    async fn segment_from_entry(&self, session_id: &str, seq: i64) -> Result<(Segment, i64)> {
        let (sid, seq) = (session_id.to_string(), seq);
        let (entry, rows) = self
            .db
            .call(move |conn| -> rusqlite::Result<(EntryRow, Vec<RawBlockRow>)> {
                let entry = conn.query_row(
                    "SELECT kind, created_at, stop_reason, model, provider, api, usage, error_message, is_error, added_tool_names
                     FROM entries WHERE session_id = ?1 AND seq = ?2",
                    (&sid, seq),
                    |r| {
                        Ok(EntryRow {
                            kind: r.get(0)?,
                            created_at: r.get(1)?,
                            stop_reason: r.get(2)?,
                            model: r.get(3)?,
                            provider: r.get(4)?,
                            api: r.get(5)?,
                            usage: r.get(6)?,
                            error_message: r.get(7)?,
                            is_error: r.get::<_, Option<i64>>(8)?.map(|v| v != 0),
                            added_tool_names: r.get(9)?,
                        })
                    },
                )?;
                let mut stmt = conn.prepare(
                    "SELECT kind, hash, mime_type, tool_call_id, tool_name
                     FROM entry_blocks WHERE session_id = ?1 AND seq = ?2 ORDER BY idx",
                )?;
                let rows = stmt
                    .query_map((&sid, seq), |r| {
                        Ok((
                            r.get::<_, String>(0)?,
                            r.get::<_, String>(1)?,
                            r.get::<_, Option<String>>(2)?,
                            r.get::<_, Option<String>>(3)?,
                            r.get::<_, Option<String>>(4)?,
                        ))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                Ok((entry, rows))
            })
            .await?;
        let rows: Vec<BlockRow> = rows
            .into_iter()
            .map(|(kind, hash, mime_type, tool_call_id, tool_name)| {
                Ok(BlockRow {
                    kind: BlockKind::from_str(&kind)?,
                    hash,
                    mime_type,
                    tool_call_id,
                    tool_name,
                })
            })
            .collect::<Result<_>>()?;
        let EntryRow {
            kind: entry_kind,
            created_at,
            stop_reason,
            model,
            provider,
            api,
            usage,
            error_message,
            is_error,
            added_tool_names,
        } = entry;

        // Decode dispatches on the entry kind first: each path handles only
        // the block kinds that kind can contain (ADR-0006), in its own
        // local scope — no cross-block ordering contract.

        let segment = match entry_kind.as_str() {
            "user" => Segment::User(UserSegment {
                content: self.decode_content_blocks(&rows).await?,
            }),
            "assistant" => Segment::Assistant(AssistantSegment {
                content: self.decode_content_blocks(&rows).await?,
                stop_reason: from_json(stop_reason.as_deref().ok_or_else(|| {
                    StoreError::Decode("assistant entry without stop_reason".into())
                })?)?,
                model: model
                    .ok_or_else(|| StoreError::Decode("assistant entry without model".into()))?,
                provider: provider
                    .ok_or_else(|| StoreError::Decode("assistant entry without provider".into()))?,
                api: api.ok_or_else(|| StoreError::Decode("assistant entry without api".into()))?,
                usage: from_json(
                    usage.as_deref().ok_or_else(|| {
                        StoreError::Decode("assistant entry without usage".into())
                    })?,
                )?,
                error_message,
            }),
            "tool_result" => {
                // Content blocks plus the one Details block; the Details row
                // also carries the message-level tool attribution (id/name),
                // which has no other column.

                let content = self.decode_content_blocks(&rows).await?;
                let mut details: Option<Value> = None;
                let mut result_ids: Option<(String, String)> = None;
                for row in &rows {
                    if row.kind != BlockKind::Details {
                        continue;
                    }
                    let bytes = self.objects.get_bytes(&row.hash).await?;
                    details = Some(from_canonical_json(&bytes)?);
                    result_ids = Some((
                        row.tool_call_id.clone().ok_or_else(|| {
                            StoreError::Decode("tool result without tool_call_id".into())
                        })?,
                        row.tool_name.clone().ok_or_else(|| {
                            StoreError::Decode("tool result without tool_name".into())
                        })?,
                    ));
                }
                let (tool_call_id, tool_name) = result_ids.ok_or_else(|| {
                    StoreError::Decode("tool result without details block".into())
                })?;
                Segment::ToolResult(ToolResultSegment {
                    tool_call_id,
                    tool_name,
                    content,
                    details: details.unwrap_or(Value::Null),
                    usage: usage.as_deref().map(from_json).transpose()?,
                    added_tool_names: added_tool_names.as_deref().map(from_json).transpose()?,
                    is_error: is_error.unwrap_or(false),
                })
            }
            "summary" => {
                // A directly appended summary is exactly one Summary text
                // block; it has no `compactions` row, so nothing else is
                // persisted for it.
                let mut summary_text: Option<String> = None;
                for row in &rows {
                    if row.kind != BlockKind::Summary {
                        continue;
                    }
                    let bytes = self.objects.get_bytes(&row.hash).await?;
                    summary_text = Some(utf8(bytes)?);
                }
                Segment::Summary(SummarySegment {
                    content: summary_text.ok_or_else(|| {
                        StoreError::Decode("summary entry without summary block".into())
                    })?,
                })
            }
            other => {
                return Err(StoreError::Decode(format!(
                    "unknown entries kind {other:?}"
                )));
            }
        };

        Ok((segment, created_at))
    }

    /// Rebuild a fold summary from its `compactions` row: the raw summary
    /// text object the map points at.
    async fn segment_from_summary(&self, session_id: &str, seq: i64) -> Result<Segment> {
        let (sid, seq) = (session_id.to_string(), seq);
        let summary_hash = self
            .db
            .call(move |conn| -> rusqlite::Result<String> {
                conn.query_row(
                    "SELECT summary_hash FROM compactions
                     WHERE session_id = ?1 AND seq = ?2",
                    (&sid, seq),
                    |r| r.get(0),
                )
            })
            .await?;
        let content = utf8(self.objects.get_bytes(&summary_hash).await?)?;
        Ok(Segment::Summary(SummarySegment::new(content)))
    }

    async fn session_created_at(&self, session_id: &str) -> Result<i64> {
        let sid = session_id.to_string();
        Ok(self
            .db
            .call(move |conn| -> rusqlite::Result<i64> {
                conn.query_row(
                    "SELECT created_at FROM sessions WHERE id = ?1",
                    (&sid,),
                    |r| r.get(0),
                )
            })
            .await?)
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
    /// into the parent view at derivation time — with `summary`. The summary
    /// text is stored as a raw content object the map's row points at
    /// (ADR-0006). Consecutive covered slots sharing one summary hash
    /// collapse into a single view item. The parent stays intact and
    /// re-derivable (undo = derive from it).
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
        let Segment::Summary(summary) = summary else {
            return Err(StoreError::InvalidDerivation {
                context: "compaction".into(),
                reason: "summary must be a Summary segment".into(),
            });
        };
        let summary_hash = self.objects.put_bytes(summary.content.as_bytes()).await?;
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
    /// derivation time, sliced per map. The one provenance route —
    /// [`SummarySegment`] carries no sources (ADR-0006 revision of
    /// ADR-0001's dual-track provenance).
    pub async fn fetch_originals(&self, session_id: &str) -> Result<Vec<CoveredRange>> {
        let row = self.session_row(session_id).await?;
        let Some(parent_id) = &row.parent_id else {
            return Ok(Vec::new());
        };
        let maps = self.compaction_maps(&row.id).await?;
        // Parent view truncated to the inherited prefix = what the maps index.
        let items = self.view_items(parent_id, row.parent_position).await?;
        let mut out = Vec::with_capacity(maps.len());
        for (_, start, end, _) in &maps {
            let mut originals = Vec::with_capacity((*end - *start) as usize);
            for item in &items[*start as usize..*end as usize] {
                originals.push(self.segment_from_item(item).await?);
            }
            out.push(CoveredRange {
                start: *start as u64,
                end: *end as u64,
                originals,
            });
        }
        Ok(out)
    }

    /// A compact session's map rows in seq order:
    /// `(seq, start, end, summary_hash)`.
    async fn compaction_maps(&self, session_id: &str) -> Result<Vec<(i64, i64, i64, String)>> {
        self.select_session(
            "SELECT seq, start, end, summary_hash FROM compactions
                 WHERE session_id = ?1 ORDER BY seq",
            session_id,
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .await
    }

    /// View items of `session_id` truncated to `limit` (None = full view).
    /// Same chain fold as [`materialize_view`](Self::materialize_view), coordinates
    /// only: cardinals are `(session_id, seq)` entries and compaction
    /// summaries, still in folded coordinates.
    async fn view_items(&self, session_id: &str, limit: Option<i64>) -> Result<Vec<ViewItem>> {
        let chain = self.chain_rows(session_id).await?;
        let mut items: Vec<ViewItem> = Vec::new();
        for row in chain.iter().rev() {
            if let Some(pos) = row.parent_position {
                items.truncate(pos as usize);
            }
            if row.kind == SessionKind::Compact {
                self.apply_compaction_items(&mut items, &row.id).await?;
            }
            items.extend(self.entry_items(&row.id).await?);
        }
        if let Some(limit) = limit {
            items.truncate(limit as usize);
        }
        Ok(items)
    }

    /// A session's own entry coordinates in seq order.
    async fn entry_items(&self, session_id: &str) -> Result<Vec<ViewItem>> {
        let sid = session_id.to_string();
        let seqs = self
            .select_session(
                "SELECT seq FROM entries WHERE session_id = ?1 ORDER BY seq",
                session_id,
                |r| r.get::<_, i64>(0),
            )
            .await?;
        Ok(seqs
            .into_iter()
            .map(|seq| ViewItem::Entry {
                session_id: sid.clone(),
                seq,
            })
            .collect())
    }

    /// Apply a compact session's map rows to its inherited prefix: per-slot
    /// assignment in seq order, consecutive slots sharing a summary hash
    /// collapse to one summary item.
    async fn apply_compaction_items(
        &self,
        items: &mut Vec<ViewItem>,
        session_id: &str,
    ) -> Result<()> {
        let maps = self.compaction_maps(session_id).await?;
        if maps.is_empty() {
            return Ok(());
        }
        let mut slots = vec![None::<(String, String, i64)>; items.len()];
        for (seq, start, end, summary_hash) in &maps {
            for slot in slots.iter_mut().take(*end as usize).skip(*start as usize) {
                *slot = Some((summary_hash.clone(), session_id.to_string(), *seq));
            }
        }
        let mut folded: Vec<ViewItem> = Vec::with_capacity(items.len());
        let mut last_summary_hash: Option<String> = None;
        for (item, slot) in items.drain(..).zip(slots) {
            match slot {
                Some((hash, sid, seq)) => {
                    if last_summary_hash.as_deref() != Some(hash.as_str()) {
                        folded.push(ViewItem::Summary {
                            session_id: sid,
                            seq,
                        });
                        last_summary_hash = Some(hash);
                    }
                }
                None => {
                    folded.push(item);
                    last_summary_hash = None;
                }
            }
        }
        *items = folded;
        Ok(())
    }

    /// The head pointer (ADR-0003): the session node last appended to, or
    /// `None` on a store that has never seen an append (fresh, or created
    /// before the pointer existed). This is the default resume target —
    /// "the session the user was last writing". Under concurrent processes the
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
        Ok(self.view_items(session_id, None).await?.len() as i64)
    }
}

impl ViewItem {
    /// The item's coordinate string — the segment's store identity in a
    /// materialized view (the old per-message hash has no successor under
    /// block granularity, ADR-0006).
    fn coordinate(&self) -> String {
        match self {
            ViewItem::Entry { session_id, seq } | ViewItem::Summary { session_id, seq } => {
                format!("{session_id}:{seq}")
            }
        }
    }
}

/// Decode raw object bytes back to a string; block text is UTF-8 by
/// definition (ADR-0006).
fn utf8(bytes: Vec<u8>) -> Result<String> {
    String::from_utf8(bytes).map_err(|e| StoreError::Decode(e.to_string()))
}

fn from_canonical_json(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice(bytes).map_err(|e| StoreError::Decode(e.to_string()))
}

fn from_json<T: serde::de::DeserializeOwned>(json: &str) -> Result<T> {
    serde_json::from_str(json).map_err(|e| StoreError::Decode(e.to_string()))
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

    /// ADR-0006: `entries.created_at` records the true write time, and the
    /// internal view surfaces it (the resume path stamps agent messages
    /// with it instead of `now()`).
    #[tokio::test]
    async fn view_carries_true_created_at() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        let before = crate::now_ms();
        store
            .append_segment(&root, &Segment::user_text("q"))
            .await
            .unwrap();
        let after = crate::now_ms();
        let view = store.materialize_view(&root).await.unwrap();
        let created_at = view[0].2;
        assert!(
            (before..=after).contains(&created_at),
            "created_at must be the true write time: {created_at} not in [{before}, {after}]"
        );
    }

    /// The append return and the materialize coordinate agree: the
    /// `"<session_id>:<seq>"` identity.
    #[tokio::test]
    async fn append_returns_the_view_coordinate() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        let first = store
            .append_segment(&root, &Segment::user_text("a"))
            .await
            .unwrap();
        let second = store
            .append_segment(&root, &Segment::user_text("b"))
            .await
            .unwrap();
        assert_eq!(first, format!("{root}:0"));
        assert_eq!(second, format!("{root}:1"));
        let view = store.materialize_view(&root).await.unwrap();
        let coords: Vec<&String> = view.iter().map(|(_, c, _)| c).collect();
        assert_eq!(coords, vec![&first, &second]);
    }
}
