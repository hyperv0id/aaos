//! Content-addressed session storage for aaos (ADR-0001).
//!
//! Two layers:
//! - **objects** (content): BLAKE3-addressed, write-once, globally
//!   deduplicated — segments, summaries, side-effect payloads.
//! - **structure** (fact source): one SQLite database in WAL mode,
//!   insert-only. Every structural change is a **derivation** — a new
//!   session row referencing (parent, position). 分叉 (fork) and 压缩
//!   (compaction) are the same operation with different record kinds;
//!   chain order is priority, and there are no per-index conflict rules.
//!
//! No HEAD pointer: a session id is the pointer, "latest" is a query, and
//! resume opens a chain by id. Rollback derives from a 书签 (bookmark);
//! nothing in the structure layer is ever updated or deleted.

mod canon;
pub mod db;
mod error;
mod object_store;
mod segment;

pub use canon::{canonical_bytes, hash_hex, segment_hash};
pub use db::{Bookmark, CoveredRange, SessionKind, SessionStore, SideEffectRecord};
pub use error::{Result, StoreError};
pub use object_store::ObjectStore;
pub use segment::{
    AssistantSegment, ContentBlock, Cost, ImageSource, Segment, StopReason, SummarySegment,
    ToolCall, ToolResultSegment, Usage, UserSegment,
};

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub(crate) fn new_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "{:x}-{:x}",
        now_ms(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}
