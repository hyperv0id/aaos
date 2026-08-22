//! Content-addressed session storage for aaos.
//!
//! Three layers, git-shaped:
//! - **objects**: content-addressed blob store ([`ObjectStore`]) — segments,
//!   summaries, side-effect payloads; write-once, globally deduplicated.
//! - **logs**: per-branch append-only record chains ([`Branch`],
//!   [`BranchWriter`]); fork and compaction are both new logs referencing a
//!   parent position — the parent is never mutated.
//! - **refs**: per-session persisted HEAD ([`SessionHead`]) — snapshot,
//!   rollback and resume are reads/writes of this pointer.
//!
//! Compaction is a fork: a new log (kind `compact`) whose map records replace
//! ranges of the parent's view with summary objects; summaries carry the
//! hashes of the segments they replace (provenance), so originals stay
//! fetchable forever.

mod canon;
mod error;
mod object_store;
mod segment;

pub use canon::{canonical_bytes, hash_hex, segment_hash};
pub use error::{Result, StoreError};
pub use object_store::ObjectStore;
pub use segment::{
    AssistantSegment, ContentBlock, Cost, ImageSource, Segment, StopReason, SummarySegment,
    ToolCall, ToolResultSegment, Usage, UserSegment,
};

pub(crate) fn now_ms() -> u64 {
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
