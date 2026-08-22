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
mod segment;

pub use canon::{canonical_bytes, hash_hex, segment_hash};
pub use error::{Result, StoreError};
pub use segment::{
    AssistantSegment, ContentBlock, Cost, ImageSource, Segment, StopReason, SummarySegment,
    ToolCall, ToolResultSegment, Usage, UserSegment,
};
