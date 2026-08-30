//! Content-addressed session storage + Agent integration for aaos (ADR-0001, ADR-0002).
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
//! One deliberate exception to insert-only: the `meta` table holds the
//! mutable **head pointer** (ADR-0003) — the session node last appended
//! to, and the default resume target. A session id remains the precise
//! pointer, and resume opens a chain by id. Rollback derives from a 书签
//! (bookmark); nothing else in the structure layer is ever updated or
//! deleted. Multi-process access: WAL + busy timeout for writers, and
//! every id carries the creating process's pid so simultaneous launches
//! cannot collide.

pub mod agent_session;
pub mod convert;
pub mod db;
mod error;
mod object_store;
mod segment;
mod side_effects;

pub use agent_session::AgentSession;
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
    // pid + in-process counter: concurrent launches never share a pid, so
    // same-millisecond ids cannot collide on the sessions primary key.
    format!(
        "{:x}-{:x}-{:x}",
        now_ms(),
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    )
}
