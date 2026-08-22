//! Branch writer — the single-writer append handle for one branch log.
//!
//! Records are appended with a synchronous std write on the blocking pool:
//! the write is immediately visible to every reader in the process (a
//! buffered `tokio::fs::File` handle is not — its `write_all` can resolve
//! before the data reaches the file). `flush()` is the durability barrier
//! (fsync); crash consistency relies on torn-tail truncation, not per-record
//! fsync.

use std::ops::Range;
use std::path::PathBuf;

use crate::branch::{create_log_with_header, Branch};
use crate::error::{Result, StoreError};
use crate::log::{
    encode_log_record, BranchKind, CompactMapRecord, HeaderRecord, LogRecord, SegmentRefRecord,
    SideEffectRecord,
};
use crate::object_store::ObjectStore;
use crate::refs::{self, SessionHead};
use crate::segment::{Segment, SummarySegment};
use crate::{new_id, now_ms};

#[derive(Debug)]
pub struct BranchWriter {
    store_root: PathBuf,
    session_id: String,
    log_relpath: String,
    objects: ObjectStore,
    position: u64,
    side_effect_seq: u64,
}

impl BranchWriter {
    /// Open (and torn-recover) an existing branch log for appending.
    pub async fn open(
        store_root: impl Into<PathBuf>,
        session_id: impl Into<String>,
        log_relpath: impl Into<String>,
    ) -> Result<Self> {
        let store_root = store_root.into();
        let session_id = session_id.into();
        let log_relpath = log_relpath.into();
        let branch = Branch::open(&store_root, &log_relpath).await?;
        let mut side_effect_seq = branch.header.inherited_seq;
        for (record, _) in &branch.records {
            if let LogRecord::SideEffect(se) = record {
                side_effect_seq = side_effect_seq.max(se.seq);
            }
        }
        Ok(Self {
            objects: ObjectStore::new(store_root.clone()),
            store_root,
            session_id,
            log_relpath,
            position: branch.log_len,
            side_effect_seq,
        })
    }

    pub fn position(&self) -> u64 {
        self.position
    }

    pub fn log_relpath(&self) -> &str {
        &self.log_relpath
    }

    pub fn side_effect_seq(&self) -> u64 {
        self.side_effect_seq
    }

    pub fn objects(&self) -> &ObjectStore {
        &self.objects
    }

    /// Put the segment into the object store and append a segment-ref record.
    /// Returns the segment hash.
    pub async fn append_segment(&mut self, segment: &Segment) -> Result<String> {
        let hash = self.objects.put(segment).await?;
        self.write_record(&LogRecord::SegmentRef(SegmentRefRecord {
            hash: hash.clone(),
            kind: segment.kind().to_string(),
            ts: now_ms(),
        }))
        .await?;
        Ok(hash)
    }

    /// Fork a subagent branch sharing this branch's current prefix.
    /// HEAD is untouched — the parent remains the session's main line.
    pub async fn fork(&mut self) -> Result<BranchWriter> {
        self.spawn_child(BranchKind::Subagent, Vec::new()).await
    }

    /// Compact: create a compact fork replacing `mappings` ranges of this
    /// branch's current view with summary objects (each carrying the hashes
    /// of the segments it replaces). If this branch is the one HEAD points
    /// at, HEAD moves to the new log. The parent log stays intact.
    pub async fn compact(
        &mut self,
        mappings: Vec<(Range<u64>, SummarySegment)>,
    ) -> Result<BranchWriter> {
        let mut maps = Vec::with_capacity(mappings.len());
        for (range, summary) in mappings {
            if range.start >= range.end {
                return Err(StoreError::InvalidLog {
                    context: self.log_relpath.clone(),
                    reason: format!("invalid compaction range {}..{}", range.start, range.end),
                });
            }
            let summary_hash = self.objects.put(&Segment::Summary(summary)).await?;
            maps.push(CompactMapRecord {
                start: range.start,
                end: range.end,
                summary_hash,
                ts: now_ms(),
            });
        }
        self.spawn_child(BranchKind::Compact, maps).await
    }

    /// Record a tool side-effect. before/after payloads are content-addressed
    /// into the object store; the record carries a session-monotonic seq.
    pub async fn append_side_effect(
        &mut self,
        tool_call_id: impl Into<String>,
        before: Option<Vec<u8>>,
        after: Option<Vec<u8>>,
        path: impl Into<String>,
    ) -> Result<u64> {
        let before_hash = match before {
            Some(bytes) => Some(self.objects.put_bytes(&bytes).await?),
            None => None,
        };
        let after_hash = match after {
            Some(bytes) => Some(self.objects.put_bytes(&bytes).await?),
            None => None,
        };
        self.side_effect_seq += 1;
        let seq = self.side_effect_seq;
        self.write_record(&LogRecord::SideEffect(SideEffectRecord {
            seq,
            tool_call_id: tool_call_id.into(),
            before_hash,
            after_hash,
            path: path.into(),
            ts: now_ms(),
        }))
        .await?;
        Ok(seq)
    }

    /// Checkpoint: persist HEAD at the current position and return it.
    pub async fn snapshot(&self) -> Result<SessionHead> {
        let head = SessionHead {
            log_relpath: self.log_relpath.clone(),
            position: self.position,
        };
        refs::write_head(&self.store_root, &self.session_id, &head).await?;
        Ok(head)
    }

    /// Durability barrier: fsync the log file.
    pub async fn flush(&mut self) -> Result<()> {
        let path = self.store_root.join(&self.log_relpath);
        crate::blocking_io(move || {
            std::fs::OpenOptions::new().write(true).open(&path)?.sync_all()
        })
        .await
    }

    async fn spawn_child(
        &mut self,
        kind: BranchKind,
        maps: Vec<CompactMapRecord>,
    ) -> Result<BranchWriter> {
        // The child pins the parent prefix at this position — make sure the
        // parent's bytes are durable before anything can reference them.
        self.flush().await?;
        let child_relpath = format!("sessions/{}/logs/{}.log", self.session_id, new_id());
        let header = HeaderRecord {
            kind,
            parent_log: Some(self.log_relpath.clone()),
            parent_position: Some(self.position),
            created_at: now_ms(),
            inherited_seq: self.side_effect_seq,
        };
        create_log_with_header(&self.store_root, &child_relpath, header).await?;
        let mut child = BranchWriter::open(&self.store_root, &self.session_id, child_relpath)
            .await?;
        for map in maps {
            child.write_record(&LogRecord::CompactMap(map)).await?;
        }
        child.flush().await?;

        if kind == BranchKind::Compact {
            // HEAD follows the main line: move it only when compacting the
            // branch it currently points at (a subagent compacting its own
            // branch leaves HEAD alone).
            let head = refs::read_head(&self.store_root, &self.session_id).await?;
            if head.log_relpath == self.log_relpath {
                refs::write_head(
                    &self.store_root,
                    &self.session_id,
                    &SessionHead {
                        log_relpath: child.log_relpath.clone(),
                        position: child.position,
                    },
                )
                .await?;
            }
        }
        Ok(child)
    }

    async fn write_record(&mut self, record: &LogRecord) -> Result<()> {
        let bytes = encode_log_record(record)?;
        let path = self.store_root.join(&self.log_relpath);
        let len = bytes.len() as u64;
        crate::blocking_io(move || {
            use std::io::Write;
            let mut file = std::fs::OpenOptions::new().append(true).open(&path)?;
            file.write_all(&bytes)
        })
        .await?;
        self.position += len;
        Ok(())
    }
}
