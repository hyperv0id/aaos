//! View materialization — what the model sees.
//!
//! Recursion along the branch chain:
//! - `view(root)` = replay segment refs
//! - `view(fork)` = `view(parent@parent_position)` + own segment refs
//! - `view(compact)` = `view(parent@parent_position)` with map ranges
//!   replaced by summaries + own segment refs
//!
//! Chain order is the priority — chained compactions resolve by
//! construction; there are no per-index conflict rules. Originals are always
//! recoverable through each summary's `sources` (content-addressed
//! provenance), independent of view positions.

use std::collections::HashSet;

use crate::branch::Branch;
use crate::error::{Result, StoreError};
use crate::log::{BranchKind, LogRecord};
use crate::object_store::ObjectStore;
use crate::segment::{Segment, SummarySegment};

#[derive(Debug, Clone, PartialEq)]
pub struct ViewItem {
    /// The visible segment (a summary where compaction replaced a range).
    pub segment: Segment,
    /// Hash of the visible object.
    pub hash: String,
}

pub async fn materialize(objects: &ObjectStore, branch: &Branch) -> Result<Vec<ViewItem>> {
    // Walk the parent chain up to the root, keeping each opened branch with
    // its prefix limit; a cyclic guard catches corrupt parent references.
    let mut chain: Vec<(Branch, Option<u64>)> = Vec::new();
    let mut visited: HashSet<String> = HashSet::new();
    let mut relpath = branch.log_relpath.clone();
    let mut prefix_limit: Option<u64> = None;
    loop {
        if !visited.insert(relpath.clone()) {
            return Err(StoreError::CyclicChain(relpath));
        }
        let current = Branch::open(&branch.store_root, &relpath).await?;
        if let Some(limit) = prefix_limit {
            if limit > current.log_len {
                return Err(StoreError::InvalidLog {
                    context: relpath.clone(),
                    reason: format!(
                        "parent position {limit} beyond log length {}",
                        current.log_len
                    ),
                });
            }
        }
        let parent = match current.header.kind {
            BranchKind::Root => None,
            BranchKind::Subagent | BranchKind::Compact => {
                let parent_log = current.header.parent_log.clone().ok_or_else(|| {
                    StoreError::InvalidLog {
                        context: relpath.clone(),
                        reason: "non-root header missing parent reference".into(),
                    }
                })?;
                let parent_position = current.header.parent_position.ok_or_else(|| {
                    StoreError::InvalidLog {
                        context: relpath.clone(),
                        reason: "non-root header missing parent position".into(),
                    }
                })?;
                Some((parent_log, parent_position))
            }
        };
        chain.push((current, prefix_limit));
        match parent {
            None => break,
            Some((parent_log, parent_position)) => {
                relpath = parent_log;
                prefix_limit = Some(parent_position);
            }
        }
    }
    chain.reverse();

    // Fold forward from the root: replay segment refs, apply each compact
    // fork's maps as they appear along the chain.
    let mut items = Vec::new();
    for (current, prefix_limit) in chain {
        if current.header.kind == BranchKind::Compact {
            items = apply_compact_maps(objects, items, &current).await?;
        }
        for (record, end) in &current.records {
            if prefix_limit.is_some_and(|limit| *end > limit) {
                continue;
            }
            if let LogRecord::SegmentRef(sr) = record {
                items.push(ViewItem {
                    segment: objects.get(&sr.hash).await?,
                    hash: sr.hash.clone(),
                });
            }
        }
    }
    Ok(items)
}

pub async fn materialize_plain(objects: &ObjectStore, branch: &Branch) -> Result<Vec<Segment>> {
    Ok(materialize(objects, branch)
        .await?
        .into_iter()
        .map(|item| item.segment)
        .collect())
}

/// Fetch the original segments behind a summary via its provenance hashes.
pub async fn fetch_originals(
    objects: &ObjectStore,
    summary: &SummarySegment,
) -> Result<Vec<Segment>> {
    let mut out = Vec::with_capacity(summary.sources.len());
    for hash in &summary.sources {
        out.push(objects.get(hash).await?);
    }
    Ok(out)
}

/// Apply a compact log's map records to the inherited parent view: per-slot
/// assignment (later records win per slot), consecutive slots sharing the
/// same summary hash collapse into one view item.
async fn apply_compact_maps(
    objects: &ObjectStore,
    parent_items: Vec<ViewItem>,
    branch: &Branch,
) -> Result<Vec<ViewItem>> {
    let mut slot_summary: Vec<Option<String>> = vec![None; parent_items.len()];
    for (record, _) in &branch.records {
        if let LogRecord::CompactMap(map) = record {
            if map.start >= map.end || map.end as usize > parent_items.len() {
                return Err(StoreError::InvalidLog {
                    context: branch.log_relpath.clone(),
                    reason: format!(
                        "map range {}..{} exceeds parent view len {}",
                        map.start,
                        map.end,
                        parent_items.len()
                    ),
                });
            }
            for slot in &mut slot_summary[map.start as usize..map.end as usize] {
                *slot = Some(map.summary_hash.clone());
            }
        }
    }

    let mut out = Vec::new();
    let mut last_summary: Option<String> = None;
    for (item, summary) in parent_items.into_iter().zip(slot_summary) {
        match summary {
            Some(hash) if last_summary.as_deref() == Some(&hash) => {}
            Some(hash) => {
                last_summary = Some(hash.clone());
                out.push(ViewItem {
                    segment: objects.get(&hash).await?,
                    hash,
                });
            }
            None => {
                last_summary = None;
                out.push(item);
            }
        }
    }
    Ok(out)
}
