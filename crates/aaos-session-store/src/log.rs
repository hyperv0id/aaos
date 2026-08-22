//! Branch log record types.
//!
//! A log file is a sequence of framed records. The first must be a header
//! carrying the branch kind and (for non-root kinds) the parent reference;
//! everything after it is content: segment refs, compaction maps
//! (compact logs only), side-effects. Timestamps live here in the structure
//! layer, never in the content objects.

use serde::{Deserialize, Serialize};

use crate::error::{Result, StoreError};
use crate::framing::{
    encode_record, TAG_COMPACT_MAP, TAG_HEADER, TAG_SEGMENT_REF, TAG_SIDE_EFFECT,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BranchKind {
    Root,
    Subagent,
    Compact,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeaderRecord {
    pub kind: BranchKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_log: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_position: Option<u64>,
    pub created_at: u64,
    /// Parent's side-effect seq at fork time — keeps WAL seq monotonic per
    /// session without walking the parent chain on open.
    pub inherited_seq: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SegmentRefRecord {
    pub hash: String,
    pub kind: String,
    pub ts: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompactMapRecord {
    /// Half-open `[start, end)` index range into the parent's materialized
    /// view at `parent_position`.
    pub start: u64,
    pub end: u64,
    pub summary_hash: String,
    pub ts: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SideEffectRecord {
    pub seq: u64,
    pub tool_call_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub before_hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub after_hash: Option<String>,
    pub path: String,
    pub ts: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LogRecord {
    Header(HeaderRecord),
    SegmentRef(SegmentRefRecord),
    CompactMap(CompactMapRecord),
    SideEffect(SideEffectRecord),
}

pub fn encode_log_record(record: &LogRecord) -> Result<Vec<u8>> {
    fn json(r: impl Serialize) -> Result<Vec<u8>> {
        serde_json::to_vec(&r).map_err(|e| StoreError::Encode(e.to_string()))
    }
    let (tag, payload) = match record {
        LogRecord::Header(r) => (TAG_HEADER, json(r)?),
        LogRecord::SegmentRef(r) => (TAG_SEGMENT_REF, json(r)?),
        LogRecord::CompactMap(r) => (TAG_COMPACT_MAP, json(r)?),
        LogRecord::SideEffect(r) => (TAG_SIDE_EFFECT, json(r)?),
    };
    Ok(encode_record(tag, &payload))
}

pub fn decode_log_record(tag: u8, payload: &[u8]) -> Result<LogRecord> {
    fn json<'a, T: Deserialize<'a>>(payload: &'a [u8]) -> Result<T> {
        serde_json::from_slice(payload).map_err(|e| StoreError::Decode(e.to_string()))
    }
    match tag {
        TAG_HEADER => Ok(LogRecord::Header(json(payload)?)),
        TAG_SEGMENT_REF => Ok(LogRecord::SegmentRef(json(payload)?)),
        TAG_COMPACT_MAP => Ok(LogRecord::CompactMap(json(payload)?)),
        TAG_SIDE_EFFECT => Ok(LogRecord::SideEffect(json(payload)?)),
        other => Err(StoreError::Decode(format!(
            "unknown record tag {other:#04x}"
        ))),
    }
}
