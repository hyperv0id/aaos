//! Canonical content encoding and hashing.
//!
//! Canonical bytes = `serde_json::to_vec(segment)`. The hash is stable
//! because serde_json is deterministic, not because it sorts: struct fields
//! serialize in declaration order (fixed at compile time), and `Value::Object`
//! keys serialize in sorted order via `BTreeMap`. Identity = BLAKE3-256 hex.
//!
//! NOTE: the sorted-key property for `Value::Object` holds only while
//! `serde_json`'s `preserve_order` feature is not enabled anywhere in the
//! dependency graph. Enabling it switches `Value::Object` to `IndexMap`
//! (insertion order), breaking content-addressing for the `details` and
//! `arguments` fields. Never add `preserve_order` to a crate that links
//! `aaos-session`.

use crate::error::{Result, StoreError};
use crate::segment::Segment;

pub fn hash_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

pub fn canonical_bytes(segment: &Segment) -> Result<Vec<u8>> {
    serde_json::to_vec(segment).map_err(|e| StoreError::Encode(e.to_string()))
}

pub fn segment_hash(segment: &Segment) -> Result<String> {
    canonical_bytes(segment).map(|bytes| hash_hex(&bytes))
}
