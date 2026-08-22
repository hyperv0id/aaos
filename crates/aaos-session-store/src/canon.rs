//! Canonical content encoding and hashing.
//!
//! Canonical bytes = `serde_json::to_vec(segment)`; serde_json emits object
//! keys in sorted order by default, so equal content hashes equal without a
//! separate sort pass. Identity = BLAKE3-256 hex.

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
