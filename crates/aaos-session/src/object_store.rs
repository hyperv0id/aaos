//! Content-addressed object store.
//!
//! Objects are write-once and globally deduplicated: identity is the
//! BLAKE3-256 hash of the content, physical path is
//! `<store>/objects/<first-2-hex>/<hash>`. Existing hash → no-op; writes go
//! to a unique `.tmp-*` then rename, so concurrent same-content writes are
//! safe (the loser's rename replaces identical bytes).

use std::path::PathBuf;

use crate::error::{Result, StoreError};
use crate::segment::Segment;

/// Canonical content encoding and hashing.
///
/// Canonical bytes = `serde_json::to_vec(segment)`. The hash is stable
/// because serde_json is deterministic, not because it sorts: struct fields
/// serialize in declaration order (fixed at compile time), and
/// `Value::Object` keys serialize in sorted order via `BTreeMap`.
/// Identity = BLAKE3-256 hex.
///
/// NOTE: the sorted-key property for `Value::Object` holds only while
/// `serde_json`'s `preserve_order` feature is not enabled anywhere in the
/// dependency graph. Enabling it switches `Value::Object` to `IndexMap`
/// (insertion order), breaking content-addressing for the `details` and
/// `arguments` fields. Never add `preserve_order` to a crate that links
/// `aaos-session`.
pub(crate) fn hash_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

pub(crate) fn canonical_bytes(segment: &Segment) -> Result<Vec<u8>> {
    serde_json::to_vec(segment).map_err(|e| StoreError::Encode(e.to_string()))
}

#[derive(Debug, Clone)]
pub struct ObjectStore {
    root: PathBuf,
}

impl ObjectStore {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    pub fn object_path(&self, hash: &str) -> Result<PathBuf> {
        validate_hash(hash)?;
        Ok(self.root.join("objects").join(&hash[..2]).join(hash))
    }

    pub async fn put_bytes(&self, bytes: &[u8]) -> Result<String> {
        let hash = hash_hex(bytes);
        let dir = self.root.join("objects").join(&hash[..2]);
        let path = dir.join(&hash);
        if tokio::fs::try_exists(&path).await? {
            return Ok(hash);
        }
        tokio::fs::create_dir_all(&dir).await?;
        let tmp = dir.join(format!(".tmp-{}-{}", std::process::id(), crate::new_id()));
        tokio::fs::write(&tmp, bytes).await?;
        tokio::fs::rename(&tmp, &path).await?;
        Ok(hash)
    }

    pub async fn get_bytes(&self, hash: &str) -> Result<Vec<u8>> {
        let path = self.object_path(hash)?;
        match tokio::fs::read(&path).await {
            Ok(bytes) => Ok(bytes),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                Err(StoreError::NotFound(hash.to_string()))
            }
            Err(e) => Err(e.into()),
        }
    }

    pub async fn put(&self, segment: &Segment) -> Result<String> {
        self.put_bytes(&canonical_bytes(segment)?).await
    }

    pub async fn get(&self, hash: &str) -> Result<Segment> {
        let bytes = self.get_bytes(hash).await?;
        serde_json::from_slice(&bytes).map_err(|e| StoreError::Decode(e.to_string()))
    }

    pub async fn contains(&self, hash: &str) -> Result<bool> {
        Ok(tokio::fs::try_exists(self.object_path(hash)?).await?)
    }
}

fn validate_hash(hash: &str) -> Result<()> {
    let ok = hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    if ok {
        Ok(())
    } else {
        Err(StoreError::InvalidHash(hash.to_string()))
    }
}
