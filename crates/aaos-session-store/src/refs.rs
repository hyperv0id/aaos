//! Session refs: manifest, HEAD, and the session-level entry points.
//!
//! HEAD is the session's "you are here" pointer — `{log, position}` written
//! atomically (tmp+rename). It moves on `create_session`, on snapshot/checkout,
//! and on compaction of the branch it points at (the main line). `open_current`
//! is crash recovery: read HEAD for the log, truncate any torn tail, continue
//! at the end. `resume` is deliberate: truncate back to the HEAD checkpoint.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::branch::{create_log_with_header, Branch};
use crate::error::{Result, StoreError};
use crate::log::{BranchKind, HeaderRecord};
use crate::writer::BranchWriter;
use crate::{new_id, now_ms};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionHead {
    pub log_relpath: String,
    pub position: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionManifest {
    pub title: String,
    pub created_at: u64,
}

pub fn session_dir(store_root: &Path, session_id: &str) -> PathBuf {
    store_root.join("sessions").join(session_id)
}

/// Session ids present under `<store>/sessions`, sorted.
pub fn session_ids(store_root: &Path) -> Result<Vec<String>> {
    let dir = store_root.join("sessions");
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut ids: Vec<String> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    ids.sort();
    Ok(ids)
}

async fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let tmp = path.with_extension(format!("tmp-{}", new_id()));
    tokio::fs::write(&tmp, bytes).await?;
    tokio::fs::rename(&tmp, path).await?;
    Ok(())
}

pub async fn write_head(store_root: &Path, session_id: &str, head: &SessionHead) -> Result<()> {
    let bytes = serde_json::to_vec(head).map_err(|e| StoreError::Encode(e.to_string()))?;
    atomic_write(&session_dir(store_root, session_id).join("HEAD"), &bytes).await
}

pub fn read_head(store_root: &Path, session_id: &str) -> Result<SessionHead> {
    let path = session_dir(store_root, session_id).join("HEAD");
    let bytes = std::fs::read(&path).map_err(|e| match e.kind() {
        std::io::ErrorKind::NotFound => StoreError::NotFound(path.display().to_string()),
        _ => e.into(),
    })?;
    serde_json::from_slice(&bytes).map_err(|e| StoreError::Decode(e.to_string()))
}

/// Create a session: manifest, root log, HEAD pointing at the empty root log.
pub async fn create_session(
    store_root: impl AsRef<Path>,
    title: impl Into<String>,
) -> Result<(String, BranchWriter)> {
    let store_root = store_root.as_ref().to_path_buf();
    let session_id = new_id();
    let dir = session_dir(&store_root, &session_id);
    tokio::fs::create_dir_all(dir.join("logs")).await?;
    let manifest = SessionManifest {
        title: title.into(),
        created_at: now_ms(),
    };
    let bytes = serde_json::to_vec(&manifest).map_err(|e| StoreError::Encode(e.to_string()))?;
    atomic_write(&dir.join("session.json"), &bytes).await?;

    let log_relpath = format!("sessions/{session_id}/logs/{}.log", new_id());
    create_log_with_header(
        &store_root,
        &log_relpath,
        HeaderRecord {
            kind: BranchKind::Root,
            parent_log: None,
            parent_position: None,
            created_at: now_ms(),
            inherited_seq: 0,
        },
    )
    .await?;

    let writer = BranchWriter::open(&store_root, &session_id, log_relpath).await?;
    write_head(
        &store_root,
        &session_id,
        &SessionHead {
            log_relpath: writer.log_relpath().to_string(),
            position: writer.position(),
        },
    )
    .await?;
    Ok((session_id, writer))
}

/// Crash recovery: continue the HEAD log at its last good record. Any torn
/// tail is truncated by `Branch::open`; committed records survive.
pub async fn open_current(
    store_root: impl AsRef<Path>,
    session_id: &str,
) -> Result<BranchWriter> {
    let head = read_head(store_root.as_ref(), session_id)?;
    BranchWriter::open(store_root.as_ref(), session_id, head.log_relpath).await
}

/// Deliberate rollback to the HEAD checkpoint: records appended after the
/// last snapshot are discarded, then the writer reopens there.
pub async fn resume(store_root: impl AsRef<Path>, session_id: &str) -> Result<BranchWriter> {
    let head = read_head(store_root.as_ref(), session_id)?;
    rollback(store_root.as_ref(), session_id, &head).await?;
    BranchWriter::open(store_root.as_ref(), session_id, head.log_relpath).await
}

/// Truncate the head's log to `head.position` and persist HEAD there.
///
/// Rolling back below a child branch's parent position is the caller's
/// responsibility — children are not tracked.
pub async fn rollback(
    store_root: impl AsRef<Path>,
    session_id: &str,
    head: &SessionHead,
) -> Result<()> {
    let store_root = store_root.as_ref();
    let branch = Branch::open(store_root, &head.log_relpath).await?;
    if head.position > branch.log_len {
        return Err(StoreError::InvalidLog {
            context: head.log_relpath.clone(),
            reason: format!(
                "rollback position {} beyond log length {}",
                head.position, branch.log_len
            ),
        });
    }
    let path = store_root.join(&head.log_relpath);
    let position = head.position;
    crate::blocking_io(move || {
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)?
            .set_len(position)
    })
    .await?;
    write_head(store_root, session_id, head).await
}
