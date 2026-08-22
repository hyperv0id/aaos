//! Branch log reader.
//!
//! `Branch::open` reads the log once: parses the header, walks records
//! validating structure, and truncates a torn tail in place. Logs are
//! immutable from the reader's perspective — only crash-tail truncation and
//! explicit rollback ever rewrite one.

use std::path::{Path, PathBuf};

use crate::error::{Result, StoreError};
use crate::framing::{read_record, ReadOutcome};
use crate::log::{decode_log_record, encode_log_record, BranchKind, HeaderRecord, LogRecord};

#[derive(Debug, Clone)]
pub struct Branch {
    pub store_root: PathBuf,
    pub log_relpath: String,
    pub header: HeaderRecord,
    /// Content records (header excluded) with their end byte offsets.
    pub records: Vec<(LogRecord, u64)>,
    /// Total good bytes (after torn-tail truncation, if any).
    pub log_len: u64,
}

impl Branch {
    pub async fn open(
        store_root: impl Into<PathBuf>,
        log_relpath: impl Into<String>,
    ) -> Result<Self> {
        let store_root = store_root.into();
        let log_relpath = log_relpath.into();
        let path = store_root.join(&log_relpath);
        let mut file = tokio::fs::File::open(&path).await.map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => StoreError::NotFound(log_relpath.clone()),
            _ => e.into(),
        })?;

        let mut consumed: u64 = 0;
        let mut header: Option<HeaderRecord> = None;
        let mut records = Vec::new();
        let mut torn = false;

        loop {
            match read_record(&mut file).await? {
                ReadOutcome::Eof => break,
                ReadOutcome::Torn => {
                    torn = true;
                    break;
                }
                ReadOutcome::Record(rec) => {
                    let decoded = decode_log_record(rec.tag, &rec.payload)?;
                    let rec_len = 5 + rec.payload.len() as u64;
                    match decoded {
                        LogRecord::Header(h) => {
                            if header.is_some() {
                                return Err(invalid(&log_relpath, "duplicate header record"));
                            }
                            validate_header(&log_relpath, &h)?;
                            header = Some(h);
                        }
                        other => {
                            let h = header
                                .as_ref()
                                .ok_or_else(|| invalid(&log_relpath, "first record is not a header"))?;
                            if matches!(other, LogRecord::CompactMap(_))
                                && h.kind != BranchKind::Compact
                            {
                                return Err(invalid(
                                    &log_relpath,
                                    "compact-map record outside a compact log",
                                ));
                            }
                            records.push((other, consumed + rec_len));
                        }
                    }
                    consumed += rec_len;
                }
            }
        }

        let header = header.ok_or_else(|| invalid(&log_relpath, "empty log"))?;
        if torn {
            let f = tokio::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .await?;
            f.set_len(consumed).await?;
        }
        Ok(Self {
            store_root,
            log_relpath,
            header,
            records,
            log_len: consumed,
        })
    }
}

fn invalid(log_relpath: &str, reason: &str) -> StoreError {
    StoreError::InvalidLog {
        context: log_relpath.to_string(),
        reason: reason.to_string(),
    }
}

fn validate_header(log_relpath: &str, header: &HeaderRecord) -> Result<()> {
    if header.kind != BranchKind::Root
        && (header.parent_log.is_none() || header.parent_position.is_none())
    {
        return Err(invalid(
            log_relpath,
            "non-root header missing parent reference",
        ));
    }
    Ok(())
}

/// Create a new log file containing only its header record.
pub async fn create_log_with_header(
    store_root: &Path,
    log_relpath: &str,
    header: HeaderRecord,
) -> Result<()> {
    let path = store_root.join(log_relpath);
    if let Some(dir) = path.parent() {
        tokio::fs::create_dir_all(dir).await?;
    }
    let bytes = encode_log_record(&LogRecord::Header(header))?;
    tokio::fs::write(&path, bytes).await?;
    Ok(())
}
