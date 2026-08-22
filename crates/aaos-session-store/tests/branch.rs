use aaos_session_store::branch::{create_log_with_header, Branch};
use aaos_session_store::log::{
    encode_log_record, BranchKind, CompactMapRecord, HeaderRecord, LogRecord, SegmentRefRecord,
};
use aaos_session_store::{now_ms, StoreError};

fn root_header() -> HeaderRecord {
    HeaderRecord {
        kind: BranchKind::Root,
        parent_log: None,
        parent_position: None,
        created_at: now_ms(),
        inherited_seq: 0,
    }
}

fn fork_header(parent_log: &str, parent_position: u64) -> HeaderRecord {
    HeaderRecord {
        kind: BranchKind::Subagent,
        parent_log: Some(parent_log.to_string()),
        parent_position: Some(parent_position),
        created_at: now_ms(),
        inherited_seq: 3,
    }
}

fn seg_ref(hash: &str) -> LogRecord {
    LogRecord::SegmentRef(SegmentRefRecord {
        hash: hash.to_string(),
        kind: "user".to_string(),
        ts: now_ms(),
    })
}

async fn append(store_root: &std::path::Path, log: &str, record: &LogRecord) {
    let bytes = encode_log_record(record).unwrap();
    use tokio::io::AsyncWriteExt;
    let mut f = tokio::fs::OpenOptions::new()
        .append(true)
        .open(store_root.join(log))
        .await
        .unwrap();
    f.write_all(&bytes).await.unwrap();
    f.sync_all().await.unwrap();
}

#[tokio::test]
async fn header_roundtrip_root() {
    let tmp = tempfile::tempdir().unwrap();
    create_log_with_header(tmp.path(), "sessions/s/logs/a.log", root_header())
        .await
        .unwrap();
    let branch = Branch::open(tmp.path(), "sessions/s/logs/a.log").await.unwrap();
    assert_eq!(branch.header.kind, BranchKind::Root);
    assert!(branch.records.is_empty());
    assert_eq!(
        branch.log_len,
        std::fs::metadata(tmp.path().join("sessions/s/logs/a.log"))
            .unwrap()
            .len()
    );
}

#[tokio::test]
async fn header_and_segment_ref_read_back() {
    let tmp = tempfile::tempdir().unwrap();
    let log = "sessions/s/logs/a.log";
    create_log_with_header(tmp.path(), log, root_header()).await.unwrap();
    append(tmp.path(), log, &seg_ref(&"a".repeat(64))).await;

    let branch = Branch::open(tmp.path(), log).await.unwrap();
    assert_eq!(branch.records.len(), 1);
    match &branch.records[0].0 {
        LogRecord::SegmentRef(sr) => assert_eq!(sr.hash, "a".repeat(64)),
        other => panic!("expected segment ref, got {other:?}"),
    }
    assert_eq!(branch.records[0].1, branch.log_len);
}

#[tokio::test]
async fn fork_header_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let log = "sessions/s/logs/child.log";
    create_log_with_header(tmp.path(), log, fork_header("sessions/s/logs/a.log", 42))
        .await
        .unwrap();
    let branch = Branch::open(tmp.path(), log).await.unwrap();
    assert_eq!(branch.header.kind, BranchKind::Subagent);
    assert_eq!(
        branch.header.parent_log.as_deref(),
        Some("sessions/s/logs/a.log")
    );
    assert_eq!(branch.header.parent_position, Some(42));
    assert_eq!(branch.header.inherited_seq, 3);
}

#[tokio::test]
async fn duplicate_header_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let log = "sessions/s/logs/a.log";
    create_log_with_header(tmp.path(), log, root_header()).await.unwrap();
    append(
        tmp.path(),
        log,
        &LogRecord::Header(fork_header("sessions/s/logs/b.log", 0)),
    )
    .await;
    match Branch::open(tmp.path(), log).await {
        Err(StoreError::InvalidLog { reason, .. }) => assert!(reason.contains("duplicate")),
        other => panic!("expected InvalidLog, got {other:?}"),
    }
}

#[tokio::test]
async fn missing_header_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let log = "sessions/s/logs/a.log";
    let bytes = encode_log_record(&seg_ref(&"a".repeat(64))).unwrap();
    tokio::fs::create_dir_all(tmp.path().join("sessions/s/logs"))
        .await
        .unwrap();
    tokio::fs::write(tmp.path().join(log), bytes).await.unwrap();
    match Branch::open(tmp.path(), log).await {
        Err(StoreError::InvalidLog { reason, .. }) => assert!(reason.contains("not a header")),
        other => panic!("expected InvalidLog, got {other:?}"),
    }
}

#[tokio::test]
async fn compact_map_outside_compact_log_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let log = "sessions/s/logs/a.log";
    create_log_with_header(tmp.path(), log, root_header()).await.unwrap();
    append(
        tmp.path(),
        log,
        &LogRecord::CompactMap(CompactMapRecord {
            start: 0,
            end: 1,
            summary_hash: "b".repeat(64),
            ts: now_ms(),
        }),
    )
    .await;
    match Branch::open(tmp.path(), log).await {
        Err(StoreError::InvalidLog { reason, .. }) => assert!(reason.contains("compact-map")),
        other => panic!("expected InvalidLog, got {other:?}"),
    }
}

#[tokio::test]
async fn non_root_header_without_parent_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let mut header = fork_header("sessions/s/logs/a.log", 1);
    header.parent_log = None;
    let log = "sessions/s/logs/b.log";
    create_log_with_header(tmp.path(), log, header).await.unwrap();
    match Branch::open(tmp.path(), log).await {
        Err(StoreError::InvalidLog { reason, .. }) => assert!(reason.contains("parent")),
        other => panic!("expected InvalidLog, got {other:?}"),
    }
}

#[tokio::test]
async fn torn_tail_truncated_on_open() {
    let tmp = tempfile::tempdir().unwrap();
    let log = "sessions/s/logs/a.log";
    create_log_with_header(tmp.path(), log, root_header()).await.unwrap();
    append(tmp.path(), log, &seg_ref(&"a".repeat(64))).await;
    let good_len = std::fs::metadata(tmp.path().join(log)).unwrap().len();
    let content = std::fs::read(tmp.path().join(log)).unwrap();
    tokio::fs::write(tmp.path().join(log), [content, vec![9, 9, 9]].concat())
        .await
        .unwrap();

    let branch = Branch::open(tmp.path(), log).await.unwrap();
    assert_eq!(branch.log_len, good_len);
    assert_eq!(branch.records.len(), 1);
    assert_eq!(
        std::fs::metadata(tmp.path().join(log)).unwrap().len(),
        good_len
    );
}
