use aaos_session_store::branch::Branch;
use aaos_session_store::log::{BranchKind, LogRecord};
use aaos_session_store::{create_session, read_head, Segment, SummarySegment};

#[tokio::test]
async fn append_writes_ref_and_object_monotonically() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    let start = w.position();

    let h1 = w.append_segment(&Segment::user_text("q1")).await.unwrap();
    let h2 = w.append_segment(&Segment::assistant_text("a1")).await.unwrap();
    assert_ne!(h1, h2);
    assert!(w.objects().contains(&h1).await.unwrap());
    assert!(w.position() > start);

    let branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    assert_eq!(branch.records.len(), 2);
    match &branch.records[0].0 {
        LogRecord::SegmentRef(sr) => {
            assert_eq!(sr.hash, h1);
            assert_eq!(sr.kind, "user");
        }
        other => panic!("expected segment ref, got {other:?}"),
    }
}

#[tokio::test]
async fn fork_writes_parent_header_and_leaves_head() {
    let tmp = tempfile::tempdir().unwrap();
    let (sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    let parent_position = w.position();

    let child = w.fork().await.unwrap();
    let branch = Branch::open(tmp.path(), child.log_relpath()).await.unwrap();
    assert_eq!(branch.header.kind, BranchKind::Subagent);
    assert_eq!(branch.header.parent_log.as_deref(), Some(w.log_relpath()));
    assert_eq!(branch.header.parent_position, Some(parent_position));

    // HEAD stays on the parent (main line).
    assert_eq!(read_head(tmp.path(), &sid).unwrap().log_relpath, w.log_relpath());
}

#[tokio::test]
async fn compact_moves_head_for_main_line() {
    let tmp = tempfile::tempdir().unwrap();
    let (sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    let h1 = w.append_segment(&Segment::user_text("q1")).await.unwrap();
    let h2 = w.append_segment(&Segment::assistant_text("a1")).await.unwrap();

    let summary = SummarySegment::new("one turn", vec![h1.clone(), h2.clone()]);
    let compacted = w.compact(vec![(0..2, summary)]).await.unwrap();

    let branch = Branch::open(tmp.path(), compacted.log_relpath()).await.unwrap();
    assert_eq!(branch.header.kind, BranchKind::Compact);
    assert!(matches!(branch.records[0].0, LogRecord::CompactMap(_)));
    assert_eq!(read_head(tmp.path(), &sid).unwrap().log_relpath, compacted.log_relpath());
}

#[tokio::test]
async fn compact_on_subagent_branch_leaves_head() {
    let tmp = tempfile::tempdir().unwrap();
    let (sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    let h1 = w.append_segment(&Segment::user_text("q1")).await.unwrap();

    let mut sub = w.fork().await.unwrap();
    let s1 = sub.append_segment(&Segment::user_text("sub")).await.unwrap();
    let summary = SummarySegment::new("sub turn", vec![h1, s1]);
    let _compacted = sub.compact(vec![(0..2, summary)]).await.unwrap();

    // HEAD still points at the parent log.
    assert_eq!(read_head(tmp.path(), &sid).unwrap().log_relpath, w.log_relpath());
}

#[tokio::test]
async fn side_effect_seq_monotonic_from_one() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    let seq1 = w
        .append_side_effect("call-1", None, Some(b"after".to_vec()), "/tmp/x")
        .await
        .unwrap();
    let seq2 = w
        .append_side_effect("call-2", Some(b"before".to_vec()), None, "/tmp/x")
        .await
        .unwrap();
    assert_eq!((seq1, seq2), (1, 2));
}
