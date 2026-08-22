use aaos_session_store::branch::Branch;
use aaos_session_store::view::{materialize, materialize_plain};
use aaos_session_store::{create_session, Segment};

#[tokio::test]
async fn replays_segment_refs_in_order() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    w.append_segment(&Segment::assistant_text("a1")).await.unwrap();
    w.append_segment(&Segment::user_text("q2")).await.unwrap();

    let branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    let segments = materialize_plain(w.objects(), &branch).await.unwrap();
    assert_eq!(segments.len(), 3);
    assert!(matches!(segments[0], Segment::User(_)));
    assert!(matches!(segments[1], Segment::Assistant(_)));
    assert!(matches!(segments[2], Segment::User(_)));

    let items = materialize(w.objects(), &branch).await.unwrap();
    let hash = aaos_session_store::segment_hash(&Segment::user_text("q1")).unwrap();
    assert_eq!(items[0].hash, hash);
}

#[tokio::test]
async fn fork_inherits_parent_prefix() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    w.append_segment(&Segment::assistant_text("a1")).await.unwrap();

    let mut child = w.fork().await.unwrap();
    child.append_segment(&Segment::user_text("sub")).await.unwrap();

    let child_branch = Branch::open(tmp.path(), child.log_relpath()).await.unwrap();
    let segments = materialize_plain(child.objects(), &child_branch).await.unwrap();
    assert_eq!(segments.len(), 3);

    // The parent's own view is unchanged.
    let parent_branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    assert_eq!(
        materialize_plain(w.objects(), &parent_branch).await.unwrap().len(),
        2
    );
}

#[tokio::test]
async fn map_range_beyond_parent_view_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();

    let summary = aaos_session_store::SummarySegment::new("s", vec![]);
    let compacted = w.compact(vec![(0..5, summary)]).await.unwrap();

    let branch = Branch::open(tmp.path(), compacted.log_relpath()).await.unwrap();
    match materialize(compacted.objects(), &branch).await {
        Err(aaos_session_store::StoreError::InvalidLog { reason, .. }) => {
            assert!(reason.contains("exceeds parent view"))
        }
        other => panic!("expected InvalidLog, got {other:?}"),
    }
}
