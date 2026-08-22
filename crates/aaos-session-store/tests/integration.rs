//! Full lifecycle: create → append → subagent fork → compact → chained
//! compact → undo via parent log → snapshot → rollback → resume.

use aaos_session_store::branch::Branch;
use aaos_session_store::view::{fetch_originals, materialize};
use aaos_session_store::{
    create_session, open_current, read_head, resume, rollback, Segment, SummarySegment,
};

async fn view(
    root: &std::path::Path,
    objects: &aaos_session_store::ObjectStore,
    log: &str,
) -> Vec<aaos_session_store::ViewItem> {
    let branch = Branch::open(root, log).await.unwrap();
    materialize(objects, &branch).await.unwrap()
}

#[tokio::test]
async fn full_lifecycle_roundtrip() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path();

    // Root: three segments.
    let (sid, mut w) = create_session(root, "main").await.unwrap();
    let _h1 = w.append_segment(&Segment::user_text("q1")).await.unwrap();
    let h2 = w.append_segment(&Segment::assistant_text("a1")).await.unwrap();
    let h3 = w.append_segment(&Segment::tool_result_text("c1", "r1")).await.unwrap();
    let root_log = w.log_relpath().to_string();
    assert_eq!(view(root, w.objects(), &root_log).await.len(), 3);

    // Subagent fork: inherits the 3-segment prefix, appends its own 2.
    let mut sub = w.fork().await.unwrap();
    sub.append_segment(&Segment::user_text("sub-q")).await.unwrap();
    sub.append_segment(&Segment::assistant_text("sub-a")).await.unwrap();
    let sub_log = sub.log_relpath().to_string();
    let sub_view = view(root, sub.objects(), &sub_log).await;
    assert_eq!(sub_view.len(), 5);
    // Fork does not move HEAD.
    assert_eq!(read_head(root, &sid).await.unwrap().log_relpath, root_log);

    // Main line compacts the middle turn: [1..3) -> summary(h2, h3).
    let summary = SummarySegment::new("turn 1 summarized", vec![h2.clone(), h3.clone()]);
    let mut compacted = w.compact(vec![(1..3, summary)]).await.unwrap();
    // HEAD followed the main line.
    assert_eq!(read_head(root, &sid).await.unwrap().log_relpath, compacted.log_relpath());

    let items = view(root, compacted.objects(), compacted.log_relpath()).await;
    assert_eq!(items.len(), 2);
    assert!(matches!(items[0].segment, Segment::User(_)));
    let summary_obj = match &items[1].segment {
        Segment::Summary(s) => s.clone(),
        other => panic!("expected summary, got {other:?}"),
    };
    let originals = fetch_originals(compacted.objects(), &summary_obj).await.unwrap();
    assert_eq!(originals.len(), 2);

    // Keep talking on the compacted branch.
    let h4 = compacted.append_segment(&Segment::user_text("q2")).await.unwrap();
    assert_eq!(view(root, compacted.objects(), compacted.log_relpath()).await.len(), 3);

    // Chained compaction: fold the first summary + q2 into one.
    let s1_hash = items[1].hash.clone();
    let s2 = SummarySegment::new("everything summarized", vec![s1_hash, h4.clone()]);
    let mut chained = compacted.compact(vec![(1..3, s2)]).await.unwrap();
    let items = view(root, chained.objects(), chained.log_relpath()).await;
    assert_eq!(items.len(), 2);
    assert!(matches!(items[0].segment, Segment::User(_)));
    assert!(matches!(items[1].segment, Segment::Summary(_)));

    // Undo compaction: the parent logs are untouched.
    assert_eq!(view(root, chained.objects(), &root_log).await.len(), 3);
    assert_eq!(view(root, chained.objects(), &sub_log).await.len(), 5);

    // Side effects continue across the whole chain.
    let seq = chained
        .append_side_effect("call-1", None, Some(b"x".to_vec()), "/tmp/x")
        .await
        .unwrap();
    assert_eq!(seq, 1);

    // Snapshot -> append -> rollback restores the checkpoint view.
    let head = chained.snapshot().await.unwrap();
    chained.append_segment(&Segment::user_text("q3")).await.unwrap();
    assert_eq!(view(root, chained.objects(), chained.log_relpath()).await.len(), 3);
    rollback(root, &sid, &head).await.unwrap();
    let w = open_current(root, &sid).await.unwrap();
    assert_eq!(view(root, w.objects(), w.log_relpath()).await.len(), 2);

    // Resume: snapshot, append past it, resume discards back to the checkpoint.
    let head = w.snapshot().await.unwrap();
    let mut w2 = open_current(root, &sid).await.unwrap();
    w2.append_segment(&Segment::user_text("q4")).await.unwrap();
    let mut w3 = resume(root, &sid).await.unwrap();
    assert_eq!(w3.position(), head.position);
    assert_eq!(view(root, w3.objects(), w3.log_relpath()).await.len(), 2);
    w3.append_segment(&Segment::user_text("q5")).await.unwrap();
    assert_eq!(view(root, w3.objects(), w3.log_relpath()).await.len(), 3);
}
