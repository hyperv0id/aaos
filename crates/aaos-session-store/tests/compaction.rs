use aaos_session_store::branch::Branch;
use aaos_session_store::view::{fetch_originals, materialize};
use aaos_session_store::{create_session, Segment, SummarySegment};

async fn view_of(
    root: &std::path::Path,
    objects: &aaos_session_store::ObjectStore,
    log: &str,
) -> Vec<aaos_session_store::ViewItem> {
    let branch = Branch::open(root, log).await.unwrap();
    materialize(objects, &branch).await.unwrap()
}

#[tokio::test]
async fn compact_replaces_range_and_originals_fetchable() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    let h = [
        w.append_segment(&Segment::user_text("q1")).await.unwrap(),
        w.append_segment(&Segment::assistant_text("a1")).await.unwrap(),
        w.append_segment(&Segment::user_text("q2")).await.unwrap(),
        w.append_segment(&Segment::assistant_text("a2")).await.unwrap(),
    ];

    let summary = SummarySegment::new("turn 1-2 summarized", vec![h[1].clone(), h[2].clone()]);
    let compacted = w.compact(vec![(1..3, summary)]).await.unwrap();

    let items = view_of(tmp.path(), compacted.objects(), compacted.log_relpath()).await;
    assert_eq!(items.len(), 3, "view shrinks by the replaced range");
    assert!(matches!(items[0].segment, Segment::User(_)));
    assert!(matches!(items[1].segment, Segment::Summary(_)));
    assert!(matches!(items[2].segment, Segment::Assistant(_)));

    let summary = match &items[1].segment {
        Segment::Summary(s) => s.clone(),
        other => panic!("expected summary, got {other:?}"),
    };
    let originals = fetch_originals(compacted.objects(), &summary).await.unwrap();
    assert_eq!(originals.len(), 2);
    assert_eq!(originals[0], Segment::assistant_text("a1"));
    assert_eq!(originals[1], Segment::user_text("q2"));
}

#[tokio::test]
async fn adjacent_ranges_collapse_into_adjacent_summaries() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    let h = [
        w.append_segment(&Segment::user_text("q1")).await.unwrap(),
        w.append_segment(&Segment::assistant_text("a1")).await.unwrap(),
        w.append_segment(&Segment::user_text("q2")).await.unwrap(),
    ];

    let s1 = SummarySegment::new("left", vec![h[0].clone()]);
    let s2 = SummarySegment::new("right", vec![h[1].clone()]);
    let compacted = w.compact(vec![(0..1, s1), (1..2, s2)]).await.unwrap();

    let items = view_of(tmp.path(), compacted.objects(), compacted.log_relpath()).await;
    assert_eq!(items.len(), 3); // S1, S2, q2
    assert_ne!(items[0].hash, items[1].hash, "adjacent distinct ranges stay distinct");
}

#[tokio::test]
async fn overlapping_maps_later_record_wins_per_slot() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    let h = [
        w.append_segment(&Segment::user_text("q1")).await.unwrap(),
        w.append_segment(&Segment::assistant_text("a1")).await.unwrap(),
        w.append_segment(&Segment::user_text("q2")).await.unwrap(),
        w.append_segment(&Segment::assistant_text("a2")).await.unwrap(),
    ];

    let s1 = SummarySegment::new("older", vec![h[0].clone(), h[1].clone()]);
    let s2 = SummarySegment::new("newer", vec![h[1].clone(), h[2].clone()]);
    let compacted = w
        .compact(vec![(0..2, s1), (1..3, s2)])
        .await
        .unwrap();

    let items = view_of(tmp.path(), compacted.objects(), compacted.log_relpath()).await;
    // slot0 -> s1, slots1-2 -> s2 (later wins), slot3 -> a2
    assert_eq!(items.len(), 3);
    assert!(matches!(items[0].segment, Segment::Summary(ref s) if s.content == "older"));
    assert!(matches!(items[1].segment, Segment::Summary(ref s) if s.content == "newer"));
    assert!(matches!(items[2].segment, Segment::Assistant(_)));
}

#[tokio::test]
async fn chained_compaction_compacts_the_compacted_view() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    w.append_segment(&Segment::assistant_text("a1")).await.unwrap();
    w.append_segment(&Segment::user_text("q2")).await.unwrap();

    let s1 = SummarySegment::new("first", vec![]);
    let mut first = w.compact(vec![(1..3, s1)]).await.unwrap();
    let items = view_of(tmp.path(), first.objects(), first.log_relpath()).await;
    assert_eq!(items.len(), 2);

    // Compact the compacted view: replace [1..2) (the summary + nothing else
    // beyond it) — chain resolves by construction.
    let s1_hash = items[1].hash.clone();
    let s2 = SummarySegment::new("second", vec![s1_hash]);
    let second = first.compact(vec![(1..2, s2)]).await.unwrap();
    let items = view_of(tmp.path(), second.objects(), second.log_relpath()).await;
    assert_eq!(items.len(), 2);
    assert!(matches!(items[1].segment, Segment::Summary(ref s) if s.content == "second"));
}

#[tokio::test]
async fn undo_compaction_via_parent_log() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    w.append_segment(&Segment::assistant_text("a1")).await.unwrap();
    w.append_segment(&Segment::user_text("q2")).await.unwrap();
    let parent_log = w.log_relpath().to_string();
    let parent_objects = w.objects().clone();

    let summary = SummarySegment::new("gone", vec![]);
    let _compacted = w.compact(vec![(0..3, summary)]).await.unwrap();

    // The parent log is untouched: materializing it still shows all three.
    let items = view_of(tmp.path(), &parent_objects, &parent_log).await;
    assert_eq!(items.len(), 3);
}
