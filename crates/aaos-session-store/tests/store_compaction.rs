//! Ticket 03 — 压缩：区间映射派生与原文取回。
//! Seam: `SessionStore::compact` / `fetch_originals` + 链式视图。

mod common;

use aaos_session_store::{CoveredRange, Segment, SessionStore};
use common::store_with;

async fn root_with(store: &SessionStore, texts: &[&str]) -> String {
    let root = store.create_root().await.unwrap();
    for text in texts {
        store
            .append_segment(&root, &Segment::user_text(*text))
            .await
            .unwrap();
    }
    root
}

#[tokio::test]
async fn compact_replaces_range_with_summary_and_originals_fetchable() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = root_with(&store, &["q1", "q2", "q3", "q4"]).await;
    let hashes: Vec<String> = store
        .materialize(&root)
        .await
        .unwrap()
        .into_iter()
        .map(|(_, h)| h)
        .collect();

    let summary = Segment::summary("earlier turns", vec![hashes[1].clone(), hashes[2].clone()]);
    let compacted = store.compact(&root, &[(1, 3)], &summary).await.unwrap();

    let view = store.materialize_plain(&compacted).await.unwrap();
    assert_eq!(view.len(), 3);
    assert_eq!(view[0], Segment::user_text("q1"));
    assert_eq!(view[1], summary);
    assert_eq!(view[2], Segment::user_text("q4"));

    // Structural route: originals come back by map range.
    let originals = store.fetch_originals(&compacted).await.unwrap();
    assert_eq!(
        originals,
        vec![CoveredRange {
            start: 1,
            end: 3,
            originals: vec![Segment::user_text("q2"), Segment::user_text("q3")]
        }]
    );

    // Content route: sources hashes resolve in the object store.
    if let Segment::Summary(s) = &view[1] {
        for hash in &s.sources {
            assert!(store.objects().get(hash).await.is_ok());
        }
    } else {
        panic!("view[1] should be a summary");
    }

    // Undo compaction: derive from the parent — no Summary, originals direct.
    let undo = store.fork(&root).await.unwrap();
    let undone = store.materialize_plain(&undo).await.unwrap();
    assert_eq!(undone.len(), 4);
    assert!(undone.iter().all(|s| s.kind() != "summary"));

    // Stability: parent appends after compaction don't shift the map.
    store
        .append_segment(&root, &Segment::user_text("q5"))
        .await
        .unwrap();
    let still = store.materialize_plain(&compacted).await.unwrap();
    assert_eq!(still.len(), 3);
    assert_eq!(still[2], Segment::user_text("q4"));
}

#[tokio::test]
async fn consecutive_ranges_sharing_a_summary_collapse_into_one_item() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = root_with(&store, &["a", "b", "c", "d"]).await;

    let summary = Segment::summary("all of it", vec![]);
    let compacted = store
        .compact(&root, &[(0, 2), (2, 4)], &summary)
        .await
        .unwrap();

    let view = store.materialize_plain(&compacted).await.unwrap();
    assert_eq!(view.len(), 1);
    assert_eq!(view[0], summary);
}

#[tokio::test]
async fn compact_session_can_append_after_its_maps() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = root_with(&store, &["a", "b"]).await;

    let summary = Segment::summary("s", vec![]);
    let compacted = store.compact(&root, &[(0, 1)], &summary).await.unwrap();
    store
        .append_segment(&compacted, &Segment::user_text("after"))
        .await
        .unwrap();

    let view = store.materialize_plain(&compacted).await.unwrap();
    assert_eq!(view.len(), 3);
    assert_eq!(view[0], summary);
    assert_eq!(view[1], Segment::user_text("b"));
    assert_eq!(view[2], Segment::user_text("after"));
}

#[tokio::test]
async fn chained_compaction_resolves_by_chain_order() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = root_with(&store, &["q1", "q2", "q3", "q4"]).await;

    let s1 = Segment::summary("first half", vec![]);
    let c1 = store.compact(&root, &[(0, 2)], &s1).await.unwrap();
    let v1 = store.materialize_plain(&c1).await.unwrap();
    assert_eq!(v1.len(), 3);

    // Indices of the second compaction address c1's compacted view.
    let s2 = Segment::summary("second", vec![]);
    let c2 = store.compact(&c1, &[(1, 2)], &s2).await.unwrap();
    let v2 = store.materialize_plain(&c2).await.unwrap();
    assert_eq!(v2.len(), 3);
    assert_eq!(v2[0], s1);
    assert_eq!(v2[1], s2);
    assert_eq!(v2[2], Segment::user_text("q4"));
}

#[tokio::test]
async fn compact_range_beyond_parent_view_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = root_with(&store, &["only"]).await;

    let err = store
        .compact(&root, &[(0, 5)], &Segment::summary("s", vec![]))
        .await
        .unwrap_err();
    assert!(
        err.to_string().contains("beyond"),
        "expected range error, got: {err}"
    );
}
