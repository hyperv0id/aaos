//! Ticket 05 — 书签、resume 与派生回退。
//! Seam: `SessionStore::snapshot` / `snapshots` + `fork_at`（派生回退）。
//! 书签永不自动恢复：回退 = 显式从书签派生，结构层零删除。

mod common;

use aaos_session_store::Segment;
use common::store_with;

#[tokio::test]
async fn bookmark_pins_a_position_and_derivation_stops_there() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    for text in ["q1", "q2"] {
        store
            .append_segment(&root, &Segment::user_text(text))
            .await
            .unwrap();
    }

    let snap = store.snapshot(&root, "checkpoint").await.unwrap();
    assert_eq!(snap.position, 2);

    // Post-bookmark appends on the original session.
    for text in ["q3", "q4"] {
        store
            .append_segment(&root, &Segment::user_text(text))
            .await
            .unwrap();
    }

    // Rollback = derive from the bookmark: lossless on both sides.
    let back = store
        .fork_at(&snap.session_id, snap.position)
        .await
        .unwrap();
    assert_eq!(
        store.materialize_plain(&back).await.unwrap(),
        vec![Segment::user_text("q1"), Segment::user_text("q2")]
    );
    assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 4);

    let bookmarks = store.snapshots(&root).await.unwrap();
    assert_eq!(bookmarks.len(), 1);
    assert_eq!(bookmarks[0].label, "checkpoint");
    assert_eq!(bookmarks[0].position, 2);
}

#[tokio::test]
async fn pre_compaction_bookmark_derives_a_summary_free_view() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    for text in ["q1", "q2", "q3"] {
        store
            .append_segment(&root, &Segment::user_text(text))
            .await
            .unwrap();
    }
    let snap = store.snapshot(&root, "pre-compact").await.unwrap();

    // Main line continues into a compaction (the parent is never mutated).
    let compacted = store
        .compact(&root, &[(0, 2)], &Segment::summary("sum", vec![]))
        .await
        .unwrap();
    store
        .append_segment(&compacted, &Segment::user_text("after"))
        .await
        .unwrap();

    let back = store
        .fork_at(&snap.session_id, snap.position)
        .await
        .unwrap();
    let view = store.materialize_plain(&back).await.unwrap();
    assert_eq!(view.len(), 3);
    assert!(
        view.iter().all(|s| s.kind() != "summary"),
        "pre-compaction derivation must show originals"
    );
}

#[tokio::test]
async fn resume_by_id_across_reopens_keeps_committed_appends() {
    let dir = tempfile::tempdir().unwrap();
    let root = {
        let store = store_with(dir.path()).await;
        let root = store.create_root().await.unwrap();
        store
            .append_segment(&root, &Segment::user_text("first"))
            .await
            .unwrap();
        root
    };

    // "Restart": a fresh handle resumes by id and appends.
    {
        let store = store_with(dir.path()).await;
        store
            .append_segment(&root, &Segment::user_text("second"))
            .await
            .unwrap();
    }

    let store = store_with(dir.path()).await;
    assert_eq!(
        store.materialize_plain(&root).await.unwrap(),
        vec![Segment::user_text("first"), Segment::user_text("second")]
    );
}

#[tokio::test]
async fn snapshot_on_missing_session_is_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let err = store.snapshot("no-such", "x").await.unwrap_err();
    assert!(err.to_string().contains("not found"), "got: {err}");
}
