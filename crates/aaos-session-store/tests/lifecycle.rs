//! Ticket 06 — full lifecycle over the SQLite structural layer:
//! root append → fork → compact → bookmark → append → derive-back → append,
//! with stage-wise length and segment-kind assertions.

use aaos_session_store::{Segment, SessionStore};

fn kinds(view: &[Segment]) -> Vec<&'static str> {
    view.iter().map(|s| s.kind()).collect()
}

#[tokio::test]
async fn full_lifecycle_root_fork_compact_bookmark_derive_back() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();

    // 1. Root: three turns.
    let root = store.create_root().await.unwrap();
    let root_hashes = [
        store.append_segment(&root, &Segment::user_text("q1")).await.unwrap(),
        store.append_segment(&root, &Segment::assistant_text("a1")).await.unwrap(),
        store.append_segment(&root, &Segment::user_text("q2")).await.unwrap(),
    ];
    let v = store.materialize_plain(&root).await.unwrap();
    assert_eq!(v.len(), 3);
    assert_eq!(kinds(&v), vec!["user", "assistant", "user"]);

    // A side effect on the root line.
    let se0 = store
        .append_side_effect(&root, "call-0", Some(b"old"), Some(b"new"), "/p/first")
        .await
        .unwrap();

    // 2. Fork: inherits 3, appends 2 — zero copies of the parent's segments.
    let child = store.fork(&root).await.unwrap();
    store
        .append_segment(&child, &Segment::assistant_text("a2"))
        .await
        .unwrap();
    store
        .append_segment(&child, &Segment::user_text("q3"))
        .await
        .unwrap();
    let v = store.materialize_plain(&child).await.unwrap();
    assert_eq!(v.len(), 5);
    assert_eq!(
        kinds(&v),
        vec!["user", "assistant", "user", "assistant", "user"]
    );
    assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 3);

    let se1 = store
        .append_side_effect(&child, "call-1", None, Some(b"child-new"), "/p/child")
        .await
        .unwrap();
    assert!(se0.seq < se1.seq);

    // 3. Compact the parent's prefix [0, 2): summary replaces q1 + a1.
    let summary = Segment::summary("opening turns", vec![root_hashes[0].clone(), root_hashes[1].clone()]);
    let compacted = store.compact(&root, &[(0, 2)], &summary).await.unwrap();
    let v = store.materialize_plain(&compacted).await.unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(kinds(&v), vec!["summary", "user"]);
    assert_eq!(
        store.fetch_originals(&compacted).await.unwrap(),
        vec![(0, 2, vec![Segment::user_text("q1"), Segment::assistant_text("a1")])]
    );

    // 4. Bookmark the compacted line, then keep appending past it.
    let snap = store.snapshot(&compacted, "stable").await.unwrap();
    assert_eq!(snap.position, 2);
    store
        .append_segment(&compacted, &Segment::assistant_text("a3"))
        .await
        .unwrap();
    assert_eq!(store.materialize_plain(&compacted).await.unwrap().len(), 3);

    // 5. Derive back from the bookmark: post-bookmark work invisible,
    //    compaction still applied (that is what the bookmark pinned).
    let back = store
        .fork_at(&snap.session_id, snap.position)
        .await
        .unwrap();
    let v = store.materialize_plain(&back).await.unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(kinds(&v), vec!["summary", "user"]);

    // Undo compaction entirely: derive from the uncompacted parent.
    let undone = store.fork_at(&root, 3).await.unwrap();
    let v = store.materialize_plain(&undone).await.unwrap();
    assert_eq!(v.len(), 3);
    assert!(v.iter().all(|s| s.kind() != "summary"));

    // 6. Continue on the rolled-back line.
    store
        .append_segment(&back, &Segment::user_text("q4"))
        .await
        .unwrap();
    let v = store.materialize_plain(&back).await.unwrap();
    assert_eq!(v.len(), 3);
    assert_eq!(kinds(&v), vec!["summary", "user", "user"]);

    // Everything survives a reopen: committed structure is durable.
    drop(store);
    let store = SessionStore::open(dir.path()).await.unwrap();
    assert_eq!(store.materialize_plain(&child).await.unwrap().len(), 5);
    assert_eq!(store.materialize_plain(&back).await.unwrap().len(), 3);
    assert_eq!(
        store.latest_session().await.unwrap(),
        Some(undone.clone()),
        "undone was the last session created"
    );
}
