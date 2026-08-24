//! Ticket 01 — 根会话追加与视图：SQLite 结构层的最小纵切。
//! Seam: `SessionStore` 公共接口（open / create_root / append_segment /
//! materialize / materialize_plain）。
#![allow(clippy::unwrap_used, clippy::expect_used)]

use aaos_session::{Segment, SessionStore};

#[tokio::test]
async fn root_append_materialize_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();

    let session = store.create_root().await.unwrap();
    let segs = vec![
        Segment::user_text("hello"),
        Segment::assistant_text("hi there"),
        Segment::tool_result_text("call-1", "42"),
    ];
    for seg in &segs {
        store.append_segment(&session, seg).await.unwrap();
    }

    let view = store.materialize(&session).await.unwrap();
    assert_eq!(view.len(), 3);
    for ((seg, hash), want) in view.iter().zip(&segs) {
        assert_eq!(seg, want);
        assert_eq!(hash.len(), 64);
        assert!(
            hash.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        );
    }
    assert_eq!(store.materialize_plain(&session).await.unwrap(), segs);
}

#[tokio::test]
async fn summary_segment_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();
    let session = store.create_root().await.unwrap();

    let summary = Segment::summary("compacted", vec!["a".repeat(64), "b".repeat(64)]);
    store.append_segment(&session, &summary).await.unwrap();

    let view = store.materialize_plain(&session).await.unwrap();
    assert_eq!(view, vec![summary]);
    assert_eq!(view[0].kind(), "summary");
}

#[tokio::test]
async fn reopen_persists_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let session = {
        let store = SessionStore::open(dir.path()).await.unwrap();
        let session = store.create_root().await.unwrap();
        store
            .append_segment(&session, &Segment::user_text("persisted"))
            .await
            .unwrap();
        session
        // store dropped here: dedicated DB thread shuts down
    };

    let store = SessionStore::open(dir.path()).await.unwrap();
    let view = store.materialize_plain(&session).await.unwrap();
    assert_eq!(view, vec![Segment::user_text("persisted")]);
}

#[tokio::test]
async fn append_missing_session_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();

    let err = store
        .append_segment("no-such-session", &Segment::user_text("x"))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not found"), "got: {err}");
}

#[tokio::test]
async fn concurrent_handle_sees_appends() {
    let dir = tempfile::tempdir().unwrap();
    let writer = SessionStore::open(dir.path()).await.unwrap();
    let session = writer.create_root().await.unwrap();

    // WAL: a second handle (same-process stand-in for the CLI reader process)
    // materializes while the writer is still open.
    let reader = SessionStore::open(dir.path()).await.unwrap();
    assert_eq!(reader.materialize_plain(&session).await.unwrap(), vec![]);

    writer
        .append_segment(&session, &Segment::user_text("seen"))
        .await
        .unwrap();
    let view = reader.materialize_plain(&session).await.unwrap();
    assert_eq!(view, vec![Segment::user_text("seen")]);
}

// --- Ticket 02: fork — derivation and chain view ---

mod common;

use common::store_with;

#[tokio::test]
async fn fork_inherits_prefix_extends_tail() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    let segs = vec![
        Segment::user_text("q1"),
        Segment::assistant_text("a1"),
        Segment::user_text("q2"),
    ];
    for seg in &segs {
        store.append_segment(&root, seg).await.unwrap();
    }

    let child = store.fork(&root).await.unwrap();
    assert_eq!(store.materialize_plain(&child).await.unwrap(), segs);

    let own = vec![Segment::assistant_text("a2"), Segment::user_text("q3")];
    for seg in &own {
        store.append_segment(&child, seg).await.unwrap();
    }

    let mut want = segs.clone();
    want.extend(own);
    assert_eq!(store.materialize_plain(&child).await.unwrap(), want);
    // Parent is immutable under the child's appends.
    assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 3);
}

#[tokio::test]
async fn fork_at_position_inherits_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    let segs = vec![
        Segment::user_text("1"),
        Segment::user_text("2"),
        Segment::user_text("3"),
        Segment::user_text("4"),
    ];
    for seg in &segs {
        store.append_segment(&root, seg).await.unwrap();
    }

    let child = store.fork_at(&root, 2).await.unwrap();
    assert_eq!(
        store.materialize_plain(&child).await.unwrap(),
        vec![Segment::user_text("1"), Segment::user_text("2")]
    );
    store
        .append_segment(&child, &Segment::user_text("child-only"))
        .await
        .unwrap();
    assert_eq!(store.materialize_plain(&child).await.unwrap().len(), 3);
    assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 4);
}

#[tokio::test]
async fn fork_beyond_parent_view_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    store
        .append_segment(&root, &Segment::user_text("only"))
        .await
        .unwrap();

    let err = store.fork_at(&root, 5).await.unwrap_err();
    assert!(
        err.to_string().contains("position"),
        "expected position error, got: {err}"
    );
}

#[tokio::test]
async fn grandchild_materializes_chain() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    store
        .append_segment(&root, &Segment::user_text("r1"))
        .await
        .unwrap();
    store
        .append_segment(&root, &Segment::user_text("r2"))
        .await
        .unwrap();

    let child = store.fork(&root).await.unwrap();
    store
        .append_segment(&child, &Segment::user_text("c1"))
        .await
        .unwrap();

    let grandchild = store.fork(&child).await.unwrap();
    store
        .append_segment(&grandchild, &Segment::user_text("g1"))
        .await
        .unwrap();

    let view = store.materialize_plain(&grandchild).await.unwrap();
    assert_eq!(
        view,
        vec![
            Segment::user_text("r1"),
            Segment::user_text("r2"),
            Segment::user_text("c1"),
            Segment::user_text("g1"),
        ]
    );
}

#[tokio::test]
async fn latest_session_is_most_recent() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    store
        .append_segment(&root, &Segment::user_text("q"))
        .await
        .unwrap();
    assert_eq!(store.latest_session().await.unwrap(), Some(root.clone()));

    let child = store.fork(&root).await.unwrap();
    assert_eq!(store.latest_session().await.unwrap(), Some(child));
}
