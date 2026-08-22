//! Ticket 01 — 根会话追加与视图：SQLite 结构层的最小纵切。
//! Seam: `SessionStore` 公共接口（open / create_root / append_segment /
//! materialize / materialize_plain）。

use aaos_session_store::{Segment, SessionStore};

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
        assert!(hash.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
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
async fn reopen_persists_sessions_and_entries() {
    let dir = tempfile::tempdir().unwrap();
    let session = {
        let store = SessionStore::open(dir.path()).await.unwrap();
        let session = store.create_root().await.unwrap();
        store.append_segment(&session, &Segment::user_text("persisted")).await.unwrap();
        session
        // store dropped here: dedicated DB thread shuts down
    };

    let store = SessionStore::open(dir.path()).await.unwrap();
    let view = store.materialize_plain(&session).await.unwrap();
    assert_eq!(view, vec![Segment::user_text("persisted")]);
}

#[tokio::test]
async fn append_to_missing_session_is_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();

    let err = store
        .append_segment("no-such-session", &Segment::user_text("x"))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not found"), "got: {err}");
}

#[tokio::test]
async fn second_handle_sees_appends_while_first_is_open() {
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
