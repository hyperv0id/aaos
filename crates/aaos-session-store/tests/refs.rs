use aaos_session_store::branch::Branch;
use aaos_session_store::view::materialize_plain;
use aaos_session_store::{
    create_session, open_current, read_head, resume, rollback, session_ids, Segment,
    SessionManifest,
};

#[tokio::test]
async fn create_session_writes_manifest_head_and_root_log() {
    let tmp = tempfile::tempdir().unwrap();
    let (sid, w) = create_session(tmp.path(), "my session").await.unwrap();

    let manifest: SessionManifest =
        serde_json::from_str(&std::fs::read_to_string(
            tmp.path().join(format!("sessions/{sid}/session.json")),
        ).unwrap())
        .unwrap();
    assert_eq!(manifest.title, "my session");

    let head = read_head(tmp.path(), &sid).unwrap();
    assert_eq!(head.log_relpath, w.log_relpath());
    assert_eq!(head.position, w.position());

    assert_eq!(session_ids(tmp.path()).unwrap(), vec![sid.clone()]);
}

#[tokio::test]
async fn snapshot_rollback_restores_view() {
    let tmp = tempfile::tempdir().unwrap();
    let (sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    let head = w.snapshot().await.unwrap();

    w.append_segment(&Segment::user_text("q2")).await.unwrap();
    w.append_segment(&Segment::user_text("q3")).await.unwrap();
    rollback(tmp.path(), &sid, &head).await.unwrap();

    let mut w = open_current(tmp.path(), &sid).await.unwrap();
    let branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    assert_eq!(materialize_plain(w.objects(), &branch).await.unwrap().len(), 1);
    assert_eq!(read_head(tmp.path(), &sid).unwrap(), head);
}

#[tokio::test]
async fn resume_truncates_to_checkpoint() {
    let tmp = tempfile::tempdir().unwrap();
    let (sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    let head = w.snapshot().await.unwrap();

    w.append_segment(&Segment::user_text("q2")).await.unwrap();
    let mut w = resume(tmp.path(), &sid).await.unwrap();
    assert_eq!(w.position(), head.position);
    w.append_segment(&Segment::user_text("q3")).await.unwrap();

    let branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    let segments = materialize_plain(w.objects(), &branch).await.unwrap();
    assert_eq!(segments.len(), 2, "q2 discarded, q3 appended after resume");
}

#[tokio::test]
async fn open_current_keeps_committed_records_after_crash_tail() {
    let tmp = tempfile::tempdir().unwrap();
    let (sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    let _head = w.snapshot().await.unwrap();
    w.append_segment(&Segment::user_text("q2")).await.unwrap();
    // No snapshot: q2 is committed but past the checkpoint. Simulate a crash
    // by appending garbage after the writer stops.
    let path = tmp.path().join(w.log_relpath());
    let content = std::fs::read(&path).unwrap();
    std::fs::write(&path, [content, vec![1, 2, 3, 4, 5, 6, 7]].concat()).unwrap();

    let mut w = open_current(tmp.path(), &sid).await.unwrap();
    let branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    assert_eq!(materialize_plain(w.objects(), &branch).await.unwrap().len(), 2);
    w.append_segment(&Segment::user_text("q3")).await.unwrap();
    let branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    assert_eq!(materialize_plain(w.objects(), &branch).await.unwrap().len(), 3);
}

#[tokio::test]
async fn rollback_beyond_log_length_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let (sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    let mut head = w.snapshot().await.unwrap();
    head.position += 10_000;
    assert!(rollback(tmp.path(), &sid, &head).await.is_err());
}
