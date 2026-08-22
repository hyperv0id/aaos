use aaos_session_store::branch::Branch;
use aaos_session_store::view::materialize_plain;
use aaos_session_store::{create_session, open_current, ObjectStore, Segment};

#[tokio::test]
async fn garbage_tail_truncated_and_append_continues_cleanly() {
    let tmp = tempfile::tempdir().unwrap();
    let (sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    w.flush().await.unwrap();

    // Crash: append garbage bytes past the last good record.
    let path = tmp.path().join(w.log_relpath());
    let content = std::fs::read(&path).unwrap();
    std::fs::write(&path, [content, vec![0xde, 0xad, 0xbe]].concat()).unwrap();

    let branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    let good_len = branch.log_len;
    assert_eq!(std::fs::metadata(&path).unwrap().len(), good_len);

    let mut w = open_current(tmp.path(), &sid).await.unwrap();
    w.append_segment(&Segment::user_text("q2")).await.unwrap();
    let branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    assert_eq!(materialize_plain(w.objects(), &branch).await.unwrap().len(), 2);
}

#[tokio::test]
async fn same_content_concurrent_put_yields_one_object() {
    let tmp = tempfile::tempdir().unwrap();
    let store = ObjectStore::new(tmp.path());
    let a = store.put_bytes(b"shared payload").await.unwrap();
    let b = {
        // A second store instance over the same root, as a separate writer
        // would have.
        let other = ObjectStore::new(tmp.path());
        other.put_bytes(b"shared payload").await.unwrap()
    };
    assert_eq!(a, b);

    let shard_dir = tmp.path().join("objects").join(&a[..2]);
    let entries: Vec<_> = std::fs::read_dir(&shard_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(&a))
        .collect();
    assert_eq!(entries.len(), 1, "exactly one object file for the hash");
}
