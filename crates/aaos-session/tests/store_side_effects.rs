//! Ticket 04 — 副作用记录与会话级 seq。
//! Seam: `SessionStore::append_side_effect` / `side_effects`。
#![allow(clippy::unwrap_used, clippy::expect_used)]

mod common;

use common::store_with;

#[tokio::test]
async fn side_effects_sequential_and_readable() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();

    let first = store
        .append_side_effect(
            &root,
            "call-1",
            Some(b"before"),
            Some(b"after"),
            "/tmp/f.txt",
        )
        .await
        .unwrap();
    let second = store
        .append_side_effect(&root, "call-2", None, Some(b"after-2"), "/tmp/g.txt")
        .await
        .unwrap();

    assert!(first.seq < second.seq, "{} !< {}", first.seq, second.seq);
    assert!(first.before_hash.is_some());
    assert!(second.before_hash.is_none());

    let records = store.side_effects(&root).await.unwrap();
    assert_eq!(records.len(), 2);
    assert_eq!(records[0].tool_call_id, "call-1");
    assert_eq!(records[0].path, "/tmp/f.txt");
    assert_eq!(records[1].tool_call_id, "call-2");
    assert_eq!(records[1].before_hash, None);
    assert!(records[0].seq < records[1].seq);

    // Payloads resolve in the object store.
    let after = records[0].after_hash.as_ref().unwrap();
    assert_eq!(store.objects().get_bytes(after).await.unwrap(), b"after");
}

#[tokio::test]
async fn seq_continues_across_fork_and_compaction() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    store
        .append_segment(&root, &aaos_session::Segment::user_text("q"))
        .await
        .unwrap();

    let s0 = store
        .append_side_effect(&root, "c0", Some(b"a"), Some(b"b"), "/p/0")
        .await
        .unwrap()
        .seq;

    let child = store.fork(&root).await.unwrap();
    let s1 = store
        .append_side_effect(&child, "c1", None, Some(b"c"), "/p/1")
        .await
        .unwrap()
        .seq;

    // Compaction of the child continues the child's seq ceiling — a compact
    // session derives from its parent and never resets the lineage seq.
    // (Sibling lineages share the parent ceiling and may reuse a seq; seq is
    // monotonic per lineage, not globally unique.)
    let compacted = store
        .compact(&child, &[(0, 1)], &aaos_session::Segment::summary("s"))
        .await
        .unwrap();
    let s2 = store
        .append_side_effect(&compacted, "c2", None, Some(b"d"), "/p/2")
        .await
        .unwrap()
        .seq;

    assert!(
        s0 < s1 && s1 < s2,
        "seqs not monotonic across derivation: {s0} {s1} {s2}"
    );
    let child_records = store.side_effects(&child).await.unwrap();
    assert_eq!(child_records.len(), 1);
    assert_eq!(child_records[0].seq, s1);
}

#[tokio::test]
async fn identical_payloads_dedup_by_hash() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();

    let first = store
        .append_side_effect(&root, "c1", None, Some(b"same-bytes"), "/p/1")
        .await
        .unwrap();
    let second = store
        .append_side_effect(&root, "c2", None, Some(b"same-bytes"), "/p/2")
        .await
        .unwrap();

    assert_eq!(first.after_hash, second.after_hash);
}
