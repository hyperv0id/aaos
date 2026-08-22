use aaos_session_store::branch::Branch;
use aaos_session_store::log::LogRecord;
use aaos_session_store::{create_session, Segment};

fn side_effects(branch: &Branch) -> Vec<&aaos_session_store::log::SideEffectRecord> {
    branch
        .records
        .iter()
        .filter_map(|(record, _)| match record {
            LogRecord::SideEffect(se) => Some(se),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn records_read_back_with_hashes_and_monotonic_seq() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_side_effect("call-1", None, Some(b"after-1".to_vec()), "/tmp/a")
        .await
        .unwrap();
    w.append_side_effect(
        "call-2",
        Some(b"before-2".to_vec()),
        Some(b"after-2".to_vec()),
        "/tmp/b",
    )
    .await
    .unwrap();

    let branch = Branch::open(tmp.path(), w.log_relpath()).await.unwrap();
    let effects = side_effects(&branch);
    assert_eq!(effects.len(), 2);
    assert_eq!(effects[0].seq, 1);
    assert!(effects[0].before_hash.is_none());
    let after = effects[0].after_hash.clone().unwrap();
    assert_eq!(w.objects().get_bytes(&after).await.unwrap(), b"after-1");

    assert_eq!(effects[1].seq, 2);
    assert_eq!(
        w.objects()
            .get_bytes(&effects[1].before_hash.clone().unwrap())
            .await
            .unwrap(),
        b"before-2"
    );
}

#[tokio::test]
async fn seq_continues_across_fork() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_side_effect("call-1", None, Some(b"a".to_vec()), "/tmp/a")
        .await
        .unwrap();
    w.append_side_effect("call-2", Some(b"b".to_vec()), None, "/tmp/a")
        .await
        .unwrap();

    let mut child = w.fork().await.unwrap();
    assert_eq!(child.side_effect_seq(), 2);
    let seq = child
        .append_side_effect("call-3", None, Some(b"c".to_vec()), "/tmp/c")
        .await
        .unwrap();
    assert_eq!(seq, 3);
}

#[tokio::test]
async fn seq_continues_across_compact() {
    let tmp = tempfile::tempdir().unwrap();
    let (_sid, mut w) = create_session(tmp.path(), "main").await.unwrap();
    w.append_segment(&Segment::user_text("q1")).await.unwrap();
    w.append_side_effect("call-1", None, Some(b"a".to_vec()), "/tmp/a")
        .await
        .unwrap();
    w.append_side_effect("call-2", None, Some(b"b".to_vec()), "/tmp/b")
        .await
        .unwrap();

    let summary = aaos_session_store::SummarySegment::new("s", vec![]);
    let mut compacted = w.compact(vec![(0..1, summary)]).await.unwrap();
    let seq = compacted
        .append_side_effect("call-3", None, Some(b"c".to_vec()), "/tmp/c")
        .await
        .unwrap();
    assert_eq!(seq, 3);
}
