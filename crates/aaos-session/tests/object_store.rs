#![allow(clippy::unwrap_used, clippy::expect_used)]
use aaos_session::{ObjectStore, Segment, StoreError};

#[tokio::test]
async fn put_get_roundtrip_typed_and_bytes() {
    let tmp = tempfile::tempdir().unwrap();
    let store = ObjectStore::new(tmp.path());
    let seg = Segment::user_text("hello");
    let hash = store.put(&seg).await.unwrap();
    assert_eq!(store.get(&hash).await.unwrap(), seg);

    let raw = store.put_bytes(b"raw bytes").await.unwrap();
    assert_eq!(store.get_bytes(&raw).await.unwrap(), b"raw bytes");
}

#[tokio::test]
async fn put_is_idempotent() {
    let tmp = tempfile::tempdir().unwrap();
    let store = ObjectStore::new(tmp.path());
    let seg = Segment::assistant_text("same");
    let first = store.put(&seg).await.unwrap();
    let second = store.put(&seg).await.unwrap();
    assert_eq!(first, second);
    assert!(store.contains(&first).await.unwrap());
}

#[tokio::test]
async fn missing_object_is_not_found() {
    let tmp = tempfile::tempdir().unwrap();
    let store = ObjectStore::new(tmp.path());
    let hash = "0".repeat(64);
    match store.get_bytes(&hash).await {
        Err(StoreError::NotFound(h)) => assert_eq!(h, hash),
        other => panic!("expected NotFound, got {other:?}"),
    }
}

#[tokio::test]
async fn shard_path_shape() {
    let tmp = tempfile::tempdir().unwrap();
    let store = ObjectStore::new(tmp.path());
    let hash = store.put_bytes(b"x").await.unwrap();
    let path = tmp.path().join("objects").join(&hash[..2]).join(&hash);
    assert!(path.exists(), "expected sharded object at {path:?}");
}

#[tokio::test]
async fn invalid_hash_rejected() {
    let tmp = tempfile::tempdir().unwrap();
    let store = ObjectStore::new(tmp.path());
    assert!(matches!(
        store.get_bytes("zz").await,
        Err(StoreError::InvalidHash(_))
    ));
    assert!(matches!(
        store.get_bytes(&"a".repeat(63)).await,
        Err(StoreError::InvalidHash(_))
    ));
}
