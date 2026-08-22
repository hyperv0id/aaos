//! Shared helpers for the integration tests.

use aaos_session_store::SessionStore;

pub async fn store_with(dir: &std::path::Path) -> SessionStore {
    SessionStore::open(dir).await.unwrap()
}
