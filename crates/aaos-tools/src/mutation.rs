use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::Mutex;

/// Serializes async runs per path: mutations to the same path never overlap,
/// while different paths proceed independently.
#[derive(Default)]
pub struct FileMutationQueue {
    locks: StdMutex<HashMap<PathBuf, Arc<Mutex<()>>>>,
}

impl FileMutationQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn run_exclusive<F, T>(&self, path: &Path, fut: F) -> T
    where
        F: std::future::Future<Output = T>,
    {
        let lock = {
            let mut map = self.locks.lock().unwrap_or_else(|e| e.into_inner());
            map.entry(path.to_path_buf())
                .or_insert_with(|| Arc::new(Mutex::new(())))
                .clone()
        };
        let _guard = lock.lock().await;
        fut.await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;

    #[tokio::test]
    async fn same_path_runs_do_not_overlap() {
        let queue = FileMutationQueue::new();
        let order = Arc::new(Mutex::new(Vec::new()));
        let path = std::path::Path::new("/tmp/same");
        let a = {
            let order = order.clone();
            let queue = &queue;
            async move {
                queue
                    .run_exclusive(path, async {
                        order.lock().await.push("a-start");
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        order.lock().await.push("a-end");
                    })
                    .await
            }
        };
        let b = {
            let order = order.clone();
            let queue = &queue;
            async move {
                queue
                    .run_exclusive(path, async {
                        order.lock().await.push("b-start");
                        order.lock().await.push("b-end");
                    })
                    .await
            }
        };
        tokio::join!(a, b);
        let order = order.lock().await.clone();
        assert!(
            order == ["a-start", "a-end", "b-start", "b-end"]
                || order == ["b-start", "b-end", "a-start", "a-end"],
            "{order:?}"
        );
    }

    #[tokio::test]
    async fn different_paths_run_independently() {
        // Different paths must NOT serialize against each other: path A holds
        // its lock until signaled by path B. If B were serialized behind A
        // (same-path behavior misapplied to distinct keys), B would never run
        // and the test would hang. Passing proves independent per-path locks.
        let queue = FileMutationQueue::new();
        let path_a = std::path::Path::new("/tmp/diff-a");
        let path_b = std::path::Path::new("/tmp/diff-b");
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();

        let a = async {
            queue
                .run_exclusive(path_a, async {
                    let _ = rx.await;
                })
                .await
        };
        let b = async {
            queue.run_exclusive(path_b, async {}).await;
            let _ = tx.send(());
        };

        // With a 2s deadline: if serialization is correct this completes
        // near-instantly; if B were stuck behind A it times out and fails.
        let both = async { tokio::join!(a, b) };
        tokio::time::timeout(Duration::from_secs(2), both)
            .await
            .expect("different paths were serialized");
    }
}
