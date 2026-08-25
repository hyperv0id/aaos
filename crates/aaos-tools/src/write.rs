//! `write` coding tool: create or fully overwrite a file.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use pi_agent_core::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::aborted;
use crate::mutation::FileMutationQueue;

/// Create the `write` tool for a session.
///
/// `cwd` is used to resolve relative `path` arguments; writes are serialized
/// through the shared per-path `queue` so concurrent mutations to the same
/// file never interleave.
pub fn create_write_tool(
    cwd: impl Into<PathBuf>,
    queue: Arc<FileMutationQueue>,
) -> Arc<dyn AgentTool> {
    Arc::new(WriteTool {
        cwd: cwd.into(),
        queue,
    })
}

struct WriteTool {
    cwd: PathBuf,
    queue: Arc<FileMutationQueue>,
}

#[async_trait]
impl AgentTool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Create or fully overwrite a file, creating missing parent directories. \
         Use for new files or full rewrites; prefer edit for targeted changes."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path relative to the session cwd, or absolute."
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write."
                }
            },
            "required": ["path", "content"],
            "additionalProperties": false
        })
    }

    async fn execute(
        &self,
        _tool_call_id: String,
        params: Value,
        signal: Option<&watch::Receiver<bool>>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        let path = params["path"]
            .as_str()
            .ok_or_else(|| "missing or non-string `path`".to_string())?;
        let content = params["content"]
            .as_str()
            .ok_or_else(|| "missing or non-string `content`".to_string())?
            .to_string();

        let resolved = self.cwd.join(path);
        let path_for_write = resolved.clone();

        if aborted(signal) {
            return Err("Operation aborted".to_string());
        }

        let byte_count = self
            .queue
            .run_exclusive(&resolved, async move {
                if let Some(parent) = path_for_write.parent()
                    && !parent.as_os_str().is_empty()
                {
                    tokio::fs::create_dir_all(parent).await?;
                }
                tokio::fs::write(&path_for_write, &content).await?;
                Ok::<usize, std::io::Error>(content.len())
            })
            .await
            .map_err(|e| e.to_string())?;

        // Late cancel after a successful write still reports success.
        Ok(AgentToolResult::text(format!(
            "Successfully wrote {byte_count} bytes to {path}"
        )))
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use pi_agent_core::types::ContentBlock;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;
    use tokio::sync::watch;

    #[tokio::test]
    async fn creates_nested_file() {
        let tmp = TempDir::new().unwrap();
        let queue = Arc::new(FileMutationQueue::new());
        let tool = create_write_tool(tmp.path(), queue);
        tool.execute(
            "1".into(),
            json!({"path": "n/a.txt", "content": "hi"}),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            fs::read_to_string(tmp.path().join("n/a.txt")).unwrap(),
            "hi"
        );
    }

    #[tokio::test]
    async fn overwrites_existing_file() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        fs::write(dir.join("f.txt"), "old").unwrap();
        let tool = create_write_tool(dir, Arc::new(FileMutationQueue::new()));
        tool.execute(
            "1".into(),
            json!({"path": "f.txt", "content": "new contents"}),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(
            fs::read_to_string(dir.join("f.txt")).unwrap(),
            "new contents"
        );
    }

    #[tokio::test]
    async fn absolute_path_is_resolved() {
        let tmp = TempDir::new().unwrap();
        let abs = tmp.path().join("abs.txt");
        let tool = create_write_tool("/some/other/cwd", Arc::new(FileMutationQueue::new()));
        tool.execute(
            "1".into(),
            json!({"path": abs.to_str().unwrap(), "content": "x"}),
            None,
            None,
        )
        .await
        .unwrap();
        assert_eq!(fs::read_to_string(&abs).unwrap(), "x");
    }

    #[tokio::test]
    async fn empty_content_writes_zero_bytes() {
        let tmp = TempDir::new().unwrap();
        let tool = create_write_tool(tmp.path(), Arc::new(FileMutationQueue::new()));
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "empty.txt", "content": ""}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "Successfully wrote 0 bytes to empty.txt");
        assert_eq!(
            fs::read_to_string(tmp.path().join("empty.txt")).unwrap(),
            ""
        );
    }

    #[tokio::test]
    async fn multibyte_utf8_byte_count() {
        let tmp = TempDir::new().unwrap();
        let tool = create_write_tool(tmp.path(), Arc::new(FileMutationQueue::new()));
        let content = "héllo"; // 6 bytes: h=1, é=2, llo=3
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "u.txt", "content": content}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("expected text"),
        };
        assert_eq!(text, "Successfully wrote 6 bytes to u.txt");
        assert_eq!(
            fs::read_to_string(tmp.path().join("u.txt")).unwrap(),
            content
        );
    }

    #[tokio::test]
    async fn aborts_before_write_when_signal_set() {
        let tmp = TempDir::new().unwrap();
        let dest = tmp.path().join("nope.txt");
        let tool = create_write_tool(tmp.path(), Arc::new(FileMutationQueue::new()));
        let (tx, rx) = watch::channel(false);
        tx.send(true).unwrap();
        let err = tool
            .execute(
                "1".into(),
                json!({"path": "nope.txt", "content": "should-not-write"}),
                Some(&rx),
                None,
            )
            .await
            .unwrap_err();
        assert_eq!(err, "Operation aborted");
        assert!(!dest.exists(), "pre-write abort must not create the file");
    }
}
