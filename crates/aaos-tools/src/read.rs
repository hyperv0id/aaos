//! `read` coding tool: read text file contents with an optional line window.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use pi_agent_core::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use serde_json::{json, Value};
use tokio::sync::watch;

use crate::path::resolve_to_cwd;
use crate::truncate::truncate_head;

/// Create the `read` tool for a session. Relative `path` arguments are
/// resolved against `cwd`.
pub fn create_read_tool(cwd: impl Into<PathBuf>) -> Arc<dyn AgentTool> {
    Arc::new(ReadTool { cwd: cwd.into() })
}

struct ReadTool {
    cwd: PathBuf,
}

/// True when the abort signal is set (operation was cancelled).
fn aborted(signal: Option<&watch::Receiver<bool>>) -> bool {
    signal.is_some_and(|s| *s.borrow())
}

fn optional_u64(params: &Value, key: &str) -> Result<Option<u64>, String> {
    match params.get(key) {
        Some(Value::Number(n)) => n
            .as_u64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be a non-negative integer")),
        Some(_) => Err(format!("{key} must be a number")),
        None => Ok(None),
    }
}

#[async_trait]
impl AgentTool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn label(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read text file contents (relative to the session cwd, or absolute). \
         Output is truncated to 2000 lines / 50KB; use offset and limit to \
         page through large files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)"
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read"
                }
            },
            "required": ["path"]
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

        let offset = match optional_u64(&params, "offset")? {
            Some(0) => return Err("offset must be >= 1 (1-indexed)".to_string()),
            other => other.map(|n| n as usize),
        };
        let limit = optional_u64(&params, "limit")?.map(|n| n as usize);

        if aborted(signal) {
            return Err("Operation aborted".to_string());
        }

        let resolved = resolve_to_cwd(path, &self.cwd);
        // read_to_string fails on binary/non-UTF-8 content (io::Error
        // InvalidData); surface it as a tool error instead of mojibake.
        let text = tokio::fs::read_to_string(&resolved)
            .await
            .map_err(|e| format!("Failed to read {path}: {e}"))?;

        if aborted(signal) {
            return Err("Operation aborted".to_string());
        }

        let total_lines = text.lines().count();
        let start_1based = offset.unwrap_or(1);
        let start_0based = start_1based - 1;
        if let Some(offset) = offset {
            if start_0based >= total_lines {
                return Err(format!(
                    "Offset {offset} is beyond end of file ({total_lines} lines total)"
                ));
            }
        }

        // Caller's window first (offset + user limit), then shared truncation.
        let window = text.lines().skip(start_0based);
        let mut content = match limit {
            Some(limit) => window.take(limit).collect::<Vec<_>>().join("\n"),
            None => window.collect::<Vec<_>>().join("\n"),
        };

        let truncation = truncate_head(&content);
        content = truncation.content;
        if truncation.truncated {
            let next_offset = start_1based + content.lines().count();
            content.push_str(&format!("\n\nUse offset={next_offset} to continue"));
        }

        Ok(AgentToolResult::text(content))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_core::types::ContentBlock;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;
    use tokio::sync::watch;

    #[tokio::test]
    async fn reads_file_with_offset_and_limit() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.txt"), "l1\nl2\nl3\nl4\n").unwrap();
        let tool = create_read_tool(tmp.path());
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "a.txt", "offset": 2, "limit": 2}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("text"),
        };
        assert!(text.contains("l2"));
        assert!(text.contains("l3"));
        assert!(!text.contains("l1"));
    }

    #[tokio::test]
    async fn empty_file_without_offset_returns_empty_text() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("empty.txt"), "").unwrap();
        let tool = create_read_tool(tmp.path());
        let result = tool
            .execute("1".into(), json!({"path": "empty.txt"}), None, None)
            .await
            .expect("empty file without offset should succeed");
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("text"),
        };
        assert_eq!(text, "");
    }

    #[tokio::test]
    async fn offset_past_eof_errors_with_total_lines() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("f.txt"), "one\ntwo\nthree\n").unwrap();
        let tool = create_read_tool(tmp.path());
        let err = tool
            .execute(
                "1".into(),
                json!({"path": "f.txt", "offset": 10}),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(
            err.contains("Offset 10 is beyond end of file (3 lines total)"),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn oversized_output_is_truncated_with_continuation_hint() {
        use crate::truncate::MAX_LINES;
        let tmp = TempDir::new().unwrap();
        let big = (0..MAX_LINES + 5)
            .map(|i| format!("L{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(tmp.path().join("big.txt"), &big).unwrap();
        let tool = create_read_tool(tmp.path());
        let result = tool
            .execute("1".into(), json!({"path": "big.txt"}), None, None)
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("text"),
        };
        assert!(text.contains("Use offset=2001 to continue"), "got: {text}");
    }

    #[tokio::test]
    async fn continuation_hint_respects_offset() {
        use crate::truncate::MAX_LINES;
        let tmp = TempDir::new().unwrap();
        let big = (0..MAX_LINES + 1000)
            .map(|i| format!("L{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(tmp.path().join("big.txt"), &big).unwrap();
        let tool = create_read_tool(tmp.path());
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "big.txt", "offset": 500}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("text"),
        };
        // 500..2500 window: truncate_head keeps 2000 lines -> next = 500 + 2000.
        assert!(text.contains("Use offset=2500 to continue"), "got: {text}");
    }

    #[tokio::test]
    async fn limit_applies_before_truncation() {
        use crate::truncate::MAX_LINES;
        let tmp = TempDir::new().unwrap();
        let big = (0..MAX_LINES + 100)
            .map(|i| format!("L{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(tmp.path().join("big.txt"), &big).unwrap();
        let tool = create_read_tool(tmp.path());
        // User limit under MAX_LINES -> no truncation, no hint.
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "big.txt", "limit": 10}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("text"),
        };
        assert_eq!(text.lines().count(), 10);
        assert!(!text.contains("to continue"));
    }

    #[tokio::test]
    async fn non_utf8_file_errors() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("bin.dat"), [0xFFu8, 0xFE, 0x00]).unwrap();
        let tool = create_read_tool(tmp.path());
        let err = tool
            .execute("1".into(), json!({"path": "bin.dat"}), None, None)
            .await
            .unwrap_err();
        assert!(
            err.contains("Failed to read bin.dat"),
            "binary files must error, got: {err}"
        );
    }

    #[tokio::test]
    async fn absolute_path_is_read() {
        let tmp = TempDir::new().unwrap();
        let abs = tmp.path().join("abs.txt");
        fs::write(&abs, "hello").unwrap();
        let tool = create_read_tool("/some/other/cwd");
        let result = tool
            .execute(
                "1".into(),
                json!({"path": abs.to_str().unwrap()}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("text"),
        };
        assert_eq!(text, "hello");
    }

    #[tokio::test]
    async fn aborts_when_signal_set() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.txt"), "l1\nl2\n").unwrap();
        let tool = create_read_tool(tmp.path());
        let (tx, rx) = watch::channel(false);
        tx.send(true).unwrap();
        let err = tool
            .execute("1".into(), json!({"path": "a.txt"}), Some(&rx), None)
            .await
            .unwrap_err();
        assert_eq!(err, "Operation aborted");
    }
}
