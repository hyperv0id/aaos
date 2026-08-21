//! `bash` coding tool (Unix only): run a command via `bash -lc` in the
//! session working directory and return the combined stdout/stderr, truncated
//! at the shared output caps.

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use pi_agent_core::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use serde_json::{json, Value};
use tokio::io::AsyncReadExt;
use tokio::process::Command;
use tokio::sync::watch;

use crate::truncate::truncate_head;

/// Create the `bash` tool for a session.
///
/// `cwd` becomes the working directory of every spawned command. The tool is
/// Unix-only: commands run with `bash -lc`. Output is truncated at the shared
/// 2000-line / 50 KiB caps; a non-zero exit, timeout, or abort becomes an error.
pub fn create_bash_tool(cwd: impl Into<PathBuf>) -> Arc<dyn AgentTool> {
    Arc::new(BashTool { cwd: cwd.into() })
}

struct BashTool {
    cwd: PathBuf,
}

#[async_trait]
impl AgentTool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn label(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute bash commands (ls, grep, find, etc.) in the session working directory."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" },
                "timeout": { "type": "number" }
            },
            "required": ["command"]
        })
    }

    async fn execute(
        &self,
        _tool_call_id: String,
        params: Value,
        signal: Option<&watch::Receiver<bool>>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        if !self.cwd.exists() {
            return Err(format!(
                "Working directory does not exist: {}",
                self.cwd.display()
            ));
        }

        let command = params
            .get("command")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "command is required".to_string())?;

        let timeout_secs = params
            .get("timeout")
            .and_then(|v| v.as_f64())
            .filter(|secs| secs.is_finite() && *secs > 0.0);

        let mut child = Command::new("bash")
            .arg("-lc")
            .arg(command)
            .current_dir(&self.cwd)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| format!("Failed to spawn bash: {e}"))?;

        // Drain stdout and stderr concurrently so the OS pipes never fill and
        // deadlock a command that produces a lot of output. The buffers are
        // concatenated stdout-then-stderr once the process has exited.
        let mut stdout = child
            .stdout
            .take()
            .ok_or_else(|| "stdout pipe unavailable".to_string())?;
        let mut stderr = child
            .stderr
            .take()
            .ok_or_else(|| "stderr pipe unavailable".to_string())?;
        let mut stdout_buf = Vec::new();
        let mut stderr_buf = Vec::new();
        let stdout_task = tokio::spawn(async move {
            stdout.read_to_end(&mut stdout_buf).await.map(|_| stdout_buf)
        });
        let stderr_task = tokio::spawn(async move {
            stderr.read_to_end(&mut stderr_buf).await.map(|_| stderr_buf)
        });

        let status_result = wait_for_child(&mut child, signal, timeout_secs).await;
        if status_result.is_err() {
            // Timeout or abort: ensure the process is dead before draining.
            let _ = child.kill().await;
        }

        let stdout_buf = drain_read_task(stdout_task).await?;
        let stderr_buf = drain_read_task(stderr_task).await?;

        let combined = String::from_utf8_lossy(&stdout_buf).into_owned()
            + &String::from_utf8_lossy(&stderr_buf);
        let output = truncate_head(&combined).content;

        match status_result {
            Ok(status) if status.success() => Ok(AgentToolResult::text(output)),
            Ok(status) => {
                let code = status.code().unwrap_or(-1);
                let error = if output.is_empty() {
                    format!("Command exited with code {code}")
                } else {
                    format!("{output}\nCommand exited with code {code}")
                };
                Err(error)
            }
            Err(error) => Err(error),
        }
    }
}

/// Wait for `child` to exit, racing the wait against an optional timeout and
/// the abort signal. Returns `Ok(ExitStatus)` when the child exits on its own,
/// or `Err` with a timeout/abort message (the caller kills the still-running
/// process in that case).
async fn wait_for_child(
    child: &mut tokio::process::Child,
    signal: Option<&watch::Receiver<bool>>,
    timeout_secs: Option<f64>,
) -> Result<std::process::ExitStatus, String> {
    let wait = child.wait();
    tokio::pin!(wait);

    let abort = async {
        match signal {
            Some(rx) => {
                let mut rx = rx.clone();
                let _ = rx.wait_for(|flag| *flag).await;
            }
            None => std::future::pending::<()>().await,
        }
    };
    tokio::pin!(abort);

    if let Some(secs) = timeout_secs {
        let sleep = tokio::time::sleep(Duration::from_secs_f64(secs));
        tokio::pin!(sleep);
        tokio::select! {
            biased;
            _ = &mut abort => Err("Command aborted".to_string()),
            _ = &mut sleep => Err(format!("Command timed out after {secs} seconds")),
            status = &mut wait => status.map_err(|e| format!("Command wait failed: {e}")),
        }
    } else {
        tokio::select! {
            biased;
            _ = &mut abort => Err("Command aborted".to_string()),
            status = &mut wait => status.map_err(|e| format!("Command wait failed: {e}")),
        }
    }
}

/// Join a spawned pipe-read task, flattening join + io errors into a tool error.
async fn drain_read_task(
    task: tokio::task::JoinHandle<Result<Vec<u8>, std::io::Error>>,
) -> Result<Vec<u8>, String> {
    task.await
        .map_err(|e| format!("output read join failed: {e}"))?
        .map_err(|e| format!("output read failed: {e}"))
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::truncate::MAX_LINES;
    use pi_agent_core::types::ContentBlock;
    use serde_json::json;
    use std::fs;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;
    use tokio::sync::watch;

    #[tokio::test]
    async fn runs_command_in_cwd() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("x.txt"), "hi").unwrap();
        let tool = create_bash_tool(tmp.path());
        let result = tool
            .execute("1".into(), json!({"command": "cat x.txt"}), None, None)
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("text"),
        };
        assert!(text.contains("hi"), "{text}");
    }

    #[tokio::test]
    async fn nonzero_exit_is_execute_err() {
        let tool = create_bash_tool("/tmp");
        let err = tool
            .execute("1".into(), json!({"command": "exit 7"}), None, None)
            .await
            .unwrap_err();
        assert!(err.contains('7'), "{err}");
        assert!(err.contains("Command exited with code"), "{err}");
    }

    #[tokio::test]
    async fn stdout_and_stderr_are_combined() {
        let tool = create_bash_tool("/tmp");
        let result = tool
            .execute(
                "1".into(),
                json!({"command": "echo out; echo err 1>&2"}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("text"),
        };
        assert!(text.contains("out"), "{text}");
        assert!(text.contains("err"), "{text}");
    }

    #[tokio::test]
    async fn timeout_kills_long_command() {
        let tool = create_bash_tool("/tmp");
        let start = Instant::now();
        let err = tool
            .execute(
                "1".into(),
                json!({"command": "sleep 10", "timeout": 1}),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "timeout should fire near 1s, took {:?}",
            start.elapsed()
        );
        assert!(err.contains("timed out"), "{err}");
        assert!(err.contains('1'), "{err}");
    }

    #[tokio::test]
    async fn abort_kills_running_command() {
        let (tx, rx) = watch::channel(false);
        let tool = create_bash_tool("/tmp");
        let tx2 = tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(200)).await;
            let _ = tx2.send(true);
        });
        let start = Instant::now();
        let err = tool
            .execute(
                "1".into(),
                json!({"command": "sleep 30", "timeout": 300}),
                Some(&rx),
                None,
            )
            .await
            .unwrap_err();
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "abort should be fast, took {:?}",
            start.elapsed()
        );
        assert!(err.contains("abort"), "{err}");
    }

    #[tokio::test]
    async fn oversized_output_is_truncated() {
        let tool = create_bash_tool("/tmp");
        // 3000 lines exceeds the 2000-line cap.
        let result = tool
            .execute("1".into(), json!({"command": "seq 1 3000"}), None, None)
            .await
            .unwrap();
        let text = match &result.content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("text"),
        };
        assert!(
            text.lines().count() <= MAX_LINES,
            "got {} lines, cap is {MAX_LINES}",
            text.lines().count()
        );
        assert!(text.starts_with("1\n") || text.starts_with("1\r"), "head: {text}");
    }

    #[tokio::test]
    async fn missing_cwd_errors_cleanly() {
        let tool = create_bash_tool("/does/not/exist/aaos-bash-7-missing");
        let err = tool
            .execute("1".into(), json!({"command": "true"}), None, None)
            .await
            .unwrap_err();
        assert!(err.contains("Working directory does not exist"), "{err}");
    }
}
