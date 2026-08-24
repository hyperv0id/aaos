//! Side-effect capture: records before/after file bytes for write/edit tools
//! and the command for bash, persisted via `append_side_effect`.
//!
//! Two cooperating pieces (spec §5.2):
//!
//! - **`before_tool_call` hook** (`install_before_hook`): runs before write/edit
//!   execution, reads the target file's current bytes, and stashes them in a
//!   [`CaptureTable`] keyed by `tool_call_id`. Bash and unknown tools record
//!   only their label (path/command/tool name, no before bytes).
//! - **`ToolExecutionEnd` listener** (`handle_tool_execution_end`): after
//!   execution, reads the target file's new bytes (write/edit only), pulls the
//!   before bytes + label from the table, and calls `append_side_effect`. The
//!   table entry is cleared after.
//!
//! `AgentToolResult` has no before/after fields (types.rs:304), so the hook is
//! the only point that can capture pre-mutation state. Parallel tool execution
//! means the after bytes are a best-effort snapshot, not a transaction.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::Mutex;

use crate::db::SessionStore;

/// A captured before-state for a tool call.
#[derive(Debug)]
struct Capture {
    /// Raw path/command or tool-name label, stored in the `side_effects.path`
    /// column.
    label: String,
    /// Resolved filesystem path for file tools (write/edit); `None` for bash
    /// and unknown tools.
    file_path: Option<PathBuf>,
    /// Pre-mutation file bytes for write/edit; `None` for bash/unknown or when
    /// the file did not exist yet (new file).
    before: Option<Vec<u8>>,
}

/// Thread-safe table mapping `tool_call_id` → captured before-state.
///
/// Entries are inserted by the `before_tool_call` hook and removed by the
/// `ToolExecutionEnd` listener, so the table never grows across turns.
#[derive(Debug, Default, Clone)]
pub struct CaptureTable {
    inner: Arc<Mutex<HashMap<String, Capture>>>,
}

impl CaptureTable {
    async fn set(&self, tool_call_id: String, capture: Capture) {
        self.inner.lock().await.insert(tool_call_id, capture);
    }

    async fn take(&self, tool_call_id: &str) -> Option<Capture> {
        self.inner.lock().await.remove(tool_call_id)
    }
}

/// Extract the side-effect label and resolved filesystem path for a tool call.
///
/// write/edit → (args.path, Some(resolved_path)); bash → (args.command, None);
/// other → (tool_name, None). When write/edit args.path is missing, the
/// label falls back to tool_name (more diagnosable than empty string).
fn tool_path(tool_name: &str, args: &Value, cwd: &Path) -> (String, Option<PathBuf>) {
    match tool_name {
        "write" | "edit" => {
            let path = args.get("path").and_then(Value::as_str);
            let label = path.unwrap_or(tool_name).to_string();
            let resolved = path.map(|p| {
                let requested = Path::new(p);
                if requested.is_absolute() {
                    requested.to_path_buf()
                } else {
                    cwd.join(requested)
                }
            });
            (label, resolved)
        }
        "bash" => (
            args.get("command")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            None,
        ),
        _ => (tool_name.to_string(), None),
    }
}

/// Read a file's bytes. Returns `None` if the file does not exist (a new
/// file being created has no before-state). Other read errors are logged
/// and also return `None` (best-effort capture).
async fn read_file_bytes(path: &Path) -> Option<Vec<u8>> {
    match tokio::fs::read(path).await {
        Ok(bytes) => Some(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => {
            #[allow(clippy::print_stderr)]
            {
                eprintln!("aaos-session: read {path:?} failed: {e}");
            }
            None
        }
    }
}

/// Read file bytes for a captured tool call. Returns None when there is no
/// resolved file path (bash, unknown tools) or when the file does not exist.
async fn read_capture_bytes(file_path: &Option<PathBuf>) -> Option<Vec<u8>> {
    match file_path {
        Some(p) => read_file_bytes(p).await,
        None => None,
    }
}

/// Install the `before_tool_call` hook that captures pre-mutation state, and
/// return the [`CaptureTable`] to be consulted by the `ToolExecutionEnd`
/// listener.
///
/// `cwd` resolves relative `path` arguments the same way the tools do.
pub fn install_before_hook(agent: &mut pi_agent_core::agent::Agent, cwd: PathBuf) -> CaptureTable {
    let table = CaptureTable::default();
    let hook_table = table.clone();
    let cwd = Arc::new(cwd);

    agent.before_tool_call = Some(Arc::new(move |ctx, _signal| {
        let table = hook_table.clone();
        let cwd = cwd.clone();
        Box::pin(async move {
            let (label, file_path) = tool_path(&ctx.tool_call.name, &ctx.args, &cwd);
            let before = read_capture_bytes(&file_path).await;
            table
                .set(
                    ctx.tool_call.id.clone(),
                    Capture {
                        label,
                        file_path,
                        before,
                    },
                )
                .await;
            Ok(pi_agent_core::types::BeforeToolCallResult::default())
        })
    }));

    table
}

/// Handle a `ToolExecutionEnd` event: look up the captured before-state, read
/// the after bytes (write/edit only), and persist the side effect.
///
/// Called from the `AgentSession` listener. Errors are logged, not propagated
/// (listeners cannot fail the turn).
///
/// Best-effort: if an abort fires after the `before_tool_call` hook ran but
/// before execution, the capture entry still exists and a row is persisted
/// asserting a mutation that never happened (before == after). This is an
/// accepted edge case of the best-effort capture model (spec §5.2 item 10).
pub async fn handle_tool_execution_end(
    store: &SessionStore,
    session_id: &str,
    tool_call_id: &str,
    table: &CaptureTable,
) {
    let Some(capture) = table.take(tool_call_id).await else {
        return;
    };
    let after = read_capture_bytes(&capture.file_path).await;
    if let Err(err) = store
        .append_side_effect(
            session_id,
            tool_call_id,
            capture.before.as_deref(),
            after.as_deref(),
            &capture.label,
        )
        .await
    {
        #[allow(clippy::print_stderr)]
        {
            eprintln!("aaos-session: side effect for {tool_call_id} failed: {err}");
        }
    }
}
