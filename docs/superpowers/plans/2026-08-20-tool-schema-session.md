# Tool Schema, Coding Tools, and AgentSession — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unblock tool-calling by hanging JSON Schema on `AgentTool`, validating with that same schema, shipping the four default coding tools, composing them in a thin `AgentSession`, and pointing the CLI at the session (the CLI may stay a print-mode test shell).

**Architecture:** Kernel owns the schema contract (`parameters()` + `validate_tool_arguments`). Provider copies `parameters()` into the OpenAI tools payload. Product crate `aaos-tools` owns read/write/edit/bash adapters and system-prompt assembly. Product crate `aaos-session` owns `AgentSession` (cwd + tools + prompt + `Agent`). `aaos-cli` only resolves catalog/auth and subscribes to session events. No TUI, no Presenter trait, no compaction, no extensions.

**Tech Stack:** Rust 2021, tokio, async-trait, serde_json, jsonschema 0.28, thiserror (library crates), clap (CLI only).

**Spec:** Session-agreed design 2026-08-20. Binding constraints are the Global Constraints below. Upstream references (behavior, not code to port wholesale): `packages/ai` `Tool.parameters` + `validateToolArguments`; `packages/coding-agent/src/core/tools/{read,write,edit,bash}.ts` schemas/snippets; `packages/coding-agent/src/core/agent-session.ts` role (shared by all modes); `packages/coding-agent/src/core/system-prompt.ts` tool list + guidelines.

## Global Constraints

- Kernel stays product-agnostic: no filesystem, bash, cwd, or prompt text inside `pi-agent-core`.
- `AgentTool::parameters()` returns a JSON Schema `serde_json::Value`. Default is `{"type":"object"}` (any object). Coding tools override with required fields.
- `AgentTool::validate` default calls `schema::validate_tool_arguments`. Tools may still override `validate` (existing `ValidationTool` tests must stay green).
- No TypeBox. No argument coercion (Pi AJV coercion is out of scope). Invalid args become error tool results; `execute` is not called.
- Provider must emit `function.parameters` as `tool.parameters()` with no empty stub object.
- Default tools are exactly `read`, `bash`, `edit`, `write` (Pi order). Image reading, TUI renderers, HTML export, skills, session persistence, and compaction are out of scope.
- `AgentSession` interface is only: `new`, `subscribe`, `prompt`, `abort`, `handle`, `agent` / `agent_mut`. Catalog and OpenAI stay injected (`StreamFn` + `Model`).
- Presentations subscribe to `AgentEvent`. Do not add `trait Frontend` / `trait Renderer`.
- CLI is a print-mode test shell: catalog → session → stdout. Keep existing CLI tests green.
- Subagents apply `~/.agents/skills/rust-best-practices/SKILL.md`: no `unwrap()` in production paths; `thiserror` in library crates; `&str`/`&Path` in parameters.
- Bounded simplifications (do not expand unless a test in this plan requires it): text-only `read`; no BOM/line-ending restoration in `edit`; bash is Unix `bash -lc` only (`#[cfg(unix)]` tests); bash output truncated in-memory (50 KiB / 2000 lines), no temp-file spill.
- `cargo test -p pi-agent-core`, `-p aaos-openai`, `-p aaos-tools`, `-p aaos-session`, `-p aaos-cli` must pass after the tasks that touch them. Full `cargo test` at the end of Task 10.
- No placeholders. Do not copy Pi TUI/extension code.

## File Structure

| File | Responsibility |
|------|----------------|
| `crates/pi-agent-core/src/schema.rs` | `validate_tool_arguments(schema, args) -> Result<Value, String>` |
| `crates/pi-agent-core/src/types.rs` | `AgentTool::parameters` + default `validate` |
| `crates/pi-agent-core/src/lib.rs` | Export `schema` |
| `crates/pi-agent-core/Cargo.toml` | Add `jsonschema = "0.28"` |
| `crates/aaos-openai/src/lib.rs` | `tools_payload` uses `parameters()` |
| `crates/aaos-tools/Cargo.toml` | New crate |
| `crates/aaos-tools/src/lib.rs` | Re-exports + `create_coding_tools` |
| `crates/aaos-tools/src/path.rs` | `resolve_to_cwd` |
| `crates/aaos-tools/src/truncate.rs` | Head truncate (2000 lines / 50 KiB) |
| `crates/aaos-tools/src/mutation.rs` | Per-path async mutex for write/edit |
| `crates/aaos-tools/src/read.rs` | `read` tool |
| `crates/aaos-tools/src/write.rs` | `write` tool |
| `crates/aaos-tools/src/edit.rs` | `edit` tool + `prepare_arguments` |
| `crates/aaos-tools/src/bash.rs` | `bash` tool |
| `crates/aaos-tools/src/prompt.rs` | `build_system_prompt` |
| `crates/aaos-session/Cargo.toml` | New crate |
| `crates/aaos-session/src/lib.rs` | `SessionOptions` + `AgentSession` |
| `crates/aaos-cli/src/main.rs` | Compose catalog + provider + session |
| `Cargo.toml` | Workspace members |

## Task Dependency Graph

| Task | Depends On | Write Paths |
|------|------------|-------------|
| 1. Kernel JSON Schema contract | — | `pi-agent-core` schema/types/lib/Cargo.toml, tool_engine tests |
| 2. Provider emits schema | Task 1 | `aaos-openai/src/lib.rs` |
| 3. Tools crate infra | — | `aaos-tools` path/truncate/mutation |
| 4. `read` tool | Tasks 1, 3 | `aaos-tools/src/read.rs` |
| 5. `write` tool | Tasks 1, 3 | `aaos-tools/src/write.rs` |
| 6. `edit` tool | Tasks 1, 3 | `aaos-tools/src/edit.rs` |
| 7. `bash` tool | Tasks 1, 3 | `aaos-tools/src/bash.rs` |
| 8. Prompt + `create_coding_tools` | Tasks 4–7 | `aaos-tools` prompt.rs, lib.rs |
| 9. `AgentSession` | Tasks 1, 8 | `aaos-session` |
| 10. CLI print shell | Tasks 2, 9 | `aaos-cli` |

---

### Task 1: Kernel JSON Schema contract

**Files:**
- Create: `crates/pi-agent-core/src/schema.rs`
- Modify: `crates/pi-agent-core/src/lib.rs`
- Modify: `crates/pi-agent-core/src/types.rs` (`AgentTool`)
- Modify: `crates/pi-agent-core/Cargo.toml`
- Modify: `crates/pi-agent-core/src/tool_engine.rs` (add schema-driven test; keep `ValidationTool` override test)

**Interfaces:**
- Consumes: `serde_json::Value` tool arguments.
- Produces: `pub fn validate_tool_arguments(schema: &Value, args: &Value) -> Result<Value, String>`; `AgentTool::parameters(&self) -> Value` (default `{"type":"object"}`); default `validate` uses `validate_tool_arguments`.

- [ ] **Step 1: Add `jsonschema` and write failing tests**

Add to `crates/pi-agent-core/Cargo.toml` dependencies:

```toml
jsonschema = "0.28"
```

Add `pub mod schema;` to `lib.rs` (module can be empty except tests will fail to compile until Step 3 — write tests in `schema.rs` under `#[cfg(test)]` after the function exists; first write the tool_engine integration test that requires `parameters()`).

In `tool_engine.rs` tests, add a tool that does **not** override `validate`, only `parameters`:

```rust
struct SchemaRequiredTool;

#[async_trait]
impl AgentTool for SchemaRequiredTool {
    fn name(&self) -> &str {
        "schema_tool"
    }
    fn label(&self) -> &str {
        "schema_tool"
    }
    fn description(&self) -> &str {
        "requires value"
    }
    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "value": { "type": "string" } },
            "required": ["value"]
        })
    }
    async fn execute(
        &self,
        _tool_call_id: String,
        _params: Value,
        _signal: Option<&watch::Receiver<bool>>,
        _on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        panic!("execute should not run when schema fails");
    }
}

#[tokio::test]
async fn schema_required_field_yields_error_result_without_executing() {
    let (emit, events) = recording_emit();
    let context = AgentContext {
        tools: vec![Arc::new(SchemaRequiredTool)],
        ..empty_context()
    };
    let assistant = assistant_with_tool_calls(vec![ToolCall {
        id: "c1".into(),
        name: "schema_tool".into(),
        arguments: json!({}),
    }]);
    let batch = execute_tool_calls(
        &assistant,
        &context,
        &AgentLoopConfig::default(),
        None,
        &emit,
    )
    .await
    .unwrap();
    assert_eq!(batch.messages.len(), 1);
    assert!(batch.messages[0].is_error);
    let text = match &batch.messages[0].content[0] {
        ContentBlock::Text { text } => text.clone(),
        _ => panic!("expected text"),
    };
    assert!(
        text.to_lowercase().contains("value"),
        "schema error should mention missing field, got {text}"
    );
    assert!(events
        .lock()
        .unwrap()
        .iter()
        .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { is_error: true, .. })));
}

#[tokio::test]
async fn schema_valid_object_reaches_execute() {
    struct OkTool;
    #[async_trait]
    impl AgentTool for OkTool {
        fn name(&self) -> &str { "schema_tool" }
        fn label(&self) -> &str { "schema_tool" }
        fn description(&self) -> &str { "ok" }
        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            })
        }
        async fn execute(
            &self,
            _id: String,
            params: Value,
            _signal: Option<&watch::Receiver<bool>>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> Result<AgentToolResult, String> {
            Ok(AgentToolResult::text(params["value"].as_str().unwrap_or("").to_string()))
        }
    }
    let (emit, _) = recording_emit();
    let context = AgentContext {
        tools: vec![Arc::new(OkTool)],
        ..empty_context()
    };
    let assistant = assistant_with_tool_calls(vec![ToolCall {
        id: "c1".into(),
        name: "schema_tool".into(),
        arguments: json!({"value": "ok"}),
    }]);
    let batch = execute_tool_calls(
        &assistant,
        &context,
        &AgentLoopConfig::default(),
        None,
        &emit,
    )
    .await
    .unwrap();
    assert!(!batch.messages[0].is_error);
    match &batch.messages[0].content[0] {
        ContentBlock::Text { text } => assert_eq!(text, "ok"),
        _ => panic!("expected text"),
    }
}
```

Also add unit tests in `schema.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn non_object_is_rejected() {
        let err = validate_tool_arguments(&json!({"type": "object"}), &json!([])).unwrap_err();
        assert!(err.contains("object"));
    }

    #[test]
    fn default_object_schema_accepts_any_keys() {
        let v = validate_tool_arguments(&json!({"type": "object"}), &json!({"a": 1})).unwrap();
        assert_eq!(v["a"], 1);
    }

    #[test]
    fn missing_required_is_err() {
        let schema = json!({
            "type": "object",
            "properties": { "path": { "type": "string" } },
            "required": ["path"]
        });
        assert!(validate_tool_arguments(&schema, &json!({})).is_err());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p pi-agent-core --lib schema_required_field -- --nocapture`

Expected: FAIL — `parameters` not on `AgentTool`, or empty-object default accepts `{}`.

- [ ] **Step 3: Implement schema helper and trait defaults**

`crates/pi-agent-core/src/schema.rs`:

```rust
use serde_json::Value;

/// Validate tool-call arguments against a JSON Schema document.
///
/// Non-objects fail before schema validation (matches the previous default
/// `AgentTool::validate`). Schema compile/validation errors become a String
/// suitable for an error tool result. Callers must not invoke this on schemas
/// with external `$ref` from inside a tokio runtime (`jsonschema::validator_for`
/// documents that restriction); our tool schemas have no `$ref`.
pub fn validate_tool_arguments(schema: &Value, args: &Value) -> Result<Value, String> {
    if !args.is_object() {
        return Err("arguments must be an object".to_string());
    }
    jsonschema::validate(schema, args).map_err(|err| err.to_string())?;
    Ok(args.clone())
}
```

In `AgentTool`:

```rust
    /// JSON Schema for this tool's arguments. Sent to the model and used by
    /// the default `validate` implementation.
    fn parameters(&self) -> Value {
        serde_json::json!({ "type": "object" })
    }

    fn validate(&self, args: &Value) -> Result<Value, String> {
        crate::schema::validate_tool_arguments(&self.parameters(), args)
    }
```

Keep `prepare_arguments` unchanged. Do not change `tool_engine` call site: it already calls `tool.validate`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pi-agent-core`

Expected: PASS, including `validation_error_yields_error_result_without_executing` (override path) and the new schema tests.

- [ ] **Step 5: Commit**

```bash
git add crates/pi-agent-core/Cargo.toml crates/pi-agent-core/src/lib.rs crates/pi-agent-core/src/schema.rs crates/pi-agent-core/src/types.rs crates/pi-agent-core/src/tool_engine.rs Cargo.lock
git commit -m "$(cat <<'EOF'
feat(pi-agent-core): validate tool args against JSON Schema

Give AgentTool a parameters() schema so the loop and providers share one contract.
EOF
)"
```

---

### Task 2: Provider emits `parameters()` as-is

**Files:**
- Modify: `crates/aaos-openai/src/lib.rs` (`tools_payload`, EchoTool test)

**Interfaces:**
- Consumes: `AgentTool::parameters()`.
- Produces: OpenAI `tools[].function.parameters` equal to that Value.

- [ ] **Step 1: Write the failing assertion**

On `EchoTool` in `aaos-openai` tests, add:

```rust
        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "x": { "type": "number" } },
                "required": ["x"]
            })
        }
```

In `request_includes_messages_tools_thinking_and_bearer`, after the name assertion:

```rust
        assert_eq!(json["tools"][0]["function"]["parameters"]["required"], json!(["x"]));
        assert_eq!(
            json["tools"][0]["function"]["parameters"]["properties"]["x"]["type"],
            "number"
        );
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p aaos-openai request_includes_messages_tools_thinking_and_bearer -- --nocapture`

Expected: FAIL — payload still has `"properties": {}` and no `required`.

- [ ] **Step 3: Forward schema**

Replace `tools_payload`:

```rust
fn tools_payload(tools: &[Arc<dyn AgentTool>]) -> Option<Value> {
    if tools.is_empty() {
        return None;
    }
    Some(Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.parameters()
                    }
                })
            })
            .collect(),
    ))
}
```

- [ ] **Step 4: Run crate tests**

Run: `cargo test -p aaos-openai`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aaos-openai/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(aaos-openai): send tool JSON Schema in Completions payload
EOF
)"
```

---

### Task 3: `aaos-tools` path, truncate, mutation queue

**Files:**
- Modify: workspace `Cargo.toml` members
- Create: `crates/aaos-tools/Cargo.toml`
- Create: `crates/aaos-tools/src/lib.rs`
- Create: `crates/aaos-tools/src/path.rs`
- Create: `crates/aaos-tools/src/truncate.rs`
- Create: `crates/aaos-tools/src/mutation.rs`

**Interfaces:**
- Consumes: `&Path` cwd + path strings.
- Produces: `pub fn resolve_to_cwd(path: &str, cwd: &Path) -> PathBuf`; `pub struct Truncation { pub content: String, pub truncated: bool, ... }`; `pub fn truncate_head(text: &str) -> Truncation`; `pub struct FileMutationQueue`; `impl FileMutationQueue { pub async fn run<F, T>(&self, path: &Path, f: F) -> T where F: Future<Output = T> }`.

- [ ] **Step 1: Scaffold crate and write failing tests**

Workspace members add `"crates/aaos-tools"`.

`Cargo.toml`:

```toml
[package]
name = "aaos-tools"
version = "0.1.0"
edition = "2021"

[dependencies]
async-trait = "0.1"
pi-agent-core = { path = "../pi-agent-core" }
serde_json = "1"
tokio = { version = "1", features = ["fs", "io-util", "macros", "process", "rt", "sync", "time"] }

[dev-dependencies]
tempfile = "3"
tokio = { version = "1", features = ["macros", "rt-multi-thread"] }
```

`lib.rs`:

```rust
pub mod mutation;
pub mod path;
pub mod truncate;
```

Tests in `path.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn relative_joins_cwd() {
        let p = resolve_to_cwd("src/lib.rs", Path::new("/tmp/proj"));
        assert_eq!(p, Path::new("/tmp/proj/src/lib.rs"));
    }

    #[test]
    fn absolute_is_unchanged() {
        let p = resolve_to_cwd("/etc/hosts", Path::new("/tmp/proj"));
        assert_eq!(p, Path::new("/etc/hosts"));
    }
}
```

Tests in `truncate.rs` (limits are `MAX_LINES: usize = 2000` and `MAX_BYTES: usize = 50 * 1024`):

```rust
    #[test]
    fn short_text_is_not_truncated() {
        let t = truncate_head("a\nb");
        assert!(!t.truncated);
        assert_eq!(t.content, "a\nb");
    }

    #[test]
    fn line_limit_truncates() {
        let input = (0..MAX_LINES + 5).map(|i| format!("L{i}")).collect::<Vec<_>>().join("\n");
        let t = truncate_head(&input);
        assert!(t.truncated);
        assert_eq!(t.content.lines().count(), MAX_LINES);
    }
```

Tests in `mutation.rs`:

```rust
#[tokio::test]
async fn same_path_runs_do_not_overlap() {
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::Mutex;

    let q = FileMutationQueue::new();
    let order = Arc::new(Mutex::new(Vec::new()));
    let path = std::path::Path::new("/tmp/same");
    let a = {
        let order = order.clone();
        let q = &q;
        async move {
            q.run(path, async {
                order.lock().await.push("a-start");
                tokio::time::sleep(Duration::from_millis(50)).await;
                order.lock().await.push("a-end");
            })
            .await
        }
    };
    let b = {
        let order = order.clone();
        let q = &q;
        async move {
            q.run(path, async {
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p aaos-tools`

Expected: FAIL — missing functions.

- [ ] **Step 3: Implement**

`path.rs`: if `Path::new(path).is_absolute()` return that `PathBuf`, else `cwd.join(path)`.

`truncate.rs`: split on `\n`; take lines until hitting `MAX_LINES` or `MAX_BYTES` (UTF-8 byte length of the joined prefix). Set `truncated` if either limit cut the input.

`mutation.rs`:

```rust
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex};
use tokio::sync::Mutex;

#[derive(Default)]
pub struct FileMutationQueue {
    locks: StdMutex<HashMap<PathBuf, Arc<Mutex<()>>>>,
}

impl FileMutationQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub async fn run<F, T>(&self, path: &Path, fut: F) -> T
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
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p aaos-tools`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/aaos-tools Cargo.lock
git commit -m "$(cat <<'EOF'
feat(aaos-tools): add path, truncate, and file mutation queue
EOF
)"
```

---

### Task 4: `read` tool

**Files:**
- Create: `crates/aaos-tools/src/read.rs`
- Modify: `crates/aaos-tools/src/lib.rs` (`pub mod read;` and re-export `create_read_tool`)

**Interfaces:**
- Consumes: `cwd: PathBuf`; args `{ path: string, offset?: number, limit?: number }` (offset 1-indexed).
- Produces: `pub fn create_read_tool(cwd: impl Into<PathBuf>) -> Arc<dyn AgentTool>`; name `"read"`; schema required `path`; `prompt_snippet()` constant `"Read file contents"`.

- [ ] **Step 1: Write failing tests in `read.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_core::types::AgentTool;
    use serde_json::json;
    use std::fs;
    use tempfile::TempDir;

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
            pi_agent_core::types::ContentBlock::Text { text } => text,
            _ => panic!("text"),
        };
        assert!(text.contains("l2"));
        assert!(text.contains("l3"));
        assert!(!text.contains("l1"));
    }

    #[test]
    fn schema_requires_path() {
        let tool = create_read_tool("/tmp");
        let schema = tool.parameters();
        assert_eq!(schema["required"], json!(["path"]));
        assert!(pi_agent_core::schema::validate_tool_arguments(&schema, &json!({})).is_err());
    }
}
```

- [ ] **Step 2: Run to verify fail**

Run: `cargo test -p aaos-tools --lib reads_file -- --nocapture`

Expected: FAIL.

- [ ] **Step 3: Implement `ReadTool`**

Struct holds `cwd: PathBuf`. `parameters()`:

```json
{
  "type": "object",
  "properties": {
    "path": { "type": "string", "description": "Path to the file to read (relative or absolute)" },
    "offset": { "type": "number", "description": "Line number to start reading from (1-indexed)" },
    "limit": { "type": "number", "description": "Maximum number of lines to read" }
  },
  "required": ["path"]
}
```

`description`: explain text files, truncation to 2000 lines / 50KB, offset/limit for large files.

`execute`: `resolve_to_cwd` → `tokio::fs::read_to_string`. If offset is set, convert to 0-based; if start >= line count, return `Err(format!("Offset {offset} is beyond end of file ({n} lines total)"))`. Apply user `limit` first, then `truncate_head`. If truncated, append a continuation note with the next `offset=` like Pi (`Use offset={next} to continue`). Binary/non-UTF8: map `from_utf8`/`read_to_string` error to `Err`. Honor abort: if `signal` is `Some` and `*borrow()` is true before/after await, `Err("Operation aborted".into())`.

`label` = `"read"`. `name` = `"read"`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p aaos-tools`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/aaos-tools
git commit -m "$(cat <<'EOF'
feat(aaos-tools): add read tool with JSON Schema
EOF
)"
```

---

### Task 5: `write` tool

**Files:**
- Create: `crates/aaos-tools/src/write.rs`
- Modify: `crates/aaos-tools/src/lib.rs`

**Interfaces:**
- Consumes: `cwd`, shared `Arc<FileMutationQueue>`, args `{ path, content }`.
- Produces: `pub fn create_write_tool(cwd: impl Into<PathBuf>, queue: Arc<FileMutationQueue>) -> Arc<dyn AgentTool>`; name `"write"`.

- [ ] **Step 1: Failing tests**

```rust
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
        assert_eq!(fs::read_to_string(tmp.path().join("n/a.txt")).unwrap(), "hi");
    }

    #[test]
    fn schema_requires_path_and_content() {
        let tool = create_write_tool("/tmp", Arc::new(FileMutationQueue::new()));
        let schema = tool.parameters();
        assert_eq!(schema["required"], json!(["path", "content"]));
    }
```

- [ ] **Step 2: Run to fail**

Run: `cargo test -p aaos-tools creates_nested_file -- --nocapture`

Expected: FAIL.

- [ ] **Step 3: Implement**

Schema: `path` + `content` strings, both required. Description: create or overwrite; creates parent directories.

`execute`: resolve path; `queue.run(&abs, async { tokio::fs::create_dir_all(parent).await?; tokio::fs::write(&abs, content).await?; })`. Return text `Successfully wrote {n} bytes to {path}`. Abort checks around awaits.

- [ ] **Step 4: Run** `cargo test -p aaos-tools` — PASS.

- [ ] **Step 5: Commit** `feat(aaos-tools): add write tool`

---

### Task 6: `edit` tool

**Files:**
- Create: `crates/aaos-tools/src/edit.rs`
- Modify: `crates/aaos-tools/src/lib.rs`

**Interfaces:**
- Consumes: `cwd`, `Arc<FileMutationQueue>`, args `{ path, edits: [{ oldText, newText }] }`.
- Produces: `pub fn create_edit_tool(cwd, queue) -> Arc<dyn AgentTool>`; name `"edit"`; `prepare_arguments` unwraps stringified `edits` or a single edit object (Pi compatibility).

- [ ] **Step 1: Failing tests**

```rust
    #[tokio::test]
    async fn replaces_unique_block() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.rs"), "fn a() {}\nfn b() {}\n").unwrap();
        let tool = create_edit_tool(tmp.path(), Arc::new(FileMutationQueue::new()));
        tool.execute(
            "1".into(),
            json!({
                "path": "a.rs",
                "edits": [{ "oldText": "fn b() {}", "newText": "fn b() { 1 }" }]
            }),
            None,
            None,
        )
        .await
        .unwrap();
        let out = fs::read_to_string(tmp.path().join("a.rs")).unwrap();
        assert!(out.contains("fn b() { 1 }"));
        assert!(out.contains("fn a() {}"));
    }

    #[tokio::test]
    async fn rejects_non_unique_old_text() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.txt"), "x\nx\n").unwrap();
        let tool = create_edit_tool(tmp.path(), Arc::new(FileMutationQueue::new()));
        let err = tool
            .execute(
                "1".into(),
                json!({"path": "a.txt", "edits": [{ "oldText": "x", "newText": "y" }]}),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(err.to_lowercase().contains("unique") || err.to_lowercase().contains("multiple"));
    }

    #[test]
    fn prepare_arguments_parses_edits_json_string() {
        let tool = create_edit_tool("/tmp", Arc::new(FileMutationQueue::new()));
        let out = tool.prepare_arguments(json!({
            "path": "a.rs",
            "edits": "[{\"oldText\":\"a\",\"newText\":\"b\"}]"
        }));
        assert!(out["edits"].is_array());
        assert_eq!(out["edits"][0]["oldText"], "a");
    }
```

- [ ] **Step 2: Run to fail** — `cargo test -p aaos-tools replaces_unique_block -- --nocapture`

- [ ] **Step 3: Implement**

Schema (match Pi field names `oldText` / `newText`):

```json
{
  "type": "object",
  "properties": {
    "path": { "type": "string", "description": "Path to the file to edit (relative or absolute)" },
    "edits": {
      "type": "array",
      "description": "One or more targeted replacements. Each edit is matched against the original file, not incrementally.",
      "items": {
        "type": "object",
        "properties": {
          "oldText": { "type": "string" },
          "newText": { "type": "string" }
        },
        "required": ["oldText", "newText"]
      }
    }
  },
  "required": ["path", "edits"]
}
```

`prepare_arguments`: if `edits` is a string, `serde_json::from_str` into array or single `{oldText,newText}` object wrapped as one-element array; if `edits` is a single object, wrap as array; if top-level `oldText`/`newText` exist, append that pair into `edits`.

`execute`: require non-empty `edits`; `queue.run`; read file; for each edit, `match_indices` on the **original** content; 0 matches → error `oldText not found`; >1 → error not unique; collect ranges; reject overlapping ranges; splice replacements from low to high index; write file. Success text: `Successfully replaced {n} block(s) in {path}.`

- [ ] **Step 4:** `cargo test -p aaos-tools` PASS.

- [ ] **Step 5: Commit** `feat(aaos-tools): add edit tool with unique oldText matching`

---

### Task 7: `bash` tool

**Files:**
- Create: `crates/aaos-tools/src/bash.rs`
- Modify: `crates/aaos-tools/src/lib.rs`

**Interfaces:**
- Consumes: `cwd`, args `{ command: string, timeout?: number }` (timeout in seconds).
- Produces: `pub fn create_bash_tool(cwd: impl Into<PathBuf>) -> Arc<dyn AgentTool>`; name `"bash"`. Unix `bash -lc`.

- [ ] **Step 1: Failing tests** (`#[cfg(unix)]`)

```rust
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
            pi_agent_core::types::ContentBlock::Text { text } => text.clone(),
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
    }

    #[test]
    fn schema_requires_command() {
        let tool = create_bash_tool("/tmp");
        assert_eq!(tool.parameters()["required"], json!(["command"]));
    }
```

- [ ] **Step 2: Run to fail**

- [ ] **Step 3: Implement**

Schema: `command` required string; `timeout` optional number (seconds).

`execute`: `tokio::process::Command::new("bash").arg("-lc").arg(command).current_dir(&self.cwd).stdout(Stdio::piped()).stderr(Stdio::piped()).kill_on_drop(true)`. Combine stdout+stderr as they arrive (read both to strings after wait, concatenate stdout then stderr if both, or interleaved is not required — concatenate `stdout` + `stderr` is fine). If `timeout` is `Some(s)`, wrap wait in `tokio::time::timeout(Duration::from_secs_f64(s))`; on elapsed, kill child and `Err(format!("Command timed out after {s} seconds"))`. Poll abort receiver with `tokio::select!` against `child.wait()`. Non-zero exit: `Err` including captured output and `Command exited with code {n}` (kernel converts `Err` to error tool result). Truncate combined output with `truncate_head` before returning success. Missing cwd: `Err(Working directory does not exist: ...)`.

- [ ] **Step 4:** `cargo test -p aaos-tools` PASS.

- [ ] **Step 5: Commit** `feat(aaos-tools): add bash tool`

---

### Task 8: System prompt + `create_coding_tools`

**Files:**
- Create: `crates/aaos-tools/src/prompt.rs`
- Modify: `crates/aaos-tools/src/lib.rs`

**Interfaces:**
- Consumes: `cwd: &Path`, `tools: &[Arc<dyn AgentTool>]`.
- Produces: `pub fn create_coding_tools(cwd: impl Into<PathBuf>) -> Vec<Arc<dyn AgentTool>>` in order read, bash, edit, write sharing one `FileMutationQueue`; `pub fn build_system_prompt(cwd: &Path, tools: &[Arc<dyn AgentTool>]) -> String`.

- [ ] **Step 1: Failing tests in `prompt.rs` / `lib.rs`**

```rust
    #[test]
    fn lists_only_present_tools_and_cwd() {
        let tools = create_coding_tools("/tmp/work");
        let names: Vec<_> = tools.iter().map(|t| t.name().to_string()).collect();
        assert_eq!(names, ["read", "bash", "edit", "write"]);
        let prompt = build_system_prompt(Path::new("/tmp/work"), &tools);
        assert!(prompt.contains("Available tools:"));
        assert!(prompt.contains("- read: Read file contents"));
        assert!(prompt.contains("- bash: Execute bash commands"));
        assert!(prompt.contains("- edit:"));
        assert!(prompt.contains("- write: Create or overwrite files"));
        assert!(prompt.contains("Current working directory: /tmp/work"));
        assert!(prompt.contains("Use write only for new files or complete rewrites"));
        assert!(prompt.contains("edits[].oldText must match exactly"));
    }

    #[test]
    fn omits_missing_tool_lines() {
        let tools = vec![create_read_tool("/tmp")];
        let prompt = build_system_prompt(Path::new("/tmp"), &tools);
        assert!(prompt.contains("- read:"));
        assert!(!prompt.contains("- bash:"));
    }
```

- [ ] **Step 2: Run to fail**

- [ ] **Step 3: Implement**

Snippet table keyed by tool name (Pi wording):

- read: snippet `Read file contents`; guidelines `Use read to examine files instead of cat or sed.`
- bash: snippet `Execute bash commands (ls, grep, find, etc.)`; if bash present and grep/find/ls tools are absent, add guideline `Use bash for file operations like ls, rg, find`
- edit: snippet `Make precise file edits with exact text replacement, including multiple disjoint edits in one call`; guidelines as in Pi edit contribution (unique oldText, one call with `edits[]`, match original file, keep oldText small)
- write: snippet `Create or overwrite files`; guideline `Use write only for new files or complete rewrites.`

Always append: `Be concise in your responses`; `Show file paths clearly when working with files`.

Template:

```text
You are an expert coding assistant. You help users by reading files, executing commands, editing code, and writing new files.

Available tools:
{lines or (none)}

In addition to the tools above, you may have access to other custom tools depending on the project.

Guidelines:
{bullets}

Current working directory: {cwd with backslashes normalized to /}
```

A tool appears in Available tools only when its name has a snippet (all four do).

`create_coding_tools`:

```rust
pub fn create_coding_tools(cwd: impl Into<PathBuf>) -> Vec<Arc<dyn AgentTool>> {
    let cwd = cwd.into();
    let queue = Arc::new(FileMutationQueue::new());
    vec![
        create_read_tool(cwd.clone()),
        create_bash_tool(cwd.clone()),
        create_edit_tool(cwd.clone(), queue.clone()),
        create_write_tool(cwd, queue),
    ]
}
```

- [ ] **Step 4:** `cargo test -p aaos-tools` PASS.

- [ ] **Step 5: Commit** `feat(aaos-tools): assemble default tools and system prompt`

---

### Task 9: Thin `AgentSession`

**Files:**
- Modify: workspace `Cargo.toml` members (`crates/aaos-session`)
- Create: `crates/aaos-session/Cargo.toml`
- Create: `crates/aaos-session/src/lib.rs`

**Interfaces:**
- Consumes: `SessionOptions { cwd: PathBuf, model: Model, stream_fn: Arc<dyn StreamFn>, thinking_level: ThinkingLevel, api_key: Option<String> }`.
- Produces:

```rust
pub struct AgentSession { /* private agent: Agent */ }
impl AgentSession {
    pub fn new(opts: SessionOptions) -> Self;
    pub fn subscribe(&self, listener: Listener) -> impl FnOnce();
    pub async fn prompt(&mut self, text: impl Into<String>) -> Result<(), AgentError>;
    pub fn abort(&self);
    pub fn handle(&self) -> AgentHandle;
    pub fn agent(&self) -> &Agent;
    pub fn agent_mut(&mut self) -> &mut Agent;
}
```

`new` sets `agent.state.{model, thinking_level, tools, system_prompt}` from `create_coding_tools` + `build_system_prompt`, and `agent.stream_fn_options.api_key`.

- [ ] **Step 1: Write failing session integration test**

`Cargo.toml` depends on `aaos-tools`, `pi-agent-core`, `tokio` (macros, rt-multi-thread, sync). Dev-dep `tempfile`.

Test: fake `StreamFn` that on call 0 captures `LlmContext` and returns a `read` tool call for `note.txt`; on later calls returns assistant text `"done"` with `StopReason::Stop`. Use `pi_agent_core::stream::mock_stream_fn` + `MockAssistantStream::new(message)` (no events required).

```rust
#[tokio::test]
async fn prompt_runs_read_tool_and_sends_schema() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let tmp = tempfile::TempDir::new().unwrap();
    std::fs::write(tmp.path().join("note.txt"), "hello from file").unwrap();
    let captured: Arc<Mutex<Option<LlmContext>>> = Arc::new(Mutex::new(None));
    let cap = captured.clone();
    let n = Arc::new(AtomicUsize::new(0));
    let n2 = n.clone();
    let stream_fn = mock_stream_fn(move |_model, ctx, _opts| {
        let i = n2.fetch_add(1, Ordering::SeqCst);
        if i == 0 {
            *cap.lock().unwrap() = Some(ctx);
            let msg = AssistantMessage {
                content: vec![ContentBlock::tool_call(
                    "c1",
                    "read",
                    json!({"path": "note.txt"}),
                )],
                stop_reason: StopReason::ToolUse,
                ..Default::default()
            };
            Box::new(MockAssistantStream::new(msg))
        } else {
            Box::new(MockAssistantStream::new(AssistantMessage::text("done")))
        }
    });
    let mut session = AgentSession::new(SessionOptions {
        cwd: tmp.path().to_path_buf(),
        model: Model { id: "t".into(), ..Model::unknown() },
        stream_fn,
        thinking_level: ThinkingLevel::Off,
        api_key: None,
    });
    session.prompt("read the note").await.unwrap();
    let ctx = captured.lock().unwrap().clone().expect("first llm call");
    let names: Vec<_> = ctx.tools.iter().map(|t| t.name().to_string()).collect();
    assert_eq!(names, ["read", "bash", "edit", "write"]);
    let read = ctx.tools.iter().find(|t| t.name() == "read").unwrap();
    assert_eq!(read.parameters()["required"], json!(["path"]));
    assert!(ctx.system_prompt.contains("Available tools:"));
    assert!(ctx.system_prompt.contains(&tmp.path().display().to_string().replace('\\', "/"))
        || ctx.system_prompt.contains("Current working directory:"));
    let tool_text: String = session
        .agent()
        .state
        .messages
        .iter()
        .filter_map(|m| m.as_tool_result())
        .flat_map(|t| t.content.iter())
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(tool_text.contains("hello from file"), "{tool_text}");
    assert!(n.load(Ordering::SeqCst) >= 2);
}
```

- [ ] **Step 2: Run to fail** — `cargo test -p aaos-session -- --nocapture`

- [ ] **Step 3: Implement `AgentSession::new` as specified.** `prompt` calls `self.agent.prompt(text).await`. `subscribe` / `abort` / `handle` delegate to `Agent`.

- [ ] **Step 4:** `cargo test -p aaos-session` PASS. Also `cargo test -p pi-agent-core` still PASS.

- [ ] **Step 5: Commit** `feat(aaos-session): compose Agent, coding tools, and system prompt`

---

### Task 10: CLI print shell talks to `AgentSession`

**Files:**
- Modify: `crates/aaos-cli/Cargo.toml` (depend on `aaos-session`)
- Modify: `crates/aaos-cli/src/main.rs` (`run_prompt`)
- Modify: `crates/aaos-cli/tests/cli.rs` (`provider_model_thinking_flags_reach_request` also asserts tools)

**Interfaces:**
- Consumes: catalog model + API key + `OpenAiCompletionsProvider` as `Arc<dyn StreamFn>`.
- Produces: CLI still prints text deltas / `--json` inner `AssistantMessageEvent`s so existing tests keep matching `"type":"text_delta"` and `"type":"done"`.

- [ ] **Step 1: Extend the captured-request CLI test**

In `provider_model_thinking_flags_reach_request`, after existing JSON asserts:

```rust
    let names: Vec<&str> = json["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["read", "bash", "edit", "write"]);
    let read = &json["tools"][0]["function"]["parameters"];
    assert_eq!(read["required"], serde_json::json!(["path"]));
    let sys = json["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    assert!(sys["content"].as_str().unwrap().contains("Available tools:"));
```

- [ ] **Step 2: Run to fail** — `cargo test -p aaos-cli provider_model_thinking_flags_reach_request -- --nocapture`

Expected: FAIL — request has no tools / no system prompt.

- [ ] **Step 3: Wire session in `run_prompt`**

Replace `provider.call(...)` streaming loop with:

```rust
    let provider: Arc<dyn StreamFn> = Arc::new(OpenAiCompletionsProvider::new());
    let mut session = aaos_session::AgentSession::new(aaos_session::SessionOptions {
        cwd: std::env::current_dir().map_err(|e| e.to_string())?,
        model,
        stream_fn: provider,
        thinking_level: thinking,
        api_key: Some(api_key),
    });

    let json_mode = cli.json;
    let (event_tx, mut event_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
    let _unsub = session.subscribe(Arc::new(move |event, _signal| {
        let tx = event_tx.clone();
        Box::pin(async move {
            let _ = tx.send(event);
        })
    }));

    let handle = session.handle();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        handle.abort();
    });

    let prompt_fut = session.prompt(prompt);
    tokio::pin!(prompt_fut);
    let mut stdout = io::stdout();
    let mut terminal: Option<StopReason> = None;
    loop {
        tokio::select! {
            biased;
            event = event_rx.recv() => {
                let Some(event) = event else { break; };
                match &event {
                    AgentEvent::MessageUpdate { assistant_event, .. } => {
                        if json_mode {
                            writeln!(stdout, "{}", event_json(assistant_event)).map_err(|e| e.to_string())?;
                        } else if let AssistantMessageEvent::TextDelta { delta, .. } = assistant_event {
                            write!(stdout, "{delta}").map_err(|e| e.to_string())?;
                            let _ = stdout.flush();
                        }
                        match assistant_event {
                            AssistantMessageEvent::Done { reason, .. } => terminal = Some(*reason),
                            AssistantMessageEvent::Error { reason, .. } => terminal = Some(*reason),
                            _ => {}
                        }
                    }
                    AgentEvent::AgentEnd { .. } => {}
                    _ => {}
                }
            }
            result = &mut prompt_fut => {
                result.map_err(|e| e.to_string())?;
                // drain remaining events
                while let Ok(event) = event_rx.try_recv() {
                    /* same match as above, extract a local fn print_event to avoid duplication */
                }
                break;
            }
        }
    }
```

Extract a `fn handle_agent_event(...)` in `main.rs` so the drain path and the loop path share one implementation (do not copy-paste the match). Keep the same exit codes as today (`130` aborted, `1` error).

`OpenAiCompletionsProvider` must be `Arc<dyn StreamFn>`: it already implements `StreamFn`. Import `AgentEvent`, `AgentSession`, `SessionOptions`. Drop unused `LlmContext` / `UserMessage` imports if unused.

If `select!` drain is awkward, simpler acceptable CLI: `subscribe` that prints immediately (sync writeln in the async listener), then `session.prompt(prompt).await` — printing from the listener is enough for tests. Prefer that if the select loop is error-prone:

```rust
    let json_mode = cli.json;
    let _unsub = session.subscribe(Arc::new(move |event, _signal| {
        Box::pin(async move {
            if let AgentEvent::MessageUpdate { assistant_event, .. } = event {
                if json_mode {
                    println!("{}", event_json(&assistant_event));
                } else if let AssistantMessageEvent::TextDelta { delta, .. } = &assistant_event {
                    print!("{delta}");
                    let _ = io::Write::flush(&mut io::stdout());
                }
            }
        })
    }));
    session.prompt(prompt).await.map_err(|e| e.to_string())?;
```

Exit code: inspect `session.agent().state` last assistant `stop_reason` / `error_message` after `prompt` returns.

This simpler listener form is the intended CLI for this task.

- [ ] **Step 4: Run tests**

Run: `cargo test -p aaos-cli`

Expected: PASS, including tools/system-prompt assertions.

Run: `cargo test`

Expected: PASS whole workspace.

- [ ] **Step 5: Commit**

```bash
git add crates/aaos-cli Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
feat(aaos-cli): run prompts through AgentSession with default tools
EOF
)"
```

---

## Self-Review

**Spec coverage:** schema contract → Task 1; provider emit → Task 2; four tools → Tasks 4–7; system prompt → Task 8; session composition → Task 9; CLI on Agent (not raw StreamFn) → Task 10. Out of scope items (TUI, Presenter trait, compaction, images) have no tasks.

**Placeholder scan:** none.

**Type consistency:** `parameters() -> Value`; `FileMutationQueue::run`; `create_*_tool`; `create_coding_tools`; `build_system_prompt(cwd, tools)`; `SessionOptions` / `AgentSession::new` / `prompt` used the same way in Tasks 9–10.
