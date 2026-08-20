use std::sync::Arc;

use async_trait::async_trait;
use pi_agent_core::types::AgentTool;
use pi_agent_core::types::AgentToolResult;
use serde_json::Value;

use crate::mutation::FileMutationQueue;
use crate::path::resolve_to_cwd;

/// Create the `edit` tool for a session.
///
/// `cwd` is used to resolve relative `path` arguments; mutations are
/// serialized through the shared per-path `queue` so concurrent edits to the
/// same file never interleave.
pub fn create_edit_tool(
    cwd: impl Into<std::path::PathBuf>,
    queue: Arc<FileMutationQueue>,
) -> Arc<dyn AgentTool> {
    Arc::new(EditTool {
        cwd: cwd.into(),
        queue,
    })
}

struct EditTool {
    cwd: std::path::PathBuf,
    queue: Arc<FileMutationQueue>,
}

impl EditTool {
    fn schema(&self) -> Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative or absolute)"
                },
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
            "required": ["path", "edits"],
            "additionalProperties": false
        })
    }

    /// Normalize Pi-style argument shapes into the canonical
    /// `{ path, edits: [{oldText, newText}] }` form:
    /// - stringified `edits` (JSON array or single object) is parsed;
    /// - a single edit object is wrapped into a one-element array;
    /// - top-level `oldText`/`newText` are appended as one more edit.
    fn prepare_arguments(&self, mut args: Value) -> Value {
        let normalized = match args.get("edits") {
            Some(Value::String(s)) => match serde_json::from_str::<Value>(s) {
                Ok(Value::Array(a)) => Value::Array(a),
                Ok(object @ Value::Object(_)) => Value::Array(vec![object]),
                _ => Value::String(s.clone()),
            },
            Some(object @ Value::Object(_)) => Value::Array(vec![object.clone()]),
            Some(other @ Value::Array(_)) => other.clone(),
            _ => Value::Array(Vec::new()),
        };
        args["edits"] = normalized;

        // Append the top-level oldText/newText pair, then drop those keys so
        // the argument object strictly matches the schema.
        let pair = match (
            args.get("oldText").and_then(Value::as_str),
            args.get("newText").and_then(Value::as_str),
        ) {
            (Some(old), Some(new)) => Some((old.to_string(), new.to_string())),
            _ => None,
        };
        if let Some((old, new)) = pair {
            if let Some(edits) = args["edits"].as_array_mut() {
                edits.push(serde_json::json!({ "oldText": old, "newText": new }));
            }
        }
        if let Some(obj) = args.as_object_mut() {
            obj.remove("oldText");
            obj.remove("newText");
        }
        args
    }
}

#[async_trait]
impl AgentTool for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    fn label(&self) -> &str {
        "edit"
    }

    fn description(&self) -> &str {
        "Apply exact-text replacements to a file. Each oldText must match the \
         original file content exactly once; several disjoint edits may be sent \
         in a single call. Use for targeted changes; prefer write for new files \
         or full rewrites."
    }

    fn parameters(&self) -> Value {
        self.schema()
    }

    fn prepare_arguments(&self, args: Value) -> Value {
        self.prepare_arguments(args)
    }

    async fn execute(
        &self,
        _tool_call_id: String,
        params: Value,
        _signal: Option<&tokio::sync::watch::Receiver<bool>>,
        _on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        let path = params["path"]
            .as_str()
            .ok_or_else(|| "missing or non-string `path`".to_string())?;
        let edits = params["edits"]
            .as_array()
            .ok_or_else(|| "missing or non-array `edits`".to_string())?;
        if edits.is_empty() {
            return Err("`edits` must contain at least one edit".to_string());
        }

        let mut parsed: Vec<(String, String)> = Vec::with_capacity(edits.len());
        for e in edits {
            let old_text = e["oldText"]
                .as_str()
                .ok_or_else(|| "each edit requires a string `oldText`".to_string())?;
            let new_text = e["newText"]
                .as_str()
                .ok_or_else(|| "each edit requires a string `newText`".to_string())?;
            parsed.push((old_text.to_string(), new_text.to_string()));
        }

        let abs = resolve_to_cwd(path, &self.cwd);
        let abs_inner = abs.clone();
        let queue = self.queue.clone();

        let n = queue
            .run(&abs, async move {
                let content = tokio::fs::read_to_string(&abs_inner)
                    .await
                    .map_err(|e| format!("failed to read {path}: {e}"))?;

                // Match every edit against the original content: zero or
                // multiple matches are both errors and leave the file untouched.
                let mut matched: Vec<(usize, usize, String)> = Vec::with_capacity(parsed.len());
                for (old_text, new_text) in &parsed {
                    let mut it = content.match_indices(old_text.as_str());
                    let Some((start, _)) = it.next() else {
                        return Err(format!("oldText {old_text:?} not found in {path}"));
                    };
                    if it.next().is_some() {
                        return Err(format!(
                            "oldText {old_text:?} is not unique in {path}; exactly one match is required"
                        ));
                    }
                    matched.push((start, start + old_text.len(), new_text.clone()));
                }

                // Disjoint ranges low-to-high; overlapping replacements are
                // rejected before anything is written.
                matched.sort_by_key(|(start, _, _)| *start);
                for pair in matched.windows(2) {
                    if pair[1].0 < pair[0].1 {
                        return Err("overlapping edits are not allowed".to_string());
                    }
                }

                let mut out = String::with_capacity(content.len());
                let mut pos = 0usize;
                for (start, end, new_text) in &matched {
                    out.push_str(&content[pos..*start]);
                    out.push_str(new_text);
                    pos = *end;
                }
                out.push_str(&content[pos..]);

                tokio::fs::write(&abs_inner, out)
                    .await
                    .map_err(|e| format!("failed to write {path}: {e}"))?;
                Ok(matched.len())
            })
            .await?;

        Ok(AgentToolResult::text(format!(
            "Successfully replaced {n} block(s) in {path}."
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    #[tokio::test]
    async fn replaces_unique_block() {
        let tmp = TempDir::new().unwrap();
        tokio::fs::write(tmp.path().join("a.rs"), "fn a() {}\nfn b() {}\n")
            .await
            .unwrap();
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
        let out = tokio::fs::read_to_string(tmp.path().join("a.rs"))
            .await
            .unwrap();
        assert!(out.contains("fn b() { 1 }"));
        assert!(out.contains("fn a() {}"));
    }

    #[tokio::test]
    async fn rejects_non_unique_old_text() {
        let tmp = TempDir::new().unwrap();
        tokio::fs::write(tmp.path().join("a.txt"), "x\nx\n")
            .await
            .unwrap();
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
        assert!(
            err.to_lowercase().contains("unique") || err.to_lowercase().contains("multiple"),
            "unexpected error: {err}"
        );
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

    #[tokio::test]
    async fn applies_disjoint_edits_in_one_call() {
        let tmp = TempDir::new().unwrap();
        tokio::fs::write(tmp.path().join("a.rs"), "alpha beta gamma\n")
            .await
            .unwrap();
        let tool = create_edit_tool(tmp.path(), Arc::new(FileMutationQueue::new()));
        let res = tool
            .execute(
                "1".into(),
                json!({
                    "path": "a.rs",
                    "edits": [
                        { "oldText": "alpha", "newText": "ALPHA" },
                        { "oldText": "gamma", "newText": "GAMMA" }
                    ]
                }),
                None,
                None,
            )
            .await
            .unwrap();
        let out = tokio::fs::read_to_string(tmp.path().join("a.rs"))
            .await
            .unwrap();
        assert_eq!(out, "ALPHA beta GAMMA\n");
        assert!(res
            .content
            .iter()
            .any(|b| matches!(b, pi_agent_core::types::ContentBlock::Text { text } if text.contains("2 block(s)"))));
    }

    #[tokio::test]
    async fn rejects_overlapping_edits_without_modifying_file() {
        let tmp = TempDir::new().unwrap();
        let original = "abcdef\n";
        tokio::fs::write(tmp.path().join("o.txt"), original)
            .await
            .unwrap();
        let tool = create_edit_tool(tmp.path(), Arc::new(FileMutationQueue::new()));
        let err = tool
            .execute(
                "1".into(),
                json!({
                    "path": "o.txt",
                    "edits": [
                        { "oldText": "bcd", "newText": "XXX" },
                        { "oldText": "cde", "newText": "YYY" }
                    ]
                }),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(err.contains("overlapping"), "unexpected error: {err}");
        let out = tokio::fs::read_to_string(tmp.path().join("o.txt"))
            .await
            .unwrap();
        assert_eq!(out, original);
    }

    #[tokio::test]
    async fn rejects_missing_oldtext_without_modifying_file() {
        let tmp = TempDir::new().unwrap();
        let original = "hello\n";
        tokio::fs::write(tmp.path().join("m.txt"), original)
            .await
            .unwrap();
        let tool = create_edit_tool(tmp.path(), Arc::new(FileMutationQueue::new()));
        let err = tool
            .execute(
                "1".into(),
                json!({"path": "m.txt", "edits": [{ "oldText": "world", "newText": "WORLD" }]}),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(err.contains("not found"), "unexpected error: {err}");
        let out = tokio::fs::read_to_string(tmp.path().join("m.txt"))
            .await
            .unwrap();
        assert_eq!(out, original);
    }

    #[tokio::test]
    async fn rejects_empty_edits() {
        let tmp = TempDir::new().unwrap();
        tokio::fs::write(tmp.path().join("e.txt"), "data\n")
            .await
            .unwrap();
        let tool = create_edit_tool(tmp.path(), Arc::new(FileMutationQueue::new()));
        let err = tool
            .execute("1".into(), json!({"path": "e.txt", "edits": []}), None, None)
            .await
            .unwrap_err();
        assert!(err.contains("at least one"), "unexpected error: {err}");
    }

    #[test]
    fn prepare_arguments_wraps_single_edit_object() {
        let tool = create_edit_tool("/tmp", Arc::new(FileMutationQueue::new()));
        let out = tool.prepare_arguments(json!({
            "path": "a.rs",
            "edits": { "oldText": "a", "newText": "b" }
        }));
        assert!(out["edits"].is_array());
        assert_eq!(out["edits"][0]["oldText"], "a");
        assert_eq!(out["edits"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn prepare_arguments_parses_stringified_single_object() {
        let tool = create_edit_tool("/tmp", Arc::new(FileMutationQueue::new()));
        let out = tool.prepare_arguments(json!({
            "path": "a.rs",
            "edits": "{\"oldText\":\"a\",\"newText\":\"b\"}"
        }));
        assert_eq!(out["edits"][0]["newText"], "b");
        assert_eq!(out["edits"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn prepare_arguments_appends_top_level_pair() {
        let tool = create_edit_tool("/tmp", Arc::new(FileMutationQueue::new()));
        let out = tool.prepare_arguments(json!({
            "path": "a.rs",
            "oldText": "x",
            "newText": "y"
        }));
        assert_eq!(out["edits"].as_array().unwrap().len(), 1);
        assert_eq!(out["edits"][0]["oldText"], "x");
        assert!(out.get("oldText").is_none());
        assert!(out.get("newText").is_none());
    }

    #[test]
    fn parameters_requires_path_and_edits() {
        let tool = create_edit_tool("/tmp", Arc::new(FileMutationQueue::new()));
        let schema = tool.parameters();
        let required = schema["required"].as_array().unwrap();
        let req: Vec<&str> = required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(req.contains(&"path"));
        assert!(req.contains(&"edits"));
        let item_required = schema["properties"]["edits"]["items"]["required"]
            .as_array()
            .unwrap();
        let ireq: Vec<&str> = item_required.iter().map(|v| v.as_str().unwrap()).collect();
        assert!(ireq.contains(&"oldText") && ireq.contains(&"newText"));
    }
}
