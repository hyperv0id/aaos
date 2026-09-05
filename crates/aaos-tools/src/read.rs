//! `read` coding tool: read text file contents with an optional line window.

use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use pi_agent_core::types::{AgentTool, AgentToolResult, AgentToolUpdateCallback};
use serde_json::{Value, json};
use tokio::sync::watch;

use crate::aborted;
use crate::skills::{SkillIndex, resolve_skill_url, split_uri_scheme};
use crate::truncate::truncate_head;

/// Create the `read` tool for a session. Relative `path` arguments are
/// resolved against `cwd`; `skill://` internal URIs are resolved against the
/// shared skill index.
pub fn create_read_tool(cwd: impl Into<PathBuf>, skills: Arc<SkillIndex>) -> Arc<dyn AgentTool> {
    Arc::new(ReadTool {
        cwd: cwd.into(),
        skills,
    })
}

struct ReadTool {
    cwd: PathBuf,
    skills: Arc<SkillIndex>,
}

/// Directory listing in oh-my-pi's `buildDirectoryResource` format: dirs
/// first, then name order, `name/` suffix for directories, and
/// `(empty directory)` when empty.
async fn skill_directory_listing(dir: &std::path::Path) -> Result<String, std::io::Error> {
    let mut entries = tokio::fs::read_dir(dir).await?;
    let mut items: Vec<(String, bool)> = Vec::new();
    while let Some(entry) = entries.next_entry().await? {
        let file_type = entry.file_type().await?;
        items.push((
            entry.file_name().to_string_lossy().into_owned(),
            file_type.is_dir(),
        ));
    }
    items.sort_by(|a, b| {
        let dir_order = (b.1 as u8).cmp(&(a.1 as u8));
        dir_order.then_with(|| a.0.cmp(&b.0))
    });
    if items.is_empty() {
        return Ok("(empty directory)".to_string());
    }
    Ok(items
        .iter()
        .map(|(name, is_dir)| format!("{name}{}", if *is_dir { "/" } else { "" }))
        .collect::<Vec<_>>()
        .join("\n"))
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

    fn description(&self) -> &str {
        "Read text file contents (relative to the session cwd, or absolute); \
         skill:// internal URIs read skills and their bundled resources. \
         Output is truncated to 2000 lines / 50KB; use offset and limit to \
         page through large files."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute), or a skill:// internal URI"
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

        // Scheme dispatch: `skill://` internal URIs resolve against the
        // skill index; anything else (including other `x://` forms) falls
        // through to the filesystem, byte-identical to before.
        let (mut content, source_path) = match split_uri_scheme(path) {
            Some((scheme, rest)) if scheme == "skill" => {
                let (host, url_path) = match rest.find('/') {
                    Some(idx) => (&rest[..idx], &rest[idx..]),
                    None => (rest, ""),
                };
                let target = resolve_skill_url(&self.skills, host, url_path).await?;
                if tokio::fs::metadata(&target).await.is_ok_and(|m| m.is_dir()) {
                    (
                        skill_directory_listing(&target)
                            .await
                            .map_err(|e| format!("Failed to read {path}: {e}"))?,
                        Some(target),
                    )
                } else {
                    (
                        tokio::fs::read_to_string(&target)
                            .await
                            .map_err(|e| format!("Failed to read {path}: {e}"))?,
                        Some(target),
                    )
                }
            }
            _ => {
                let resolved = self.cwd.join(path);
                // read_to_string fails on binary/non-UTF-8 content (io::Error
                // InvalidData); surface it as a tool error instead of mojibake.
                let text = tokio::fs::read_to_string(&resolved)
                    .await
                    .map_err(|e| format!("Failed to read {path}: {e}"))?;
                if aborted(signal) {
                    return Err("Operation aborted".to_string());
                }
                (text, None)
            }
        };

        if aborted(signal) {
            return Err("Operation aborted".to_string());
        }

        let total_lines = content.lines().count();
        let start_1based = offset.unwrap_or(1);
        let start_0based = start_1based - 1;
        if let Some(offset) = offset
            && start_0based >= total_lines
        {
            return Err(format!(
                "Offset {offset} is beyond end of file ({total_lines} lines total)"
            ));
        }

        // Caller's window first (offset + user limit), then shared truncation.
        let window = content.lines().skip(start_0based);
        content = match limit {
            Some(limit) => window.take(limit).collect::<Vec<_>>().join("\n"),
            None => window.collect::<Vec<_>>().join("\n"),
        };

        let truncation = truncate_head(&content);
        content = truncation.content;
        if truncation.truncated {
            let next_offset = start_1based + content.lines().count();
            content.push_str(&format!("\n\nUse offset={next_offset} to continue"));
        }

        // Skill results carry the resolved real path for the model, both as a
        // first line in the visible text and in details.sourcePath.
        let result = match source_path {
            Some(source) => {
                let source_display = source.display().to_string();
                let mut result =
                    AgentToolResult::text(format!("Source: {source_display}\n\n{content}"));
                result.details = json!({ "sourcePath": source_display });
                result
            }
            None => AgentToolResult::text(content),
        };
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    #![expect(clippy::panic)]
    use super::*;
    use pi_agent_core::types::ContentBlock;
    use serde_json::json;
    use std::fs;
    use std::sync::Arc;
    use tempfile::TempDir;
    use tokio::sync::watch;

    fn text_of(result: &AgentToolResult) -> &str {
        match &result.content[0] {
            ContentBlock::Text { text } => text,
            _ => panic!("expected text block"),
        }
    }

    fn empty_index() -> Arc<SkillIndex> {
        Arc::new(SkillIndex::discover(
            std::path::Path::new("/nonexistent"),
            std::path::Path::new("/nonexistent"),
        ))
    }

    fn write_skill(dir: &std::path::Path, name: &str, frontmatter: &str) {
        let skill_dir = dir.join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(skill_dir.join("SKILL.md"), frontmatter).unwrap();
    }

    #[tokio::test]
    async fn reads_file_with_offset_and_limit() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.txt"), "l1\nl2\nl3\nl4\n").unwrap();
        let tool = create_read_tool(tmp.path(), empty_index());
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "a.txt", "offset": 2, "limit": 2}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = text_of(&result);
        assert!(text.contains("l2"));
        assert!(text.contains("l3"));
        assert!(!text.contains("l1"));
    }

    #[tokio::test]
    async fn empty_file_returns_empty_text() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("empty.txt"), "").unwrap();
        let tool = create_read_tool(tmp.path(), empty_index());
        let result = tool
            .execute("1".into(), json!({"path": "empty.txt"}), None, None)
            .await
            .expect("empty file without offset should succeed");
        let text = text_of(&result);
        assert_eq!(text, "");
    }

    #[tokio::test]
    async fn offset_past_eof_errors() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("f.txt"), "one\ntwo\nthree\n").unwrap();
        let tool = create_read_tool(tmp.path(), empty_index());
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
    async fn oversized_output_has_continuation_hint() {
        use crate::truncate::MAX_LINES;
        let tmp = TempDir::new().unwrap();
        let big = (0..MAX_LINES + 5)
            .map(|i| format!("L{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(tmp.path().join("big.txt"), &big).unwrap();
        let tool = create_read_tool(tmp.path(), empty_index());
        let result = tool
            .execute("1".into(), json!({"path": "big.txt"}), None, None)
            .await
            .unwrap();
        let text = text_of(&result);
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
        let tool = create_read_tool(tmp.path(), empty_index());
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "big.txt", "offset": 500}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = text_of(&result);
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
        let tool = create_read_tool(tmp.path(), empty_index());
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
        let text = text_of(&result);
        assert_eq!(text.lines().count(), 10);
        assert!(!text.contains("to continue"));
    }

    #[tokio::test]
    async fn non_utf8_file_errors() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("bin.dat"), [0xFFu8, 0xFE, 0x00]).unwrap();
        let tool = create_read_tool(tmp.path(), empty_index());
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
        let tool = create_read_tool("/some/other/cwd", empty_index());
        let result = tool
            .execute(
                "1".into(),
                json!({"path": abs.to_str().unwrap()}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = text_of(&result);
        assert_eq!(text, "hello");
    }

    #[tokio::test]
    async fn aborts_when_signal_set() {
        let tmp = TempDir::new().unwrap();
        fs::write(tmp.path().join("a.txt"), "l1\nl2\n").unwrap();
        let tool = create_read_tool(tmp.path(), empty_index());
        let (tx, rx) = watch::channel(false);
        tx.send(true).unwrap();
        let err = tool
            .execute("1".into(), json!({"path": "a.txt"}), Some(&rx), None)
            .await
            .unwrap_err();
        assert_eq!(err, "Operation aborted");
    }

    // --- skill:// internal URI dispatch ---

    fn skill_index_with(dir: &std::path::Path) -> Arc<SkillIndex> {
        Arc::new(SkillIndex::discover(
            std::path::Path::new("/nonexistent"),
            dir,
        ))
    }

    #[tokio::test]
    async fn skill_url_reads_skill_md() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\n# Fmt\nrun fmt\n",
        );
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let result = tool
            .execute("1".into(), json!({"path": "skill://fmt"}), None, None)
            .await
            .unwrap();
        let text = text_of(&result);
        assert!(text.contains("# Fmt"), "got: {text}");
        assert!(text.contains("run fmt"), "got: {text}");
        assert!(
            text.starts_with("Source: "),
            "sourcePath must be visible in text: {text}"
        );
        assert!(
            text.contains(&skills_dir.join("fmt/SKILL.md").display().to_string()),
            "real path must appear: {text}"
        );
        assert_eq!(
            result.details["sourcePath"],
            json!(skills_dir.join("fmt/SKILL.md").display().to_string())
        );
    }

    #[tokio::test]
    async fn skill_url_trailing_slash_reads_skill_md() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\nbody-content\n",
        );
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let result = tool
            .execute("1".into(), json!({"path": "skill://fmt/"}), None, None)
            .await
            .unwrap();
        let text = text_of(&result);
        assert!(text.contains("body-content"), "got: {text}");
    }

    #[tokio::test]
    async fn skill_url_explicit_skill_md_is_same_file() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\nSAME-BODY\n",
        );
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "skill://fmt/SKILL.md"}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = text_of(&result);
        assert!(text.contains("SAME-BODY"), "got: {text}");
    }

    #[tokio::test]
    async fn skill_url_reads_bundled_file_and_lists_directory() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\n# Fmt\n",
        );
        fs::create_dir_all(skills_dir.join("fmt/scripts")).unwrap();
        fs::write(
            skills_dir.join("fmt/scripts/run.sh"),
            "#!/bin/sh\necho hi\n",
        )
        .unwrap();
        fs::write(skills_dir.join("fmt/notes.txt"), "note text\n").unwrap();
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));

        let result = tool
            .execute(
                "1".into(),
                json!({"path": "skill://fmt/scripts/run.sh"}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = text_of(&result);
        assert!(text.contains("echo hi"), "got: {text}");
        assert!(
            text.contains(&skills_dir.join("fmt/scripts/run.sh").display().to_string()),
            "sourcePath for bundled file: {text}"
        );

        // `skill://fmt` is SKILL.md; the root directory is addressed as
        // `skill://fmt/..` — instead list a subdirectory directly.
        let listing = tool
            .execute(
                "1".into(),
                json!({"path": "skill://fmt/scripts"}),
                None,
                None,
            )
            .await
            .unwrap();
        let listing_text = text_of(&listing);
        assert!(
            listing_text.contains("run.sh"),
            "listing must contain run.sh: {listing_text}"
        );
        assert_eq!(
            listing.details["sourcePath"],
            json!(skills_dir.join("fmt/scripts").display().to_string())
        );

        // Root directory listing via the parent: `skill://fmt/SKILL.md` is a
        // file, so exercise listing at the skill root by addressing the dir.
        let root_listing = tool
            .execute("1".into(), json!({"path": "skill://fmt/."}), None, None)
            .await
            .unwrap();
        let root_text = text_of(&root_listing);
        // dirs first: scripts/ then files.
        assert!(
            root_text.contains("scripts/\nSKILL.md")
                || root_text.contains("scripts/\nnotes.txt\nSKILL.md"),
            "dirs-first listing: {root_text}"
        );
        assert!(root_text.contains("notes.txt"), "got: {root_text}");
    }

    #[tokio::test]
    async fn skill_url_empty_directory_lists_empty() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\n# Fmt\n",
        );
        fs::create_dir_all(skills_dir.join("fmt/empty")).unwrap();
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let result = tool
            .execute("1".into(), json!({"path": "skill://fmt/empty"}), None, None)
            .await
            .unwrap();
        let text = text_of(&result);
        assert!(text.contains("(empty directory)"), "got: {text}");
    }

    #[tokio::test]
    async fn skill_url_percent_decodes_path_segment() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\n# Fmt\n",
        );
        fs::write(skills_dir.join("fmt/notes v2.txt"), "decoded name\n").unwrap();
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "skill://fmt/notes%20v2.txt"}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = text_of(&result);
        assert!(text.contains("decoded name"), "got: {text}");
    }

    #[tokio::test]
    async fn skill_url_malformed_percent_errors() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\n# Fmt\n",
        );
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let err = tool
            .execute(
                "1".into(),
                json!({"path": "skill://fmt/notes%2.txt"}),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(err.contains("invalid percent-encoding"), "got: {err}");
    }

    #[tokio::test]
    async fn skill_url_rejects_dotdot() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\n# Fmt\n",
        );
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let err = tool
            .execute(
                "1".into(),
                json!({"path": "skill://fmt/../../etc/passwd"}),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(
            err.contains("Path traversal") || err.contains(".."),
            "got: {err}"
        );
    }

    #[tokio::test]
    async fn skill_url_rejects_absolute_path() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\n# Fmt\n",
        );
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let err = tool
            .execute(
                "1".into(),
                json!({"path": "skill://fmt//etc/passwd"}),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(err.contains("Absolute paths are not allowed"), "got: {err}");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn skill_url_rejects_symlink_escape() {
        use std::os::unix::fs::symlink;
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\n# Fmt\n",
        );
        let outside = tmp.path().join("outside-secret.txt");
        fs::write(&outside, "secret").unwrap();
        symlink(&outside, skills_dir.join("fmt/escaped.txt")).unwrap();
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let err = tool
            .execute(
                "1".into(),
                json!({"path": "skill://fmt/escaped.txt"}),
                None,
                None,
            )
            .await
            .unwrap_err();
        assert!(
            err.contains("outside the skill directory") || err.contains("File not found"),
            "symlink escape must be rejected, got: {err}"
        );
    }

    #[tokio::test]
    async fn skill_url_unknown_skill_lists_available() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(&skills_dir, "fmt", "---\ndescription: format code\n---\n");
        write_skill(&skills_dir, "build", "---\ndescription: build it\n---\n");
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let err = tool
            .execute("1".into(), json!({"path": "skill://nope"}), None, None)
            .await
            .unwrap_err();
        assert!(
            err.contains("Unknown skill: nope"),
            "unknown skill error: {err}"
        );
        assert!(
            err.contains("Available: build, fmt") || err.contains("Available: fmt, build"),
            "available list: {err}"
        );
    }

    #[tokio::test]
    async fn skill_url_empty_name_errors() {
        let tmp = TempDir::new().unwrap();
        let tool = create_read_tool(tmp.path(), empty_index());
        let err = tool
            .execute("1".into(), json!({"path": "skill://"}), None, None)
            .await
            .unwrap_err();
        assert!(
            err.contains("requires a skill name"),
            "empty skill name: {err}"
        );
    }

    #[tokio::test]
    async fn skill_read_applies_offset_limit_and_source_line() {
        let tmp = TempDir::new().unwrap();
        let skills_dir = tmp.path().join(".agents/skills");
        fs::create_dir_all(&skills_dir).unwrap();
        write_skill(
            &skills_dir,
            "fmt",
            "---\ndescription: format code\n---\nl1\nl2\nl3\nl4\n",
        );
        let tool = create_read_tool(tmp.path(), skill_index_with(&skills_dir));
        let result = tool
            .execute(
                "1".into(),
                json!({"path": "skill://fmt", "offset": 2, "limit": 2}),
                None,
                None,
            )
            .await
            .unwrap();
        let text = text_of(&result);
        // The Source: line is prepended after windowing; the window itself
        // still starts at file line 2 (frontmatter line 2 is `description:`).
        let body = text.split_once("\n\n").map(|(_, b)| b).unwrap_or(text);
        assert!(body.contains("description: format code"), "got: {text}");
        assert!(!body.contains("l1"), "got: {text}");
        assert!(text.starts_with("Source: "), "got: {text}");
    }
}
