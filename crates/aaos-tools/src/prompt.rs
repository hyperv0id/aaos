//! Default coding-tool assembly and the system prompt they are described with.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pi_agent_core::types::AgentTool;

use crate::mutation::FileMutationQueue;
use crate::skills::SkillIndex;
use crate::write::create_write_tool;
use crate::{create_bash_tool, create_edit_tool, create_read_tool};

/// Assemble the default coding tools in Pi order, sharing one mutation queue
/// between `edit` and `write`.
pub fn create_coding_tools(
    cwd: impl Into<PathBuf>,
    skills: Arc<SkillIndex>,
) -> Vec<Arc<dyn AgentTool>> {
    let cwd = cwd.into();
    let queue = Arc::new(FileMutationQueue::new());
    vec![
        create_read_tool(cwd.clone(), skills),
        create_bash_tool(cwd.clone()),
        create_edit_tool(cwd.clone(), queue.clone()),
        create_write_tool(cwd, queue),
    ]
}

/// One-line tool list entries, keyed by `AgentTool::name()`.
fn tool_list_entry(name: &str) -> Option<&'static str> {
    match name {
        "read" => Some("Read file contents"),
        "bash" => Some("Execute bash commands (ls, grep, find, etc.)"),
        "edit" => Some(
            "Make precise file edits with exact text replacement, including multiple disjoint edits in one call",
        ),
        "write" => Some("Create or overwrite files"),
        _ => None,
    }
}

/// Per-tool guideline bullets contributed when that tool is present.
fn tool_guidelines(name: &str) -> &'static [&'static str] {
    match name {
        "read" => &["Use read to examine files instead of cat or sed."],
        "edit" => &[
            "Use edit for precise changes (edits[].oldText must match exactly)",
            "When changing multiple separate locations in one file, use one edit call with multiple entries in edits[] instead of multiple edit calls",
            "Each edits[].oldText is matched against the original file, not after earlier edits are applied. Do not emit overlapping or nested edits. Merge nearby changes into one edit.",
            "Keep edits[].oldText as small as possible while still being unique in the file. Do not pad with large unchanged regions.",
        ],
        "write" => &["Use write only for new files or complete rewrites."],
        _ => &[],
    }
}

/// Build the system prompt from `cwd` and the tools actually present.
///
/// A tool is listed under Available tools only when its name has a list entry.
/// The bash-for-file-ops guideline is included only when `bash` is present.
pub fn build_system_prompt(
    cwd: &Path,
    tools: &[Arc<dyn AgentTool>],
    skills: &SkillIndex,
) -> String {
    let tool_lines: Vec<String> = tools
        .iter()
        .filter_map(|tool| {
            tool_list_entry(tool.name()).map(|entry| format!("- {}: {entry}", tool.name()))
        })
        .collect();
    let tools_list = if tool_lines.is_empty() {
        "(none)".to_string()
    } else {
        tool_lines.join("\n")
    };

    let has_tool = |name: &str| tools.iter().any(|tool| tool.name() == name);
    let mut guidelines: Vec<&str> = Vec::new();
    if has_tool("bash") {
        guidelines.push("Use bash for file operations like ls, rg, find");
    }
    for tool in tools {
        guidelines.extend_from_slice(tool_guidelines(tool.name()));
    }
    guidelines.push("Be concise in your responses");
    guidelines.push("Show file paths clearly when working with files");
    let guidelines = guidelines
        .iter()
        .map(|line| format!("- {line}"))
        .collect::<Vec<_>>()
        .join("\n");

    let cwd = cwd.to_string_lossy().replace('\\', "/");

    let skills_block = if has_tool("read") && !skills.is_empty() {
        format!("\n\n{}\n", skills.prompt_block())
    } else {
        "\n".to_string()
    };

    format!(
        "You are an expert coding assistant. You help users by reading files, executing commands, editing code, and writing new files.\n\
         \n\
         Available tools:\n\
         {tools_list}\n\
         \n\
         In addition to the tools above, you may have access to other custom tools depending on the project.\n\
         \n\
         Guidelines:\n\
         {guidelines}{skills_block}\n\
         Current working directory: {cwd}"
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::path::Path;

    use crate::skills::SkillIndex;
    use crate::{build_system_prompt, create_coding_tools, create_read_tool};

    fn write_skill(dir: &Path, name: &str, description: &str) {
        let skill_dir = dir.join(name);
        std::fs::create_dir_all(&skill_dir).unwrap();
        std::fs::write(
            skill_dir.join("SKILL.md"),
            format!("---\ndescription: {description}\n---\nbody\n"),
        )
        .unwrap();
    }

    #[test]
    fn lists_only_present_tools_and_cwd() {
        let tools = create_coding_tools(
            "/tmp/work",
            std::sync::Arc::new(SkillIndex::discover(
                Path::new("/nonexistent"),
                Path::new("/nonexistent"),
            )),
        );
        let names: Vec<_> = tools.iter().map(|t| t.name().to_string()).collect();
        assert_eq!(names, ["read", "bash", "edit", "write"]);
        let prompt = build_system_prompt(
            Path::new("/tmp/work"),
            &tools,
            &SkillIndex::discover(Path::new("/nonexistent"), Path::new("/nonexistent")),
        );
        assert!(prompt.contains("Available tools:"));
        assert!(prompt.contains("- read: Read file contents"));
        assert!(prompt.contains("- bash: Execute bash commands"));
        assert!(prompt.contains("- edit:"));
        assert!(prompt.contains("- write: Create or overwrite files"));
        assert!(prompt.contains("Current working directory: /tmp/work"));
        assert!(prompt.contains("Use write only for new files or complete rewrites"));
        assert!(prompt.contains("edits[].oldText must match exactly"));
        assert!(prompt.contains("Use bash for file operations like ls, rg, find"));
        let prompt_win = build_system_prompt(
            Path::new("C:\\tmp\\work"),
            &tools,
            &SkillIndex::discover(Path::new("/nonexistent"), Path::new("/nonexistent")),
        );
        assert!(prompt_win.contains("Current working directory: C:/tmp/work"));
        assert!(!prompt_win.contains("C:\\tmp\\work"));
    }

    #[test]
    fn omits_missing_tool_lines() {
        let tools = vec![create_read_tool(
            "/tmp",
            std::sync::Arc::new(SkillIndex::discover(
                Path::new("/nonexistent"),
                Path::new("/nonexistent"),
            )),
        )];
        let prompt = build_system_prompt(
            Path::new("/tmp"),
            &tools,
            &SkillIndex::discover(Path::new("/nonexistent"), Path::new("/nonexistent")),
        );
        assert!(prompt.contains("- read:"));
        assert!(!prompt.contains("- bash:"));
        assert!(!prompt.contains("Use bash for file operations like ls, rg, find"));
    }

    // --- skills index injection ---

    #[test]
    fn injects_skills_block_with_uris_and_no_real_paths() {
        let tmp = tempfile::TempDir::new().unwrap();
        let project_skills = tmp.path().join(".agents/skills");
        std::fs::create_dir_all(&project_skills).unwrap();
        write_skill(&project_skills, "fmt", "format code per house style");
        write_skill(&project_skills, "build", "build and test the workspace");
        let skills = SkillIndex::discover(Path::new("/nonexistent"), &project_skills);
        let tools = create_coding_tools(tmp.path(), std::sync::Arc::new(skills.clone()));
        let prompt = build_system_prompt(tmp.path(), &tools, &skills);
        assert!(prompt.contains("Matching skill → MUST read `skill://<name>` first."));
        assert!(prompt.contains("- fmt: format code per house style — skill://fmt"));
        assert!(prompt.contains("- build: build and test the workspace — skill://build"));
        let block = prompt
            .split("<skills>")
            .nth(1)
            .and_then(|s| s.split("</skills>").next())
            .unwrap();
        let tmp_str = tmp.path().display().to_string();
        assert!(
            !block.contains("SKILL.md"),
            "no path leakage in index: {block}"
        );
        assert!(
            !block.contains(&tmp_str),
            "no real path leakage in index: {block}"
        );
    }

    #[test]
    fn skills_block_only_when_read_tool_present() {
        let tmp = tempfile::TempDir::new().unwrap();
        let project_skills = tmp.path().join(".agents/skills");
        std::fs::create_dir_all(&project_skills).unwrap();
        write_skill(&project_skills, "fmt", "format code");
        let skills = SkillIndex::discover(Path::new("/nonexistent"), &project_skills);
        let no_read = vec![crate::create_bash_tool(tmp.path())];
        let prompt = build_system_prompt(tmp.path(), &no_read, &skills);
        assert!(!prompt.contains("<skills>"));
        assert!(!prompt.contains("skill://"));
    }

    #[test]
    fn empty_skills_index_omits_block() {
        let tmp = tempfile::TempDir::new().unwrap();
        let skills = SkillIndex::discover(Path::new("/nonexistent"), Path::new("/nonexistent"));
        let tools = create_coding_tools(tmp.path(), std::sync::Arc::new(skills.clone()));
        let prompt = build_system_prompt(tmp.path(), &tools, &skills);
        assert!(!prompt.contains("<skills>"));
        assert!(!prompt.contains("skill://"));
        // No-skills prompt must stay byte-identical to before skills existed:
        // one blank line between the last guideline and the cwd line.
        assert!(
            prompt.contains(
                "- Show file paths clearly when working with files\n\nCurrent working directory:"
            ),
            "blank line before cwd regressed: {prompt}"
        );
    }
}
