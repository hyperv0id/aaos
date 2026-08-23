//! Default coding-tool assembly and the system prompt they are described with.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pi_agent_core::types::AgentTool;

use crate::mutation::FileMutationQueue;
use crate::write::create_write_tool;
use crate::{create_bash_tool, create_edit_tool, create_read_tool};

/// Assemble the default coding tools in Pi order, sharing one mutation queue
/// between `edit` and `write`.
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
/// The bash-for-file-ops guideline is included only when `bash` is present
/// and no tool is named `grep`, `find`, or `ls`.
pub fn build_system_prompt(cwd: &Path, tools: &[Arc<dyn AgentTool>]) -> String {
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
    if has_tool("bash") && !has_tool("grep") && !has_tool("find") && !has_tool("ls") {
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

    format!(
        "You are an expert coding assistant. You help users by reading files, executing commands, editing code, and writing new files.\n\
         \n\
         Available tools:\n\
         {tools_list}\n\
         \n\
         In addition to the tools above, you may have access to other custom tools depending on the project.\n\
         \n\
         Guidelines:\n\
         {guidelines}\n\
         \n\
         Current working directory: {cwd}"
    )
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::path::Path;

    use crate::{build_system_prompt, create_coding_tools, create_read_tool};

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
        assert!(prompt.contains("Use bash for file operations like ls, rg, find"));
        let prompt_win = build_system_prompt(Path::new("C:\\tmp\\work"), &tools);
        assert!(prompt_win.contains("Current working directory: C:/tmp/work"));
        assert!(!prompt_win.contains("C:\\tmp\\work"));
    }

    #[test]
    fn omits_missing_tool_lines() {
        let tools = vec![create_read_tool("/tmp")];
        let prompt = build_system_prompt(Path::new("/tmp"), &tools);
        assert!(prompt.contains("- read:"));
        assert!(!prompt.contains("- bash:"));
        assert!(!prompt.contains("Use bash for file operations like ls, rg, find"));
    }
}
