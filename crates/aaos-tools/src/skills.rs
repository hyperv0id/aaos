//! Skill discovery, frontmatter parsing, prompt index, and `skill://` resolution.
//!
//! Skills are single-level discoveries under the user (`~/.agents/skills/`) and
//! project (`.agents/skills/`) roots: an immediate child directory
//! containing `SKILL.md` is one skill (nested directories are not
//! discovered). A project-level skill overrides the user-level skill with the
//! same name. SKILL.md frontmatter: `name` defaults to the directory name;
//! `description` is required — a skill without one is skipped. Other
//! frontmatter fields are ignored.
//!
//! The system prompt only ever receives the pure index (name, description,
//! `skill://` URI — no real paths); skill bodies and bundled resources are read
//! on demand via the `read` tool, which annotates results with the resolved real
//! path.

use std::path::{Path, PathBuf};

/// A single discovered skill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    /// Directory that backs the skill (contains `SKILL.md`).
    pub dir: PathBuf,
}

/// Immutable skill index, discovered once at startup.
///
/// Project-level skills override user-level skills of the same name; the index
/// is sorted by name for stable prompt output.
#[derive(Debug, Clone)]
pub struct SkillIndex {
    skills: Vec<Skill>,
}

impl SkillIndex {
    /// Discover skills under the given user-level and project-level roots.
    ///
    /// Missing roots are skipped; entries that are not directories (or that
    /// lack `SKILL.md`) are ignored. Project-level skills win over same-named
    /// user-level skills. The result is sorted by `name`.
    pub fn discover(user_skills_dir: &Path, project_skills_dir: &Path) -> SkillIndex {
        let mut by_name: std::collections::HashMap<String, Skill> =
            std::collections::HashMap::new();
        for root in [user_skills_dir, project_skills_dir] {
            let Ok(entries) = std::fs::read_dir(root) else {
                continue; // Missing root → skipped.
            };
            for entry in entries.flatten() {
                let dir = entry.path();
                // Single-level discovery: only immediate child directories.
                let Ok(file_type) = entry.file_type() else {
                    continue;
                };
                if !file_type.is_dir() {
                    continue;
                }
                let Some((name, description)) = load_skill_meta(&dir) else {
                    continue; // No SKILL.md or no description → not a skill.
                };
                // Project level wins: later root overrides earlier.
                by_name.insert(
                    name.clone(),
                    Skill {
                        name,
                        description,
                        dir,
                    },
                );
            }
        }
        let mut skills: Vec<Skill> = by_name.into_values().collect();
        skills.sort_by(|a, b| a.name.cmp(&b.name));
        Self { skills }
    }

    pub fn is_empty(&self) -> bool {
        self.skills.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.skills.iter().map(|s| s.name.as_str())
    }

    pub fn get(&self, name: &str) -> Option<&Skill> {
        self.skills.iter().find(|s| s.name == name)
    }

    /// Render the `<skills>` index block for the system prompt.
    ///
    /// Contains an instruction sentence, one `- name: description` line per
    /// skill (each carrying its `skill://` URI), and no real filesystem paths.
    pub fn prompt_block(&self) -> String {
        if self.skills.is_empty() {
            return String::new();
        }
        let mut block =
            String::from("Matching skill → MUST read `skill://<name>` first.\n<skills>\n");
        for skill in &self.skills {
            block.push_str(&format!(
                "- {}: {} — skill://{}\n",
                skill.name, skill.description, skill.name
            ));
        }
        block.push_str("</skills>");
        block
    }
}

/// Read and parse `dir/SKILL.md`; returns `(name, description)`.
///
/// `name` defaults to the directory name when the frontmatter omits it (or
/// leaves it blank); `description` is required — a missing or empty
/// description makes the directory not a skill. Unknown frontmatter fields
/// are ignored. Returns `None` when SKILL.md is absent or unreadable.
fn load_skill_meta(dir: &Path) -> Option<(String, String)> {
    let content = std::fs::read_to_string(dir.join("SKILL.md")).ok()?;
    let (name, description) = parse_skill_frontmatter(&content)?;
    let name = match name {
        Some(name) => name,
        None => dir.file_name()?.to_string_lossy().into_owned(),
    };
    let description = description?;
    Some((name, description))
}

/// Line-based SKILL.md frontmatter parsing, per the pi convention: a leading
/// `---` line, `key: value` lines, closed by a `---` line. Only `name` and
/// `description` are read; everything else is ignored.
fn parse_skill_frontmatter(content: &str) -> Option<(Option<String>, Option<String>)> {
    let mut lines = content.lines();
    if lines.next()? != "---" {
        return None;
    }
    let mut name = None;
    let mut description = None;
    for line in lines {
        if line == "---" {
            return Some((name, description));
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        match key.trim() {
            "name" if !value.is_empty() => name = Some(value.to_string()),
            "description" if !value.is_empty() => description = Some(value.to_string()),
            _ => {} // Other fields are ignored.
        }
    }
    // Unclosed frontmatter: treat as no frontmatter.
    None
}

/// Resolve a `skill://` URL against the index, returning the canonicalized
/// target path (real path, inside the skill directory).
///
/// Mirrors oh-my-pi's `skill://` semantics: `skill://name` and `skill://name/`
/// read `SKILL.md`; `skill://name/path` reads a file (or lists a
/// directory) inside the skill directory. Unknown skills error and list the
/// available names. Absolute paths, `..`, and symlink escapes are rejected;
/// a canonicalized target must land inside the canonicalized skill directory.
pub async fn resolve_skill_url(
    skills: &SkillIndex,
    host: &str,
    url_path: &str,
) -> Result<PathBuf, String> {
    if host.is_empty() {
        return Err("skill:// URL requires a skill name: skill://<name>".to_string());
    }
    let skill = skills
        .get(host)
        .ok_or_else(|| unknown_skill_error(skills, host))?;
    let relative = if url_path.is_empty() || url_path == "/" {
        "SKILL.md".to_string()
    } else {
        let decoded = percent_decode(&url_path[1..]).map_err(|e| format!("{e} in skill:// URL"))?;
        validate_relative_path(&decoded)?;
        decoded
    };
    let target = skill.dir.join(&relative);

    let skill_dir = tokio::fs::canonicalize(&skill.dir)
        .await
        .map_err(|_| format!("File not found: {}", target.display()))?;
    let path = tokio::fs::canonicalize(&target)
        .await
        .map_err(|_| format!("File not found: {}", target.display()))?;
    if path != skill_dir && !path.starts_with(&skill_dir) {
        return Err(format!(
            "skill:// path resolves outside the skill directory: skill://{host}{url_path}"
        ));
    }
    Ok(path)
}

fn unknown_skill_error(skills: &SkillIndex, name: &str) -> String {
    let names = skills.names().collect::<Vec<_>>().join(", ");
    let available = if names.is_empty() { "none" } else { &names };
    format!("Unknown skill: {name}\nAvailable: {available}")
}

fn hex_val(b: u8) -> u8 {
    match b {
        b'0'..=b'9' => b - b'0',
        b'a'..=b'f' => b - b'a' + 10,
        b'A'..=b'F' => b - b'A' + 10,
        _ => unreachable!("hexdigit checked by caller"),
    }
}

/// Percent-decode a path segment sequence (`%XX` escapes; `+` stays literal,
/// since it is only a space in query strings). Errors on malformed escapes or
/// non-UTF-8 bytes.
fn percent_decode(input: &str) -> Result<String, String> {
    let bytes = input.as_bytes();
    let mut decoded_bytes = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' {
            if i + 2 >= bytes.len()
                || !bytes[i + 1].is_ascii_hexdigit()
                || !bytes[i + 2].is_ascii_hexdigit()
            {
                return Err("invalid percent-encoding".to_string());
            }
            let hi = hex_val(bytes[i + 1]);
            let lo = hex_val(bytes[i + 2]);
            decoded_bytes.push((hi << 4) | lo);
            i += 3;
        } else {
            decoded_bytes.push(bytes[i]);
            i += 1;
        }
    }
    String::from_utf8(decoded_bytes).map_err(|_| "non-UTF-8 path".to_string())
}

/// Validate a percent-decoded skill-relative path: reject absolute paths and
/// `..` segments (checking both `/` and `\` separators, per oh-my-pi parity).
fn validate_relative_path(decoded: &str) -> Result<(), String> {
    if std::path::Path::new(decoded).is_absolute() {
        return Err("Absolute paths are not allowed in skill:// URLs".to_string());
    }
    if decoded.split(['/', '\\']).any(|seg| seg == "..") {
        return Err("Path traversal (..) is not allowed in skill:// URLs".to_string());
    }
    Ok(())
}

/// Split a path argument into `(scheme, rest)` when it carries a URI scheme
/// (`scheme://…`, case-insensitive, per the oh-my-pi parse semantics).
/// Filesystem paths without a scheme yield `None`.
pub fn split_uri_scheme(input: &str) -> Option<(String, &str)> {
    // `^([a-z][a-z0-9+.-]*):\/\/([^/?#]*)` per omp's parse.ts; the scheme
    // is lowercased for matching.
    let colon = input.find("://")?;
    let scheme = &input[..colon];
    let valid = !scheme.is_empty()
        && scheme.chars().enumerate().all(|(i, c)| {
            (i == 0 && c.is_ascii_alphabetic())
                || c.is_ascii_alphanumeric()
                || c == '+'
                || c == '.'
                || c == '-'
        });
    if !valid {
        return None;
    }
    Some((scheme.to_ascii_lowercase(), &input[colon + 3..]))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn write_skill(dir: &Path, name: &str, frontmatter: &str) -> PathBuf {
        let skill_dir = dir.join(name);
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(skill_dir.join("SKILL.md"), frontmatter).unwrap();
        skill_dir
    }

    // --- discovery + frontmatter ---

    #[test]
    fn discovers_user_and_project_skills_two_levels() {
        let tmp = TempDir::new().unwrap();
        let user = tmp.path().join("user/.agents/skills");
        let project = tmp.path().join("project/.agents/skills");
        fs::create_dir_all(&user).unwrap();
        fs::create_dir_all(&project).unwrap();
        write_skill(
            &user,
            "user-skill",
            "---\nname: user-skill\ndescription: user-level skill\n---\nbody\n",
        );
        write_skill(
            &project,
            "project-skill",
            "---\ndescription: project-level skill\n---\nbody\n",
        );
        let index = SkillIndex::discover(&user, &project);
        assert_eq!(
            index.names().collect::<Vec<_>>(),
            ["project-skill", "user-skill"]
        );
        assert_eq!(
            index.get("user-skill").unwrap().description,
            "user-level skill"
        );
        assert_eq!(
            index.get("project-skill").unwrap().description,
            "project-level skill"
        );
    }

    #[test]
    fn project_level_overrides_user_level_same_name() {
        let tmp = TempDir::new().unwrap();
        let user = tmp.path().join("user/.agents/skills");
        let project = tmp.path().join("project/.agents/skills");
        fs::create_dir_all(&user).unwrap();
        fs::create_dir_all(&project).unwrap();
        write_skill(&user, "foo", "---\ndescription: user version\n---\n");
        write_skill(&project, "foo", "---\ndescription: project version\n---\n");
        let index = SkillIndex::discover(&user, &project);
        assert_eq!(index.names().collect::<Vec<_>>(), ["foo"]);
        assert_eq!(index.get("foo").unwrap().description, "project version");
        assert_eq!(index.get("foo").unwrap().dir, project.join("foo"));
    }

    #[test]
    fn nested_skills_dirs_are_not_discovered() {
        let tmp = TempDir::new().unwrap();
        let user = tmp.path().join("user/.agents/skills");
        fs::create_dir_all(&user).unwrap();
        fs::create_dir_all(user.join("outer").join("inner")).unwrap();
        fs::write(
            user.join("outer/inner/SKILL.md"),
            "---\ndescription: nested\n---\n",
        )
        .unwrap();
        let index = SkillIndex::discover(&user, Path::new("/nonexistent"));
        assert!(index.is_empty());
    }

    #[test]
    fn missing_dirs_and_dir_without_skill_md_are_ignored() {
        let tmp = TempDir::new().unwrap();
        let user = tmp.path().join("user/.agents/skills");
        fs::create_dir_all(&user).unwrap();
        fs::create_dir_all(user.join("no-md")).unwrap();
        fs::write(user.join("random.txt"), "not a skill").unwrap();
        let index = SkillIndex::discover(&user, Path::new("/nonexistent"));
        assert!(index.is_empty());
    }

    #[test]
    fn frontmatter_name_defaults_to_dir_name_and_unknown_fields_ignored() {
        let tmp = TempDir::new().unwrap();
        let user = tmp.path().join("user/.agents/skills");
        fs::create_dir_all(&user).unwrap();
        write_skill(
            &user,
            "dir-name",
            "---\ndescription: from frontmatter\nother: ignored\n---\nbody\n",
        );
        let index = SkillIndex::discover(&user, Path::new("/nonexistent"));
        assert_eq!(
            index.get("dir-name").unwrap().description,
            "from frontmatter"
        );
    }

    #[test]
    fn frontmatter_name_overrides_dir_name() {
        let tmp = TempDir::new().unwrap();
        let user = tmp.path().join("user/.agents/skills");
        fs::create_dir_all(&user).unwrap();
        write_skill(
            &user,
            "dir-name",
            "---\nname: renamed\ndescription: d\n---\n",
        );
        let index = SkillIndex::discover(&user, Path::new("/nonexistent"));
        assert_eq!(index.names().collect::<Vec<_>>(), ["renamed"]);
        assert_eq!(index.get("renamed").unwrap().dir, user.join("dir-name"));
    }

    #[test]
    fn missing_or_empty_description_skips_skill() {
        let tmp = TempDir::new().unwrap();
        let user = tmp.path().join("user/.agents/skills");
        fs::create_dir_all(&user).unwrap();
        write_skill(&user, "no-desc", "---\nname: no-desc\n---\n");
        write_skill(&user, "empty-desc", "---\ndescription:\n---\n");
        write_skill(&user, "good", "---\ndescription: ok\n---\n");
        write_skill(&user, "no-frontmatter", "plain body, no frontmatter\n");
        let index = SkillIndex::discover(&user, Path::new("/nonexistent"));
        assert_eq!(index.names().collect::<Vec<_>>(), ["good"]);
    }

    // --- index block ---

    #[test]
    fn prompt_block_lists_skills_with_uris_and_no_real_paths() {
        let tmp = TempDir::new().unwrap();
        let user = tmp.path().join("user/.agents/skills");
        fs::create_dir_all(&user).unwrap();
        write_skill(&user, "aa", "---\ndescription: first skill\n---\n");
        write_skill(&user, "bb", "---\ndescription: second skill\n---\n");
        let index = SkillIndex::discover(&user, Path::new("/nonexistent"));
        let block = index.prompt_block();
        let tmp_str = tmp.path().display().to_string();
        assert!(
            block.contains("Matching skill → MUST read `skill://<name>` first."),
            "{block}"
        );
        assert_eq!(
            block,
            "Matching skill → MUST read `skill://<name>` first.\n<skills>\n- aa: first skill — skill://aa\n- bb: second skill — skill://bb\n</skills>"
        );
        assert!(
            !block.contains("SKILL.md"),
            "index must not leak paths: {block}"
        );
        assert!(
            !block.contains(&tmp_str),
            "index must not leak real paths: {block}"
        );
    }

    #[test]
    fn empty_index_has_no_prompt_block() {
        let index = SkillIndex::discover(Path::new("/nonexistent"), Path::new("/nonexistent"));
        assert_eq!(index.prompt_block(), "");
    }

    // --- URI splitting ---

    #[test]
    fn split_uri_scheme_recognizes_case_insensitive_skill() {
        assert_eq!(
            split_uri_scheme("skill://foo"),
            Some(("skill".to_string(), "foo"))
        );
        assert_eq!(
            split_uri_scheme("SKILL://foo/bar"),
            Some(("skill".to_string(), "foo/bar"))
        );
        assert_eq!(split_uri_scheme("src/read.rs"), None);
        assert_eq!(split_uri_scheme("C:\\work\\file.txt"), None);
        assert_eq!(
            split_uri_scheme("other://x"),
            Some(("other".to_string(), "x"))
        );
    }
}
