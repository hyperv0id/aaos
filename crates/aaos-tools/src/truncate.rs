pub const MAX_LINES: usize = 2000;
pub const MAX_BYTES: usize = 50 * 1024;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Truncation {
    pub content: String,
    /// True when either the line or byte cap cut the input.
    pub truncated: bool,
}

/// Keep the head of `text`: at most `MAX_LINES` lines and `MAX_BYTES` UTF-8
/// bytes. Stops as soon as either cap is hit; the byte cap never splits a
/// multi-byte character.
pub fn truncate_head(text: &str) -> Truncation {
    let mut out = String::new();
    let mut truncated = false;
    let mut lines = text.split('\n');

    for _ in 0..MAX_LINES {
        let Some(line) = lines.next() else { break };
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
        if out.len() > MAX_BYTES {
            // Trim to the byte cap without splitting a UTF-8 char.
            let mut end = MAX_BYTES;
            while !out.is_char_boundary(end) {
                end -= 1;
            }
            out.truncate(end);
            truncated = true;
            break;
        }
    }
    if lines.next().is_some() {
        truncated = true;
    }
    Truncation {
        content: out,
        truncated,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_not_truncated() {
        let truncation = truncate_head("a\nb");
        assert!(!truncation.truncated);
        assert_eq!(truncation.content, "a\nb");
    }

    #[test]
    fn line_limit_truncates() {
        let input = (0..MAX_LINES + 5)
            .map(|i| format!("L{i}"))
            .collect::<Vec<_>>()
            .join("\n");
        let truncation = truncate_head(&input);
        assert!(truncation.truncated);
        assert_eq!(truncation.content.lines().count(), MAX_LINES);
    }

    #[test]
    fn byte_cap_never_splits_utf8() {
        // Single line of multibyte chars exceeding the byte cap.
        let big: String = "é".repeat(MAX_BYTES + 100);
        let truncation = truncate_head(&big);
        assert!(truncation.truncated);
        assert!(truncation.content.len() <= MAX_BYTES);
        // Content must remain valid UTF-8 (it is a String, but assert boundary invariant).
        assert!(truncation
            .content
            .is_char_boundary(truncation.content.len()));
    }

    #[test]
    fn exact_byte_cap_is_not_truncated() {
        let exact: String = "a".repeat(MAX_BYTES);
        let truncation = truncate_head(&exact);
        assert!(!truncation.truncated);
        assert_eq!(truncation.content.len(), MAX_BYTES);
    }

    #[test]
    fn empty_input_is_not_truncated() {
        let truncation = truncate_head("");
        assert!(!truncation.truncated);
        assert_eq!(truncation.content, "");
    }
}
