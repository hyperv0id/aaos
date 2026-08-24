//! Segment wire types — the store's own content model.
//!
//! Field shapes mirror `pi_agent_core::types` message types (minus
//! timestamps, which live on log records so objects stay dedupable —
//! git's blob-vs-commit split), so a later bridge crate only moves
//! fields. `Summary` is store-native: it carries the hashes of the
//! segments it replaces (provenance).

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Minimal semantic unit of a conversation (content layer, content-addressed).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Segment {
    User(UserSegment),
    Assistant(AssistantSegment),
    ToolResult(ToolResultSegment),
    Summary(SummarySegment),
}

impl Segment {
    pub fn kind(&self) -> &'static str {
        match self {
            Segment::User(_) => "user",
            Segment::Assistant(_) => "assistant",
            Segment::ToolResult(_) => "tool_result",
            Segment::Summary(_) => "summary",
        }
    }

    pub fn user_text(text: impl Into<String>) -> Self {
        Segment::User(UserSegment {
            content: vec![ContentBlock::Text { text: text.into() }],
        })
    }

    pub fn assistant_text(text: impl Into<String>) -> Self {
        Segment::Assistant(AssistantSegment {
            content: vec![ContentBlock::Text { text: text.into() }],
            stop_reason: StopReason::Stop,
            model: "unknown".into(),
            provider: "unknown".into(),
            api: "unknown".into(),
            usage: Usage::default(),
            error_message: None,
        })
    }

    pub fn tool_result_text(tool_call_id: impl Into<String>, text: impl Into<String>) -> Self {
        Segment::ToolResult(ToolResultSegment {
            tool_call_id: tool_call_id.into(),
            tool_name: "unknown".into(),
            content: vec![ContentBlock::Text { text: text.into() }],
            details: Value::Null,
            usage: None,
            added_tool_names: None,
            is_error: false,
        })
    }

    pub fn summary(content: impl Into<String>, sources: Vec<String>) -> Self {
        Segment::Summary(SummarySegment::new(content, sources))
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UserSegment {
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AssistantSegment {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub model: String,
    pub provider: String,
    pub api: String,
    pub usage: Usage,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolResultSegment {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    pub details: Value,
    pub usage: Option<Usage>,
    pub added_tool_names: Option<Vec<String>>,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SummarySegment {
    pub content: String,
    /// Hashes of the summarized segments — provenance for compaction.
    pub sources: Vec<String>,
    /// Model that generated the summary (`None` for hand-written ones).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

impl SummarySegment {
    pub fn new(content: impl Into<String>, sources: Vec<String>) -> Self {
        Self {
            content: content.into(),
            sources,
            model: None,
        }
    }

    pub fn generated_by(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Text { text: String },
    Image { source: ImageSource },
    Thinking { text: String },
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageSource {
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    Pending,
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_tokens: u64,
    pub cost: Cost,
}

#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::object_store::{canonical_bytes, hash_hex};

    /// Golden vector: pins the canonical encoding and the content hash of a
    /// fixed segment. The segment carries a `details: Value::Object` whose
    /// keys are inserted in non-sorted order (`z` before `a`); the canonical
    /// bytes and hash only stay stable while serde_json serializes object
    /// keys in sorted order (the default, BTreeMap-backed `Value::Object`).
    /// Enabling serde_json's `preserve_order` feature anywhere in the
    /// dependency graph switches `Value::Object` to insertion-order
    /// `IndexMap`, which would change both the bytes and the hash — this
    /// test fails in that case. See `object_store.rs` for the full invariant.
    #[test]
    fn canonical_bytes_and_hash_are_pinned() {
        let seg = Segment::ToolResult(ToolResultSegment {
            tool_call_id: "call-1".into(),
            tool_name: "unknown".into(),
            content: vec![ContentBlock::Text { text: "42".into() }],
            details: serde_json::json!({"z": 1, "a": 2}),
            usage: None,
            added_tool_names: None,
            is_error: false,
        });
        let bytes = canonical_bytes(&seg).unwrap();
        let want_bytes = br#"{"type":"tool_result","tool_call_id":"call-1","tool_name":"unknown","content":[{"type":"text","text":"42"}],"details":{"a":2,"z":1},"usage":null,"added_tool_names":null,"is_error":false}"#;
        assert_eq!(
            bytes, want_bytes,
            "canonical encoding drifted — serde_json key ordering changed?"
        );
        let hash = hash_hex(&bytes);
        assert_eq!(
            hash, "58e12efde4108fdbb1c9a4dd879667fd9c6f88997a53422b7ba58ffd626e0116",
            "content hash drifted — canonical encoding changed?"
        );
    }

    #[test]
    fn hash_is_64_lowercase_hex() {
        let hash = hash_hex(&canonical_bytes(&Segment::assistant_text("hi")).unwrap());
        assert_eq!(hash.len(), 64);
        assert!(
            hash.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        );
    }

    #[test]
    fn kind_distinguishes_variants() {
        assert_eq!(Segment::user_text("x").kind(), "user");
        assert_eq!(Segment::assistant_text("x").kind(), "assistant");
        assert_eq!(Segment::tool_result_text("c1", "x").kind(), "tool_result");
        assert_eq!(Segment::summary("s", vec![]).kind(), "summary");
    }

    #[test]
    fn serde_roundtrip_all_variants() {
        let segs = vec![
            Segment::user_text("q"),
            Segment::assistant_text("a"),
            Segment::tool_result_text("c1", "r"),
            Segment::summary("s", vec!["a".repeat(64)]),
            Segment::Summary(SummarySegment::new("llm summary", vec![]).generated_by("gpt-x")),
        ];
        for seg in segs {
            let bytes = canonical_bytes(&seg).unwrap();
            let back: Segment = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(back, seg);
        }
    }

    #[test]
    fn summary_model_field_is_absent_when_none() {
        let bytes = canonical_bytes(&Segment::summary("s", vec![])).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(
            !text.contains("model"),
            "manual summary omits model: {text}"
        );
    }

    #[test]
    fn wire_shape_is_internally_tagged() {
        let bytes = canonical_bytes(&Segment::user_text("q")).unwrap();
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["type"], "user");
        assert_eq!(value["content"][0]["type"], "text");
    }
}
