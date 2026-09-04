//! Segment wire types — the store's in-memory content model.
//!
//! Field shapes mirror `pi_agent_core::types` message types (minus
//! timestamps: the write path records the true write time in
//! `entries.created_at`, and the resume path restores it — see
//! `convert.rs`). Segments are the memory currency only: on disk (ADR-0006)
//! a segment is decomposed into per-block raw-content objects with the
//! structure and metadata in the DB, so no serialization of the whole
//! segment survives. `Summary` is store-native: it replaces a range of the
//! parent view at compaction, and provenance is structural
//! (`compactions` ranges / `fetch_originals`) — the segment itself carries
//! no sources.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Minimal semantic unit of a conversation (content layer, content-addressed).
#[derive(Debug, Clone, PartialEq)]
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

    pub fn summary(content: impl Into<String>) -> Self {
        Segment::Summary(SummarySegment::new(content))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UserSegment {
    pub content: Vec<ContentBlock>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantSegment {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub model: String,
    pub provider: String,
    pub api: String,
    pub usage: Usage,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResultSegment {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    pub details: Value,
    pub usage: Option<Usage>,
    pub added_tool_names: Option<Vec<String>>,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SummarySegment {
    pub content: String,
}

impl SummarySegment {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            content: content.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text { text: String },
    Image { source: ImageSource },
    Thinking { text: String },
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageSource {
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq)]
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
    use crate::object_store::hash_hex;

    #[test]
    fn hash_is_64_lowercase_hex() {
        let hash = hash_hex(b"hi");
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
        assert_eq!(Segment::summary("s").kind(), "summary");
    }

    #[test]
    fn summary_carries_only_content() {
        let Segment::Summary(s) = Segment::summary("the gist") else {
            panic!("expected summary segment");
        };
        assert_eq!(s.content, "the gist");
    }
}
