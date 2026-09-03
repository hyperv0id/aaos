//! `Message` ↔ `Segment` conversion — the agent↔store isomorphic projection.
//!
//! Field shapes are verified-identical (ADR-0002); the conversion moves
//! fields one by one (no serialization round-trip). One asymmetry:
//!
//! - **Write** (`&Message` → `Segment`) drops `timestamp`: segments don't
//!   carry it. The write path records the true write time in
//!   `entries.created_at` instead (ADR-0006), and the resume path restores
//!   it via `message_from_segment`. The standalone `TryFrom` impls have
//!   no store context and fall back to [`pi_agent_core::types::now`].
//!
//! Read direction fails on [`Segment::Summary`] — summaries are store-native
//! with no agent-side equivalent; the resume path renders them as a user
//! message instead.

use pi_agent_core::types::{
    self, AssistantMessage, ContentBlock as AgentContentBlock, Cost as AgentCost,
    ImageSource as AgentImageSource, Message, StopReason as AgentStopReason,
    ToolCall as AgentToolCall, ToolResultMessage, Usage as AgentUsage, UserMessage,
};

use crate::segment::{
    AssistantSegment, ContentBlock as StoreContentBlock, Cost as StoreCost,
    ImageSource as StoreImageSource, Segment, StopReason as StoreStopReason, SummarySegment,
    ToolCall as StoreToolCall, ToolResultSegment, Usage as StoreUsage, UserSegment,
};

/// Why a segment could not become a `Message`.
#[derive(Debug)]
pub enum ConvertError {
    /// Store-native summary segments have no agent-side message type.
    /// Carries the segment so callers can render its content.
    Summary(SummarySegment),
}

// ---------------------------------------------------------------------------
// Write direction: agent in-memory state → store segments.
// ---------------------------------------------------------------------------

impl From<&Message> for Segment {
    fn from(message: &Message) -> Self {
        match message {
            Message::User(m) => Segment::from(m),
            Message::Assistant(m) => Segment::from(m),
            Message::ToolResult(m) => Segment::from(m),
        }
    }
}

impl From<&UserMessage> for Segment {
    fn from(message: &UserMessage) -> Self {
        Segment::User(UserSegment {
            content: to_store_content(&message.content),
        })
    }
}

impl From<&AssistantMessage> for Segment {
    fn from(message: &AssistantMessage) -> Self {
        Segment::Assistant(AssistantSegment {
            content: to_store_content(&message.content),
            stop_reason: to_store_stop_reason(message.stop_reason),
            model: message.model.clone(),
            provider: message.provider.clone(),
            api: message.api.clone(),
            usage: to_store_usage(&message.usage),
            error_message: message.error_message.clone(),
        })
    }
}

impl From<&ToolResultMessage> for Segment {
    fn from(message: &ToolResultMessage) -> Self {
        Segment::ToolResult(ToolResultSegment {
            tool_call_id: message.tool_call_id.clone(),
            tool_name: message.tool_name.clone(),
            content: to_store_content(&message.content),
            details: message.details.clone(),
            usage: message.usage.as_ref().map(to_store_usage),
            added_tool_names: message.added_tool_names.clone(),
            is_error: message.is_error,
        })
    }
}

fn to_store_content(blocks: &[AgentContentBlock]) -> Vec<StoreContentBlock> {
    blocks
        .iter()
        .map(|block| match block {
            AgentContentBlock::Text { text } => StoreContentBlock::Text { text: text.clone() },
            AgentContentBlock::Image { source } => StoreContentBlock::Image {
                source: to_store_image_source(source),
            },
            AgentContentBlock::Thinking { text } => {
                StoreContentBlock::Thinking { text: text.clone() }
            }
            AgentContentBlock::ToolCall(call) => {
                StoreContentBlock::ToolCall(to_store_tool_call(call))
            }
        })
        .collect()
}

fn to_store_image_source(source: &AgentImageSource) -> StoreImageSource {
    StoreImageSource {
        mime_type: source.mime_type.clone(),
        bytes: source.bytes.clone(),
    }
}

fn to_store_tool_call(call: &AgentToolCall) -> StoreToolCall {
    StoreToolCall {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments: call.arguments.clone(),
    }
}

fn to_store_usage(usage: &AgentUsage) -> StoreUsage {
    StoreUsage {
        input: usage.input,
        output: usage.output,
        cache_read: usage.cache_read,
        cache_write: usage.cache_write,
        total_tokens: usage.total_tokens,
        cost: to_store_cost(usage.cost),
    }
}

fn to_store_cost(cost: AgentCost) -> StoreCost {
    StoreCost {
        input: cost.input,
        output: cost.output,
        cache_read: cost.cache_read,
        cache_write: cost.cache_write,
        total: cost.total,
    }
}

fn to_store_stop_reason(reason: AgentStopReason) -> StoreStopReason {
    match reason {
        AgentStopReason::Pending => StoreStopReason::Pending,
        AgentStopReason::Stop => StoreStopReason::Stop,
        AgentStopReason::Length => StoreStopReason::Length,
        AgentStopReason::ToolUse => StoreStopReason::ToolUse,
        AgentStopReason::Error => StoreStopReason::Error,
        AgentStopReason::Aborted => StoreStopReason::Aborted,
        AgentStopReason::Deferred => StoreStopReason::Deferred,
    }
}

// ---------------------------------------------------------------------------
// Read direction: store segments → agent in-memory state.
// ---------------------------------------------------------------------------

impl TryFrom<Segment> for Message {
    type Error = ConvertError;

    fn try_from(segment: Segment) -> Result<Self, Self::Error> {
        message_from_segment(segment, types::now())
    }
}

impl TryFrom<UserSegment> for Message {
    type Error = ConvertError;

    fn try_from(segment: UserSegment) -> Result<Self, Self::Error> {
        Ok(user_message_from(segment, types::now()))
    }
}

impl TryFrom<AssistantSegment> for Message {
    type Error = ConvertError;

    fn try_from(segment: AssistantSegment) -> Result<Self, Self::Error> {
        Ok(assistant_message_from(segment, types::now()))
    }
}

impl TryFrom<ToolResultSegment> for Message {
    type Error = ConvertError;

    fn try_from(segment: ToolResultSegment) -> Result<Self, Self::Error> {
        Ok(tool_result_message_from(segment, types::now()))
    }
}

/// Convert a stored segment stamped with its true write time
/// (`entries.created_at`, ADR-0006) — the resume path. The standalone
/// `TryFrom` impls above have no store context and fall back to `now()`.
pub(crate) fn message_from_segment(
    segment: Segment,
    timestamp: u64,
) -> Result<Message, ConvertError> {
    match segment {
        Segment::User(s) => Ok(user_message_from(s, timestamp)),
        Segment::Assistant(s) => Ok(assistant_message_from(s, timestamp)),
        Segment::ToolResult(s) => Ok(tool_result_message_from(s, timestamp)),
        Segment::Summary(s) => Err(ConvertError::Summary(s)),
    }
}

fn user_message_from(segment: UserSegment, timestamp: u64) -> Message {
    Message::User(UserMessage {
        content: from_store_content(segment.content),
        timestamp,
    })
}

fn assistant_message_from(segment: AssistantSegment, timestamp: u64) -> Message {
    Message::Assistant(AssistantMessage {
        content: from_store_content(segment.content),
        stop_reason: from_store_stop_reason(segment.stop_reason),
        model: segment.model,
        provider: segment.provider,
        api: segment.api,
        usage: from_store_usage(segment.usage),
        error_message: segment.error_message,
        timestamp,
    })
}

fn tool_result_message_from(segment: ToolResultSegment, timestamp: u64) -> Message {
    Message::ToolResult(ToolResultMessage {
        tool_call_id: segment.tool_call_id,
        tool_name: segment.tool_name,
        content: from_store_content(segment.content),
        details: segment.details,
        usage: segment.usage.map(from_store_usage),
        added_tool_names: segment.added_tool_names,
        is_error: segment.is_error,
        timestamp,
    })
}
fn from_store_content(blocks: Vec<StoreContentBlock>) -> Vec<AgentContentBlock> {
    blocks
        .into_iter()
        .map(|block| match block {
            StoreContentBlock::Text { text } => AgentContentBlock::Text { text },
            StoreContentBlock::Image { source } => AgentContentBlock::Image {
                source: from_store_image_source(source),
            },
            StoreContentBlock::Thinking { text } => AgentContentBlock::Thinking { text },
            StoreContentBlock::ToolCall(call) => {
                AgentContentBlock::ToolCall(from_store_tool_call(call))
            }
        })
        .collect()
}

fn from_store_image_source(source: StoreImageSource) -> AgentImageSource {
    AgentImageSource {
        mime_type: source.mime_type,
        bytes: source.bytes,
    }
}

fn from_store_tool_call(call: StoreToolCall) -> AgentToolCall {
    AgentToolCall {
        id: call.id,
        name: call.name,
        arguments: call.arguments,
    }
}

fn from_store_usage(usage: StoreUsage) -> AgentUsage {
    AgentUsage {
        input: usage.input,
        output: usage.output,
        cache_read: usage.cache_read,
        cache_write: usage.cache_write,
        total_tokens: usage.total_tokens,
        cost: from_store_cost(usage.cost),
    }
}

fn from_store_cost(cost: StoreCost) -> AgentCost {
    AgentCost {
        input: cost.input,
        output: cost.output,
        cache_read: cost.cache_read,
        cache_write: cost.cache_write,
        total: cost.total,
    }
}

fn from_store_stop_reason(reason: StoreStopReason) -> AgentStopReason {
    match reason {
        StoreStopReason::Pending => AgentStopReason::Pending,
        StoreStopReason::Stop => AgentStopReason::Stop,
        StoreStopReason::Length => AgentStopReason::Length,
        StoreStopReason::ToolUse => AgentStopReason::ToolUse,
        StoreStopReason::Error => AgentStopReason::Error,
        StoreStopReason::Aborted => AgentStopReason::Aborted,
        StoreStopReason::Deferred => AgentStopReason::Deferred,
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use serde_json::{Value, json};

    /// A fully-populated assistant message exercising every field.
    fn full_assistant() -> AssistantMessage {
        AssistantMessage {
            content: vec![
                AgentContentBlock::text("thinking through"),
                AgentContentBlock::Thinking {
                    text: "hidden reasoning".into(),
                },
                AgentContentBlock::tool_call("call-1", "read_file", json!({"path": "/tmp/x"})),
            ],
            stop_reason: AgentStopReason::ToolUse,
            model: "test-model".into(),
            provider: "test-provider".into(),
            api: "test-api".into(),
            usage: AgentUsage {
                input: 11,
                output: 22,
                cache_read: 33,
                cache_write: 44,
                total_tokens: 55,
                cost: AgentCost {
                    input: 0.1,
                    output: 0.2,
                    cache_read: 0.3,
                    cache_write: 0.4,
                    total: 1.0,
                },
            },
            error_message: None,
            timestamp: 12345,
        }
    }

    fn full_tool_result() -> ToolResultMessage {
        ToolResultMessage {
            tool_call_id: "call-1".into(),
            tool_name: "read_file".into(),
            content: vec![AgentContentBlock::text("file body")],
            details: json!({"bytes": 9}),
            usage: Some(AgentUsage::default()),
            added_tool_names: Some(vec!["extra_tool".into()]),
            is_error: false,
            timestamp: 67890,
        }
    }

    #[test]
    fn user_roundtrip_preserves_fields_except_timestamp() {
        let original = UserMessage {
            content: vec![
                AgentContentBlock::text("hello"),
                AgentContentBlock::text("world"),
            ],
            timestamp: 42,
        };
        let restored = Message::try_from(Segment::from(&original)).unwrap();
        assert_eq!(
            restored,
            Message::User(UserMessage {
                content: original.content.clone(),
                timestamp: restored.as_user().unwrap().timestamp,
            }),
            "content survives; only timestamp may differ"
        );
        assert_ne!(restored.as_user().unwrap().timestamp, 42);
    }

    #[test]
    fn assistant_roundtrip_preserves_all_fields_except_timestamp() {
        let original = full_assistant();
        let restored = Message::try_from(Segment::from(&original)).unwrap();
        let Message::Assistant(restored) = &restored else {
            panic!("expected assistant message");
        };
        assert_eq!(restored.content, original.content);
        assert_eq!(restored.stop_reason, original.stop_reason);
        assert_eq!(restored.model, original.model);
        assert_eq!(restored.provider, original.provider);
        assert_eq!(restored.api, original.api);
        assert_eq!(restored.usage, original.usage);
        assert_eq!(restored.error_message, original.error_message);
        assert_ne!(restored.timestamp, original.timestamp);
    }

    #[test]
    fn assistant_error_message_survives_roundtrip() {
        let original = AssistantMessage {
            error_message: Some("boom".into()),
            stop_reason: AgentStopReason::Error,
            ..full_assistant()
        };
        let restored = Message::try_from(Segment::from(&original)).unwrap();
        assert_eq!(
            restored.as_assistant().unwrap().error_message,
            Some("boom".into())
        );
    }

    #[test]
    fn tool_result_roundtrip_preserves_all_fields_except_timestamp() {
        let original = full_tool_result();
        let restored = Message::try_from(Segment::from(&original)).unwrap();
        let Message::ToolResult(restored) = &restored else {
            panic!("expected tool result message");
        };
        assert_eq!(restored.tool_call_id, original.tool_call_id);
        assert_eq!(restored.tool_name, original.tool_name);
        assert_eq!(restored.content, original.content);
        assert_eq!(restored.details, original.details);
        assert_eq!(restored.usage, original.usage);
        assert_eq!(restored.added_tool_names, original.added_tool_names);
        assert_eq!(restored.is_error, original.is_error);
        assert_ne!(restored.timestamp, original.timestamp);
    }

    #[test]
    fn summary_segment_fails_conversion() {
        let segment = Segment::summary("the gist");
        let err = Message::try_from(segment).unwrap_err();
        let ConvertError::Summary(summary) = err;
        assert_eq!(summary.content, "the gist");
    }

    #[test]
    fn image_blocks_roundtrip_byte_exact() {
        let original = UserMessage {
            content: vec![AgentContentBlock::Image {
                source: AgentImageSource {
                    mime_type: "image/png".into(),
                    bytes: vec![0x89, 0x50, 0x4e, 0x47],
                },
            }],
            timestamp: 7,
        };
        let restored = Message::try_from(Segment::from(&original)).unwrap();
        assert_eq!(restored.as_user().unwrap().content, original.content);
    }

    #[test]
    fn stop_reason_roundtrips_all_variants() {
        for reason in [
            AgentStopReason::Pending,
            AgentStopReason::Stop,
            AgentStopReason::Length,
            AgentStopReason::ToolUse,
            AgentStopReason::Error,
            AgentStopReason::Aborted,
            AgentStopReason::Deferred,
        ] {
            let message = AssistantMessage {
                stop_reason: reason,
                ..full_assistant()
            };
            let restored = Message::try_from(Segment::from(&message)).unwrap();
            assert_eq!(restored.as_assistant().unwrap().stop_reason, reason);
        }
    }

    #[test]
    fn null_details_survive_roundtrip() {
        let original = ToolResultMessage {
            details: Value::Null,
            ..full_tool_result()
        };
        let restored = Message::try_from(Segment::from(&original)).unwrap();
        assert_eq!(restored.as_tool_result().unwrap().details, Value::Null);
    }
}
