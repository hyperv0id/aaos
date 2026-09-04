//! Pure, testable compaction logic: token estimation, cut-point selection,
//! and the deterministic transcript builder for compacted segments.
//!
//! Aligns with pi `packages/coding-agent/src/core/compaction/compaction.ts`:
//! - usage-anchored context tokens (compaction.ts:183-214) with local
//!   chars/4 estimation as fallback (compaction.ts:280-350);
//! - reserve-based trigger threshold (compaction.ts:235-249);
//! - tail-accumulating cut points that never split a `tool_call`/`tool_result`
//!   pair (compaction.ts:373-480).
//!
//! The transcript builder is deterministic: it renders the compacted
//! segments inline (user/assistant dialogue text, short tool calls) and
//! points at the content-addressed object paths for long tool arguments,
//! images, and all tool results. Thinking blocks and tool-result details
//! are process, not content: they are dropped, not unloaded — originals
//! stay forensically retrievable via `fetch_originals`. No model
//! involvement — compaction never calls an LLM.
//!
//! Object paths are block-granular (ADR-0006): every content block is its
//! own object — raw text bytes, image payload, canonical JSON — so the
//! builder recomputes a block's hash from the segment itself
//! (`db::block_bytes` + `object_store::hash_hex`) instead
//! of reading `entry_blocks`. Content addressing makes the recomputed hash
//! exact for any segment that came out of the store (encode∘decode is the
//! identity on bytes), which is the caller's contract: pass materialized
//! segments and the referenced paths exist.

use pi_agent_core::types::{AssistantMessage, ContentBlock, Message, StopReason, UserMessage};

use crate::convert::ConvertError;
use crate::db::{block_bytes, canonical_json};
use crate::object_store::{ObjectStore, hash_hex};
use crate::segment::Segment;

/// The conversation is summarized when its context tokens exceed
/// `context_window - reserve_tokens`. pi `CompactionSettings.reserveTokens`
/// default (compaction.ts:148-161).
pub const DEFAULT_RESERVE_TOKENS: u64 = 16_384;

/// Token budget for the retained recent tail when choosing a cut point. pi
/// `CompactionSettings.keepRecentTokens` default.
pub const DEFAULT_KEEP_RECENT_TOKENS: u64 = 20_000;

/// Estimated characters for one image block — pi `ESTIMATED_IMAGE_CHARS`
/// (compaction.ts:269), ≈ 1200 tokens at chars per token.
const ESTIMATED_IMAGE_CHARS: u64 = 4_800;

/// Estimated characters per token in the local fallback estimation.
const CHARS_PER_TOKEN: u64 = 4;

/// Serialized tool-call arguments of at most this many *characters* are
/// inlined into the transcript; longer ones are truncated to a preview of
/// this length and the transcript points at the object path instead.
/// Character-granular, not bytes: canonical JSON may contain multibyte
/// UTF-8.
const INLINE_ARG_MAX_CHARS: usize = 100;

/// First line of the transcript: explains the format and that full outputs
/// live at the referenced paths.
pub const TRANSCRIPT_PREAMBLE: &str = "The conversation history before this point was compacted into this transcript. Full tool outputs and arguments are at the referenced absolute paths and can be read with the read tool.";

/// Local token estimate of one message: text chars/4, images at
/// `ESTIMATED_IMAGE_CHARS` chars, assistant thinking and tool-call
/// name+arguments included, tool results counted by their content text.
/// Conservative — pi `estimateTokens` (compaction.ts:280-350).
pub fn estimate_tokens(msg: &Message) -> u64 {
    match msg {
        Message::User(user) => estimate_content_tokens(&user.content),
        Message::Assistant(assistant) => {
            let mut chars: u64 = 0;
            for block in &assistant.content {
                match block {
                    ContentBlock::Text { text } => chars += text.len() as u64,
                    ContentBlock::Thinking { text } => chars += text.len() as u64,
                    ContentBlock::Image { .. } => chars += ESTIMATED_IMAGE_CHARS,
                    ContentBlock::ToolCall(call) => {
                        chars += call.name.len() as u64
                            + serde_json::to_string(&call.arguments)
                                .unwrap_or_default()
                                .len() as u64
                    }
                }
            }
            chars.div_ceil(CHARS_PER_TOKEN)
        }
        Message::ToolResult(result) => estimate_content_tokens(&result.content),
    }
}

/// Context token count of a message list, usage-anchored: the last
/// non-aborted/non-error assistant message with `usage.total_tokens > 0`
/// anchors the count; everything after it is locally estimated. With no usable
/// anchor, a pure estimate of the whole list. pi `estimateContextTokens`
/// (compaction.ts:183-214).
pub fn context_tokens(messages: &[Message]) -> u64 {
    for (idx, msg) in messages.iter().enumerate().rev() {
        if let Some(tokens) = msg.as_assistant().and_then(anchor_tokens) {
            let mut total = tokens;
            for after in &messages[idx + 1..] {
                total += estimate_tokens(after);
            }
            return total;
        }
    }
    messages.iter().map(estimate_tokens).sum()
}

/// Whether compaction should trigger: context tokens exceed
/// `context_window - reserve_tokens`. pi `shouldCompact` (compaction.ts:235-
/// 249). A reserve larger than the window saturates the threshold to zero.
pub fn should_compact(current_tokens: u64, context_window: u64, reserve_tokens: u64) -> bool {
    current_tokens > context_window.saturating_sub(reserve_tokens)
}

/// Find the cut point that keeps approximately `keep_recent_tokens` of the
/// tail, never splitting a `tool_call`/`tool_result` pair.
///
/// Walks from the tail toward the head accumulating [`estimate_tokens`] until
/// the budget is met; the candidate index is then moved forward (toward the
/// head) to the nearest legal boundary: never a `tool_result`, never right
/// after an assistant message with tool calls (their results would be
/// orphaned in the retained tail). If the retained tail covers the whole list
/// (cut at 0), returns `None` — an empty compaction ("Nothing to compact",
/// pi compaction.ts:1960-1963).
///
/// Returns the first kept index. pi's `isSplitTurn` is intentionally not
/// modeled: the transcript embeds the whole compacted range, so a mid-turn
/// cut needs no turn-prefix summary.
pub fn find_cut_point(messages: &[Message], keep_recent_tokens: u64) -> Option<usize> {
    if messages.is_empty() {
        return None;
    }

    // Walk backwards from the tail, accumulating estimated tokens until the
    // budget is met. Candidate = first index to keep. Zero-token messages are
    // skipped (pi skips them too), so an all-empty list never crosses the
    // budget and the loop ends with candidate still 0.
    let mut accumulated: u64 = 0;
    let mut candidate: usize = 0;
    for (i, msg) in messages.iter().enumerate().rev() {
        let tokens = estimate_tokens(msg);
        if tokens == 0 {
            continue;
        }
        accumulated += tokens;
        if accumulated >= keep_recent_tokens {
            candidate = i;
            break;
        }
    }

    // Move forward (toward the head) to the nearest legal boundary: never a
    // tool result, and never right after an assistant message with tool calls
    // (the retained tail would start with their orphaned results).
    while candidate > 0 && !is_legal_cut(messages, candidate) {
        candidate -= 1;
    }

    // Retained tail covers everything: an empty compaction.
    if candidate == 0 {
        return None;
    }

    Some(candidate)
}

/// Deterministic transcript of the compacted segments: the content of the
/// `Segment::Summary` that replaces them.
///
/// Rendering rules:
/// - preamble line (history was compacted; full outputs at the referenced
///   absolute paths);
/// - user/assistant dialogue text inline as `[User] {text}` /
///   `[Assistant] {text}`;
/// - thinking blocks dropped entirely (process, not content);
/// - tool calls with serialized arguments ≤ 100 chars inline as
///   `[Tool call] name({args})`; longer ones as a 100-char preview plus
///   the path: `[Tool call] name({preview}…) — full arguments at
///   {object path}` (character-granular: canonical JSON can contain
///   multibyte UTF-8);
/// - images as `[Image] at {object path}` — the object holding the raw
///   image bytes;
/// - tool results as one `[Tool result] {object path}` line per content
///   block (block-granular objects, ADR-0006); the `details` payload
///   never renders — tool display/truncation metadata, process not
///   content (like thinking), originals retrievable via
///   `fetch_originals`; a result with no content renders
///   `[Tool result] (empty)`;
/// - a `Segment::Summary` inside the range embeds its content as-is (it is
///   already a transcript with paths — re-compaction stays transitive).
///
/// Every referenced path is recomputed from the segment's own bytes
/// (content addressing); pass store-materialized segments and the paths
/// are guaranteed to exist.
pub fn build_transcript(
    segments: &[Segment],
    objects: &ObjectStore,
) -> crate::error::Result<String> {
    let mut parts = vec![TRANSCRIPT_PREAMBLE.to_string()];
    for segment in segments {
        match segment {
            Segment::User(user) => {
                for block in &user.content {
                    match block {
                        crate::segment::ContentBlock::Text { text } => {
                            if !text.is_empty() {
                                parts.push(format!("[User] {text}"));
                            }
                        }
                        crate::segment::ContentBlock::Image { .. } => {
                            parts.push(format!(
                                "[Image] at {}",
                                objects
                                    .object_path(&hash_hex(&block_bytes(block)?))?
                                    .display()
                            ));
                        }
                        // Other variants never appear in user segments.
                        _ => {}
                    }
                }
            }
            Segment::Assistant(assistant) => {
                for block in &assistant.content {
                    match block {
                        crate::segment::ContentBlock::Thinking { .. } => {
                            // Thinking is process, not content: dropped, not unloaded.
                        }
                        crate::segment::ContentBlock::Text { text } => {
                            parts.push(format!("[Assistant] {text}"));
                        }
                        crate::segment::ContentBlock::ToolCall(call) => {
                            let args_bytes = canonical_json(&call.arguments)?;
                            let args = String::from_utf8(args_bytes)
                                .map_err(|e| crate::error::StoreError::Encode(e.to_string()))?;
                            if args.chars().count() <= INLINE_ARG_MAX_CHARS {
                                parts.push(format!("[Tool call] {}({args})", call.name));
                            } else {
                                let preview: String =
                                    args.chars().take(INLINE_ARG_MAX_CHARS).collect::<String>();
                                parts.push(format!(
                                    "[Tool call] {}({preview}…) — full arguments at {}",
                                    call.name,
                                    objects.object_path(&hash_hex(args.as_bytes()))?.display()
                                ));
                            }
                        }
                        crate::segment::ContentBlock::Image { .. } => {
                            parts.push(format!(
                                "[Image] at {}",
                                objects
                                    .object_path(&hash_hex(&block_bytes(block)?))?
                                    .display()
                            ));
                        }
                    }
                }
            }
            Segment::ToolResult(result) => {
                if result.content.is_empty() {
                    parts.push("[Tool result] (empty)".to_string());
                }
                for block in &result.content {
                    parts.push(format!(
                        "[Tool result] {}",
                        objects
                            .object_path(&hash_hex(&block_bytes(block)?))?
                            .display()
                    ));
                }
                // Details are display/truncation metadata — process, not
                // content — so they stay out of the transcript; forensics
                // go through `fetch_originals`.
            }
            Segment::Summary(summary) => {
                if !summary.content.is_empty() {
                    parts.push(summary.content.clone());
                }
            }
        }
    }
    Ok(parts.join("\n"))
}

/// Convert materialized store segments to in-memory messages — the
/// compaction coordinator's read path.
///
/// `Summary` segments have no agent-side message type (see
/// [`crate::convert`]); each becomes a bare user message whose text is the
/// summary content. Both read paths (this one and
/// [`crate::agent_session::AgentSession::resume`], which additionally
/// stamps true write times) render Summary as a bare user message.
///
/// When the view contains a Summary, every assistant `usage` is zeroed
/// (in-memory only; the persisted `entries.usage` column stays faithful).
/// A pre-compaction assistant's `usage.total_tokens` describes the
/// PRE-compaction context, so an anchor that survives compaction in the
/// retained tail is stale: it overstates the post-compaction context and
/// misjudges metering. Zeroing makes [`context_tokens`] fall back to pure
/// estimation — the documented no-anchor path — until the next assistant
/// response lands with fresh usage.
pub fn view_messages(segments: &[Segment]) -> Vec<Message> {
    let mut messages: Vec<Message> = segments
        .iter()
        .map(|segment| match Message::try_from(segment.clone()) {
            Ok(message) => message,
            Err(ConvertError::Summary(summary)) => Message::User(UserMessage::new(summary.content)),
        })
        .collect();
    if segments
        .iter()
        .any(|segment| matches!(segment, Segment::Summary(_)))
    {
        for message in &mut messages {
            if let Message::Assistant(assistant) = message {
                assistant.usage = Default::default();
            }
        }
    }
    messages
}

/// `usage.total_tokens` of an assistant message, if it is a usable anchor:
/// non-aborted, non-error, and with a positive token count. pi
/// `getAssistantUsage` (compaction.ts:160-173).
fn anchor_tokens(assistant: &AssistantMessage) -> Option<u64> {
    match assistant.stop_reason {
        StopReason::Aborted | StopReason::Error => None,
        _ if assistant.usage.total_tokens > 0 => Some(assistant.usage.total_tokens),
        _ => None,
    }
}

fn estimate_content_tokens(blocks: &[ContentBlock]) -> u64 {
    let mut chars: u64 = 0;
    for block in blocks {
        match block {
            ContentBlock::Text { text } => chars += text.len() as u64,
            ContentBlock::Image { .. } => chars += ESTIMATED_IMAGE_CHARS,
            ContentBlock::Thinking { .. } | ContentBlock::ToolCall(_) => {}
        }
    }
    chars.div_ceil(CHARS_PER_TOKEN)
}

/// Whether cutting at `index` is legal: the cut message is not a tool result,
/// and no open tool call crosses the cut — every assistant tool call at or
/// before the cut must have its matching result before the cut, so the
/// retained tail never starts with an orphaned tool result. pi
/// `findValidCutPoints` + DSH `toolPairingBalancedBefore`.
fn is_legal_cut(messages: &[Message], index: usize) -> bool {
    debug_assert!(index > 0 && index < messages.len());
    if matches!(messages[index], Message::ToolResult(_)) {
        return false;
    }
    let mut open_calls: i64 = 0;
    for msg in &messages[..index] {
        match msg {
            Message::Assistant(assistant) => open_calls += assistant.tool_calls().len() as i64,
            Message::ToolResult(_) => open_calls -= 1,
            Message::User(_) => {}
        }
    }
    open_calls == 0
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::segment::{
        AssistantSegment, ContentBlock as StoreBlock, ImageSource as StoreImageSource,
        ToolCall as StoreToolCall, UserSegment,
    };
    use pi_agent_core::types::{
        ImageSource as AgentImageSource, ToolCall as AgentToolCall, ToolResultMessage,
        Usage as AgentUsage, UserMessage,
    };

    fn user(text: &str) -> Message {
        Message::User(UserMessage::new(text))
    }

    fn assistant(text: &str, reason: StopReason) -> Message {
        Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::text(text)],
            stop_reason: reason,
            model: "test".into(),
            provider: "test".into(),
            api: "test".into(),
            usage: AgentUsage::default(),
            error_message: None,
            timestamp: 0,
        })
    }

    fn assistant_with_usage(text: &str, total_tokens: u64, reason: StopReason) -> Message {
        Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::text(text)],
            stop_reason: reason,
            model: "test".into(),
            provider: "test".into(),
            api: "test".into(),
            usage: AgentUsage {
                total_tokens,
                ..AgentUsage::default()
            },
            error_message: None,
            timestamp: 0,
        })
    }

    /// Assistant message whose only content is one tool call. The call is
    /// *unpaired* by default — no matching tool result anywhere.
    fn tool_call_message(id: &str, name: &str, args: serde_json::Value) -> Message {
        Message::Assistant(AssistantMessage {
            content: vec![ContentBlock::ToolCall(AgentToolCall {
                id: id.into(),
                name: name.into(),
                arguments: args,
            })],
            stop_reason: StopReason::ToolUse,
            model: "test".into(),
            provider: "test".into(),
            api: "test".into(),
            usage: AgentUsage::default(),
            error_message: None,
            timestamp: 0,
        })
    }

    fn tool_result(text: &str, id: &str) -> Message {
        Message::ToolResult(ToolResultMessage {
            tool_call_id: id.into(),
            tool_name: "read_file".into(),
            content: vec![ContentBlock::text(text)],
            details: serde_json::Value::Null,
            usage: None,
            added_tool_names: None,
            is_error: false,
            timestamp: 0,
        })
    }

    /// Store-native assistant segment with the given content blocks.
    fn store_assistant(blocks: Vec<StoreBlock>) -> Segment {
        Segment::Assistant(AssistantSegment {
            content: blocks,
            stop_reason: crate::segment::StopReason::Stop,
            model: "test".into(),
            provider: "test".into(),
            api: "test".into(),
            usage: crate::segment::Usage::default(),
            error_message: None,
        })
    }

    #[test]
    fn estimate_pure_text_path() {
        // 8 chars → ceil(8/4) = 2
        assert_eq!(estimate_tokens(&user("abcdefgh")), 2);
        // 9 chars → ceil(9/4) = 3
        assert_eq!(estimate_tokens(&user("abcdefghi")), 3);
    }

    #[test]
    fn estimate_includes_thinking_and_tool_call() {
        let msg = Message::Assistant(AssistantMessage {
            content: vec![
                ContentBlock::text("abcd"),
                ContentBlock::Thinking {
                    text: "efgh".into(),
                },
                ContentBlock::ToolCall(AgentToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "/tmp/x"}),
                }),
            ],
            stop_reason: StopReason::Stop,
            model: "m".into(),
            provider: "p".into(),
            api: "a".into(),
            usage: AgentUsage::default(),
            error_message: None,
            timestamp: 0,
        });
        // text(4) + thinking(4) + name(4) + arguments(`{"path":"/tmp/x"}` = 17) = 29 → ceil(29/4) = 8
        assert_eq!(estimate_tokens(&msg), 8);
    }

    #[test]
    fn estimate_image_counts_1200_tokens() {
        let msg = Message::User(UserMessage {
            content: vec![
                ContentBlock::text("abcd"),
                ContentBlock::Image {
                    source: AgentImageSource {
                        mime_type: "image/png".into(),
                        bytes: vec![0u8; 100],
                    },
                },
            ],
            timestamp: 0,
        });
        // 4 chars + 4800 chars = 4804 → ceil(4804/4) = 1201
        assert_eq!(estimate_tokens(&msg), 1201);
    }

    #[test]
    fn estimate_tool_result_counts_content_text() {
        assert_eq!(estimate_tokens(&tool_result("abcdefgh", "c1")), 2);
    }

    #[test]
    fn context_tokens_uses_last_usage_anchor_plus_tail() {
        let messages = vec![
            user("aaaa"),                                         // est 1
            assistant_with_usage("bbbb", 1000, StopReason::Stop), // anchor 1000
            user("cccc"),                                         // est 1
            tool_result("dddd", "c1"),                            // est 1
        ];
        assert_eq!(context_tokens(&messages), 1000 + 1 + 1);
    }

    #[test]
    fn context_tokens_skips_aborted_and_error_anchors() {
        let messages = vec![
            assistant_with_usage("aaaa", 500, StopReason::Aborted), // skipped
            assistant_with_usage("bbbb", 700, StopReason::Error),   // skipped
            assistant_with_usage("cccc", 1000, StopReason::Stop),   // anchor
            user("dddd"),                                           // est 1
        ];
        assert_eq!(context_tokens(&messages), 1000 + 1);
    }

    #[test]
    fn context_tokens_falls_back_to_pure_estimate_when_no_anchor() {
        let messages = vec![
            user("aaaa"),                                         // est 1
            assistant_with_usage("bbbb", 0, StopReason::Stop),    // not an anchor: 0 tokens
            user("cccc"),                                         // est 1
            assistant_with_usage("dddd", 0, StopReason::Aborted), // not an anchor: aborted
        ];
        assert_eq!(context_tokens(&messages), 4);
    }

    #[test]
    fn should_compact_boundary() {
        // threshold = 1000 - 100 = 900; 900 is not > 900
        assert!(!should_compact(900, 1000, 100));
        assert!(should_compact(901, 1000, 100));
        // reserve larger than the window: threshold saturates to 0
        assert!(should_compact(1, 100, 1000));
    }

    #[test]
    fn find_cut_point_never_cuts_on_tool_result() {
        // u0(20 chars→5), a1(20→5), tr2(60→15), u3(20→5)
        let messages = vec![
            user("01234567890123456789"),
            assistant("01234567890123456789", StopReason::Stop),
            tool_result(
                "012345678901234567890123456789012345678901234567890123456789",
                "c1",
            ),
            user("01234567890123456789"),
        ];
        // Tail: u3=5 < 10; tr2=20 ≥ 10 → candidate 2 (tool result, illegal)
        // → move forward to 1 (assistant, message before = user) → keep [1..]
        let cut = find_cut_point(&messages, 10).expect("cut point");
        assert_eq!(cut, 1);
        assert!(matches!(messages[cut], Message::Assistant(_)));
        // The retained tail must not start with a tool result.
        assert!(!matches!(messages[cut], Message::ToolResult(_)));
    }

    #[test]
    fn find_cut_point_moves_forward_past_unpaired_tool_call() {
        // u0(20→5), a1 = tool call c1 (name "read"=4 + args 27 → 31 chars → 8),
        // u2(20→5) sits between the call and its result, tr3(20→5), a4(20→5).
        let messages = vec![
            user("01234567890123456789"),
            tool_call_message(
                "c1",
                "read",
                serde_json::json!({"path": "/tmp/xxxxxxxxxxx"}),
            ),
            user("01234567890123456789"),
            tool_result("01234567890123456789", "c1"),
            assistant("01234567890123456789", StopReason::Stop),
        ];
        // Tail: a4=5 < 12; tr3=10 < 12; u2=15 ≥ 12 → candidate 2 (user).
        // messages[1] is an assistant with an unpaired tool call (its result
        // sits at 3, in the retained tail) → illegal → move forward to 1.
        let cut = find_cut_point(&messages, 12).expect("cut point");
        assert_eq!(cut, 1);
        // The pair must end up together in the retained tail.
        let tail = &messages[cut..];
        assert!(matches!(tail[0], Message::Assistant(_)));
        assert!(
            tail.iter()
                .any(|m| matches!(m, Message::ToolResult(r) if r.tool_call_id == "c1"))
        );
    }

    #[test]
    fn find_cut_point_empty_compaction_returns_none() {
        let messages = vec![user("short"), assistant("short", StopReason::Stop)];
        // Entire list = 2+2 = 4 tokens < 1000 → budget never met → cut at 0 → None
        assert_eq!(find_cut_point(&messages, 1000), None);

        // Budget met exactly at index 0 (single giant message) → None too.
        let big = vec![user(&"x".repeat(40))];
        assert_eq!(find_cut_point(&big, 5), None);
    }

    #[test]
    fn build_transcript_renders_all_segment_kinds() {
        let dir = tempfile::tempdir().unwrap();
        let objects = ObjectStore::new(dir.path());

        let user_seg = Segment::user_text("hello");
        let assistant_seg = store_assistant(vec![StoreBlock::Text {
            text: "hi there".into(),
        }]);
        let thinking_seg = store_assistant(vec![StoreBlock::Thinking { text: "hmm".into() }]);
        let call_seg = store_assistant(vec![StoreBlock::ToolCall(StoreToolCall {
            id: "c1".into(),
            name: "read".into(),
            arguments: serde_json::json!({"path": "/tmp/x"}),
        })]);
        let long_call_args = serde_json::json!({"command": "x".repeat(2500)});
        let long_call_seg = store_assistant(vec![StoreBlock::ToolCall(StoreToolCall {
            id: "c2".into(),
            name: "bash".into(),
            arguments: long_call_args.clone(),
        })]);
        let result_seg = Segment::tool_result_text("c1", "file content");
        let embedded_seg = Segment::summary("already compacted");

        let segments = vec![
            user_seg,
            assistant_seg,
            thinking_seg,
            call_seg,
            long_call_seg,
            result_seg,
            embedded_seg,
        ];
        let transcript = build_transcript(&segments, &objects).unwrap();

        assert!(transcript.starts_with(TRANSCRIPT_PREAMBLE));
        assert!(transcript.contains("[User] hello"));
        assert!(transcript.contains("[Assistant] hi there"));
        assert!(
            !transcript.contains("hmm"),
            "thinking is dropped: {transcript}"
        );
        assert!(!transcript.contains("[Thinking]"), "{transcript}");
        assert!(transcript.contains(r#"[Tool call] read({"path":"/tmp/x"})"#));
        // Long arguments render as a 100-char preview plus the path to the
        // canonical-JSON arguments object; the path is recomputed from the
        // segment's own bytes (content addressing, ADR-0006).
        let args_json = String::from_utf8(canonical_json(&long_call_args).unwrap()).unwrap();
        let preview: String = args_json.chars().take(INLINE_ARG_MAX_CHARS).collect();
        let args_path = objects
            .object_path(&hash_hex(&canonical_json(&long_call_args).unwrap()))
            .unwrap();
        assert!(
            transcript.contains(&format!(
                "[Tool call] bash({preview}…) — full arguments at {}",
                args_path.display()
            )),
            "long-arg preview and path resolve: {transcript}"
        );
        assert!(
            !transcript.contains(&"x".repeat(2500)),
            "full long arguments are not inlined: {transcript}"
        );
        let result_path = objects.object_path(&hash_hex(b"file content")).unwrap();
        assert!(
            transcript.contains(&format!("[Tool result] {}", result_path.display())),
            "tool result path resolves: {transcript}"
        );
        assert!(
            transcript.contains("already compacted"),
            "embedded summary content is inlined"
        );
        // Images render as an [Image] line pointing at the raw-bytes object.
        let image_seg = Segment::User(UserSegment {
            content: vec![
                StoreBlock::Text {
                    text: "with image".into(),
                },
                StoreBlock::Image {
                    source: StoreImageSource {
                        mime_type: "image/png".into(),
                        bytes: vec![1u8, 2, 3],
                    },
                },
            ],
        });
        let image_path = objects.object_path(&hash_hex(&[1u8, 2, 3])).unwrap();
        let transcript =
            build_transcript(&[Segment::user_text("hello"), image_seg], &objects).unwrap();
        assert!(transcript.contains("[User] hello"));
        assert!(
            transcript.contains(&format!("[Image] at {}", image_path.display())),
            "image path resolves: {transcript}"
        );
    }

    #[test]
    fn build_transcript_tool_result_drops_details_and_empty() {
        let dir = tempfile::tempdir().unwrap();
        let objects = ObjectStore::new(dir.path());

        // Details never reach the transcript: only the content block's
        // object path renders; the details object path does not.
        let with_details = Segment::ToolResult(crate::segment::ToolResultSegment {
            tool_call_id: "c1".into(),
            tool_name: "bash".into(),
            content: vec![StoreBlock::Text { text: "out".into() }],
            details: serde_json::json!({"exit": 0}),
            usage: None,
            added_tool_names: None,
            is_error: false,
        });
        let transcript = build_transcript(&[with_details], &objects).unwrap();
        let details_path = objects
            .object_path(&hash_hex(
                &canonical_json(&serde_json::json!({"exit": 0})).unwrap(),
            ))
            .unwrap();
        assert!(
            !transcript.contains(details_path.to_str().unwrap()),
            "details object is not referenced: {transcript}"
        );
        let content_path = objects.object_path(&hash_hex(b"out")).unwrap();
        assert!(
            transcript.contains(&format!("[Tool result] {}", content_path.display())),
            "content path renders: {transcript}"
        );

        // No content: the explicit empty marker, with or without details.
        let empty = |details: serde_json::Value| {
            Segment::ToolResult(crate::segment::ToolResultSegment {
                tool_call_id: "c2".into(),
                tool_name: "bash".into(),
                content: vec![],
                details,
                usage: None,
                added_tool_names: None,
                is_error: false,
            })
        };
        for details in [serde_json::Value::Null, serde_json::json!({"exit": 1})] {
            let transcript = build_transcript(&[empty(details)], &objects).unwrap();
            assert!(
                transcript.contains("[Tool result] (empty)"),
                "empty marker regardless of details: {transcript}"
            );
            assert!(
                !transcript.contains("/tmp/"),
                "no path line for an empty result: {transcript}"
            );
        }
    }

    #[test]
    fn build_transcript_tool_call_char_boundary() {
        let dir = tempfile::tempdir().unwrap();
        let objects = ObjectStore::new(dir.path());

        // Canonical JSON does not escape non-ASCII: `{"cmd":"` is 8 chars and
        // `"}` is 2, so 90 令 chars give exactly 100 → inline; 91 → 101 →
        // truncated. Proves the threshold is character-granular and
        // multibyte-safe (a byte comparison would treat 令 as 3 bytes).
        let inline_args = serde_json::json!({"cmd": "令".repeat(90)});
        let inline_seg = store_assistant(vec![StoreBlock::ToolCall(StoreToolCall {
            id: "c1".into(),
            name: "bash".into(),
            arguments: inline_args,
        })]);
        let inline_args_json = String::from_utf8(
            canonical_json(&serde_json::json!({"cmd": "令".repeat(90)})).unwrap(),
        )
        .unwrap();
        assert_eq!(inline_args_json.chars().count(), 100);
        let transcript = build_transcript(&[inline_seg], &objects).unwrap();
        assert!(
            transcript.contains(&format!("[Tool call] bash({inline_args_json})")),
            "exactly 100 chars stays inline: {transcript}"
        );

        let truncated_args = serde_json::json!({"cmd": "令".repeat(91)});
        let truncated_seg = store_assistant(vec![StoreBlock::ToolCall(StoreToolCall {
            id: "c2".into(),
            name: "bash".into(),
            arguments: truncated_args.clone(),
        })]);
        let truncated_args_json =
            String::from_utf8(canonical_json(&truncated_args).unwrap()).unwrap();
        assert_eq!(truncated_args_json.chars().count(), 101);
        let preview: String = truncated_args_json
            .chars()
            .take(INLINE_ARG_MAX_CHARS)
            .collect();
        let args_path = objects
            .object_path(&hash_hex(&canonical_json(&truncated_args).unwrap()))
            .unwrap();
        let transcript = build_transcript(&[truncated_seg], &objects).unwrap();
        assert!(
            transcript.contains(&format!(
                "[Tool call] bash({preview}…) — full arguments at {}",
                args_path.display()
            )),
            "101 chars is truncated with path: {transcript}"
        );
        assert!(
            !transcript.contains(&truncated_args_json),
            "full 91-令 arguments are not inlined: {transcript}"
        );
    }

    #[tokio::test]
    async fn build_transcript_paths_match_appended_objects() {
        // End-to-end over the real write path: after `append_segment`, the
        // recomputed transcript paths exist on disk and hold the block bytes.
        let dir = tempfile::tempdir().unwrap();
        let store = crate::db::SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        let result_text = "R".repeat(100);
        store
            .append_segment(&root, &Segment::tool_result_text("c1", &result_text))
            .await
            .unwrap();
        let image_bytes = vec![9u8; 16];
        store
            .append_segment(
                &root,
                &Segment::User(UserSegment {
                    content: vec![StoreBlock::Image {
                        source: StoreImageSource {
                            mime_type: "image/png".into(),
                            bytes: image_bytes.clone(),
                        },
                    }],
                }),
            )
            .await
            .unwrap();
        let segments = store.materialize_plain(&root).await.unwrap();
        let transcript = build_transcript(&segments, store.objects()).unwrap();
        let path = store
            .objects()
            .object_path(&hash_hex(result_text.as_bytes()))
            .unwrap();
        assert!(transcript.contains(&format!("[Tool result] {}", path.display())));
        assert_eq!(std::fs::read_to_string(path).unwrap(), result_text);
        let image_path = store
            .objects()
            .object_path(&hash_hex(&image_bytes))
            .unwrap();

        assert!(
            transcript.contains(&format!("[Image] at {}", image_path.display())),
            "image line points at the store object: {transcript}"
        );
        assert_eq!(std::fs::read(image_path).unwrap(), image_bytes);
    }
    #[test]
    fn view_messages_zeroes_assistant_usage_when_summary_present() {
        let mut anchored = Segment::assistant_text("a1");
        if let Segment::Assistant(a) = &mut anchored {
            a.usage.total_tokens = 100_000;
        }
        let segments = vec![
            Segment::user_text("q1"),
            anchored,
            Segment::summary("the gist"),
        ];
        let messages = view_messages(&segments);
        let Message::Assistant(assistant) = &messages[1] else {
            panic!("assistant must stay an assistant message");
        };
        assert_eq!(
            assistant.usage.total_tokens, 0,
            "a view containing a summary carries no pre-compaction usage anchors"
        );
    }

    #[test]
    fn view_messages_keeps_assistant_usage_without_summary() {
        let mut anchored = Segment::assistant_text("a1");
        if let Segment::Assistant(a) = &mut anchored {
            a.usage.total_tokens = 100_000;
        }
        let segments = vec![Segment::user_text("q1"), anchored];
        let messages = view_messages(&segments);
        let Message::Assistant(assistant) = &messages[1] else {
            panic!("assistant must stay an assistant message");
        };
        assert_eq!(
            assistant.usage.total_tokens, 100_000,
            "usage untouched when no summary is present"
        );
    }
    #[test]
    fn view_messages_renders_summary_as_bare_user_message() {
        let segments = vec![
            Segment::user_text("q1"),
            Segment::summary("the gist"),
            Segment::assistant_text("a1"),
        ];
        let messages = view_messages(&segments);
        assert_eq!(messages.len(), 3);
        let Message::User(summary_msg) = &messages[1] else {
            panic!("summary must render as a user message");
        };
        assert_eq!(summary_msg.content, vec![ContentBlock::text("the gist")]);
    }
}
