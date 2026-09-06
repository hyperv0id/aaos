//! Shared test fixtures for this crate's test modules: the message-view
//! helper and the store seeding routines behind the compaction/session
//! assertions. `#[cfg(test)]` only.
#![allow(clippy::unwrap_used)]

use aaos_session::{
    AssistantSegment, ContentBlock as StoreBlock, Segment, SessionStore,
    StopReason as StoreStopReason, ToolCall as StoreToolCall, Usage as StoreUsage,
};
use pi_agent_core::types::{ContentBlock, Message};

pub(crate) fn first_text(msg: &Message) -> String {
    let content = match msg {
        Message::User(u) => &u.content,
        Message::Assistant(a) => &a.content,
        Message::ToolResult(t) => &t.content,
    };
    content
        .iter()
        .find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

/// Seed `n` user/assistant text turns, each ~`chars` chars long.
pub(crate) async fn seed_turns(store: &SessionStore, root: &str, n: usize, chars: usize) {
    for i in 0..n {
        store
            .append_segment(
                root,
                &Segment::user_text(format!("u{i}-{}", "x".repeat(chars))),
            )
            .await
            .unwrap();
        store
            .append_segment(
                root,
                &Segment::assistant_text(format!("a{i}-{}", "y".repeat(chars))),
            )
            .await
            .unwrap();
    }
}

/// Seed one tool round-trip: an assistant tool call + a `result_chars`-long
/// tool result. Tool results are the context bulk compaction replaces with a
/// path, so they make the projection strictly smaller.
pub(crate) async fn seed_tool_turn(
    store: &SessionStore,
    root: &str,
    call_id: &str,
    name: &str,
    args: serde_json::Value,
    result_chars: usize,
) {
    store
        .append_segment(
            root,
            &Segment::Assistant(AssistantSegment {
                content: vec![StoreBlock::ToolCall(StoreToolCall {
                    id: call_id.into(),
                    name: name.into(),
                    arguments: args,
                })],
                stop_reason: StoreStopReason::ToolUse,
                model: "test".into(),
                provider: "test".into(),
                api: "test".into(),
                usage: StoreUsage::default(),
                error_message: None,
            }),
        )
        .await
        .unwrap();
    store
        .append_segment(
            root,
            &Segment::tool_result_text(call_id, "R".repeat(result_chars)),
        )
        .await
        .unwrap();
}
