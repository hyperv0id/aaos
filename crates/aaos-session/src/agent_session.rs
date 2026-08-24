//! `AgentSession`: binds one `SessionStore` + one session node + one [`Agent`].
//!
//! The agent's `state.messages` is the sole in-memory transcript; the store
//! is the persistent projection. They sync through the [`convert`](crate::convert) layer:
//!
//! - **Write**: a `MessageEnd` listener turns every inbound message (user
//!   prompt, steering, assistant output, tool result — all commit through
//!   `MessageEnd`, see agent.rs `process_event`) into a `Segment` and awaits
//!   `append_segment` inline. Awaiting synchronously is what makes append
//!   order equal event order; no fire-and-forget.
//! - **Read**: [`resume`](AgentSession::resume) replaces (never pushes onto)
//!   `agent.state.messages` with the materialized view of a node, repairing
//!   dangling tool calls in memory only.
//!
//! The session id is shared with the listener through an `Arc<RwLock<_>>` so
//! a `/fork`-style switch redirects subsequent appends to the new node.

use std::collections::HashMap;
use std::sync::Arc;

use pi_agent_core::agent::Agent;
use pi_agent_core::types::{AgentEvent, ContentBlock, Message, ToolResultMessage, UserMessage};
use tokio::sync::RwLock;

use crate::Result;
use crate::convert::ConvertError;
use crate::db::SessionStore;
use crate::segment::Segment;

/// Handle binding a session store, a session node id, and an agent.
pub struct AgentSession {
    store: SessionStore,
    /// Current node id (first resume target, then append target). Shared
    /// with the persist listener so a session switch redirects appends.
    session_id: Arc<RwLock<String>>,
    agent: Agent,
}

impl AgentSession {
    /// Bind `agent` to the initial `session_id` node and register the
    /// `MessageEnd` persist listener.
    pub fn new(store: SessionStore, agent: Agent, session_id: impl Into<String>) -> Self {
        let session_id = Arc::new(RwLock::new(session_id.into()));
        let (listener_store, listener_session_id) = (store.clone(), session_id.clone());
        // The returned unregister handle is intentionally dropped: the
        // listener lives as long as the agent it is bound to.
        let _ = agent.subscribe(Arc::new(move |event, _abort| {
            let store = listener_store.clone();
            let session_id = listener_session_id.clone();
            Box::pin(async move {
                if let AgentEvent::MessageEnd { message } = event {
                    let segment = Segment::from(&message);
                    let current_id = session_id.read().await.clone();
                    if let Err(err) = store.append_segment(&current_id, &segment).await {
                        // Listeners cannot propagate errors; surface the
                        // failure so persistence loss is not silent.
                        #[allow(clippy::print_stderr)]
                        {
                            eprintln!("aaos-session: append to session {current_id} failed: {err}");
                        }
                    }
                }
            })
        }));
        Self {
            store,
            session_id,
            agent,
        }
    }

    /// Current session node id (the next append target).
    pub async fn current_session_id(&self) -> String {
        self.session_id.read().await.clone()
    }

    /// Load the view of `session_id` into `agent.state.messages` and make it
    /// the current node.
    ///
    /// Replaces (does not push onto) the message list — resuming a fork on a
    /// non-empty agent would otherwise duplicate the old prefix. Returns the
    /// number of messages loaded.
    pub async fn resume(&mut self, session_id: &str) -> Result<usize> {
        let segments = self.store.materialize_plain(session_id).await?;
        let messages = materialize_messages(segments);
        let messages = repair_dangling_tool_calls(messages);
        let count = messages.len();
        self.agent.state.messages = messages;
        *self.session_id.write().await = session_id.to_string();
        Ok(count)
    }
}

/// Convert segments to messages; `Summary` segments become user messages
/// with a provenance prefix (the agent has no matching message type).
fn materialize_messages(segments: Vec<Segment>) -> Vec<Message> {
    segments
        .into_iter()
        .map(|segment| match Message::try_from(segment) {
            Ok(message) => message,
            Err(ConvertError::Summary(summary)) => Message::User(UserMessage {
                content: vec![ContentBlock::text(format!(
                    "[compacted summary] {}",
                    summary.content
                ))],
                timestamp: pi_agent_core::types::now(),
            }),
        })
        .collect()
}

/// Repair dangling tool calls in the in-memory view only (never written back
/// to the store): every `tool_call` of an assistant message that has no
/// matching tool result *after* it gets a synthesized error result in the
/// same shape as `fail_tool_calls_from_truncated_message` (agent_loop.rs)
/// emits, with an interruption-specific message.
fn repair_dangling_tool_calls(messages: Vec<Message>) -> Vec<Message> {
    let mut last_result_index: HashMap<String, usize> = HashMap::new();
    for (index, message) in messages.iter().enumerate() {
        if let Message::ToolResult(result) = message {
            last_result_index.insert(result.tool_call_id.clone(), index);
        }
    }

    let mut repaired = Vec::with_capacity(messages.len());
    for (index, message) in messages.into_iter().enumerate() {
        let synthesized: Vec<ToolResultMessage> = message
            .as_assistant()
            .map(|assistant| {
                assistant
                    .tool_calls()
                    .into_iter()
                    .filter_map(|call| {
                        let answered = last_result_index
                            .get(&call.id)
                            .is_some_and(|result_index| *result_index > index);
                        if answered {
                            return None;
                        }
                        Some(ToolResultMessage {
                            tool_call_id: call.id.clone(),
                            tool_name: call.name.clone(),
                            content: vec![ContentBlock::text(format!(
                                "Tool call \"{name}\" was not executed: the previous run was \
                                 interrupted",
                                name = call.name
                            ))],
                            details: serde_json::json!({}),
                            usage: None,
                            added_tool_names: None,
                            is_error: true,
                            timestamp: pi_agent_core::types::now(),
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        repaired.push(message);
        repaired.extend(synthesized.into_iter().map(Message::ToolResult));
    }
    repaired
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use crate::segment::{
        AssistantSegment, ContentBlock as StoreContentBlock, StopReason as StoreStopReason,
        ToolCall as StoreToolCall, Usage as StoreUsage,
    };
    use pi_agent_core::stream::simple_text_response;
    use pi_agent_core::types::{Model, StreamFn};

    async fn test_store() -> (SessionStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        (store, dir)
    }

    fn make_agent(stream_fn: Arc<dyn StreamFn>) -> Agent {
        let mut agent = Agent::new(stream_fn);
        agent.state.model = Model {
            id: "test".into(),
            provider: "test".into(),
            api: "test".into(),
            ..Default::default()
        };
        agent
    }

    fn user_text(segment: &Segment) -> &str {
        match segment {
            Segment::User(user) => match &user.content[0] {
                StoreContentBlock::Text { text } => text,
                other => panic!("expected text block, got {other:?}"),
            },
            other => panic!("expected user segment, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn prompt_turn_persists() {
        let (store, _dir) = test_store().await;
        let session_id = store.create_root().await.unwrap();
        let agent = make_agent(simple_text_response("hello there"));
        let mut session = AgentSession::new(store.clone(), agent, session_id.clone());

        assert_eq!(session.resume(&session_id).await.unwrap(), 0);
        session.agent.prompt("first question").await.unwrap();

        let segments = store.materialize_plain(&session_id).await.unwrap();
        assert_eq!(segments.len(), 2, "user + assistant persisted");
        assert_eq!(segments[0].kind(), "user");
        assert_eq!(user_text(&segments[0]), "first question");
        assert_eq!(segments[1].kind(), "assistant");
    }

    #[tokio::test]
    async fn resume_repairs_dangling_tool_call() {
        let (store, _dir) = test_store().await;
        let session_id = store.create_root().await.unwrap();

        store
            .append_segment(&session_id, &Segment::user_text("run the tool"))
            .await
            .unwrap();
        let call = StoreToolCall {
            id: "call-1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/tmp/x"}),
        };
        store
            .append_segment(
                &session_id,
                &Segment::Assistant(AssistantSegment {
                    content: vec![StoreContentBlock::ToolCall(call)],
                    stop_reason: StoreStopReason::ToolUse,
                    model: "test-model".into(),
                    provider: "test-provider".into(),
                    api: "test-api".into(),
                    usage: StoreUsage::default(),
                    error_message: None,
                }),
            )
            .await
            .unwrap();

        let mut session = AgentSession::new(
            store,
            make_agent(simple_text_response("done")),
            session_id.clone(),
        );
        assert_eq!(session.resume(&session_id).await.unwrap(), 3);

        let messages = &session.agent.state.messages;
        assert_eq!(messages.len(), 3);
        assert!(matches!(&messages[0], Message::User(_)));
        assert!(matches!(&messages[1], Message::Assistant(_)));
        let Message::ToolResult(result) = messages.last().unwrap() else {
            panic!(
                "expected a synthesized tool result last, got {:?}",
                messages.last()
            );
        };
        assert!(result.is_error);
        assert_eq!(result.tool_call_id, "call-1");
        assert_eq!(
            result.content,
            vec![ContentBlock::text(
                "Tool call \"read_file\" was not executed: the previous run was interrupted"
            )]
        );

        // The repaired transcript is continuable: the last message is a
        // tool result, never an assistant, so continue_run is not rejected.
        session.agent.continue_run().await.unwrap();

        let messages = &session.agent.state.messages;
        assert_eq!(messages.len(), 4);
        assert!(matches!(messages.last().unwrap(), Message::Assistant(_)));
    }

    #[tokio::test]
    async fn resume_keeps_complete_tool_pair() {
        let (store, _dir) = test_store().await;
        let session_id = store.create_root().await.unwrap();

        store
            .append_segment(&session_id, &Segment::user_text("look at the file"))
            .await
            .unwrap();
        let call = StoreToolCall {
            id: "call-1".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/tmp/x"}),
        };
        store
            .append_segment(
                &session_id,
                &Segment::Assistant(AssistantSegment {
                    content: vec![StoreContentBlock::ToolCall(call)],
                    stop_reason: StoreStopReason::ToolUse,
                    model: "test-model".into(),
                    provider: "test-provider".into(),
                    api: "test-api".into(),
                    usage: StoreUsage::default(),
                    error_message: None,
                }),
            )
            .await
            .unwrap();
        store
            .append_segment(
                &session_id,
                &Segment::tool_result_text("call-1", "the file content"),
            )
            .await
            .unwrap();

        let mut session = AgentSession::new(
            store,
            make_agent(simple_text_response("ok")),
            session_id.clone(),
        );
        assert_eq!(session.resume(&session_id).await.unwrap(), 3);
        let messages = &session.agent.state.messages;
        assert_eq!(messages.len(), 3, "no synthesis for a complete pair");
        assert!(matches!(messages.last().unwrap(), Message::ToolResult(_)));
    }
}
