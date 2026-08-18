//! Provider stream seam: fake in-memory provider streams for driving the agent loop.
//!
//! Real providers are out of scope for the embryo. Tests program a
//! [`MockAssistantStream`] with the event sequence they expect the agent loop
//! to observe, plus the final [`AssistantMessage`] that `result` returns, and
//! hand it to the loop through [`mock_stream_fn`] (or the
//! [`simple_text_response`] / [`tool_use_response`] one-liners).

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;

use crate::types::{
    AssistantEventStream, AssistantMessage, AssistantMessageEvent, ContentBlock, LlmContext,
    StopReason, StreamFn, StreamFnOptions, ToolCall,
};

/// In-memory fake provider stream.
///
/// `next_event` pops the pre-programmed event queue until exhausted;
/// `result` returns the pre-programmed final message.
pub struct MockAssistantStream {
    events: VecDeque<AssistantMessageEvent>,
    final_message: AssistantMessage,
}

impl MockAssistantStream {
    pub fn new(final_message: AssistantMessage) -> Self {
        Self {
            events: VecDeque::new(),
            final_message,
        }
    }

    /// Append an event to the programmed sequence.
    pub fn push(&mut self, event: AssistantMessageEvent) {
        self.events.push_back(event);
    }
}

#[async_trait]
impl AssistantEventStream for MockAssistantStream {
    async fn next_event(&mut self) -> Option<AssistantMessageEvent> {
        self.events.pop_front()
    }

    async fn result(self: Box<Self>) -> AssistantMessage {
        self.final_message
    }
}

/// Wraps a plain closure factory so it satisfies the [`StreamFn`] trait seam.
///
/// The factory is invoked once per `call` with the model, LLM context, and
/// stream options. The abort signal is ignored: fake providers return their
/// programmed stream immediately instead of long-polling.
pub fn mock_stream_fn<F>(factory: F) -> Arc<dyn StreamFn>
where
    F: FnMut(String, LlmContext, StreamFnOptions) -> Box<dyn AssistantEventStream>
        + Send
        + Sync
        + 'static,
{
    Arc::new(ClosureStreamFn {
        factory: Mutex::new(factory),
    })
}

struct ClosureStreamFn<F> {
    factory: Mutex<F>,
}

#[async_trait]
impl<F> StreamFn for ClosureStreamFn<F>
where
    F: FnMut(String, LlmContext, StreamFnOptions) -> Box<dyn AssistantEventStream>
        + Send
        + Sync
        + 'static,
{
    async fn call(
        &self,
        model: String,
        context: LlmContext,
        options: StreamFnOptions,
        _abort: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Box<dyn AssistantEventStream>, String> {
        let mut factory = self
            .factory
            .lock()
            .map_err(|_| "mock stream fn factory poisoned".to_string())?;
        Ok(factory(model, context, options))
    }
}

/// A stream fn that immediately yields an assistant message containing `text`
/// with a `stop` stop reason. No events are emitted.
pub fn simple_text_response(text: &str) -> Arc<dyn StreamFn> {
    let message = AssistantMessage::text(text);
    mock_stream_fn(move |_model, _context, _options| {
        Box::new(MockAssistantStream::new(message.clone()))
    })
}

/// A stream fn that immediately yields an assistant message containing the
/// given tool calls, with `stop_reason` (typically `StopReason::ToolUse`).
/// No events are emitted.
pub fn tool_use_response(tool_calls: Vec<ToolCall>, stop_reason: StopReason) -> Arc<dyn StreamFn> {
    let message = AssistantMessage {
        content: tool_calls.into_iter().map(ContentBlock::ToolCall).collect(),
        stop_reason,
        ..Default::default()
    };
    mock_stream_fn(move |_model, _context, _options| {
        Box::new(MockAssistantStream::new(message.clone()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn empty_llm_context() -> LlmContext {
        LlmContext {
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: Vec::new(),
        }
    }

    fn abort_channel() -> tokio::sync::watch::Receiver<bool> {
        tokio::sync::watch::channel(true).1
    }

    #[tokio::test]
    async fn fake_provider_yields_programmed_events_and_result() {
        let final_message = AssistantMessage::text("hi");
        let empty = AssistantMessage::text("");
        let mut stream = MockAssistantStream::new(final_message.clone());
        stream.push(AssistantMessageEvent::Start {
            partial: empty.clone(),
        });
        stream.push(AssistantMessageEvent::TextStart {
            content_index: 0,
            partial: empty.clone(),
        });
        stream.push(AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "h".into(),
            partial: AssistantMessage::text("h"),
        });
        stream.push(AssistantMessageEvent::TextDelta {
            content_index: 0,
            delta: "i".into(),
            partial: final_message.clone(),
        });
        stream.push(AssistantMessageEvent::TextEnd {
            content_index: 0,
            content: "hi".into(),
            partial: final_message.clone(),
        });
        stream.push(AssistantMessageEvent::Done {
            reason: StopReason::Stop,
            message: final_message.clone(),
        });

        assert_eq!(
            stream.next_event().await,
            Some(AssistantMessageEvent::Start { partial: empty })
        );
        assert_eq!(
            stream.next_event().await,
            Some(AssistantMessageEvent::TextStart {
                content_index: 0,
                partial: AssistantMessage::text(""),
            })
        );
        assert_eq!(
            stream.next_event().await,
            Some(AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "h".into(),
                partial: AssistantMessage::text("h"),
            })
        );
        assert_eq!(
            stream.next_event().await,
            Some(AssistantMessageEvent::TextDelta {
                content_index: 0,
                delta: "i".into(),
                partial: final_message.clone(),
            })
        );
        assert_eq!(
            stream.next_event().await,
            Some(AssistantMessageEvent::TextEnd {
                content_index: 0,
                content: "hi".into(),
                partial: final_message.clone(),
            })
        );
        assert_eq!(
            stream.next_event().await,
            Some(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                message: final_message.clone(),
            })
        );
        // Queue exhausted: subsequent polls return None.
        assert_eq!(stream.next_event().await, None);

        let result = Box::new(stream).result().await;
        assert_eq!(result, final_message);
    }

    #[tokio::test]
    async fn mock_stream_fn_invokes_factory_with_call_arguments() {
        let final_message = AssistantMessage::text("hello");
        let programmed = final_message.clone();
        let stream_fn = mock_stream_fn(move |model, _context, _options| {
            assert_eq!(model, "test-model");
            Box::new(MockAssistantStream::new(programmed.clone()))
        });

        let mut stream = stream_fn
            .call(
                "test-model".to_string(),
                empty_llm_context(),
                StreamFnOptions::default(),
                abort_channel(),
            )
            .await
            .expect("stream fn call should succeed");

        assert_eq!(stream.next_event().await, None);
        assert_eq!(stream.result().await, final_message);
    }

    #[tokio::test]
    async fn simple_text_response_produces_text_message() {
        let stream_fn = simple_text_response("hello world");
        let stream = stream_fn
            .call(
                "fake-model".to_string(),
                empty_llm_context(),
                StreamFnOptions::default(),
                abort_channel(),
            )
            .await
            .expect("stream fn call should succeed");

        let message = stream.result().await;
        assert_eq!(message.content, vec![ContentBlock::text("hello world")]);
        assert_eq!(message.stop_reason, StopReason::Stop);
    }

    #[tokio::test]
    async fn tool_use_response_produces_tool_call_message() {
        let calls = vec![ToolCall {
            id: "c1".into(),
            name: "echo".into(),
            arguments: json!({ "v": "a" }),
        }];
        let stream_fn = tool_use_response(calls.clone(), StopReason::ToolUse);
        let stream = stream_fn
            .call(
                "fake-model".to_string(),
                empty_llm_context(),
                StreamFnOptions::default(),
                abort_channel(),
            )
            .await
            .expect("stream fn call should succeed");

        let message = stream.result().await;
        assert_eq!(message.stop_reason, StopReason::ToolUse);
        assert_eq!(message.tool_calls(), calls.iter().collect::<Vec<_>>());
    }
}
