use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context as TaskContext, Poll};

use futures::future::BoxFuture;
use futures::Stream;
use tokio::sync::{mpsc, watch};
use tokio::task::{JoinError, JoinHandle};

use crate::tool_engine::{create_error_tool_result, execute_tool_calls, ExecutedToolBatch};
use crate::types::{
    AgentContext, AgentEvent, AgentLoopConfig, AssistantMessage, AssistantMessageEvent, ContentBlock,
    Message, StopReason, StreamFn, ToolCall, ToolResultMessage,
};

/// A running agent loop that yields events and resolves to the new messages produced.
pub struct AgentRun {
    events: mpsc::Receiver<AgentEvent>,
    handle: Option<JoinHandle<Vec<Message>>>,
}

impl AgentRun {
    pub async fn next_event(&mut self) -> Option<AgentEvent> {
        self.events.recv().await
    }

    /// Await the spawned loop once, retaining a panic/cancellation join error
    /// instead of silently defaulting. Later calls return an empty result.
    ///
    /// # Errors
    ///
    /// Returns the spawn [`JoinError`] if the low-level loop panicked or was
    /// cancelled.
    pub async fn result(&mut self) -> Result<Vec<Message>, JoinError> {
        match self.handle.take() {
            Some(handle) => handle.await,
            None => Ok(Vec::new()),
        }
    }
}

impl Stream for AgentRun {
    type Item = AgentEvent;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut TaskContext<'_>) -> Poll<Option<Self::Item>> {
        self.events.poll_recv(cx)
    }
}

type EventSink = Arc<dyn Fn(AgentEvent) -> BoxFuture<'static, ()> + Send + Sync>;

fn create_agent_stream<F, Fut>(abort: watch::Receiver<bool>, f: F) -> AgentRun
where
    F: FnOnce(EventSink, watch::Receiver<bool>) -> Fut + Send + 'static,
    Fut: Future<Output = Vec<Message>> + Send + 'static,
{
    let (events_tx, events_rx) = mpsc::channel::<AgentEvent>(256);

    let sink: EventSink = Arc::new(move |event| {
        let tx = events_tx.clone();
        Box::pin(async move {
            let _ = tx.send(event).await;
        })
    });

    let handle = tokio::spawn(async move { f(sink, abort).await });

    AgentRun {
        events: events_rx,
        handle: Some(handle),
    }
}

/// Start an agent loop with new prompt messages.
pub fn agent_loop(
    prompts: Vec<Message>,
    context: AgentContext,
    config: AgentLoopConfig,
    abort: watch::Receiver<bool>,
    stream_fn: Arc<dyn StreamFn>,
) -> AgentRun {
    create_agent_stream(abort, move |emit, abort| async move {
        let prompts_clone = prompts.clone();
        let mut new_messages = prompts_clone.clone();
        let mut current_context = AgentContext {
            system_prompt: context.system_prompt,
            messages: [context.messages, prompts_clone].concat(),
            tools: context.tools,
        };

        emit(AgentEvent::AgentStart).await;
        emit(AgentEvent::TurnStart).await;
        for prompt in &prompts {
            emit(AgentEvent::MessageStart {
                message: prompt.clone(),
            })
            .await;
            emit(AgentEvent::MessageEnd {
                message: prompt.clone(),
            })
            .await;
        }

        run_loop(
            &mut current_context,
            &mut new_messages,
            config,
            abort,
            stream_fn,
            emit,
        )
        .await;

        new_messages
    })
}

/// Continue an agent loop from an existing context without adding user message events.
pub fn agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    abort: watch::Receiver<bool>,
    stream_fn: Arc<dyn StreamFn>,
) -> AgentRun {
    if context.messages.is_empty() {
        panic!("Cannot continue: no messages in context");
    }

    if matches!(context.messages.last(), Some(Message::Assistant(_))) {
        panic!("Cannot continue from message role: assistant");
    }

    create_agent_stream(abort, move |emit, abort| async move {
        let mut new_messages: Vec<Message> = Vec::new();
        let mut current_context = context;

        emit(AgentEvent::AgentStart).await;
        emit(AgentEvent::TurnStart).await;

        run_loop(
            &mut current_context,
            &mut new_messages,
            config,
            abort,
            stream_fn,
            emit,
        )
        .await;

        new_messages
    })
}

async fn run_loop(
    current_context: &mut AgentContext,
    new_messages: &mut Vec<Message>,
    mut config: AgentLoopConfig,
    mut abort: watch::Receiver<bool>,
    stream_fn: Arc<dyn StreamFn>,
    emit: EventSink,
) {
    let mut first_turn = true;
    let mut pending_messages: Vec<Message> = Vec::new();

    loop {
        let mut has_more_tool_calls = true;

        while has_more_tool_calls || !pending_messages.is_empty() {
            if !first_turn {
                emit(AgentEvent::TurnStart).await;
            }
            first_turn = false;

            // Inject pending steering messages before the next assistant response.
            for message in pending_messages.drain(..) {
                emit(AgentEvent::MessageStart {
                    message: message.clone(),
                })
                .await;
                emit(AgentEvent::MessageEnd {
                    message: message.clone(),
                })
                .await;
                current_context.messages.push(message.clone());
                new_messages.push(message);
            }

            let message = stream_assistant_response(
                current_context,
                &config,
                &mut abort,
                stream_fn.clone(),
                emit.clone(),
            )
            .await;

            let assistant_message = match message.as_assistant().cloned() {
                Some(am) => am,
                None => {
                    // Should not happen; treat as error stop.
                    let am = AssistantMessage {
                        content: vec![ContentBlock::text("Internal error: expected assistant message")],
                        stop_reason: StopReason::Error,
                        error_message: Some("expected assistant message".into()),
                        ..Default::default()
                    };
                    emit(AgentEvent::TurnEnd {
                        message: Message::Assistant(am.clone()),
                        tool_results: vec![],
                    })
                    .await;
                    emit(AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    })
                    .await;
                    return;
                }
            };

            new_messages.push(message.clone());

            if assistant_message.stop_reason == StopReason::Error
                || assistant_message.stop_reason == StopReason::Aborted
            {
                emit(AgentEvent::TurnEnd {
                    message: message.clone(),
                    tool_results: vec![],
                })
                .await;
                emit(AgentEvent::AgentEnd {
                    messages: new_messages.clone(),
                })
                .await;
                return;
            }

            let tool_calls = assistant_message.tool_calls();
            let mut tool_results: Vec<ToolResultMessage> = Vec::new();
            has_more_tool_calls = false;

            if !tool_calls.is_empty() {
                let batch = if assistant_message.stop_reason == StopReason::Length {
                    fail_tool_calls_from_truncated_message(
                        tool_calls.into_iter().cloned().collect(),
                        emit.clone(),
                    )
                    .await
                } else {
                    execute_tool_calls(
                        &assistant_message,
                        current_context,
                        &config,
                        Some(&abort),
                        &emit,
                    )
                    .await
                    .unwrap_or_else(|_| ExecutedToolBatch {
                        messages: vec![],
                        terminate: false,
                    })
                };

                tool_results = batch.messages;
                has_more_tool_calls = !batch.terminate;

                for result in &tool_results {
                    current_context.messages.push(Message::ToolResult(result.clone()));
                    new_messages.push(Message::ToolResult(result.clone()));
                }
            }

            emit(AgentEvent::TurnEnd {
                message: message.clone(),
                tool_results: tool_results.clone(),
            })
            .await;

            // prepare_next_turn hook
            if let Some(ref hook) = config.prepare_next_turn {
                let ctx = crate::types::PrepareNextTurnContext {
                    message: assistant_message.clone(),
                    tool_results: tool_results.clone(),
                    context: current_context.clone(),
                    new_messages: new_messages.clone(),
                };
                if let Ok(Some(update)) = hook(ctx).await {
                    if let Some(ctx) = update.context {
                        *current_context = ctx;
                    }
                    if let Some(model) = update.model {
                        config.model = model;
                    }
                    if let Some(tl) = update.thinking_level {
                        config.thinking_level = Some(tl);
                    }
                }
            }

            // should_stop_after_turn hook
            if let Some(ref hook) = config.should_stop_after_turn {
                let ctx = crate::types::ShouldStopAfterTurnContext {
                    message: assistant_message.clone(),
                    tool_results: tool_results.clone(),
                    context: current_context.clone(),
                    new_messages: new_messages.clone(),
                };
                if let Ok(true) = hook(ctx).await {
                    emit(AgentEvent::AgentEnd {
                        messages: new_messages.clone(),
                    })
                    .await;
                    return;
                }
            }

            // Poll steering messages for next turn.
            if let Some(ref hook) = config.get_steering_messages {
                if let Ok(msgs) = hook().await {
                    pending_messages = msgs;
                }
            }
        }

        // Agent would stop here. Check for follow-up messages.
        if let Some(ref hook) = config.get_follow_up_messages {
            if let Ok(msgs) = hook().await {
                if !msgs.is_empty() {
                    pending_messages = msgs;
                    continue;
                }
            }
        }

        break;
    }

    emit(AgentEvent::AgentEnd {
        messages: new_messages.clone(),
    })
    .await;
}

async fn stream_assistant_response(
    current_context: &mut AgentContext,
    config: &AgentLoopConfig,
    abort: &mut watch::Receiver<bool>,
    stream_fn: Arc<dyn StreamFn>,
    emit: EventSink,
) -> Message {
    // transform_context
    let messages = if let Some(ref hook) = config.transform_context {
        match hook(current_context.messages.clone()).await {
            Ok(msgs) => msgs,
            Err(_) => current_context.messages.clone(),
        }
    } else {
        current_context.messages.clone()
    };

    let llm_messages = (config.convert_to_llm)(messages);

    let llm_context = crate::types::LlmContext {
        system_prompt: current_context.system_prompt.clone(),
        messages: llm_messages,
        tools: current_context.tools.clone(),
    };

    let options = config.stream_fn_options.clone();

    let mut stream = match stream_fn
        .call(config.model.clone(), llm_context, options, abort.clone())
        .await
    {
        Ok(stream) => stream,
        Err(e) => {
            let am = AssistantMessage {
                content: vec![ContentBlock::text("")],
                stop_reason: StopReason::Error,
                error_message: Some(e),
                ..Default::default()
            };
            emit(AgentEvent::MessageStart {
                message: Message::Assistant(am.clone()),
            })
            .await;
            emit(AgentEvent::MessageEnd {
                message: Message::Assistant(am.clone()),
            })
            .await;
            return Message::Assistant(am);
        }
    };

    let mut partial_message: Option<AssistantMessage> = None;
    let mut added_partial = false;

    while let Some(event) = stream.next_event().await {
        match event {
            AssistantMessageEvent::Start { partial } => {
                partial_message = Some(partial.clone());
                current_context.messages.push(Message::Assistant(partial.clone()));
                added_partial = true;
                emit(AgentEvent::MessageStart {
                    message: Message::Assistant(partial),
                })
                .await;
            }
            AssistantMessageEvent::TextStart
            | AssistantMessageEvent::TextDelta { .. }
            | AssistantMessageEvent::TextEnd
            | AssistantMessageEvent::ThinkingStart
            | AssistantMessageEvent::ThinkingDelta { .. }
            | AssistantMessageEvent::ThinkingEnd
            | AssistantMessageEvent::ToolCallStart
            | AssistantMessageEvent::ToolCallDelta { .. }
            | AssistantMessageEvent::ToolCallEnd => {
                if let Some(ref partial) = partial_message {
                    if added_partial {
                        if let Some(last) = current_context.messages.last_mut() {
                            *last = Message::Assistant(partial.clone());
                        }
                    }
                    emit(AgentEvent::MessageUpdate {
                        message: Message::Assistant(partial.clone()),
                        assistant_event: event.clone(),
                    })
                    .await;
                }
            }
            AssistantMessageEvent::Done | AssistantMessageEvent::Error => {
                let final_message = stream.result().await;
                if added_partial {
                    *current_context.messages.last_mut().unwrap() =
                        Message::Assistant(final_message.clone());
                } else {
                    current_context.messages.push(Message::Assistant(final_message.clone()));
                }
                if !added_partial {
                    emit(AgentEvent::MessageStart {
                        message: Message::Assistant(final_message.clone()),
                    })
                    .await;
                }
                emit(AgentEvent::MessageEnd {
                    message: Message::Assistant(final_message.clone()),
                })
                .await;
                return Message::Assistant(final_message);
            }
        }
    }

    // Stream ended without Done/Error.
    let final_message = stream.result().await;
    if added_partial {
        *current_context.messages.last_mut().unwrap() = Message::Assistant(final_message.clone());
    } else {
        current_context.messages.push(Message::Assistant(final_message.clone()));
        emit(AgentEvent::MessageStart {
            message: Message::Assistant(final_message.clone()),
        })
        .await;
    }
    emit(AgentEvent::MessageEnd {
        message: Message::Assistant(final_message.clone()),
    })
    .await;
    Message::Assistant(final_message)
}

async fn fail_tool_calls_from_truncated_message(
    tool_calls: Vec<ToolCall>,
    emit: EventSink,
) -> ExecutedToolBatch {
    let mut messages = Vec::new();
    for tool_call in tool_calls {
        emit(AgentEvent::ToolExecutionStart {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            args: tool_call.arguments.clone(),
        })
        .await;

        let result = create_error_tool_result(&format!(
            "Tool call \"{}\" was not executed: the response hit the output token limit, so its arguments may be truncated. Re-issue the tool call with complete arguments.",
            tool_call.name
        ));
        emit(AgentEvent::ToolExecutionEnd {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            result: result.clone(),
            is_error: true,
        })
        .await;

        let tool_result = ToolResultMessage {
            tool_call_id: tool_call.id,
            tool_name: tool_call.name,
            content: result.content,
            details: result.details,
            usage: result.usage,
            is_error: true,
            timestamp: now(),
        };

        emit(AgentEvent::MessageStart {
            message: Message::ToolResult(tool_result.clone()),
        })
        .await;
        emit(AgentEvent::MessageEnd {
            message: Message::ToolResult(tool_result.clone()),
        })
        .await;

        messages.push(tool_result);
    }

    ExecutedToolBatch {
        messages,
        terminate: false,
    }
}

fn now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stream::{mock_stream_fn, simple_text_response, MockAssistantStream};
    use crate::types::{
        AgentLoopConfig, AgentTool, AgentToolResult, AssistantMessage, ContentBlock, Message,
        StopReason, UserMessage,
    };
    use async_trait::async_trait;
    use serde_json::Value;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;

    fn empty_context() -> AgentContext {
        AgentContext::empty()
    }

    fn default_config() -> AgentLoopConfig {
        AgentLoopConfig {
            model: "test-model".into(),
            provider: "test-provider".into(),
            api: "test-api".into(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn pure_text_turn_lifecycle() {
        let config = default_config();
        let stream_fn = simple_text_response("Hello");
        let mut run = agent_loop(
            vec![Message::User(UserMessage::new("hi"))],
            empty_context(),
            config,
            watch::channel(false).1,
            stream_fn,
        );
        let mut events = Vec::new();
        while let Some(event) = run.next_event().await {
            events.push(event);
        }
        let result = run.result().await.expect("run should not panic");

        assert_eq!(events[0], AgentEvent::AgentStart);
        assert_eq!(events[1], AgentEvent::TurnStart);
        assert!(matches!(events[2], AgentEvent::MessageStart { .. }));
        assert!(matches!(events[3], AgentEvent::MessageEnd { .. }));
        assert!(matches!(events[4], AgentEvent::MessageStart { .. }));
        assert!(matches!(events[5], AgentEvent::MessageEnd { .. }));
        assert!(matches!(events[6], AgentEvent::TurnEnd { .. }));
        assert_eq!(
            events[7],
            AgentEvent::AgentEnd {
                messages: result.clone()
            }
        );
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn one_tool_call_then_final_response() {
        let config = default_config();

        let counter = Arc::new(AtomicUsize::new(0));
        let counter2 = counter.clone();
        let stream_fn = mock_stream_fn(move |_model, _ctx, _opts| {
            let c = counter2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(AssistantMessage {
                    content: vec![ContentBlock::tool_call(
                        "c1",
                        "echo",
                        Value::String("hello".into()),
                    )],
                    stop_reason: StopReason::ToolUse,
                    ..Default::default()
                }))
            } else {
                Box::new(MockAssistantStream::new(
                    AssistantMessage::text("done").with_stop_reason(StopReason::Stop),
                ))
            }
        });

        struct EchoTool;
        #[async_trait]
        impl AgentTool for EchoTool {
            fn name(&self) -> &str {
                "echo"
            }
            fn label(&self) -> &str {
                "Echo"
            }
            fn description(&self) -> &str {
                "echo"
            }
            async fn execute(
                &self,
                _id: String,
                params: Value,
                _signal: Option<&watch::Receiver<bool>>,
                _on_update: Option<crate::types::AgentToolUpdateCallback>,
            ) -> Result<AgentToolResult, String> {
                Ok(AgentToolResult::text(format!(
                    "echoed: {}",
                    params.as_str().unwrap_or("")
                )))
            }
        }

        let mut context = empty_context();
        context.tools.push(Arc::new(EchoTool));

        let mut run = agent_loop(
            vec![Message::User(UserMessage::new("call echo"))],
            context,
            config,
            watch::channel(false).1,
            stream_fn,
        );
        let mut events = Vec::new();
        while let Some(event) = run.next_event().await {
            events.push(event);
        }
        let result = run.result().await.expect("run should not panic");

        let has_tool_start = events.iter().any(|e| {
            matches!(e, AgentEvent::ToolExecutionStart { tool_name, .. } if tool_name == "echo")
        });
        let has_tool_end = events.iter().any(|e| {
            matches!(e, AgentEvent::ToolExecutionEnd { tool_name, .. } if tool_name == "echo")
        });
        assert!(has_tool_start);
        assert!(has_tool_end);
        assert_eq!(result.len(), 4); // user, assistant(toolUse), toolResult, assistant(stop)
    }

    #[tokio::test]
    async fn length_truncated_tool_call_never_executes() {
        let config = default_config();

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count2 = call_count.clone();
        let stream_fn = mock_stream_fn(move |_model, _ctx, _opts| {
            let c = call_count2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(AssistantMessage {
                    content: vec![ContentBlock::tool_call(
                        "c1",
                        "echo",
                        Value::String("hel".into()),
                    )],
                    stop_reason: StopReason::Length,
                    ..Default::default()
                }))
            } else {
                Box::new(MockAssistantStream::new(AssistantMessage::text("retry")))
            }
        });

        struct EchoTool;
        #[async_trait]
        impl AgentTool for EchoTool {
            fn name(&self) -> &str { "echo" }
            fn label(&self) -> &str { "Echo" }
            fn description(&self) -> &str { "echo" }
            async fn execute(
                &self,
                _id: String,
                _params: Value,
                _signal: Option<&watch::Receiver<bool>>,
                _on_update: Option<crate::types::AgentToolUpdateCallback>,
            ) -> Result<AgentToolResult, String> {
                panic!("tool must not execute after length truncation");
            }
        }

        let mut context = empty_context();
        context.tools.push(Arc::new(EchoTool));

        let mut run = agent_loop(
            vec![Message::User(UserMessage::new("call echo"))],
            context,
            config,
            watch::channel(false).1,
            stream_fn,
        );
        let mut events = Vec::new();
        while let Some(event) = run.next_event().await {
            events.push(event);
        }
        let result = run.result().await.expect("run should not panic");

        let has_error_tool_end = events.iter().any(|e| matches!(e, AgentEvent::ToolExecutionEnd { tool_name, is_error, .. } if tool_name == "echo" && *is_error));
        assert!(has_error_tool_end);
        assert_eq!(result.len(), 4); // user, assistant(length), toolResult(error), assistant(retry)
    }

    #[tokio::test]
    async fn steering_message_injected_after_tool_batch() {
        let mut config = default_config();
        let steering = Arc::new(AtomicBool::new(false));
        let steering2 = steering.clone();
        config.get_steering_messages = Some(Arc::new(move || {
            let done = steering2.load(Ordering::SeqCst);
            let steering3 = steering.clone();
            Box::pin(async move {
                if !done {
                    steering3.store(true, Ordering::SeqCst);
                    Ok(vec![Message::User(UserMessage::new("steer"))])
                } else {
                    Ok(vec![])
                }
            })
        }));

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count2 = call_count.clone();
        let stream_fn = mock_stream_fn(move |_model, _ctx, _opts| {
            let c = call_count2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(AssistantMessage {
                    content: vec![ContentBlock::tool_call("c1", "noop", Value::Object(Default::default()))],
                    stop_reason: StopReason::ToolUse,
                    ..Default::default()
                }))
            } else {
                Box::new(MockAssistantStream::new(AssistantMessage::text("ok")))
            }
        });

        struct NoopTool;
        #[async_trait]
        impl AgentTool for NoopTool {
            fn name(&self) -> &str { "noop" }
            fn label(&self) -> &str { "Noop" }
            fn description(&self) -> &str { "noop" }
            async fn execute(&self, _id: String, _params: Value, _signal: Option<&watch::Receiver<bool>>, _on_update: Option<crate::types::AgentToolUpdateCallback>) -> Result<AgentToolResult, String> {
                Ok(AgentToolResult::text("done"))
            }
        }

        let mut context = empty_context();
        context.tools.push(Arc::new(NoopTool));

        let mut run = agent_loop(
            vec![Message::User(UserMessage::new("call noop"))],
            context,
            config,
            watch::channel(false).1,
            stream_fn,
        );
        let mut events = Vec::new();
        while let Some(event) = run.next_event().await {
            events.push(event);
        }
        let result = run.result().await.expect("run should not panic");

        let user_message_count = events.iter().filter(|e| matches!(e, AgentEvent::MessageEnd { message } if message.role() == "user")).count();
        assert_eq!(user_message_count, 2); // initial prompt + steering
        assert_eq!(result.len(), 5); // user, assistant(toolUse), toolResult, user(steer), assistant(ok)
    }

    #[tokio::test]
    async fn follow_up_message_continues_after_stop() {
        let mut config = default_config();
        let follow_up = Arc::new(AtomicBool::new(false));
        let follow_up2 = follow_up.clone();
        config.get_follow_up_messages = Some(Arc::new(move || {
            let done = follow_up2.load(Ordering::SeqCst);
            let follow_up3 = follow_up.clone();
            Box::pin(async move {
                if !done {
                    follow_up3.store(true, Ordering::SeqCst);
                    Ok(vec![Message::User(UserMessage::new("follow up"))])
                } else {
                    Ok(vec![])
                }
            })
        }));

        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count2 = call_count.clone();
        let stream_fn = mock_stream_fn(move |_model, _ctx, _opts| {
            call_count2.fetch_add(1, Ordering::SeqCst);
            Box::new(MockAssistantStream::new(AssistantMessage::text("ok")))
        });

        let mut run = agent_loop(
            vec![Message::User(UserMessage::new("hi"))],
            empty_context(),
            config,
            watch::channel(false).1,
            stream_fn,
        );
        let mut events = Vec::new();
        while let Some(event) = run.next_event().await {
            events.push(event);
        }
        let result = run.result().await.expect("run should not panic");

        let turn_starts = events.iter().filter(|e| matches!(e, AgentEvent::TurnStart)).count();
        assert_eq!(turn_starts, 2);
        assert_eq!(result.len(), 4); // user, assistant, user(followUp), assistant
    }
}

