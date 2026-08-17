use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use futures::future::{join_all, BoxFuture};
use parking_lot::Mutex;
use serde_json::Value;
use tokio::sync::watch;
use tokio::task::JoinHandle;

use crate::types::{
    AgentContext, AgentEvent, AgentLoopConfig, AgentTool, AgentToolResult, AgentToolUpdateCallback,
    AfterToolCallContext, AssistantMessage, BeforeToolCallContext, BeforeToolCallResult,
    ContentBlock, Message, ToolCall, ToolExecutionMode, ToolResultMessage,
};

/// Async event sink used by the tool engine.
pub type EventSink = Arc<dyn Fn(AgentEvent) -> BoxFuture<'static, ()> + Send + Sync>;

/// The outcome of executing every tool call in a single assistant message.
#[derive(Debug, Clone)]
pub struct ExecutedToolBatch {
    pub messages: Vec<ToolResultMessage>,
    pub terminate: bool,
}

struct PreparedToolCall {
    tool_call: ToolCall,
    tool: Arc<dyn AgentTool>,
    args: Value,
}

enum ImmediateOutcome {
    Result(AgentToolResult, bool),
}

enum PreparedCall {
    Ready(PreparedToolCall),
    Immediate(ImmediateOutcome),
}

struct FinalizedOutcome {
    tool_call: ToolCall,
    result: AgentToolResult,
    is_error: bool,
}

/// Execute every tool call embedded in `assistant_message`.
///
/// Emits `tool_execution_start`/`tool_execution_end` for each call, and
/// `message_start`/`message_end` for each resulting `ToolResultMessage`.
/// Returns the batch of messages plus the batch terminate flag.
pub async fn execute_tool_calls(
    assistant_message: &AssistantMessage,
    context: &AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&watch::Receiver<bool>>,
    emit: &EventSink,
) -> Result<ExecutedToolBatch, String> {
    let tool_calls: Vec<&ToolCall> = assistant_message.tool_calls();
    if tool_calls.is_empty() {
        return Ok(ExecutedToolBatch {
            messages: Vec::new(),
            terminate: false,
        });
    }

    let sequential = config.tool_execution == ToolExecutionMode::Sequential
        || tool_calls.iter().any(|tc| {
            context
                .tools
                .iter()
                .any(|t| t.name() == tc.name && t.execution_mode() == ToolExecutionMode::Sequential)
        });

    if sequential {
        execute_sequential(assistant_message, context, config, signal, emit, &tool_calls).await
    } else {
        execute_parallel(assistant_message, context, config, signal, emit, &tool_calls).await
    }
}

async fn execute_sequential(
    assistant_message: &AssistantMessage,
    context: &AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&watch::Receiver<bool>>,
    emit: &EventSink,
    tool_calls: &[&ToolCall],
) -> Result<ExecutedToolBatch, String> {
    let mut finalized_calls: Vec<FinalizedOutcome> = Vec::with_capacity(tool_calls.len());
    let mut messages: Vec<ToolResultMessage> = Vec::with_capacity(tool_calls.len());

    for tool_call in tool_calls.iter().copied() {
        emit_tool_execution_start(tool_call, emit).await;

        let finalized = match prepare_tool_call(assistant_message, tool_call, context, config).await? {
            PreparedCall::Ready(prepared) => {
                let (result, is_error) =
                    execute_prepared_tool_call(&prepared, signal, emit).await;
                finalize_executed_tool_call(
                    assistant_message,
                    &prepared.tool_call,
                    &prepared.args,
                    result,
                    is_error,
                    context,
                    config,
                )
                .await?
            }
            PreparedCall::Immediate(ImmediateOutcome::Result(result, is_error)) => {
                FinalizedOutcome {
                    tool_call: tool_call.clone(),
                    result,
                    is_error,
                }
            }
        };

        emit_tool_execution_end(&finalized, emit).await;
        let message = create_tool_result_message(&finalized);
        emit_tool_result_message(&message, emit).await;
        finalized_calls.push(finalized);
        messages.push(message);
    }

    Ok(ExecutedToolBatch {
        messages,
        terminate: should_terminate_batch(&finalized_calls),
    })
}

async fn execute_parallel(
    assistant_message: &AssistantMessage,
    context: &AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&watch::Receiver<bool>>,
    emit: &EventSink,
    tool_calls: &[&ToolCall],
) -> Result<ExecutedToolBatch, String> {
    let mut finalized_entries: Vec<Option<FinalizedOutcome>> =
        (0..tool_calls.len()).map(|_| None).collect();
    let mut pending_futures = Vec::new();

    for (index, tool_call) in tool_calls.iter().copied().enumerate() {
        emit_tool_execution_start(tool_call, emit).await;

        match prepare_tool_call(assistant_message, tool_call, context, config).await? {
            PreparedCall::Ready(prepared) => {
                let fut = async move {
                    let (result, is_error) =
                        execute_prepared_tool_call(&prepared, signal, emit).await;
                    let finalized = finalize_executed_tool_call(
                        assistant_message,
                        &prepared.tool_call,
                        &prepared.args,
                        result,
                        is_error,
                        context,
                        config,
                    )
                    .await?;
                    emit_tool_execution_end(&finalized, emit).await;
                    Ok::<(usize, FinalizedOutcome), String>((index, finalized))
                };
                pending_futures.push(fut);
            }
            PreparedCall::Immediate(ImmediateOutcome::Result(result, is_error)) => {
                let finalized = FinalizedOutcome {
                    tool_call: tool_call.clone(),
                    result,
                    is_error,
                };
                emit_tool_execution_end(&finalized, emit).await;
                finalized_entries[index] = Some(finalized);
            }
        }
    }

    let completed = join_all(pending_futures)
        .await
        .into_iter()
        .collect::<Result<Vec<(usize, FinalizedOutcome)>, String>>()?;
    for (index, finalized) in completed {
        finalized_entries[index] = Some(finalized);
    }

    let mut messages: Vec<ToolResultMessage> = Vec::with_capacity(finalized_entries.len());
    let mut all_terminate = true;
    for finalized in finalized_entries.into_iter().flatten() {
        all_terminate = all_terminate && finalized.result.terminate;
        let message = create_tool_result_message(&finalized);
        emit_tool_result_message(&message, emit).await;
        messages.push(message);
    }

    let terminate = !messages.is_empty() && all_terminate;
    Ok(ExecutedToolBatch {
        messages,
        terminate,
    })
}

async fn prepare_tool_call(
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    context: &AgentContext,
    config: &AgentLoopConfig,
) -> Result<PreparedCall, String> {
    let tool = context
        .tools
        .iter()
        .find(|t| t.name() == tool_call.name)
        .cloned();

    let Some(tool) = tool else {
        return Ok(PreparedCall::Immediate(ImmediateOutcome::Result(
            create_error_tool_result(&format!("Tool {} not found", tool_call.name)),
            true,
        )));
    };

    let prepared_args = tool.prepare_arguments(tool_call.arguments.clone());
    let args = match tool.validate(&prepared_args) {
        Ok(validated) => validated,
        Err(err) => {
            return Ok(PreparedCall::Immediate(ImmediateOutcome::Result(
                create_error_tool_result(&err),
                true,
            )));
        }
    };

    if let Some(before) = &config.before_tool_call {
        let before_ctx = BeforeToolCallContext {
            assistant_message: assistant_message.clone(),
            tool_call: tool_call.clone(),
            args: args.clone(),
            context: context.clone(),
        };
        let before_result: BeforeToolCallResult = before(before_ctx).await?;
        if before_result.block {
            let reason = before_result
                .reason
                .unwrap_or_else(|| "Tool execution was blocked".to_string());
            let mut result = create_error_tool_result(&reason);
            result.terminate = before_result.terminate;
            return Ok(PreparedCall::Immediate(ImmediateOutcome::Result(
                result, true,
            )));
        }
    }

    Ok(PreparedCall::Ready(PreparedToolCall {
        tool_call: tool_call.clone(),
        tool,
        args,
    }))
}

async fn execute_prepared_tool_call(
    prepared: &PreparedToolCall,
    signal: Option<&watch::Receiver<bool>>,
    emit: &EventSink,
) -> (AgentToolResult, bool) {
    let accepting_updates = Arc::new(AtomicBool::new(true));
    let update_handles: Arc<Mutex<Vec<JoinHandle<()>>>> = Arc::new(Mutex::new(Vec::new()));

    let on_update = {
        let accepting = Arc::clone(&accepting_updates);
        let handles = Arc::clone(&update_handles);
        let tool_call_id = prepared.tool_call.id.clone();
        let tool_name = prepared.tool_call.name.clone();
        let args = prepared.tool_call.arguments.clone();
        let emit = Arc::clone(emit);
        Some(Box::new(move |partial: AgentToolResult| {
            if !accepting.load(Ordering::SeqCst) {
                return;
            }
            let event = AgentEvent::ToolExecutionUpdate {
                tool_call_id: tool_call_id.clone(),
                tool_name: tool_name.clone(),
                args: args.clone(),
                partial_result: partial,
            };
            let sink = Arc::clone(&emit);
            let handle = tokio::spawn(async move { sink(event).await });
            handles.lock().push(handle);
        }) as AgentToolUpdateCallback)
    };

    let result = prepared
        .tool
        .execute(
            prepared.tool_call.id.clone(),
            prepared.args.clone(),
            signal,
            on_update,
        )
        .await;

    accepting_updates.store(false, Ordering::SeqCst);
    let handles = std::mem::take(&mut *update_handles.lock());
    for handle in handles {
        let _ = handle.await;
    }

    match result {
        Ok(result) => (result, false),
        Err(err) => (create_error_tool_result(&err), true),
    }
}

async fn finalize_executed_tool_call(
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    args: &Value,
    mut result: AgentToolResult,
    mut is_error: bool,
    context: &AgentContext,
    config: &AgentLoopConfig,
) -> Result<FinalizedOutcome, String> {
    if let Some(after) = &config.after_tool_call {
        let after_ctx = AfterToolCallContext {
            assistant_message: assistant_message.clone(),
            tool_call: tool_call.clone(),
            args: args.clone(),
            result: result.clone(),
            is_error,
            context: context.clone(),
        };
        let after_result = after(after_ctx).await?;
        result.content = after_result.content.unwrap_or(result.content);
        result.details = after_result.details.unwrap_or(result.details);
        result.usage = after_result.usage.or(result.usage);
        if let Some(v) = after_result.is_error {
            is_error = v;
        }
        if let Some(v) = after_result.terminate {
            result.terminate = v;
        }
    }

    Ok(FinalizedOutcome {
        tool_call: tool_call.clone(),
        result,
        is_error,
    })
}

async fn emit_tool_execution_start(tool_call: &ToolCall, emit: &EventSink) {
    emit(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        args: tool_call.arguments.clone(),
    })
    .await;
}

async fn emit_tool_execution_end(finalized: &FinalizedOutcome, emit: &EventSink) {
    emit(AgentEvent::ToolExecutionEnd {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        result: finalized.result.clone(),
        is_error: finalized.is_error,
    })
    .await;
}

async fn emit_tool_result_message(message: &ToolResultMessage, emit: &EventSink) {
    emit(AgentEvent::MessageStart {
        message: Message::ToolResult(message.clone()),
    })
    .await;
    emit(AgentEvent::MessageEnd {
        message: Message::ToolResult(message.clone()),
    })
    .await;
}

fn create_tool_result_message(finalized: &FinalizedOutcome) -> ToolResultMessage {
    ToolResultMessage {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        content: finalized.result.content.clone(),
        details: finalized.result.details.clone(),
        usage: finalized.result.usage,
        is_error: finalized.is_error,
        timestamp: now(),
    }
}

fn should_terminate_batch(finalized_calls: &[FinalizedOutcome]) -> bool {
    !finalized_calls.is_empty() && finalized_calls.iter().all(|f| f.result.terminate)
}

/// Build an error tool result from a plain text message.
pub fn create_error_tool_result(message: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![ContentBlock::Text {
            text: message.to_string(),
        }],
        details: Value::Null,
        usage: None,
        terminate: false,
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct EchoTool {
        name: &'static str,
        mode: ToolExecutionMode,
    }

    #[async_trait]
    impl AgentTool for EchoTool {
        fn name(&self) -> &str {
            self.name
        }
        fn label(&self) -> &str {
            self.name
        }
        fn description(&self) -> &str {
            "echo"
        }
        fn execution_mode(&self) -> ToolExecutionMode {
            self.mode
        }
        async fn execute(
            &self,
            _tool_call_id: String,
            params: Value,
            _signal: Option<&watch::Receiver<bool>>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> Result<AgentToolResult, String> {
            let text = params
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            Ok(AgentToolResult {
                content: vec![ContentBlock::Text { text: text.to_string() }],
                details: params.clone(),
                usage: None,
                terminate: false,
            })
        }
    }

    struct ValidationTool;

    #[async_trait]
    impl AgentTool for ValidationTool {
        fn name(&self) -> &str {
            "validation_tool"
        }
        fn label(&self) -> &str {
            "validation_tool"
        }
        fn description(&self) -> &str {
            "validates"
        }
        fn validate(&self, _args: &Value) -> Result<Value, String> {
            Err("missing required field 'value'".to_string())
        }
        async fn execute(
            &self,
            _tool_call_id: String,
            _params: Value,
            _signal: Option<&watch::Receiver<bool>>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> Result<AgentToolResult, String> {
            panic!("execute should not be called when validation fails")
        }
    }

    struct BlockingHookTool;

    #[async_trait]
    impl AgentTool for BlockingHookTool {
        fn name(&self) -> &str {
            "blocking_tool"
        }
        fn label(&self) -> &str {
            "blocking_tool"
        }
        fn description(&self) -> &str {
            "blocked"
        }
        async fn execute(
            &self,
            _tool_call_id: String,
            _params: Value,
            _signal: Option<&watch::Receiver<bool>>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> Result<AgentToolResult, String> {
            panic!("execute should not be called when blocked")
        }
    }

    fn assistant_with_tool_calls(tool_calls: Vec<ToolCall>) -> AssistantMessage {
        AssistantMessage {
            content: tool_calls
                .into_iter()
                .map(ContentBlock::ToolCall)
                .collect(),
            stop_reason: crate::types::StopReason::ToolUse,
            ..Default::default()
        }
    }

    fn empty_context() -> AgentContext {
        AgentContext::empty()
    }

    fn echo_tool(name: &'static str) -> Arc<dyn AgentTool> {
        Arc::new(EchoTool {
            name,
            mode: ToolExecutionMode::Parallel,
        })
    }

    fn recording_emit() -> (EventSink, Arc<Mutex<Vec<AgentEvent>>>) {
        let events = Arc::new(Mutex::new(Vec::new()));
        let events2 = Arc::clone(&events);
        let sink: EventSink = Arc::new(move |event: AgentEvent| {
            events2.lock().push(event);
            Box::pin(async {})
        });
        (sink, events)
    }

    #[tokio::test]
    async fn successful_execution_returns_content_and_details() {
        let (emit, events) = recording_emit();
        let tool = echo_tool("echo");
        let context = AgentContext {
            tools: vec![tool],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "echo".into(),
            arguments: json!({"message": "hello"}),
        }]);

        let batch = execute_tool_calls(&assistant, &context, &AgentLoopConfig::default(), None, &emit)
            .await
            .unwrap();

        assert_eq!(batch.messages.len(), 1);
        assert_eq!(batch.messages[0].content, vec![ContentBlock::Text { text: "hello".into() }]);
        assert_eq!(batch.messages[0].details, json!({"message": "hello"}));
        assert!(!batch.messages[0].is_error);
        assert!(!batch.terminate);

        let ev = events.lock();
        assert!(matches!(ev[0], AgentEvent::ToolExecutionStart { .. }));
        assert!(matches!(ev[1], AgentEvent::ToolExecutionEnd { .. }));
        assert!(matches!(ev[2], AgentEvent::MessageStart { .. }));
        assert!(matches!(ev[3], AgentEvent::MessageEnd { .. }));
    }

    #[tokio::test]
    async fn validation_error_yields_error_result_without_executing() {
        let (emit, events) = recording_emit();
        let context = AgentContext {
            tools: vec![Arc::new(ValidationTool)],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "validation_tool".into(),
            arguments: json!({}),
        }]);

        let batch = execute_tool_calls(&assistant, &context, &AgentLoopConfig::default(), None, &emit)
            .await
            .unwrap();

        assert_eq!(batch.messages.len(), 1);
        assert!(batch.messages[0].is_error);
        assert_eq!(batch.messages[0].content.len(), 1);
        let text = match &batch.messages[0].content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("expected text block"),
        };
        assert_eq!(text, "missing required field 'value'");
        assert!(!batch.terminate);

        assert!(events.lock().iter().any(|e| matches!(
            e,
            AgentEvent::ToolExecutionEnd { is_error: true, .. }
        )));
    }

    #[tokio::test]
    async fn missing_tool_yields_error_result() {
        let (emit, _events) = recording_emit();
        let assistant = assistant_with_tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "absent".into(),
            arguments: json!({}),
        }]);

        let batch = execute_tool_calls(
            &assistant,
            &AgentContext::empty(),
            &AgentLoopConfig::default(),
            None,
            &emit,
        )
        .await
        .unwrap();

        assert_eq!(batch.messages.len(), 1);
        assert!(batch.messages[0].is_error);
        let text = match &batch.messages[0].content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("expected text block"),
        };
        assert_eq!(text, "Tool absent not found");
        assert!(!batch.terminate);
    }

    #[tokio::test]
    async fn before_tool_call_block_with_terminate() {
        let (emit, _events) = recording_emit();
        let tool = Arc::new(BlockingHookTool);
        let context = AgentContext {
            tools: vec![tool],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "blocking_tool".into(),
            arguments: json!({}),
        }]);

        let mut config = AgentLoopConfig::default();
        let blocked = Arc::new(AtomicBool::new(false));
        let blocked2 = Arc::clone(&blocked);
        config.before_tool_call = Some(Arc::new(move |_ctx| {
            blocked2.store(true, Ordering::SeqCst);
            Box::pin(async move {
                Ok(BeforeToolCallResult {
                    block: true,
                    reason: Some("policy".to_string()),
                    terminate: true,
                })
            })
        }));

        let batch = execute_tool_calls(&assistant, &context, &config, None, &emit)
            .await
            .unwrap();

        assert!(blocked.load(Ordering::SeqCst));
        assert_eq!(batch.messages.len(), 1);
        assert!(batch.messages[0].is_error);
        assert!(batch.terminate);
    }

    #[tokio::test]
    async fn after_tool_call_overrides_content_and_terminate() {
        let (emit, _events) = recording_emit();
        let tool = echo_tool("echo");
        let context = AgentContext {
            tools: vec![tool],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "echo".into(),
            arguments: json!({"message": "hello"}),
        }]);

        let mut config = AgentLoopConfig::default();
        let called = Arc::new(AtomicBool::new(false));
        let called2 = Arc::clone(&called);
        config.after_tool_call = Some(Arc::new(move |_ctx| {
            called2.store(true, Ordering::SeqCst);
            Box::pin(async move {
                Ok(crate::types::AfterToolCallResult {
                    content: Some(vec![ContentBlock::Text { text: "overridden".into() }]),
                    terminate: Some(true),
                    ..Default::default()
                })
            })
        }));

        let batch = execute_tool_calls(&assistant, &context, &config, None, &emit)
            .await
            .unwrap();

        assert!(called.load(Ordering::SeqCst));
        assert_eq!(batch.messages[0].content, vec![ContentBlock::Text { text: "overridden".into() }]);
        assert!(batch.terminate);
    }

    #[tokio::test]
    async fn parallel_completion_order_different_from_source_order() {
        use tokio::time::{sleep, Duration};

        struct DelayedTool {
            name: &'static str,
            delay_ms: u64,
            marker: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl AgentTool for DelayedTool {
            fn name(&self) -> &str {
                self.name
            }
            fn label(&self) -> &str {
                self.name
            }
            fn description(&self) -> &str {
                "delayed"
            }
            async fn execute(
                &self,
                _tool_call_id: String,
                _params: Value,
                _signal: Option<&watch::Receiver<bool>>,
                _on_update: Option<AgentToolUpdateCallback>,
            ) -> Result<AgentToolResult, String> {
                sleep(Duration::from_millis(self.delay_ms)).await;
                self.marker.fetch_add(1, Ordering::SeqCst);
                Ok(AgentToolResult::text(format!("done {}", self.name)))
            }
        }

        let order = Arc::new(AtomicUsize::new(0));
        let tool_a = Arc::new(DelayedTool {
            name: "a",
            delay_ms: 50,
            marker: Arc::clone(&order),
        });
        let tool_b = Arc::new(DelayedTool {
            name: "b",
            delay_ms: 10,
            marker: Arc::clone(&order),
        });

        let (emit, events) = recording_emit();
        let context = AgentContext {
            tools: vec![tool_a, tool_b],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![
            ToolCall {
                id: "c1".into(),
                name: "a".into(),
                arguments: json!({}),
            },
            ToolCall {
                id: "c2".into(),
                name: "b".into(),
                arguments: json!({}),
            },
        ]);

        let batch = execute_tool_calls(&assistant, &context, &AgentLoopConfig::default(), None, &emit)
            .await
            .unwrap();

        assert_eq!(batch.messages.len(), 2);
        assert_eq!(batch.messages[0].tool_call_id, "c1");
        assert_eq!(batch.messages[1].tool_call_id, "c2");

        // tool_execution_end events should reflect completion order: b then a.
        let end_events: Vec<_> = events
            .lock()
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(end_events, vec!["c2".to_string(), "c1".to_string()]);
    }
}
