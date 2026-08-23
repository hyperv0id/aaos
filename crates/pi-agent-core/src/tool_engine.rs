use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use futures::future::{BoxFuture, join_all};
use serde_json::Value;
use tokio::sync::watch;

use crate::types::{
    AfterToolCallContext, AgentContext, AgentEvent, AgentLoopConfig, AgentTool, AgentToolResult,
    AgentToolUpdateCallback, AssistantMessage, BeforeToolCallContext, BeforeToolCallResult,
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
        Ok(execute_sequential(
            assistant_message,
            context,
            config,
            signal,
            emit,
            &tool_calls,
        )
        .await)
    } else {
        Ok(execute_parallel(
            assistant_message,
            context,
            config,
            signal,
            emit,
            &tool_calls,
        )
        .await)
    }
}

async fn execute_sequential(
    assistant_message: &AssistantMessage,
    context: &AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&watch::Receiver<bool>>,
    emit: &EventSink,
    tool_calls: &[&ToolCall],
) -> ExecutedToolBatch {
    let mut finalized_calls: Vec<FinalizedOutcome> = Vec::with_capacity(tool_calls.len());
    let mut messages: Vec<ToolResultMessage> = Vec::with_capacity(tool_calls.len());

    for tool_call in tool_calls.iter().copied() {
        emit_tool_execution_start(tool_call, emit).await;

        let finalized =
            match prepare_tool_call(assistant_message, tool_call, context, config, signal).await {
                PreparedCall::Ready(prepared) => {
                    let (result, is_error) =
                        execute_prepared_tool_call(&prepared, signal, emit).await;
                    finalize_executed_tool_call(
                        assistant_message,
                        &prepared.tool_call,
                        &prepared.args,
                        (result, is_error),
                        context,
                        config,
                        signal,
                    )
                    .await
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

        if aborted(signal) {
            break;
        }
    }

    ExecutedToolBatch {
        messages,
        terminate: should_terminate_batch(&finalized_calls),
    }
}

async fn execute_parallel(
    assistant_message: &AssistantMessage,
    context: &AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&watch::Receiver<bool>>,
    emit: &EventSink,
    tool_calls: &[&ToolCall],
) -> ExecutedToolBatch {
    // One slot per source-order index; every call (immediate or async) lands a
    // finalized outcome here, so message events can be emitted in source order
    // after all executions complete.
    let mut finalized_entries: Vec<Option<FinalizedOutcome>> =
        (0..tool_calls.len()).map(|_| None).collect();
    let mut pending_futures = Vec::new();

    for (index, tool_call) in tool_calls.iter().copied().enumerate() {
        emit_tool_execution_start(tool_call, emit).await;

        match prepare_tool_call(assistant_message, tool_call, context, config, signal).await {
            PreparedCall::Ready(prepared) => {
                let fut = async move {
                    let (result, is_error) =
                        execute_prepared_tool_call(&prepared, signal, emit).await;
                    let finalized = finalize_executed_tool_call(
                        assistant_message,
                        &prepared.tool_call,
                        &prepared.args,
                        (result, is_error),
                        context,
                        config,
                        signal,
                    )
                    .await;
                    emit_tool_execution_end(&finalized, emit).await;
                    (index, finalized)
                };
                pending_futures.push(fut);
                if aborted(signal) {
                    break;
                }
            }
            PreparedCall::Immediate(ImmediateOutcome::Result(result, is_error)) => {
                let finalized = FinalizedOutcome {
                    tool_call: tool_call.clone(),
                    result,
                    is_error,
                };
                emit_tool_execution_end(&finalized, emit).await;
                finalized_entries[index] = Some(finalized);
                if aborted(signal) {
                    break;
                }
            }
        }
    }

    for (index, finalized) in join_all(pending_futures).await {
        finalized_entries[index] = Some(finalized);
    }

    let finalized_calls: Vec<FinalizedOutcome> = finalized_entries.into_iter().flatten().collect();
    let mut messages: Vec<ToolResultMessage> = Vec::with_capacity(finalized_calls.len());
    for finalized in &finalized_calls {
        let message = create_tool_result_message(finalized);
        emit_tool_result_message(&message, emit).await;
        messages.push(message);
    }

    ExecutedToolBatch {
        messages,
        terminate: should_terminate_batch(&finalized_calls),
    }
}

async fn prepare_tool_call(
    assistant_message: &AssistantMessage,
    tool_call: &ToolCall,
    context: &AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&watch::Receiver<bool>>,
) -> PreparedCall {
    // No abort check on entry: upstream emits `tool_execution_start` and runs
    // prepare unconditionally; an already-set signal surfaces as an
    // "Operation aborted" immediate result from the checks below.

    let tool = context
        .tools
        .iter()
        .find(|t| t.name() == tool_call.name)
        .cloned();

    let Some(tool) = tool else {
        return PreparedCall::Immediate(ImmediateOutcome::Result(
            create_error_tool_result(&format!("Tool {} not found", tool_call.name)),
            true,
        ));
    };

    let prepared_args = tool.prepare_arguments(tool_call.arguments.clone());
    let mut args = match tool.validate(&prepared_args) {
        Ok(validated) => validated,
        Err(err) => {
            return PreparedCall::Immediate(ImmediateOutcome::Result(
                create_error_tool_result(&err),
                true,
            ));
        }
    };

    if let Some(before) = &config.before_tool_call {
        let hook_signal = signal.cloned().unwrap_or_else(|| watch::channel(false).1);
        let before_ctx = BeforeToolCallContext {
            assistant_message: assistant_message.clone(),
            tool_call: tool_call.clone(),
            args: args.clone(),
            context: context.clone(),
        };
        // A failing hook becomes an immediate error result for this call so the
        // engine still emits a paired tool_execution_end and keeps going (Pi
        // converts hook errors into error tool results instead of throwing).
        let before_result: BeforeToolCallResult = match before(before_ctx, hook_signal).await {
            Ok(result) => result,
            Err(err) => {
                return PreparedCall::Immediate(ImmediateOutcome::Result(
                    create_error_tool_result(&err),
                    true,
                ));
            }
        };
        // The hook may return replacement args. Upstream mutates the validated
        // args object in place; the returned override replaces them with no
        // revalidation, matching the observable contract.
        if let Some(override_args) = before_result.args_override {
            args = override_args;
        }
        if aborted(signal) {
            return PreparedCall::Immediate(ImmediateOutcome::Result(
                create_error_tool_result("Operation aborted"),
                true,
            ));
        }
        if before_result.block {
            let reason = before_result
                .reason
                .unwrap_or_else(|| "Tool execution was blocked".to_string());
            let mut result = create_error_tool_result(&reason);
            result.terminate = before_result.terminate;
            return PreparedCall::Immediate(ImmediateOutcome::Result(result, true));
        }
    }
    if aborted(signal) {
        return PreparedCall::Immediate(ImmediateOutcome::Result(
            create_error_tool_result("Operation aborted"),
            true,
        ));
    }

    PreparedCall::Ready(PreparedToolCall {
        tool_call: tool_call.clone(),
        tool,
        args,
    })
}

async fn execute_prepared_tool_call(
    prepared: &PreparedToolCall,
    signal: Option<&watch::Receiver<bool>>,
    emit: &EventSink,
) -> (AgentToolResult, bool) {
    // Race-free update delivery: the callback pushes events into an unbounded
    // channel; after execute settles, accepting_updates is set to false (no
    // new callbacks pass the gate), then all queued events are drained and
    // emitted in push order. No JoinHandles, no spawn, no check-then-spawn gap.
    let accepting_updates = Arc::new(AtomicBool::new(true));
    let (update_tx, mut update_rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();

    let on_update = {
        let accepting = Arc::clone(&accepting_updates);
        let tool_call_id = prepared.tool_call.id.clone();
        let tool_name = prepared.tool_call.name.clone();
        let args = prepared.tool_call.arguments.clone();
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
            let _ = update_tx.send(event);
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

    // No new callbacks can pass the gate after this store.
    accepting_updates.store(false, Ordering::SeqCst);

    // Drain all events that passed the check, in push (FIFO) order.
    while let Ok(event) = update_rx.try_recv() {
        emit(event).await;
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
    outcome: (AgentToolResult, bool),
    context: &AgentContext,
    config: &AgentLoopConfig,
    signal: Option<&watch::Receiver<bool>>,
) -> FinalizedOutcome {
    let (mut result, mut is_error) = outcome;
    if let Some(after) = &config.after_tool_call {
        let hook_signal = signal.cloned().unwrap_or_else(|| watch::channel(false).1);
        let after_ctx = AfterToolCallContext {
            assistant_message: assistant_message.clone(),
            tool_call: tool_call.clone(),
            args: args.clone(),
            result: result.clone(),
            is_error,
            context: context.clone(),
        };
        // A failing hook replaces the outcome with an error result so the call
        // still emits a paired tool_execution_end (Pi behavior).
        match after(after_ctx, hook_signal).await {
            Ok(after_result) => {
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
            Err(err) => {
                result = create_error_tool_result(&err);
                is_error = true;
            }
        }
    }

    FinalizedOutcome {
        tool_call: tool_call.clone(),
        result,
        is_error,
    }
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
    // Mirror upstream: include addedToolNames only when non-empty.
    let added_tool_names = finalized
        .result
        .added_tool_names
        .as_ref()
        .filter(|names| !names.is_empty())
        .cloned();
    ToolResultMessage {
        tool_call_id: finalized.tool_call.id.clone(),
        tool_name: finalized.tool_call.name.clone(),
        content: finalized.result.content.clone(),
        details: finalized.result.details.clone(),
        usage: finalized.result.usage,
        added_tool_names,
        is_error: finalized.is_error,
        timestamp: now(),
    }
}

fn should_terminate_batch(finalized_calls: &[FinalizedOutcome]) -> bool {
    !finalized_calls.is_empty() && finalized_calls.iter().all(|f| f.result.terminate)
}

fn aborted(signal: Option<&watch::Receiver<bool>>) -> bool {
    signal.is_some_and(|s| *s.borrow())
}

/// Build an error tool result from a plain text message.
pub fn create_error_tool_result(message: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![ContentBlock::Text {
            text: message.to_string(),
        }],
        details: serde_json::json!({}),
        usage: None,
        added_tool_names: None,
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
    use std::sync::Mutex;
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
            let text = params.get("message").and_then(|v| v.as_str()).unwrap_or("");
            Ok(AgentToolResult {
                content: vec![ContentBlock::Text {
                    text: text.to_string(),
                }],
                details: params.clone(),
                usage: None,
                added_tool_names: None,
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
            content: tool_calls.into_iter().map(ContentBlock::ToolCall).collect(),
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
            events2.lock().unwrap().push(event);
            Box::pin(async {})
        });
        (sink, events)
    }

    #[tokio::test]
    async fn successful_execution_returns_content() {
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

        let batch = execute_tool_calls(
            &assistant,
            &context,
            &AgentLoopConfig::default(),
            None,
            &emit,
        )
        .await
        .unwrap();

        assert_eq!(batch.messages.len(), 1);
        assert_eq!(
            batch.messages[0].content,
            vec![ContentBlock::Text {
                text: "hello".into()
            }]
        );
        assert_eq!(batch.messages[0].details, json!({"message": "hello"}));
        assert!(!batch.messages[0].is_error);
        assert!(!batch.terminate);

        let ev = events.lock().unwrap();
        assert!(matches!(ev[0], AgentEvent::ToolExecutionStart { .. }));
        assert!(matches!(ev[1], AgentEvent::ToolExecutionEnd { .. }));
        assert!(matches!(ev[2], AgentEvent::MessageStart { .. }));
        assert!(matches!(ev[3], AgentEvent::MessageEnd { .. }));
    }

    #[tokio::test]
    async fn validation_error_skips_execution() {
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

        let batch = execute_tool_calls(
            &assistant,
            &context,
            &AgentLoopConfig::default(),
            None,
            &emit,
        )
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

        assert!(
            events
                .lock()
                .unwrap()
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { is_error: true, .. }))
        );
    }

    struct SchemaRequiredTool;

    #[async_trait]
    impl AgentTool for SchemaRequiredTool {
        fn name(&self) -> &str {
            "schema_tool"
        }
        fn label(&self) -> &str {
            "schema_tool"
        }
        fn description(&self) -> &str {
            "requires value"
        }
        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "value": { "type": "string" } },
                "required": ["value"]
            })
        }
        async fn execute(
            &self,
            _tool_call_id: String,
            _params: Value,
            _signal: Option<&watch::Receiver<bool>>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> Result<AgentToolResult, String> {
            panic!("execute should not run when schema fails");
        }
    }

    #[tokio::test]
    async fn schema_required_field_skips_execution() {
        let (emit, events) = recording_emit();
        let context = AgentContext {
            tools: vec![Arc::new(SchemaRequiredTool)],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "schema_tool".into(),
            arguments: json!({}),
        }]);
        let batch = execute_tool_calls(
            &assistant,
            &context,
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
            _ => panic!("expected text"),
        };
        assert!(
            text.to_lowercase().contains("value"),
            "schema error should mention missing field, got {text}"
        );
        assert!(
            events
                .lock()
                .unwrap()
                .iter()
                .any(|e| matches!(e, AgentEvent::ToolExecutionEnd { is_error: true, .. }))
        );
    }

    #[tokio::test]
    async fn schema_valid_object_reaches_execute() {
        struct OkTool;
        #[async_trait]
        impl AgentTool for OkTool {
            fn name(&self) -> &str {
                "schema_tool"
            }
            fn label(&self) -> &str {
                "schema_tool"
            }
            fn description(&self) -> &str {
                "ok"
            }
            fn parameters(&self) -> Value {
                json!({
                    "type": "object",
                    "properties": { "value": { "type": "string" } },
                    "required": ["value"]
                })
            }
            async fn execute(
                &self,
                _id: String,
                params: Value,
                _signal: Option<&watch::Receiver<bool>>,
                _on_update: Option<AgentToolUpdateCallback>,
            ) -> Result<AgentToolResult, String> {
                Ok(AgentToolResult::text(
                    params["value"].as_str().unwrap_or("").to_string(),
                ))
            }
        }
        let (emit, _) = recording_emit();
        let context = AgentContext {
            tools: vec![Arc::new(OkTool)],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "schema_tool".into(),
            arguments: json!({"value": "ok"}),
        }]);
        let batch = execute_tool_calls(
            &assistant,
            &context,
            &AgentLoopConfig::default(),
            None,
            &emit,
        )
        .await
        .unwrap();
        assert!(!batch.messages[0].is_error);
        match &batch.messages[0].content[0] {
            ContentBlock::Text { text } => assert_eq!(text, "ok"),
            _ => panic!("expected text"),
        }
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
        config.before_tool_call = Some(Arc::new(move |_ctx, _signal| {
            blocked2.store(true, Ordering::SeqCst);
            Box::pin(async move {
                Ok(BeforeToolCallResult {
                    block: true,
                    reason: Some("policy".to_string()),
                    terminate: true,
                    ..Default::default()
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
    async fn after_hook_overrides_content_and_terminate() {
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
        config.after_tool_call = Some(Arc::new(move |_ctx, _signal| {
            called2.store(true, Ordering::SeqCst);
            Box::pin(async move {
                Ok(crate::types::AfterToolCallResult {
                    content: Some(vec![ContentBlock::Text {
                        text: "overridden".into(),
                    }]),
                    terminate: Some(true),
                    ..Default::default()
                })
            })
        }));

        let batch = execute_tool_calls(&assistant, &context, &config, None, &emit)
            .await
            .unwrap();

        assert!(called.load(Ordering::SeqCst));
        assert_eq!(
            batch.messages[0].content,
            vec![ContentBlock::Text {
                text: "overridden".into()
            }]
        );
        assert!(batch.terminate);
    }

    #[tokio::test]
    async fn parallel_completion_order_vs_source() {
        use tokio::time::{Duration, sleep};

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

        let batch = execute_tool_calls(
            &assistant,
            &context,
            &AgentLoopConfig::default(),
            None,
            &emit,
        )
        .await
        .unwrap();

        assert_eq!(batch.messages.len(), 2);
        assert_eq!(batch.messages[0].tool_call_id, "c1");
        assert_eq!(batch.messages[1].tool_call_id, "c2");

        // tool_execution_end events should reflect completion order: b then a.
        let end_events: Vec<_> = events
            .lock()
            .unwrap()
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(end_events, vec!["c2".to_string(), "c1".to_string()]);
    }

    struct FailingTool;

    #[async_trait]
    impl AgentTool for FailingTool {
        fn name(&self) -> &str {
            "failing_tool"
        }
        fn label(&self) -> &str {
            "failing_tool"
        }
        fn description(&self) -> &str {
            "always fails"
        }
        async fn execute(
            &self,
            _tool_call_id: String,
            _params: Value,
            _signal: Option<&watch::Receiver<bool>>,
            _on_update: Option<AgentToolUpdateCallback>,
        ) -> Result<AgentToolResult, String> {
            Err("execution exploded".to_string())
        }
    }

    #[tokio::test]
    async fn before_hook_error_continues_batch() {
        let (emit, events) = recording_emit();
        let context = AgentContext {
            tools: vec![echo_tool("echo")],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![
            ToolCall {
                id: "c1".into(),
                name: "echo".into(),
                arguments: json!({"message": "first"}),
            },
            ToolCall {
                id: "c2".into(),
                name: "echo".into(),
                arguments: json!({"message": "second"}),
            },
        ]);

        let mut config = AgentLoopConfig::default();
        let hook_calls = Arc::new(AtomicUsize::new(0));
        let hook_calls2 = Arc::clone(&hook_calls);
        config.before_tool_call = Some(Arc::new(move |_ctx, _signal| {
            let n = hook_calls2.fetch_add(1, Ordering::SeqCst);
            if n == 0 {
                Box::pin(async move {
                    Err::<BeforeToolCallResult, String>("before hook failed".to_string())
                })
            } else {
                Box::pin(async move { Ok(BeforeToolCallResult::default()) })
            }
        }));

        let batch = execute_tool_calls(&assistant, &context, &config, None, &emit)
            .await
            .unwrap();

        // The failing hook becomes an error result for c1; c2 still executes.
        assert_eq!(batch.messages.len(), 2);
        assert!(batch.messages[0].is_error);
        let text = match &batch.messages[0].content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("expected text block"),
        };
        assert_eq!(text, "before hook failed");
        assert!(!batch.messages[1].is_error);
        assert!(!batch.terminate);

        let ev = events.lock().unwrap();
        let starts = ev
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolExecutionStart { .. }))
            .count();
        let ends = ev
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
            .count();
        assert_eq!(starts, 2);
        assert_eq!(ends, 2);
    }

    #[tokio::test]
    async fn after_hook_error_replaces_result() {
        let (emit, events) = recording_emit();
        let context = AgentContext {
            tools: vec![echo_tool("echo")],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "echo".into(),
            arguments: json!({"message": "hello"}),
        }]);

        let config = AgentLoopConfig {
            after_tool_call: Some(Arc::new(|_ctx, _signal| {
                Box::pin(async move {
                    Err::<crate::types::AfterToolCallResult, String>(
                        "after hook failed".to_string(),
                    )
                })
            })),
            ..AgentLoopConfig::default()
        };

        let batch = execute_tool_calls(&assistant, &context, &config, None, &emit)
            .await
            .unwrap();

        // The failing hook replaces the executed result with an error result.
        assert_eq!(batch.messages.len(), 1);
        assert!(batch.messages[0].is_error);
        let text = match &batch.messages[0].content[0] {
            ContentBlock::Text { text } => text.clone(),
            _ => panic!("expected text block"),
        };
        assert_eq!(text, "after hook failed");
        assert!(!batch.terminate);

        let ev = events.lock().unwrap();
        let starts = ev
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolExecutionStart { .. }))
            .count();
        let ends = ev
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
            .count();
        assert_eq!(starts, 1);
        assert_eq!(ends, 1);
    }

    #[tokio::test]
    async fn tool_execute_error() {
        let (emit, events) = recording_emit();
        let context = AgentContext {
            tools: vec![Arc::new(FailingTool)],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![ToolCall {
            id: "c1".into(),
            name: "failing_tool".into(),
            arguments: json!({}),
        }]);

        let batch = execute_tool_calls(
            &assistant,
            &context,
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
        assert_eq!(text, "execution exploded");
        assert!(!batch.terminate);

        let ev = events.lock().unwrap();
        let starts = ev
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolExecutionStart { .. }))
            .count();
        let ends = ev
            .iter()
            .filter(|e| matches!(e, AgentEvent::ToolExecutionEnd { .. }))
            .count();
        assert_eq!(starts, 1);
        assert_eq!(ends, 1);
    }

    #[tokio::test]
    async fn sequential_tool_enforces_order() {
        use tokio::time::{Duration, sleep};

        struct DelayTool {
            name: &'static str,
            mode: ToolExecutionMode,
            delay_ms: u64,
        }

        #[async_trait]
        impl AgentTool for DelayTool {
            fn name(&self) -> &str {
                self.name
            }
            fn label(&self) -> &str {
                self.name
            }
            fn description(&self) -> &str {
                "delayed"
            }
            fn execution_mode(&self) -> ToolExecutionMode {
                self.mode
            }
            async fn execute(
                &self,
                _tool_call_id: String,
                _params: Value,
                _signal: Option<&watch::Receiver<bool>>,
                _on_update: Option<AgentToolUpdateCallback>,
            ) -> Result<AgentToolResult, String> {
                sleep(Duration::from_millis(self.delay_ms)).await;
                Ok(AgentToolResult::text(format!("done {}", self.name)))
            }
        }

        let (emit, events) = recording_emit();
        let context = AgentContext {
            tools: vec![
                Arc::new(DelayTool {
                    name: "slow",
                    mode: ToolExecutionMode::Parallel,
                    delay_ms: 50,
                }),
                Arc::new(DelayTool {
                    name: "strict",
                    mode: ToolExecutionMode::Sequential,
                    delay_ms: 1,
                }),
            ],
            ..empty_context()
        };
        let assistant = assistant_with_tool_calls(vec![
            ToolCall {
                id: "c1".into(),
                name: "slow".into(),
                arguments: json!({}),
            },
            ToolCall {
                id: "c2".into(),
                name: "strict".into(),
                arguments: json!({}),
            },
        ]);

        // Config defaults to Parallel; the Sequential tool must force the
        // whole batch to run sequentially.
        let batch = execute_tool_calls(
            &assistant,
            &context,
            &AgentLoopConfig::default(),
            None,
            &emit,
        )
        .await
        .unwrap();

        assert_eq!(batch.messages.len(), 2);
        assert_eq!(batch.messages[0].tool_call_id, "c1");
        assert_eq!(batch.messages[1].tool_call_id, "c2");

        // Sequential: c1 fully finishes (including its end event) before c2
        // even starts.
        let ev = events.lock().unwrap();
        let start_c2 = ev
            .iter()
            .position(|e| {
                matches!(
                    e,
                    AgentEvent::ToolExecutionStart { tool_call_id, .. } if tool_call_id == "c2"
                )
            })
            .unwrap();
        assert!(ev.iter().take(start_c2).any(|e| {
            matches!(
                e,
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } if tool_call_id == "c1"
            )
        }));
        let end_events: Vec<_> = ev
            .iter()
            .filter_map(|e| match e {
                AgentEvent::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(end_events, vec!["c1".to_string(), "c2".to_string()]);
    }
}
