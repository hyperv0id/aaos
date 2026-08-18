use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::BoxFuture;
use serde_json::Value;

/// Accepted input modalities for a model.
///
/// Mirrors upstream `input: ("text" | "image")[]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelInput {
    Text,
    Image,
}

/// Per-million-token pricing rates.
///
/// Mirrors upstream `ModelCostRates`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ModelCost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

/// Full model metadata passed to the stream function.
///
/// Mirrors upstream `Model<TApi>`. Omits provider-adapter fields
/// (`thinkingLevelMap`, `samplingParams`, `headers`, `compat`) —
/// provider adapters are out of scope for this crate.
#[derive(Debug, Clone, PartialEq)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    pub reasoning: bool,
    pub input: Vec<ModelInput>,
    pub cost: ModelCost,
    pub context_window: u64,
    pub max_tokens: u64,
}

impl Model {
    /// Mirrors upstream `DEFAULT_MODEL` literal.
    pub fn unknown() -> Self {
        Self {
            id: "unknown".into(),
            name: "unknown".into(),
            api: "unknown".into(),
            provider: "unknown".into(),
            base_url: String::new(),
            reasoning: false,
            input: Vec::new(),
            cost: ModelCost::default(),
            context_window: 0,
            max_tokens: 0,
        }
    }
}

impl Default for Model {
    fn default() -> Self {
        Self::unknown()
    }
}

/// Reason an assistant message finished.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    Pending,
    Stop,
    Length,
    ToolUse,
    Error,
    Aborted,
    Deferred,
}

impl fmt::Display for StopReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            StopReason::Pending => "pending",
            StopReason::Stop => "stop",
            StopReason::Length => "length",
            StopReason::ToolUse => "toolUse",
            StopReason::Error => "error",
            StopReason::Aborted => "aborted",
            StopReason::Deferred => "deferred",
        };
        write!(f, "{}", s)
    }
}

/// Requested reasoning/thinking level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ThinkingLevel {
    #[default]
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

/// A block of content inside a message.
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text { text: String },
    Image { source: ImageSource },
    Thinking { text: String },
    ToolCall(ToolCall),
}

impl ContentBlock {
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text { text: text.into() }
    }

    pub fn tool_call(id: impl Into<String>, name: impl Into<String>, arguments: Value) -> Self {
        ContentBlock::ToolCall(ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        })
    }
}

/// Raw image bytes plus MIME type.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageSource {
    pub mime_type: String,
    pub bytes: Vec<u8>,
}

/// A tool call emitted by the assistant.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Token and cost usage.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Usage {
    pub input: u64,
    pub output: u64,
    pub cache_read: u64,
    pub cache_write: u64,
    pub total_tokens: u64,
    pub cost: Cost,
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cost {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}

/// A message in the agent transcript.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

impl Message {
    pub fn role(&self) -> &'static str {
        match self {
            Message::User(_) => "user",
            Message::Assistant(_) => "assistant",
            Message::ToolResult(_) => "toolResult",
        }
    }

    pub fn as_assistant(&self) -> Option<&AssistantMessage> {
        match self {
            Message::Assistant(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_user(&self) -> Option<&UserMessage> {
        match self {
            Message::User(m) => Some(m),
            _ => None,
        }
    }

    pub fn as_tool_result(&self) -> Option<&ToolResultMessage> {
        match self {
            Message::ToolResult(m) => Some(m),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct UserMessage {
    pub content: Vec<ContentBlock>,
    pub timestamp: u64,
}

impl UserMessage {
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            timestamp: now(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub model: String,
    pub provider: String,
    pub api: String,
    pub usage: Usage,
    pub error_message: Option<String>,
    pub timestamp: u64,
}

impl Default for AssistantMessage {
    fn default() -> Self {
        Self {
            content: Vec::new(),
            stop_reason: StopReason::Stop,
            model: "unknown".into(),
            provider: "unknown".into(),
            api: "unknown".into(),
            usage: Usage::default(),
            error_message: None,
            timestamp: now(),
        }
    }
}

impl AssistantMessage {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            ..Default::default()
        }
    }

    pub fn with_stop_reason(mut self, reason: StopReason) -> Self {
        self.stop_reason = reason;
        self
    }

    pub fn with_model(
        mut self,
        model: impl Into<String>,
        provider: impl Into<String>,
        api: impl Into<String>,
    ) -> Self {
        self.model = model.into();
        self.provider = provider.into();
        self.api = api.into();
        self
    }

    pub fn tool_calls(&self) -> Vec<&ToolCall> {
        self.content
            .iter()
            .filter_map(|c| match c {
                ContentBlock::ToolCall(tc) => Some(tc),
                _ => None,
            })
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    pub details: Value,
    pub usage: Option<Usage>,
    /// Names of tools introduced by this result and available from this
    /// transcript point onward. Mirrors upstream `addedToolNames`. Only
    /// present when non-empty.
    pub added_tool_names: Option<Vec<String>>,
    pub is_error: bool,
    pub timestamp: u64,
}

/// Result produced by a tool execution.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AgentToolResult {
    pub content: Vec<ContentBlock>,
    pub details: Value,
    pub usage: Option<Usage>,
    /// Names of tools introduced by this result and available from this
    /// transcript point onward. Mirrors upstream `addedToolNames`.
    pub added_tool_names: Option<Vec<String>>,
    pub terminate: bool,
}

impl AgentToolResult {
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            ..Default::default()
        }
    }

    pub fn error(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentBlock::text(text)],
            ..Default::default()
        }
    }
}

/// Callback used by tools to stream partial execution updates.
pub type AgentToolUpdateCallback = Box<dyn Fn(AgentToolResult) + Send + Sync>;

/// Tool execution strategy for calls from a single assistant message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolExecutionMode {
    #[default]
    Parallel,
    Sequential,
}

/// Tool definition used by the agent runtime.
#[async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn label(&self) -> &str;
    fn description(&self) -> &str;
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Parallel
    }

    /// Optional pre-validation argument adapter.
    fn prepare_arguments(&self, args: Value) -> Value {
        args
    }

    /// Validate arguments. Default accepts any object.
    fn validate(&self, args: &Value) -> Result<Value, String> {
        if args.is_object() {
            Ok(args.clone())
        } else {
            Err("arguments must be an object".to_string())
        }
    }

    /// Execute the tool call. Errors are converted to error tool results.
    async fn execute(
        &self,
        tool_call_id: String,
        params: Value,
        signal: Option<&tokio::sync::watch::Receiver<bool>>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String>;
}

/// Queue drain mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueueMode {
    #[default]
    OneAtATime,
    All,
}

/// Public agent state.
#[derive(Clone)]
pub struct AgentState {
    pub system_prompt: String,
    pub model: Model,
    pub thinking_level: ThinkingLevel,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<Message>,
    pub is_streaming: bool,
    pub streaming_message: Option<Message>,
    pub pending_tool_calls: std::collections::HashSet<String>,
    pub error_message: Option<String>,
}

impl fmt::Debug for AgentState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentState")
            .field("system_prompt", &self.system_prompt)
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field(
                "tools",
                &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
            )
            .field("messages", &self.messages)
            .field("is_streaming", &self.is_streaming)
            .field("streaming_message", &self.streaming_message)
            .field("pending_tool_calls", &self.pending_tool_calls)
            .field("error_message", &self.error_message)
            .finish()
    }
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            system_prompt: String::new(),
            model: Model::unknown(),
            thinking_level: ThinkingLevel::default(),
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            streaming_message: None,
            pending_tool_calls: std::collections::HashSet::new(),
            error_message: None,
        }
    }
}

/// Snapshot of context passed to the low-level agent loop.
#[derive(Clone)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl fmt::Debug for AgentContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentContext")
            .field("system_prompt", &self.system_prompt)
            .field("messages", &self.messages)
            .field(
                "tools",
                &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl AgentContext {
    pub fn empty() -> Self {
        Self {
            system_prompt: String::new(),
            messages: Vec::new(),
            tools: Vec::new(),
        }
    }
}

/// Event emitted by the agent runtime.
#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        messages: Vec<Message>,
    },
    TurnStart,
    TurnEnd {
        message: Message,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart {
        message: Message,
    },
    MessageUpdate {
        message: Message,
        assistant_event: AssistantMessageEvent,
    },
    MessageEnd {
        message: Message,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
    },
}

/// Provider stream event, mirroring upstream `AssistantMessageEvent`.
///
/// Every incremental event (text/thinking/toolcall start, delta, end) carries
/// the `contentIndex` of the block it refers to plus the full `partial`
/// assistant message as it stands at that point of the stream — the consumer
/// replaces the in-flight message wholesale instead of accumulating deltas.
///
/// Streams emit `start` before partial updates and terminate with either
/// `done` carrying the final successful message or `error` carrying the final
/// message with stop reason `Aborted`/`Error`.
#[derive(Debug, Clone, PartialEq)]
pub enum AssistantMessageEvent {
    Start {
        partial: AssistantMessage,
    },
    TextStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    TextDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    TextEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ThinkingStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ThinkingDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ThinkingEnd {
        content_index: usize,
        content: String,
        partial: AssistantMessage,
    },
    ToolCallStart {
        content_index: usize,
        partial: AssistantMessage,
    },
    ToolCallDelta {
        content_index: usize,
        delta: String,
        partial: AssistantMessage,
    },
    ToolCallEnd {
        content_index: usize,
        tool_call: ToolCall,
        partial: AssistantMessage,
    },
    /// Stream terminated successfully. `reason` is one of `Stop`, `Length`,
    /// `ToolUse`, `Deferred` (upstream `Extract<StopReason, ...>`); enforced at
    /// runtime by the stream function, not by the type system — a full nested
    /// enum would duplicate `StopReason` and force conversion at every provider
    /// boundary for no compile-time payoff (the loop never discriminates).
    Done {
        reason: StopReason,
        message: AssistantMessage,
    },
    /// Stream terminated in failure. `reason` is `Aborted` or `Error`, same
    /// runtime-guarantee approach as `Done`.
    Error {
        reason: StopReason,
        error: AssistantMessage,
    },
}

/// Options forwarded to the stream function.
#[derive(Debug, Clone, Default)]
pub struct StreamFnOptions {
    pub api_key: Option<String>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<HashMap<ThinkingLevel, u64>>,
    pub thinking_level: Option<ThinkingLevel>,
}

/// LLM-facing context built from the agent context.
#[derive(Clone)]
pub struct LlmContext {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

impl fmt::Debug for LlmContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LlmContext")
            .field("system_prompt", &self.system_prompt)
            .field("messages", &self.messages)
            .field(
                "tools",
                &self.tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
            )
            .finish()
    }
}

/// An async stream of assistant events with a final result.
#[async_trait]
pub trait AssistantEventStream: Send + Unpin {
    async fn next_event(&mut self) -> Option<AssistantMessageEvent>;
    async fn result(self: Box<Self>) -> AssistantMessage;
}

/// Provider-neutral stream function seam.
#[async_trait]
pub trait StreamFn: Send + Sync {
    async fn call(
        &self,
        model: Model,
        context: LlmContext,
        options: StreamFnOptions,
        abort: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Box<dyn AssistantEventStream>, String>;
}

/// Context passed to `before_tool_call`.
#[derive(Debug, Clone)]
pub struct BeforeToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: ToolCall,
    pub args: Value,
    pub context: AgentContext,
}

/// Result returned from `before_tool_call`.
#[derive(Debug, Clone, Default)]
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
    /// Hint that the agent should stop after the current tool batch when this
    /// call is blocked. Early termination only happens when every finalized
    /// tool result in the batch sets this to true.
    pub terminate: bool,
    /// Replacement for the validated tool arguments. Upstream mutates the
    /// validated args object in place; Rust hooks return the mutated value
    /// explicitly. When `Some`, the override REPLACES the validated args with
    /// NO revalidation: `execute` and `after_tool_call` see the override.
    pub args_override: Option<Value>,
}

/// Context passed to `after_tool_call`.
#[derive(Debug, Clone)]
pub struct AfterToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: ToolCall,
    pub args: Value,
    pub result: AgentToolResult,
    pub is_error: bool,
    pub context: AgentContext,
}

/// Result returned from `after_tool_call`. Omitted fields keep original values.
#[derive(Debug, Clone, Default)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<ContentBlock>>,
    pub details: Option<Value>,
    pub usage: Option<Usage>,
    pub is_error: Option<bool>,
    pub terminate: Option<bool>,
}

/// Context passed to `should_stop_after_turn`.
#[derive(Debug, Clone)]
pub struct ShouldStopAfterTurnContext {
    pub message: AssistantMessage,
    pub tool_results: Vec<ToolResultMessage>,
    pub context: AgentContext,
    pub new_messages: Vec<Message>,
}

pub type PrepareNextTurnContext = ShouldStopAfterTurnContext;

/// Replacement runtime state returned by `prepare_next_turn`.
#[derive(Debug, Clone, Default)]
pub struct AgentLoopTurnUpdate {
    pub context: Option<AgentContext>,
    pub model: Option<Model>,
    pub thinking_level: Option<ThinkingLevel>,
}

pub type BeforeToolCallHook = Arc<
    dyn Fn(
            BeforeToolCallContext,
            tokio::sync::watch::Receiver<bool>,
        ) -> BoxFuture<'static, Result<BeforeToolCallResult, String>>
        + Send
        + Sync,
>;

pub type AfterToolCallHook = Arc<
    dyn Fn(
            AfterToolCallContext,
            tokio::sync::watch::Receiver<bool>,
        ) -> BoxFuture<'static, Result<AfterToolCallResult, String>>
        + Send
        + Sync,
>;

pub type ShouldStopAfterTurnHook = Arc<
    dyn Fn(
            ShouldStopAfterTurnContext,
            tokio::sync::watch::Receiver<bool>,
        ) -> BoxFuture<'static, Result<bool, String>>
        + Send
        + Sync,
>;

pub type PrepareNextTurnHook = Arc<
    dyn Fn(
            PrepareNextTurnContext,
            tokio::sync::watch::Receiver<bool>,
        ) -> BoxFuture<'static, Result<Option<AgentLoopTurnUpdate>, String>>
        + Send
        + Sync,
>;

pub type GetMessagesHook =
    Arc<dyn Fn() -> BoxFuture<'static, Result<Vec<Message>, String>> + Send + Sync>;

pub type ConvertToLlm =
    Arc<dyn Fn(Vec<Message>) -> BoxFuture<'static, Result<Vec<Message>, String>> + Send + Sync>;

pub type TransformContext = Arc<
    dyn Fn(
            Vec<Message>,
            tokio::sync::watch::Receiver<bool>,
        ) -> BoxFuture<'static, Result<Vec<Message>, String>>
        + Send
        + Sync,
>;

/// Configuration for the low-level agent loop.
#[derive(Clone)]
pub struct AgentLoopConfig {
    pub model: Model,
    pub thinking_level: Option<ThinkingLevel>,
    pub tool_execution: ToolExecutionMode,
    pub before_tool_call: Option<BeforeToolCallHook>,
    pub after_tool_call: Option<AfterToolCallHook>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnHook>,
    pub prepare_next_turn: Option<PrepareNextTurnHook>,
    pub get_steering_messages: Option<GetMessagesHook>,
    pub get_follow_up_messages: Option<GetMessagesHook>,
    pub convert_to_llm: ConvertToLlm,
    pub transform_context: Option<TransformContext>,
    pub stream_fn_options: StreamFnOptions,
}

impl fmt::Debug for AgentLoopConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AgentLoopConfig")
            .field("model", &self.model)
            .field("thinking_level", &self.thinking_level)
            .field("tool_execution", &self.tool_execution)
            .field("before_tool_call", &self.before_tool_call.is_some())
            .field("after_tool_call", &self.after_tool_call.is_some())
            .field(
                "should_stop_after_turn",
                &self.should_stop_after_turn.is_some(),
            )
            .field("prepare_next_turn", &self.prepare_next_turn.is_some())
            .field(
                "get_steering_messages",
                &self.get_steering_messages.is_some(),
            )
            .field(
                "get_follow_up_messages",
                &self.get_follow_up_messages.is_some(),
            )
            .field("stream_fn_options", &self.stream_fn_options)
            .finish()
    }
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            model: Model::unknown(),
            thinking_level: None,
            tool_execution: ToolExecutionMode::default(),
            before_tool_call: None,
            after_tool_call: None,
            should_stop_after_turn: None,
            prepare_next_turn: None,
            get_steering_messages: None,
            get_follow_up_messages: None,
            convert_to_llm: Arc::new(|m| Box::pin(async move { Ok(m) })),
            transform_context: None,
            stream_fn_options: StreamFnOptions::default(),
        }
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

    #[test]
    fn stop_reason_display() {
        assert_eq!(StopReason::ToolUse.to_string(), "toolUse");
        assert_eq!(StopReason::Length.to_string(), "length");
    }

    #[test]
    fn message_roles() {
        assert_eq!(Message::User(UserMessage::new("hi")).role(), "user");
        assert_eq!(
            Message::Assistant(AssistantMessage::text("hi")).role(),
            "assistant"
        );
    }

    #[test]
    fn assistant_tool_calls_extraction() {
        let msg = AssistantMessage {
            content: vec![
                ContentBlock::text("use tools"),
                ContentBlock::tool_call("c1", "echo", serde_json::json!({"v": "a"})),
                ContentBlock::tool_call("c2", "echo", serde_json::json!({"v": "b"})),
            ],
            stop_reason: StopReason::ToolUse,
            ..Default::default()
        };
        assert_eq!(msg.tool_calls().len(), 2);
    }
}
