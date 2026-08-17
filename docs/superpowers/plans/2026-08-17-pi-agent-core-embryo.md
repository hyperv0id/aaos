# Pi Agent Core Embryo — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust clean-room embryo of Pi's `packages/agent` kernel (`Agent`, `agentLoop`, event lifecycle, tool execution, queues, abort) with deterministic fake-provider acceptance tests that prove behavioral 1:1 equivalence via normalized traces.

**Architecture:** A single crate `crates/pi-agent-core` exposes provider-neutral message/event types, an `Agent` state machine, and low-level `agent_loop` / `agent_loop_continue` functions. Tool execution is async and supports parallel/sequential modes. A `StreamFn` seam accepts a fake in-memory provider for tests. A `TraceCollector` records normalized events/state snapshots so acceptance tests compare behavior, not timestamps or object identity.

**Tech Stack:** Rust 1.80+, `tokio` (async runtime & test harness), `futures` (`Stream` trait), `serde` (optional, for snapshots). No external JSON-schema validator in this embryo; validation is structural/manual.

**Spec:** `.scratch/pi-rust-replica-research-corrected.md` (the corrected research note that supersedes the earlier sidecar conclusion).

## Global Constraints

- Kernel only: replicate `packages/agent` (`Agent`, `agent-loop.ts`, `types.ts`) and provider-neutral parts of `packages/ai`. Do **not** implement real HTTP providers, OAuth, model registry, compaction, session persistence, skills, extensions, or TUI rendering.
- Behavioral/trace equivalence: same deterministic input sequence must produce the same normalized event order, message roles, stop reasons, tool invocation behavior, and lifecycle barriers.
- No Node/Bun runtime dependency in production code; tests may use `tokio`.
- All public types/functions must be `Send` where Pi expects them to cross async boundaries.
- `agent_loop` / `agent_loop_continue` must never throw outside a normal event lifecycle; provider/runtime failures become assistant messages with `stop_reason: Error` or `Aborted` and a complete `agent_end` event.
- `isStreaming`-equivalent barrier: the run is idle only after `agent_end` listeners settle.
- Abort signal must propagate to `StreamFn`, hooks, tool execution, and event listeners.
- Tool execution default is parallel; any tool with `execution_mode: Sequential` forces the whole batch to sequential.
- `length`-truncated assistant messages must not execute any tool calls; emit error tool results and continue.
- `beforeToolCall` runs after validation; `afterToolCall` replaces fields field-by-field with no deep merge.
- Steering messages inject after the current assistant turn's tool calls complete; follow-up messages run only when the agent would otherwise stop.
- Queue drain modes: `"all"` drains every queued message at the drain point; `"one_at_a_time"` drains only the oldest.
- No placeholders, no speculative abstractions. Lean code; prefer stdlib/`tokio`/`futures` over new crates.

---

## File Map

| File | Responsibility |
|---|---|
| `Cargo.toml` | Workspace manifest (single crate workspace for future expansion) |
| `crates/pi-agent-core/Cargo.toml` | Crate manifest |
| `crates/pi-agent-core/src/lib.rs` | Re-exports |
| `crates/pi-agent-core/src/types.rs` | Messages, content blocks, events, state, tool trait, config, stop reasons |
| `crates/pi-agent-core/src/stream.rs` | `StreamFn` seam, `AssistantEventStream` trait, fake in-memory provider |
| `crates/pi-agent-core/src/tool_engine.rs` | Tool argument preparation, validation, `beforeToolCall`/`afterToolCall`, parallel/sequential execution |
| `crates/pi-agent-core/src/agent_loop.rs` | `agent_loop`, `agent_loop_continue`, inner/outer loops, queue draining, lifecycle events |
| `crates/pi-agent-core/src/agent.rs` | `Agent` wrapper, `prompt`/`continue`/`steer`/`follow_up`/`abort`/`reset`/`wait_for_idle` |
| `crates/pi-agent-core/src/trace.rs` | `TraceCollector` and normalized snapshot helpers |
| `crates/pi-agent-core/tests/acceptance.rs` | 12 acceptance tests matching the research spec matrix |

---

## Task Dependency Graph

| Task | Depends On | Produces Files |
|---|---|---|
| 1. Types & scaffold | — | `Cargo.toml`, crate `Cargo.toml`, `lib.rs`, `types.rs` |
| 2. StreamFn seam & fake provider | Task 1 | `stream.rs` |
| 3. Tool execution engine | Task 1 | `tool_engine.rs` |
| 4. Agent loop | Task 2, Task 3 | `agent_loop.rs` |
| 5. Agent wrapper | Task 4 | `agent.rs` |
| 6. Trace collector & acceptance tests | Task 5 | `trace.rs`, `tests/acceptance.rs` |

---

## Task 1: Cargo Scaffold and Core Types

**Files:**
- Create: `Cargo.toml`
- Create: `crates/pi-agent-core/Cargo.toml`
- Create: `crates/pi-agent-core/src/lib.rs`
- Create: `crates/pi-agent-core/src/types.rs`

**Interfaces:**
- Produces: `Message`, `AssistantMessage`, `UserMessage`, `ToolResultMessage`, `ContentBlock`, `StopReason`, `ThinkingLevel`, `Usage`, `AgentEvent`, `AgentState`, `AgentContext`, `AgentTool`, `AgentToolResult`, `AgentLoopConfig`, `QueueMode`, `ToolExecutionMode`, `BeforeToolCallContext`, `AfterToolCallContext`.

- [ ] **Step 1: Create workspace `Cargo.toml`**

```toml
[workspace]
members = ["crates/pi-agent-core"]
resolver = "2"
```

- [ ] **Step 2: Create crate `crates/pi-agent-core/Cargo.toml`**

```toml
[package]
name = "pi-agent-core"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["sync", "rt", "macros", "time"] }
futures = "0.3"
async-trait = "0.1"
serde = { version = "1", features = ["derive"], optional = true }

[dev-dependencies]
tokio-test = "0.4"
```

- [ ] **Step 3: Define stop reasons and thinking level in `types.rs`**

```rust
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
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
```

- [ ] **Step 4: Define content blocks in `types.rs`**

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ContentBlock {
    Text { text: String },
    Image { source: ImageSource }, // keep minimal: bytes + mime type
    Thinking { text: String },
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value, // use json::Value for raw args
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImageSource {
    pub mime_type: String,
    pub bytes: Vec<u8>,
}
```

- [ ] **Step 5: Define messages in `types.rs`**

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    User(UserMessage),
    Assistant(AssistantMessage),
    ToolResult(ToolResultMessage),
}

#[derive(Debug, Clone, PartialEq)]
pub struct UserMessage {
    pub content: Vec<ContentBlock>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssistantMessage {
    pub content: Vec<ContentBlock>,
    pub stop_reason: StopReason,
    pub model: String,      // provider model id
    pub provider: String,
    pub api: String,
    pub usage: Usage,
    pub error_message: Option<String>,
    pub timestamp: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    pub details: serde_json::Value,
    pub usage: Option<Usage>,
    pub is_error: bool,
    pub timestamp: u64,
}
```

- [ ] **Step 6: Define `Usage` in `types.rs`**

```rust
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
```

- [ ] **Step 7: Define events in `types.rs`**

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd { messages: Vec<Message> },
    TurnStart,
    TurnEnd {
        message: Message,
        tool_results: Vec<ToolResultMessage>,
    },
    MessageStart { message: Message },
    MessageUpdate {
        message: Message,
        assistant_event: AssistantMessageEvent,
    },
    MessageEnd { message: Message },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionUpdate {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
        partial_result: AgentToolResult,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result: AgentToolResult,
        is_error: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssistantMessageEvent {
    Start { partial: AssistantMessage },
    TextStart,
    TextDelta { text: String },
    TextEnd,
    ThinkingStart,
    ThinkingDelta { text: String },
    ThinkingEnd,
    ToolCallStart,
    ToolCallDelta { text: String },
    ToolCallEnd,
    Done,
    Error,
}
```

- [ ] **Step 8: Define tool trait and result in `types.rs`**

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct AgentToolResult {
    pub content: Vec<ContentBlock>,
    pub details: serde_json::Value,
    pub usage: Option<Usage>,
    pub terminate: bool,
}

pub type AgentToolUpdateCallback = Box<dyn Fn(AgentToolResult) + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolExecutionMode {
    #[default]
    Parallel,
    Sequential,
}

#[async_trait::async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn label(&self) -> &str;
    fn description(&self) -> &str;
    fn execution_mode(&self) -> ToolExecutionMode { ToolExecutionMode::Parallel }

    /// Optional pre-validation argument adapter. Must return args shaped for validation.
    fn prepare_arguments(&self, args: serde_json::Value) -> serde_json::Value { args }

    /// Validate arguments. Default checks that `args` is an object.
    fn validate(&self, args: &serde_json::Value) -> Result<serde_json::Value, String> {
        if args.is_object() { Ok(args.clone()) } else { Err("arguments must be an object".into()) }
    }

    async fn execute(
        &self,
        tool_call_id: String,
        params: serde_json::Value,
        signal: Option<&tokio::sync::watch::Receiver<bool>>,
        on_update: Option<AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String>;
}
```

Use `async-trait` crate or switch to `Pin<Box<dyn Future>>` if you prefer no extra crate; add `async-trait` to `Cargo.toml` if used.

- [ ] **Step 9: Define state, context, config, queue mode in `types.rs`**

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QueueMode {
    #[default]
    OneAtATime,
    All,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentState {
    pub system_prompt: String,
    pub model: String,
    pub provider: String,
    pub api: String,
    pub thinking_level: ThinkingLevel,
    pub tools: Vec<Arc<dyn AgentTool>>,
    pub messages: Vec<Message>,
    pub is_streaming: bool,
    pub streaming_message: Option<Message>,
    pub pending_tool_calls: std::collections::HashSet<String>,
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentContext {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    pub model: String,
    pub provider: String,
    pub api: String,
    pub thinking_level: Option<ThinkingLevel>,
    pub tool_execution: ToolExecutionMode,
    pub before_tool_call: Option<Arc<dyn Fn(BeforeToolCallContext) -> futures::future::BoxFuture<'static, Result<BeforeToolCallResult, String>> + Send + Sync>>,
    pub after_tool_call: Option<Arc<dyn Fn(AfterToolCallContext) -> futures::future::BoxFuture<'static, Result<AfterToolCallResult, String>> + Send + Sync>>,
    pub should_stop_after_turn: Option<Arc<dyn Fn(ShouldStopAfterTurnContext) -> futures::future::BoxFuture<'static, Result<bool, String>> + Send + Sync>>,
    pub prepare_next_turn: Option<Arc<dyn Fn(PrepareNextTurnContext) -> futures::future::BoxFuture<'static, Result<Option<AgentLoopTurnUpdate>, String>> + Send + Sync>>,
    pub get_steering_messages: Option<Arc<dyn Fn() -> futures::future::BoxFuture<'static, Result<Vec<Message>, String>> + Send + Sync>>,
    pub get_follow_up_messages: Option<Arc<dyn Fn() -> futures::future::BoxFuture<'static, Result<Vec<Message>, String>> + Send + Sync>>,
    pub convert_to_llm: Arc<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>,
    pub transform_context: Option<Arc<dyn Fn(Vec<Message>) -> futures::future::BoxFuture<'static, Result<Vec<Message>, String>> + Send + Sync>>,
}

#[derive(Debug, Clone)]
pub struct BeforeToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: ToolCall,
    pub args: serde_json::Value,
    pub context: AgentContext,
}

#[derive(Debug, Clone, Default)]
pub struct BeforeToolCallResult {
    pub block: bool,
    pub reason: Option<String>,
    pub terminate: bool,
}

#[derive(Debug, Clone)]
pub struct AfterToolCallContext {
    pub assistant_message: AssistantMessage,
    pub tool_call: ToolCall,
    pub args: serde_json::Value,
    pub result: AgentToolResult,
    pub is_error: bool,
    pub context: AgentContext,
}

#[derive(Debug, Clone, Default)]
pub struct AfterToolCallResult {
    pub content: Option<Vec<ContentBlock>>,
    pub details: Option<serde_json::Value>,
    pub usage: Option<Usage>,
    pub is_error: Option<bool>,
    pub terminate: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct ShouldStopAfterTurnContext {
    pub message: AssistantMessage,
    pub tool_results: Vec<ToolResultMessage>,
    pub context: AgentContext,
    pub new_messages: Vec<Message>,
}

pub type PrepareNextTurnContext = ShouldStopAfterTurnContext;

#[derive(Debug, Clone)]
pub struct AgentLoopTurnUpdate {
    pub context: Option<AgentContext>,
    pub model: Option<String>,
    pub thinking_level: Option<ThinkingLevel>,
}
```

Note: `BoxFuture` and `Arc<dyn AgentTool>` require `Send + Sync`. Make sure all trait bounds are consistent.

- [ ] **Step 10: Create `lib.rs` re-exports**

```rust
pub mod agent;
pub mod agent_loop;
pub mod stream;
pub mod tool_engine;
pub mod trace;
pub mod types;
```

- [ ] **Step 11: Build to verify types compile**

Run: `cargo build -p pi-agent-core`
Expected: compiles (will warn about unused code, that's fine).

- [ ] **Step 12: Commit**

```bash
git add Cargo.toml crates/pi-agent-core/Cargo.toml crates/pi-agent-core/src/lib.rs crates/pi-agent-core/src/types.rs
git commit -m "feat(pi-agent-core): scaffold crate and core types"
```

---

## Task 2: StreamFn Seam and Fake Provider

**Files:**
- Create: `crates/pi-agent-core/src/stream.rs`
- Modify: `crates/pi-agent-core/src/types.rs` (add `ModelInfo` and `StreamFnOptions` if needed)

**Interfaces:**
- Consumes: `AssistantMessage`, `AssistantMessageEvent`, `Message`, `Usage`, `StopReason`, `ThinkingLevel` from Task 1.
- Produces: `StreamFn` type, `AssistantEventStream` trait, `MockAssistantStream` builder, `MockAssistantEventStream` type used by tests.

- [ ] **Step 1: Define LLM request context in `types.rs`**

```rust
#[derive(Debug, Clone)]
pub struct LlmContext {
    pub system_prompt: String,
    pub messages: Vec<Message>,
    pub tools: Vec<Arc<dyn AgentTool>>,
}

#[derive(Debug, Clone)]
pub struct StreamFnOptions {
    pub api_key: Option<String>,
    pub session_id: Option<String>,
    pub thinking_budgets: Option<std::collections::HashMap<ThinkingLevel, u64>>,
    pub signal: Option<tokio::sync::watch::Receiver<bool>>,
}

pub type StreamFn = Arc<
    dyn Fn(String, LlmContext, StreamFnOptions) -> futures::future::BoxFuture<'static, Result<Box<dyn AssistantEventStream>, String>>
        + Send
        + Sync,
>;

#[async_trait::async_trait]
pub trait AssistantEventStream: Send + Unpin {
    async fn next_event(&mut self) -> Option<AssistantMessageEvent>;
    async fn result(self: Box<Self>) -> AssistantMessage;
}
```

- [ ] **Step 2: Implement `MockAssistantStream` in `stream.rs`**

Provide a builder that tests can use to pre-program assistant events:

```rust
use pi_agent_core::types::*;
use std::collections::VecDeque;
use futures::future::BoxFuture;

pub struct MockAssistantStream {
    events: VecDeque<AssistantMessageEvent>,
    final_message: AssistantMessage,
}

impl MockAssistantStream {
    pub fn new(final_message: AssistantMessage) -> Self { ... }
    pub fn push(&mut self, event: AssistantMessageEvent) { ... }
}

#[async_trait::async_trait]
impl AssistantEventStream for MockAssistantStream {
    async fn next_event(&mut self) -> Option<AssistantMessageEvent> { self.events.pop_front() }
    async fn result(self: Box<Self>) -> AssistantMessage { self.final_message }
}
```

- [ ] **Step 3: Provide a `stream_fn` factory**

```rust
pub fn mock_stream_fn<F>(mut factory: F) -> StreamFn
where
    F: FnMut(String, LlmContext, StreamFnOptions) -> Box<dyn AssistantEventStream> + Send + Sync + 'static,
{
    Arc::new(move |model, ctx, opts| {
        let stream = factory(model, ctx, opts);
        Box::pin(async move { Ok(stream) })
    })
}
```

Also provide helpers for common cases:

```rust
pub fn simple_text_response(text: &str) -> StreamFn { ... }
pub fn tool_use_response(tool_calls: Vec<ToolCall>, stop_reason: StopReason) -> StreamFn { ... }
```

- [ ] **Step 4: Add unit test for fake provider event sequence**

In `stream.rs` module tests or a new `tests/stream.rs`:

```rust
#[tokio::test]
async fn fake_provider_yields_programmed_events() {
    let mut stream = MockAssistantStream::new(AssistantMessage { ... });
    stream.push(AssistantMessageEvent::TextDelta { text: "hi".into() });
    stream.push(AssistantMessageEvent::Done);
    // consume events and assert result
}
```

Run: `cargo test -p pi-agent-core --lib stream`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/pi-agent-core/src/stream.rs crates/pi-agent-core/src/types.rs
git commit -m "feat(pi-agent-core): StreamFn seam and fake provider"
```

---

## Task 3: Tool Execution Engine

**Files:**
- Create: `crates/pi-agent-core/src/tool_engine.rs`

**Interfaces:**
- Consumes: `AgentTool`, `ToolCall`, `AgentToolResult`, `AgentContext`, `BeforeToolCallContext`, `AfterToolCallContext`, `AgentEvent` from Task 1.
- Produces: `execute_tool_calls(assistant_message, context, config, signal, emit) -> Result<ExecutedToolBatch, String>` where `emit` is an async event sink.

- [ ] **Step 1: Define `ExecutedToolBatch` and internal outcome types**

```rust
#[derive(Debug, Clone)]
pub struct ExecutedToolBatch {
    pub messages: Vec<ToolResultMessage>,
    pub terminate: bool,
}

struct PreparedToolCall {
    tool_call: ToolCall,
    tool: Arc<dyn AgentTool>,
    args: serde_json::Value,
}

enum ImmediateOutcome {
    Result(AgentToolResult, bool),
}
```

- [ ] **Step 2: Implement `prepare_tool_call`**

Flow:
1. Look up tool by `tool_call.name`. If missing, return error result.
2. Call `tool.prepare_arguments(tool_call.arguments)`.
3. Call `tool.validate(prepared_args)`. On validation error, return error result.
4. If `config.before_tool_call` is set, call it. If it returns `block: true`, build error result with reason; propagate `terminate` flag.
5. Return `PreparedToolCall`.

Use `Arc<dyn AgentTool>` for shared tool ownership.

- [ ] **Step 3: Implement `execute_prepared_tool_call`**

Call `tool.execute(tool_call_id, args, signal, on_update)`. Wrap the `on_update` callback so that late calls after the future settles are ignored (match Pi's behavior). Convert thrown errors into error `AgentToolResult` with text content.

- [ ] **Step 4: Implement `finalize_executed_tool_call`**

If `config.after_tool_call` is set, call it and replace fields field-by-field:

```rust
if let Some(after) = &config.after_tool_call {
    let result_after = after(ctx).await?;
    result.content = result_after.content.unwrap_or(result.content);
    result.details = result_after.details.unwrap_or(result.details);
    result.usage = result_after.usage.or(result.usage);
    if let Some(v) = result_after.is_error { is_error = v; }
    if let Some(v) = result_after.terminate { result.terminate = v; }
}
```

- [ ] **Step 5: Implement sequential execution**

For each tool call: emit `tool_execution_start`, prepare, execute (or immediate), finalize, emit `tool_execution_end`, build `ToolResultMessage`, emit `message_start`/`message_end`. Collect messages and determine `terminate`.

- [ ] **Step 6: Implement parallel execution**

1. For each tool call: sequentially emit `tool_execution_start`, prepare. If immediate (missing tool / blocked / validation error), emit `tool_execution_end` immediately and store finalized outcome. Otherwise spawn an async task/future that executes and finalizes and emits `tool_execution_end`.
2. Wait for all futures. Because each future emits its own `tool_execution_end`, completion order is naturally reflected.
3. After all futures complete, iterate in assistant source order and emit `message_start`/`message_end` for each `ToolResultMessage`.
4. Compute `terminate` from all finalized outcomes.

Important: the `emit` closure must be `Send` and `Sync` if shared across tasks. Use `Arc<dyn Fn(AgentEvent) -> BoxFuture<'static, ()> + Send + Sync>`.

- [ ] **Step 7: Implement `create_error_tool_result`**

```rust
pub fn create_error_tool_result(message: &str) -> AgentToolResult {
    AgentToolResult {
        content: vec![ContentBlock::Text { text: message.into() }],
        details: serde_json::Value::Null,
        usage: None,
        terminate: false,
    }
}
```

- [ ] **Step 8: Add module tests for prepare/execute/finalize**

Test cases:
- Successful execution returns content/details.
- Validation error yields error result, no `execute` called.
- `beforeToolCall` block + terminate yields error result and sets terminate.
- `afterToolCall` overrides content and terminate.
- Missing tool yields error result.

Run: `cargo test -p pi-agent-core --lib tool_engine`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add crates/pi-agent-core/src/tool_engine.rs
git commit -m "feat(pi-agent-core): tool execution engine"
```

---

## Task 4: Agent Loop

**Files:**
- Create: `crates/pi-agent-core/src/agent_loop.rs`

**Interfaces:**
- Consumes: `AgentContext`, `AgentLoopConfig`, `StreamFn`, `AssistantEventStream`, `AgentEvent`, tool engine from Tasks 2-3.
- Produces: `agent_loop(prompts, context, config, signal, stream_fn) -> AgentRun`, `agent_loop_continue(context, config, signal, stream_fn) -> AgentRun`.

- [ ] **Step 1: Define `AgentRun` return type**

`AgentRun` is both a `Stream` of `AgentEvent` and a future that resolves to `Vec<Message>` (the new messages produced by this run).

```rust
pub struct AgentRun {
    events: tokio::sync::mpsc::Receiver<AgentEvent>,
    handle: tokio::task::JoinHandle<Vec<Message>>,
}

impl AgentRun {
    pub async fn next_event(&mut self) -> Option<AgentEvent> { self.events.recv().await }
    pub async fn result(self) -> Vec<Message> { self.handle.await.unwrap_or_default() }
}

impl futures::Stream for AgentRun {
    type Item = AgentEvent;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.events.poll_recv(cx)
    }
}
```

Use `tokio::sync::mpsc::channel(256)` (bounded but large enough).

- [ ] **Step 2: Implement `run_agent_loop` (prompt path)**

1. Build `current_context` by appending prompts to `context.messages`.
2. Push `agent_start`, `turn_start`, then `message_start`/`message_end` for each prompt.
3. Call `run_loop`.
4. Return `new_messages` (just the prompts plus everything produced in this run).

- [ ] **Step 3: Implement `run_agent_loop_continue`**

1. Error if `context.messages.is_empty()`.
2. Error if last message role is `Assistant`.
3. Push `agent_start`, `turn_start`.
4. Call `run_loop`.
5. Return `new_messages` (only what the run produced, not pre-existing context).

- [ ] **Step 4: Implement `run_loop` inner/outer loop**

Pseudocode matching Pi:

```rust
let mut current_context = initial_context.clone();
let mut config = config.clone();
let mut first_turn = true;
let mut pending_messages = config.get_steering_messages.map(|f| f().await?).unwrap_or_default();

loop {
    let mut has_more_tool_calls = true;
    while has_more_tool_calls || !pending_messages.is_empty() {
        if !first_turn { emit(turn_start).await; }
        first_turn = false;

        for msg in pending_messages.drain(..) {
            emit(message_start/end).await;
            current_context.messages.push(msg.clone());
            new_messages.push(msg);
        }

        let message = stream_assistant_response(...).await;
        new_messages.push(Message::Assistant(message.clone()));

        if message.stop_reason == Error || message.stop_reason == Aborted {
            emit(turn_end).await;
            emit(agent_end).await;
            return;
        }

        let tool_calls = message.tool_calls();
        if tool_calls.is_empty() {
            emit(turn_end { message, tool_results: vec![] }).await;
        } else {
            let batch = if message.stop_reason == Length {
                fail_tool_calls_from_truncated_message(tool_calls, emit).await
            } else {
                execute_tool_calls(...).await
            };
            for tr in &batch.messages {
                current_context.messages.push(Message::ToolResult(tr.clone()));
                new_messages.push(Message::ToolResult(tr.clone()));
            }
            emit(turn_end { message, tool_results: batch.messages }).await;
            has_more_tool_calls = !batch.terminate;
        }

        if let Some(update) = config.prepare_next_turn(...) {
            if let Some(ctx) = update.context { current_context = ctx; }
            if let Some(m) = update.model { config.model = m; }
            if let Some(tl) = update.thinking_level { config.thinking_level = Some(tl); }
        }

        if let Some(should_stop) = config.should_stop_after_turn {
            if should_stop(...).await? {
                emit(agent_end).await;
                return;
            }
        }

        pending_messages = config.get_steering_messages.map(|f| f().await?).unwrap_or_default();
    }

    let follow_ups = config.get_follow_up_messages.map(|f| f().await?).unwrap_or_default();
    if !follow_ups.is_empty() {
        pending_messages = follow_ups;
        continue;
    }
    break;
}

emit(agent_end).await;
```

- [ ] **Step 5: Implement `stream_assistant_response`**

1. Apply `transform_context` if present.
2. Call `convert_to_llm`.
3. Build `LlmContext` and call `stream_fn`.
4. Iterate `AssistantMessageEvent`s:
   - `Start` / stream deltas: update partial message in `current_context.messages`, emit `message_start`/`message_update`.
   - `Done`/`Error`: get final message via `stream.result()`, replace partial or push, emit `message_end`, return.

Ensure a `StreamFn` that returns no events still emits `message_start` and `message_end` for the final message.

- [ ] **Step 6: Implement `fail_tool_calls_from_truncated_message`**

For each tool call: emit `tool_execution_start`, build error result ("output token limit"), emit `tool_execution_end`, emit `message_start`/`message_end` for tool result. Return `terminate: false`.

- [ ] **Step 7: Add loop unit tests**

In `agent_loop.rs` module tests or `tests/`:
- Pure text turn produces lifecycle events.
- One tool call + final response.
- Parallel execution completion order vs source order.
- Sequential override by tool.
- `length` stop reason skips execution.
- `should_stop_after_turn` ends the run.
- Steering injection after tool batch.
- Follow-up continuation.

Use the fake provider from Task 2.

Run: `cargo test -p pi-agent-core --lib agent_loop`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/pi-agent-core/src/agent_loop.rs
git commit -m "feat(pi-agent-core): agent loop with lifecycle and queues"
```

---

## Task 5: Agent Wrapper

**Files:**
- Create: `crates/pi-agent-core/src/agent.rs`

**Interfaces:**
- Consumes: `AgentState`, `AgentLoopConfig`, `agent_loop`, `agent_loop_continue`, `QueueMode`.
- Produces: `Agent` struct with `prompt`, `continue_run`, `steer`, `follow_up`, `abort`, `reset`, `wait_for_idle`, `subscribe`.

- [ ] **Step 1: Define `Agent` and `PendingMessageQueue`**

```rust
pub struct Agent {
    state: AgentState,
    listeners: Vec<Arc<dyn Fn(AgentEvent, tokio::sync::watch::Receiver<bool>) -> BoxFuture<'static, ()> + Send + Sync>>,
    steering_queue: PendingMessageQueue,
    follow_up_queue: PendingMessageQueue,
    active_run: Option<ActiveRun>,
}

struct PendingMessageQueue {
    mode: QueueMode,
    messages: Vec<Message>,
}

struct ActiveRun {
    promise: tokio::sync::oneshot::Receiver<()>,
    abort_tx: tokio::sync::watch::Sender<bool>,
}
```

- [ ] **Step 2: Implement queue methods**

`steer`, `follow_up`, `clear_steering_queue`, `clear_follow_up_queue`, `has_queued_messages`, `steering_mode`, `follow_up_mode` setters.

- [ ] **Step 3: Implement `prompt`**

1. Error if `active_run` exists.
2. Normalize input (string -> UserMessage with timestamp).
3. Create `AgentLoopConfig` from agent fields; wire `get_steering_messages` to drain steering queue (with optional `skip_initial_steering_poll`).
4. Run `agent_loop` with lifecycle wrapper that awaits listeners, tracks streaming state, and handles run failure.

- [ ] **Step 4: Implement `continue_run`**

1. Error if `active_run` exists.
2. If last message is assistant: drain steering; if empty drain follow-up; if still empty error "Cannot continue from message role: assistant".
3. Run `agent_loop_continue` or `agent_loop` with the queued messages.

- [ ] **Step 5: Implement `abort`, `wait_for_idle`, `reset`**

- `abort`: send `true` through abort watch channel.
- `wait_for_idle`: await `active_run.promise` if any.
- `reset`: error if `active_run` exists; clear messages, streaming state, pending tool calls, error message, queues.

- [ ] **Step 6: Implement `subscribe` and listener settlement**

`subscribe(listener)` adds a listener. During a run, every emitted event is passed to every listener with the current abort signal. Listeners are awaited in order. `agent_end` listeners must settle before the run's promise resolves and `is_streaming` becomes false.

- [ ] **Step 7: Implement run failure handling**

If the executor (or `StreamFn`) throws outside the normal protocol, build an assistant error/aborted message and emit `message_start`, `message_end`, `turn_end`, `agent_end` through `process_events`.

- [ ] **Step 8: Add wrapper unit tests**

Tests:
- `prompt` emits full lifecycle.
- `prompt` rejects while streaming.
- `continue_run` rejects while streaming.
- `abort` propagates signal and completes lifecycle.
- `wait_for_idle` waits for async listeners.
- `reset` rejects during run and clears state after run.
- steering/follow-up queue semantics.

Run: `cargo test -p pi-agent-core --lib agent`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add crates/pi-agent-core/src/agent.rs
git commit -m "feat(pi-agent-core): Agent stateful wrapper"
```

---

## Task 6: Trace Collector and Acceptance Tests

**Files:**
- Create: `crates/pi-agent-core/src/trace.rs`
- Create: `crates/pi-agent-core/tests/acceptance.rs`

**Interfaces:**
- Consumes: `Agent`, `AgentEvent`, `Message`, `ContentBlock`, `StopReason`, `ToolCall`, `AgentToolResult` from previous tasks.
- Produces: `TraceCollector` helper and 12 passing acceptance tests.

- [ ] **Step 1: Implement `TraceCollector`**

```rust
#[derive(Debug, Clone, Default)]
pub struct TraceCollector {
    entries: Vec<TraceEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TraceEntry {
    Event { event_type: String },
    MessageStart { role: String },
    MessageEnd { role: String, stop_reason: Option<String> },
    ToolExecutionStart { tool_call_id: String, tool_name: String, args: serde_json::Value },
    ToolExecutionEnd { tool_call_id: String, tool_name: String, result_summary: String, is_error: bool },
    ToolResult { tool_call_id: String, tool_name: String, is_error: bool },
    TurnEnd { tool_result_ids: Vec<String> },
    StateSnapshot { is_streaming: bool, pending: Vec<String> },
}

impl TraceCollector {
    pub fn new() -> Self { Default::default() }
    pub async fn collect_run(&mut self, run: &mut AgentRun) -> Vec<Message> { ... }
    pub fn observe_event(&mut self, event: &AgentEvent) { ... }
    pub fn snapshot_state(&mut self, state: &AgentState) { ... }
    pub fn entries(&self) -> &[TraceEntry] { ... }
}
```

Normalization rules:
- `ToolExecutionUpdate` events are ignored in default trace (they are secondary; can be optionally captured).
- Timestamps are not recorded.
- `result_summary` is a compact string derived from content (e.g. first text block or "error").

- [ ] **Step 2: Provide test helpers**

In `tests/acceptance.rs` or a `src/test_helpers.rs` module (cfg(test)):

```rust
fn user_message(text: &str) -> Message { ... }
fn text_content(text: &str) -> Vec<ContentBlock> { ... }
fn tool_call(id: &str, name: &str, args: serde_json::Value) -> ContentBlock { ... }
fn echo_tool(name: &str) -> Arc<dyn AgentTool> { ... }
fn recording_tool(name: &str, log: Arc<Mutex<Vec<String>>>) -> Arc<dyn AgentTool> { ... }
```

- [ ] **Step 3: Write 12 acceptance tests**

Each test:
1. Builds `Agent` with deterministic fake `StreamFn` and tools.
2. Runs `prompt`/`continue_run`.
3. Collects trace and asserts normalized sequence equals expected.
4. Asserts final transcript roles and stop reasons.

Tests:

1. `text_only_turn`
2. `one_tool_call_then_final_response`
3. `two_parallel_tool_calls_completion_order_vs_source_order`
4. `sequential_execution_mode_forces_sequential`
5. `steering_queue_one_at_a_time`
6. `steering_queue_all`
7. `follow_up_queue_one_at_a_time`
8. `follow_up_queue_all`
9. `abort_while_provider_pending`
10. `stream_fn_throw_converts_to_error_lifecycle`
11. `length_truncated_tool_call_never_executes`
12. `before_and_after_tool_hooks_and_terminate`
13. `agent_loop_continue_no_user_message_events` (bonus, keep if time permits)
14. `prompt_continue_reject_while_streaming` (bonus)

- [ ] **Step 4: Run all acceptance tests**

Run: `cargo test -p pi-agent-core --test acceptance`
Expected: all 12 pass.

- [ ] **Step 5: Run full crate test suite**

Run: `cargo test -p pi-agent-core`
Expected: all unit + acceptance tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/pi-agent-core/src/trace.rs crates/pi-agent-core/tests/acceptance.rs
git commit -m "test(pi-agent-core): trace collector and 12 acceptance tests"
```

---

## Self-Review Checklist

- [ ] Spec coverage: every contract in `.scratch/pi-rust-replica-research-corrected.md` §3 maps to at least one task.
- [ ] Placeholder scan: no "TBD", "TODO", "implement later", or open-ended "handle edge cases" steps.
- [ ] Type consistency: `AgentEvent`, `Message`, `AgentToolResult`, `ToolCall`, `AgentLoopConfig` names match across tasks.
- [ ] Trace-equivalence: Task 6 compares normalized event order/roles/IDs, not timestamps.
- [ ] Testability: every task has a runnable test command.
