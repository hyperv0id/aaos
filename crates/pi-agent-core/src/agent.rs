use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use futures::future::BoxFuture;
use tokio::sync::watch;

use crate::agent_loop::{
    AgentRun, LoopError, agent_loop, agent_loop_continue, thinking_level_to_option,
};
use crate::types::{
    AfterToolCallHook, AgentContext, AgentEvent, AgentLoopConfig, AgentState, AssistantMessage,
    BeforeToolCallHook, ContentBlock, ConvertToLlm, Message, PrepareNextTurnHook, QueueMode,
    ShouldStopAfterTurnHook, StopReason, StreamFn, StreamFnOptions, ThinkingLevel,
    ToolExecutionMode, TransformContext, UserMessage,
};

/// A callback that receives every agent event plus the active abort signal.
pub type Listener =
    Arc<dyn Fn(AgentEvent, watch::Receiver<bool>) -> BoxFuture<'static, ()> + Send + Sync>;

/// Re-entrancy and validation errors from `Agent::prompt` / `Agent::continue_run`.
///
/// Upstream throws synchronously for these conditions; the Rust port returns
/// `Err` so callers can handle them without catching panics. The error strings
/// match the upstream messages verbatim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentError {
    /// `Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.`
    AlreadyProcessing,
    /// `Agent is already processing. Wait for completion before continuing.`
    AlreadyProcessingContinue,
    /// `No messages to continue from`
    NoMessagesToContinueFrom,
    /// `Cannot continue from message role: assistant`
    CannotContinueFromAssistant,
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::AlreadyProcessing => write!(
                f,
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
            ),
            AgentError::AlreadyProcessingContinue => write!(
                f,
                "Agent is already processing. Wait for completion before continuing."
            ),
            AgentError::NoMessagesToContinueFrom => write!(f, "No messages to continue from"),
            AgentError::CannotContinueFromAssistant => {
                write!(f, "Cannot continue from message role: assistant")
            }
        }
    }
}

impl std::error::Error for AgentError {}

fn lock_listeners(listeners: &Arc<Mutex<Vec<Listener>>>) -> MutexGuard<'_, Vec<Listener>> {
    match listeners.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

struct PendingMessageQueue {
    inner: Arc<Mutex<PendingMessageQueueState>>,
}

struct PendingMessageQueueState {
    mode: QueueMode,
    messages: Vec<Message>,
}

impl PendingMessageQueue {
    fn new(mode: QueueMode) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PendingMessageQueueState {
                mode,
                messages: Vec::new(),
            })),
        }
    }

    fn lock(&self) -> MutexGuard<'_, PendingMessageQueueState> {
        match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    fn enqueue(&self, message: Message) {
        self.lock().messages.push(message);
    }

    fn has_items(&self) -> bool {
        !self.lock().messages.is_empty()
    }

    fn drain(&self) -> Vec<Message> {
        let mut state = self.lock();
        match state.mode {
            QueueMode::All => std::mem::take(&mut state.messages),
            QueueMode::OneAtATime => {
                if state.messages.is_empty() {
                    Vec::new()
                } else {
                    vec![state.messages.remove(0)]
                }
            }
        }
    }

    fn clear(&self) {
        self.lock().messages.clear();
    }

    fn mode(&self) -> QueueMode {
        self.lock().mode
    }

    fn set_mode(&self, mode: QueueMode) {
        self.lock().mode = mode;
    }
}

impl Clone for PendingMessageQueue {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

/// Per-run state. The run keeps its abort channel alive via the
/// [`RunState::active_abort`] sender; this struct only holds the idle
/// barrier sender so `wait_for_idle` observers can complete.
struct ActiveRun {
    idle_tx: watch::Sender<bool>,
}

/// Shared run state accessible from both [`Agent`] and [`AgentHandle`].
#[derive(Clone)]
struct RunState {
    active_abort: Arc<Mutex<Option<watch::Sender<bool>>>>,
    active_idle: Arc<Mutex<Option<watch::Receiver<bool>>>>,
}

/// A cloneable handle that can steer, abort, and inspect an active [`Agent`]
/// run concurrently — even while `prompt(&mut self).await` is pending.
///
/// Shares Arc'd queues and run-state with the owning [`Agent`], so operations
/// are effective immediately. Created via [`Agent::handle`].
#[derive(Clone)]
pub struct AgentHandle {
    steering_queue: PendingMessageQueue,
    follow_up_queue: PendingMessageQueue,
    run_state: RunState,
}

impl AgentHandle {
    pub fn steer(&self, message: Message) {
        self.steering_queue.enqueue(message);
    }

    pub fn follow_up(&self, message: Message) {
        self.follow_up_queue.enqueue(message);
    }

    pub fn clear_steering_queue(&self) {
        self.steering_queue.clear();
    }

    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue.clear();
    }

    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    pub fn has_queued_messages(&self) -> bool {
        self.steering_queue.has_items() || self.follow_up_queue.has_items()
    }

    pub fn steering_mode(&self) -> QueueMode {
        self.steering_queue.mode()
    }

    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.steering_queue.set_mode(mode);
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue.mode()
    }

    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_queue.set_mode(mode);
    }

    pub fn abort(&self) {
        if let Some(tx) = self.run_state.active_abort.lock().unwrap().as_ref() {
            let _ = tx.send(true);
        }
    }

    /// Capture an [`AgentAbortHandle`] bound to the currently active run, if
    /// any. When no run is active, the returned handle is inert. Captured
    /// during a run, the handle goes stale once that run finishes.
    pub fn abort_handle(&self) -> AgentAbortHandle {
        AgentAbortHandle {
            abort_tx: self.run_state.active_abort.lock().unwrap().clone(),
        }
    }

    pub fn signal(&self) -> Option<watch::Receiver<bool>> {
        self.run_state
            .active_abort
            .lock()
            .unwrap()
            .as_ref()
            .map(|tx| tx.subscribe())
    }

    pub fn wait_for_idle(&self) -> BoxFuture<'static, ()> {
        let idle = self.run_state.active_idle.lock().unwrap().clone();
        Box::pin(async move {
            let Some(mut idle) = idle else {
                return;
            };
            if *idle.borrow_and_update() {
                return;
            }
            while idle.changed().await.is_ok() {
                if *idle.borrow_and_update() {
                    return;
                }
            }
        })
    }
}

/// A cloneable handle that can abort an active [`Agent`] run.
///
/// Captures the *specific* run's abort sender at creation time. When that run
/// finishes, the handle goes dead — `abort()` becomes a no-op and cannot
/// affect a later run. This matches upstream's per-run `AbortController`.
#[derive(Clone)]
pub struct AgentAbortHandle {
    abort_tx: Option<watch::Sender<bool>>,
}

impl AgentAbortHandle {
    pub fn abort(&self) {
        if let Some(tx) = &self.abort_tx {
            let _ = tx.send(true);
        }
    }
}

/// Stateful wrapper around the low-level agent loop.
///
/// Mirrors Pi's `Agent` class: owns transcript, tools, listeners, and queued
/// messages, and exposes the lifecycle methods used by TUI/shell code.
pub struct Agent {
    pub state: AgentState,
    pub stream_fn: Arc<dyn StreamFn>,
    pub before_tool_call: Option<BeforeToolCallHook>,
    pub after_tool_call: Option<AfterToolCallHook>,
    pub should_stop_after_turn: Option<ShouldStopAfterTurnHook>,
    pub prepare_next_turn: Option<PrepareNextTurnHook>,
    pub convert_to_llm: ConvertToLlm,
    pub transform_context: Option<TransformContext>,
    pub tool_execution: ToolExecutionMode,
    pub stream_fn_options: StreamFnOptions,

    listeners: Arc<Mutex<Vec<Listener>>>,
    steering_queue: PendingMessageQueue,
    follow_up_queue: PendingMessageQueue,
    active_run: Option<ActiveRun>,
    run_state: RunState,
}
impl Agent {
    pub fn new(stream_fn: Arc<dyn StreamFn>) -> Self {
        Self {
            state: AgentState::default(),
            stream_fn,
            before_tool_call: None,
            after_tool_call: None,
            should_stop_after_turn: None,
            prepare_next_turn: None,
            convert_to_llm: Arc::new(|m| Box::pin(async move { Ok(m) })),
            transform_context: None,
            tool_execution: ToolExecutionMode::default(),
            stream_fn_options: StreamFnOptions::default(),
            listeners: Arc::new(Mutex::new(Vec::new())),
            steering_queue: PendingMessageQueue::new(QueueMode::OneAtATime),
            follow_up_queue: PendingMessageQueue::new(QueueMode::OneAtATime),
            active_run: None,
            run_state: RunState {
                active_abort: Arc::new(Mutex::new(None)),
                active_idle: Arc::new(Mutex::new(None)),
            },
        }
    }

    pub fn subscribe(&self, listener: Listener) -> impl FnOnce() + use<> {
        let listeners = self.listeners.clone();
        {
            let mut guard = lock_listeners(&listeners);
            if !guard
                .iter()
                .any(|existing| Arc::ptr_eq(existing, &listener))
            {
                guard.push(listener.clone());
            }
        }
        move || {
            let mut listeners = lock_listeners(&listeners);
            if let Some(index) = listeners
                .iter()
                .position(|existing| Arc::ptr_eq(existing, &listener))
            {
                listeners.remove(index);
            }
        }
    }

    /// Returns a cloneable handle that can steer, abort, and inspect the agent
    /// concurrently — even while a `prompt(&mut self).await` is pending.
    pub fn handle(&self) -> AgentHandle {
        AgentHandle {
            steering_queue: self.steering_queue.clone(),
            follow_up_queue: self.follow_up_queue.clone(),
            run_state: self.run_state.clone(),
        }
    }

    pub fn steering_mode(&self) -> QueueMode {
        self.steering_queue.mode()
    }

    pub fn set_steering_mode(&self, mode: QueueMode) {
        self.steering_queue.set_mode(mode);
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue.mode()
    }

    pub fn set_follow_up_mode(&self, mode: QueueMode) {
        self.follow_up_queue.set_mode(mode);
    }

    pub fn steer(&self, message: Message) {
        self.steering_queue.enqueue(message);
    }

    pub fn follow_up(&self, message: Message) {
        self.follow_up_queue.enqueue(message);
    }

    pub fn clear_steering_queue(&self) {
        self.steering_queue.clear();
    }

    pub fn clear_follow_up_queue(&self) {
        self.follow_up_queue.clear();
    }

    pub fn clear_all_queues(&self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    pub fn has_queued_messages(&self) -> bool {
        self.steering_queue.has_items() || self.follow_up_queue.has_items()
    }

    pub fn signal(&self) -> Option<watch::Receiver<bool>> {
        self.run_state
            .active_abort
            .lock()
            .unwrap()
            .as_ref()
            .map(|tx| tx.subscribe())
    }

    pub fn abort_handle(&self) -> AgentAbortHandle {
        AgentAbortHandle {
            abort_tx: self.run_state.active_abort.lock().unwrap().clone(),
        }
    }

    pub fn abort(&self) {
        if let Some(tx) = self.run_state.active_abort.lock().unwrap().as_ref() {
            let _ = tx.send(true);
        }
    }

    pub fn wait_for_idle(&self) -> BoxFuture<'static, ()> {
        let idle = self.run_state.active_idle.lock().unwrap().clone();
        Box::pin(async move {
            let Some(mut idle) = idle else {
                return;
            };
            if *idle.borrow_and_update() {
                return;
            }
            while idle.changed().await.is_ok() {
                if *idle.borrow_and_update() {
                    return;
                }
            }
        })
    }

    pub fn reset(&mut self) {
        if self.active_run.is_some() {
            panic!("Agent is already processing. Wait for completion before resetting.");
        }
        self.state.messages.clear();
        self.state.is_streaming = false;
        self.state.streaming_message = None;
        self.state.pending_tool_calls.clear();
        self.state.error_message = None;
        self.clear_all_queues();
    }

    pub async fn prompt(&mut self, input: impl Into<PromptInput>) -> Result<(), AgentError> {
        if self.active_run.is_some() {
            return Err(AgentError::AlreadyProcessing);
        }
        let messages = input.into().into_messages();
        self.run_prompt_messages(messages, false).await;
        Ok(())
    }

    pub async fn continue_run(&mut self) -> Result<(), AgentError> {
        if self.active_run.is_some() {
            return Err(AgentError::AlreadyProcessingContinue);
        }

        if self.state.messages.is_empty() {
            return Err(AgentError::NoMessagesToContinueFrom);
        }

        let last_role = self.state.messages.last().map(|m| m.role());
        if last_role == Some("assistant") {
            let steering = self.steering_queue.drain();
            if !steering.is_empty() {
                self.run_prompt_messages(steering, true).await;
                return Ok(());
            }
            let follow_ups = self.follow_up_queue.drain();
            if !follow_ups.is_empty() {
                self.run_prompt_messages(follow_ups, false).await;
                return Ok(());
            }
            return Err(AgentError::CannotContinueFromAssistant);
        }

        self.run_continuation().await;
        Ok(())
    }

    async fn run_prompt_messages(
        &mut self,
        messages: Vec<Message>,
        skip_initial_steering_poll: bool,
    ) {
        let (abort_tx, _abort_rx) = watch::channel(false);
        let (idle_tx, _idle_rx) = watch::channel(false);

        *self.run_state.active_abort.lock().unwrap() = Some(abort_tx.clone());
        *self.run_state.active_idle.lock().unwrap() = Some(idle_tx.subscribe());

        let abort_rx = abort_tx.subscribe();

        let context = self.create_context_snapshot();
        let config = self.create_loop_config(skip_initial_steering_poll);
        let stream_fn = self.stream_fn.clone();
        let listeners = self.listeners.clone();
        let message_prefix = self.state.messages.len();

        self.state.is_streaming = true;
        self.state.streaming_message = None;
        self.state.error_message = None;

        self.active_run = Some(ActiveRun { idle_tx });

        let mut run = agent_loop(messages, context, config, abort_rx, stream_fn);
        let run_error = self.drain_run_events(&mut run, &listeners, &abort_tx).await;
        if let Some(error) = run_error {
            self.handle_loop_error(&listeners, &abort_tx, error, message_prefix)
                .await;
        }

        self.finish_run();
    }

    async fn run_continuation(&mut self) {
        let (abort_tx, _abort_rx) = watch::channel(false);
        let (idle_tx, _idle_rx) = watch::channel(false);

        *self.run_state.active_abort.lock().unwrap() = Some(abort_tx.clone());
        *self.run_state.active_idle.lock().unwrap() = Some(idle_tx.subscribe());

        let abort_rx = abort_tx.subscribe();

        let context = self.create_context_snapshot();
        let config = self.create_loop_config(false);
        let stream_fn = self.stream_fn.clone();
        let listeners = self.listeners.clone();
        let message_prefix = self.state.messages.len();

        self.state.is_streaming = true;
        self.state.streaming_message = None;
        self.state.error_message = None;

        self.active_run = Some(ActiveRun { idle_tx });

        let mut run = match agent_loop_continue(context, config, abort_rx, stream_fn) {
            Ok(run) => run,
            Err(_) => {
                // Synchronous validation failed; finish_run() will clean
                // up abort/idle state.
                self.finish_run();
                return;
            }
        };
        let run_error = self.drain_run_events(&mut run, &listeners, &abort_tx).await;
        if let Some(error) = run_error {
            self.handle_loop_error(&listeners, &abort_tx, error, message_prefix)
                .await;
        }

        self.finish_run();
    }

    /// When the loop returns a hook error, the error lifecycle (message_start,
    /// message_end, turn_end, agent_end) was already emitted through the event
    /// channel before the task returned Err — so the Agent's state is already
    /// updated via `process_event`. We only need to synthesize the terminal
    /// lifecycle for join failures (panic/cancellation), where no events were
    /// emitted.
    async fn handle_loop_error(
        &mut self,
        listeners: &Arc<Mutex<Vec<Listener>>>,
        abort_tx: &watch::Sender<bool>,
        error: LoopError,
        message_prefix: usize,
    ) {
        match error {
            LoopError::Hook(_) => {
                // Lifecycle already emitted by the loop via
                // emit_hook_error_lifecycle; state already updated through
                // process_event. Nothing to do.
            }
            LoopError::Join(e) => {
                self.complete_failed_run(
                    listeners,
                    abort_tx,
                    &format!("agent run failed: {e}"),
                    message_prefix,
                )
                .await;
            }
        }
    }

    async fn drain_run_events(
        &mut self,
        run: &mut AgentRun,
        listeners: &Arc<Mutex<Vec<Listener>>>,
        abort_tx: &watch::Sender<bool>,
    ) -> Option<LoopError> {
        while let Some(event) = run.next_event().await {
            self.process_event(&event);
            let snapshot = lock_listeners(listeners).clone();
            for listener in snapshot {
                listener(event.clone(), abort_tx.subscribe()).await;
            }
        }
        run.result().await.err()
    }

    async fn complete_failed_run(
        &mut self,
        listeners: &Arc<Mutex<Vec<Listener>>>,
        abort_tx: &watch::Sender<bool>,
        error_message: &str,
        message_prefix: usize,
    ) {
        let assistant_message = AssistantMessage {
            content: vec![ContentBlock::text("")],
            stop_reason: StopReason::Error,
            error_message: Some(error_message.to_string()),
            ..Default::default()
        };
        let message = Message::Assistant(assistant_message);

        let pre_terminal = [
            AgentEvent::MessageStart {
                message: message.clone(),
            },
            AgentEvent::MessageEnd {
                message: message.clone(),
            },
            AgentEvent::TurnEnd {
                message,
                tool_results: Vec::new(),
            },
        ];
        for event in pre_terminal {
            self.process_event(&event);
            let snapshot = lock_listeners(listeners).clone();
            for listener in snapshot {
                listener(event.clone(), abort_tx.subscribe()).await;
            }
        }

        let agent_end = AgentEvent::AgentEnd {
            messages: self.state.messages[message_prefix..].to_vec(),
        };
        self.process_event(&agent_end);
        let snapshot = lock_listeners(listeners).clone();
        for listener in snapshot {
            listener(agent_end.clone(), abort_tx.subscribe()).await;
        }
    }

    fn process_event(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::MessageStart { message } => {
                self.state.streaming_message = Some(message.clone());
            }
            AgentEvent::MessageUpdate { message, .. } => {
                self.state.streaming_message = Some(message.clone());
            }
            AgentEvent::MessageEnd { message } => {
                self.state.streaming_message = None;
                self.state.messages.push(message.clone());
            }
            AgentEvent::ToolExecutionStart { tool_call_id, .. } => {
                self.state.pending_tool_calls.insert(tool_call_id.clone());
            }
            AgentEvent::ToolExecutionEnd { tool_call_id, .. } => {
                self.state.pending_tool_calls.remove(tool_call_id);
            }
            AgentEvent::TurnEnd { message, .. } => {
                if let Some(err) = message.as_assistant().and_then(|m| m.error_message.clone()) {
                    self.state.error_message = Some(err);
                }
            }
            AgentEvent::AgentEnd { .. } => {
                self.state.streaming_message = None;
            }
            _ => {}
        }
    }

    fn finish_run(&mut self) {
        self.state.is_streaming = false;
        self.state.streaming_message = None;
        self.state.pending_tool_calls.clear();
        if let Some(run) = &self.active_run {
            let _ = run.idle_tx.send(true);
        }
        *self.run_state.active_abort.lock().unwrap() = None;
        *self.run_state.active_idle.lock().unwrap() = None;
        self.active_run = None;
    }

    fn create_context_snapshot(&self) -> AgentContext {
        AgentContext {
            system_prompt: self.state.system_prompt.clone(),
            messages: self.state.messages.clone(),
            tools: self.state.tools.clone(),
        }
    }

    fn create_loop_config(&self, skip_initial_steering_poll: bool) -> AgentLoopConfig {
        let skip = Arc::new(AtomicBool::new(skip_initial_steering_poll));
        let steering_queue = self.steering_queue.clone();
        let follow_up_queue = self.follow_up_queue.clone();

        let get_steering: Option<crate::types::GetMessagesHook> = Some(Arc::new(move || {
            let queue = steering_queue.clone();
            let skip = skip.clone();
            Box::pin(async move {
                if skip.swap(false, Ordering::SeqCst) {
                    Ok(Vec::new())
                } else {
                    Ok(queue.drain())
                }
            })
        }));

        let get_follow_up: Option<crate::types::GetMessagesHook> = Some(Arc::new(move || {
            let queue = follow_up_queue.clone();
            Box::pin(async move { Ok(queue.drain()) })
        }));

        AgentLoopConfig {
            model: self.state.model.clone(),
            thinking_level: self.thinking_level(),
            tool_execution: self.tool_execution,
            before_tool_call: self.before_tool_call.clone(),
            after_tool_call: self.after_tool_call.clone(),
            should_stop_after_turn: self.should_stop_after_turn.clone(),
            prepare_next_turn: self.prepare_next_turn.clone(),
            get_steering_messages: get_steering,
            get_follow_up_messages: get_follow_up,
            convert_to_llm: self.convert_to_llm.clone(),
            transform_context: self.transform_context.clone(),
            stream_fn_options: self.stream_fn_options.clone(),
        }
    }

    fn thinking_level(&self) -> Option<ThinkingLevel> {
        thinking_level_to_option(self.state.thinking_level)
    }
}

/// Input accepted by `Agent::prompt`.
pub enum PromptInput {
    Text(String),
    Messages(Vec<Message>),
}

impl PromptInput {
    fn into_messages(self) -> Vec<Message> {
        match self {
            PromptInput::Text(text) => vec![Message::User(UserMessage {
                content: vec![ContentBlock::Text { text }],
                timestamp: now(),
            })],
            PromptInput::Messages(messages) => messages,
        }
    }
}

impl From<String> for PromptInput {
    fn from(text: String) -> Self {
        PromptInput::Text(text)
    }
}

impl From<&str> for PromptInput {
    fn from(text: &str) -> Self {
        PromptInput::Text(text.to_string())
    }
}

impl From<Vec<Message>> for PromptInput {
    fn from(messages: Vec<Message>) -> Self {
        PromptInput::Messages(messages)
    }
}

impl From<Message> for PromptInput {
    fn from(message: Message) -> Self {
        PromptInput::Messages(vec![message])
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
    use crate::stream::simple_text_response;
    use crate::types::{AssistantEventStream, LlmContext, Model, StopReason, StreamFnOptions};
    use async_trait::async_trait;

    #[tokio::test]
    async fn full_lifecycle_emits_events() {
        let mut agent = Agent::new(simple_text_response("Hello"));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
        let _ = agent.subscribe(Arc::new(move |event, _signal| {
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send(event);
            })
        }));

        agent.prompt("hi").await.unwrap();

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }
        assert_eq!(events[0], AgentEvent::AgentStart);
        assert!(matches!(events.last(), Some(AgentEvent::AgentEnd { .. })));
        assert!(!agent.state.is_streaming);
        assert_eq!(agent.state.messages.len(), 2);
    }

    #[tokio::test]
    async fn prompt_rejects_while_streaming() {
        let mut agent = Agent::new(simple_text_response("Hello"));
        let (abort_tx, _) = watch::channel(false);
        let (idle_tx, idle_rx) = watch::channel(false);
        *agent.run_state.active_abort.lock().unwrap() = Some(abort_tx.clone());
        agent.active_run = Some(ActiveRun { idle_tx });
        let _ = idle_rx;
        // prompt() now returns Err(AgentError::AlreadyProcessing) instead of
        // panicking — matching upstream's rejected-promise behaviour.
        let result = agent.prompt("hi").await;
        assert_eq!(result, Err(AgentError::AlreadyProcessing));
    }

    #[tokio::test]
    async fn reset_rejects_during_run() {
        let mut agent = Agent::new(simple_text_response("Hello"));
        let (abort_tx, _) = watch::channel(false);
        let (idle_tx, idle_rx) = watch::channel(false);
        *agent.run_state.active_abort.lock().unwrap() = Some(abort_tx.clone());
        agent.active_run = Some(ActiveRun { idle_tx });
        let _ = idle_rx;
        let handle = tokio::spawn(async move {
            agent.reset();
        });
        assert!(handle.await.is_err());
    }

    #[tokio::test]
    async fn abort_completes_lifecycle() {
        // abort() with no active run is a no-op.
        let agent = Agent::new(simple_text_response("Hello"));
        agent.abort();
        assert!(agent.active_run.is_none());
    }

    #[tokio::test]
    async fn wait_for_idle_shared_barrier() {
        use std::time::Duration;

        let mut agent = Agent::new(simple_text_response("Hello"));
        let (idle_tx, idle_rx) = watch::channel(false);
        *agent.run_state.active_idle.lock().unwrap() = Some(idle_rx);
        agent.active_run = Some(ActiveRun {
            idle_tx: idle_tx.clone(),
        });

        let mut wait1 = agent.wait_for_idle();
        let mut wait2 = agent.wait_for_idle();

        tokio::select! {
            _ = &mut wait1 => panic!("waiter 1 resolved before the shared idle signal flipped true"),
            _ = &mut wait2 => panic!("waiter 2 resolved before the shared idle signal flipped true"),
            _ = tokio::time::sleep(Duration::from_millis(50)) => {}
        }

        let _ = idle_tx.send(true);

        tokio::time::timeout(Duration::from_secs(5), async {
            (&mut wait1).await;
            (&mut wait2).await;
        })
        .await
        .expect("both waiters must resolve once the shared idle signal flips true");
    }

    struct PanickingStreamFn;

    #[async_trait]
    impl StreamFn for PanickingStreamFn {
        async fn call(
            &self,
            _model: Model,
            _context: LlmContext,
            _options: StreamFnOptions,
            _abort: watch::Receiver<bool>,
        ) -> Result<Box<dyn AssistantEventStream>, String> {
            panic!("stream provider panicked");
        }
    }

    #[tokio::test]
    async fn panicking_provider_error_lifecycle() {
        let mut agent = Agent::new(Arc::new(PanickingStreamFn));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
        let _ = agent.subscribe(Arc::new(move |event, _signal| {
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send(event);
            })
        }));

        agent.prompt("hi").await.unwrap();

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        let terminal: Vec<&AgentEvent> = events.iter().rev().take(4).rev().collect();
        assert!(matches!(terminal[0], AgentEvent::MessageStart { .. }));
        assert!(matches!(terminal[1], AgentEvent::MessageEnd { .. }));
        assert!(matches!(terminal[2], AgentEvent::TurnEnd { .. }));
        assert!(matches!(terminal[3], AgentEvent::AgentEnd { .. }));

        assert!(!agent.state.is_streaming);
        let assistant = agent
            .state
            .messages
            .last()
            .and_then(Message::as_assistant)
            .expect("terminal error message recorded");
        assert_eq!(assistant.stop_reason, StopReason::Error);
        let error_message = assistant
            .error_message
            .as_deref()
            .expect("error message should be explanatory");
        assert!(error_message.contains("stream provider panicked"));
        assert!(agent.state.error_message.is_some());
    }
}
