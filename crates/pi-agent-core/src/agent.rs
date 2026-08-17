use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use futures::future::BoxFuture;
use tokio::sync::{oneshot, watch};
use tokio::task::JoinError;

use crate::agent_loop::{agent_loop, agent_loop_continue, AgentRun};
use crate::types::{
    AfterToolCallHook, AgentContext, AgentEvent, AgentLoopConfig, AgentState, AssistantMessage,
    BeforeToolCallHook, ContentBlock, ConvertToLlm, Message, PrepareNextTurnHook, QueueMode,
    ShouldStopAfterTurnHook, StopReason, StreamFn, StreamFnOptions, ThinkingLevel,
    ToolExecutionMode, TransformContext, UserMessage,
};

/// A callback that receives every agent event plus the active abort signal.
pub type Listener =
    Arc<dyn Fn(AgentEvent, watch::Receiver<bool>) -> BoxFuture<'static, ()> + Send + Sync>;

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

struct ActiveRun {
    promise: oneshot::Receiver<()>,
}

/// A cloneable handle that can abort an active [`Agent`] run.
#[derive(Clone)]
pub struct AgentAbortHandle {
    abort_tx: watch::Sender<bool>,
}

impl AgentAbortHandle {
    pub fn abort(&self) {
        let _ = self.abort_tx.send(true);
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

    listeners: Vec<Listener>,
    steering_queue: PendingMessageQueue,
    follow_up_queue: PendingMessageQueue,
    active_run: Option<ActiveRun>,
    abort_tx: watch::Sender<bool>,
}
impl Agent {
    pub fn new(stream_fn: Arc<dyn StreamFn>) -> Self {
        let (abort_tx, _abort_rx) = watch::channel(false);
        Self {
            state: AgentState::default(),
            stream_fn,
            before_tool_call: None,
            after_tool_call: None,
            should_stop_after_turn: None,
            prepare_next_turn: None,
            convert_to_llm: Arc::new(|m| m),
            transform_context: None,
            tool_execution: ToolExecutionMode::default(),
            stream_fn_options: StreamFnOptions::default(),
            listeners: Vec::new(),
            steering_queue: PendingMessageQueue::new(QueueMode::OneAtATime),
            follow_up_queue: PendingMessageQueue::new(QueueMode::OneAtATime),
            active_run: None,
            abort_tx,
        }
    }

    pub fn subscribe(&mut self, listener: Listener) -> impl FnOnce() {
        self.listeners.push(listener);
        let idx = self.listeners.len() - 1;
        move || {
            // Best-effort removal is not needed for the embryo.
            let _ = idx;
        }
    }

    pub fn steering_mode(&self) -> QueueMode {
        self.steering_queue.mode()
    }

    pub fn set_steering_mode(&mut self, mode: QueueMode) {
        self.steering_queue.set_mode(mode);
    }

    pub fn follow_up_mode(&self) -> QueueMode {
        self.follow_up_queue.mode()
    }

    pub fn set_follow_up_mode(&mut self, mode: QueueMode) {
        self.follow_up_queue.set_mode(mode);
    }

    pub fn steer(&mut self, message: Message) {
        self.steering_queue.enqueue(message);
    }

    pub fn follow_up(&mut self, message: Message) {
        self.follow_up_queue.enqueue(message);
    }

    pub fn clear_steering_queue(&mut self) {
        self.steering_queue.clear();
    }

    pub fn clear_follow_up_queue(&mut self) {
        self.follow_up_queue.clear();
    }

    pub fn clear_all_queues(&mut self) {
        self.clear_steering_queue();
        self.clear_follow_up_queue();
    }

    pub fn has_queued_messages(&self) -> bool {
        self.steering_queue.has_items() || self.follow_up_queue.has_items()
    }

    pub fn signal(&self) -> Option<watch::Receiver<bool>> {
        if self.active_run.is_some() {
            Some(self.abort_tx.subscribe())
        } else {
            None
        }
    }

    pub fn abort_handle(&self) -> AgentAbortHandle {
        AgentAbortHandle {
            abort_tx: self.abort_tx.clone(),
        }
    }

    pub fn abort(&self) {
        let _ = self.abort_tx.send(true);
    }

    pub async fn wait_for_idle(&mut self) {
        if let Some(run) = self.active_run.take() {
            let _ = run.promise.await;
        }
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

    /// Start a new prompt from text or messages.
    pub async fn prompt(&mut self, input: impl Into<PromptInput>) {
        if self.active_run.is_some() {
            panic!(
                "Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."
            );
        }
        let messages = input.into().into_messages();
        self.run_prompt_messages(messages, false).await;
    }

    /// Continue from the current transcript.
    pub async fn continue_run(&mut self) {
        if self.active_run.is_some() {
            panic!("Agent is already processing. Wait for completion before continuing.");
        }

        let last_role = self.state.messages.last().map(|m| m.role());
        if last_role == Some("assistant") {
            let steering = self.steering_queue.drain();
            if !steering.is_empty() {
                self.run_prompt_messages(steering, true).await;
                return;
            }
            let follow_ups = self.follow_up_queue.drain();
            if !follow_ups.is_empty() {
                self.run_prompt_messages(follow_ups, false).await;
                return;
            }
            panic!("Cannot continue from message role: assistant");
        }

        self.run_continuation().await;
    }

    async fn run_prompt_messages(
        &mut self,
        messages: Vec<Message>,
        skip_initial_steering_poll: bool,
    ) {
        let _ = self.abort_tx.send(false);
        let abort_tx = self.abort_tx.clone();
        let abort_rx = abort_tx.subscribe();
        let (done_tx, done_rx) = oneshot::channel();

        let context = self.create_context_snapshot();
        let config = self.create_loop_config(skip_initial_steering_poll);
        let stream_fn = self.stream_fn.clone();
        let listeners: Vec<Listener> = self.listeners.clone();
        let message_prefix = self.state.messages.len();

        self.state.is_streaming = true;
        self.state.streaming_message = None;
        self.state.error_message = None;

        self.active_run = Some(ActiveRun { promise: done_rx });

        let mut run = agent_loop(messages, context, config, abort_rx, stream_fn);
        let run_error = self.drain_run_events(&mut run, &listeners, &abort_tx).await;
        if let Some(error) = run_error {
            self.complete_failed_run(&listeners, &abort_tx, error, message_prefix)
                .await;
        }
        let _ = done_tx.send(());

        self.finish_run();
    }

    async fn run_continuation(&mut self) {
        let _ = self.abort_tx.send(false);
        let abort_tx = self.abort_tx.clone();
        let abort_rx = abort_tx.subscribe();
        let (done_tx, done_rx) = oneshot::channel();

        let context = self.create_context_snapshot();
        let config = self.create_loop_config(false);
        let stream_fn = self.stream_fn.clone();
        let listeners: Vec<Listener> = self.listeners.clone();
        let message_prefix = self.state.messages.len();

        self.state.is_streaming = true;
        self.state.streaming_message = None;
        self.state.error_message = None;

        self.active_run = Some(ActiveRun { promise: done_rx });

        let mut run = agent_loop_continue(context, config, abort_rx, stream_fn);
        let run_error = self.drain_run_events(&mut run, &listeners, &abort_tx).await;
        if let Some(error) = run_error {
            self.complete_failed_run(&listeners, &abort_tx, error, message_prefix)
                .await;
        }
        let _ = done_tx.send(());

        self.finish_run();
    }

    /// Drain a low-level run, settling listeners per event, and surface a join
    /// failure once the buffered events are exhausted.
    async fn drain_run_events(
        &mut self,
        run: &mut AgentRun,
        listeners: &[Listener],
        abort_tx: &watch::Sender<bool>,
    ) -> Option<JoinError> {
        while let Some(event) = run.next_event().await {
            self.process_event(&event);
            for listener in listeners {
                listener(event.clone(), abort_tx.subscribe()).await;
            }
        }
        run.result().await.err()
    }

    /// Append a synthetic assistant error terminal lifecycle so a failed run
    /// still settles listeners before state is marked idle.
    async fn complete_failed_run(
        &mut self,
        listeners: &[Listener],
        abort_tx: &watch::Sender<bool>,
        error: JoinError,
        message_prefix: usize,
    ) {
        let assistant_message = AssistantMessage {
            content: vec![ContentBlock::text("")],
            stop_reason: StopReason::Error,
            error_message: Some(format!("agent run failed: {error}")),
            ..Default::default()
        };
        let message = Message::Assistant(assistant_message);

        // Process and settle the synthetic message/turn events first so the
        // error message is in `state.messages` before AgentEnd is built.
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
            for listener in listeners {
                listener(event.clone(), abort_tx.subscribe()).await;
            }
        }

        let agent_end = AgentEvent::AgentEnd {
            messages: self.state.messages[message_prefix..].to_vec(),
        };
        self.process_event(&agent_end);
        for listener in listeners {
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
        self.active_run = None;
        let _ = self.abort_tx.send(false);
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
            provider: self.state.provider.clone(),
            api: self.state.api.clone(),
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
        match self.state.thinking_level {
            ThinkingLevel::Off => None,
            other => Some(other),
        }
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
    use crate::types::{AssistantEventStream, LlmContext, StopReason, StreamFnOptions};
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

        agent.prompt("hi").await;

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
        agent.active_run = Some(ActiveRun {
            promise: oneshot::channel().1,
        });
        let handle = tokio::spawn(async move {
            agent.prompt("hi").await;
        });
        assert!(handle.await.is_err());
    }

    #[tokio::test]
    async fn reset_rejects_during_run() {
        let mut agent = Agent::new(simple_text_response("Hello"));
        agent.active_run = Some(ActiveRun {
            promise: oneshot::channel().1,
        });
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

    struct PanickingStreamFn;

    #[async_trait]
    impl StreamFn for PanickingStreamFn {
        async fn call(
            &self,
            _model: String,
            _context: LlmContext,
            _options: StreamFnOptions,
            _abort: watch::Receiver<bool>,
        ) -> Result<Box<dyn AssistantEventStream>, String> {
            panic!("stream provider panicked");
        }
    }

    #[tokio::test]
    async fn panicking_provider_yields_complete_error_lifecycle() {
        let mut agent = Agent::new(Arc::new(PanickingStreamFn));
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AgentEvent>();
        let _ = agent.subscribe(Arc::new(move |event, _signal| {
            let tx = tx.clone();
            Box::pin(async move {
                let _ = tx.send(event);
            })
        }));

        agent.prompt("hi").await;

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
