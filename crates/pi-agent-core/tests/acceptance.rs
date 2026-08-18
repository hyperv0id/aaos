use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use pi_agent_core::agent::Agent;
use pi_agent_core::agent_loop::agent_loop_continue;
use pi_agent_core::stream::{mock_stream_fn, simple_text_response, MockAssistantStream};
use pi_agent_core::trace::{TraceCollector, TraceEntry};
use pi_agent_core::types::{
    AfterToolCallResult, AgentContext, AgentEvent, AgentLoopConfig, AgentTool, AgentToolResult,
    AssistantMessage, AssistantMessageEvent, BeforeToolCallResult, ContentBlock, Message,
    QueueMode, StopReason, StreamFn, StreamFnOptions, ToolCall, ToolExecutionMode, UserMessage,
};
use serde_json::{json, Value};
use tokio::sync::watch;

fn text_msg(text: &str) -> Message {
    Message::User(UserMessage::new(text))
}

fn assistant_text(text: &str) -> AssistantMessage {
    AssistantMessage::text(text)
}

fn assistant_tool_use(calls: Vec<ToolCall>, stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content: calls.into_iter().map(ContentBlock::ToolCall).collect(),
        stop_reason,
        ..Default::default()
    }
}

fn tool_call(id: &str, name: &str, args: Value) -> ToolCall {
    ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: args,
    }
}

fn make_agent(stream_fn: Arc<dyn StreamFn>, tools: Vec<Arc<dyn AgentTool>>) -> Agent {
    let mut agent = Agent::new(stream_fn);
    agent.state.model = "test".into();
    agent.state.provider = "test".into();
    agent.state.api = "test".into();
    agent.state.tools = tools;
    agent
}

fn subscribe_trace(agent: &mut Agent) -> Arc<Mutex<TraceCollector>> {
    let trace = Arc::new(Mutex::new(TraceCollector::new()));
    let trace2 = trace.clone();
    let _ = agent.subscribe(Arc::new(move |event, _signal| {
        let trace = trace2.clone();
        Box::pin(async move {
            trace.lock().unwrap().observe_event(&event);
        })
    }));
    trace
}

struct EchoTool {
    name: String,
    log: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl AgentTool for EchoTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn label(&self) -> &str {
        "Echo"
    }
    fn description(&self) -> &str {
        "echo"
    }
    async fn execute(
        &self,
        _tool_call_id: String,
        params: Value,
        _signal: Option<&watch::Receiver<bool>>,
        _on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        self.log.lock().unwrap().push(params.clone());
        Ok(AgentToolResult::text(format!("echoed: {}", params)))
    }
}

struct SlowEchoTool {
    first_started: Arc<AtomicBool>,
    release_first: Arc<tokio::sync::Notify>,
    log: Arc<Mutex<Vec<Value>>>,
}

#[async_trait]
impl AgentTool for SlowEchoTool {
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
        _tool_call_id: String,
        params: Value,
        _signal: Option<&watch::Receiver<bool>>,
        _on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        let v = params.get("v").and_then(|v| v.as_str()).unwrap_or("");
        if v == "first" {
            self.first_started.store(true, Ordering::SeqCst);
            self.release_first.notified().await;
        }
        self.log.lock().unwrap().push(params.clone());
        Ok(AgentToolResult::text(format!("echoed: {}", params)))
    }
}

struct RecordingSequentialTool {
    order: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentTool for RecordingSequentialTool {
    fn name(&self) -> &str {
        "seq"
    }
    fn label(&self) -> &str {
        "Seq"
    }
    fn description(&self) -> &str {
        "seq"
    }
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Sequential
    }
    async fn execute(
        &self,
        _tool_call_id: String,
        params: Value,
        _signal: Option<&watch::Receiver<bool>>,
        _on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        let v = params
            .get("v")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        self.order.lock().unwrap().push(v);
        Ok(AgentToolResult::text("ok"))
    }
}

struct BlockingSeqTool {
    first_started: Arc<AtomicBool>,
    release_first: Arc<tokio::sync::Notify>,
    order: Arc<Mutex<Vec<String>>>,
}

#[async_trait]
impl AgentTool for BlockingSeqTool {
    fn name(&self) -> &str {
        "seq"
    }
    fn label(&self) -> &str {
        "Seq"
    }
    fn description(&self) -> &str {
        "seq"
    }
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Sequential
    }
    async fn execute(
        &self,
        _tool_call_id: String,
        params: Value,
        _signal: Option<&watch::Receiver<bool>>,
        _on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
    ) -> Result<AgentToolResult, String> {
        let v = params
            .get("v")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if v == "first" {
            self.first_started.store(true, Ordering::SeqCst);
            self.release_first.notified().await;
        }
        self.order.lock().unwrap().push(v);
        Ok(AgentToolResult::text("ok"))
    }
}

struct FailingStreamFn;

#[async_trait]
impl StreamFn for FailingStreamFn {
    async fn call(
        &self,
        _model: String,
        _context: pi_agent_core::types::LlmContext,
        _options: StreamFnOptions,
        _abort: watch::Receiver<bool>,
    ) -> Result<Box<dyn pi_agent_core::types::AssistantEventStream>, String> {
        Err("provider exploded".into())
    }
}

struct HangingStreamFn;

#[async_trait]
impl StreamFn for HangingStreamFn {
    async fn call(
        &self,
        _model: String,
        _context: pi_agent_core::types::LlmContext,
        _options: StreamFnOptions,
        mut abort: watch::Receiver<bool>,
    ) -> Result<Box<dyn pi_agent_core::types::AssistantEventStream>, String> {
        let _ = abort.changed().await;
        assert!(*abort.borrow());
        let msg = AssistantMessage {
            content: vec![ContentBlock::Text { text: "".into() }],
            stop_reason: StopReason::Aborted,
            error_message: Some("aborted".into()),
            ..Default::default()
        };
        Ok(Box::new(MockAssistantStream::new(msg)))
    }
}

#[tokio::test]
async fn text_only_turn() {
    let mut agent = make_agent(simple_text_response("Hello"), vec![]);
    let trace = subscribe_trace(&mut agent);

    agent.prompt("hi").await;

    let entries = trace.lock().unwrap().entries().to_vec();
    assert_eq!(
        entries[0],
        TraceEntry::Event {
            event_type: "agent_start".into()
        }
    );
    assert_eq!(
        entries[1],
        TraceEntry::Event {
            event_type: "turn_start".into()
        }
    );
    assert!(matches!(entries[2], TraceEntry::MessageStart { ref role } if role == "user"));
    assert!(matches!(entries[3], TraceEntry::MessageEnd { ref role, .. } if role == "user"));
    assert!(matches!(entries[4], TraceEntry::MessageStart { ref role } if role == "assistant"));
    assert!(
        matches!(entries[5], TraceEntry::MessageEnd { ref role, stop_reason: Some(ref s) } if role == "assistant" && s == "stop")
    );
    assert!(
        matches!(entries[6], TraceEntry::TurnEnd { ref tool_result_ids } if tool_result_ids.is_empty())
    );
    assert_eq!(
        entries[7],
        TraceEntry::Event {
            event_type: "agent_end".into()
        }
    );
    assert_eq!(agent.state.messages.len(), 2);
}

#[tokio::test]
async fn streaming_updates_carry_latest_partial() {
    let stream_fn = mock_stream_fn(move |_model, _context, _options| {
        let mut mock = MockAssistantStream::new(assistant_text("Hi!"));
        mock.push(AssistantMessageEvent::Start {
            partial: assistant_text(""),
        });
        mock.push(AssistantMessageEvent::TextStart);
        mock.push(AssistantMessageEvent::TextDelta {
            text: "H".to_string(),
        });
        mock.push(AssistantMessageEvent::TextDelta {
            text: "i".to_string(),
        });
        mock.push(AssistantMessageEvent::TextDelta {
            text: "!".to_string(),
        });
        mock.push(AssistantMessageEvent::TextEnd);
        mock.push(AssistantMessageEvent::Done);
        Box::new(mock)
    });

    let mut agent = make_agent(stream_fn, vec![]);
    let updates = Arc::new(Mutex::new(Vec::new()));
    let updates2 = updates.clone();
    let _ = agent.subscribe(Arc::new(move |event, _signal| {
        let updates = updates2.clone();
        Box::pin(async move {
            if let AgentEvent::MessageUpdate {
                message,
                assistant_event: AssistantMessageEvent::TextDelta { .. },
            } = event
            {
                let text = message
                    .as_assistant()
                    .map(|assistant| {
                        assistant
                            .content
                            .iter()
                            .filter_map(|block| match block {
                                ContentBlock::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<String>()
                    })
                    .unwrap_or_default();
                updates.lock().unwrap().push(text);
            }
        })
    }));

    agent.prompt("stream").await;

    let updates = updates.lock().unwrap().clone();
    assert_eq!(
        updates,
        vec!["H".to_string(), "Hi".to_string(), "Hi!".to_string()]
    );
    assert_eq!(agent.state.messages.len(), 2);
}

#[tokio::test]
async fn one_tool_call_then_final_response() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            let c = calls2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(assistant_tool_use(
                    vec![tool_call("c1", "echo", json!({"v": "hello"}))],
                    StopReason::ToolUse,
                )))
            } else {
                Box::new(MockAssistantStream::new(assistant_text("done")))
            }
        }),
        vec![Arc::new(EchoTool {
            name: "echo".into(),
            log: Arc::new(Mutex::new(vec![])),
        })],
    );
    let trace = subscribe_trace(&mut agent);

    agent.prompt("call echo").await;

    let entries = trace.lock().unwrap().entries().to_vec();
    assert!(entries.iter().any(
        |e| matches!(e, TraceEntry::ToolExecutionStart { tool_name, .. } if tool_name == "echo")
    ));
    assert!(entries.iter().any(|e| matches!(e, TraceEntry::ToolExecutionEnd { tool_name, is_error: false, .. } if tool_name == "echo")));
    assert!(entries
        .iter()
        .any(|e| matches!(e, TraceEntry::MessageStart { role } if role == "toolResult")));
    assert_eq!(agent.state.messages.len(), 4);
}

#[tokio::test]
async fn two_parallel_tool_calls_completion_order_vs_source_order() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let first_started = Arc::new(AtomicBool::new(false));
    let release_first = Arc::new(tokio::sync::Notify::new());

    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            let c = calls2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(assistant_tool_use(
                    vec![
                        tool_call("c1", "echo", json!({"v": "first"})),
                        tool_call("c2", "echo", json!({"v": "second"})),
                    ],
                    StopReason::ToolUse,
                )))
            } else {
                Box::new(MockAssistantStream::new(assistant_text("done")))
            }
        }),
        vec![],
    );

    let tool_log = Arc::new(Mutex::new(Vec::new()));
    agent.state.tools = vec![Arc::new(SlowEchoTool {
        first_started: first_started.clone(),
        release_first: release_first.clone(),
        log: tool_log.clone(),
    })];
    let trace = subscribe_trace(&mut agent);

    let release_handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(5)).await;
            if first_started.load(Ordering::SeqCst) {
                release_first.notify_one();
                break;
            }
        }
    });

    agent.prompt("call both").await;
    let _ = release_handle.await;

    let entries = trace.lock().unwrap().entries().to_vec();
    let tool_ends: Vec<_> = entries
        .iter()
        .filter_map(|e| match e {
            TraceEntry::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_ends, vec!["c2", "c1"]);

    let turn_end = entries.iter().find_map(|e| match e {
        TraceEntry::TurnEnd { tool_result_ids } => Some(tool_result_ids.clone()),
        _ => None,
    });
    assert_eq!(turn_end, Some(vec!["c1".into(), "c2".into()]));

    let log = tool_log.lock().unwrap();
    assert!(log.contains(&json!({"v": "first"})));
    assert!(log.contains(&json!({"v": "second"})));
}

#[tokio::test]
async fn sequential_execution_override() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let order = Arc::new(Mutex::new(Vec::new()));
    let order2 = order.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            let c = calls2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(assistant_tool_use(
                    vec![
                        tool_call("c1", "seq", json!({"v": "a"})),
                        tool_call("c2", "seq", json!({"v": "b"})),
                    ],
                    StopReason::ToolUse,
                )))
            } else {
                Box::new(MockAssistantStream::new(assistant_text("done")))
            }
        }),
        vec![Arc::new(RecordingSequentialTool { order: order2 })],
    );
    let trace = subscribe_trace(&mut agent);

    agent.prompt("run sequential").await;

    let entries = trace.lock().unwrap().entries().to_vec();
    let tool_ends: Vec<_> = entries
        .iter()
        .filter_map(|e| match e {
            TraceEntry::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_ends, vec!["c1", "c2"]);

    let logged = order.lock().unwrap();
    assert_eq!(logged.len(), 2);
    assert_eq!(logged[0], "a");
    assert_eq!(logged[1], "b");
}

#[tokio::test]
async fn steering_queue_one_at_a_time() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            calls2.fetch_add(1, Ordering::SeqCst);
            Box::new(MockAssistantStream::new(assistant_text("ok")))
        }),
        vec![],
    );
    agent.set_steering_mode(QueueMode::OneAtATime);
    agent.steer(text_msg("steer-1"));
    agent.steer(text_msg("steer-2"));
    let trace = subscribe_trace(&mut agent);

    agent.prompt("start").await;

    let entries = trace.lock().unwrap().entries().to_vec();

    // Steering messages are polled before the first assistant response:
    // start, steer-1 (initial poll), assistant, steer-2 (after turn 1), assistant.
    let user_count = entries
        .iter()
        .filter(|e| matches!(e, TraceEntry::MessageStart { role } if role == "user"))
        .count();
    assert_eq!(user_count, 3); // start + steer-1 + steer-2

    // The first assistant MessageEnd must come after steer-1's MessageEnd.
    let steer1_end = entries
        .iter()
        .position(|e| matches!(e, TraceEntry::MessageEnd { ref role, .. } if role == "user"))
        .expect("at least one user MessageEnd");
    let first_asst_end = entries
        .iter()
        .position(|e| matches!(e, TraceEntry::MessageEnd { ref role, .. } if role == "assistant"))
        .expect("at least one assistant MessageEnd");
    assert!(
        steer1_end < first_asst_end,
        "steering message must appear before first assistant MessageEnd"
    );

    // Two assistant turns: steer-1 in turn 1, steer-2 in turn 2.
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn steering_queue_all() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            calls2.fetch_add(1, Ordering::SeqCst);
            Box::new(MockAssistantStream::new(assistant_text("ok")))
        }),
        vec![],
    );
    agent.set_steering_mode(QueueMode::All);
    agent.steer(text_msg("steer-1"));
    agent.steer(text_msg("steer-2"));
    let trace = subscribe_trace(&mut agent);

    agent.prompt("start").await;

    let entries = trace.lock().unwrap().entries().to_vec();

    // Both steering messages are polled before the first assistant response.
    let user_count = entries
        .iter()
        .filter(|e| matches!(e, TraceEntry::MessageStart { role } if role == "user"))
        .count();
    assert_eq!(user_count, 3); // start + steer-1 + steer-2

    // Both steering MessageEnds must come before the first assistant MessageEnd.
    let user_ends: Vec<_> = entries
        .iter()
        .filter(|e| matches!(e, TraceEntry::MessageEnd { ref role, .. } if role == "user"))
        .collect();
    assert_eq!(user_ends.len(), 3); // start + steer-1 + steer-2
    let first_asst_end = entries
        .iter()
        .position(|e| matches!(e, TraceEntry::MessageEnd { ref role, .. } if role == "assistant"))
        .expect("at least one assistant MessageEnd");
    // The third user MessageEnd (steer-2) must precede the first assistant MessageEnd.
    let steer2_end = entries
        .iter()
        .rposition(|e| matches!(e, TraceEntry::MessageEnd { ref role, .. } if role == "user"))
        .expect("at least one user MessageEnd");
    assert!(
        steer2_end < first_asst_end,
        "both steering messages must appear before first assistant MessageEnd"
    );

    // Single assistant turn: both steering messages injected before it.
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn continue_with_steering_skips_initial_poll() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            calls2.fetch_add(1, Ordering::SeqCst);
            Box::new(MockAssistantStream::new(assistant_text("ok")))
        }),
        vec![],
    );
    agent.set_steering_mode(QueueMode::All);
    let trace = subscribe_trace(&mut agent);

    // First prompt to establish an assistant tail.
    agent.prompt("initial").await;

    // Queue two steering messages, then continue from the assistant tail.
    agent.steer(text_msg("steer-1"));
    agent.steer(text_msg("steer-2"));
    agent.continue_run().await;

    let entries = trace.lock().unwrap().entries().to_vec();

    // Both steering messages should appear as user messages in the trace.
    let user_count = entries
        .iter()
        .filter(|e| matches!(e, TraceEntry::MessageStart { role } if role == "user"))
        .count();
    assert_eq!(user_count, 3); // initial + steer-1 + steer-2

    // The continue run should produce exactly one assistant response.
    // skip_initial_steering_poll prevents the hook from double-draining.
    assert_eq!(calls.load(Ordering::SeqCst), 2); // 1 prompt + 1 continue
}

#[tokio::test]
async fn follow_up_queue_one_at_a_time() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            calls2.fetch_add(1, Ordering::SeqCst);
            Box::new(MockAssistantStream::new(assistant_text("ok")))
        }),
        vec![],
    );
    agent.set_follow_up_mode(QueueMode::OneAtATime);
    agent.follow_up(text_msg("follow-1"));
    agent.follow_up(text_msg("follow-2"));
    let trace = subscribe_trace(&mut agent);

    agent.prompt("start").await;

    let entries = trace.lock().unwrap().entries().to_vec();
    let user_count = entries
        .iter()
        .filter(|e| matches!(e, TraceEntry::MessageStart { role } if role == "user"))
        .count();
    assert_eq!(user_count, 3); // start + follow-1 (turn 2) + follow-2 (turn 3)
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[tokio::test]
async fn follow_up_queue_all() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            calls2.fetch_add(1, Ordering::SeqCst);
            Box::new(MockAssistantStream::new(assistant_text("ok")))
        }),
        vec![],
    );
    agent.set_follow_up_mode(QueueMode::All);
    agent.follow_up(text_msg("follow-1"));
    agent.follow_up(text_msg("follow-2"));
    let trace = subscribe_trace(&mut agent);

    agent.prompt("start").await;

    let entries = trace.lock().unwrap().entries().to_vec();
    let user_count = entries
        .iter()
        .filter(|e| matches!(e, TraceEntry::MessageStart { role } if role == "user"))
        .count();
    assert_eq!(user_count, 3); // start + follow-1 + follow-2 (both in turn 2)
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn abort_stops_sequential_tool_batch() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let first_started = Arc::new(AtomicBool::new(false));
    let release_first = Arc::new(tokio::sync::Notify::new());
    let order = Arc::new(Mutex::new(Vec::new()));
    let order2 = order.clone();

    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            let c = calls2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(assistant_tool_use(
                    vec![
                        tool_call("c1", "seq", json!({"v": "first"})),
                        tool_call("c2", "seq", json!({"v": "second"})),
                        tool_call("c3", "seq", json!({"v": "third"})),
                    ],
                    StopReason::ToolUse,
                )))
            } else {
                Box::new(MockAssistantStream::new(assistant_text("done")))
            }
        }),
        vec![],
    );
    agent.state.tools = vec![Arc::new(BlockingSeqTool {
        first_started: first_started.clone(),
        release_first: release_first.clone(),
        order: order2,
    })];
    let trace = subscribe_trace(&mut agent);
    let abort_handle = agent.abort_handle();

    let helper = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(5)).await;
            if first_started.load(Ordering::SeqCst) {
                break;
            }
        }
        abort_handle.abort();
        release_first.notify_one();
    });

    agent.prompt("run sequential").await;
    let _ = helper.await;

    let entries = trace.lock().unwrap().entries().to_vec();
    let tool_ends: Vec<_> = entries
        .iter()
        .filter_map(|e| match e {
            TraceEntry::ToolExecutionEnd { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(tool_ends, vec!["c1".to_string()]);

    let logged = order.lock().unwrap();
    assert_eq!(logged.len(), 1);
    assert_eq!(logged[0], "first");

    assert_eq!(
        entries.last(),
        Some(&TraceEntry::Event {
            event_type: "agent_end".into()
        })
    );
}

#[tokio::test]
async fn abort_checked_before_tool_preparation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            let c = calls2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(assistant_tool_use(
                    vec![
                        tool_call("c1", "echo", json!({"v": "a"})),
                        tool_call("c2", "echo", json!({"v": "b"})),
                        tool_call("c3", "echo", json!({"v": "c"})),
                    ],
                    StopReason::ToolUse,
                )))
            } else {
                Box::new(MockAssistantStream::new(assistant_text("done")))
            }
        }),
        vec![Arc::new(EchoTool {
            name: "echo".into(),
            log: Arc::new(Mutex::new(vec![])),
        })],
    );

    let abort_handle = agent.abort_handle();
    agent.before_tool_call = Some(Arc::new(move |_ctx| {
        let handle = abort_handle.clone();
        Box::pin(async move {
            handle.abort();
            Ok(BeforeToolCallResult::default())
        })
    }));
    let trace = subscribe_trace(&mut agent);

    agent.prompt("call all").await;

    let entries = trace.lock().unwrap().entries().to_vec();
    let starts: Vec<_> = entries
        .iter()
        .filter_map(|e| match e {
            TraceEntry::ToolExecutionStart { tool_call_id, .. } => Some(tool_call_id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(starts, vec!["c1".to_string()]);

    assert_eq!(
        entries.last(),
        Some(&TraceEntry::Event {
            event_type: "agent_end".into()
        })
    );
}

#[tokio::test]
async fn abort_while_provider_pending() {
    let mut agent = make_agent(Arc::new(HangingStreamFn), vec![]);
    let trace = subscribe_trace(&mut agent);
    let abort_handle = agent.abort_handle();

    let handle = tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(20)).await;
        abort_handle.abort();
    });

    agent.prompt("start").await;
    let _ = handle.await;

    let entries = trace.lock().unwrap().entries().to_vec();
    assert!(entries.iter().any(
        |e| matches!(e, TraceEntry::MessageEnd { stop_reason: Some(ref s), .. } if s == "aborted")
    ));
    assert_eq!(
        entries.last(),
        Some(&TraceEntry::Event {
            event_type: "agent_end".into()
        })
    );
}

#[tokio::test]
async fn provider_throw_converts_to_error_lifecycle() {
    let mut agent = make_agent(Arc::new(FailingStreamFn), vec![]);
    let trace = subscribe_trace(&mut agent);

    agent.prompt("start").await;

    let entries = trace.lock().unwrap().entries().to_vec();
    assert!(entries.iter().any(
        |e| matches!(e, TraceEntry::MessageEnd { stop_reason: Some(ref s), .. } if s == "error")
    ));
    assert!(entries
        .iter()
        .any(|e| matches!(e, TraceEntry::MessageEnd { ref role, .. } if role == "assistant")));
    assert_eq!(
        entries.last(),
        Some(&TraceEntry::Event {
            event_type: "agent_end".into()
        })
    );
    assert_eq!(agent.state.error_message, Some("provider exploded".into()));
}

#[tokio::test]
async fn length_truncated_tool_call_never_executes() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            let c = calls2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(assistant_tool_use(
                    vec![tool_call("c1", "echo", json!({"v": "hel"}))],
                    StopReason::Length,
                )))
            } else {
                Box::new(MockAssistantStream::new(assistant_text("retry")))
            }
        }),
        vec![Arc::new(EchoTool {
            name: "echo".into(),
            log: Arc::new(Mutex::new(vec![])),
        })],
    );
    let trace = subscribe_trace(&mut agent);

    agent.prompt("call echo").await;

    let entries = trace.lock().unwrap().entries().to_vec();
    assert!(
        entries
            .iter()
            .any(|e| matches!(e, TraceEntry::ToolExecutionEnd { tool_name, is_error: true, .. } if tool_name == "echo"))
    );
    assert_eq!(agent.state.messages.len(), 4);
}

#[tokio::test]
async fn before_and_after_tool_hooks_and_terminate() {
    let calls = Arc::new(AtomicUsize::new(0));
    let calls2 = calls.clone();
    let mut agent = make_agent(
        mock_stream_fn(move |_model, _ctx, _opts| {
            let c = calls2.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Box::new(MockAssistantStream::new(assistant_tool_use(
                    vec![
                        tool_call("c1", "blocked", json!({"v": "blocked"})),
                        tool_call("c2", "echo", json!({"v": "hello"})),
                    ],
                    StopReason::ToolUse,
                )))
            } else {
                Box::new(MockAssistantStream::new(assistant_text("should not run")))
            }
        }),
        vec![
            Arc::new(EchoTool {
                name: "blocked".into(),
                log: Arc::new(Mutex::new(vec![])),
            }),
            Arc::new(EchoTool {
                name: "echo".into(),
                log: Arc::new(Mutex::new(vec![])),
            }),
        ],
    );

    agent.before_tool_call = Some(Arc::new(|ctx| {
        Box::pin(async move {
            Ok(BeforeToolCallResult {
                block: ctx.tool_call.name == "blocked",
                reason: Some("Blocked by policy".into()),
                terminate: ctx.tool_call.name == "blocked",
            })
        })
    }));
    agent.after_tool_call = Some(Arc::new(|_ctx| {
        Box::pin(async move {
            Ok(AfterToolCallResult {
                content: Some(vec![ContentBlock::text("after hook")]),
                is_error: Some(true),
                terminate: Some(true),
                ..Default::default()
            })
        })
    }));

    let trace = subscribe_trace(&mut agent);

    agent.prompt("call echo").await;

    let entries = trace.lock().unwrap().entries().to_vec();
    for tool_name in ["blocked", "echo"] {
        assert!(entries.iter().any(|entry| {
            matches!(
                entry,
                TraceEntry::ToolExecutionEnd {
                    tool_name: actual,
                    is_error: true,
                    ..
                } if actual == tool_name
            )
        }));
    }
    let echo_result = agent
        .state
        .messages
        .iter()
        .filter_map(Message::as_tool_result)
        .find(|result| result.tool_name == "echo")
        .expect("echo tool result");
    assert_eq!(echo_result.content, vec![ContentBlock::text("after hook")]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn agent_loop_continue_no_user_message_events() {
    let mut context = AgentContext::empty();
    context.messages = vec![text_msg("Hello")];
    let config = AgentLoopConfig {
        model: "test".into(),
        provider: "test".into(),
        api: "test".into(),
        ..Default::default()
    };

    let mut run = agent_loop_continue(
        context,
        config,
        watch::channel(false).1,
        simple_text_response("Response"),
    );

    let mut events = Vec::new();
    while let Some(event) = run.next_event().await {
        events.push(event);
    }
    let result = run.result().await.expect("run should not panic");

    let user_events: Vec<_> = events
        .iter()
        .filter(|e| matches!(e, AgentEvent::MessageEnd { message } if message.role() == "user"))
        .collect();
    assert!(user_events.is_empty());
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].role(), "assistant");
}
