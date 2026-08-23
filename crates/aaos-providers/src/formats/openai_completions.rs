//! OpenAI Chat Completions streaming API format (SSE), also used by every
//! OpenAI-compatible endpoint.

use std::collections::BTreeMap;
use std::error::Error as StdError;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use pi_agent_core::types::{
    AgentTool, AssistantEventStream, AssistantMessage, AssistantMessageEvent, ContentBlock,
    LlmContext, Message, Model, StopReason, StreamFn, StreamFnOptions, ThinkingLevel, ToolCall,
};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::sync::watch;

/// `Model::api` key dispatching to this format.
pub const API: &str = "openai-completions";

pub fn reasoning_effort(level: ThinkingLevel) -> Option<&'static str> {
    match level {
        ThinkingLevel::Off => None,
        ThinkingLevel::Minimal => Some("minimal"),
        ThinkingLevel::Low => Some("low"),
        ThinkingLevel::Medium => Some("medium"),
        ThinkingLevel::High => Some("high"),
        ThinkingLevel::XHigh => Some("xhigh"),
        ThinkingLevel::Max => Some("max"),
    }
}

async fn wait_aborted(abort: &mut watch::Receiver<bool>) {
    if *abort.borrow() {
        return;
    }
    let _ = abort.wait_for(|v| *v).await;
}

/// reqwest's Display omits the source chain, so "error sending request for url"
/// hides timeouts and TLS failures unless we walk `.source()`.
fn error_chain(err: &dyn StdError) -> String {
    let mut msg = err.to_string();
    let mut source = err.source();
    while let Some(inner) = source {
        msg.push_str(": ");
        msg.push_str(&inner.to_string());
        source = inner.source();
    }
    msg
}

fn chat_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat/completions") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat/completions")
    }
}

fn tools_payload(tools: &[Arc<dyn AgentTool>]) -> Option<Value> {
    if tools.is_empty() {
        return None;
    }
    Some(Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "type": "function",
                    "function": {
                        "name": t.name(),
                        "description": t.description(),
                        "parameters": t.parameters()
                    }
                })
            })
            .collect(),
    ))
}

fn content_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

pub fn build_request_body(model: &Model, context: &LlmContext, options: &StreamFnOptions) -> Value {
    let mut messages = Vec::new();
    if !context.system_prompt.is_empty() {
        messages.push(json!({"role": "system", "content": context.system_prompt}));
    }
    for msg in &context.messages {
        match msg {
            Message::User(u) => messages.push(json!({
                "role": "user",
                "content": content_text(&u.content)
            })),
            Message::Assistant(a) => {
                let mut tool_calls = Vec::new();
                for block in &a.content {
                    if let ContentBlock::ToolCall(tc) = block {
                        tool_calls.push(json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments.to_string()
                            }
                        }));
                    }
                }
                let mut obj = serde_json::Map::new();
                obj.insert("role".into(), json!("assistant"));
                obj.insert("content".into(), json!(content_text(&a.content)));
                if !tool_calls.is_empty() {
                    obj.insert("tool_calls".into(), Value::Array(tool_calls));
                }
                messages.push(Value::Object(obj));
            }
            Message::ToolResult(t) => messages.push(json!({
                "role": "tool",
                "tool_call_id": t.tool_call_id,
                "content": content_text(&t.content)
            })),
        }
    }

    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(model.id));
    body.insert("stream".into(), json!(true));
    body.insert("messages".into(), Value::Array(messages));
    if let Some(tools) = tools_payload(&context.tools) {
        body.insert("tools".into(), tools);
    }
    let thinking = options.thinking_level.unwrap_or(ThinkingLevel::Off);
    if let Some(effort) = reasoning_effort(thinking) {
        body.insert("reasoning_effort".into(), json!(effort));
    }
    Value::Object(body)
}

struct OpenAiStream {
    rx: tokio::sync::mpsc::UnboundedReceiver<AssistantMessageEvent>,
    final_message: AssistantMessage,
}

#[async_trait]
impl AssistantEventStream for OpenAiStream {
    async fn next_event(&mut self) -> Option<AssistantMessageEvent> {
        let ev = self.rx.recv().await?;
        match &ev {
            AssistantMessageEvent::Done { message, .. } => self.final_message = message.clone(),
            AssistantMessageEvent::Error { error, .. } => self.final_message = error.clone(),
            _ => {}
        }
        Some(ev)
    }

    async fn result(self: Box<Self>) -> AssistantMessage {
        self.final_message
    }
}

pub struct OpenAiCompletionsProvider {
    client: Client,
}

impl OpenAiCompletionsProvider {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .user_agent("aaos")
                .connect_timeout(Duration::from_secs(15))
                .read_timeout(Duration::from_secs(30))
                .build()
                .expect("reqwest client"),
        }
    }
}

impl Default for OpenAiCompletionsProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StreamFn for OpenAiCompletionsProvider {
    async fn call(
        &self,
        model: Model,
        context: LlmContext,
        options: StreamFnOptions,
        abort: watch::Receiver<bool>,
    ) -> Result<Box<dyn AssistantEventStream>, String> {
        let api_key = options.api_key.clone().unwrap_or_default();
        let body = build_request_body(&model, &context, &options);
        let url = chat_url(&model.base_url);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        let client = self.client.clone();
        let final_seed = AssistantMessage {
            model: model.id.clone(),
            provider: model.provider.clone(),
            api: model.api.clone(),
            stop_reason: StopReason::Pending,
            ..Default::default()
        };
        tokio::spawn(async move {
            run_stream(client, url, api_key, model, body, abort, tx).await;
        });
        Ok(Box::new(OpenAiStream {
            rx,
            final_message: final_seed,
        }))
    }
}

async fn run_stream(
    client: Client,
    url: String,
    api_key: String,
    model: Model,
    body: Value,
    mut abort: watch::Receiver<bool>,
    tx: tokio::sync::mpsc::UnboundedSender<AssistantMessageEvent>,
) {
    let mut builder = EventBuilder::new(&model, tx);
    if *abort.borrow() {
        builder.abort();
        return;
    }

    let request = client
        .post(&url)
        .header("Authorization", format!("Bearer {api_key}"))
        .header("Content-Type", "application/json")
        .json(&body);

    let response = tokio::select! {
        biased;
        _ = wait_aborted(&mut abort) => {
            builder.abort();
            return;
        }
        result = request.send() => result,
    };

    let response = match response {
        Ok(r) => r,
        Err(e) => {
            if *abort.borrow() {
                builder.abort();
            } else {
                builder.error(error_chain(&e));
            }
            return;
        }
    };

    if !response.status().is_success() {
        let status = response.status();
        let text = response.text().await.unwrap_or_default();
        builder.error(format!("HTTP {status}: {text}"));
        return;
    }

    let mut byte_stream = response.bytes_stream();
    let mut buf = String::new();

    loop {
        if *abort.borrow() {
            builder.abort();
            return;
        }
        let chunk = tokio::select! {
            biased;
            _ = wait_aborted(&mut abort) => {
                builder.abort();
                return;
            }
            next = byte_stream.next() => next,
        };
        match chunk {
            Some(Ok(bytes)) => {
                buf.push_str(&String::from_utf8_lossy(&bytes));
                while let Some(pos) = buf.find("\n\n") {
                    let frame = buf[..pos].to_string();
                    buf = buf[pos + 2..].to_string();
                    if let Err(msg) = builder.push_sse(&frame) {
                        builder.error(msg);
                        return;
                    }
                    if builder.finished {
                        return;
                    }
                }
            }
            Some(Err(e)) => {
                if *abort.borrow() {
                    builder.abort();
                } else {
                    builder.error(error_chain(&e));
                }
                return;
            }
            None => {
                if !builder.finished {
                    builder.close_done();
                }
                return;
            }
        }
    }
}

struct PendingTool {
    id: String,
    name: String,
    arguments: String,
    content_index: usize,
}

struct EventBuilder {
    tx: tokio::sync::mpsc::UnboundedSender<AssistantMessageEvent>,
    message: AssistantMessage,
    started: bool,
    open: OpenBlock,
    tools: BTreeMap<u32, PendingTool>,
    finished: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum OpenBlock {
    None,
    Text,
    Thinking,
    Tool(u32),
}

impl EventBuilder {
    fn new(model: &Model, tx: tokio::sync::mpsc::UnboundedSender<AssistantMessageEvent>) -> Self {
        let message = AssistantMessage {
            content: Vec::new(),
            stop_reason: StopReason::Pending,
            model: model.id.clone(),
            provider: model.provider.clone(),
            api: model.api.clone(),
            ..Default::default()
        };
        Self {
            tx,
            message,
            started: false,
            open: OpenBlock::None,
            tools: BTreeMap::new(),
            finished: false,
        }
    }

    fn emit(&mut self, event: AssistantMessageEvent) {
        let _ = self.tx.send(event);
    }

    fn ensure_start(&mut self) {
        if !self.started {
            self.started = true;
            self.emit(AssistantMessageEvent::Start {
                partial: self.message.clone(),
            });
        }
    }

    fn push_sse(&mut self, frame: &str) -> Result<(), String> {
        let mut data = String::new();
        for line in frame.lines() {
            if let Some(rest) = line.strip_prefix("data:") {
                let rest = rest.trim_start();
                if !data.is_empty() {
                    data.push('\n');
                }
                data.push_str(rest);
            } else if !line.is_empty() && !line.starts_with(':') && !line.starts_with("event:") {
                return Err(format!("malformed SSE line: {line}"));
            }
        }
        if data.is_empty() {
            return Ok(());
        }
        if data.trim() == "[DONE]" {
            self.close_done();
            return Ok(());
        }
        let value: Value =
            serde_json::from_str(&data).map_err(|e| format!("malformed SSE JSON: {e}"))?;
        if let Some(err) = value.get("error") {
            let msg = err
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("provider error")
                .to_string();
            self.error(msg);
            return Ok(());
        }
        self.apply_chunk(&value);
        Ok(())
    }

    fn apply_chunk(&mut self, value: &Value) {
        self.ensure_start();
        let choice = value.pointer("/choices/0").cloned().unwrap_or(Value::Null);
        let delta = choice.get("delta").cloned().unwrap_or(Value::Null);

        if let Some(reason) = delta.get("reasoning_content").and_then(|v| v.as_str())
            && !reason.is_empty()
        {
            self.push_thinking(reason);
        }
        if let Some(content) = delta.get("content").and_then(|v| v.as_str())
            && !content.is_empty()
        {
            self.push_text(content);
        }
        if let Some(calls) = delta.get("tool_calls").and_then(|v| v.as_array()) {
            for call in calls {
                self.push_tool(call);
            }
        }
        if let Some(reason) = choice.get("finish_reason").and_then(|v| v.as_str())
            && !reason.is_nullish()
            && reason != "null"
        {
            self.finish_reason(reason);
        }
    }

    fn close_open(&mut self) {
        match self.open {
            OpenBlock::Text => {
                let idx = self.current_index();
                let content = match &self.message.content[idx] {
                    ContentBlock::Text { text } => text.clone(),
                    _ => String::new(),
                };
                self.emit(AssistantMessageEvent::TextEnd {
                    content_index: idx,
                    content,
                    partial: self.message.clone(),
                });
            }
            OpenBlock::Thinking => {
                let idx = self.current_index();
                let content = match &self.message.content[idx] {
                    ContentBlock::Thinking { text } => text.clone(),
                    _ => String::new(),
                };
                self.emit(AssistantMessageEvent::ThinkingEnd {
                    content_index: idx,
                    content,
                    partial: self.message.clone(),
                });
            }
            OpenBlock::Tool(i) => {
                if let Some(tool) = self.tools.remove(&i) {
                    let parsed = parse_tool_args(&tool.arguments);
                    let tc = ToolCall {
                        id: tool.id,
                        name: tool.name,
                        arguments: parsed,
                    };
                    self.message.content[tool.content_index] = ContentBlock::ToolCall(tc.clone());
                    self.emit(AssistantMessageEvent::ToolCallEnd {
                        content_index: tool.content_index,
                        tool_call: tc,
                        partial: self.message.clone(),
                    });
                }
            }
            OpenBlock::None => {}
        }
        self.open = OpenBlock::None;
    }

    fn current_index(&self) -> usize {
        self.message.content.len().saturating_sub(1)
    }

    fn push_text(&mut self, delta: &str) {
        if self.open != OpenBlock::Text {
            self.close_open();
            self.message.content.push(ContentBlock::text(""));
            let idx = self.current_index();
            self.open = OpenBlock::Text;
            self.emit(AssistantMessageEvent::TextStart {
                content_index: idx,
                partial: self.message.clone(),
            });
        }
        let idx = self.current_index();
        if let ContentBlock::Text { text } = &mut self.message.content[idx] {
            text.push_str(delta);
        }
        self.emit(AssistantMessageEvent::TextDelta {
            content_index: idx,
            delta: delta.to_string(),
            partial: self.message.clone(),
        });
    }

    fn push_thinking(&mut self, delta: &str) {
        if self.open != OpenBlock::Thinking {
            self.close_open();
            self.message.content.push(ContentBlock::Thinking {
                text: String::new(),
            });
            let idx = self.current_index();
            self.open = OpenBlock::Thinking;
            self.emit(AssistantMessageEvent::ThinkingStart {
                content_index: idx,
                partial: self.message.clone(),
            });
        }
        let idx = self.current_index();
        if let ContentBlock::Thinking { text } = &mut self.message.content[idx] {
            text.push_str(delta);
        }
        self.emit(AssistantMessageEvent::ThinkingDelta {
            content_index: idx,
            delta: delta.to_string(),
            partial: self.message.clone(),
        });
    }

    fn push_tool(&mut self, call: &Value) {
        let index = call.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        if self.open != OpenBlock::Tool(index) && self.open != OpenBlock::None {
            self.close_open();
        }
        if !self.tools.contains_key(&index) {
            let content_index = self.message.content.len();
            self.message.content.push(ContentBlock::ToolCall(ToolCall {
                id: String::new(),
                name: String::new(),
                arguments: json!({}),
            }));
            self.tools.insert(
                index,
                PendingTool {
                    id: String::new(),
                    name: String::new(),
                    arguments: String::new(),
                    content_index,
                },
            );
        }
        {
            let entry = self.tools.get_mut(&index).unwrap();
            if let Some(id) = call.get("id").and_then(|v| v.as_str()) {
                entry.id = id.to_string();
            }
            if let Some(name) = call.pointer("/function/name").and_then(|v| v.as_str()) {
                entry.name = name.to_string();
            }
        }
        let args_delta = call
            .pointer("/function/arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let (content_index, id, name, args_so_far, starting) = {
            let entry = self.tools.get_mut(&index).unwrap();
            let starting = self.open != OpenBlock::Tool(index);
            if !args_delta.is_empty() {
                entry.arguments.push_str(&args_delta);
            }
            (
                entry.content_index,
                entry.id.clone(),
                entry.name.clone(),
                entry.arguments.clone(),
                starting,
            )
        };
        if starting {
            self.open = OpenBlock::Tool(index);
            self.message.content[content_index] = ContentBlock::ToolCall(ToolCall {
                id: id.clone(),
                name: name.clone(),
                arguments: json!({}),
            });
            self.emit(AssistantMessageEvent::ToolCallStart {
                content_index,
                partial: self.message.clone(),
            });
        } else if let ContentBlock::ToolCall(tc) = &mut self.message.content[content_index] {
            if !id.is_empty() {
                tc.id = id;
            }
            if !name.is_empty() {
                tc.name = name;
            }
        }
        if !args_delta.is_empty() {
            if let ContentBlock::ToolCall(tc) = &mut self.message.content[content_index] {
                tc.arguments = json!(args_so_far);
            }
            self.emit(AssistantMessageEvent::ToolCallDelta {
                content_index,
                delta: args_delta,
                partial: self.message.clone(),
            });
        }
    }

    fn finish_reason(&mut self, reason: &str) {
        self.close_open();
        self.message.stop_reason = match reason {
            "length" => StopReason::Length,
            "tool_calls" => StopReason::ToolUse,
            _ => StopReason::Stop,
        };
        self.emit(AssistantMessageEvent::Done {
            reason: self.message.stop_reason,
            message: self.message.clone(),
        });
        self.finished = true;
    }

    fn close_done(&mut self) {
        if self.finished {
            return;
        }
        self.ensure_start();
        self.close_open();
        if self.message.stop_reason == StopReason::Pending {
            self.message.stop_reason = StopReason::Stop;
        }
        self.emit(AssistantMessageEvent::Done {
            reason: self.message.stop_reason,
            message: self.message.clone(),
        });
        self.finished = true;
    }

    fn abort(&mut self) {
        if self.finished {
            return;
        }
        self.ensure_start();
        self.close_open();
        self.message.stop_reason = StopReason::Aborted;
        self.message.error_message = Some("Aborted".into());
        self.emit(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error: self.message.clone(),
        });
        self.finished = true;
    }

    fn error(&mut self, msg: String) {
        if self.finished {
            return;
        }
        self.ensure_start();
        self.close_open();
        self.message.stop_reason = StopReason::Error;
        self.message.error_message = Some(msg);
        self.emit(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: self.message.clone(),
        });
        self.finished = true;
    }
}

fn parse_tool_args(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| json!({}))
}

trait Nullish {
    fn is_nullish(&self) -> bool;
}
impl Nullish for str {
    fn is_nullish(&self) -> bool {
        self.is_empty() || self == "null"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pi_agent_core::types::{AgentToolResult, ToolResultMessage, UserMessage};
    use std::net::SocketAddr;
    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::watch;

    #[test]
    fn error_chain_appends_source() {
        #[derive(Debug)]
        struct Inner;
        impl std::fmt::Display for Inner {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("operation timed out")
            }
        }
        impl StdError for Inner {}

        #[derive(Debug)]
        struct Outer(Inner);
        impl std::fmt::Display for Outer {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("error sending request for url (https://example)")
            }
        }
        impl StdError for Outer {
            fn source(&self) -> Option<&(dyn StdError + 'static)> {
                Some(&self.0)
            }
        }

        let msg = error_chain(&Outer(Inner));
        assert!(msg.contains("error sending request"));
        assert!(msg.contains("operation timed out"));
    }

    struct EchoTool;

    #[async_trait]
    impl AgentTool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn label(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echoes"
        }
        fn parameters(&self) -> Value {
            json!({
                "type": "object",
                "properties": { "x": { "type": "number" } },
                "required": ["x"]
            })
        }
        async fn execute(
            &self,
            _tool_call_id: String,
            _params: Value,
            _signal: Option<&watch::Receiver<bool>>,
            _on_update: Option<pi_agent_core::types::AgentToolUpdateCallback>,
        ) -> Result<AgentToolResult, String> {
            Ok(AgentToolResult::text("ok"))
        }
    }

    fn model(base: &str) -> Model {
        Model {
            id: "deepseek-v4-flash".into(),
            name: "flash".into(),
            api: "openai-completions".into(),
            provider: "deepseek".into(),
            base_url: base.into(),
            reasoning: true,
            input: vec![],
            cost: Default::default(),
            context_window: 1000,
            max_tokens: 100,
        }
    }

    fn sse(chunks: &[&str]) -> String {
        chunks
            .iter()
            .map(|c| format!("data: {c}\n\n"))
            .collect::<String>()
            + "data: [DONE]\n\n"
    }

    async fn serve(
        status: u16,
        body: String,
        delay_ms: u64,
    ) -> (SocketAddr, tokio::task::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            if let Ok((mut sock, _)) = listener.accept().await {
                let mut buf = vec![0u8; 8192];
                let _ = sock.read(&mut buf).await;
                if delay_ms > 0 {
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                }
                let header = format!(
                    "HTTP/1.1 {status} OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = sock.write_all(header.as_bytes()).await;
                if delay_ms > 0 {
                    for chunk in body.as_bytes().chunks(24) {
                        if sock.write_all(chunk).await.is_err() {
                            break;
                        }
                        let _ = sock.flush().await;
                        tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    }
                } else {
                    let _ = sock.write_all(body.as_bytes()).await;
                }
            }
        });
        (addr, handle)
    }

    async fn collect(
        addr: SocketAddr,
        context: LlmContext,
        options: StreamFnOptions,
        abort: watch::Receiver<bool>,
    ) -> (Vec<AssistantMessageEvent>, AssistantMessage) {
        let provider = OpenAiCompletionsProvider::new();
        let mut m = model(&format!("http://{addr}"));
        m.base_url = format!("http://{addr}");
        let mut stream = provider.call(m, context, options, abort).await.unwrap();
        let mut events = Vec::new();
        while let Some(ev) = stream.next_event().await {
            events.push(ev);
        }
        let final_msg = stream.result().await;
        (events, final_msg)
    }

    fn captured_request_server() -> (
        tokio::sync::oneshot::Receiver<String>,
        impl std::future::Future<Output = SocketAddr>,
    ) {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let fut = async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            tokio::spawn(async move {
                if let Ok((mut sock, _)) = listener.accept().await {
                    let mut buf = vec![0u8; 16384];
                    let n = sock.read(&mut buf).await.unwrap_or(0);
                    let _ = tx.send(String::from_utf8_lossy(&buf[..n]).into_owned());
                    let body =
                        sse(&[r#"{"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}"#]);
                    let header = format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        body.len()
                    );
                    let _ = sock.write_all(header.as_bytes()).await;
                    let _ = sock.write_all(body.as_bytes()).await;
                }
            });
            addr
        };
        (rx, fut)
    }

    #[tokio::test]
    async fn request_includes_messages_tools_thinking_and_bearer() {
        let (rx, fut) = captured_request_server();
        let addr = fut.await;
        let context = LlmContext {
            system_prompt: "sys".into(),
            messages: vec![Message::User(UserMessage::new("hello"))],
            tools: vec![Arc::new(EchoTool)],
        };
        let options = StreamFnOptions {
            api_key: Some("cchub-key".into()),
            thinking_level: Some(ThinkingLevel::High),
            ..Default::default()
        };
        let (_tx, abort) = watch::channel(false);
        let _ = collect(addr, context, options, abort).await;
        let raw = rx.await.unwrap();
        assert!(
            raw.to_ascii_lowercase().contains("bearer cchub-key"),
            "{raw}"
        );
        assert!(
            raw.to_ascii_lowercase().contains("user-agent: aaos"),
            "{raw}"
        );
        let body = raw.split("\r\n\r\n").nth(1).unwrap_or("");
        let json: Value = serde_json::from_str(body).unwrap();
        assert_eq!(json["model"], "deepseek-v4-flash");
        assert_eq!(json["reasoning_effort"], "high");
        assert_eq!(json["messages"][0]["role"], "system");
        assert_eq!(json["messages"][1]["role"], "user");
        assert_eq!(json["tools"][0]["function"]["name"], "echo");
        assert_eq!(
            json["tools"][0]["function"]["parameters"]["required"],
            json!(["x"])
        );
        assert_eq!(
            json["tools"][0]["function"]["parameters"]["properties"]["x"]["type"],
            "number"
        );
    }

    #[tokio::test]
    async fn text_events_are_emitted() {
        let body = sse(&[
            r#"{"choices":[{"delta":{"content":"Hel"}}]}"#,
            r#"{"choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}"#,
        ]);
        let (addr, h) = serve(200, body, 0).await;
        let (_tx, abort) = watch::channel(false);
        let (events, msg) = collect(
            addr,
            LlmContext {
                system_prompt: String::new(),
                messages: vec![Message::User(UserMessage::new("q"))],
                tools: vec![],
            },
            StreamFnOptions {
                api_key: Some("k".into()),
                thinking_level: Some(ThinkingLevel::High),
                ..Default::default()
            },
            abort,
        )
        .await;
        let _ = h.await;
        assert!(matches!(
            events.first(),
            Some(AssistantMessageEvent::Start { .. })
        ));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AssistantMessageEvent::TextStart { .. }))
        );
        assert!(events.iter().any(
            |e| matches!(e, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hel")
        ));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AssistantMessageEvent::TextEnd { .. }))
        );
        assert!(matches!(
            events.last(),
            Some(AssistantMessageEvent::Done {
                reason: StopReason::Stop,
                ..
            })
        ));
        assert_eq!(content_text(&msg.content), "Hello");
    }

    #[tokio::test]
    async fn thinking_and_tool_call_events() {
        let body = sse(&[
            r#"{"choices":[{"delta":{"reasoning_content":"hmm"}}]}"#,
            r#"{"choices":[{"delta":{"reasoning_content":"!"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":"{\"x\""}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":1}"}}]},"finish_reason":"tool_calls"}]}"#,
        ]);
        let (addr, h) = serve(200, body, 0).await;
        let (_tx, abort) = watch::channel(false);
        let (events, msg) = collect(
            addr,
            LlmContext {
                system_prompt: String::new(),
                messages: vec![Message::User(UserMessage::new("q"))],
                tools: vec![Arc::new(EchoTool)],
            },
            StreamFnOptions {
                api_key: Some("k".into()),
                ..Default::default()
            },
            abort,
        )
        .await;
        let _ = h.await;
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AssistantMessageEvent::ThinkingStart { .. }))
        );
        assert!(events.iter().any(
            |e| matches!(e, AssistantMessageEvent::ThinkingDelta { delta, .. } if delta == "hmm")
        ));
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AssistantMessageEvent::ThinkingEnd { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AssistantMessageEvent::ToolCallStart { .. }))
        );
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AssistantMessageEvent::ToolCallDelta { .. }))
        );
        let end = events
            .iter()
            .find_map(|e| match e {
                AssistantMessageEvent::ToolCallEnd { tool_call, .. } => Some(tool_call),
                _ => None,
            })
            .unwrap();
        assert_eq!(end.name, "echo");
        assert_eq!(end.arguments, json!({"x":1}));
        AgentTool::validate(&EchoTool, &end.arguments).unwrap();
        assert_eq!(msg.stop_reason, StopReason::ToolUse);
    }

    #[tokio::test]
    async fn http_error_and_malformed_sse_stay_in_stream() {
        let (addr, h) = serve(401, "nope".into(), 0).await;
        let (_tx, abort) = watch::channel(false);
        let (events, msg) = collect(
            addr,
            LlmContext {
                system_prompt: String::new(),
                messages: vec![],
                tools: vec![],
            },
            StreamFnOptions {
                api_key: Some("k".into()),
                ..Default::default()
            },
            abort,
        )
        .await;
        let _ = h.await;
        assert!(matches!(
            events.last(),
            Some(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                ..
            })
        ));
        assert_eq!(msg.stop_reason, StopReason::Error);

        let (addr, h) = serve(200, "not-sse\n\n".into(), 0).await;
        let (_tx, abort) = watch::channel(false);
        let (events, _) = collect(
            addr,
            LlmContext {
                system_prompt: String::new(),
                messages: vec![],
                tools: vec![],
            },
            StreamFnOptions {
                api_key: Some("k".into()),
                ..Default::default()
            },
            abort,
        )
        .await;
        let _ = h.await;
        assert!(matches!(
            events.last(),
            Some(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                ..
            })
        ));
    }

    #[test]
    fn request_body_includes_tool_result_messages() {
        let model = model("http://example");
        let context = LlmContext {
            system_prompt: String::new(),
            messages: vec![
                Message::User(UserMessage::new("q")),
                Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_1".into(),
                    tool_name: "echo".into(),
                    content: vec![ContentBlock::text("pong")],
                    details: json!({}),
                    usage: None,
                    added_tool_names: None,
                    is_error: false,
                    timestamp: 0,
                }),
            ],
            tools: vec![],
        };
        let body = build_request_body(
            &model,
            &context,
            &StreamFnOptions {
                thinking_level: Some(ThinkingLevel::High),
                ..Default::default()
            },
        );
        assert_eq!(body["messages"][1]["role"], "tool");
        assert_eq!(body["messages"][1]["tool_call_id"], "call_1");
        assert_eq!(body["reasoning_effort"], "high");
    }

    #[tokio::test]
    async fn sse_provider_error_stays_in_stream() {
        let body = "data: {\"error\":{\"message\":\"upstream failed\"}}\n\n";
        let (addr, h) = serve(200, body.into(), 0).await;
        let (_tx, abort) = watch::channel(false);
        let (events, msg) = collect(
            addr,
            LlmContext {
                system_prompt: String::new(),
                messages: vec![],
                tools: vec![],
            },
            StreamFnOptions {
                api_key: Some("k".into()),
                ..Default::default()
            },
            abort,
        )
        .await;
        let _ = h.await;
        assert!(matches!(
            events.last(),
            Some(AssistantMessageEvent::Error {
                reason: StopReason::Error,
                ..
            })
        ));
        assert!(msg.error_message.unwrap().contains("upstream failed"));
    }

    #[tokio::test]
    async fn abort_cancels_body_and_emits_aborted() {
        let body = sse(&[
            r#"{"choices":[{"delta":{"content":"aaaa"}}]}"#,
            r#"{"choices":[{"delta":{"content":"bbbb"}}]}"#,
        ]);
        let (addr, h) = serve(200, body, 80).await;
        let (tx, abort) = watch::channel(false);
        let provider = OpenAiCompletionsProvider::new();
        let m = model(&format!("http://{addr}"));
        let handle = tokio::spawn(async move {
            let mut stream = provider
                .call(
                    m,
                    LlmContext {
                        system_prompt: String::new(),
                        messages: vec![],
                        tools: vec![],
                    },
                    StreamFnOptions {
                        api_key: Some("k".into()),
                        ..Default::default()
                    },
                    abort,
                )
                .await
                .unwrap();
            let mut events = Vec::new();
            while let Some(ev) = stream.next_event().await {
                events.push(ev);
            }
            (events, stream.result().await)
        });
        tokio::time::sleep(Duration::from_millis(50)).await;
        let _ = tx.send(true);
        let (events, msg) = handle.await.unwrap();
        let _ = h.await;
        assert!(events.iter().any(|e| matches!(
            e,
            AssistantMessageEvent::Error {
                reason: StopReason::Aborted,
                ..
            }
        )));
        assert_eq!(msg.stop_reason, StopReason::Aborted);
    }
}
