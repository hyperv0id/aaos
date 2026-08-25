//! Cohere Chat v2 streaming API format (SSE).
//!
//! `POST {base}/chat` with `Authorization: Bearer {api_key}` authentication
//! and `Content-Type: application/json`. The base URL already contains the
//! version path (e.g. `https://api.cohere.com/v2`); this adapter appends
//! `/chat`. Responses are named-event SSE frames (`event: <type>` +
//! `data: <json>`) terminated by the `message-end` event.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use pi_agent_core::types::{
    AgentTool, AssistantEventStream, AssistantMessage, AssistantMessageEvent, ContentBlock,
    LlmContext, Message, Model, StopReason, StreamFn, StreamFnOptions, ToolCall,
};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::sync::watch;

#[cfg(test)]
use pi_agent_core::types::ModelInput;

use super::sse::{self, Ctx, SseFormat, content_text, parse_tool_args, supports_images};

/// `Model::api` key dispatching to this format.
pub const API: &str = "cohere-chat";

/// Reserved EventBuilder block index for `tool-plan-delta` thinking deltas.
///
/// Cohere's `tool-plan-delta` events carry no `index` field and precede any
/// content/tool-call blocks (whose indices start at 0), so the tool plan
/// accumulates under this never-colliding key.
const PLAN_INDEX: usize = usize::MAX;

/// Build the `/chat` endpoint from a base URL that already contains the
/// version path. Trims trailing slashes, avoids duplicating an existing
/// `/chat` suffix, and appends `/chat` otherwise.
fn chat_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/chat") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/chat")
    }
}

/// Cohere tool schema: `{ type: "function", function: { name, description, parameters } }`.
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

/// Serialize a user message's content blocks into the Cohere Chat v2 content
/// format.
///
/// - Text-only messages become a plain string (Cohere accepts this).
/// - Messages with images become a content-block array with
///   `{type: "text", text}` and
///   `{type: "image_url", image_url: {url: "data:...;base64,..."}}` blocks.
/// - When the model cannot accept images, image blocks are dropped and the
///   surviving text flattens back to a plain string.
///
/// Returns `None` when the message has no serializable content.
fn serialize_user_content(blocks: &[ContentBlock], model: &Model) -> Option<Value> {
    let has_images = blocks
        .iter()
        .any(|b| matches!(b, ContentBlock::Image { .. }));

    if !has_images {
        let text = content_text(blocks);
        if text.is_empty() {
            return None;
        }
        return Some(Value::String(text));
    }

    // Image present but model doesn't accept images: emit only the text
    // blocks, dropping image blocks entirely.
    let can_send_images = supports_images(model);

    let mut parts: Vec<Value> = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } if !text.is_empty() => {
                parts.push(json!({ "type": "text", "text": text }));
            }
            ContentBlock::Image { source } if can_send_images => {
                parts.push(json!({
                    "type": "image_url",
                    "image_url": {
                        "url": format!(
                            "data:{};base64,{}",
                            source.mime_type,
                            base64::engine::general_purpose::STANDARD.encode(&source.bytes)
                        )
                    }
                }));
            }
            _ => {}
        }
    }
    if parts.is_empty() {
        return None;
    }
    // If images were dropped (model can't accept them) and only text blocks
    // survive, flatten to a plain string — matching the text-only path.
    let has_image = parts
        .iter()
        .any(|p| p.get("type").and_then(|t| t.as_str()) == Some("image_url"));
    if !has_image {
        let text = parts
            .iter()
            .filter_map(|p| {
                if p.get("type").and_then(|t| t.as_str()) == Some("text") {
                    p.get("text").and_then(|t| t.as_str()).map(String::from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("");
        if text.is_empty() {
            return None;
        }
        return Some(Value::String(text));
    }

    Some(Value::Array(parts))
}

/// Serialize an assistant message's text content for request replay. Tool
/// calls are serialized separately into the `tool_calls` field by
/// `build_request_body`; thinking blocks are not replayed.
fn serialize_assistant_content(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } if !text.is_empty() => {
                Some(json!({ "type": "text", "text": text }))
            }
            _ => None,
        })
        .collect()
}

/// Build the Cohere Chat v2 request body: `{ model, stream, messages, tools }`.
///
/// The system prompt becomes the first `role: "system"` message; tool results
/// use the dedicated `role: "tool"` with a `tool_call_id`.
fn build_request_body(model: &Model, context: &LlmContext, _options: &StreamFnOptions) -> Value {
    let mut messages: Vec<Value> = Vec::new();

    if !context.system_prompt.is_empty() {
        messages.push(json!({ "role": "system", "content": context.system_prompt }));
    }

    for msg in &context.messages {
        match msg {
            Message::User(u) => {
                if let Some(content) = serialize_user_content(&u.content, model) {
                    messages.push(json!({ "role": "user", "content": content }));
                }
            }
            Message::Assistant(a) => {
                let content = serialize_assistant_content(&a.content);
                let tool_calls: Vec<Value> = a
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolCall(tc) => Some(json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments.to_string()
                            }
                        })),
                        _ => None,
                    })
                    .collect();
                if content.is_empty() && tool_calls.is_empty() {
                    continue;
                }
                let mut msg = serde_json::Map::new();
                msg.insert("role".into(), json!("assistant"));
                msg.insert("content".into(), Value::Array(content));
                if !tool_calls.is_empty() {
                    msg.insert("tool_calls".into(), Value::Array(tool_calls));
                }
                messages.push(Value::Object(msg));
            }
            Message::ToolResult(t) => {
                messages.push(json!({
                    "role": "tool",
                    "content": content_text(&t.content),
                    "tool_call_id": t.tool_call_id
                }));
            }
        }
    }

    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(model.id));
    body.insert("stream".into(), json!(true));
    body.insert("messages".into(), Value::Array(messages));

    if let Some(tools) = tools_payload(&context.tools) {
        body.insert("tools".into(), tools);
    }

    Value::Object(body)
}

pub struct CohereChatProvider {
    client: Client,
}

impl CohereChatProvider {
    pub fn new() -> Result<Self, reqwest::Error> {
        Ok(Self {
            client: Client::builder()
                .user_agent("aaos")
                .connect_timeout(Duration::from_secs(15))
                .read_timeout(Duration::from_secs(30))
                .build()?,
        })
    }
}

#[async_trait]
impl StreamFn for CohereChatProvider {
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
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
        let client = self.client.clone();
        let final_seed = AssistantMessage {
            model: model.id.clone(),
            provider: model.provider.clone(),
            api: model.api.clone(),
            stop_reason: StopReason::Pending,
            ..Default::default()
        };
        let headers = [("Authorization", format!("Bearer {api_key}"))];
        tokio::spawn(async move {
            sse::run_stream::<CohereFormat>(client, url, model, body, abort, tx, headers).await;
        });
        Ok(sse::SseStream::boxed(rx, final_seed))
    }
}

struct PendingTool {
    id: String,
    name: String,
    arguments: String,
    content_index: usize,
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum OpenBlock {
    #[default]
    None,
    Text(usize),
    Thinking(usize),
    Tool(usize),
}

/// Cohere chunk interpretation: named events keyed by stream index, mapped
/// onto our content slots through `block_map`; tool plans accumulate under
/// the reserved [`PLAN_INDEX`].
#[derive(Default)]
struct CohereFormat {
    open: OpenBlock,
    /// Maps Cohere stream index → pending tool accumulator.
    pending_tools: BTreeMap<usize, PendingTool>,
    /// Maps Cohere stream index → our content[] index, so deltas for a known
    /// block find the right slot.
    block_map: BTreeMap<usize, usize>,
}
#[cfg(test)]
type EventBuilder = super::sse::EventBuilder<CohereFormat>;

impl SseFormat for CohereFormat {
    /// Dispatch one decoded payload. Cohere streams carry named events;
    /// `event:` supplies the dispatch key.
    fn apply_chunk(&mut self, cx: &mut Ctx<'_>, event: Option<&str>, value: &Value) {
        cx.ensure_start();
        match event.unwrap_or("") {
            "message-start" => {
                // Stream is live; no content to record in the simplified model.
            }
            "content-start" => {
                // Cohere's content-start carries no content payload — the
                // following content-delta opens the text block via push_text.
            }
            "content-delta" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let text = value
                    .pointer("/delta/message/content/text")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                if !text.is_empty() {
                    self.push_text(cx, index, text);
                }
            }
            "content-end" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                self.close_block(cx, index);
            }
            "tool-plan-delta" => {
                let text = value
                    .pointer("/delta/message/tool_plan")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                if !text.is_empty() {
                    self.push_thinking(cx, PLAN_INDEX, text);
                }
            }
            "tool-call-start" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let tool = value
                    .pointer("/delta/message/tool_calls/0")
                    .cloned()
                    .unwrap_or(Value::Null);
                let id = tool
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = tool
                    .pointer("/function/name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                self.start_tool(cx, index, id, name);
            }
            "tool-call-delta" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let partial = value
                    .pointer("/delta/message/tool_calls/0/function/arguments")
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                if !partial.is_empty() {
                    self.push_tool_delta(cx, index, partial);
                }
            }
            "tool-call-end" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                self.close_block(cx, index);
            }
            "message-end" => {
                if let Some(reason) = value
                    .pointer("/delta/finish_reason")
                    .and_then(|v| v.as_str())
                {
                    self.finish_reason(cx, reason);
                } else if !*cx.finished {
                    // Terminal marker without an explicit finish reason.
                    self.close_done(cx);
                }
            }
            _ => {}
        }
    }

    fn close_open(&mut self, cx: &mut Ctx<'_>) {
        match self.open {
            OpenBlock::Text(_) => {
                let idx = cx.current_index();
                let content = match &cx.message.content[idx] {
                    ContentBlock::Text { text } => text.clone(),
                    _ => String::new(),
                };
                cx.emit(AssistantMessageEvent::TextEnd {
                    content_index: idx,
                    content,
                    partial: cx.message.clone(),
                });
            }
            OpenBlock::Thinking(_) => {
                let idx = cx.current_index();
                let content = match &cx.message.content[idx] {
                    ContentBlock::Thinking { text } => text.clone(),
                    _ => String::new(),
                };
                cx.emit(AssistantMessageEvent::ThinkingEnd {
                    content_index: idx,
                    content,
                    partial: cx.message.clone(),
                });
            }
            OpenBlock::Tool(index) => {
                if let Some(tool) = self.pending_tools.remove(&index) {
                    let parsed = parse_tool_args(&tool.arguments);
                    let tc = ToolCall {
                        id: tool.id,
                        name: tool.name,
                        arguments: parsed,
                    };
                    cx.message.content[tool.content_index] = ContentBlock::ToolCall(tc.clone());
                    cx.emit(AssistantMessageEvent::ToolCallEnd {
                        content_index: tool.content_index,
                        tool_call: tc,
                        partial: cx.message.clone(),
                    });
                }
            }
            OpenBlock::None => {}
        }
        self.open = OpenBlock::None;
    }

    fn stop_reason(&self, reason: &str) -> StopReason {
        match reason {
            "MAX_TOKENS" | "max_tokens" => StopReason::Length,
            "TOOL_CALL" | "tool_use" => StopReason::ToolUse,
            _ => StopReason::Stop,
        }
    }
}

impl CohereFormat {
    fn start_text(&mut self, cx: &mut Ctx<'_>, index: usize, initial: &str) {
        self.close_open(cx);
        cx.message.content.push(ContentBlock::text(initial));
        let content_index = cx.message.content.len() - 1;
        self.block_map.insert(index, content_index);
        self.open = OpenBlock::Text(index);
        cx.emit(AssistantMessageEvent::TextStart {
            content_index,
            partial: cx.message.clone(),
        });
        if !initial.is_empty() {
            cx.emit(AssistantMessageEvent::TextDelta {
                content_index,
                delta: initial.to_string(),
                partial: cx.message.clone(),
            });
        }
    }

    fn start_thinking(&mut self, cx: &mut Ctx<'_>, index: usize, initial: &str) {
        self.close_open(cx);
        cx.message.content.push(ContentBlock::Thinking {
            text: initial.into(),
        });
        let content_index = cx.message.content.len() - 1;
        self.block_map.insert(index, content_index);
        self.open = OpenBlock::Thinking(index);
        cx.emit(AssistantMessageEvent::ThinkingStart {
            content_index,
            partial: cx.message.clone(),
        });
        if !initial.is_empty() {
            cx.emit(AssistantMessageEvent::ThinkingDelta {
                content_index,
                delta: initial.to_string(),
                partial: cx.message.clone(),
            });
        }
    }

    fn start_tool(&mut self, cx: &mut Ctx<'_>, index: usize, id: String, name: String) {
        self.close_open(cx);
        let content_index = cx.message.content.len();
        cx.message.content.push(ContentBlock::ToolCall(ToolCall {
            id: id.clone(),
            name: name.clone(),
            arguments: json!({}),
        }));
        self.block_map.insert(index, content_index);
        self.pending_tools.insert(
            index,
            PendingTool {
                id,
                name,
                arguments: String::new(),
                content_index,
            },
        );
        self.open = OpenBlock::Tool(index);
        cx.emit(AssistantMessageEvent::ToolCallStart {
            content_index,
            partial: cx.message.clone(),
        });
    }

    fn push_text(&mut self, cx: &mut Ctx<'_>, index: usize, delta: &str) {
        let content_index = match self.block_map.get(&index) {
            Some(&ci) => ci,
            None => {
                // Delta without a preceding content-start: open a text block.
                self.start_text(cx, index, "");
                self.block_map[&index]
            }
        };
        if !matches!(self.open, OpenBlock::Text(i) if i == index) {
            self.close_open(cx);
            self.open = OpenBlock::Text(index);
        }
        if let ContentBlock::Text { text } = &mut cx.message.content[content_index] {
            text.push_str(delta);
        }
        cx.emit(AssistantMessageEvent::TextDelta {
            content_index,
            delta: delta.to_string(),
            partial: cx.message.clone(),
        });
    }

    fn push_thinking(&mut self, cx: &mut Ctx<'_>, index: usize, delta: &str) {
        let content_index = match self.block_map.get(&index) {
            Some(&ci) => ci,
            None => {
                self.start_thinking(cx, index, "");
                self.block_map[&index]
            }
        };
        if !matches!(self.open, OpenBlock::Thinking(i) if i == index) {
            self.close_open(cx);
            self.open = OpenBlock::Thinking(index);
        }
        if let ContentBlock::Thinking { text } = &mut cx.message.content[content_index] {
            text.push_str(delta);
        }
        cx.emit(AssistantMessageEvent::ThinkingDelta {
            content_index,
            delta: delta.to_string(),
            partial: cx.message.clone(),
        });
    }

    fn push_tool_delta(&mut self, cx: &mut Ctx<'_>, index: usize, partial: &str) {
        let content_index = match self.block_map.get(&index) {
            Some(&ci) => ci,
            None => return, // Delta for an unopened tool block — ignore.
        };
        // The tool entry is removed on close_open; a late delta after
        // tool-call-end is a protocol violation — tolerate it by returning
        // early instead of indexing a missing BTreeMap key (which would
        // panic and kill the spawned stream task).
        let accumulated = match self.pending_tools.get_mut(&index) {
            Some(entry) => {
                entry.arguments.push_str(partial);
                entry.arguments.clone()
            }
            None => return,
        };
        if let ContentBlock::ToolCall(tc) = &mut cx.message.content[content_index] {
            tc.arguments = parse_tool_args(&accumulated);
        }
        cx.emit(AssistantMessageEvent::ToolCallDelta {
            content_index,
            delta: partial.to_string(),
            partial: cx.message.clone(),
        });
    }

    fn close_block(&mut self, cx: &mut Ctx<'_>, index: usize) {
        // Unknown stream index — not a block we opened.
        if !self.block_map.contains_key(&index) {
            return;
        }
        // Only close if this is the currently open block.
        let is_current = match self.open {
            OpenBlock::None => false,
            OpenBlock::Text(i) | OpenBlock::Thinking(i) => i == index,
            OpenBlock::Tool(i) => i == index,
        };
        if is_current {
            self.close_open(cx);
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    use pi_agent_core::types::{AgentToolResult, ImageSource, ToolResultMessage, UserMessage};

    use std::net::SocketAddr;

    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use tokio::net::TcpListener;

    use tokio::sync::watch;

    fn model(base: &str) -> Model {
        Model {
            id: "command-r-plus".into(),
            name: "Command R+".into(),
            api: API.into(),
            provider: "cohere".into(),
            base_url: base.into(),
            reasoning: false,
            input: vec![],
            cost: Default::default(),
            context_window: 128_000,
            max_tokens: 4096,
        }
    }

    fn sse_event(event: &str, data: &str) -> String {
        format!("event: {event}\ndata: {data}\n\n")
    }

    fn collect_events(frames: &[String]) -> Vec<AssistantMessageEvent> {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
        let m = model("http://example");
        let mut builder = EventBuilder::new(&m, tx);
        for frame in frames {
            builder.push_sse(frame).expect("frame should parse");
            if builder.finished {
                break;
            }
        }
        if !builder.finished {
            builder.close_done();
        }
        let mut events = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            events.push(ev);
        }
        events
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
        let provider = CohereChatProvider::new().expect("HTTP client");
        let m = model(&format!("http://{addr}"));
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
                        "event: content-start\ndata: {\"type\":\"content-start\",\"index\":0}\n\n"
                            .to_string()
                            + "event: content-delta\ndata: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"hi\"}}}}\n\n"
                            + "event: content-end\ndata: {\"type\":\"content-end\",\"index\":0}\n\n"
                            + "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\"}}\n\n";
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

    mod url {
        use super::*;

        #[test]
        fn appends_tail_segment() {
            assert_eq!(
                chat_url("https://api.cohere.com/v2"),
                "https://api.cohere.com/v2/chat"
            );
        }

        #[test]
        fn trims_trailing_slash() {
            assert_eq!(
                chat_url("https://api.cohere.com/v2/"),
                "https://api.cohere.com/v2/chat"
            );
            assert_eq!(
                chat_url("https://api.cohere.com/v2//"),
                "https://api.cohere.com/v2/chat"
            );
        }

        #[test]
        fn does_not_duplicate_suffix() {
            assert_eq!(
                chat_url("https://api.cohere.com/v2/chat"),
                "https://api.cohere.com/v2/chat"
            );
            assert_eq!(
                chat_url("https://api.cohere.com/v2/chat/"),
                "https://api.cohere.com/v2/chat"
            );
        }
    }

    mod request_body {
        use super::*;

        #[test]
        fn includes_all_sections() {
            let model = model("https://api.cohere.com/v2");
            let context = LlmContext {
                system_prompt: "be helpful".into(),
                messages: vec![Message::User(UserMessage::new("hello"))],
                tools: vec![Arc::new(EchoTool)],
            };
            let body = build_request_body(&model, &context, &StreamFnOptions::default());
            assert_eq!(body["model"], "command-r-plus");
            assert_eq!(body["stream"], true);
            assert_eq!(body["messages"][0]["role"], "system");
            assert_eq!(body["messages"][0]["content"], "be helpful");
            assert_eq!(body["messages"][1]["role"], "user");
            assert_eq!(body["messages"][1]["content"], "hello");
            assert_eq!(body["tools"][0]["type"], "function");
            assert_eq!(body["tools"][0]["function"]["name"], "echo");
            assert_eq!(
                body["tools"][0]["function"]["parameters"]["required"],
                json!(["x"])
            );
        }

        #[test]
        fn serializes_assistant_tool_calls() {
            let model = model("http://example");
            let context = LlmContext {
                system_prompt: String::new(),
                messages: vec![
                    Message::User(UserMessage::new("run echo")),
                    Message::Assistant(AssistantMessage {
                        content: vec![
                            ContentBlock::text("calling echo"),
                            ContentBlock::tool_call("call_1", "echo", json!({"x": 1})),
                        ],
                        stop_reason: StopReason::ToolUse,
                        ..Default::default()
                    }),
                ],
                tools: vec![],
            };
            let body = build_request_body(&model, &context, &StreamFnOptions::default());
            let assistant_msg = &body["messages"][1];
            assert_eq!(assistant_msg["role"], "assistant");
            assert_eq!(assistant_msg["content"][0]["type"], "text");
            assert_eq!(assistant_msg["content"][0]["text"], "calling echo");
            assert_eq!(assistant_msg["tool_calls"][0]["id"], "call_1");
            assert_eq!(assistant_msg["tool_calls"][0]["type"], "function");
            assert_eq!(assistant_msg["tool_calls"][0]["function"]["name"], "echo");
            assert_eq!(
                assistant_msg["tool_calls"][0]["function"]["arguments"],
                "{\"x\":1}"
            );
        }

        #[test]
        fn serializes_tool_results() {
            let model = model("http://example");
            let context = LlmContext {
                system_prompt: String::new(),
                messages: vec![Message::ToolResult(ToolResultMessage {
                    tool_call_id: "call_1".into(),
                    tool_name: "echo".into(),
                    content: vec![ContentBlock::text("pong")],
                    details: json!({}),
                    usage: None,
                    added_tool_names: None,
                    is_error: false,
                    timestamp: 0,
                })],
                tools: vec![],
            };
            let body = build_request_body(&model, &context, &StreamFnOptions::default());
            let msg = &body["messages"][0];
            assert_eq!(msg["role"], "tool");
            assert_eq!(msg["content"], "pong");
            assert_eq!(msg["tool_call_id"], "call_1");
        }
    }

    mod vision {
        use super::*;

        #[test]
        fn image_serialized_as_content_array() {
            let mut m = model("http://example");
            m.input = vec![ModelInput::Text, ModelInput::Image];
            let context = LlmContext {
                system_prompt: String::new(),
                messages: vec![Message::User(UserMessage {
                    content: vec![
                        ContentBlock::text("what is this"),
                        ContentBlock::Image {
                            source: ImageSource {
                                mime_type: "image/png".into(),
                                bytes: vec![1, 2, 3],
                            },
                        },
                    ],
                    timestamp: 0,
                })],
                tools: vec![],
            };
            let body = build_request_body(&m, &context, &StreamFnOptions::default());
            let content = &body["messages"][0]["content"];
            assert!(
                content.is_array(),
                "content should be an array when images are present"
            );
            assert_eq!(content[0]["type"], "text");
            assert_eq!(content[0]["text"], "what is this");
            assert_eq!(content[1]["type"], "image_url");
            // base64 of [1,2,3] is "AQID"
            assert_eq!(content[1]["image_url"]["url"], "data:image/png;base64,AQID");
        }

        #[test]
        fn drops_when_unsupported() {
            // model.input has no Image — image blocks must be dropped.
            let m = model("http://example");
            let context = LlmContext {
                system_prompt: String::new(),
                messages: vec![Message::User(UserMessage {
                    content: vec![
                        ContentBlock::text("describe"),
                        ContentBlock::Image {
                            source: ImageSource {
                                mime_type: "image/png".into(),
                                bytes: vec![1, 2, 3],
                            },
                        },
                    ],
                    timestamp: 0,
                })],
                tools: vec![],
            };
            let body = build_request_body(&m, &context, &StreamFnOptions::default());
            // Text-only message → content is a string, image dropped.
            assert_eq!(body["messages"][0]["content"], "describe");
        }
    }

    mod sse {
        use super::*;

        #[test]
        fn text_stream_emits_all_events() {
            let frames = vec![
                sse_event("message-start", r#"{"type":"message-start"}"#),
                sse_event("content-start", r#"{"type":"content-start","index":0}"#),
                sse_event(
                    "content-delta",
                    r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"Hel"}}}}"#,
                ),
                sse_event(
                    "content-delta",
                    r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"lo"}}}}"#,
                ),
                sse_event("content-end", r#"{"type":"content-end","index":0}"#),
                sse_event(
                    "message-end",
                    r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE"}}"#,
                ),
            ];
            let events = collect_events(&frames);
            assert!(
                matches!(events.first(), Some(AssistantMessageEvent::Start { .. })),
                "first event should be Start, got {events:?}",
            );
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, AssistantMessageEvent::TextStart { .. })),
                "expected TextStart, events: {events:?}",
            );
            assert!(
                events.iter().any(|e| matches!(
                    e,
                    AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hel"
                )),
                "expected TextDelta \"Hel\", events: {events:?}",
            );
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, AssistantMessageEvent::TextEnd { .. })),
                "expected TextEnd, events: {events:?}",
            );
            assert!(
                matches!(
                    events.last(),
                    Some(AssistantMessageEvent::Done {
                        reason: StopReason::Stop,
                        ..
                    }),
                ),
                "last event should be Done(Stop), got {events:?}",
            );
        }

        #[test]
        fn tool_call_stream_emits_events() {
            let frames = vec![
                sse_event("message-start", r#"{"type":"message-start"}"#),
                sse_event(
                    "tool-plan-delta",
                    r#"{"type":"tool-plan-delta","delta":{"message":{"tool_plan":"hmm"}}}"#,
                ),
                sse_event(
                    "tool-call-start",
                    r#"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"echo","arguments":""}}]}}}"#,
                ),
                sse_event(
                    "tool-call-delta",
                    r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":[{"function":{"arguments":"{\"x\""}}]}}}"#,
                ),
                sse_event(
                    "tool-call-delta",
                    r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":[{"function":{"arguments":":1}"}}]}}}"#,
                ),
                sse_event("tool-call-end", r#"{"type":"tool-call-end","index":0}"#),
                sse_event(
                    "message-end",
                    r#"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL"}}"#,
                ),
            ];
            let events = collect_events(&frames);
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, AssistantMessageEvent::ThinkingStart { .. })),
                "expected ThinkingStart, events: {events:?}",
            );
            assert!(
                events.iter().any(|e| matches!(
                    e,
                    AssistantMessageEvent::ThinkingDelta { delta, .. } if delta == "hmm"
                )),
                "expected ThinkingDelta \"hmm\", events: {events:?}",
            );
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, AssistantMessageEvent::ThinkingEnd { .. })),
                "expected ThinkingEnd, events: {events:?}",
            );
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, AssistantMessageEvent::ToolCallStart { .. })),
                "expected ToolCallStart, events: {events:?}",
            );
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, AssistantMessageEvent::ToolCallDelta { .. })),
                "expected ToolCallDelta, events: {events:?}",
            );
            let end = events
                .iter()
                .find_map(|e| match e {
                    AssistantMessageEvent::ToolCallEnd { tool_call, .. } => Some(tool_call.clone()),
                    _ => None,
                })
                .expect("expected ToolCallEnd");
            assert_eq!(end.name, "echo");
            assert_eq!(end.arguments, json!({"x": 1}));
            assert!(
                matches!(
                    events.last(),
                    Some(AssistantMessageEvent::Done {
                        reason: StopReason::ToolUse,
                        ..
                    }),
                ),
                "last event should be Done(ToolUse), got {events:?}",
            );
        }

        #[test]
        fn max_tokens_maps_to_length() {
            let frames = vec![
                sse_event("message-start", r#"{"type":"message-start"}"#),
                sse_event("content-start", r#"{"type":"content-start","index":0}"#),
                sse_event(
                    "content-delta",
                    r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"x"}}}}"#,
                ),
                sse_event("content-end", r#"{"type":"content-end","index":0}"#),
                sse_event(
                    "message-end",
                    r#"{"type":"message-end","delta":{"finish_reason":"MAX_TOKENS"}}"#,
                ),
            ];
            let events = collect_events(&frames);
            assert!(
                matches!(
                    events.last(),
                    Some(AssistantMessageEvent::Done {
                        reason: StopReason::Length,
                        ..
                    }),
                ),
                "MAX_TOKENS should map to Length, got {events:?}",
            );
        }

        #[test]
        fn provider_error_event_emits_error() {
            let frame = sse_event(
                "error",
                r#"{"type":"error","error":{"message":"upstream failed"}}"#,
            );
            let events = collect_events(&[frame]);
            assert!(
                matches!(
                    events.last(),
                    Some(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        ..
                    }),
                ),
                "error event should emit Error, got {events:?}",
            );
        }

        #[test]
        fn late_tool_delta_after_close_ok() {
            // Protocol violation: tool-call-end arrives (closing the block and
            // removing the pending-tool entry), then a late tool-call-delta for
            // the same index. Must not panic on the BTreeMap index lookup.
            let frames = vec![
                sse_event("message-start", r#"{"type":"message-start"}"#),
                sse_event(
                    "tool-call-start",
                    r#"{"type":"tool-call-start","index":0,"delta":{"message":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"echo","arguments":""}}]}}}"#,
                ),
                sse_event(
                    "tool-call-delta",
                    r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":[{"function":{"arguments":"{\"x\":1}"}}]}}}"#,
                ),
                sse_event("tool-call-end", r#"{"type":"tool-call-end","index":0}"#),
                // Late delta after close — must be tolerated, not panic.
                sse_event(
                    "tool-call-delta",
                    r#"{"type":"tool-call-delta","index":0,"delta":{"message":{"tool_calls":[{"function":{"arguments":"{}"}}]}}}"#,
                ),
                sse_event(
                    "message-end",
                    r#"{"type":"message-end","delta":{"finish_reason":"TOOL_CALL"}}"#,
                ),
            ];
            let events = collect_events(&frames);
            assert!(
                matches!(
                    events.last(),
                    Some(AssistantMessageEvent::Done {
                        reason: StopReason::ToolUse,
                        ..
                    }),
                ),
                "late delta should not prevent Done, got {events:?}",
            );
        }

        #[test]
        fn out_of_order_stop_ignored() {
            // content-end for index 0 while block 1 is open should not close
            // block 1. A stale/duplicate stop for a different index must be
            // ignored, preserving Start/End pairing.
            let frames = vec![
                sse_event("message-start", r#"{"type":"message-start"}"#),
                sse_event("content-start", r#"{"type":"content-start","index":0}"#),
                sse_event(
                    "content-delta",
                    r#"{"type":"content-delta","index":0,"delta":{"message":{"content":{"text":"first"}}}}"#,
                ),
                sse_event("content-end", r#"{"type":"content-end","index":0}"#),
                sse_event("content-start", r#"{"type":"content-start","index":1}"#),
                sse_event(
                    "content-delta",
                    r#"{"type":"content-delta","index":1,"delta":{"message":{"content":{"text":"second"}}}}"#,
                ),
                // Duplicate stop for index 0 — should be ignored, not close block 1.
                sse_event("content-end", r#"{"type":"content-end","index":0}"#),
                sse_event("content-end", r#"{"type":"content-end","index":1}"#),
                sse_event(
                    "message-end",
                    r#"{"type":"message-end","delta":{"finish_reason":"COMPLETE"}}"#,
                ),
            ];
            let events = collect_events(&frames);
            // Block 1 should still get its TextEnd before Done.
            let text_ends: Vec<_> = events
                .iter()
                .filter_map(|e| match e {
                    AssistantMessageEvent::TextEnd { content, .. } => Some(content.clone()),
                    _ => None,
                })
                .collect();
            assert!(
                text_ends.iter().any(|t| t == "second"),
                "block 1 should close with \"second\", got text_ends: {text_ends:?}",
            );
        }
    }

    mod http {
        use super::*;

        #[tokio::test]
        async fn includes_auth_headers() {
            let (rx, fut) = captured_request_server();
            let addr = fut.await;
            let context = LlmContext {
                system_prompt: "sys".into(),
                messages: vec![Message::User(UserMessage::new("hello"))],
                tools: vec![Arc::new(EchoTool)],
            };
            let options = StreamFnOptions {
                api_key: Some("cchub-key".into()),
                ..Default::default()
            };
            let (_tx, abort) = watch::channel(false);
            let _ = collect(addr, context, options, abort).await;
            let raw = rx.await.unwrap();
            assert_eq!(
                raw.lines().next().unwrap_or(""),
                "POST /chat HTTP/1.1",
                "{raw}"
            );
            assert!(
                raw.to_ascii_lowercase()
                    .contains("authorization: bearer cchub-key"),
                "{raw}"
            );
            assert!(
                raw.to_ascii_lowercase().contains("user-agent: aaos"),
                "{raw}"
            );
            let body = raw.split("\r\n\r\n").nth(1).unwrap_or("");
            let json: Value = serde_json::from_str(body).unwrap();
            assert_eq!(json["model"], "command-r-plus");
            assert_eq!(json["stream"], true);
            assert_eq!(json["messages"][0]["role"], "system");
            assert_eq!(json["messages"][0]["content"], "sys");
            assert_eq!(json["messages"][1]["role"], "user");
            assert_eq!(json["messages"][1]["content"], "hello");
            assert_eq!(json["tools"][0]["type"], "function");
            assert_eq!(json["tools"][0]["function"]["name"], "echo");
        }

        #[tokio::test]
        async fn text_events_are_emitted() {
            let body = [
                "event: message-start\ndata: {\"type\":\"message-start\"}\n\n",
                "event: content-start\ndata: {\"type\":\"content-start\",\"index\":0}\n\n",
                "event: content-delta\ndata: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"Hel\"}}}}\n\n",
                "event: content-delta\ndata: {\"type\":\"content-delta\",\"index\":0,\"delta\":{\"message\":{\"content\":{\"text\":\"lo\"}}}}\n\n",
                "event: content-end\ndata: {\"type\":\"content-end\",\"index\":0}\n\n",
                "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"COMPLETE\"}}\n\n",
            ]
            .concat();
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
                    ..Default::default()
                },
                abort,
            )
            .await;
            h.await.unwrap();
            assert!(
                matches!(events.first(), Some(AssistantMessageEvent::Start { .. })),
                "first event should be Start, got {events:?}",
            );
            assert!(
                events.iter().any(|e| matches!(
                    e,
                    AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hel"
                )),
                "expected TextDelta \"Hel\", events: {events:?}",
            );
            assert!(
                matches!(
                    events.last(),
                    Some(AssistantMessageEvent::Done {
                        reason: StopReason::Stop,
                        ..
                    }),
                ),
                "last event should be Done(Stop), got {events:?}",
            );
            assert_eq!(content_text(&msg.content), "Hello");
        }

        #[tokio::test]
        async fn thinking_and_tool_call_events() {
            let body = [
                "event: message-start\ndata: {\"type\":\"message-start\"}\n\n",
                "event: tool-plan-delta\ndata: {\"type\":\"tool-plan-delta\",\"delta\":{\"message\":{\"tool_plan\":\"hmm\"}}}\n\n",
                "event: tool-call-start\ndata: {\"type\":\"tool-call-start\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"echo\",\"arguments\":\"\"}}]}}}\n\n",
                "event: tool-call-delta\ndata: {\"type\":\"tool-call-delta\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{\\\"x\\\":\"}}]}}}\n\n",
                "event: tool-call-delta\ndata: {\"type\":\"tool-call-delta\",\"index\":0,\"delta\":{\"message\":{\"tool_calls\":[{\"function\":{\"arguments\":\"1}\"}}]}}}\n\n",
                "event: tool-call-end\ndata: {\"type\":\"tool-call-end\",\"index\":0}\n\n",
                "event: message-end\ndata: {\"type\":\"message-end\",\"delta\":{\"finish_reason\":\"TOOL_CALL\"}}\n\n",
            ]
            .concat();
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
            h.await.unwrap();
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, AssistantMessageEvent::ThinkingStart { .. })),
                "expected ThinkingStart, events: {events:?}",
            );
            assert!(
                events.iter().any(|e| matches!(
                    e,
                    AssistantMessageEvent::ThinkingDelta { delta, .. } if delta == "hmm"
                )),
                "expected ThinkingDelta \"hmm\", events: {events:?}",
            );
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, AssistantMessageEvent::ToolCallStart { .. })),
                "expected ToolCallStart, events: {events:?}",
            );
            let end = events
                .iter()
                .find_map(|e| match e {
                    AssistantMessageEvent::ToolCallEnd { tool_call, .. } => Some(tool_call.clone()),
                    _ => None,
                })
                .unwrap();
            assert_eq!(end.name, "echo");
            assert_eq!(end.arguments, json!({"x": 1}));
            AgentTool::validate(&EchoTool, &end.arguments).unwrap();
            assert_eq!(msg.stop_reason, StopReason::ToolUse);
        }

        #[tokio::test]
        async fn error_stays_in_stream() {
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
            h.await.unwrap();
            assert!(
                matches!(
                    events.last(),
                    Some(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        ..
                    }),
                ),
                "last event should be Error, got {events:?}",
            );
            assert_eq!(msg.stop_reason, StopReason::Error);
        }
    }
}
