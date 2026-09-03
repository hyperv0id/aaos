//! Anthropic Messages streaming API format (SSE).
//!
//! `POST {base}/messages` with `x-api-key` authentication and
//! `anthropic-version: 2023-06-01` header. Borrowing providers
//! (freemodel/kimi/minimax/subconscious/thinkingmachines) share this
//! adapter — their base URLs differ but the wire format is identical
//! (spec §2, §5).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use base64::Engine;
use pi_agent_core::types::{
    AgentTool, AssistantEventStream, AssistantMessage, AssistantMessageEvent, ContentBlock,
    LlmContext, Message, Model, StopReason, StreamFn, StreamFnOptions, ThinkingLevel, ToolCall,
};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::sync::watch;

#[cfg(test)]
use pi_agent_core::types::ModelInput;

use super::sse::{self, Ctx, SseFormat, content_text, parse_tool_args, supports_images};

/// `Model::api` key dispatching to this format.
pub const API: &str = "anthropic-messages";

/// Anthropic API version header value.
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Default token budget for extended thinking when the model is reasoning-capable.
const DEFAULT_THINKING_BUDGET: u64 = 1024;

/// Build the `/messages` endpoint from a base URL that already contains the
/// version path (spec §2). Trims trailing slashes, avoids duplicating an
/// existing `/messages` suffix, and appends `/messages` otherwise.
fn messages_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    if trimmed.ends_with("/messages") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/messages")
    }
}

/// Anthropic tool schema: `{ name, description, input_schema }` (spec §2).
fn tools_payload(tools: &[Arc<dyn AgentTool>]) -> Option<Value> {
    if tools.is_empty() {
        return None;
    }
    Some(Value::Array(
        tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name(),
                    "description": t.description(),
                    "input_schema": t.parameters()
                })
            })
            .collect(),
    ))
}

/// Serialize a user message's content blocks into the Anthropic content format.
///
/// - Text-only messages become a plain string (Anthropic accepts this).
/// - Messages with images become a content-block array with
///   `{type: "image", source: {type: "base64", media_type, data}}` (spec §6).
/// - Image-only messages get a placeholder text block prepended, since
///   Anthropic requires at least one text block (Pi `anthropic-messages.ts`).
///
/// Returns `None` when the message would only contain an image that the model
/// cannot accept — the caller drops such messages rather than sending an
/// unsupported block.
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

    // Image present but model doesn't accept images: emit only the text blocks,
    // dropping image blocks entirely (spec §6: don't send image blocks when
    // `model.input` lacks `Image`).
    let can_send_images = supports_images(model);

    let mut parts: Vec<Value> = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } if !text.is_empty() => {
                parts.push(json!({ "type": "text", "text": text }));
            }
            ContentBlock::Image { source } if can_send_images => {
                parts.push(json!({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": source.mime_type,
                        "data": base64::engine::general_purpose::STANDARD.encode(&source.bytes)
                    }
                }));
            }
            _ => {}
        }
    }
    // If images were dropped (model can't accept them) and only text blocks
    // survive, flatten to a plain string — matching the text-only path and
    // avoiding a single-element array where Anthropic accepts a string.
    let has_image = parts
        .iter()
        .any(|p| p.get("type").and_then(|t| t.as_str()) == Some("image"));
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

    // Images present: Anthropic requires at least one text block (Pi
    // `anthropic-messages.ts` adds a placeholder for image-only messages).
    let has_text = parts
        .iter()
        .any(|p| p.get("type").and_then(|t| t.as_str()) == Some("text"));
    if !has_text {
        parts.insert(0, json!({ "type": "text", "text": "(see attached image)" }));
    }

    Some(Value::Array(parts))
}

/// Serialize an assistant message's content blocks for request replay.
fn serialize_assistant_content(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } if !text.is_empty() => {
                Some(json!({ "type": "text", "text": text }))
            }
            ContentBlock::ToolCall(tc) => Some(json!({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments
            })),
            // Thinking blocks are not replayed in the simplified aaos model
            // (no signature tracking). Pi replays them with signatures; aaos
            // omits thinking from the request body.
            _ => None,
        })
        .collect()
}

/// Build the Anthropic Messages request body (spec §2, issue 08).
fn build_request_body(model: &Model, context: &LlmContext, options: &StreamFnOptions) -> Value {
    let mut messages: Vec<Value> = Vec::new();
    for msg in &context.messages {
        match msg {
            Message::User(u) => {
                if let Some(content) = serialize_user_content(&u.content, model) {
                    messages.push(json!({ "role": "user", "content": content }));
                }
            }
            Message::Assistant(a) => {
                let blocks = serialize_assistant_content(&a.content);
                if !blocks.is_empty() {
                    messages.push(json!({ "role": "assistant", "content": blocks }));
                }
            }
            Message::ToolResult(t) => {
                messages.push(json!({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": t.tool_call_id,
                        "content": content_text(&t.content),
                        "is_error": t.is_error
                    }]
                }));
            }
        }
    }

    let mut body = serde_json::Map::new();
    body.insert("model".into(), json!(model.id));
    body.insert("stream".into(), json!(true));
    body.insert("messages".into(), Value::Array(messages));
    body.insert(
        "max_tokens".into(),
        json!(if model.max_tokens > 0 {
            model.max_tokens
        } else {
            4096
        }),
    );

    if !context.system_prompt.is_empty() {
        body.insert("system".into(), json!(context.system_prompt));
    }

    if let Some(tools) = tools_payload(&context.tools) {
        body.insert("tools".into(), tools);
    }

    let thinking = options.thinking_level.unwrap_or(ThinkingLevel::Off);
    if model.reasoning && thinking != ThinkingLevel::Off {
        body.insert(
            "thinking".into(),
            json!({ "type": "enabled", "budget_tokens": DEFAULT_THINKING_BUDGET }),
        );
    }

    Value::Object(body)
}

pub struct AnthropicMessagesProvider {
    client: Client,
}

impl AnthropicMessagesProvider {
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
impl StreamFn for AnthropicMessagesProvider {
    async fn call(
        &self,
        model: Model,
        context: LlmContext,
        options: StreamFnOptions,
        abort: watch::Receiver<bool>,
    ) -> Result<Box<dyn AssistantEventStream>, String> {
        let api_key = options.api_key.clone().unwrap_or_default();
        let body = build_request_body(&model, &context, &options);
        let url = messages_url(&model.base_url);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
        let client = self.client.clone();
        let final_seed = AssistantMessage {
            model: model.id.clone(),
            provider: model.provider.clone(),
            api: model.api.clone(),
            stop_reason: StopReason::Pending,
            ..Default::default()
        };
        let headers = [
            ("x-api-key", api_key.clone()),
            ("anthropic-version", ANTHROPIC_VERSION.to_string()),
        ];
        tokio::spawn(async move {
            sse::run_stream::<AnthropicFormat>(client, url, model, body, abort, tx, headers).await;
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

/// Anthropic chunk interpretation: named events keyed by provider content
/// block index, mapped onto our content slots through `block_map`.
#[derive(Default)]
struct AnthropicFormat {
    open: OpenBlock,
    /// Maps Anthropic content block index → pending tool accumulator.
    pending_tools: BTreeMap<usize, PendingTool>,
    /// Maps Anthropic content block index → our content[] index, so deltas
    /// for a known block find the right slot.
    block_map: BTreeMap<usize, usize>,
}
#[cfg(test)]
type EventBuilder = super::sse::EventBuilder<AnthropicFormat>;

impl SseFormat for AnthropicFormat {
    /// Dispatch one decoded payload. Anthropic streams carry named events;
    /// `event:` supplies the dispatch key.
    fn apply_chunk(&mut self, cx: &mut Ctx<'_>, event: Option<&str>, value: &Value) {
        cx.ensure_start();
        match event.unwrap_or("") {
            "message_start" => {
                // message_start carries input-side usage: input_tokens plus
                // cache-creation/read tokens (issue 70).
                self.apply_start_usage(cx, value);
            }
            "content_block_start" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let block = value.get("content_block");
                let block_type = block
                    .and_then(|b| b.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                match block_type {
                    "text" => {
                        let initial = block
                            .and_then(|b| b.get("text"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        self.start_text(cx, index, initial);
                    }
                    "thinking" => {
                        let initial = block
                            .and_then(|b| b.get("thinking"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        self.start_thinking(cx, index, initial);
                    }
                    "tool_use" => {
                        let id = block
                            .and_then(|b| b.get("id"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let name = block
                            .and_then(|b| b.get("name"))
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        self.start_tool(cx, index, id, name);
                    }
                    "redacted_thinking" => {
                        // Anthropic redacts thinking for some models/orgs.
                        // Surface as a Thinking block with a redaction note
                        // instead of silently dropping it (Pi maps to a
                        // thinking block with "[Reasoning redacted]").
                        self.start_thinking(cx, index, "[Reasoning redacted]");
                    }
                    _ => {}
                }
            }
            "content_block_delta" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                let delta = value.get("delta");
                let delta_type = delta
                    .and_then(|d| d.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("");
                match delta_type {
                    "text_delta" => {
                        let text = delta
                            .and_then(|d| d.get("text"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        if !text.is_empty() {
                            self.push_text(cx, index, text);
                        }
                    }
                    "thinking_delta" => {
                        let text = delta
                            .and_then(|d| d.get("thinking"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        if !text.is_empty() {
                            self.push_thinking(cx, index, text);
                        }
                    }
                    "input_json_delta" => {
                        let partial = delta
                            .and_then(|d| d.get("partial_json"))
                            .and_then(|t| t.as_str())
                            .unwrap_or("");
                        if !partial.is_empty() {
                            self.push_tool_delta(cx, index, partial);
                        }
                    }
                    _ => {}
                }
            }
            "content_block_stop" => {
                let index = value.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
                self.close_block(cx, index);
            }
            "message_delta" => {
                // Terminal delta carries output_tokens; input-side figures
                // arrived on message_start.
                self.apply_delta_usage(cx, value);
                if let Some(reason) = value.pointer("/delta/stop_reason").and_then(|v| v.as_str()) {
                    self.finish_reason(cx, reason);
                }
            }
            "message_stop" if !*cx.finished => {
                self.close_done(cx);
            }
            _ => {}
        }
    }

    fn close_open(&mut self, cx: &mut Ctx<'_>) {
        match self.open {
            OpenBlock::Text(index) => {
                let content_index = self
                    .block_map
                    .get(&index)
                    .copied()
                    .unwrap_or_else(|| cx.current_index());
                let content = match &cx.message.content[content_index] {
                    ContentBlock::Text { text } => text.clone(),
                    _ => String::new(),
                };
                cx.emit(AssistantMessageEvent::TextEnd {
                    content_index,
                    content,
                    partial: cx.message.clone(),
                });
            }
            OpenBlock::Thinking(index) => {
                let content_index = self
                    .block_map
                    .get(&index)
                    .copied()
                    .unwrap_or_else(|| cx.current_index());
                let content = match &cx.message.content[content_index] {
                    ContentBlock::Thinking { text } => text.clone(),
                    _ => String::new(),
                };
                cx.emit(AssistantMessageEvent::ThinkingEnd {
                    content_index,
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
            "max_tokens" => StopReason::Length,
            "tool_use" => StopReason::ToolUse,
            _ => StopReason::Stop,
        }
    }
}

impl AnthropicFormat {
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
                // Delta without a preceding content_block_start: open a text block.
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
        // content_block_stop is a protocol violation — tolerate it by
        // returning early instead of panicking on the map lookup.
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
        // Only close if this stop refers to the currently open block.
        // A stale or duplicate stop for a different block is ignored
        // rather than closing the wrong block and breaking Start/End pairing.
        let is_current = match self.open {
            OpenBlock::None => false,
            OpenBlock::Text(i) | OpenBlock::Thinking(i) | OpenBlock::Tool(i) => i == index,
        };
        if is_current {
            self.close_open(cx);
        }
    }

    /// Record input-side usage from `message_start`:
    /// `message.usage.input_tokens`, `cache_creation_input_tokens`,
    /// `cache_read_input_tokens`. Anthropic's `input_tokens` excludes cache
    /// tokens, so the running total is input + output + cache_read +
    /// cache_write (output arrives later on `message_delta`, applied by
    /// [`Self::apply_delta_usage`]).
    fn apply_start_usage(&self, cx: &mut Ctx<'_>, value: &Value) {
        let usage = value
            .pointer("/message/usage")
            .cloned()
            .unwrap_or(Value::Null);
        let mut u = cx.message.usage;
        u.input = usage
            .get("input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        u.cache_write = usage
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        u.cache_read = usage
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        u.total_tokens = u.input + u.output + u.cache_read + u.cache_write;
        cx.message.usage = u;
    }

    /// Record output-side usage from the terminal `message_delta` and
    /// recompute the running total.
    fn apply_delta_usage(&self, cx: &mut Ctx<'_>, value: &Value) {
        let mut u = cx.message.usage;
        u.output = value
            .pointer("/usage/output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        u.total_tokens = u.input + u.output + u.cache_read + u.cache_write;
        cx.message.usage = u;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    use pi_agent_core::types::{
        AgentToolResult, ImageSource, ToolResultMessage, Usage, UserMessage,
    };

    use std::net::SocketAddr;

    use std::time::Duration;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use tokio::net::TcpListener;

    use tokio::sync::watch;

    fn model(base: &str) -> Model {
        Model {
            id: "claude-sonnet-4-20250514".into(),
            name: "Claude".into(),
            api: API.into(),
            provider: "anthropic".into(),
            base_url: base.into(),
            reasoning: true,
            input: vec![],
            cost: Default::default(),
            context_window: 200_000,
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

    fn done_message(events: &[AssistantMessageEvent]) -> AssistantMessage {
        events
            .iter()
            .find_map(|e| match e {
                AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
                _ => None,
            })
            .expect("expected a Done event")
    }

    struct EchoTool;

    #[async_trait]
    impl AgentTool for EchoTool {
        fn name(&self) -> &str {
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
        let provider = AnthropicMessagesProvider::new().expect("HTTP client");
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
                    let body = "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
                        .to_string()
                        + "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n"
                        + "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
                        + "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n"
                        + "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
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
                messages_url("https://api.anthropic.com/v1"),
                "https://api.anthropic.com/v1/messages"
            );
            // Borrowing provider: freemodel base carries /v1, adapter appends /messages.
            assert_eq!(
                messages_url("https://cc.freemodel.dev/v1"),
                "https://cc.freemodel.dev/v1/messages"
            );
        }

        #[test]
        fn trims_trailing_slash() {
            assert_eq!(
                messages_url("https://api.anthropic.com/v1/"),
                "https://api.anthropic.com/v1/messages"
            );
            assert_eq!(
                messages_url("https://api.anthropic.com/v1//"),
                "https://api.anthropic.com/v1/messages"
            );
        }

        #[test]
        fn does_not_duplicate_suffix() {
            assert_eq!(
                messages_url("https://api.anthropic.com/v1/messages"),
                "https://api.anthropic.com/v1/messages"
            );
            assert_eq!(
                messages_url("https://api.anthropic.com/v1/messages/"),
                "https://api.anthropic.com/v1/messages"
            );
        }
    }

    mod request_body {
        use super::*;

        #[test]
        fn includes_all_sections() {
            let model = model("https://api.anthropic.com/v1");
            let context = LlmContext {
                system_prompt: "be helpful".into(),
                messages: vec![Message::User(UserMessage::new("hello"))],
                tools: vec![Arc::new(EchoTool)],
            };
            let body = build_request_body(
                &model,
                &context,
                &StreamFnOptions {
                    thinking_level: Some(ThinkingLevel::High),
                    ..Default::default()
                },
            );
            assert_eq!(body["model"], "claude-sonnet-4-20250514");
            assert_eq!(body["stream"], true);
            assert_eq!(body["max_tokens"], 4096);
            assert_eq!(body["system"], "be helpful");
            assert_eq!(body["messages"][0]["role"], "user");
            assert_eq!(body["messages"][0]["content"], "hello");
            assert_eq!(body["tools"][0]["name"], "echo");
            assert_eq!(body["tools"][0]["input_schema"]["required"], json!(["x"]));
            assert_eq!(body["thinking"]["type"], "enabled");
            assert_eq!(body["thinking"]["budget_tokens"], DEFAULT_THINKING_BUDGET);
        }

        #[test]
        fn omits_thinking_when_off() {
            let model = model("https://api.anthropic.com/v1");
            let context = LlmContext {
                system_prompt: String::new(),
                messages: vec![Message::User(UserMessage::new("hi"))],
                tools: vec![],
            };
            let body = build_request_body(
                &model,
                &context,
                &StreamFnOptions {
                    thinking_level: Some(ThinkingLevel::Off),
                    ..Default::default()
                },
            );
            assert!(body.get("thinking").is_none());
        }

        #[test]
        fn includes_tool_result_messages() {
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
            let body = build_request_body(&model, &context, &StreamFnOptions::default());
            // Tool results are wrapped in a user message with tool_result content blocks.
            let tool_msg = &body["messages"][1];
            assert_eq!(tool_msg["role"], "user");
            assert_eq!(tool_msg["content"][0]["type"], "tool_result");
            assert_eq!(tool_msg["content"][0]["tool_use_id"], "call_1");
            assert_eq!(tool_msg["content"][0]["content"], "pong");
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
            assert_eq!(assistant_msg["content"][1]["type"], "tool_use");
            assert_eq!(assistant_msg["content"][1]["id"], "call_1");
            assert_eq!(assistant_msg["content"][1]["name"], "echo");
            assert_eq!(assistant_msg["content"][1]["input"]["x"], 1);
        }
    }

    mod vision {
        use super::*;

        #[test]
        fn image_serialized_as_source_block() {
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
                "content should be array when images present"
            );
            assert_eq!(content[0]["type"], "text");
            assert_eq!(content[0]["text"], "what is this");
            assert_eq!(content[1]["type"], "image");
            assert_eq!(content[1]["source"]["type"], "base64");
            assert_eq!(content[1]["source"]["media_type"], "image/png");
            // base64 of [1,2,3] is "AQID"
            assert_eq!(content[1]["source"]["data"], "AQID");
        }

        #[test]
        fn image_only_message_placeholder() {
            let mut m = model("http://example");
            m.input = vec![ModelInput::Text, ModelInput::Image];
            let context = LlmContext {
                system_prompt: String::new(),
                messages: vec![Message::User(UserMessage {
                    content: vec![ContentBlock::Image {
                        source: ImageSource {
                            mime_type: "image/png".into(),
                            bytes: vec![1, 2, 3],
                        },
                    }],
                    timestamp: 0,
                })],
                tools: vec![],
            };
            let body = build_request_body(&m, &context, &StreamFnOptions::default());
            let content = &body["messages"][0]["content"];
            assert!(content.is_array());
            // First block should be the placeholder text.
            assert_eq!(content[0]["type"], "text");
            assert_eq!(content[0]["text"], "(see attached image)");
            assert_eq!(content[1]["type"], "image");
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
                sse_event(
                    "message_start",
                    r#"{"type":"message_start","message":{"id":"msg_1"}}"#,
                ),
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hel"}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"lo"}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                sse_event(
                    "message_delta",
                    r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
                ),
                sse_event("message_stop", r#"{"type":"message_stop"}"#),
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
        fn thinking_stream_emits_events() {
            let frames = vec![
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"hmm"}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"!"}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"done"}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":1}"#,
                ),
                sse_event(
                    "message_delta",
                    r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
                ),
                sse_event("message_stop", r#"{"type":"message_stop"}"#),
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
            // Text block after thinking.
            assert!(
                events.iter().any(
                    |e| matches!(e, AssistantMessageEvent::TextDelta { delta, .. } if delta == "done")
                ),
                "expected TextDelta \"done\", events: {events:?}",
            );
        }

        #[test]
        fn tool_use_stream_emits_events() {
            let frames = vec![
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"echo","input":{}}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"x\""}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":":1}"}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                sse_event(
                    "message_delta",
                    r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
                ),
                sse_event("message_stop", r#"{"type":"message_stop"}"#),
            ];
            let events = collect_events(&frames);
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
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"..."}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                sse_event(
                    "message_delta",
                    r#"{"type":"message_delta","delta":{"stop_reason":"max_tokens"}}"#,
                ),
                sse_event("message_stop", r#"{"type":"message_stop"}"#),
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
                "max_tokens should map to Length, got {events:?}",
            );
        }

        #[test]
        fn provider_error_event_emits_error() {
            let frame = sse_event(
                "error",
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
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
        fn redacted_thinking_block() {
            let frames = vec![
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"answer"}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":1}"#,
                ),
                sse_event(
                    "message_delta",
                    r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
                ),
                sse_event("message_stop", r#"{"type":"message_stop"}"#),
            ];
            let events = collect_events(&frames);
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, AssistantMessageEvent::ThinkingStart { .. })),
                "redacted_thinking should surface as ThinkingStart, events: {events:?}",
            );
            assert!(
                events.iter().any(|e| matches!(
                    e,
                    AssistantMessageEvent::ThinkingEnd { content, .. } if content == "[Reasoning redacted]"
                )),
                "redacted_thinking should surface as ThinkingEnd with redaction note, events: {events:?}",
            );
        }

        #[test]
        fn late_tool_delta_after_close_ok() {
            // Protocol violation: content_block_stop arrives, then a late
            // input_json_delta for the same index. Must not panic.
            let frames = vec![
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call_1","name":"echo","input":{}}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                // Late delta after close — must be tolerated, not panic.
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{}"}}"#,
                ),
                sse_event(
                    "message_delta",
                    r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"}}"#,
                ),
                sse_event("message_stop", r#"{"type":"message_stop"}"#),
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
            // content_block_stop for index 0 while block 1 is open should
            // not close block 1.
            let frames = vec![
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"first"}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
                ),
                sse_event(
                    "content_block_delta",
                    r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"second"}}"#,
                ),
                // Duplicate stop for index 0 — should be ignored, not close block 1.
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":1}"#,
                ),
                sse_event(
                    "message_delta",
                    r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#,
                ),
                sse_event("message_stop", r#"{"type":"message_stop"}"#),
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

        #[test]
        fn usage_parsed_from_message_events() {
            let frames = vec![
                sse_event(
                    "message_start",
                    r#"{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":25,"cache_creation_input_tokens":11,"cache_read_input_tokens":7}}}"#,
                ),
                sse_event(
                    "content_block_start",
                    r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":"hi"}}"#,
                ),
                sse_event(
                    "content_block_stop",
                    r#"{"type":"content_block_stop","index":0}"#,
                ),
                sse_event(
                    "message_delta",
                    r#"{"type":"message_delta","usage":{"output_tokens":9},"delta":{"stop_reason":"end_turn"}}"#,
                ),
                sse_event("message_stop", r#"{"type":"message_stop"}"#),
            ];
            let events = collect_events(&frames);
            let done = done_message(&events);
            assert_eq!(
                done.usage,
                Usage {
                    input: 25,
                    output: 9,
                    cache_read: 7,
                    cache_write: 11,
                    total_tokens: 52,
                    cost: Default::default(),
                }
            );
        }
    }

    mod http {
        use super::*;

        #[tokio::test]
        async fn includes_auth_and_headers() {
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
            assert_eq!(
                raw.lines().next().unwrap_or(""),
                "POST /messages HTTP/1.1",
                "{raw}"
            );
            assert!(
                raw.to_ascii_lowercase().contains("x-api-key: cchub-key"),
                "{raw}"
            );
            assert!(raw.contains("anthropic-version: 2023-06-01"), "{raw}");
            assert!(
                raw.to_ascii_lowercase().contains("user-agent: aaos"),
                "{raw}"
            );
            let body = raw.split("\r\n\r\n").nth(1).unwrap_or("");
            let json: Value = serde_json::from_str(body).unwrap();
            assert_eq!(json["model"], "claude-sonnet-4-20250514");
            assert_eq!(json["system"], "sys");
            assert_eq!(json["messages"][0]["role"], "user");
            assert_eq!(json["messages"][0]["content"], "hello");
            assert_eq!(json["tools"][0]["name"], "echo");
            assert_eq!(json["tools"][0]["input_schema"]["required"], json!(["x"]));
        }

        #[tokio::test]
        async fn text_events_are_emitted() {
            let body = [
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n",
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
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
                matches!(events.first(), Some(AssistantMessageEvent::Start { .. }),),
                "first event should be Start, got {events:?}",
            );
            assert!(
                events.iter().any(
                    |e| matches!(e, AssistantMessageEvent::TextDelta { delta, .. } if delta == "Hel")
                ),
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
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"hmm\"}}\n\n",
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"call_1\",\"name\":\"echo\",\"input\":{}}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"x\\\":\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"1}\"}}\n\n",
                "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
                "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"}}\n\n",
                "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
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

        #[tokio::test]
        async fn provider_error_stays_in_stream() {
            let body = "event: error\ndata: {\"type\":\"error\",\"error\":{\"message\":\"upstream failed\"}}\n\n";
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
            assert!(msg.error_message.unwrap().contains("upstream failed"));
        }

        #[tokio::test]
        async fn abort_cancels_body() {
            let body = [
                "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"aaaa\"}}\n\n",
                "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"bbbb\"}}\n\n",
            ]
            .concat();
            let (addr, h) = serve(200, body, 80).await;
            let (tx, abort) = watch::channel(false);
            let (seen_delta_tx, seen_delta_rx) = tokio::sync::oneshot::channel::<()>();
            let provider = AnthropicMessagesProvider::new().expect("HTTP client");
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
                let mut signal = Some(seen_delta_tx);
                while let Some(ev) = stream.next_event().await {
                    if matches!(ev, AssistantMessageEvent::TextDelta { .. })
                        && let Some(tx) = signal.take()
                    {
                        let _ = tx.send(());
                    }
                    events.push(ev);
                }
                (events, stream.result().await)
            });
            seen_delta_rx.await.unwrap();
            let _ = tx.send(true);
            let (events, msg) = handle.await.unwrap();
            h.await.unwrap();
            assert!(
                events.iter().any(|e| matches!(
                    e,
                    AssistantMessageEvent::Error {
                        reason: StopReason::Aborted,
                        ..
                    }
                )),
                "expected an Aborted error, events: {events:?}",
            );
            assert_eq!(msg.stop_reason, StopReason::Aborted);
        }
    }
}
