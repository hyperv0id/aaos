//! Google Generative Language streaming API format (SSE).
//!
//! `POST {base}/models/{model_id}:streamGenerateContent?alt=sse` with
//! `x-goog-api-key` authentication. Each SSE frame is a `data:`-only
//! `GenerateContentResponse` JSON (no `event:` lines); the stream ends when
//! the HTTP body ends — there is no `[DONE]` sentinel.

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

use super::sse::{self, Ctx, SseFormat, content_text, supports_images};

/// `Model::api` key dispatching to this format.
pub const API: &str = "google-genai";

/// Build the `:streamGenerateContent?alt=sse` endpoint from a base URL that
/// already contains the version path. Trims trailing slashes, avoids
/// duplicating an existing `:streamGenerateContent` suffix (appending
/// `?alt=sse` when the suffix lacks the query), and appends the full tail
/// otherwise.
fn stream_generate_url(base_url: &str, model_id: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');
    let tail = format!("/models/{model_id}:streamGenerateContent");
    if trimmed.ends_with(&format!("{tail}?alt=sse")) {
        return trimmed.to_string();
    }
    if trimmed.ends_with(&tail) {
        return format!("{trimmed}?alt=sse");
    }
    format!("{trimmed}{tail}?alt=sse")
}

/// Google tool schema: `[{ "functionDeclarations": [{ name, description,
/// parameters }] }]`.
fn tools_payload(tools: &[Arc<dyn AgentTool>]) -> Option<Value> {
    if tools.is_empty() {
        return None;
    }
    Some(Value::Array(vec![json!({
        "functionDeclarations": tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name(),
                    "description": t.description(),
                    "parameters": t.parameters()
                })
            })
            .collect::<Vec<_>>()
    })]))
}

/// Serialize a user message's content blocks into Google `parts` entries.
///
/// Text blocks become `{"text": ...}`; image blocks become
/// `{"inlineData": {"mimeType", "data"}}` with base64 payload (spec §6).
/// Image blocks are dropped when the model does not accept image input —
/// never sent silently. Returns an empty vec when every block was dropped;
/// the caller skips such messages.
fn serialize_user_parts(blocks: &[ContentBlock], model: &Model) -> Vec<Value> {
    let can_send_images = supports_images(model);
    let mut parts = Vec::new();
    for block in blocks {
        match block {
            ContentBlock::Text { text } if !text.is_empty() => {
                parts.push(json!({ "text": text }));
            }
            ContentBlock::Image { source } if can_send_images => {
                parts.push(json!({
                    "inlineData": {
                        "mimeType": source.mime_type,
                        "data": base64::engine::general_purpose::STANDARD.encode(&source.bytes)
                    }
                }));
            }
            _ => {}
        }
    }
    parts
}

/// Serialize an assistant message's content blocks for request replay:
/// `{"text": ...}` and `{"functionCall": {"name", "args"}}` parts.
fn serialize_assistant_parts(blocks: &[ContentBlock]) -> Vec<Value> {
    blocks
        .iter()
        .filter_map(|block| match block {
            ContentBlock::Text { text } if !text.is_empty() => Some(json!({ "text": text })),
            ContentBlock::ToolCall(tc) => Some(json!({
                "functionCall": { "name": tc.name, "args": tc.arguments }
            })),
            // Thinking blocks are not replayed in the simplified aaos model.
            _ => None,
        })
        .collect()
}

/// Build the Google GenerateContent request body (spec §2, issue 09).
fn build_request_body(model: &Model, context: &LlmContext, _options: &StreamFnOptions) -> Value {
    let mut contents: Vec<Value> = Vec::new();
    for msg in &context.messages {
        match msg {
            Message::User(u) => {
                let parts = serialize_user_parts(&u.content, model);
                if !parts.is_empty() {
                    contents.push(json!({ "role": "user", "parts": parts }));
                }
            }
            Message::Assistant(a) => {
                let parts = serialize_assistant_parts(&a.content);
                if !parts.is_empty() {
                    contents.push(json!({ "role": "model", "parts": parts }));
                }
            }
            Message::ToolResult(t) => {
                contents.push(json!({
                    "role": "function",
                    "parts": [{
                        "functionResponse": {
                            "name": t.tool_name,
                            "response": { "content": content_text(&t.content) }
                        }
                    }]
                }));
            }
        }
    }

    let mut body = serde_json::Map::new();
    body.insert("contents".into(), Value::Array(contents));
    body.insert("stream".into(), json!(true));
    body.insert("generationConfig".into(), json!({ "candidateCount": 1 }));

    if !context.system_prompt.is_empty() {
        body.insert(
            "systemInstruction".into(),
            json!({ "parts": [{ "text": context.system_prompt }] }),
        );
    }

    if let Some(tools) = tools_payload(&context.tools) {
        body.insert("tools".into(), tools);
    }

    Value::Object(body)
}

pub struct GoogleGenAiProvider {
    client: Client,
}

impl GoogleGenAiProvider {
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
impl StreamFn for GoogleGenAiProvider {
    async fn call(
        &self,
        model: Model,
        context: LlmContext,
        options: StreamFnOptions,
        abort: watch::Receiver<bool>,
    ) -> Result<Box<dyn AssistantEventStream>, String> {
        let api_key = options.api_key.clone().unwrap_or_default();
        let body = build_request_body(&model, &context, &options);
        let url = stream_generate_url(&model.base_url, &model.id);
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<AssistantMessageEvent>();
        let client = self.client.clone();
        let final_seed = AssistantMessage {
            model: model.id.clone(),
            provider: model.provider.clone(),
            api: model.api.clone(),
            stop_reason: StopReason::Pending,
            ..Default::default()
        };
        let headers = [("x-goog-api-key", api_key)];
        tokio::spawn(async move {
            sse::run_stream::<GoogleFormat>(client, url, model, body, abort, tx, headers).await;
        });
        Ok(sse::SseStream::boxed(rx, final_seed))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum OpenBlock {
    #[default]
    None,
    Text,
    Thinking,
    /// Content index of the open tool block. Google sends complete
    /// `functionCall` parts, so a tool block opens and closes within one part.
    Tool(u32),
}

/// Google chunk interpretation: `data:`-only frames carrying complete parts;
/// position-driven blocks with no provider-side indices to track.
#[derive(Default)]
struct GoogleFormat {
    open: OpenBlock,
}
#[cfg(test)]
type EventBuilder = super::sse::EventBuilder<GoogleFormat>;

impl SseFormat for GoogleFormat {
    fn apply_chunk(&mut self, cx: &mut Ctx<'_>, _event: Option<&str>, value: &Value) {
        cx.ensure_start();
        let candidate = value
            .pointer("/candidates/0")
            .cloned()
            .unwrap_or(Value::Null);

        if let Some(parts) = candidate
            .pointer("/content/parts")
            .and_then(|v| v.as_array())
        {
            for part in parts {
                let text = part.get("text").and_then(|t| t.as_str()).unwrap_or("");
                let is_thought = part
                    .get("thought")
                    .and_then(|t| t.as_bool())
                    .unwrap_or(false);
                if is_thought {
                    if !text.is_empty() {
                        self.push_thinking(cx, text);
                    }
                } else if !text.is_empty() {
                    self.push_text(cx, text);
                } else if let Some(fc) = part.get("functionCall") {
                    let name = fc.get("name").and_then(|n| n.as_str()).unwrap_or("");
                    let args = fc.get("args").cloned().unwrap_or_else(|| json!({}));
                    if !name.is_empty() {
                        self.push_tool(cx, name, &args);
                    }
                }
            }
        }

        // The final chunk carries `usageMetadata`; record it before the
        // finishReason terminates the stream so the final message reports
        // real token usage (issue 70).
        self.apply_usage(cx, value);
        if let Some(reason) = candidate.get("finishReason").and_then(|v| v.as_str())
            && !reason.is_empty()
        {
            self.finish_reason(cx, reason);
        }
    }

    fn close_open(&mut self, cx: &mut Ctx<'_>) {
        match self.open {
            OpenBlock::Text => {
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
            OpenBlock::Thinking => {
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
            OpenBlock::Tool(idx) => {
                let content_index = idx as usize;
                let tool_call = match &cx.message.content[content_index] {
                    ContentBlock::ToolCall(tc) => tc.clone(),
                    // Unreachable in practice — push_tool always pushes a
                    // ToolCall block before opening a Tool block; keep the
                    // stream alive rather than panicking.
                    _ => ToolCall {
                        id: String::new(),
                        name: String::new(),
                        arguments: json!({}),
                    },
                };
                cx.emit(AssistantMessageEvent::ToolCallEnd {
                    content_index,
                    tool_call,
                    partial: cx.message.clone(),
                });
            }
            OpenBlock::None => {}
        }
        self.open = OpenBlock::None;
    }

    fn stop_reason(&self, reason: &str) -> StopReason {
        match reason {
            "MAX_TOKENS" => StopReason::Length,
            // STOP, SAFETY, RECITATION, MALFORMED_FUNCTION_CALL, etc.
            _ => StopReason::Stop,
        }
    }
}

impl GoogleFormat {
    fn push_text(&mut self, cx: &mut Ctx<'_>, delta: &str) {
        if self.open != OpenBlock::Text {
            self.close_open(cx);
            cx.message.content.push(ContentBlock::text(""));
            let idx = cx.current_index();
            self.open = OpenBlock::Text;
            cx.emit(AssistantMessageEvent::TextStart {
                content_index: idx,
                partial: cx.message.clone(),
            });
        }
        let idx = cx.current_index();
        if let ContentBlock::Text { text } = &mut cx.message.content[idx] {
            text.push_str(delta);
        }
        cx.emit(AssistantMessageEvent::TextDelta {
            content_index: idx,
            delta: delta.to_string(),
            partial: cx.message.clone(),
        });
    }

    fn push_thinking(&mut self, cx: &mut Ctx<'_>, delta: &str) {
        if self.open != OpenBlock::Thinking {
            self.close_open(cx);
            cx.message.content.push(ContentBlock::Thinking {
                text: String::new(),
            });
            let idx = cx.current_index();
            self.open = OpenBlock::Thinking;
            cx.emit(AssistantMessageEvent::ThinkingStart {
                content_index: idx,
                partial: cx.message.clone(),
            });
        }
        let idx = cx.current_index();
        if let ContentBlock::Thinking { text } = &mut cx.message.content[idx] {
            text.push_str(delta);
        }
        cx.emit(AssistantMessageEvent::ThinkingDelta {
            content_index: idx,
            delta: delta.to_string(),
            partial: cx.message.clone(),
        });
    }

    /// Record usage from the final chunk's top-level `usageMetadata`:
    /// `promptTokenCount` (input, incl. cache), `candidatesTokenCount`
    /// (output), `cachedContentTokenCount` (cache read). Google reports no
    /// cache-write figure; `total_tokens` is the provider's authoritative
    /// `totalTokenCount` when present, else the component sum.
    fn apply_usage(&self, cx: &mut Ctx<'_>, value: &Value) {
        let Some(metadata) = value.get("usageMetadata") else {
            return;
        };
        let prompt = metadata
            .get("promptTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let cached = metadata
            .get("cachedContentTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let output = metadata
            .get("candidatesTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let mut u = cx.message.usage;
        u.input = prompt.saturating_sub(cached);
        u.output = output;
        u.cache_read = cached;
        u.total_tokens = metadata
            .get("totalTokenCount")
            .and_then(|v| v.as_u64())
            .unwrap_or(u.input + u.output + u.cache_read);
        cx.message.usage = u;
    }

    /// Push a complete Google `functionCall` part: open a tool block, emit
    /// `ToolCallStart`, then close it immediately with `ToolCallEnd` — Google
    /// delivers the full call (name + args) in a single part.
    fn push_tool(&mut self, cx: &mut Ctx<'_>, name: &str, args: &Value) {
        self.close_open(cx);
        let content_index = cx.message.content.len();
        cx.message.content.push(ContentBlock::ToolCall(ToolCall {
            id: String::new(),
            name: name.to_string(),
            arguments: args.clone(),
        }));
        self.open = OpenBlock::Tool(content_index as u32);
        cx.emit(AssistantMessageEvent::ToolCallStart {
            content_index,
            partial: cx.message.clone(),
        });
        self.close_open(cx);
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
            id: "gemini-2.5-flash".into(),
            name: "Gemini 2.5 Flash".into(),
            api: API.into(),
            provider: "google".into(),
            base_url: base.into(),
            reasoning: true,
            input: vec![],
            cost: Default::default(),
            context_window: 1_000_000,
            max_tokens: 8192,
        }
    }

    fn sse_data(data: &str) -> String {
        format!("data: {data}\n\n")
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
        let provider = GoogleGenAiProvider::new().expect("HTTP client");
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
                        "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"hi\"}]}}]}\n\n"
                            .to_string()
                            + "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"!\"}]},\"finishReason\":\"STOP\"}]}\n\n";
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
                stream_generate_url(
                    "https://generativelanguage.googleapis.com/v1beta",
                    "gemini-2.5-flash"
                ),
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
            );
        }

        #[test]
        fn trims_trailing_slash() {
            assert_eq!(
                stream_generate_url(
                    "https://generativelanguage.googleapis.com/v1beta/",
                    "gemini-2.5-flash"
                ),
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
            );
            assert_eq!(
                stream_generate_url(
                    "https://generativelanguage.googleapis.com/v1beta//",
                    "gemini-2.5-flash"
                ),
                "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse"
            );
        }

        #[test]
        fn does_not_duplicate_suffix() {
            let tail = "/models/gemini-2.5-flash:streamGenerateContent";
            // Suffix without the query: append ?alt=sse.
            assert_eq!(
                stream_generate_url(&format!("https://example.com{tail}"), "gemini-2.5-flash"),
                format!("https://example.com{tail}?alt=sse")
            );
            // Full suffix already present: unchanged.
            assert_eq!(
                stream_generate_url(
                    &format!("https://example.com{tail}?alt=sse"),
                    "gemini-2.5-flash"
                ),
                format!("https://example.com{tail}?alt=sse")
            );
        }
    }

    mod request_body {
        use super::*;

        #[test]
        fn includes_all_sections() {
            let model = model("https://generativelanguage.googleapis.com/v1beta");
            let context = LlmContext {
                system_prompt: "be helpful".into(),
                messages: vec![Message::User(UserMessage::new("hello"))],
                tools: vec![Arc::new(EchoTool)],
            };
            let body = build_request_body(&model, &context, &StreamFnOptions::default());
            assert_eq!(body["contents"][0]["role"], "user");
            assert_eq!(body["contents"][0]["parts"][0]["text"], "hello");
            assert_eq!(body["systemInstruction"]["parts"][0]["text"], "be helpful");
            assert_eq!(body["tools"][0]["functionDeclarations"][0]["name"], "echo");
            assert_eq!(
                body["tools"][0]["functionDeclarations"][0]["parameters"]["required"],
                json!(["x"])
            );
            assert_eq!(body["generationConfig"]["candidateCount"], 1);
            assert_eq!(body["stream"], true);
        }

        #[test]
        fn omits_optional_sections() {
            let model = model("http://example");
            let context = LlmContext {
                system_prompt: String::new(),
                messages: vec![],
                tools: vec![],
            };
            let body = build_request_body(&model, &context, &StreamFnOptions::default());
            assert!(body.get("systemInstruction").is_none());
            assert!(body.get("tools").is_none());
            assert_eq!(body["contents"], json!([]));
        }

        #[test]
        fn serializes_assistant_and_tool_msgs() {
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
            // Assistant replay: role "model" with text + functionCall parts.
            assert_eq!(body["contents"][1]["role"], "model");
            assert_eq!(body["contents"][1]["parts"][0]["text"], "calling echo");
            assert_eq!(
                body["contents"][1]["parts"][1]["functionCall"]["name"],
                "echo"
            );
            assert_eq!(
                body["contents"][1]["parts"][1]["functionCall"]["args"]["x"],
                1
            );
            // Tool result: role "function" with functionResponse part.
            assert_eq!(body["contents"][2]["role"], "function");
            assert_eq!(
                body["contents"][2]["parts"][0]["functionResponse"]["name"],
                "echo"
            );
            assert_eq!(
                body["contents"][2]["parts"][0]["functionResponse"]["response"]["content"],
                "pong"
            );
        }
    }

    mod vision {
        use super::*;

        #[test]
        fn image_serialized_as_inline_data() {
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
            let parts = &body["contents"][0]["parts"];
            assert_eq!(parts[0]["text"], "what is this");
            assert_eq!(parts[1]["inlineData"]["mimeType"], "image/png");
            // base64 of [1,2,3] is "AQID"
            assert_eq!(parts[1]["inlineData"]["data"], "AQID");
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
            let parts = &body["contents"][0]["parts"];
            assert_eq!(
                parts,
                &json!([{ "text": "describe" }]),
                "image part must be dropped"
            );
        }
    }

    mod sse {
        use super::*;

        #[test]
        fn text_stream_emits_all_events() {
            let frames = vec![
                sse_data(r#"{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}"#),
                sse_data(
                    r#"{"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}]}"#,
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
                events.iter().any(|e| matches!(
                    e,
                    AssistantMessageEvent::TextDelta { delta, .. } if delta == "lo"
                )),
                "expected TextDelta \"lo\", events: {events:?}",
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
            let frames = vec![sse_data(
                r#"{"candidates":[{"content":{"parts":[{"functionCall":{"name":"echo","args":{"x":1}}}]},"finishReason":"STOP"}]}"#,
            )];
            let events = collect_events(&frames);
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
                .expect("expected ToolCallEnd");
            assert_eq!(end.name, "echo");
            assert_eq!(end.arguments, json!({"x": 1}));
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
                sse_data(
                    r#"{"candidates":[{"content":{"parts":[{"thought":true,"text":"hmm"}]}}]}"#,
                ),
                sse_data(r#"{"candidates":[{"content":{"parts":[{"text":"done"}]}}]}"#),
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
            // Text after thinking.
            assert!(
                events.iter().any(|e| matches!(
                    e,
                    AssistantMessageEvent::TextDelta { delta, .. } if delta == "done"
                )),
                "expected TextDelta \"done\", events: {events:?}",
            );
        }

        #[test]
        fn max_tokens_maps_to_length() {
            let frames = vec![sse_data(
                r#"{"candidates":[{"content":{"parts":[{"text":"..."}]},"finishReason":"MAX_TOKENS"}]}"#,
            )];
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
            let frames = vec![sse_data(r#"{"error":{"message":"bad"}}"#)];
            let events = collect_events(&frames);
            assert!(
                matches!(
                    events.last(),
                    Some(AssistantMessageEvent::Error {
                        reason: StopReason::Error,
                        ..
                    }),
                ),
                "error response should emit Error, got {events:?}",
            );
        }

        #[test]
        fn usage_metadata_parsed_from_final_chunk() {
            let frames = vec![
                sse_data(r#"{"candidates":[{"content":{"parts":[{"text":"Hel"}]}}]}"#),
                sse_data(
                    r#"{"candidates":[{"content":{"parts":[{"text":"lo"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":20,"totalTokenCount":130,"cachedContentTokenCount":30}}"#,
                ),
            ];
            let events = collect_events(&frames);
            let done = events
                .iter()
                .find_map(|e| match e {
                    AssistantMessageEvent::Done { message, .. } => Some(message.clone()),
                    _ => None,
                })
                .expect("expected a Done event");
            assert_eq!(
                done.usage,
                pi_agent_core::types::Usage {
                    input: 70,
                    output: 20,
                    cache_read: 30,
                    cache_write: 0,
                    total_tokens: 130,
                    cost: Default::default(),
                }
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
                api_key: Some("google-key".into()),
                ..Default::default()
            };
            let (_tx, abort) = watch::channel(false);
            let _ = collect(addr, context, options, abort).await;
            let raw = rx.await.unwrap();
            assert_eq!(
                raw.lines().next().unwrap_or(""),
                "POST /models/gemini-2.5-flash:streamGenerateContent?alt=sse HTTP/1.1",
                "{raw}"
            );
            assert!(
                raw.to_ascii_lowercase()
                    .contains("x-goog-api-key: google-key"),
                "{raw}"
            );
            assert!(
                raw.to_ascii_lowercase().contains("user-agent: aaos"),
                "{raw}"
            );
            let body = raw.split("\r\n\r\n").nth(1).unwrap_or("");
            let json: Value = serde_json::from_str(body).unwrap();
            assert_eq!(json["contents"][0]["role"], "user");
            assert_eq!(json["contents"][0]["parts"][0]["text"], "hello");
            assert_eq!(json["systemInstruction"]["parts"][0]["text"], "sys");
            assert_eq!(json["tools"][0]["functionDeclarations"][0]["name"], "echo");
            assert_eq!(json["generationConfig"]["candidateCount"], 1);
            assert_eq!(json["stream"], true);
        }

        #[tokio::test]
        async fn text_events_are_emitted() {
            let body = [
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"Hel\"}]}}]}\n\n",
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"lo\"}]},\"finishReason\":\"STOP\"}]}\n\n",
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
        async fn abort_cancels_body() {
            let body = [
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"aaaa\"}]}}]}\n\n",
                "data: {\"candidates\":[{\"content\":{\"parts\":[{\"text\":\"bbbb\"}]}}]}\n\n",
            ]
            .concat();
            let (addr, h) = serve(200, body, 80).await;
            let (tx, abort) = watch::channel(false);
            let (seen_delta_tx, seen_delta_rx) = tokio::sync::oneshot::channel::<()>();
            let provider = GoogleGenAiProvider::new().expect("HTTP client");
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
