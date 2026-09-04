//! Shared SSE streaming engine for the wire-format adapters.
//!
//! Every adapter streams the same mechanical shape: POST a JSON body with
//! format-specific authentication headers, split `\n\n`-separated SSE
//! frames off the HTTP body, decode each frame's data payload, and feed it
//! to a per-format [`SseFormat`] interpreter that translates provider
//! chunks into [`AssistantMessageEvent`]s. This module owns that machinery
//! once — [`run_stream`], the [`EventBuilder`] shell, and the terminal
//! lifecycle (`finish_reason`/`close_done`/`abort`/`error`). Adapters
//! contribute only their authentication headers and an [`SseFormat`]
//! implementation with three hooks: `apply_chunk`, `close_open`, and
//! `stop_reason`.

use std::error::Error as StdError;

use async_trait::async_trait;
use pi_agent_core::types::{
    AssistantEventStream, AssistantMessage, AssistantMessageEvent, ContentBlock, Model, ModelInput,
    StopReason,
};
use reqwest::Client;
use serde_json::{Value, json};
use tokio::sync::{mpsc, watch};

/// Park until the abort flag flips; returns immediately if already set.
pub(super) async fn wait_aborted(abort: &mut watch::Receiver<bool>) {
    if *abort.borrow() {
        return;
    }
    let _ = abort.wait_for(|v| *v).await;
}

/// Flatten an error's source chain into one message.
///
/// reqwest's Display omits the source chain, so "error sending request for url"
/// hides timeouts and TLS failures unless we walk `.source()`.
pub(super) fn error_chain(err: &dyn StdError) -> String {
    let mut msg = err.to_string();
    let mut source = err.source();
    while let Some(inner) = source {
        msg.push_str(": ");
        msg.push_str(&inner.to_string());
        source = inner.source();
    }
    msg
}

/// Extract text from content blocks, joining all `Text` blocks.
pub(super) fn content_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Whether the model accepts image input (spec §6).
pub(super) fn supports_images(model: &Model) -> bool {
    model.input.contains(&ModelInput::Image)
}

/// Parse one SSE frame (the text between two `\n\n` separators) into its
/// optional `event:` name and accumulated `data:` payload.
///
/// Comment (`:`) and empty lines are ignored; any other unrecognized line
/// is a protocol error. Named-event formats (Anthropic, Cohere) consume the
/// event name; delta-driven formats ignore it — both previously tolerated
/// `event:` lines, so unified parsing is behavior-preserving.
pub(super) fn parse_sse_frame(frame: &str) -> Result<(Option<String>, String), String> {
    let mut event: Option<String> = None;
    let mut data = String::new();
    for line in frame.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            event = Some(rest.trim().to_string());
        } else if let Some(rest) = line.strip_prefix("data:") {
            let rest = rest.trim_start();
            if !data.is_empty() {
                data.push('\n');
            }
            data.push_str(rest);
        } else if !line.is_empty() && !line.starts_with(':') {
            return Err(format!("malformed SSE line: {line}"));
        }
    }
    Ok((event, data))
}

/// Extract a provider-level `{"error": {"message": …}}` payload, if present.
pub(super) fn provider_error(value: &Value) -> Option<String> {
    Some(
        value
            .get("error")?
            .get("message")
            .and_then(|m| m.as_str())
            .unwrap_or("provider error")
            .to_string(),
    )
}

/// Parse accumulated tool-call argument JSON, falling back to `{}`.
pub(super) fn parse_tool_args(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| json!({}))
}

/// Receiver half of the engine: yields events until terminal, tracking the
/// final assistant message for [`AssistantEventStream::result`].
pub(super) struct SseStream {
    rx: mpsc::UnboundedReceiver<AssistantMessageEvent>,
    final_message: AssistantMessage,
}

impl SseStream {
    /// Box the stream end returned from `StreamFn::call`.
    pub(super) fn boxed(
        rx: mpsc::UnboundedReceiver<AssistantMessageEvent>,
        final_message: AssistantMessage,
    ) -> Box<dyn AssistantEventStream> {
        Box::new(Self { rx, final_message })
    }
}

#[async_trait]
impl AssistantEventStream for SseStream {
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

/// Engine-owned state handed to [`SseFormat`] hooks.
///
/// Splitting these fields out lets format code drive message/event state
/// while the engine keeps the builder shell — disjoint borrows instead of
/// self-referential callbacks.
pub(super) struct Ctx<'a> {
    /// Formats append content blocks directly; block layout is theirs.
    pub(super) message: &'a mut AssistantMessage,
    tx: &'a mpsc::UnboundedSender<AssistantMessageEvent>,
    started: &'a mut bool,
    pub(super) finished: &'a mut bool,
}

impl Ctx<'_> {
    pub(super) fn emit(&mut self, event: AssistantMessageEvent) {
        let _ = self.tx.send(event);
    }

    pub(super) fn ensure_start(&mut self) {
        if !*self.started {
            *self.started = true;
            self.emit(AssistantMessageEvent::Start {
                partial: self.message.clone(),
            });
        }
    }

    /// Index of the last content block (`usize::MAX`-safe on empty).
    pub(super) fn current_index(&self) -> usize {
        self.message.content.len().saturating_sub(1)
    }
}

/// Per-format interpretation of one provider's SSE stream.
///
/// The engine owns frame parsing ([`EventBuilder::push_sse`]) and the
/// terminal lifecycle; a format supplies three hooks:
///
/// - [`apply_chunk`](SseFormat::apply_chunk): interpret one decoded payload
/// - [`close_open`](SseFormat::close_open): tear down the open block
/// - [`stop_reason`](SseFormat::stop_reason): translate the provider's
///   finish-reason vocabulary onto [`StopReason`]
pub(super) trait SseFormat: Default {
    /// Interpret one decoded provider payload.
    ///
    /// `event` carries the frame's `event:` name when the stream uses named
    /// events (Anthropic, Cohere); delta-driven formats ignore it.
    fn apply_chunk(&mut self, cx: &mut Ctx<'_>, event: Option<&str>, value: &Value);

    /// Emit the End event for whichever block is open, resetting to none.
    fn close_open(&mut self, cx: &mut Ctx<'_>);

    /// Map a provider finish-reason token onto [`StopReason`].
    fn stop_reason(&self, reason: &str) -> StopReason;

    /// Flush the open block and emit `Done` carrying the mapped reason.
    fn finish_reason(&mut self, cx: &mut Ctx<'_>, reason: &str) {
        self.close_open(cx);
        cx.message.stop_reason = self.stop_reason(reason);
        cx.emit(AssistantMessageEvent::Done {
            reason: cx.message.stop_reason,
            message: cx.message.clone(),
        });
        *cx.finished = true;
    }

    /// Synthetic successful termination when the body ends mid-stream.
    fn close_done(&mut self, cx: &mut Ctx<'_>) {
        if *cx.finished {
            return;
        }
        finish_done(self, cx);
    }

    /// Abort: flush the open block and mark the message aborted.
    fn abort(&mut self, cx: &mut Ctx<'_>) {
        if *cx.finished {
            return;
        }
        cx.ensure_start();
        self.close_open(cx);
        cx.message.stop_reason = StopReason::Aborted;
        cx.message.error_message = Some("Aborted".into());
        cx.emit(AssistantMessageEvent::Error {
            reason: StopReason::Aborted,
            error: cx.message.clone(),
        });
        *cx.finished = true;
    }

    /// Terminal failure carrying `msg`; ignored once finished.
    fn error(&mut self, cx: &mut Ctx<'_>, msg: String) {
        if *cx.finished {
            return;
        }
        cx.ensure_start();
        self.close_open(cx);
        cx.message.stop_reason = StopReason::Error;
        cx.message.error_message = Some(msg);
        cx.emit(AssistantMessageEvent::Error {
            reason: StopReason::Error,
            error: cx.message.clone(),
        });
        *cx.finished = true;
    }
}

/// Shared synthetic-termination tail: flush the open block, promote a
/// `Pending` stop reason to `Stop`, and emit the terminal `Done`.
///
/// Standalone (not a trait method) so an adapter that overrides
/// [`SseFormat::close_done`] can reuse the fall-through tail; calling the
/// trait default from an override would virtual-dispatch back into the
/// override.
pub(super) fn finish_done<F: SseFormat>(format: &mut F, cx: &mut Ctx<'_>) {
    cx.ensure_start();
    format.close_open(cx);
    if cx.message.stop_reason == StopReason::Pending {
        cx.message.stop_reason = StopReason::Stop;
    }
    cx.emit(AssistantMessageEvent::Done {
        reason: cx.message.stop_reason,
        message: cx.message.clone(),
    });
    *cx.finished = true;
}

/// Engine shell around a format: parses SSE frames and drives the shared
/// terminal lifecycle.
///
/// Adapters alias this with their format — `type EventBuilder =
/// sse::EventBuilder<MyFormat>` — so unit tests drive `push_sse`,
/// `finished`, and `close_done` exactly as before the extraction.
pub(super) struct EventBuilder<F: SseFormat> {
    tx: mpsc::UnboundedSender<AssistantMessageEvent>,
    message: AssistantMessage,
    started: bool,
    /// Set once a terminal Done/Error was emitted.
    pub(crate) finished: bool,
    format: F,
}

/// Assemble the hook context from disjoint engine fields; passing the
/// fields individually keeps the format borrow separate from the state
/// borrows at each call site.
fn ctx_of<'a>(
    tx: &'a mpsc::UnboundedSender<AssistantMessageEvent>,
    message: &'a mut AssistantMessage,
    started: &'a mut bool,
    finished: &'a mut bool,
) -> Ctx<'a> {
    Ctx {
        tx,
        message,
        started,
        finished,
    }
}

impl<F: SseFormat> EventBuilder<F> {
    pub(super) fn new(model: &Model, tx: mpsc::UnboundedSender<AssistantMessageEvent>) -> Self {
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
            finished: false,
            format: F::default(),
        }
    }

    /// Parse and dispatch one SSE frame.
    ///
    /// Ordering mirrors the original readers: the OpenAI `[DONE]` sentinel
    /// is checked before JSON parsing (harmless no-op for the other
    /// formats, which never send it); provider-level `error` payloads
    /// become terminal errors; everything else goes to the format hook.
    pub(super) fn push_sse(&mut self, frame: &str) -> Result<(), String> {
        let (event, data) = parse_sse_frame(frame)?;
        if data.is_empty() {
            return Ok(());
        }
        if data.trim() == "[DONE]" {
            self.close_done();
            return Ok(());
        }
        let value: Value =
            serde_json::from_str(&data).map_err(|e| format!("malformed SSE JSON: {e}"))?;
        if let Some(msg) = provider_error(&value) {
            self.error(msg);
            return Ok(());
        }
        let mut cx = ctx_of(
            &self.tx,
            &mut self.message,
            &mut self.started,
            &mut self.finished,
        );
        self.format.apply_chunk(&mut cx, event.as_deref(), &value);
        Ok(())
    }

    pub(super) fn close_done(&mut self) {
        let mut cx = ctx_of(
            &self.tx,
            &mut self.message,
            &mut self.started,
            &mut self.finished,
        );
        self.format.close_done(&mut cx);
    }

    pub(super) fn abort(&mut self) {
        let mut cx = ctx_of(
            &self.tx,
            &mut self.message,
            &mut self.started,
            &mut self.finished,
        );
        self.format.abort(&mut cx);
    }

    pub(super) fn error(&mut self, msg: String) {
        let mut cx = ctx_of(
            &self.tx,
            &mut self.message,
            &mut self.started,
            &mut self.finished,
        );
        self.format.error(&mut cx, msg);
    }
}

/// Drive one provider SSE response end-to-end: abort-aware request send,
/// non-2xx handling, `\n\n` frame splitting over [`reqwest::Response::chunk`]
/// reads, and terminal lifecycle.
///
/// `headers` carries the adapter's authentication headers in wire order;
/// the engine appends `Content-Type` after them.
pub(super) async fn run_stream<F: SseFormat>(
    client: Client,
    url: String,
    model: Model,
    body: Value,
    mut abort: watch::Receiver<bool>,
    tx: mpsc::UnboundedSender<AssistantMessageEvent>,
    headers: impl IntoIterator<Item = (&'static str, String)>,
) {
    let mut builder: EventBuilder<F> = EventBuilder::new(&model, tx);
    if *abort.borrow() {
        builder.abort();
        return;
    }

    let request = headers
        .into_iter()
        .fold(client.post(&url), |request, (name, value)| {
            request.header(name, value)
        })
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

    let mut response = response;
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
            next = response.chunk() => next,
        };
        match chunk {
            Ok(Some(bytes)) => {
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
            Err(e) => {
                if *abort.borrow() {
                    builder.abort();
                } else {
                    builder.error(error_chain(&e));
                }
                return;
            }
            // Body ended mid-stream: synthesize a successful termination.
            Ok(None) => {
                if !builder.finished {
                    builder.close_done();
                }
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;

    /// Minimal format: records the forwarded event name and payload count,
    /// starting the stream so lifecycle events are observable.
    #[derive(Default)]
    struct ProbeFormat {
        chunks: usize,
        last_event: Option<String>,
    }

    impl SseFormat for ProbeFormat {
        fn apply_chunk(&mut self, cx: &mut Ctx<'_>, event: Option<&str>, value: &Value) {
            assert_eq!(value["n"], 1);
            self.last_event = event.map(String::from);
            cx.ensure_start();
            self.chunks += 1;
        }

        fn close_open(&mut self, _cx: &mut Ctx<'_>) {}

        fn stop_reason(&self, reason: &str) -> StopReason {
            match reason {
                "max_tokens" => StopReason::Length,
                _ => StopReason::Stop,
            }
        }
    }

    fn builder() -> (
        EventBuilder<ProbeFormat>,
        tokio::sync::mpsc::UnboundedReceiver<AssistantMessageEvent>,
    ) {
        let (tx, rx) = mpsc::unbounded_channel();
        (EventBuilder::new(&Model::unknown(), tx), rx)
    }

    fn drain(rx: &mut tokio::sync::mpsc::UnboundedReceiver<AssistantMessageEvent>) -> Vec<String> {
        let mut names = Vec::new();
        while let Ok(ev) = rx.try_recv() {
            names.push(match ev {
                AssistantMessageEvent::Start { .. } => "Start".into(),
                AssistantMessageEvent::TextDelta { .. } => "TextDelta".into(),
                AssistantMessageEvent::Done { .. } => "Done".into(),
                AssistantMessageEvent::Error { .. } => "Error".into(),
                _ => "Other".into(),
            });
        }
        names
    }

    #[test]
    fn frame_parses_event_and_data_lines() {
        let (mut b, mut rx) = builder();
        b.push_sse("event: probe\ndata: {\"n\":1}\n\n")
            .expect("frame parses");
        assert_eq!(b.format.last_event.as_deref(), Some("probe"));
        assert_eq!(drain(&mut rx), vec!["Start"]);
        assert!(!b.finished);
    }

    #[test]
    fn multi_data_lines_join_with_newline() {
        let (mut b, mut rx) = builder();
        // `{"n":` + newline + `1}` reassembles to valid JSON — the SSE
        // data-field newline rule.
        b.push_sse("data: {\"n\":\ndata: 1}").expect("frame parses");
        assert_eq!(
            b.format.last_event, None,
            "data-only frames carry no event name"
        );
        assert_eq!(b.format.chunks, 1);
        assert_eq!(drain(&mut rx), vec!["Start"]);
    }

    #[test]
    fn comment_and_blank_lines_are_ignored() {
        let (mut b, mut rx) = builder();
        b.push_sse(": keepalive\n\n")
            .expect("comment-only frame parses");
        assert_eq!(b.format.chunks, 0);
        assert!(!b.finished);
        assert!(drain(&mut rx).is_empty());
    }

    #[test]
    fn malformed_line_is_an_error() {
        let (mut b, _rx) = builder();
        let err = b.push_sse("data: {\"n\":1}\njunk here").unwrap_err();
        assert!(err.contains("malformed SSE line: junk here"), "{err}");
    }

    #[test]
    fn done_sentinel_closes_before_json_parse() {
        let (mut b, mut rx) = builder();
        b.push_sse("data: [DONE]").expect("[DONE] handled");
        assert!(b.finished, "[DONE] must finish the stream");
        // `[DONE]` is not valid JSON; reaching the parser would have errored.
        assert_eq!(drain(&mut rx), vec!["Start", "Done"]);
    }

    #[test]
    fn provider_error_payload_becomes_terminal() {
        let (mut b, mut rx) = builder();
        b.push_sse("data: {\"error\":{\"message\":\"boom\"}}")
            .expect("frame parses");
        assert!(b.finished);
        assert_eq!(drain(&mut rx), vec!["Start", "Error"]);
    }

    #[test]
    fn close_done_is_idempotent_and_fills_stop_reason() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut message = AssistantMessage::default();
        let mut started = false;
        let mut finished = false;
        {
            let mut state = Ctx {
                tx: &tx,
                message: &mut message,
                started: &mut started,
                finished: &mut finished,
            };
            let mut fmt = ProbeFormat::default();
            fmt.close_done(&mut state);
            fmt.close_done(&mut state);
        }
        assert!(finished);
        assert_eq!(message.stop_reason, StopReason::Stop);
        assert_eq!(drain(&mut rx), vec!["Start", "Done"],);
    }

    #[test]
    fn terminal_defaults_guard_on_finished() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut message = AssistantMessage::default();
        let mut started = false;
        let mut finished = false;
        let mut fmt = ProbeFormat::default();
        {
            let mut state = Ctx {
                tx: &tx,
                message: &mut message,
                started: &mut started,
                finished: &mut finished,
            };
            fmt.finish_reason(&mut state, "max_tokens");
            fmt.abort(&mut state);
            fmt.error(&mut state, "late".into());
        }
        assert_eq!(
            message.stop_reason,
            StopReason::Length,
            "post-terminal calls must not overwrite the reason"
        );
        assert!(finished);
        assert_eq!(
            drain(&mut rx),
            vec!["Done"],
            "abort/error after Done must emit nothing"
        );
    }

    #[test]
    fn abort_and_error_default_lifecycles() {
        assert_eq!(
            run_terminal(StopReason::Aborted, ProbeFormat::abort),
            "abort"
        );
        assert_eq!(
            run_terminal(StopReason::Error, |f: &mut ProbeFormat, s: &mut Ctx<'_>| {
                f.error(s, "x".into());
            }),
            "error"
        );
    }

    /// Drive one terminal default against a fresh builder; may panic on
    /// wrong state, returns the terminal class for the assertion above.
    fn run_terminal(
        expected: StopReason,
        trigger: fn(&mut ProbeFormat, &mut Ctx<'_>),
    ) -> &'static str {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut message = AssistantMessage::default();
        let mut started = false;
        let mut finished = false;
        {
            let mut state = Ctx {
                tx: &tx,
                message: &mut message,
                started: &mut started,
                finished: &mut finished,
            };
            let mut fmt = ProbeFormat::default();
            trigger(&mut fmt, &mut state);
        }
        assert!(finished);
        assert_eq!(message.stop_reason, expected);
        assert_eq!(drain(&mut rx).len(), 2, "Start + one terminal event");
        if expected == StopReason::Aborted {
            "abort"
        } else {
            "error"
        }
    }
}
