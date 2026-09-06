//! Compaction coordinator: orchestrates conversation compaction for a
//! session node — cut-point selection, deterministic transcript building,
//! projection validation, the `store.compact` commit, and the auto-trigger
//! hooks (`transform_context` + `prepare_next_turn`).
//!
//! Design notes
//! ------------
//! - The coordinator is **session-agnostic**: `compact()` takes a node id and
//!   returns the new compacted id plus the injected view; it never touches the
//!   agent. Callers decide whether/how to switch the session's append target
//!   and resync `state.messages`.
//! - Compaction is **deterministic**: the compacted `Segment::Summary`
//!   content is a transcript of the compacted range (see
//!   `aaos_session::compaction::build_transcript`). No LLM call happens; the
//!   coordinator retains only the live model's `context_window` for the
//!   trigger checks — it never re-resolves or calls a model.
//! - Provenance is **structural only** (ADR-0006): the summary segment
//!   carries no `sources`; originals are retrievable through
//!   `SessionStore::fetch_originals` on the compacted node, and the
//!   transcript's object paths are recomputed from block bytes.
//! - The hook-facing methods (`pre_request_hook` for `transform_context`,
//!   `post_turn_hook` for `prepare_next_turn`) return the injected view on
//!   success and `None` on refusal/failure (stderr already surfaced) so the
//!   hooks are non-blocking: the caller keeps the original context.
//! - Per-run state guards: `compacted_this_run` (at most one auto-compaction
//!   per run), `overflow_retry_attempted` (at most one overflow recovery per
//!   run — a second overflow fails the run), `pending_resync` (the caller
//!   resyncs the in-memory view after the run via `take_pending_resync`).
//!   All call sites are awaited serially in one task (hooks inside the agent
//!   loop, `/compact` between REPL prompts), so no re-entrancy guard is
//!   needed.

use std::sync::{Arc, Mutex};

use aaos_session::compaction::{
    DEFAULT_KEEP_RECENT_TOKENS, DEFAULT_RESERVE_TOKENS, build_transcript, context_tokens,
    find_cut_point, should_compact, view_messages,
};
use aaos_session::{Segment, SessionStore, SummarySegment};
use pi_agent_core::agent::Agent;
use pi_agent_core::types::{
    AgentContext, AgentLoopTurnUpdate, AssistantMessage, Message, Model, StopReason, Usage,
    UserMessage,
};

/// Compaction settings, constructed explicitly by the caller. Manual
/// `/compact` ignores `enabled`; the auto hooks check it.
#[derive(Debug, Clone, Copy)]
pub struct CompactionSettings {
    /// Whether the auto hooks may trigger compaction.
    pub enabled: bool,
    /// Token budget reserved for the model's output window.
    pub reserve_tokens: u64,
    /// Token budget for the retained recent tail.
    pub keep_recent_tokens: u64,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: DEFAULT_RESERVE_TOKENS,
            keep_recent_tokens: DEFAULT_KEEP_RECENT_TOKENS,
        }
    }
}

impl CompactionSettings {
    /// Pure parse of the three env values (as `Option<&str>`, `None` = unset)
    /// onto a [`CompactionSettings`]. Reading the environment belongs to the
    /// frontend; this stays pure so it is testable without mutating process
    /// env — the workspace denies `unsafe_code`, so tests cannot set env
    /// vars.
    pub fn from_env_values(
        enabled: Option<&str>,
        reserve_tokens: Option<&str>,
        keep_recent_tokens: Option<&str>,
    ) -> Self {
        let enabled = match enabled {
            Some(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no"),
            None => true,
        };
        let reserve_tokens = reserve_tokens
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_RESERVE_TOKENS);
        let keep_recent_tokens = keep_recent_tokens
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_KEEP_RECENT_TOKENS);
        Self {
            enabled,
            reserve_tokens,
            keep_recent_tokens,
        }
    }
}

/// The result of a successful compaction.
#[derive(Debug)]
pub struct CompactionOutcome {
    /// The id of the new compacted node.
    pub compacted_id: String,
    /// Estimated context tokens before compaction.
    pub before_tokens: u64,
    /// Estimated context tokens after compaction (summary + retained tail).
    pub after_tokens: u64,
    /// The messages the live agent should run with: transcript user
    /// message + the retained tail (store view + in-memory lagging tail).
    pub injected_view: Vec<Message>,
}

/// Refusals and failures, rendered exactly to stderr by the caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionError {
    /// The conversation is too short to compact.
    NothingToCompact,
    /// The compaction or commit failed.
    Failed(String),
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompactionError::NothingToCompact => write!(f, "Nothing to compact"),
            CompactionError::Failed(reason) => write!(f, "Compaction failed: {reason}"),
        }
    }
}

impl std::error::Error for CompactionError {}

/// Per-run state, mutated only by `begin_run` and the hooks. All call sites
/// are awaited serially in one task, so the `Mutex` merely guards brief flag
/// updates and needs no re-entrancy protection.
#[derive(Default)]
struct CoordinatorState {
    /// A compaction committed this run (auto or manual) — at most one
    /// auto-attempt per run.
    compacted_this_run: bool,
    /// Overflow recovery already attempted this run — at most one re-prompt.
    overflow_retry_attempted: bool,
    /// Pending node switch for the caller: `Some(id)` when a compaction
    /// committed but the session has not been resynced yet.
    pending_resync: Option<String>,
}

/// Compaction coordinator for one [`SessionStore`].
pub struct CompactionCoordinator {
    store: SessionStore,
    settings: CompactionSettings,
    /// Live model's context window, used by the auto-trigger checks. Only
    /// the window is retained — the coordinator never re-resolves a model.
    context_window: u64,
    state: Mutex<CoordinatorState>,
}

impl CompactionCoordinator {
    /// Build a coordinator. `model` is the live model; only its
    /// `context_window` is kept, for the auto-trigger checks.
    pub fn new(store: SessionStore, settings: CompactionSettings, model: &Model) -> Self {
        Self {
            store,
            settings,
            context_window: model.context_window,
            state: Mutex::new(CoordinatorState::default()),
        }
    }

    /// Reset per-run flags; call before each prompt.
    pub fn begin_run(&self) {
        let mut state = lock_state(&self.state);
        state.compacted_this_run = false;
        state.overflow_retry_attempted = false;
        state.pending_resync = None;
    }

    /// Take the pending resync target, if a compaction committed and the
    /// caller has not resynced yet. The caller must `session.resume(&id)`.
    pub fn take_pending_resync(&self) -> Option<String> {
        let mut state = lock_state(&self.state);
        state.pending_resync.take()
    }

    /// Pre-request (transform_context) check: when auto-compaction is enabled
    /// and the outgoing context exceeds the window threshold, compact the
    /// current node. On success returns the injected view — the transcript
    /// message plus the retained tail — for the caller to use as this
    /// request's messages; on refusal/failure prints to stderr and returns
    /// `None` so the caller keeps the original messages — non-blocking.
    pub async fn pre_request_hook(
        &self,
        messages: &[Message],
        session_id: &str,
    ) -> Option<CompactionOutcome> {
        if lock_state(&self.state).compacted_this_run {
            return None;
        }
        if !self.settings.enabled {
            return None;
        }
        if !should_compact(
            context_tokens(messages),
            self.context_window,
            self.settings.reserve_tokens,
        ) {
            return None;
        }
        match self.compact(session_id).await {
            Ok(outcome) => Some(outcome),
            Err(err) => {
                #[allow(clippy::print_stderr)]
                {
                    eprintln!("compaction failed: {err}");
                }
                None
            }
        }
    }

    /// Turn-end (prepare_next_turn) check:
    /// - (a) overflow branch — the just-finished message hit the context
    ///   window (an overflow-class error message, or silent overflow via
    ///   usage.input + usage.cache_read > context_window). Recovery compacts
    ///   once per run; a second overflow returns `Err` to fail the run.
    /// - (b) threshold branch — a normal message with the context still above
    ///   the window threshold and no compaction yet this run.
    ///
    /// On success returns `Some(outcome)` whose `injected_view` replaces the
    /// context for subsequent turns; on refusal/failure `Ok(None)` (stderr
    /// already surfaced); a hard overflow-after-recovery returns `Err`.
    ///
    /// Overflow recovery is a rescue path, not threshold auto-compaction:
    /// the overflow branch runs even when `settings.enabled` is false.
    pub async fn post_turn_hook(
        &self,
        assistant: &AssistantMessage,
        context: &AgentContext,
        session_id: &str,
    ) -> Result<Option<CompactionOutcome>, String> {
        let compacted_this_run = lock_state(&self.state).compacted_this_run;

        // Overflow recovery (a) is a rescue path, not threshold
        // auto-compaction: it ignores `settings.enabled`.
        let overflow = (assistant.stop_reason == StopReason::Error
            && assistant
                .error_message
                .as_deref()
                .is_some_and(is_overflow_message))
            || is_silent_overflow(self.context_window, &assistant.usage);
        let threshold = self.settings.enabled
            && !compacted_this_run
            && should_compact(
                context_tokens(&context.messages),
                self.context_window,
                self.settings.reserve_tokens,
            );
        if !overflow && !threshold {
            return Ok(None);
        }

        if overflow {
            let state = lock_state(&self.state);
            if state.overflow_retry_attempted {
                return Err("context overflow persists after compaction".to_string());
            }
        }

        // Compact once for either trigger; log a failure and keep the
        // original context (non-blocking).
        if overflow {
            // The recovery attempt counts even when compaction itself fails:
            // a second overflow in the same run still fails the run.
            lock_state(&self.state).overflow_retry_attempted = true;
        }
        let outcome = match self.compact(session_id).await {
            Ok(outcome) => outcome,
            Err(err) => {
                #[allow(clippy::print_stderr)]
                {
                    eprintln!("compaction failed: {err}");
                }
                return Ok(None);
            }
        };

        // Issue #70 §3.5: on overflow recovery the failed/truncated assistant
        // message is excluded from the retry context while remaining in
        // session history (the segment stays persisted in the store).
        // Silent overflow (a completed message) is not stripped.
        if overflow {
            Ok(Some(strip_failed_assistant(outcome)))
        } else {
            Ok(Some(outcome))
        }
    }

    /// Compact `session_id`'s view into a new node. Manual `/compact` and the
    /// hooks both call this.
    ///
    /// `NothingToCompact` returns `Err` without touching the store; a missing
    /// session surfaces as `Failed` from the store's materialize. On success,
    /// returns the outcome; the caller is responsible for switching the
    /// session to `compacted_id` and resyncing the in-memory view.
    pub async fn compact(&self, session_id: &str) -> Result<CompactionOutcome, CompactionError> {
        let segments = self
            .store
            .materialize_plain(session_id)
            .await
            .map_err(|e| CompactionError::Failed(e.to_string()))?;
        if segments.is_empty() {
            return Err(CompactionError::NothingToCompact);
        }

        // Message view: summary segments become bare user messages (the
        // shared `view_messages` conversion — the resume path renders
        // Summary identically).
        let messages = view_messages(&segments);

        let first_kept = match find_cut_point(&messages, self.settings.keep_recent_tokens) {
            Some(i) => i,
            None => return Err(CompactionError::NothingToCompact),
        };

        // Deterministic transcript of the compacted prefix. Provenance is
        // structural (ADR-0006): the summary carries no sources; the
        // transcript's block-level object paths are recomputed from the
        // segments' own bytes and resolve in the store.
        let before_tokens = context_tokens(&messages);
        let transcript = build_transcript(&segments[..first_kept], self.store.objects())
            .map_err(|e| CompactionError::Failed(e.to_string()))?;

        // Projection validation: the compacted view must be strictly smaller.
        // The projected view contains a summary by construction, so its
        // assistant usage is zeroed (in-memory only): a pre-compaction
        // `usage.total_tokens` surviving in the retained tail is a stale
        // anchor that would re-anchor `after_tokens` at ≈ `before_tokens`
        // and refuse every compaction. Zeroed usage falls back to pure
        // estimation — and the zeroed view is what `injected_view` returns.
        let mut projected = vec![Message::User(UserMessage::new(transcript.clone()))];
        projected.extend_from_slice(&messages[first_kept..]);
        for message in &mut projected {
            if let Message::Assistant(assistant) = message {
                assistant.usage = Default::default();
            }
        }
        let after_tokens = context_tokens(&projected);

        if after_tokens >= before_tokens {
            return Err(CompactionError::Failed(
                "compaction would not reduce context".into(),
            ));
        }

        // Commit: one map row covering everything before the cut.
        let summary = Segment::Summary(SummarySegment::new(transcript.clone()));
        let compacted_id = self
            .store
            .compact(session_id, &[(0, first_kept as u64)], &summary)
            .await
            .map_err(|e| CompactionError::Failed(e.to_string()))?;

        {
            let mut state = lock_state(&self.state);
            state.compacted_this_run = true;
            state.pending_resync = Some(compacted_id.clone());
        }

        Ok(CompactionOutcome {
            compacted_id,
            before_tokens,
            after_tokens,
            injected_view: projected,
        })
    }
}

/// Overflow recovery (issue #70 §3.5): the failed/truncated assistant
/// message is excluded from the retry context while staying in session
/// history — strip a trailing Error assistant from the injected view (the
/// persisted segment stays in the store).
fn strip_failed_assistant(outcome: CompactionOutcome) -> CompactionOutcome {
    let mut injected_view = outcome.injected_view;
    if matches!(
        injected_view.last(),
        Some(Message::Assistant(a)) if a.stop_reason == StopReason::Error
    ) {
        injected_view.pop();
    }
    CompactionOutcome {
        injected_view,
        ..outcome
    }
}

/// Whether an error message describes a context-window overflow. Rate-limit /
/// 429 / throttling messages are explicitly excluded — those are transient
/// and belong to the retry path, not compaction recovery.
fn is_overflow_message(msg: &str) -> bool {
    let lower = msg.to_ascii_lowercase();
    if lower.contains("rate") || lower.contains("429") || lower.contains("throttl") {
        return false;
    }
    lower.contains("prompt is too long")
        || lower.contains("context window")
        || lower.contains("maximum prompt length")
        || lower.contains("context length exceeded")
}

/// Silent overflow: the turn consumed more input+cache tokens than the
/// context window without an error message. A zero window (unknown model)
/// means no check.
fn is_silent_overflow(context_window: u64, usage: &Usage) -> bool {
    context_window > 0 && usage.input.saturating_add(usage.cache_read) > context_window
}

/// Lock the coordinator state, recovering from poison like the agent crate
/// does (a poisoned mutex is a bug, not a reason to crash the session).
fn lock_state(state: &Mutex<CoordinatorState>) -> std::sync::MutexGuard<'_, CoordinatorState> {
    match state.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Install the auto-trigger compaction hooks on `agent`:
/// - `transform_context` compacts pre-request when the outgoing context
///   exceeds the window; on success the injected view replaces the messages
///   for this request and the append target switches to the compacted node.
/// - `prepare_next_turn` compacts post-turn on threshold overshoot or
///   context overflow; on success the injected view replaces the context for
///   subsequent in-run turns. A second overflow per run fails the run.
///
/// Both hooks share the coordinator's per-run state; `node_handle` is the
/// session's node-id lock (same lock the persist listener reads), so the
/// append-target switch is atomic with respect to post-compaction appends.
pub fn install_compaction_hooks(
    agent: &mut Agent,
    coordinator: &Arc<CompactionCoordinator>,
    node_handle: Arc<tokio::sync::RwLock<String>>,
) {
    let transform_coordinator = coordinator.clone();
    let transform_handle = node_handle.clone();
    agent.transform_context = Some(Arc::new(move |messages, _abort| {
        let coordinator = transform_coordinator.clone();
        let node_handle = transform_handle.clone();
        Box::pin(async move {
            let current_id = node_handle.read().await.clone();
            match coordinator.pre_request_hook(&messages, &current_id).await {
                Some(outcome) => {
                    *node_handle.write().await = outcome.compacted_id.clone();
                    Ok(outcome.injected_view)
                }
                None => Ok(messages),
            }
        })
    }));

    let prepare_coordinator = coordinator.clone();
    let prepare_handle = node_handle.clone();
    agent.prepare_next_turn = Some(Arc::new(move |ctx, _abort| {
        let coordinator = prepare_coordinator.clone();
        let node_handle = prepare_handle.clone();
        Box::pin(async move {
            let current_id = node_handle.read().await.clone();
            match coordinator
                .post_turn_hook(&ctx.message, &ctx.context, &current_id)
                .await
            {
                Ok(Some(outcome)) => {
                    *node_handle.write().await = outcome.compacted_id.clone();
                    Ok(Some(AgentLoopTurnUpdate {
                        context: Some(AgentContext {
                            system_prompt: ctx.context.system_prompt.clone(),
                            messages: outcome.injected_view,
                            tools: ctx.context.tools.clone(),
                        }),
                        model: None,
                        thinking_level: None,
                    }))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(e),
            }
        })
    }));
}
#[cfg(test)]
mod tests {
    // Store plumbing unwraps and expect-failures are the test idiom here;
    // failure paths themselves are asserted through returned errors.
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use aaos_session::compaction::TRANSCRIPT_PREAMBLE;
    use aaos_session::{Segment, SessionStore};
    use pi_agent_core::types::{
        AgentContext, AssistantMessage, Message, Model, StopReason, Usage, UserMessage,
    };

    use super::{
        CompactionCoordinator, CompactionError, CompactionOutcome, CompactionSettings,
        is_overflow_message, is_silent_overflow, strip_failed_assistant,
    };

    fn settings(enabled: bool, reserve_tokens: u64, keep_recent_tokens: u64) -> CompactionSettings {
        CompactionSettings {
            enabled,
            reserve_tokens,
            keep_recent_tokens,
        }
    }

    fn model_with_window(context_window: u64) -> Model {
        Model {
            context_window,
            ..Model::unknown()
        }
    }

    fn user(text: &str) -> Message {
        Message::User(UserMessage::new(text))
    }

    /// Assistant usage anchor: drives `context_tokens` like a real response.
    fn anchored_assistant(total_tokens: u64) -> Message {
        Message::Assistant(AssistantMessage {
            usage: Usage {
                total_tokens,
                ..Usage::default()
            },
            ..AssistantMessage::default()
        })
    }

    fn error_assistant(message: &str) -> AssistantMessage {
        AssistantMessage {
            stop_reason: StopReason::Error,
            error_message: Some(message.to_string()),
            ..AssistantMessage::default()
        }
    }

    fn agent_context(messages: Vec<Message>) -> AgentContext {
        AgentContext {
            system_prompt: String::new(),
            messages,
            tools: Vec::new(),
        }
    }

    async fn fresh_store() -> (tempfile::TempDir, SessionStore) {
        let tmp = tempfile::tempdir().unwrap();
        let store = SessionStore::open(tmp.path()).await.unwrap();
        (tmp, store)
    }

    /// Store fixture: u("hello"), u("world"), a("ok") anchored at 100_000
    /// tokens. With `keep_recent_tokens = 1` the cut lands at 2, and the
    /// anchor makes `before_tokens` (100k) dwarf the transcript, so the
    /// shrink check passes.
    async fn store_with_compactable_fixture() -> (tempfile::TempDir, SessionStore, String) {
        let (tmp, store) = fresh_store().await;
        let id = store.create_root().await.unwrap();
        store
            .append_segment(&id, &Segment::user_text("hello"))
            .await
            .unwrap();
        store
            .append_segment(&id, &Segment::user_text("world"))
            .await
            .unwrap();
        let mut anchored = Segment::assistant_text("ok");
        if let Segment::Assistant(a) = &mut anchored {
            a.usage.total_tokens = 100_000;
        }

        store.append_segment(&id, &anchored).await.unwrap();
        (tmp, store, id)
    }
    /// First text block of a user message, `None` for any other shape —
    /// lets tests unwrap a malformed transcript without `panic!`.
    fn transcript_text(message: &Message) -> Option<&str> {
        let Message::User(user) = message else {
            return None;
        };
        match user.content.first() {
            Some(pi_agent_core::types::ContentBlock::Text { text }) => Some(text),
            _ => None,
        }
    }

    // ---- CompactionSettings::from_env_values ----

    #[test]
    fn from_env_values_unset_falls_back_to_defaults() {
        let parsed = CompactionSettings::from_env_values(None, None, None);
        let default = CompactionSettings::default();
        assert!(parsed.enabled);
        assert_eq!(parsed.reserve_tokens, default.reserve_tokens);
        assert_eq!(parsed.keep_recent_tokens, default.keep_recent_tokens);
    }

    #[test]
    fn from_env_values_disabled_strings_case_and_whitespace_insensitive() {
        for value in ["0", "false", "no", "FALSE", "No", " false ", "\tno\n"] {
            let parsed = CompactionSettings::from_env_values(Some(value), None, None);
            assert!(!parsed.enabled, "value {value:?} must disable");
        }
    }

    #[test]
    fn from_env_values_enabled_strings() {
        for value in ["1", "true", "YES", "on", ""] {
            let parsed = CompactionSettings::from_env_values(Some(value), None, None);
            assert!(parsed.enabled, "value {value:?} must enable");
        }
    }

    #[test]
    fn from_env_values_numeric_parse_and_invalid_fallbacks() {
        let parsed = CompactionSettings::from_env_values(None, Some("4096"), Some(" 12345 "));
        assert_eq!(parsed.reserve_tokens, 4096);
        assert_eq!(parsed.keep_recent_tokens, 12345);

        let defaults = CompactionSettings::default();
        for reserve in ["", "abc", "-1", "12.5", "99999999999999999999999"] {
            let parsed = CompactionSettings::from_env_values(None, Some(reserve), None);
            assert_eq!(
                parsed.reserve_tokens, defaults.reserve_tokens,
                "{reserve:?}"
            );
        }
        let parsed = CompactionSettings::from_env_values(None, None, Some("not a number"));
        assert_eq!(parsed.keep_recent_tokens, defaults.keep_recent_tokens);
    }

    // ---- is_overflow_message ----

    #[test]
    fn is_overflow_message_matches_overflow_phrasings() {
        for msg in [
            "prompt is too long: 300000 tokens",
            "Prompt is too long",
            "request exceeds the context window",
            "maximum prompt length of 128000 tokens",
            "context length exceeded",
        ] {
            assert!(
                is_overflow_message(msg),
                "{msg:?} must classify as overflow"
            );
        }
    }

    #[test]
    fn is_overflow_message_excludes_rate_limits_and_unrelated() {
        for msg in [
            "rate limit exceeded, retry later",
            "HTTP 429 too many requests",
            "request throttled by provider",
            "internal server error",
            // A rate-limit message wins even when it mentions overflow words.
            "rate limit: prompt is too long",
        ] {
            assert!(!is_overflow_message(msg), "{msg:?} must not be overflow");
        }
    }

    // ---- is_silent_overflow ----

    #[test]
    fn is_silent_overflow_boundaries() {
        let usage = |input: u64, cache_read: u64| Usage {
            input,
            cache_read,
            ..Usage::default()
        };
        // input + cache_read above the window.
        assert!(is_silent_overflow(1000, &usage(900, 200)));
        // Exactly at the window: not an overflow.
        assert!(!is_silent_overflow(1000, &usage(1000, 0)));
        // Below the window.
        assert!(!is_silent_overflow(1000, &usage(500, 100)));
        // Zero window (unknown model): no check.
        assert!(!is_silent_overflow(0, &usage(999_999, 0)));
        // input + cache_read saturates at the window itself: never exceeds.
        assert!(!is_silent_overflow(u64::MAX, &usage(u64::MAX, 1)));
        // No wraparound producing a false negative on normal magnitudes.
        assert!(is_silent_overflow(u64::MAX - 1, &usage(u64::MAX, 1)));
    }

    // ---- strip_failed_assistant ----

    #[test]
    fn strip_failed_assistant_pops_trailing_error_assistant() {
        let outcome = CompactionOutcome {
            compacted_id: "c".to_string(),
            before_tokens: 10,
            after_tokens: 5,
            injected_view: vec![user("q"), Message::Assistant(error_assistant("boom"))],
        };
        let stripped = strip_failed_assistant(outcome);
        assert_eq!(stripped.injected_view, vec![user("q")]);
        assert_eq!(stripped.compacted_id, "c");
        assert_eq!(stripped.before_tokens, 10);
        assert_eq!(stripped.after_tokens, 5);
    }

    #[test]
    fn strip_failed_assistant_keeps_non_error_tail() {
        let stop_tail = CompactionOutcome {
            compacted_id: "c".to_string(),
            before_tokens: 10,
            after_tokens: 5,
            injected_view: vec![user("q"), Message::Assistant(AssistantMessage::default())],
        };
        let stripped = strip_failed_assistant(stop_tail);
        assert_eq!(stripped.injected_view.len(), 2);

        let user_tail = CompactionOutcome {
            compacted_id: "c".to_string(),
            before_tokens: 10,
            after_tokens: 5,
            injected_view: vec![Message::Assistant(AssistantMessage::default()), user("q")],
        };
        let stripped = strip_failed_assistant(user_tail);
        assert_eq!(
            stripped.injected_view,
            vec![Message::Assistant(AssistantMessage::default()), user("q")]
        );
    }

    // ---- compact ----

    #[tokio::test]
    async fn compact_missing_session_is_failed() {
        let (_tmp, store, _) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let err = coordinator.compact("missing").await.unwrap_err();
        assert!(matches!(err, CompactionError::Failed(_)));
    }

    #[tokio::test]
    async fn compact_empty_view_is_nothing_to_compact() {
        let (_tmp, store) = fresh_store().await;
        let id = store.create_root().await.unwrap();
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let err = coordinator.compact(&id).await.unwrap_err();
        assert_eq!(err, CompactionError::NothingToCompact);
        assert_eq!(err.to_string(), "Nothing to compact");
        assert!(coordinator.take_pending_resync().is_none());
    }

    #[tokio::test]
    async fn compact_tail_under_keep_budget_is_nothing_to_compact() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        // Default keep budget dwarfs the fixture's estimates: cut at 0.
        let coordinator = CompactionCoordinator::new(
            store,
            settings(true, 0, 20_000),
            &model_with_window(200_000),
        );
        let err = coordinator.compact(&id).await.unwrap_err();
        assert_eq!(err, CompactionError::NothingToCompact);
    }

    #[tokio::test]
    async fn compact_no_shrink_is_failed_and_store_untouched() {
        let (_tmp, store) = fresh_store().await;
        let id = store.create_root().await.unwrap();
        for segment in [
            Segment::user_text("hello"),
            Segment::assistant_text("hi"),
            Segment::user_text("bye"),
            Segment::assistant_text("ok"),
        ] {
            store.append_segment(&id, &segment).await.unwrap();
        }
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let err = coordinator.compact(&id).await.unwrap_err();
        assert!(
            matches!(
                &err,
                CompactionError::Failed(reason) if reason.contains("would not reduce context")
            ),
            "{err:?}"
        );
        // Refusal must not touch the store nor arm the resync.
        assert_eq!(
            coordinator
                .store
                .materialize_plain(&id)
                .await
                .unwrap()
                .len(),
            4
        );
        assert!(coordinator.take_pending_resync().is_none());
    }

    #[tokio::test]
    async fn compact_success_commits_summary_and_sets_resync() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let outcome = coordinator.compact(&id).await.unwrap();
        assert!(outcome.after_tokens < outcome.before_tokens);

        // Injected view: transcript user message + retained tail.
        let text = transcript_text(&outcome.injected_view[0])
            .expect("first injected message must be the transcript user message");
        assert!(text.starts_with(TRANSCRIPT_PREAMBLE));
        assert!(text.contains("[User] hello"));
        assert!(matches!(&outcome.injected_view[1], Message::Assistant(_)));

        // Store view of the compacted node: summary + retained tail.
        let view = coordinator
            .store
            .materialize_plain(&outcome.compacted_id)
            .await
            .unwrap();
        assert_eq!(view.len(), 2);
        assert!(matches!(view[0], Segment::Summary(_)));
        assert!(matches!(view[1], Segment::Assistant(_)));

        // Resync target armed once, consumed by take.
        assert_eq!(
            coordinator.take_pending_resync().as_deref(),
            Some(outcome.compacted_id.as_str())
        );
        assert!(coordinator.take_pending_resync().is_none());
    }

    // ---- pre_request_hook ----

    #[tokio::test]
    async fn pre_request_hook_disabled_returns_none() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(false, 0, 1), &model_with_window(200_000));
        let messages = vec![user("hi"), anchored_assistant(250_000)];
        assert!(coordinator.pre_request_hook(&messages, &id).await.is_none());
        assert!(coordinator.take_pending_resync().is_none());
        assert_eq!(
            coordinator
                .store
                .materialize_plain(&id)
                .await
                .unwrap()
                .len(),
            3
        );
    }

    #[tokio::test]
    async fn pre_request_hook_below_threshold_returns_none() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let messages = vec![user("hi"), anchored_assistant(50)];
        assert!(coordinator.pre_request_hook(&messages, &id).await.is_none());
        assert!(coordinator.take_pending_resync().is_none());
    }

    #[tokio::test]
    async fn pre_request_hook_above_threshold_compacts() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let messages = vec![user("hi"), anchored_assistant(250_000)];
        let outcome = coordinator
            .pre_request_hook(&messages, &id)
            .await
            .expect("must compact above threshold");
        assert!(outcome.after_tokens < outcome.before_tokens);
        assert!(matches!(outcome.injected_view[0], Message::User(_)));
        assert_eq!(
            coordinator.take_pending_resync().as_deref(),
            Some(outcome.compacted_id.as_str())
        );
    }

    #[tokio::test]
    async fn pre_request_hook_compacts_once_per_run_and_begin_run_resets() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let messages = vec![user("hi"), anchored_assistant(250_000)];
        assert!(coordinator.pre_request_hook(&messages, &id).await.is_some());
        // Second request in the same run: guarded off.
        assert!(coordinator.pre_request_hook(&messages, &id).await.is_none());
        // begin_run clears the per-run flags and the resync target.
        coordinator.begin_run();
        assert!(coordinator.take_pending_resync().is_none());
    }

    #[tokio::test]
    async fn pre_request_hook_compaction_failure_returns_none() {
        let (_tmp, store) = fresh_store().await;
        let root = store.create_root().await.unwrap();
        // Empty view: compact() refuses with NothingToCompact.
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let messages = vec![user("hi"), anchored_assistant(250_000)];
        assert!(
            coordinator
                .pre_request_hook(&messages, &root)
                .await
                .is_none()
        );
        assert!(coordinator.take_pending_resync().is_none());
    }

    // ---- post_turn_hook ----

    #[tokio::test]
    async fn post_turn_hook_normal_turn_below_threshold_returns_none() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let context = agent_context(vec![user("hi"), anchored_assistant(50)]);
        let assistant = AssistantMessage::default();
        assert!(
            coordinator
                .post_turn_hook(&assistant, &context, &id)
                .await
                .unwrap()
                .is_none()
        );
        assert!(coordinator.take_pending_resync().is_none());
    }

    #[tokio::test]
    async fn post_turn_hook_rate_limit_error_is_not_overflow() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let context = agent_context(vec![user("hi"), anchored_assistant(50)]);
        let assistant = error_assistant("rate limit exceeded");
        assert!(
            coordinator
                .post_turn_hook(&assistant, &context, &id)
                .await
                .unwrap()
                .is_none()
        );
        assert!(coordinator.take_pending_resync().is_none());
    }

    #[tokio::test]
    async fn post_turn_hook_overflow_error_recovers_even_when_disabled() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(false, 0, 1), &model_with_window(200_000));
        let context = agent_context(vec![user("hi")]);
        let assistant = error_assistant("prompt is too long: 300000 tokens");
        let outcome = coordinator
            .post_turn_hook(&assistant, &context, &id)
            .await
            .unwrap()
            .expect("overflow recovery must run even when disabled");
        assert!(outcome.after_tokens < outcome.before_tokens);
        assert_eq!(
            coordinator.take_pending_resync().as_deref(),
            Some(outcome.compacted_id.as_str())
        );
    }

    #[tokio::test]
    async fn post_turn_hook_second_overflow_fails_the_run() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(false, 0, 1), &model_with_window(200_000));
        let context = agent_context(vec![user("hi")]);
        let assistant = error_assistant("prompt is too long");
        assert!(
            coordinator
                .post_turn_hook(&assistant, &context, &id)
                .await
                .unwrap()
                .is_some()
        );
        let err = coordinator
            .post_turn_hook(&assistant, &context, &id)
            .await
            .unwrap_err();
        assert!(err.contains("overflow persists"), "{err}");
    }

    #[tokio::test]
    async fn post_turn_hook_silent_overflow_recovers_even_when_disabled() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        // Usage beyond the window with no error message; disabled settings
        // rule out the threshold branch, isolating the silent-overflow path.
        let coordinator =
            CompactionCoordinator::new(store, settings(false, 0, 1), &model_with_window(1_000));
        let context = agent_context(vec![user("hi")]);
        let assistant = AssistantMessage {
            usage: Usage {
                input: 900,
                cache_read: 200,
                ..Usage::default()
            },
            ..AssistantMessage::default()
        };
        let outcome = coordinator
            .post_turn_hook(&assistant, &context, &id)
            .await
            .unwrap()
            .expect("silent overflow must trigger recovery");
        assert!(coordinator.take_pending_resync().is_some());
        assert!(matches!(outcome.injected_view[0], Message::User(_)));
    }

    #[tokio::test]
    async fn post_turn_hook_threshold_branch_compacts_and_keeps_tail() {
        let (_tmp, store, id) = store_with_compactable_fixture().await;
        let coordinator =
            CompactionCoordinator::new(store, settings(true, 0, 1), &model_with_window(200_000));
        let context = agent_context(vec![user("hi"), anchored_assistant(250_000)]);
        let assistant = AssistantMessage::default();
        let outcome = coordinator
            .post_turn_hook(&assistant, &context, &id)
            .await
            .unwrap()
            .expect("threshold branch must compact");
        // Not an overflow: nothing stripped, tail intact.
        assert!(matches!(
            outcome.injected_view.last(),
            Some(Message::Assistant(_))
        ));
        assert_eq!(
            coordinator.take_pending_resync().as_deref(),
            Some(outcome.compacted_id.as_str())
        );
    }
}

/// Transcript-level coordinator tests (migrated from the CLI; assertion
/// semantics unchanged): deterministic transcript content, projection
/// validation, stale-anchor zeroing, and the manual-compact/disabled
/// interplay. All settings are constructed directly (never from env) so the
/// tests are hermetic against process-env leakage.
#[cfg(test)]
mod transcript_tests {
    // Test-support expects and assert-style panics are the idiom here; the
    // production paths above stay panic-free.
    #![expect(clippy::panic)]
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use aaos_session::compaction::{
        DEFAULT_KEEP_RECENT_TOKENS, DEFAULT_RESERVE_TOKENS, TRANSCRIPT_PREAMBLE,
    };
    use aaos_session::{
        AgentSession, AssistantSegment, ContentBlock as StoreBlock, Segment, SessionStore,
        StopReason as StoreStopReason, ToolCall as StoreToolCall, Usage as StoreUsage,
    };
    use pi_agent_core::agent::Agent;
    use pi_agent_core::stream::{MockAssistantStream, mock_stream_fn, simple_text_response};
    use pi_agent_core::types::{AssistantMessage, ContentBlock, Message, Model, StopReason, Usage};
    use serde_json::json;

    use super::{CompactionCoordinator, CompactionError, CompactionSettings};

    fn test_model() -> Model {
        Model {
            id: "test".into(),
            ..Model::unknown()
        }
    }

    fn first_text(msg: &Message) -> String {
        let content = match msg {
            Message::User(u) => &u.content,
            Message::Assistant(a) => &a.content,
            Message::ToolResult(t) => &t.content,
        };
        content
            .iter()
            .find_map(|b| match b {
                ContentBlock::Text { text } => Some(text.clone()),
                _ => None,
            })
            .unwrap_or_default()
    }

    /// Seed `n` user/assistant text turns, each ~`chars` chars long.
    async fn seed_turns(store: &SessionStore, root: &str, n: usize, chars: usize) {
        for i in 0..n {
            store
                .append_segment(
                    root,
                    &Segment::user_text(format!("u{i}-{}", "x".repeat(chars))),
                )
                .await
                .unwrap();
            store
                .append_segment(
                    root,
                    &Segment::assistant_text(format!("a{i}-{}", "y".repeat(chars))),
                )
                .await
                .unwrap();
        }
    }

    /// Seed one tool round-trip: an assistant tool call + a `result_chars`-
    /// long tool result. Tool results are the context bulk compaction
    /// replaces with a path, so they make the projection strictly smaller.
    async fn seed_tool_turn(
        store: &SessionStore,
        root: &str,
        call_id: &str,
        name: &str,
        args: serde_json::Value,
        result_chars: usize,
    ) {
        store
            .append_segment(
                root,
                &Segment::Assistant(AssistantSegment {
                    content: vec![StoreBlock::ToolCall(StoreToolCall {
                        id: call_id.into(),
                        name: name.into(),
                        arguments: args,
                    })],
                    stop_reason: StoreStopReason::ToolUse,
                    model: "test".into(),
                    provider: "test".into(),
                    api: "test".into(),
                    usage: StoreUsage::default(),
                    error_message: None,
                }),
            )
            .await
            .unwrap();
        store
            .append_segment(
                root,
                &Segment::tool_result_text(call_id, "R".repeat(result_chars)),
            )
            .await
            .unwrap();
    }

    /// Absolute object paths referenced by a transcript's path lines:
    /// the `[Tool result] {path}` payload, `[Tool call] … — full
    /// arguments at {path}`, and `[Image] at {path}` (block-granular
    /// objects, ADR-0006).
    fn transcript_paths(transcript: &str) -> Vec<&str> {
        transcript
            .lines()
            .filter_map(|line| {
                let path = match line.strip_prefix("[Tool result] ") {
                    Some(rest) if rest != "(empty)" => rest,
                    _ => line.rsplit_once(" at ").map(|(_, path)| path.trim())?,
                };
                Some(path.trim()).filter(|path| path.starts_with('/'))
            })
            .collect()
    }

    /// (a) Happy path: enough content → compact creates a node whose
    /// summary segment is a deterministic transcript (seeded texts inline,
    /// tool results replaced by absolute paths that resolve), with model None (deterministic, no
    /// generating model; ADR-0006 structural provenance); resuming onto the compacted node
    /// yields the transcript user-message + retained tail.
    #[tokio::test]
    async fn happy_path_creates_node_and_resumes() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        // Tool round-trip FIRST (its 4000-char result is the context bulk
        // the compaction replaces with a path), then 5 text turns; the cut
        // lands inside the text turns, so the result is compacted away.
        seed_tool_turn(
            &store,
            &root,
            "c1",
            "read",
            json!({"path": "/tmp/note.txt"}),
            4000,
        )
        .await;
        seed_turns(&store, &root, 5, 100).await;

        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 60,
        };
        let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

        let outcome = coordinator.compact(&root).await.expect("compact ok");
        assert_ne!(outcome.compacted_id, root);
        assert!(
            outcome.before_tokens > outcome.after_tokens,
            "{} -> {}",
            outcome.before_tokens,
            outcome.after_tokens
        );

        // Transcript content: preamble, seeded texts inline, tool result
        // at the absolute path, tool call inline.
        let summary = &first_text(&outcome.injected_view[0]);
        assert!(summary.starts_with(TRANSCRIPT_PREAMBLE), "{summary}");
        assert!(summary.contains("[User] u0-"), "{summary}");
        assert!(summary.contains("[Assistant] a0-"), "{summary}");
        let paths = transcript_paths(summary);
        let result_path = summary
            .lines()
            .find_map(|line| line.strip_prefix("[Tool result] "))
            .expect("transcript references the result object");
        assert!(
            summary.contains(&format!("[Tool result] {result_path}")),
            "{summary}"
        );
        assert!(
            summary.contains(r#"[Tool call] read({"path":"/tmp/note.txt"})"#),
            "{summary}"
        );
        assert!(!summary.contains("RRRR"), "result text is not inlined");
        assert!(
            !summary.contains("<summary>"),
            "no pi-style summary wrapper"
        );

        // Every referenced object path resolves on disk; the result
        // object holds the raw output bytes (block-granular, ADR-0006).
        assert!(
            !paths.is_empty(),
            "transcript references at least one object"
        );
        for path in &paths {
            assert!(std::fs::exists(path).unwrap(), "object exists: {path}");
        }
        assert_eq!(
            std::fs::read_to_string(result_path).unwrap(),
            "R".repeat(4000),
            "result object holds the raw output bytes"
        );

        // Summary segment persisted: provenance is
        // structural (ADR-0006): `fetch_originals` covers the prefix.
        let view = store
            .materialize_plain(&outcome.compacted_id)
            .await
            .unwrap();
        let Segment::Summary(s) = &view[0] else {
            panic!("first segment must be a summary");
        };
        assert_eq!(s.content, *summary);
        let originals = store.fetch_originals(&outcome.compacted_id).await.unwrap();
        assert_eq!(originals.len(), 1, "one compaction map");
        assert!(
            originals[0].originals.len() > 1,
            "the covered prefix is retrievable: {:?}",
            originals[0]
        );

        // Resume onto the compacted node: transcript user-message + tail.
        let mut session = AgentSession::new(
            store.clone(),
            Agent::new(simple_text_response("ok")),
            &root,
            dir.path(),
        );
        session.resume(&outcome.compacted_id).await.unwrap();
        let messages = &session.state().messages;
        // Summary renders as a bare user message: exactly the summary
        // content (no provenance prefix — the preamble is
        // self-describing).
        assert_eq!(
            first_text(&messages[0]),
            *summary,
            "summary renders bare as a user message"
        );
        assert!(first_text(&messages[0]).contains(TRANSCRIPT_PREAMBLE));
        assert!(
            messages.len() < 12,
            "tail must be shorter than the full transcript"
        );
        assert_eq!(
            session.current_session_id().await,
            outcome.compacted_id,
            "resume switched the append target"
        );
        // Head unchanged: compaction derives but does not append.
        assert_eq!(store.head().await.unwrap().as_deref(), Some(root.as_str()));
    }

    /// (b) Nothing to compact: a tiny session refuses without creating a node.
    #[tokio::test]
    async fn tiny_session_refuses_nothing_to_compact() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        store
            .append_segment(&root, &Segment::user_text("hi"))
            .await
            .unwrap();

        let settings = CompactionSettings::default();
        let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

        let err = coordinator.compact(&root).await.unwrap_err();
        assert_eq!(err, CompactionError::NothingToCompact);
        // No node created: root's view is unchanged, head unchanged.
        assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 1);
        assert_eq!(store.head().await.unwrap().as_deref(), Some(root.as_str()));
    }

    /// (c) Re-compaction: a compacted node compacts again naturally. The
    /// second node's transcript embeds the first transcript's text, the
    /// first node's object paths still resolve, and the view is correct.
    #[tokio::test]
    async fn recompaction_embeds_previous_transcript() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        seed_tool_turn(&store, &root, "c1", "read", json!({"path": "/a"}), 4000).await;
        seed_turns(&store, &root, 4, 100).await;

        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 60,
        };
        let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

        let first = coordinator.compact(&root).await.expect("first compact ok");
        // Keep compacting on the compacted node.
        let first_view = store.materialize_plain(&first.compacted_id).await.unwrap();
        let Segment::Summary(first_summary) = &first_view[0] else {
            panic!("first node starts with a summary");
        };
        let first_transcript = first_summary.content.clone();
        let first_paths: Vec<String> = transcript_paths(&first_transcript)
            .iter()
            .map(|p| p.to_string())
            .collect();
        assert!(
            !first_paths.is_empty(),
            "first transcript references object paths"
        );

        // Append a second tool round-trip and enough turns after it that
        // the second cut lands in the text tail and the new result is
        // compacted away; the old transcript is embedded verbatim.
        seed_tool_turn(
            &store,
            &first.compacted_id,
            "c2",
            "bash",
            json!({"command": "ls"}),
            4000,
        )
        .await;
        seed_turns(&store, &first.compacted_id, 4, 100).await;
        let second = coordinator
            .compact(&first.compacted_id)
            .await
            .expect("second compact ok");
        assert_ne!(second.compacted_id, first.compacted_id);

        // Second node: the old transcript is embedded verbatim (it is
        // already a transcript with paths — transitive), and the old
        // object paths still resolve.
        let second_view = store.materialize_plain(&second.compacted_id).await.unwrap();
        let Segment::Summary(second_summary) = &second_view[0] else {
            panic!("second node starts with a summary");
        };
        assert!(
            second_summary.content.contains(&first_transcript),
            "first transcript embedded: {}",
            second_summary.content
        );
        assert!(
            second_summary.content.contains("[User] u0-"),
            "new turns rendered inline: {}",
            second_summary.content
        );
        for path in &first_paths {
            assert!(
                std::fs::exists(path).unwrap(),
                "old object path still resolves: {path}"
            );
        }
        for path in transcript_paths(&second_summary.content) {
            assert!(
                std::fs::exists(path).unwrap(),
                "second transcript's paths resolve: {path}"
            );
        }

        // Resume onto the second node: transcript + retained tail.
        let mut session = AgentSession::new(
            store.clone(),
            Agent::new(simple_text_response("ok")),
            &root,
            dir.path(),
        );
        session.resume(&second.compacted_id).await.unwrap();
        let messages = &session.state().messages;
        // Summary renders as a bare user message (no provenance prefix).
        assert_eq!(
            first_text(&messages[0]),
            second_summary.content,
            "second summary renders bare as a user message"
        );
        assert!(
            first_text(&messages[0]).contains(TRANSCRIPT_PREAMBLE),
            "second summary keeps the preamble"
        );
        // Only the embedded first transcript may mention the old prefix;
        // the retained tail must not.
        assert!(
            messages[1..].iter().all(|m| !first_text(m).contains("u0-")),
            "old prefix not in the tail"
        );
        assert_eq!(
            session.current_session_id().await,
            second.compacted_id,
            "resume switched to the second node"
        );
    }

    /// (d) Degenerate projection: a transcript larger than the prefix it
    /// replaces is rejected — no node is created. Constructed with a
    /// tool-free dialogue where the transcript preamble dominates.
    #[tokio::test]
    async fn degenerate_projection_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        // Text-only dialogue: no tool results to shed, so the transcript
        // (preamble + role labels) is never smaller than the prefix it
        // replaces.
        seed_turns(&store, &root, 6, 100).await;

        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 60,
        };
        let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

        let err = coordinator.compact(&root).await.unwrap_err();
        match &err {
            CompactionError::Failed(m) => {
                assert!(m.contains("reduce"), "expected reduce, got {m}");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 12);
    }

    /// Stale usage anchors: a pre-compaction assistant in the retained
    /// tail carries a `usage.total_tokens` describing the PRE-compaction
    /// context. The projected view contains a summary by construction,
    /// so its assistant usage must be zeroed before metering —
    /// `after_tokens` falls back to pure estimation instead of
    /// re-anchoring on the stale total, which refuses every compaction
    /// with "would not reduce context".
    #[tokio::test]
    async fn stale_tail_anchor_does_not_refuse_compaction() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        // Tool round-trip for the compactable bulk, then text turns; the
        // LAST assistant carries a large pre-compaction usage total (the
        // normal end of a live turn) and lands in the retained tail.
        seed_tool_turn(&store, &root, "c1", "read", json!({"path": "/a"}), 4000).await;
        seed_turns(&store, &root, 4, 100).await;
        store
            .append_segment(
                &root,
                &Segment::Assistant(AssistantSegment {
                    content: vec![StoreBlock::Text {
                        text: "stale anchor".into(),
                    }],
                    stop_reason: StoreStopReason::Stop,
                    model: "test".into(),
                    provider: "test".into(),
                    api: "test".into(),
                    usage: StoreUsage {
                        total_tokens: 100_000,
                        ..StoreUsage::default()
                    },
                    error_message: None,
                }),
            )
            .await
            .unwrap();

        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 16_384,
            keep_recent_tokens: 60,
        };
        let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

        let outcome = coordinator
            .compact(&root)
            .await
            .expect("compaction must not be refused by the stale tail anchor");
        assert!(
            outcome.after_tokens < outcome.before_tokens,
            "{} -> {}",
            outcome.before_tokens,
            outcome.after_tokens
        );
        // The injected view carries a summary, so no assistant in it may
        // hold a pre-compaction usage anchor.
        for message in &outcome.injected_view {
            if let Message::Assistant(a) = message {
                assert_eq!(
                    a.usage.total_tokens, 0,
                    "injected assistant must carry zero usage"
                );
            }
        }
    }

    /// Overflow recovery excludes the failed/truncated assistant from the
    /// retry context (issue #70 §3.5) while the store keeps it in session
    /// history. Driven through `post_turn_hook` directly: an Error-stop
    /// assistant ends the run, so the retry context is the hook's injected
    /// view, consumed by the caller.
    #[tokio::test]
    async fn overflow_recovery_strips_failed_assistant_from_retry_context() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        // Seed enough context that the recovery compaction succeeds,
        // and persist the failed/truncated assistant as the last
        // segment of the session history (the message the overflow
        // branch strips from the injected view).
        seed_tool_turn(&store, &root, "c1", "bash", json!({"command": "ls"}), 4000).await;
        seed_turns(&store, &root, 5, 100).await;
        let failed_text = "FAILED MESSAGE TEXT";
        store
            .append_segment(
                &root,
                &Segment::Assistant(AssistantSegment {
                    content: vec![StoreBlock::Text {
                        text: failed_text.into(),
                    }],
                    stop_reason: StoreStopReason::Error,
                    model: "test".into(),
                    provider: "test".into(),
                    api: "test".into(),
                    usage: StoreUsage::default(),
                    error_message: Some("provider: prompt is too long for context window".into()),
                }),
            )
            .await
            .unwrap();

        let model = Model {
            id: "test".into(),
            context_window: 100,
            ..Model::unknown()
        };
        let settings = CompactionSettings {
            enabled: true,
            reserve_tokens: 50,
            keep_recent_tokens: 60,
        };
        let coordinator = CompactionCoordinator::new(store.clone(), settings, &model);
        coordinator.begin_run();

        let failed = AssistantMessage {
            content: vec![ContentBlock::text(failed_text)],
            stop_reason: StopReason::Error,
            error_message: Some("provider: prompt is too long for context window".into()),
            ..Default::default()
        };
        let context = pi_agent_core::types::AgentContext {
            system_prompt: "sys".into(),
            messages: vec![],
            tools: vec![],
        };
        let outcome = coordinator
            .post_turn_hook(&failed, &context, &root)
            .await
            .expect("overflow recovery succeeds")
            .expect("a compaction committed");

        // Injected view: transcript first, failed assistant excluded.
        assert!(
            first_text(&outcome.injected_view[0]).contains(TRANSCRIPT_PREAMBLE),
            "injected view starts with the transcript: {:?}",
            outcome.injected_view[0]
        );
        assert!(
            !outcome
                .injected_view
                .iter()
                .any(|m| first_text(m) == failed_text),
            "failed assistant excluded from the injected view: {:?}",
            outcome.injected_view
        );

        // The failed message is still persisted in session history
        // (compact derives; the segment stays in the store).
        let head = store.head().await.unwrap().unwrap();
        let view = store.materialize_plain(&head).await.unwrap();
        assert!(
            view.iter().any(|seg| matches!(
                seg,
                Segment::Assistant(a)
                    if a.stop_reason == StoreStopReason::Error
                        && a.content.iter().any(|b| matches!(
                            b,
                            StoreBlock::Text { text }
                                if text == failed_text
                        ))
            )),
            "failed assistant stays in the store"
        );
    }

    /// (e) AAOS_COMPACTION_ENABLED=0 does not block manual /compact —
    /// `enabled=false` settings still compact on explicit request.
    #[tokio::test]
    async fn manual_compact_ignores_disabled() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::open(dir.path()).await.unwrap();
        let root = store.create_root().await.unwrap();
        seed_tool_turn(&store, &root, "c1", "read", json!({"path": "/a"}), 4000).await;
        seed_turns(&store, &root, 5, 100).await;

        let settings = CompactionSettings {
            enabled: false,
            reserve_tokens: 50,
            keep_recent_tokens: 60,
        };
        let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());
        let outcome = coordinator.compact(&root).await.expect("manual compact ok");
        assert_ne!(outcome.compacted_id, root);
        let view = store
            .materialize_plain(&outcome.compacted_id)
            .await
            .unwrap();
        let Segment::Summary(s) = &view[0] else {
            panic!("manual compact created a summary");
        };
        assert!(s.content.starts_with(TRANSCRIPT_PREAMBLE));
    }

    /// Auto-trigger hooks: transform_context + prepare_next_turn wired
    /// through `install_compaction_hooks`. All settings constructed
    /// directly (never from_env) so tests are hermetic.
    mod hooks {
        use super::*;

        use crate::compaction::install_compaction_hooks;
        use crate::session::turn_outcome;
        use std::sync::{Arc, Mutex};
        use tokio::sync::RwLock;

        /// A session agent with a recording fake stream and compaction hooks
        /// installed, bound to a seeded session.
        async fn hooked_session(
            dir: &tempfile::TempDir,
            store: &SessionStore,
            root: &str,
            settings: CompactionSettings,
            model: &Model,
            record: Arc<Mutex<Vec<String>>>,
        ) -> (AgentSession, Arc<CompactionCoordinator>, Arc<Mutex<usize>>) {
            let llm_calls = Arc::new(Mutex::new(0usize));
            let llm_calls_for_stream = llm_calls.clone();
            let record_for_stream = record.clone();
            let stream_fn = mock_stream_fn(move |_model, ctx, _opts| {
                *llm_calls_for_stream.lock().unwrap() += 1;
                let texts: Vec<String> = ctx.messages.iter().map(first_text).collect();
                record_for_stream
                    .lock()
                    .unwrap()
                    .push(texts.join("\n---\n"));
                Box::new(MockAssistantStream::new(AssistantMessage::text("ok")))
            });
            let mut agent = Agent::new(stream_fn);
            agent.state.model = model.clone();
            agent.state.system_prompt = "sys".to_string();
            agent.stream_fn_options.api_key = None;
            let mut session = AgentSession::new(store.clone(), agent, root.to_string(), dir.path());
            // Load the seeded transcript so the prompt's context is the full
            // conversation (which is what the hooks see and measure).
            session.resume(root).await.unwrap();
            let coordinator = Arc::new(CompactionCoordinator::new(store.clone(), settings, model));
            let node_handle: Arc<RwLock<String>> = session.session_id_lock();
            install_compaction_hooks(session.agent_mut(), &coordinator, node_handle);
            (session, coordinator, llm_calls)
        }

        /// (a) transform_context threshold trigger: a big seeded session
        /// compacts during prompt(); the request the fake stream receives is
        /// the transcript message, not the full old prefix; after the run
        /// the session resyncs onto the compacted node; appends land on
        /// the compacted node.
        #[tokio::test]
        async fn transform_context_threshold_compacts_and_resyncs() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            // ~1100 tokens total with a 4000-char tool result in the prefix.
            seed_tool_turn(&store, &root, "c1", "read", json!({"path": "/a"}), 4000).await;
            seed_turns(&store, &root, 6, 100).await;

            let model = Model {
                id: "test".into(),
                context_window: 100,
                ..Model::unknown()
            };
            let settings = CompactionSettings {
                enabled: true,
                reserve_tokens: 50, // threshold: 100-50 = 50 → way above
                keep_recent_tokens: 60,
            };
            let record: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let (mut session, coordinator, llm_calls) =
                hooked_session(&dir, &store, &root, settings, &model, record.clone()).await;

            coordinator.begin_run();
            session.agent_mut().prompt("hello").await.unwrap();
            let resynced = coordinator.take_pending_resync();
            if let Some(id) = resynced {
                session.resume(&id).await.unwrap();
            }

            // The request the fake stream saw: transcript message, not the
            // old prefix. (Record locked in a scoped block so no guard
            // crosses the store awaits below.)
            {
                let calls = record.lock().unwrap();
                assert_eq!(calls.len(), 1, "one session request");
                assert!(
                    calls[0].contains(TRANSCRIPT_PREAMBLE),
                    "transcript in request: {}",
                    calls[0]
                );
                assert!(
                    !calls[0].contains(&"R".repeat(4000)),
                    "tool-result bulk must be gone from the request: {}",
                    calls[0]
                );
            }

            // Session resynced onto the compacted node: state = transcript + tail.
            let messages = &session.state().messages;
            assert!(
                first_text(&messages[0]).contains(TRANSCRIPT_PREAMBLE),
                "first message is the transcript"
            );
            assert!(
                messages
                    .iter()
                    .all(|m| !first_text(m).contains(&"R".repeat(4000))),
                "no tool-result bulk in state"
            );
            let last = messages.last().unwrap();
            assert_eq!(
                first_text(last),
                "ok",
                "the run's assistant landed in state"
            );

            // Appends landed on the compacted node: head moved to it.
            let head = store.head().await.unwrap().unwrap();
            assert_ne!(head, root, "head moved off root");
            let view = store.materialize_plain(&head).await.unwrap();
            let Segment::Summary(s) = &view[0] else {
                panic!("compacted node starts with a summary");
            };
            assert!(s.content.starts_with(TRANSCRIPT_PREAMBLE));
            assert!(*llm_calls.lock().unwrap() >= 1);
        }

        /// (b) no double compaction: threshold still exceeded after the
        /// compaction → only one compact node per run.
        #[tokio::test]
        async fn no_double_compaction_per_run() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            // Tool result in the compacted prefix; the retained tail (4
            // turns ≈ 100 tokens) still exceeds the threshold after.
            seed_tool_turn(&store, &root, "c1", "bash", json!({"command": "ls"}), 4000).await;
            seed_turns(&store, &root, 12, 100).await; // ~300 tokens

            let model = Model {
                id: "test".into(),
                context_window: 100,
                ..Model::unknown()
            };
            let settings = CompactionSettings {
                enabled: true,
                reserve_tokens: 50,      // threshold 50
                keep_recent_tokens: 100, // retained tail ~100 tokens > 50
            };
            let record: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let (mut session, coordinator, _) =
                hooked_session(&dir, &store, &root, settings, &model, record.clone()).await;

            coordinator.begin_run();
            session.agent_mut().prompt("hello").await.unwrap();
            if let Some(id) = coordinator.take_pending_resync() {
                session.resume(&id).await.unwrap();
            }

            // Exactly one compact derivation along the chain.
            let head = store.head().await.unwrap().unwrap();
            assert_ne!(head, root);
            let originals = store.fetch_originals(&head).await.unwrap();
            assert_eq!(originals.len(), 1, "one compaction: {originals:?}");
            let view = store.materialize_plain(&head).await.unwrap();
            let summaries = view
                .iter()
                .filter(|seg| matches!(seg, Segment::Summary(_)))
                .count();
            assert_eq!(summaries, 1, "exactly one summary segment");
        }

        /// context window triggers one compaction; a second overflow in the
        /// same run fails it (the run ends with an Error stop reason).
        #[tokio::test]
        async fn overflow_recovery_once_then_fails() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            // Small session — below threshold, so no pre-request compaction.
            seed_tool_turn(&store, &root, "c1", "bash", json!({"command": "ls"}), 4000).await;
            seed_turns(&store, &root, 2, 100).await;

            let model = Model {
                id: "test".into(),
                context_window: 100,
                ..Model::unknown()
            };
            let settings = CompactionSettings {
                enabled: true,
                reserve_tokens: 50,
                keep_recent_tokens: 60,
            };
            // Scripted session stream: turn 1 requests an unknown tool (so a
            // second turn happens) with a huge usage; turn 2 silently
            // overflows again.
            let calls = Arc::new(Mutex::new(0usize));
            let calls_for_stream = calls.clone();
            let session_stream = mock_stream_fn(move |_model, _ctx, _opts| {
                let mut count = calls_for_stream.lock().unwrap();
                let call = *count;
                *count += 1;
                let msg = if call == 0 {
                    AssistantMessage {
                        content: vec![ContentBlock::tool_call("c1", "absent_tool", json!({}))],
                        stop_reason: StopReason::ToolUse,
                        usage: Usage {
                            input: 10_000,
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                } else {
                    AssistantMessage {
                        content: vec![ContentBlock::text("big response")],
                        usage: Usage {
                            input: 10_000,
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                };
                Box::new(MockAssistantStream::new(msg))
            });
            let mut agent = Agent::new(session_stream);
            agent.state.model = model.clone();
            agent.state.system_prompt = "sys".to_string();
            agent.stream_fn_options.api_key = None;
            let mut session = AgentSession::new(store.clone(), agent, root.clone(), dir.path());
            session.resume(&root).await.unwrap();
            let coordinator = Arc::new(CompactionCoordinator::new(store.clone(), settings, &model));
            let node_handle: Arc<RwLock<String>> = session.session_id_lock();
            install_compaction_hooks(session.agent_mut(), &coordinator, node_handle);

            // One run: first overflow compacts, second overflow fails the run.
            coordinator.begin_run();
            session.agent_mut().prompt("hello").await.unwrap();
            assert_eq!(*calls.lock().unwrap(), 2, "two turns happened");
            let state = session.state();
            let (stop_reason, error_message) = turn_outcome(state);
            assert_eq!(stop_reason, Some(StopReason::Error));
            assert!(
                error_message
                    .as_deref()
                    .unwrap_or_default()
                    .contains("overflow"),
                "overflow surfaced: {error_message:?}"
            );
            // One compaction happened during the run.
            let head = store.head().await.unwrap().unwrap();
            assert_ne!(head, root);
            let view = store.materialize_plain(&head).await.unwrap();
            assert!(matches!(view[0], Segment::Summary(_)));
        }

        /// (d) disabled via settings: no auto compaction fires; the full
        /// context goes to the model and no compact node is created.
        #[tokio::test]
        async fn disabled_no_auto_compaction() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            seed_turns(&store, &root, 6, 200).await;

            let model = Model {
                id: "test".into(),
                context_window: 100,
                ..Model::unknown()
            };
            let settings = CompactionSettings {
                enabled: false,
                reserve_tokens: 50,
                keep_recent_tokens: 60,
            };
            let record: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
            let (mut session, coordinator, _) =
                hooked_session(&dir, &store, &root, settings, &model, record.clone()).await;

            coordinator.begin_run();
            session.agent_mut().prompt("hello").await.unwrap();

            // No pending resync, no compact node.
            assert!(coordinator.take_pending_resync().is_none());
            let head = store.head().await.unwrap().unwrap();
            assert_eq!(head, root, "head stays on root when disabled");
            // The request carried the full original prefix.
            let calls = record.lock().unwrap();
            assert!(calls[0].contains("u0-"), "full context sent: {}", calls[0]);
            assert!(
                !calls[0].contains(TRANSCRIPT_PREAMBLE),
                "no transcript injected"
            );
        }
    }

    /// `CompactionSettings` env parsing: defaults when unset, overrides
    /// parse, invalid values fall back to defaults. Exercises the
    /// pure `from_env_values` core — the workspace denies
    /// `unsafe_code`, so tests cannot mutate process env.
    #[test]
    fn from_env_defaults_overrides_and_fallbacks() {
        let defaults = CompactionSettings::from_env_values(None, None, None);
        assert!(defaults.enabled);
        assert_eq!(defaults.reserve_tokens, DEFAULT_RESERVE_TOKENS);
        assert_eq!(defaults.keep_recent_tokens, DEFAULT_KEEP_RECENT_TOKENS);

        let parsed =
            CompactionSettings::from_env_values(Some("false"), Some("4096"), Some("12345"));
        assert!(!parsed.enabled, "false disables");
        assert_eq!(parsed.reserve_tokens, 4096);
        assert_eq!(parsed.keep_recent_tokens, 12345);

        // Invalid values fall back to defaults; unrecognized strings
        // count as enabled.
        let fallback = CompactionSettings::from_env_values(Some("bogus"), Some("nan"), Some("-7"));
        assert!(fallback.enabled);
        assert_eq!(fallback.reserve_tokens, DEFAULT_RESERVE_TOKENS);
        assert_eq!(fallback.keep_recent_tokens, DEFAULT_KEEP_RECENT_TOKENS);

        // Disabled strings: "0", "false", "no" (case-insensitive).
        for v in ["0", "FALSE", "No"] {
            assert!(
                !CompactionSettings::from_env_values(Some(v), None, None).enabled,
                "{v}"
            );
        }
    }
}
