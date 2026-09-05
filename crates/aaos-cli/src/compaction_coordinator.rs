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

use std::sync::Mutex;

use aaos_session::compaction::{
    DEFAULT_KEEP_RECENT_TOKENS, DEFAULT_RESERVE_TOKENS, build_transcript, context_tokens,
    find_cut_point, should_compact, view_messages,
};
use aaos_session::{Segment, SessionStore, SummarySegment};
use pi_agent_core::types::{
    AgentContext, AssistantMessage, Message, Model, StopReason, Usage, UserMessage,
};

/// Compaction settings, read once from the environment at coordinator
/// construction. Manual `/compact` ignores `enabled`; the auto hooks check it.
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
    /// Parse `AAOS_COMPACTION_ENABLED`, `AAOS_COMPACTION_RESERVE_TOKENS`,
    /// `AAOS_COMPACTION_KEEP_RECENT_TOKENS`. Disabled strings: "0", "false",
    /// "no" (case-insensitive); numeric parses fall back to defaults.
    pub fn from_env() -> Self {
        Self::from_env_values(
            std::env::var("AAOS_COMPACTION_ENABLED").ok().as_deref(),
            std::env::var("AAOS_COMPACTION_RESERVE_TOKENS")
                .ok()
                .as_deref(),
            std::env::var("AAOS_COMPACTION_KEEP_RECENT_TOKENS")
                .ok()
                .as_deref(),
        )
    }

    /// Pure parse of the three env values (as `Option<&str>`, `None` = unset)
    /// onto a [`CompactionSettings`]. Testable without mutating process env —
    /// the workspace denies `unsafe_code`, so tests cannot set env vars.
    pub(crate) fn from_env_values(
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
