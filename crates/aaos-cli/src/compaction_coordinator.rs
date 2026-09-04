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
