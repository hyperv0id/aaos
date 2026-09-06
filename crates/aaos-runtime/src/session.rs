//! Session assembly and the run primitives every frontend shares: resolve
//! the node to continue, build the agent, bind it to the store, resume its
//! view, then drive turns (`run_turn`) and manual compactions
//! (`compact_now`) through a [`SessionHandle`].
use std::path::PathBuf;
use std::sync::Arc;

use aaos_session::{AgentSession, SessionStore};
use pi_agent_core::types::{AgentState, StopReason};

use crate::compaction::{
    CompactionCoordinator, CompactionError, CompactionOutcome, CompactionSettings,
    install_compaction_hooks,
};
use crate::event::{EventSink, SessionEvent};
use crate::model::{AgentConfig, EnvConfig, build_agent, build_coordinator};
use pi_agent_core::agent::AgentHandle;

/// Session decision config: which node to continue and whether to fork
/// (from the CLI's `--session`/`--fork`, or a TUI's own session picker).
#[derive(Debug, Clone, Default)]
pub struct RuntimeConfig {
    pub session_id: Option<String>,
    pub fork: bool,
}

/// Compaction settings carrier for the session config (see
/// `aaos_runtime::compaction`).
#[derive(Debug, Clone, Copy)]
pub struct CompactionConfig {
    pub settings: CompactionSettings,
}

/// Aggregate runtime configuration: session decisions, agent assembly, host
/// environment adaptation, compaction settings, and the explicit working
/// directories (no `std::env` reads inside the runtime).
#[derive(Clone)]
pub struct SessionConfig {
    pub runtime: RuntimeConfig,
    pub agent: AgentConfig,
    pub env: EnvConfig,
    pub compaction: CompactionConfig,
    pub cwd: PathBuf,
    pub user_skills_dir: PathBuf,
}

/// Resolve the session node to continue. An explicit session id wins and
/// resumes that node in place (`fork` derives a new session from it
/// instead); an unknown id errors rather than silently starting a session
/// nothing points at. The default continues the user's head session — the persisted
/// head pointer (the node last appended to; `latest_created_session` for
/// stores that predate the pointer) — as a fresh derivation: the derivation
/// inherits the full view, while each process appends to its own node, so n
/// concurrent runs never cross-write one session. An empty store gets a
/// fresh root.
pub async fn resolve_session_node(
    store: &SessionStore,
    cfg: &RuntimeConfig,
) -> Result<String, String> {
    if let Some(id) = cfg.session_id.as_deref() {
        if cfg.fork {
            return store.fork(id).await.map_err(|e| e.to_string());
        }
        if !store.session_exists(id).await.map_err(|e| e.to_string())? {
            return Err(format!("session not found: {id}"));
        }
        return Ok(id.to_string());
    }
    let target = match store.head().await.map_err(|e| e.to_string())? {
        Some(id) => id,
        None => match store
            .latest_created_session()
            .await
            .map_err(|e| e.to_string())?
        {
            Some(id) => id,
            None => return store.create_root().await.map_err(|e| e.to_string()),
        },
    };
    store.fork(&target).await.map_err(|e| e.to_string())
}

/// Assemble a session for both entry modes: open the store, resolve the node
/// to continue, build the agent, bind it via `AgentSession::new` (MessageEnd
/// → append_segment listener) and `resume` its view into `state.messages`
/// (replacing it, with dangling tool-call repair). `cwd` and the skills
/// directory come from `config` — the runtime never reads `std::env`.
///
/// The returned [`SessionHandle`] is fully wired: `sink` is installed as a
/// regular agent listener at assembly time (every kernel event is moved into
/// [`SessionEvent::Agent`] and handed to `sink.on_event` — the drain loop's
/// fan-out, synchronous, no channel), the compaction coordinator is built
/// from `config.compaction.settings` and the resolved live model, and the
/// auto-trigger hooks are installed on the agent. Callers must pass the sink
/// before any prompt runs (agent events are only emitted from `prompt`), so
/// no event is lost.
pub async fn create_session(
    config: &SessionConfig,
    sink: Arc<dyn EventSink>,
) -> Result<SessionHandle, String> {
    let store = SessionStore::open(&config.env.paths.config_dir)
        .await
        .map_err(|e| e.to_string())?;
    let session_id = resolve_session_node(&store, &config.runtime).await?;
    let mut session = AgentSession::new(
        store,
        build_agent(
            &config.agent,
            &config.env,
            &config.cwd,
            &config.user_skills_dir,
        )
        .await?,
        &session_id,
        config.cwd.clone(),
    );
    let listener_sink = sink.clone();
    let _ = session.agent().subscribe(Arc::new(move |event, _signal| {
        let sink = listener_sink.clone();
        Box::pin(async move {
            sink.on_event(SessionEvent::Agent(event));
        })
    }));
    session
        .resume(&session_id)
        .await
        .map_err(|e| e.to_string())?;
    let coordinator = build_coordinator(
        &session.agent().state.model,
        session.store(),
        config.compaction.settings,
        sink.clone(),
    );
    let node_handle = session.session_id_lock();
    install_compaction_hooks(session.agent_mut(), &coordinator, node_handle);
    Ok(SessionHandle {
        session,
        coordinator,
    })
}

/// The result of one turn (`SessionHandle::run_turn`).
pub struct TurnOutcome {
    /// The last assistant message's stop reason.
    pub stop_reason: Option<StopReason>,
    /// Its error message, falling back to the session-level error.
    pub error_message: Option<String>,
    /// True when a compaction committed during this turn (auto, or a manual
    /// `compact_now` that ran into the turn's window) and the view resynced.
    pub compacted_this_run: bool,
}

/// A wired, runnable session: the bound agent session and its compaction
/// coordinator. The frontend's event sink outlives the handle through its
/// own clones (the agent listener and the coordinator each hold one).
/// Frontends drive turns and manual compactions through it; events arrive
/// only through the sink, never through the accessor paths.
pub struct SessionHandle {
    session: AgentSession,
    coordinator: Arc<CompactionCoordinator>,
}

impl SessionHandle {
    /// Read-only session view. Events never travel through this path —
    /// they are routed through the sink.
    pub fn session(&self) -> &AgentSession {
        &self.session
    }

    /// The kernel's concurrency handle (steer/abort), taken by `&self` so a
    /// frontend can clone and call it from another task while a `run_turn`
    /// holds the exclusive borrow (steer/abort stay concurrent with a
    /// pending prompt).
    pub fn handle(&self) -> AgentHandle {
        self.session.agent().handle()
    }

    /// Drive one turn — the sequence every frontend shares: reset per-run
    /// state (`begin_run`), prompt, resync the in-memory view if a
    /// compaction committed mid-run, and resolve the turn outcome.
    pub async fn run_turn(&mut self, prompt: &str) -> Result<TurnOutcome, String> {
        self.coordinator.begin_run();
        self.session
            .agent_mut()
            .prompt(prompt)
            .await
            .map_err(|e| e.to_string())?;
        if let Some(id) = self.coordinator.take_pending_resync() {
            self.session.resume(&id).await.map_err(|e| e.to_string())?;
        }
        let mut outcome = TurnOutcome::from_state(self.session.state());
        outcome.compacted_this_run = self.coordinator.compacted_this_run();
        Ok(outcome)
    }

    /// Manual compaction (the REPL `/compact` primitive): compact the
    /// current node, switch onto the compacted node, and consume the
    /// pending resync the coordinator recorded so the next turn's post-run
    /// resync doesn't re-resume the same node. The caller renders the
    /// outcome (`Compacted into X (a → b tokens)` is frontend copy).
    ///
    /// If the post-compaction resume fails, the pending resync the
    /// coordinator recorded is left set but is inert: the next turn's
    /// `begin_run` clears it before any resync, so the post-run resync
    /// never lands on the compacted node — the in-memory view stays as-is
    /// and the failure surfaces to the caller verbatim.
    pub async fn compact_now(&mut self) -> Result<CompactionOutcome, CompactionError> {
        let current_id = self.session.current_session_id().await;
        let outcome = self.coordinator.compact(&current_id).await?;
        if let Err(e) = self.session.resume(&outcome.compacted_id).await {
            return Err(CompactionError::Failed(format!(
                "resume onto {} failed: {e}",
                outcome.compacted_id
            )));
        }
        let _ = self.coordinator.take_pending_resync();
        Ok(outcome)
    }

    /// Whether this session persisted at least one segment (side-effect
    /// records are not counted) — the REPL save-notice's basis.
    pub fn has_persisted(&self) -> bool {
        self.session.has_persisted_segments()
    }

    /// The current session node id (the next append target).
    pub async fn current_session_id(&self) -> String {
        self.session.current_session_id().await
    }
}

impl TurnOutcome {
    /// Resolve a finished turn's stop reason and error message from the
    /// agent state: the last assistant message's stop reason, plus its
    /// error message falling back to the session-level error.
    /// `compacted_this_run` is left `false` — the caller fills it from the
    /// coordinator's per-run flag. Shared by [`SessionHandle`] and the
    /// compaction hooks tests.
    pub(crate) fn from_state(state: &AgentState) -> Self {
        let last = state.messages.iter().rev().find_map(|m| m.as_assistant());
        let stop_reason = last.map(|m| m.stop_reason);
        let error_message = last
            .and_then(|m| m.error_message.clone())
            .or_else(|| state.error_message.clone());
        Self {
            stop_reason,
            error_message,
            compacted_this_run: false,
        }
    }
}

#[cfg(test)]
mod tests {
    // Store plumbing unwraps and struct-update shorthands are the test
    // idiom; production paths stay panic-free.
    #![expect(clippy::needless_update)]
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use aaos_session::{Segment, SessionStore};

    use super::RuntimeConfig;
    use super::resolve_session_node;

    /// The `resolve_session_node` decision rules; names state one rule each.
    mod resolve_session {
        use super::*;

        /// Issue #61: the default run continues the head session as a fresh
        /// derivation — its own node, the head's full view — and the head only
        /// moves when something is actually appended.
        #[tokio::test]
        async fn default_derives_own_line() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            store
                .append_segment(&root, &Segment::user_text("q"))
                .await
                .unwrap();

            let resolved = resolve_session_node(&store, &RuntimeConfig::default())
                .await
                .unwrap();
            assert_ne!(
                resolved, root,
                "the default run continues on its own session"
            );
            assert_eq!(
                store.materialize_plain(&resolved).await.unwrap(),
                vec![Segment::user_text("q")],
                "the derivation inherits the head's view"
            );
            assert_eq!(
                store.head().await.unwrap().as_deref(),
                Some(root.as_str()),
                "head follows appends, not derivations"
            );
        }

        #[tokio::test]
        async fn explicit_session_resumes_in_place() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            store
                .append_segment(&root, &Segment::user_text("q"))
                .await
                .unwrap();

            let cfg = RuntimeConfig {
                session_id: Some(root.clone()),
                ..Default::default()
            };
            assert_eq!(
                resolve_session_node(&store, &cfg).await.unwrap(),
                root,
                "--session resumes the node itself"
            );

            let forked = resolve_session_node(
                &store,
                &RuntimeConfig {
                    session_id: Some(root.clone()),
                    fork: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            assert_ne!(forked, root);
            assert_eq!(
                store.materialize_plain(&forked).await.unwrap(),
                vec![Segment::user_text("q")]
            );
        }

        /// `--session` must fail loudly on an unknown node instead of silently
        /// starting from a session nothing points at; the `--fork` path is checked
        /// by the store's own lookup.
        #[tokio::test]
        async fn unknown_id_errors() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();

            let err = resolve_session_node(
                &store,
                &RuntimeConfig {
                    session_id: Some("nope".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
            assert!(err.contains("nope"), "{err}");

            let err = resolve_session_node(
                &store,
                &RuntimeConfig {
                    session_id: Some("nope".into()),
                    fork: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
            assert!(err.contains("nope"), "{err}");
        }

        #[tokio::test]
        async fn empty_store_creates_root() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();

            let root = resolve_session_node(&store, &RuntimeConfig::default())
                .await
                .unwrap();
            assert!(store.materialize_plain(&root).await.unwrap().is_empty());
            assert_eq!(store.head().await.unwrap(), None, "no appends, no head");
        }

        /// A store written before the head pointer existed: the fallback picks
        /// the newest created session and derives from it.
        #[tokio::test]
        async fn legacy_store_still_resumes() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            let child = store.fork(&root).await.unwrap();
            assert_eq!(store.head().await.unwrap(), None);

            let resolved = resolve_session_node(&store, &RuntimeConfig::default())
                .await
                .unwrap();
            assert_ne!(resolved, child);
            assert!(store.materialize_plain(&resolved).await.unwrap().is_empty());
        }
    }
    /// SessionHandle behavior tests: the new public run primitives
    /// (`run_turn`, `compact_now`) and their contract with the compaction
    /// coordinator's per-run state. Fixtures follow the compaction module's
    /// test idioms (mock stream, tempfile store, NoopSink).
    mod session_handle {
        use std::sync::Arc;

        use crate::session::SessionHandle;
        use aaos_session::compaction::TRANSCRIPT_PREAMBLE;
        use aaos_session::{AgentSession, SessionStore};
        use pi_agent_core::agent::Agent;
        use pi_agent_core::stream::{MockAssistantStream, mock_stream_fn};
        use pi_agent_core::types::{AssistantMessage, Model, StopReason};

        use crate::compaction::{
            CompactionCoordinator, CompactionSettings, install_compaction_hooks,
        };
        use crate::event::NoopSink;
        use crate::test_support::{first_text, seed_tool_turn, seed_turns};

        /// A wired SessionHandle over a seeded session: fake stream answering
        /// "ok", auto-compaction hooks installed, coordinator over the same
        /// store — the same assembly create_session performs, with the model
        /// and stream swapped for mocks.
        async fn handle_on(
            dir: &tempfile::TempDir,
            store: &SessionStore,
            root: &str,
            settings: CompactionSettings,
            model: &Model,
        ) -> SessionHandle {
            let stream_fn = mock_stream_fn(|_model, _ctx, _opts| {
                Box::new(MockAssistantStream::new(AssistantMessage::text("ok")))
            });
            let mut agent = Agent::new(stream_fn);
            agent.state.model = model.clone();
            agent.state.system_prompt = "sys".to_string();
            agent.stream_fn_options.api_key = None;
            let mut session = AgentSession::new(store.clone(), agent, root.to_string(), dir.path());
            session.resume(root).await.unwrap();
            let coordinator = Arc::new(CompactionCoordinator::new(
                store.clone(),
                settings,
                model,
                Arc::new(NoopSink),
            ));
            let node_handle = session.session_id_lock();
            install_compaction_hooks(session.agent_mut(), &coordinator, node_handle);
            SessionHandle {
                session,
                coordinator,
            }
        }

        /// One-shot semantics: a single `run_turn` drives the prompt to a
        /// clean stop, persists the turn on the session's own node, and
        /// reports no compaction for a plain turn.
        #[tokio::test]
        async fn run_turn_one_shot_persists_and_reports() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            let mut handle = handle_on(
                &dir,
                &store,
                &root,
                CompactionSettings {
                    enabled: false,
                    ..Default::default()
                },
                &Model::unknown(),
            )
            .await;

            let outcome = handle.run_turn("hello").await.unwrap();
            assert_eq!(outcome.stop_reason, Some(StopReason::Stop));
            assert_eq!(outcome.error_message, None);
            assert!(!outcome.compacted_this_run, "plain turn, no compaction");
            assert!(handle.has_persisted(), "the turn persisted");

            let node = handle.current_session_id().await;
            assert_eq!(node, root, "appends land on the bound node");
            let view = store.materialize_plain(&node).await.unwrap();
            assert_eq!(view.len(), 2, "user + assistant segments persisted");

            let messages = &handle.session().state().messages;
            let last = messages.last().unwrap();
            assert_eq!(first_text(last), "ok", "the assistant landed in state");
        }

        /// A REPL loop reuses the same primitive for every turn: two
        /// consecutive `run_turn`s on one handle each run cleanly and append.
        #[tokio::test]
        async fn repl_reuses_run_turn_across_turns() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            let mut handle = handle_on(
                &dir,
                &store,
                &root,
                CompactionSettings {
                    enabled: false,
                    ..Default::default()
                },
                &Model::unknown(),
            )
            .await;

            for i in 0..2 {
                let outcome = handle.run_turn(&format!("turn {i}")).await.unwrap();
                assert_eq!(outcome.stop_reason, Some(StopReason::Stop), "turn {i}");
                assert_eq!(outcome.error_message, None, "turn {i}");
            }
            let view = store.materialize_plain(&root).await.unwrap();
            assert_eq!(view.len(), 4, "two user + two assistant segments");
        }

        /// Manual compact (§9.1 risk 3): `compact_now` compacts, switches
        /// onto the compacted node, and consumes the pending resync — so the
        /// following `run_turn` does not re-resume, and exactly one
        /// compaction exists along the chain.
        #[tokio::test]
        async fn compact_now_then_run_turn_does_not_double_resume() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            seed_tool_turn(
                &store,
                &root,
                "c1",
                "read",
                serde_json::json!({"path": "/a"}),
                4000,
            )
            .await;
            seed_turns(&store, &root, 5, 100).await;
            // enabled: false — the manual path ignores it, and the auto hooks
            // stay inert so the only compaction is the manual one.
            let mut handle = handle_on(
                &dir,
                &store,
                &root,
                CompactionSettings {
                    enabled: false,
                    reserve_tokens: 50,
                    keep_recent_tokens: 60,
                },
                &Model::unknown(),
            )
            .await;

            let outcome = handle.compact_now().await.expect("manual compact ok");
            assert_ne!(outcome.compacted_id, root);
            assert!(
                outcome.before_tokens > outcome.after_tokens,
                "{} -> {}",
                outcome.before_tokens,
                outcome.after_tokens
            );
            // The view switched onto the compacted node and the pending
            // resync was consumed: nothing left for the next turn to re-do.
            let messages = &handle.session().state().messages;
            assert!(
                first_text(&messages[0]).starts_with(TRANSCRIPT_PREAMBLE),
                "state starts with the transcript"
            );
            assert!(
                messages.iter().all(|m| !first_text(m).contains("RRRR")),
                "tool-result bulk is gone from state"
            );
            assert_eq!(
                handle.current_session_id().await,
                outcome.compacted_id,
                "appends now target the compacted node"
            );

            let next = handle.run_turn("after compact").await.unwrap();
            assert_eq!(next.stop_reason, Some(StopReason::Stop));
            assert!(!next.compacted_this_run, "new run, no compaction fired");

            // Exactly one compaction derivation along the chain.
            let head = store.head().await.unwrap().unwrap();
            assert_eq!(head, outcome.compacted_id, "the turn appended to it");
            let originals = store.fetch_originals(&head).await.unwrap();
            assert_eq!(originals.len(), 1, "one compaction: {originals:?}");
        }

        /// Auto-compaction during `run_turn` sets `compacted_this_run` and
        /// the post-run resync lands the view on the compacted node.
        #[tokio::test]
        async fn run_turn_reports_compacted_this_run() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            // ~1100 tokens total with a 4000-char tool result in the prefix;
            // window 100 / reserve 50 → the threshold triggers pre-request.
            seed_tool_turn(
                &store,
                &root,
                "c1",
                "read",
                serde_json::json!({"path": "/a"}),
                4000,
            )
            .await;
            seed_turns(&store, &root, 6, 100).await;
            let model = Model {
                id: "test".into(),
                context_window: 100,
                ..Model::unknown()
            };
            let mut handle = handle_on(
                &dir,
                &store,
                &root,
                CompactionSettings {
                    enabled: true,
                    reserve_tokens: 50,
                    keep_recent_tokens: 60,
                },
                &model,
            )
            .await;

            let outcome = handle.run_turn("hello").await.unwrap();
            assert_eq!(outcome.stop_reason, Some(StopReason::Stop));
            assert!(outcome.compacted_this_run, "auto compaction committed");

            let messages = &handle.session().state().messages;
            assert!(
                first_text(&messages[0]).starts_with(TRANSCRIPT_PREAMBLE),
                "post-run resync landed on the compacted node"
            );
            let last = messages.last().unwrap();
            assert_eq!(first_text(last), "ok", "the run's assistant is in state");
            let head = store.head().await.unwrap().unwrap();
            assert_ne!(head, root, "head moved to the compacted node");
        }
    }
}
