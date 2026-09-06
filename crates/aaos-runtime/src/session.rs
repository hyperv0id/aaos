//! Session assembly and the post-run resync: resolve the node to continue,
//! build the agent, bind it to the store, and resume its view — the shared
//! assembly sequence every frontend runs.
use std::path::PathBuf;
use std::sync::Arc;

use aaos_session::{AgentSession, SessionStore};
use pi_agent_core::types::{AgentState, StopReason};

use crate::compaction::{CompactionCoordinator, CompactionSettings};
use crate::model::{AgentConfig, EnvConfig, build_agent};

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

/// Build the session for both entry modes: open the store, resolve the node
/// to continue, build the agent, bind it via `AgentSession::new` (MessageEnd
/// → append_segment listener) and `resume` its view into `state.messages`
/// (replacing it, with dangling tool-call repair). `cwd` and the skills
/// directory come from `config` — the runtime never reads `std::env`.
pub async fn create_session(config: &SessionConfig) -> Result<AgentSession, String> {
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
    session
        .resume(&session_id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(session)
}

/// Resync the session's in-memory view after a run if a compaction committed
/// mid-run (auto or manual): `take_pending_resync` yields the compacted node.
pub async fn resync_after_run(
    session: &mut AgentSession,
    coordinator: &Arc<CompactionCoordinator>,
) -> Result<(), String> {
    if let Some(id) = coordinator.take_pending_resync() {
        session.resume(&id).await.map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Resolve the outcome of a finished turn: the last assistant message's stop
/// reason, plus its error message falling back to the session-level error.
pub fn turn_outcome(state: &AgentState) -> (Option<StopReason>, Option<String>) {
    let last = state.messages.iter().rev().find_map(|m| m.as_assistant());
    let stop_reason = last.map(|m| m.stop_reason);
    let error_message = last
        .and_then(|m| m.error_message.clone())
        .or_else(|| state.error_message.clone());
    (stop_reason, error_message)
}

#[cfg(test)]
mod tests {
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
}
