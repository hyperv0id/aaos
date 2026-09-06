//! Session-bound orchestration pieces shared by every frontend: the
//! post-run resync and the turn-outcome extraction.

use aaos_session::AgentSession;
use pi_agent_core::types::{AgentState, StopReason};

use crate::compaction::CompactionCoordinator;
use std::sync::Arc;

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
