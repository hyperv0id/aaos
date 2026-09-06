//! Event contract (design doc §3): the runtime routes everything a frontend
//! observes through a single [`EventSink::on_event`] method. Agent events are
//! wrapped in [`SessionEvent::Agent`] by the forwarding listener installed at
//! assembly time; compaction hook failures surface as
//! [`SessionEvent::CompactionFailed`]. No forwarding channel/queue is built —
//! the sink is called synchronously at the point where each event arises
//! (agent events through the drain loop's fan-out, compaction failures
//! directly at the hook failure site).

use pi_agent_core::types::AgentEvent;

/// Session-level event union: kernel agent events plus runtime extension
/// events. Owned (`'static`) so a future frontend can move events into its
/// own consumer channel without a signature change.
#[expect(
    clippy::large_enum_variant,
    reason = "the Agent variant carries the kernel's own owned AgentEvent by design (design doc §3.2); boxing it would defeat the move-into-channel contract"
)]
#[derive(Debug)]
pub enum SessionEvent {
    /// A kernel agent event (the drain loop's per-listener clone, moved in).
    Agent(AgentEvent),
    /// A compaction hook failed; the error text is what the frontend
    /// renders (the CLI prints it verbatim to stderr).
    CompactionFailed { error: String },
}

/// The frontend-injected event router: one method receives every event the
/// runtime produces. Implementations forward each variant to their own
/// rendering path.
pub trait EventSink: Send + Sync {
    fn on_event(&self, event: SessionEvent);
}

/// A sink that drops every event: for tests and disabled-event scenarios.
pub struct NoopSink;

impl EventSink for NoopSink {
    fn on_event(&self, _event: SessionEvent) {}
}
