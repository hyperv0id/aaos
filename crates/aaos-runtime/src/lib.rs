//! aaos-runtime: the shared runtime layer under frontends — assembles a
//! runnable session from configuration, drives one prompt into a persisted,
//! compactable, observable agent run, and hands results and events back to
//! the frontend for rendering.
//!
//! Dependency direction: this crate depends on `aaos-session`,
//! `aaos-providers`, `aaos-tools`, and `pi-agent-core`; frontends (the CLI,
//! a future TUI) depend on this crate. It knows nothing about any frontend:
//! no terminal protocol, no product defaults, no process lifecycle.
pub mod compaction;
pub mod event;
pub mod model;
pub mod session;

#[cfg(test)]
mod test_support;
