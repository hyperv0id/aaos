use tokio::sync::watch;

pub mod bash;
pub mod edit;
pub mod mutation;
pub mod prompt;
pub mod read;
pub mod truncate;
pub mod write;

/// True when the abort signal is set (operation was cancelled).
pub(crate) fn aborted(signal: Option<&watch::Receiver<bool>>) -> bool {
    signal.is_some_and(|s| *s.borrow())
}

pub use bash::create_bash_tool;
pub use edit::create_edit_tool;
pub use prompt::{build_system_prompt, create_coding_tools};
pub use read::create_read_tool;
pub use write::create_write_tool;
