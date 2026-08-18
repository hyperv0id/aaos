# Task 7 Report: prompt/continue_run → Result + cleanup sweep

## Status: COMPLETE

## Changes

### 1. AgentError enum + Result returns (agent.rs)

Added `pub enum AgentError` in `agent.rs` with four variants mapping verbatim to upstream error strings:
- `AlreadyProcessing` — `"Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion."`
- `AlreadyProcessingContinue` — `"Agent is already processing. Wait for completion before continuing."`
- `NoMessagesToContinueFrom` — `"No messages to continue from"`
- `CannotContinueFromAssistant` — `"Cannot continue from message role: assistant"`

`impl fmt::Display` + `impl std::error::Error` (crate-local, no thiserror — String convention).

Signature changes:
- `pub async fn prompt(&mut self, input: impl Into<PromptInput>) -> Result<(), AgentError>` — re-entrancy returns `Err(AgentError::AlreadyProcessing)` instead of panicking.
- `pub async fn continue_run(&mut self) -> Result<(), AgentError>` — re-entrancy returns `Err(AgentError::AlreadyProcessingContinue)`; empty transcript returns `Err(AgentError::NoMessagesToContinueFrom)`; assistant-tail with no queued messages returns `Err(AgentError::CannotContinueFromAssistant)`.
- `reset()` KEEPS panic (upstream throws; existing `reset_rejects_during_run` test uses task-panic/JoinError pattern). Ledger note: this mirrors upstream's `throw new Error(...)` — Rust panic is the closest equivalent for a synchronous void method that the existing tests already cover via `catch_unwind`/JoinError.

The internal `run_prompt_messages` / `run_continuation` methods are unchanged — they return `()` and handle run errors through the lifecycle event path (not Result), matching upstream's `runWithLifecycle` catch → `handleRunFailure`.

### 2. Deferred cleanup items

#### Task 6(a): Share thinking_level_to_option
- Made `thinking_level_to_option` `pub(crate)` in `agent_loop.rs`.
- `Agent::thinking_level()` now calls `thinking_level_to_option(self.state.thinking_level)` instead of inline `match`.
- Added `thinking_level_to_option` to the `use crate::agent_loop::{...}` import in `agent.rs`.

#### Task 6(b): Rewrite update_after_settle_is_dropped
- Rewrote the test to actually invoke the callback post-settle using the Mutex-stash approach.
- `LateUpdateTool` holds `Arc<Mutex<Option<AgentToolUpdateCallback>>>`. In `execute`, the callback is called once normally ("during" — delivered), then stashed into the Mutex.
- After `prompt().await.unwrap()`, the test retrieves the stashed callback and calls it with "late".
- Asserts `update_count == 1` — the late call was dropped by the `accepting_updates` gate (now false after settle).
- Removed the unused `update_after_settle: Arc<AtomicBool>` field.

#### Task 6(c): Drop dead Notify
- Removed `use tokio::sync::Notify;` from `update_emitted_during_execution_delivered_before_end`.
- Removed `release: Arc<Notify>` field from `UpdatingTool`.
- Removed `self.release.notify_one()` call (nobody awaited it).
- Removed `let release = Arc::new(Notify::new());`.
- Updated tool construction to `UpdatingTool {}`.

#### Task 4: Fix stale doc comments
- `AgentRun::result()` doc: `Returns the spawn [JoinError]...` → `Returns [LoopError] — either a hook rejection or the wrapped [tokio::task::JoinError]...`
- `trace::Collector::collect_run()` doc: `Returns the spawn [tokio::task::JoinError]...` → `Returns [crate::agent_loop::LoopError] — either a hook rejection or the wrapped [tokio::task::JoinError]...`

### 3. Test updates

#### agent.rs tests
- `prompt_rejects_while_streaming`: changed from spawning a task and asserting `handle.await.is_err()` (task panic) to calling `agent.prompt("hi").await` directly and asserting `result == Err(AgentError::AlreadyProcessing)`. No panic.
- `reset_rejects_during_run`: UNCHANGED — still uses task-panic/JoinError (reset keeps panic).
- Two `agent.prompt("hi").await;` calls in `full_lifecycle_emits_events` and the streaming events test: added `.unwrap()`.

#### acceptance.rs tests
- All 36 `agent.prompt("...").await;` / `guard.prompt("...").await;` call sites: added `.unwrap()` (they expect success).
- `continue_run_empty_transcript_does_not_pollute_state`: rewrote from spawning a task + asserting JoinError to calling `agent.continue_run().await` directly and asserting `result.is_err()`. Simplified from `Arc<tokio::sync::Mutex<Agent>>` to direct `&mut Agent` (no concurrency needed).
- `agent.continue_run().await;` (steering_queue_all test, line 762): added `.unwrap()`.

## Test summary

```
cargo test -p pi-agent-core
  lib: 32 passed; 0 failed; 0 ignored; 0 warnings
  acceptance: 40 passed; 0 failed; 0 ignored; 0 warnings
  doc: 0 passed; 0 failed
Total: 72 passed, 0 warnings — pristine
```

## Concerns

1. **reset() keeps panic**: Upstream `reset()` throws; the Rust port panics (existing test pattern). This is a deliberate divergence from the Result pattern applied to prompt/continue_run. Rationale: `reset()` is synchronous and void in upstream; a panic mirrors `throw` for Rust callers. If a future refactor moves reset to `Result<(), AgentError>`, the `reset_rejects_during_run` test must be updated to assert `Err` instead of task-panic/JoinError.
2. **AgentError variant naming**: `AlreadyProcessing` vs `AlreadyProcessingContinue` — upstream uses slightly different messages for prompt vs continue re-entrancy. Two variants capture this distinction. An alternative would be a single `AlreadyProcessing` variant with the continue-specific message, but that would lose the verbatim upstream string match. Current approach: separate variants, each with its exact upstream string.
3. **No new test for continue_run re-entrancy**: The brief asks for a new test asserting prompt returns `Err(AgentError::AlreadyProcessing)`. The existing `prompt_rejects_while_streaming` test now does this. No equivalent test for `continue_run` re-entrancy was added (the empty-transcript test covers `NoMessagesToContinueFrom`). Adding a `continue_run` re-entrancy test would require simulating an active run state, which the existing `prompt_rejects_while_streaming` pattern already demonstrates.
