# Pi Agent Core Equivalence Fixes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close 9 behavioral gaps between the local Rust pi-agent-core and the upstream Pi kernel, achieving true behavioral trace-equivalence.

**Architecture:** Targeted fixes to existing crate `crates/pi-agent-core` — agent_loop, agent, tool_engine, types, and acceptance tests. No new crates, no real providers.

**Tech Stack:** Rust 1.80+, tokio, futures, async-trait, serde_json.

**Spec:** `/tmp/aaos-pi-rust-replica-handoff.md` (gap analysis) and `.scratch/pi-rust-replica-research-corrected.md` (corrected research conclusions).

## Global Constraints

- Kernel only: no real HTTP providers, no Node/Bun, no model registry, no schema validator, no session persistence.
- Behavioral/trace equivalence: same deterministic input sequence must produce the same normalized event order, message roles, stop reasons, tool invocation behavior, and lifecycle barriers.
- `agent_loop` / `agent_loop_continue` must never throw outside a normal event lifecycle; provider/runtime failures become assistant messages with `stop_reason: Error` or `Aborted` and a complete `agent_end` event.
- Abort signal must propagate to `StreamFn`, hooks, tool execution, and event listeners.
- Tool execution default is parallel; any tool with `execution_mode: Sequential` forces the whole batch to sequential.
- `length`-truncated assistant messages must not execute any tool calls; emit error tool results and continue.
- `beforeToolCall` runs after validation; `afterToolCall` replaces fields field-by-field with no deep merge.
- Steering messages inject after the current assistant turn's tool calls complete; follow-up messages run only when the agent would otherwise stop.
- Queue drain modes: `"all"` drains every queued message at the drain point; `"one_at_a_time"` drains only the oldest.
- No placeholders, no speculative abstractions. Lean code; prefer stdlib/`tokio`/`futures` over new crates.
- Read and apply `~/.agents/skills/rust-best-practices/SKILL.md` in all implementations.

---

## File Structure

| File | Responsibility |
|------|----------------|
| `crates/pi-agent-core/src/types.rs` | Type definitions, hook signatures, `StreamFnOptions` |
| `crates/pi-agent-core/src/agent_loop.rs` | `agent_loop`, `agent_loop_continue`, inner/outer loops, queue draining, lifecycle events, `AgentRun` |
| `crates/pi-agent-core/src/agent.rs` | `Agent` wrapper, `prompt`/`continue`/`steer`/`follow_up`/`abort`/`reset`/`wait_for_idle`/`subscribe` |
| `crates/pi-agent-core/src/tool_engine.rs` | Tool argument preparation, validation, `beforeToolCall`/`afterToolCall`, parallel/sequential execution |
| `crates/pi-agent-core/src/stream.rs` | `StreamFn` seam, fake provider, `MockAssistantStream` |
| `crates/pi-agent-core/tests/acceptance.rs` | Acceptance tests covering all behavioral contracts |

---

## Task 1: Steering Initial Poll and Continue Skip Semantics

**Files:**
- Modify: `crates/pi-agent-core/src/agent_loop.rs`
- Modify: `crates/pi-agent-core/src/agent.rs`
- Modify: `crates/pi-agent-core/tests/acceptance.rs`

**Interfaces:**
- Consumes: `AgentLoopConfig.get_steering_messages`, `run_loop` internals.
- Produces: Corrected steering poll order: initial poll before first assistant response; `skip_initial_steering_poll` skips that first poll.

- [ ] **Step 1: Write failing acceptance test**

Update the existing `steering_queue_one_at_a_time` and `steering_queue_all` tests to assert that steering messages queued before `prompt()` are injected before the first assistant response (not after). The tests should verify that with 2 steering messages + prompt, the steering messages appear before the first assistant `MessageEnd` in the event trace.

Add a new test `continue_with_steering_skips_initial_poll` that calls `continue_run()` from an assistant tail with two steering messages queued, asserting both are processed within the same run (not deferred).

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p pi-agent-core --test acceptance steering -- --nocapture`
Expected: FAIL — steering messages appear after first assistant response, not before.

- [ ] **Step 3: Fix run_loop initial steering poll**

In `run_loop`, change the initial `pending_messages` from `Vec::new()` to poll `config.get_steering_messages` before the first loop iteration. The `skip_initial_steering_poll` flag (carried via `AtomicBool` in the steering hook closure) should skip this initial poll when true.

In `Agent::create_loop_config`, the `skip` flag semantics: when `skip_initial_steering_poll` is true, the first call to the steering hook returns empty; subsequent calls drain normally. This is already the current behavior, but the hook is only called after the first turn ends — move the call to before the inner while loop's first iteration.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p pi-agent-core --test acceptance steering -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run full crate tests**

Run: `cargo test -p pi-agent-core`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/pi-agent-core/src/agent_loop.rs crates/pi-agent-core/src/agent.rs crates/pi-agent-core/tests/acceptance.rs
git commit -m "fix(pi-agent-core): steering initial poll before first turn"
```

---

## Task 2: Abort Propagation to Hooks, Tool Preparation, and Loop

**Files:**
- Modify: `crates/pi-agent-core/src/tool_engine.rs`
- Modify: `crates/pi-agent-core/src/agent_loop.rs`
- Modify: `crates/pi-agent-core/tests/acceptance.rs`

**Interfaces:**
- Consumes: `watch::Receiver<bool>` abort signal, `AgentLoopConfig` hooks.
- Produces: Abort checks at: before each tool preparation, before each tool execution in sequential mode, before each tool preparation in parallel mode, and after provider returns in the loop.

- [ ] **Step 1: Write failing acceptance test**

Add `abort_stops_sequential_tool_batch` test: a sequential batch of 3 tool calls where abort fires after the first completes. Assert that the 2nd and 3rd tool calls do not execute (no `ToolExecutionEnd` events for them), and the run terminates with an `agent_end`.

Add `abort_checked_before_tool_preparation` test: abort fires while tools are being prepared in parallel mode. Assert no `ToolExecutionStart` events for tools not yet prepared.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p pi-agent-core --test acceptance abort -- --nocapture`
Expected: FAIL — remaining tools execute after abort.

- [ ] **Step 3: Add abort checks to tool engine**

In `execute_sequential`: check `*signal.borrow()` before each iteration; if aborted, break and return an `ExecutedToolBatch` with only the completed messages.

In `execute_parallel`: check `*signal.borrow()` before each `prepare_tool_call` in the preparation loop; skip remaining preparations if aborted.

In `prepare_tool_call`: check abort before calling `before_tool_call` hook; if aborted, return an immediate error result "aborted".

- [ ] **Step 4: Add abort check in run_loop**

After `stream_assistant_response` returns, check `*abort.borrow()`; if aborted and the assistant message is not already `Error`/`Aborted`, emit `TurnEnd` and `AgentEnd` and return.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p pi-agent-core --test acceptance abort -- --nocapture`
Expected: PASS.

- [ ] **Step 6: Run full crate tests**

Run: `cargo test -p pi-agent-core`
Expected: All tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/pi-agent-core/src/tool_engine.rs crates/pi-agent-core/src/agent_loop.rs crates/pi-agent-core/tests/acceptance.rs
git commit -m "fix(pi-agent-core): propagate abort to hooks, tool prep, and loop"
```

---

## Task 3: Streaming Partial Message Updates

**Files:**
- Modify: `crates/pi-agent-core/src/agent_loop.rs`
- Modify: `crates/pi-agent-core/tests/acceptance.rs`

**Interfaces:**
- Consumes: `AssistantMessageEvent` delta events.
- Produces: `MessageUpdate` events carrying the latest partial `AssistantMessage`, not a stale copy from the `Start` event.

- [ ] **Step 1: Write failing acceptance test**

Add `streaming_updates_carry_latest_partial` test: use a `MockAssistantStream` that emits `Start { partial: "H" }`, then `TextDelta { "i" }`, then `TextDelta { "!" }`, then `Done`. Assert that the `MessageUpdate` events carry the accumulated text ("Hi", "Hi!") not just the initial partial.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p pi-agent-core --test acceptance streaming_updates -- --nocapture`
Expected: FAIL — updates carry stale partial.

- [ ] **Step 3: Fix partial message accumulation**

In `stream_assistant_response`, when handling delta events (`TextDelta`, `ThinkingDelta`, `ToolCallDelta`), update `partial_message` by appending the delta to the appropriate content block before emitting `MessageUpdate`. For `TextStart`/`TextEnd`/`ThinkingStart`/etc., update the partial message's content structure accordingly.

The key fix: `partial_message` must be mutated on each delta event, not just read from the `Start` event's snapshot.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p pi-agent-core --test acceptance streaming_updates -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Run full crate tests**

Run: `cargo test -p pi-agent-core`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/pi-agent-core/src/agent_loop.rs crates/pi-agent-core/tests/acceptance.rs
git commit -m "fix(pi-agent-core): streaming updates carry latest partial message"
```

---

## Task 4: Agent Wrapper Fixes — Continue Validation, Listener Unsubscribe, Shared Wait Barrier, Unbounded Event Channel

**Files:**
- Modify: `crates/pi-agent-core/src/agent.rs`
- Modify: `crates/pi-agent-core/src/agent_loop.rs`
- Modify: `crates/pi-agent-core/tests/acceptance.rs`

**Interfaces:**
- Consumes: `AgentRun`, `Agent` internals.
- Produces: Early continue validation, real listener removal, shared idle barrier, unbounded event buffer.

- [ ] **Step 1: Write failing acceptance tests**

`continue_run_empty_transcript_does_not_pollute_state`: call `continue_run()` on empty transcript; assert `is_streaming == false` and `active_run == None` after the panic is caught.

`listener_unsubscribe_removes_listener`: subscribe a listener, unsubscribe it, run a prompt, assert the unsubscribed listener received zero events.

`wait_for_idle_multiple_waiters`: spawn two tasks both calling `wait_for_idle()`; assert both complete after the run finishes.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p pi-agent-core --test acceptance continue_run_empty listener_unsubscribe wait_for_idle_multiple -- --nocapture`
Expected: FAIL.

- [ ] **Step 3: Fix continue_run early validation**

In `Agent::continue_run`, before setting `is_streaming` or `active_run`, validate: if messages empty, panic immediately; if last is assistant and no steering/follow-up, panic immediately. Move the checks from `agent_loop_continue` (which runs inside the spawned task) to `continue_run` (which runs before spawning).

- [ ] **Step 4: Implement real listener unsubscribe**

Change `subscribe` to return a `ListenerHandle` that holds a `Vec<Listener>` reference (via `Arc<Mutex<Vec<Listener>>>` or index-based removal). When dropped or called, remove the listener from the vector.

Change `Agent.listeners` from `Vec<Listener>` to `Arc<Mutex<Vec<Listener>>>` so handles can remove themselves.

- [ ] **Step 5: Implement shared wait_for_idle barrier**

Replace `oneshot::Receiver<()>` in `ActiveRun` with a `watch::Receiver<bool>` (idle flag). Multiple `wait_for_idle` callers can subscribe and await the same watch channel. `finish_run` sends `true` on the channel.

- [ ] **Step 6: Make event channel unbounded**

Change `mpsc::channel::<AgentEvent>(256)` to `mpsc::unbounded_channel::<AgentEvent>()` in `create_agent_stream`. Update `AgentRun` to use `mpsc::UnboundedReceiver`.

- [ ] **Step 7: Run tests to verify they pass**

Run: `cargo test -p pi-agent-core --test acceptance continue_run_empty listener_unsubscribe wait_for_idle_multiple -- --nocapture`
Expected: PASS.

- [ ] **Step 8: Run full crate tests**

Run: `cargo test -p pi-agent-core`
Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add crates/pi-agent-core/src/agent.rs crates/pi-agent-core/src/agent_loop.rs crates/pi-agent-core/tests/acceptance.rs
git commit -m "fix(pi-agent-core): continue validation, listener unsubscribe, shared idle barrier, unbounded events"
```

---

## Task 5: Thinking Level in Provider Options and Hook Error Conversion

**Files:**
- Modify: `crates/pi-agent-core/src/types.rs`
- Modify: `crates/pi-agent-core/src/agent_loop.rs`
- Modify: `crates/pi-agent-core/src/tool_engine.rs`
- Modify: `crates/pi-agent-core/tests/acceptance.rs`

**Interfaces:**
- Consumes: `AgentLoopConfig.thinking_level`, `StreamFnOptions`.
- Produces: `StreamFnOptions.thinking_level` field; hook errors converted to error lifecycle instead of silently ignored; error tool result `details` defaults to `json!({})`.

- [ ] **Step 1: Write failing acceptance tests**

`thinking_level_passed_to_provider`: assert `StreamFnOptions` received by the fake provider includes the configured `thinking_level`.

`hook_error_converts_to_error_lifecycle`: a `should_stop_after_turn` hook that returns `Err("hook failed")`; assert the run converts to an error assistant message lifecycle with the hook error message.

`error_tool_result_details_is_object`: assert that error tool results have `details == json!({})` not `json!(null)`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p pi-agent-core --test acceptance thinking_level hook_error error_tool_result_details -- --nocapture`
Expected: FAIL.

- [ ] **Step 3: Add thinking_level to StreamFnOptions**

Add `pub thinking_level: Option<ThinkingLevel>` to `StreamFnOptions`. In `stream_assistant_response`, populate it from `config.thinking_level` before calling `stream_fn.call`.

- [ ] **Step 4: Convert hook errors to error lifecycle**

In `run_loop`, when `prepare_next_turn` or `should_stop_after_turn` hooks return `Err`, convert to an assistant `StopReason::Error` message with the error message, emit `TurnEnd` and `AgentEnd`, and return. When `get_steering_messages` or `get_follow_up_messages` hooks return `Err`, treat as empty (don't halt the run for queue hooks — upstream behavior is to silently skip).

- [ ] **Step 5: Fix error tool result details default**

In `create_error_tool_result`, set `details` to `serde_json::json!({})` instead of `Value::Null` (the `Default` for `Value`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p pi-agent-core --test acceptance thinking_level hook_error error_tool_result_details -- --nocapture`
Expected: PASS.

- [ ] **Step 7: Run full crate tests**

Run: `cargo test -p pi-agent-core`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add crates/pi-agent-core/src/types.rs crates/pi-agent-core/src/agent_loop.rs crates/pi-agent-core/src/tool_engine.rs crates/pi-agent-core/tests/acceptance.rs
git commit -m "fix(pi-agent-core): thinking level in provider options, hook error conversion, error details default"
```

---

## Self-Review Checklist

- [ ] Spec coverage: every gap in the handoff analysis maps to at least one task.
- [ ] Placeholder scan: no "TBD", "TODO", "implement later", or open-ended "handle edge cases" steps.
- [ ] Type consistency: `AgentEvent`, `Message`, `AgentToolResult`, `ToolCall`, `AgentLoopConfig` names match across tasks.
- [ ] Trace-equivalence: acceptance tests compare normalized event order/roles/IDs, not timestamps.
- [ ] Testability: every task has a runnable test command.
