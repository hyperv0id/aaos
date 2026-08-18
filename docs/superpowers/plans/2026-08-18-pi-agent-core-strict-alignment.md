# Pi Agent Core Strict Alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close all 15 gaps (6 P1, 9 P2) from the 2026-08-18 gap analysis so `crates/pi-agent-core` is strictly aligned with upstream Pi `packages/agent` (earendil-works/pi @ main, sparse clone at `/tmp/pi-upstream`). Alignment is the only requirement: every behavioral decision resolves to upstream source, not to Rust idiom preference. Known upstream defects are replicated as-is and recorded in the ledger, never "fixed" locally.

**Architecture:** Targeted changes to existing crate `crates/pi-agent-core` — types.rs, agent.rs, agent_loop.rs, tool_engine.rs, stream.rs, tests/acceptance.rs. No new crates. Upstream reference files: `packages/agent/src/agent.ts` (606 ln), `agent-loop.ts` (805 ln), `types.ts` (443 ln), `packages/agent/test/agent.test.ts`, `agent-loop.test.ts`.

**Tech Stack:** Rust 1.80+, tokio, futures, async-trait, serde_json.

**Spec (binding authority):** upstream sources at `/tmp/pi-upstream/packages/agent/src/*.ts` and `/tmp/pi-upstream/packages/agent/test/*.ts`, plus `packages/ai/src/types.ts` (Model, AssistantMessageEvent). The gap analysis in `.superpowers/sdd/pi-agent-core-strict-alignment/gap-analysis.md` motivates but does not override upstream source.

## Global Constraints

- Strict alignment with upstream Pi (`/tmp/pi-upstream`): same event ordering, same lifecycle barriers, same hook contracts, same state semantics. When upstream behavior is arguably deficient, replicate it anyway and record a ledger note — discussion is deferred.
- Hooks (beforeToolCall/afterToolCall/shouldStopAfterTurn/prepareNextTurn/transformContext) receive the run's abort signal, matching upstream signatures that take `signal?: AbortSignal`.
- All test evidence from `cargo test -p pi-agent-core` (full suite green, output pristine).
- Rust API ergonomics (Result vs panic, Send bounds) follow the gap analysis's Rust-specific rulings (see Task 7), not blind JS transliteration.
- Subagents use `skill://rust-best-practices` during implementation.
- No placeholders; no speculative abstractions beyond what alignment requires.

## Task Dependency Graph

| Task | Depends On | Write Paths |
|---|---|---|
| 1. Model struct replaces String model | — | types.rs, agent_loop.rs, agent.rs, stream.rs, tests |
| 2. Upstream stream event protocol | — | types.rs, stream.rs, agent_loop.rs, tests |
| 3. Hook signals + before-hook arg mutation + post-hook abort re-check | — | types.rs, tool_engine.rs, agent.rs, tests |
| 4. Loop-level hook error bubbling + convertToLlm async + steering hook errors | Task 2 (event enum shape) | types.rs, agent_loop.rs, agent.rs, tests |
| 5. Run-scoped abort + concurrent steering/followUp API | — | agent.rs, tests |
| 6. agent_loop_continue validation + addedToolNames + Off→None unification + listener dedup + update-callback race | Task 2 (ToolCall shape from stream protocol) | types.rs, agent_loop.rs, agent.rs, tool_engine.rs, tests |
| 7. prompt/continue → Result; error taxonomy; whole-branch review + cleanup | Tasks 1-6 | agent.rs, types.rs, tests |

---

## Task 1: Model struct replaces String model (P2 #1)

**Files:** Modify `crates/pi-agent-core/src/types.rs`, `agent_loop.rs`, `agent.rs`, `stream.rs`, `tests/acceptance.rs`.

**Upstream authority:** `types.ts` Model interface (id, name, api, provider, baseUrl, reasoning, thinkingLevelMap?, input, cost, contextWindow, maxTokens, samplingParams?, headers?); `agent.ts` DEFAULT_MODEL; `AgentLoopTurnUpdate.model?: Model<any>`.

**Change:**
- Add `pub struct Model { pub id: String, pub name: String, pub api: String, pub provider: String, pub base_url: String, pub reasoning: bool, pub input: Vec<ModelInput>, pub cost: ModelCost, pub context_window: u64, pub max_tokens: u64 }` with `ModelInput { enum Text, Image }` (serialized names "text"/"image"), `ModelCost { input, output, cache_read, cache_write: f64, total: f64 }`.
- `Model::unknown()` mirrors upstream DEFAULT_MODEL (id/name/api/provider "unknown", empty base_url, reasoning false, empty input, zero cost, zero windows).
- `AgentState.model: Model`; `AgentState::default()` uses `Model::unknown()`.
- `AgentLoopConfig.model: Model`; `AgentLoopTurnUpdate.model: Option<Model>`.
- `StreamFn::call(model: Model, ...)` takes Model by value.
- Update all constructions/call sites/tests. AssistantMessage.model stays String (upstream AssistantMessage has `model: string` — check `/tmp/pi-upstream/packages/ai/src/types.ts`).
- [ ] Acceptance test: thinking/model metadata reach StreamFn — assert the StreamFn receives a Model whose id/provider match configured state.

## Task 2: Upstream stream event protocol (P2 #2)

**Files:** Modify `types.rs`, `stream.rs`, `agent_loop.rs`, `tests/acceptance.rs`.

**Upstream authority:** `packages/ai/src/types.ts` `AssistantMessageEvent` union:
```
| { type: "start"; partial: AssistantMessage }
| { type: "text_start"; contentIndex: number; partial: AssistantMessage }
| { type: "text_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
| { type: "text_end"; contentIndex: number; content: string; partial: AssistantMessage }
| { type: "thinking_start" | "thinking_delta" | "thinking_end"; ...; partial }
| { type: "toolcall_start"; contentIndex; partial }
| { type: "toolcall_delta"; contentIndex; delta; partial }
| { type: "toolcall_end"; contentIndex; toolCall: ToolCall; partial }
| { type: "done"; reason: "stop"|"length"|"toolUse"|"deferred"; message: AssistantMessage }
| { type: "error"; reason: "aborted"|"error"; error: AssistantMessage }
```

**Change:**
- Reshape `AssistantMessageEvent` enum to carry `content_index: usize`, `partial: AssistantMessage` on every incremental event; `ToolCallStart` additionally carries `tool_call: ToolCall` upstream — NO: upstream `toolcall_start` has NO toolCall field, only `toolcall_end` does. Follow upstream exactly: `ToolCallStart { content_index, partial }`, `ToolCallDelta { content_index, delta, partial }`, `ToolCallEnd { content_index, tool_call: ToolCall, partial }`, `TextEnd/ThinkingEnd { content_index, content: String, partial }`, `Done { reason: StopReason-lite, message }`, `Error { reason, error: AssistantMessage }`.
- The loop's `stream_assistant_response` consumes partial from the event (upstream sets `partialMessage = event.partial` on each event) instead of accumulating deltas locally; remove local accumulation (append_text_delta/append_thinking_delta, tool_call_args buffer). The `Start` push of partial into context.messages is kept.
- `MockAssistantStream` / `mock_stream_fn` updated for the new shape. All existing tests reprogrammed.
- [ ] Acceptance test: streamed tool call Start(id/name empty)-Delta-End(full ToolCall) builds the message via partial updates; message_update events carry the partial from each event.

## Task 3: Hook signals + before-hook arg mutation + post-hook abort re-check (P1 #2,#3,#4)

**Files:** Modify `types.rs`, `tool_engine.rs`, `agent.rs`, `tests/abort.rs` (new).

**Upstream authority:**
- `agent-loop.ts` prepareToolCall: before-hook receives `(ctx, signal)`; after the hook resolves, `if (signal?.aborted) return immediate "Operation aborted"`; block path returns error result with reason; after the block check ANOTHER `if (signal?.aborted)` before returning prepared.
- `agent-loop.ts` executeToolCallsSequential/Parallel: `tool_execution_start` emitted BEFORE prepareToolCall (no abort check before start in upstream — start events always emit for every call in toolCalls even when aborted mid-batch? NO — upstream emits start unconditionally per call, abort breaks AFTER each finalized call, and prepareToolCall returns "Operation aborted" immediate results when aborted). Match this exactly: emit start, then prepare (which aborts to "Operation aborted" immediate result), execute, finalize, emit end, THEN check `signal?.aborted` to break.

Wait — check upstream sequential: the loop body is `await emit(start); preparation = await prepareToolCall(...); ... await emitToolExecutionEnd; ...; if (signal?.aborted) break;`. Yes: start always emitted; break checked after each call completes. Parallel: same — start emitted, prepare, push, `if (signal?.aborted) break;` after push.
- beforeToolCall args: upstream passes the SAME validated args object into the hook ctx and into execute (`preparation.args = validatedArgs`), so in-place mutation by the hook mutates what executes. Rust rendering: `BeforeToolCallContext.args` stays the validated Value; the hook returns args via `BeforeToolCallResult` — NO, upstream returns no args from the hook; mutation is in-place JS. Rust cannot mutate a captured Value in-place through an Arc closure. Ruling: add `args_override: Option<Value>` to `BeforeToolCallResult` (hook returns modified args; upstream tests mutate in place, Rust returns explicitly — the observable contract "hook-modified args are executed without revalidation" is identical). Note: upstream executes mutated args without revalidation; Rust must NOT revalidate the override.
- afterToolCall merge: field-by-field `??` — Rust already does unwrap_or; verify is_error handling matches (`afterResult.isError ?? isError`).
- Hook type changes: `BeforeToolCallHook = Arc<dyn Fn(BeforeToolCallContext, watch::Receiver<bool>) -> BoxFuture<...>>` etc. — all five hook types take signal. Listener type already takes signal.
- [ ] Acceptance tests: (a) before-hook returns args_override → tool executes overridden args, no revalidation; (b) before-hook triggers abort → "Operation aborted" error result, tool not executed; (c) hooks receive signal that reflects abort mid-run.

## Task 4: Loop-level hook errors bubble + async convertToLlm + steering hook errors (P1 #5, P2 #6)

**Files:** Modify `types.rs`, `agent_loop.rs`, `agent.rs`, `tests/acceptance.rs`.

**Upstream authority:**
- `buildProviderContext` (agent-loop.ts): `messages = await config.transformContext(messages, signal)` — no try/catch; a rejection bubbles out of streamAssistantResponse → runLoop → runAgentLoop → Agent.runWithLifecycle catch → handleRunFailure → error assistant message lifecycle. Same for `convertToLlm` (`await config.convertToLlm(messages)`).
- `runLoop` initial/turn steering poll: `(await config.getSteeringMessages?.()) || []` — hook rejection bubbles to runWithLifecycle. Follow-up poll: same.
- Wait — upstream runLoop lines: `let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];` — yes, no catch. Rejection propagates. Rust: initial poll `hook().await?` propagating a loop error.
- BUT the low-level `agentLoop()`/`agentLoopContinue()` (EventStream-returning) are `void runAgentLoop(...).then(...)` — a rejection there never calls `stream.end()`. Hmm — upstream contract: the run rejects, EventStream stays open? Actually `runAgentLoop` rejection: `void (...).then(messages => stream.end(messages))` — no `.catch`, so rejection is an unhandled rejection and the EventStream never ends. The Agent wrapper catches it via runWithLifecycle. For Rust: agent_loop/agent_loop_continue are spawned tasks; a hook error becomes `Err` from the loop → the spawned task returns Err... but AgentRun resolves `Result<Vec<Message>, JoinError>`. Match the OBSERVABLE Agent-level behavior: loop hook errors produce the error assistant lifecycle via Agent.runWithLifecycle's catch (already exists for panics; extend to returned errors). For the low-level API: agent_loop returns AgentRun whose result becomes an error — replicate upstream by having the loop task emit the error lifecycle itself? NO — upstream low-level does NOT emit anything on hook rejection; the stream just never ends. That's a defect. Ruling: low-level Rust agent_loop surfaces hook errors by emitting the error assistant lifecycle (agent must never hang); note the deviation with rationale: upstream low-level stream hangs (unhandled rejection), Rust cannot hang a spawned task's consumer; the Agent-level behavior (error lifecycle via runWithLifecycle) is identical either way. Record ledger note.
- Async convertToLlm: `ConvertToLlm = Arc<dyn Fn(Vec<Message>) -> BoxFuture<'static, Vec<Message>>>` (upstream may return Promise; errors bubble — but upstream doc says convertToLlm "must not throw or reject" contract; a rejection still bubbles through buildProviderContext. Rust: return Result<Vec<Message>, String>? Upstream signature returns `Message[] | Promise<Message[]>`, NOT Result. Hook rejection = error path. Rust: `BoxFuture<'static, Result<Vec<Message>, String>>` aligning with TransformContext.)

Hmm wait — upstream contract explicitly says convertToLlm must not throw; if it does, bubbles to error lifecycle. Rust: Result with Err → error lifecycle. Aligned.

- [ ] Acceptance tests: transform_context Err → error lifecycle (no next LLM call); convert_to_llm Err → error lifecycle; steering hook Err → error lifecycle.

## Task 5: Run-scoped abort + concurrent steering/followUp (P1 #1)

**Files:** Modify `agent.rs`, `tests/acceptance.rs`.

**Upstream authority:** `agent.ts` runWithLifecycle: fresh `AbortController` per run; `activeRun.abortController`; `abort()` only affects activeRun; finishRun clears activeRun so old signals go dead. steer/followUp enqueue on the queue; the run drains via hooks — callable while prompt() promise is pending.

**Change:**
- `ActiveRun { abort_tx: watch::Sender<bool>, idle: watch::Receiver<bool> }` — fresh channel per run; `Agent::abort()` sends on active run's channel if any (no-op otherwise); `abort_handle()` returns handle bound to the active run's sender (None when idle → handle whose abort() no-ops). `signal()` subscribes to active run.
- prompt/steer/follow_up concurrent API: steer/follow_up remain `&mut self` but the run must not hold the borrow across await. Restructure: `prompt(&mut self, input)` spawns the drain task and returns... JS `prompt()` returns a Promise awaited by caller. Rust ruling (gap analysis P1#1): make steering injectable DURING run — keep `prompt(&mut self).await` but move run state behind `Arc<Mutex<RunState>>` so steer/follow_up can be called from another task holding a cloned Arc. Minimal: Agent fields `steering_queue`/`follow_up_queue` are already Arc<Mutex<...>> (PendingMessageQueue.inner). steer(&mut self) only touches the queue — the problem is only that `prompt().await` holds `&mut Agent`. Solution: internal `run_shared: Arc<Mutex<Agent>>`-style split is invasive. Cleaner: `Agent::prompt` takes `&mut self`, spawns the loop, and awaits a future that does NOT borrow self (drain via moved Arc clones); caller can then use `Arc<Mutex<Agent>>` themselves. Actually the API already supports concurrency via the existing Arc<Mutex<Agent>> pattern IF prompt doesn't hold the mutex — it currently does (self.drain_run_events(&mut run...) borrows self). Fix: refactor drain_run_events to operate on Arc-cloned listener/state pieces so prompt's future borrows only a `&mut AgentState`-equivalent behind its own Arc<Mutex>. 

Design: move `state` handling into `process_event(&self.state-arc, event)`. Concretely: `Agent { inner: Arc<Mutex<AgentInner>> }`? That's a big refactor. Alternative accepted-upstream-shape: keep `Agent::prompt(&mut self)` semantics (awaits run to completion like JS await) and add the concurrency seam at the queue level: steer/follow_up already only need `&self` (queue is Arc<Mutex>) — change `steer(&self, ...)`/`follow_up(&self, ...)`/`has_queued_messages(&self)`/queue-clear methods to `&self` receivers. Then `Arc<Mutex<Agent>>` + prompt(&mut) still blocks... The user with `Arc<tokio::sync::Mutex<Agent>>` calls `lock().await.prompt()` — holds the lock. Gap analysis says this is THE core gap. Real fix: split Agent into `Agent { shared: Arc<AgentShared> }` where AgentShared holds queues+listeners+abort+state behind individual Mutexes, prompt(&mut self) spawns+drains via shared clone, and steer(&self) works without the outer mutex. prompt keeps `&mut self` (re-entrancy guard via active_run flag on shared), steering works concurrently from any clone of `AgentHandle`/`&Agent`.

Concrete API: `Agent` stays the owner struct. `pub fn handle(&self) -> AgentHandle` returns a cheap cloneable handle with `steer`, `follow_up`, `abort`, `has_queued_messages`, `wait_for_idle`, `signal`. `Agent::prompt` internals reworked to not hold exclusive borrow during the await: state mutations happen through `Arc<Mutex<AgentState>>`. This matches upstream observable behavior (steer callable mid-run) while keeping Rust ownership sane.
- [ ] Acceptance test: mid-run steering — spawn prompt, wait for tool execution start (via Notify), steer via handle, assert steering message injected before next LLM call (visible in transcript/events); prompt completes normally.

## Task 6: agent_loop_continue validation + addedToolNames + Off→None + listener dedup + update-callback race (P1 #6, P2 #4,#7,#8,#9)

**Files:** Modify `types.rs`, `agent_loop.rs`, `agent.rs`, `tool_engine.rs`, `tests/acceptance.rs`.

**Upstream authority:**
- agent-loop.ts agentLoopContinue: throws "Cannot continue: no messages in context" / "Cannot continue from message role: assistant" at CALL time (synchronous throw). Rust: `agent_loop_continue` returns `Result<AgentRun, ContinueError>` — upstream throws synchronously before creating the stream; Rust returns Err synchronously. Agent::continue_run keeps its own drain-first logic (agent.ts continue()).
- addedToolNames: `AgentToolResult.added_tool_names: Option<Vec<String>>`; `createToolResultMessage` copies `...(finalized.result.addedToolNames?.length ? { addedToolNames } : {})` — Rust: copy `Option<Vec<String>>` into ToolResultMessage.added_tool_names when Some-and-nonempty; field stays Option (upstream omits the key when absent).
- Off→None: runLoop prepareNextTurn merge: `reasoning: nextTurnSnapshot.thinkingLevel === undefined ? config.reasoning : nextTurnSnapshot.thinkingLevel === "off" ? undefined : nextTurnSnapshot.thinkingLevel` — Rust: `thinking_level = if tl == Off { None } else { Some(tl) }` in the dynamic path, matching the initial path.
- Listener dedup: upstream `Set<Listener>` — same function registered twice fires once. Rust Vec<Listener> with Arc::ptr_eq dedup on subscribe.
- Update-callback race: upstream `acceptingUpdates` boolean + pushed promises awaited after execute settles; late calls ignored. Rust current: AtomicBool + tokio::spawn + JoinHandle list — the race: check-then-spawn gap. Upstream has NO such gap because the callback runs synchronously on the same thread (promise push). Rust: eliminate the race by NOT spawning — make the update callback push the event into the same ordered event flow: since the emit sink is an unbounded mpsc clone, callback can `let _ = tx.send(event)` synchronously? The EventSink is `Arc<dyn Fn(AgentEvent) -> BoxFuture>` — for updates we need fire-and-forget. Approach: keep tokio::spawn but register the handle BEFORE re-checking accepting flag... cleanest race-free: callback does `if !accepting.load() return; spawn; store handle under mutex` and execute_prepared does `accepting.store(false); take handles; await all` — the race window is a callback that passed the check but hasn't pushed the handle when take() runs. Fix: after store(false), loop-poll? No — restructure: callback pushes into a `Mutex<Vec<AgentEvent>>`-plus-Notify? Simplest correct: callback checks accepting; if set, pushes event DIRECTLY into an unbounded channel owned by execute_prepared; after execute settles, store(false), then drain the channel fully (all events that passed the check are in the channel; any later call sees false and drops). Awaiting settlement = draining channel (sync, no spawn) then emitting in order via emit sink. No JoinHandles, no race.
- [ ] Acceptance tests: continue with empty context / assistant tail → Err variants; tool result carries added_tool_names when set; prepare_next_turn Off → provider sees None; duplicate listener registered twice fires once; update emitted before tool end is delivered, update after settle dropped.

## Task 7: prompt/continue → Result + final review (P2 #10)

**Files:** Modify `agent.rs`, `types.rs`, `tests/acceptance.rs`.

**Upstream authority:** agent.ts prompt()/continue() return rejected promises on re-entrancy ("Agent is already processing...") and validation errors ("No messages to continue from"). Rust: `pub async fn prompt(&mut self, ...) -> Result<(), AgentError>` and `continue_run(&mut self) -> Result<(), AgentError>`; `AgentError` enum: `AlreadyProcessing`, `NoMessagesToContinueFrom`, `CannotContinueFromAssistant` (agent.ts messages), plus the loop's internal error path stays as lifecycle events (not Result errors). reset() panic → Result-taking? upstream throws — Rust: keep panic for reset (mirrors throw; callers in tests use catch_unwind/JoinError) OR return Result. Gap analysis: "如果目标是 Rust API 的惯用行为，应该改为 Result" — apply to prompt/continue_run; reset stays panic (upstream throws Error; Rust panic in async = task failure, acceptable and already tested). Actually for strict alignment AND Rust idiom: make reset() also return Result<(), AgentError> — hmm, upstream reset throws only while processing. Ruling: prompt/continue_run → Result; reset → panic preserved (existing test `reset_rejects_during_run` uses task panic), ledger note.

- [ ] Update all existing tests/call sites (agent.rs tests, acceptance.rs) for Result returns.
- This task ends with the whole-branch final review + cleanup pass.

## Self-Review Checklist

- [ ] All 15 gaps from the analysis have a corresponding task row and test.
- [ ] Every task cites its upstream authority lines.
- [ ] No new dependencies; tokio/futures/async-trait/serde_json only.
- [ ] Full suite green: `cargo test -p pi-agent-core` pristine output.
