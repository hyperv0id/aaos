# Session Storage Implementation Plan

> **⚠ SUPERSEDED（2026-08-22）**：本计划的结构层（追加日志链 + HEAD 引用 + framing/torn-tail）已被 [ADR-0001](../../adr/0001-sqlite-structural-source-of-truth.md) 翻转为 SQLite 事实源 + 统一派生模型。资产层（对象库 / Segment / canonical JSON）原样保留；链式语义（链序即优先级、区间映射、sources 出处）被 ADR 继承。词汇表见根 `CONTEXT.md`。

> **For agentic workers:** Use subagent-driven-development or executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 内容寻址对象库 + 分叉日志链 + 每会话 HEAD 引用。压缩 = compact fork（区间→摘要映射 + HEAD 迁移），快照/回滚/resume 走 HEAD。Standalone crate, no kernel dependency.

**Tech Stack:** Rust 1.80+, tokio, blake3, serde + serde_json, thiserror, tempfile (dev).

**Spec:** `SESSION-STORAGE.md` (reproduced below). **Out of scope:** GC（本次不做，只留可达性边界注记）、压缩策略执行器（已定：LLM 生成摘要，Pi 做法；落在 kernel/session 侧，见边界 3）、视图缓存、`AgentSession` 接线。

## Reproduced Spec: SESSION-STORAGE.md

```markdown
# Agent 会话的存储结构设计（aaos）

> 配套 DESIGN.md。只到架构层，只陈述最终设计。

## 一、设计

会话存储 = **内容寻址对象库 + 分叉日志链 + 每会话 HEAD 引用**。

- 对象库存内容：按哈希寻址，写一次、全局一份；压缩后原文仍在。
- 日志链存结构：一个分支一个追加文件；分叉/压缩 = 新文件引用父日志位置，零复制。
- HEAD 存"现在在哪"：每会话一个原子小文件（日志, 位置）；快照/回滚/resume 即读写 HEAD。

## 二、模型

- **段**：最小语义单元（消息、工具结果、摘要）。
- **追加**：日志尾部加段引用记录。
- **分叉**：子代理共享父前缀，各自追加。
- **压缩 = 分叉**：新日志（kind=compact）+ 区间→摘要映射记录；视图 = 父视图替换区间；压缩主线时 HEAD 迁移到新日志。父日志不动，撤销压缩 = HEAD 指回父日志。
- **摘要带出处**：摘要对象记录被压缩段的哈希列表；原文永远可寻址。
- **视图**：沿引用链递归重放；链序即优先级，无补丁式冲突规则。

## 三、结构

### 1. 对象库（内容层）
- 每段内容按哈希寻址，写一次、不可变、全局一份。
- 物理只追加。崩溃恢复 = 丢弃不完整尾记录。
- 压缩不删原文：摘要是新对象（含出处），原文继续可寻址。

### 2. 日志链（结构层）
- 一个分支一个线性追加文件。记录：头（root/subagent/compact + 父引用 + 继承 seq + 时间戳）、段引用、压缩映射、副作用。
- 分叉/压缩 = 新文件 + 头部引用（父日志, 位置）。多级分叉自然成链。
- 压缩映射：`{start, end, summary_hash}`，区间为半开 `[start, end)`，相对父视图；同槽后写覆盖先写；连续同摘要槽合并为一段。
- 副作用 seq 会话级单调，fork/compact 经头部继承。
- 一致性：记录级整写、单写者追加、尾记录检测。

### 3. 视图（呈现层）
- `view(root)` = 重放段引用；`view(fork)` = `view(parent@pos)` + 本日志；`view(compact)` = `view(parent@pos)` 应用映射 + 本日志。
- 缓存约束：日志不可变，前缀缓存永远有效（缓存本身本次不做）。

### 4. HEAD / 引用（会话层）
- `sessions/<id>/HEAD` = `{log, position}`，tmp+rename 原子写。
- HEAD 在建会话、主线压缩、显式快照/checkout 时更新；分叉（子代理）与普通追加不动 HEAD。
- 崩溃恢复 = 打开日志时截断撕裂尾（与回滚分开）；resume = 回到 HEAD 检查点（故意丢弃检查点后的记录）。

### 5. 副作用日志
- 工具动作 before/after 内容进对象库，记录进日志流，seq 会话级连续；恢复时重放或截断（undo/redo 本次不做）。

## 四、复杂度

| 操作 | 成本 |
|---|---|
| 追加段 | O(1) 日志记录 + O(段大小) 写对象 |
| 分叉 / 压缩 | O(1) 新日志 + k 条映射 + 摘要对象写 |
| 回滚 / 快照 | O(1) 指针；视图重放 O(活动段数 + 链长) |
| 视图材料化 | O(活动段数 + 链长) |
| 取回原文 | O(1)（经摘要 sources） |

## 五、边界
1. 不引入数据库服务。
2. GC 本次不做。可达性界（未来）：全部会话 HEAD 闭包 ∪ 摘要 sources 闭包。
3. 压缩策略：**调用 LLM 生成摘要**（同 Pi 做法——上下文压力或手动触发，保留最近若干段，其余由模型摘要）——由 kernel/session 层实现；本层只提供 `compact` + 出处，`SummarySegment.content` 即模型输出，`model` 记录生成方。
4. 回滚低于子分支的父位置 = 调用方责任（不跟踪子分支）。
```

## Design Decisions

1. **Object identity** = BLAKE3-256 hex of `serde_json::to_vec(segment)` (serde_json sorts keys by default). Path = `<store>/objects/<hh>/<hash>` (shard by first 2 hex chars).

2. **Segment** = the store's own enum (`User`/`Assistant`/`ToolResult`/`Summary`), mirroring kernel `Message` field shapes but not importing the kernel. `SummarySegment` carries `sources: Vec<String>` — hashes of the summarized visible segments (provenance; content-addressed, stable across forks). A later bridge crate adds `From` conversions.

3. **Log record** = `u32 BE length (payload)` + `u8 tag` + payload. No CRC — internal format, torn-tail detected by length. Tags: `0x00` header, `0x01` segment-ref, `0x02` compact-map, `0x03` side-effect. Payload length cap 64 MiB — beyond reads as torn.

4. **Branch header** = first record (tag `0x00`): `{ "kind": "root"|"subagent"|"compact", "parent_log"?, "parent_position"?, "created_at": <ms>, "inherited_seq": <u64> }`. `parent_position` = byte offset into parent log marking the end of the inherited prefix. `inherited_seq` = parent's side-effect seq at fork time (0 for root) — keeps WAL seq globally monotonic per session without parent-chain walks on open.

5. **Layout**: `<store>/objects/<hh>/<hash>`; `<store>/sessions/<sid>/session.json` (manifest: title, created_at), `<store>/sessions/<sid>/HEAD` (`{"log","position"}`), `<store>/sessions/<sid>/logs/<id>.log`. All log relpaths (`sessions/<sid>/logs/<id>.log`) are relative to the store root so parent references are unambiguous.

6. **HEAD policy**: HEAD tracks the session's main line. Updated atomically (tmp+rename) on `create_session`, `snapshot`/explicit checkout, and `compact` **when compacting the branch HEAD currently points at**; compacting a non-HEAD branch (a subagent's) leaves HEAD alone. `fork` (subagent) and plain appends never touch HEAD — the parent keeps writing its own log — after a crash, `open_current` reads HEAD for the current log and the open pass truncates only the torn tail (committed records survive). `resume` is different and deliberate: truncate to the HEAD checkpoint, discarding post-checkpoint records.

7. **Compaction = compact fork**: `writer.compact(mappings)` creates a new log (header kind=compact, parent = current log at current position), then writes one compact-map record per mapping: `{start, end, summary_hash, ts}`. Indices are 0-based half-open `[start, end)` into the **parent's materialized view at parent_position** (stable: parent log immutable). Map application: per-slot assignment, later records overwrite earlier ones per slot; consecutive slots sharing the same summary hash collapse into one view item. Compact-map records are only valid in compact logs (validated on open). If the compacted log is the one HEAD points at (the main line), HEAD moves to the new log; the parent log stays intact and forkable — undo-compaction = HEAD back to the parent.

8. **View recursion** (with cyclic-chain guard via visited log set):
   - `view(root)` = replay segment refs
   - `view(fork)` = `view(parent@parent_position)` + own records
   - `view(compact)` = `view(parent@parent_position)` with maps applied + own segment refs
   Chain order is the priority — chained compactions resolve by construction; no per-index conflict rules.

9. **fetch originals** via `SummarySegment.sources` → object store gets. No positional bookkeeping.

10. **Timestamps** (`ts` unix ms) live in record payloads (structure layer), never in objects (content layer) — git's commit-vs-blob separation. Objects stay dedupable.

11. **Side-effect record** = `{ "seq", "tool_call_id", "before_hash"?, "after_hash"?, "path", "ts" }`. before/after byte payloads content-addressed into the object store. Writer's seq = `max(header.inherited_seq, own max seq)`. Undo/redo replay is out of scope.

12. **Single-writer per branch** = owned `BranchWriter` handle, moved between await points. Concurrent appends to the same branch are a caller bug.

13. **No SQLite, no GC** in this plan. Session listing derivable by walking `sessions/`.

14. **Crash recovery** = torn-tail only. On open, scan in a single pass, truncate to last good record. `flush()` calls `sync_all()` (best-effort).

15. **Record writes** go through a synchronous std append on the blocking pool, NOT a long-lived `tokio::fs::File` handle: tokio's buffered `write_all` can resolve before the bytes reach the file, making fresh readers see stale content. Std appends are immediately visible process-wide; `flush()` remains the fsync durability barrier.

16. **Object store write-once**: existing hash → no-op. Write to unique `.tmp-<hash8>-<pid>-<ctr>` then rename; concurrent same-hash writes are safe (identical content).

17. **Rollback below a child's parent_position** would corrupt that child's view; children are not tracked — caller's responsibility (documented edge).

18. **IDs**: `format!("{:x}-{:x}", now_ms, atomic_counter)` for session and log ids.

## File Structure

```
crates/aaos-session/
  Cargo.toml
  src/
    lib.rs          — crate root, re-exports, now_ms/new_id utils
    segment.rs      — Segment enum + wire types (SummarySegment.sources)
    canon.rs        — canonical JSON + BLAKE3 hashing
    object_store.rs — content-addressed store (put/get bytes + typed segment)
    framing.rs      — length+tag record framing, torn-tail
    log.rs          — HeaderRecord (root/subagent/compact), record types
    branch.rs       — Branch (single-pass open, torn-tail truncate)
    writer.rs       — BranchWriter (append / fork / compact / side-effect)
    view.rs         — recursive materialize + map application + fetch_originals
    refs.rs         — session manifest, HEAD, create/open_current/resume/rollback
    error.rs        — thiserror error enum
  tests/
    object_store.rs, framing.rs, branch.rs, writer.rs, view.rs,
    compaction.rs, refs.rs, wal.rs, recovery.rs, integration.rs
```

Workspace: add `crates/aaos-session` to root `Cargo.toml` `[workspace] members`. No existing files modified except root `Cargo.toml`.

## Core Interfaces (pseudo-signatures)

```rust
// segment.rs
enum Segment { User(UserSegment), Assistant(AssistantSegment),
               ToolResult(ToolResultSegment), Summary(SummarySegment) }
fn canonical_bytes(&Segment) -> Vec<u8>   // serde_json::to_vec
fn segment_hash(&Segment) -> String       // BLAKE3 hex
fn segment_kind(&Segment) -> &'static str

// object_store.rs
struct ObjectStore { root: PathBuf }
impl ObjectStore {
    fn put_bytes(&self, &[u8]) -> String
    fn get_bytes(&self, hash) -> Vec<u8>
    fn put(&self, &Segment) -> String
    fn get(&self, hash) -> Segment
    fn contains(&self, hash) -> bool
}

// framing.rs
fn encode_record(tag: u8, payload: &[u8]) -> Vec<u8>
async fn read_record(src) -> ReadOutcome   // Eof | Torn | Record(DecodedRecord)

// log.rs
struct HeaderRecord { kind: Root|Subagent|Compact, parent_log: Option<String>,
                      parent_position: Option<u64>, created_at: u64, inherited_seq: u64 }
enum LogRecord { Header(HeaderRecord), SegmentRef(SegmentRefRecord),
                 CompactMap(CompactMapRecord), SideEffect(SideEffectRecord) }

// branch.rs
struct Branch { store_root, log_relpath, header, records: Vec<(LogRecord, u64)>, log_len }
impl Branch {
    async fn open(store_root, log_relpath) -> Self   // single pass + torn-tail truncate
}
async fn create_log_with_header(store_root, relpath, HeaderRecord) -> ()

// writer.rs
struct BranchWriter { store_root, session_id, log_relpath, objects, position, side_effect_seq, file }
impl BranchWriter {
    async fn open(store_root, session_id, log_relpath) -> Self
    async fn append_segment(&mut self, &Segment) -> String
    async fn fork(&mut self) -> BranchWriter                        // kind=subagent, HEAD untouched
    async fn compact(&mut self, mappings: Vec<(Range<u64>, SummarySegment)>) -> BranchWriter  // HEAD moves iff compacting the HEAD branch
    async fn append_side_effect(&mut self, tool_call_id, before: Option<Vec<u8>>, after: Option<Vec<u8>>, path) -> u64
    async fn snapshot(&self) -> SessionHead                         // writes HEAD
    fn position() -> u64
    async fn flush() -> ()
}

// view.rs
struct ViewItem { segment: Segment, hash: String }                  // hash of the VISIBLE object
async fn materialize(&ObjectStore, &Branch) -> Vec<ViewItem>
async fn materialize_plain(&ObjectStore, &Branch) -> Vec<Segment>
async fn fetch_originals(&ObjectStore, &SummarySegment) -> Vec<Segment>

// refs.rs
struct SessionHead { log_relpath: String, position: u64 }
struct SessionManifest { title: String, created_at: u64 }
async fn create_session(store_root, title) -> (session_id, BranchWriter)
async fn read_head(store_root, session_id) -> SessionHead
async fn write_head(store_root, session_id, &SessionHead) -> ()
async fn open_current(store_root, session_id) -> BranchWriter       // torn-recover, continue at end
async fn resume(store_root, session_id) -> BranchWriter             // deliberate truncate to checkpoint
async fn rollback(store_root, session_id, &SessionHead) -> ()       // truncate + set HEAD
```

## Task Dependency Graph

| Task | Depends On | Deliverable |
|---|---|---|
| 1. Crate scaffold | — | empty crate compiles |
| 2. Segment types + canon + hash | 1 | `Segment`, `canonical_bytes`, `segment_hash`, `SummarySegment.sources` |
| 3. Object store | 2 | `put`/`get`/`put_bytes`/`get_bytes`, dedup |
| 4. Record framing | 1 | encode/decode, torn-tail by length, 64 MiB cap |
| 5. Branch log reader | 4 | `Branch::open` (single-pass, torn truncate, header-kind validation) |
| 6. BranchWriter | 3,4,5 | append / fork / compact (HEAD moves) / side-effect (seq inherit) |
| 7. View | 3,5,6 | recursive materialize, map apply + collapse, `fetch_originals` |
| 8. Compaction test | 7 | range replace, sources fetchable, chained compaction, undo via parent log |
| 9. refs: HEAD + rollback/resume | 6 | create_session, snapshot, open_current vs resume semantics |
| 10. Side-effect WAL test | 6 | monotonic seq, continuity across fork/compact |
| 11. Crash recovery test | 3,4,6 | torn-tail truncated on open, idempotent writes |
| 12. Integration test | 2-11 | append→fork→compact→chain→undo→snapshot→rollback→resume roundtrip |

---

### Task 1: Crate scaffold

- [x] Add `crates/aaos-session` to workspace `members`.
- [x] Create `Cargo.toml` (deps: blake3, serde, serde_json, thiserror, tokio; dev: tempfile, tokio macros/rt).
- [x] Stub `lib.rs` with module doc.
- [x] `cargo build && cargo test -p aaos-session` → 0 tests pass.
- [x] Commit.

---

### Task 2: Segment types + canonical JSON + hashing

- [x] `segment.rs`: `Segment` enum + wire types (`UserSegment`, `AssistantSegment`, `ToolResultSegment`, `SummarySegment { content, sources: Vec<String> }`, `ContentBlock`, `ToolCall`, `Usage`, `Cost`, `StopReason`, `ImageSource`) — mirror `pi-agent-core` `Message` field shapes by reading that crate (no import). Derive `Serialize`/`Deserialize`; tag-bearing enums use `#[serde(tag = "type", rename_all = "snake_case")]`.
- [x] `canon.rs`: `canonical_bytes` = `serde_json::to_vec`. `segment_hash` = BLAKE3 hex. `hash_hex(bytes)`. `segment_kind` = match on variant.
- [x] Tests: canonical bytes deterministic, hash is 64 lowercase hex, kind distinguishes variants, serde roundtrip.
- [x] Commit.

---

### Task 3: Object store

- [x] `error.rs`: `StoreError { Io, InvalidHash, NotFound, Decode, Encode, InvalidLog }` via thiserror.
- [x] `object_store.rs`: `put_bytes`/`get_bytes` (arbitrary bytes, BLAKE3 hash), `put`/`get` (typed Segment). Write-once via unique `.tmp-*` + rename. Two-level shard path. `contains`. Missing file → `NotFound`; non-64-hex → `InvalidHash`.
- [x] Tests: put→get roundtrip, idempotent put, missing→NotFound, shard path shape.
- [x] Commit.

---

### Task 4: Record framing

- [x] `framing.rs`: `encode_record` = `u32 BE len(payload)` + `tag` + payload. `read_record` → `Eof` (clean) | `Torn` (short payload / tag / over-cap len) | `Record`. Cap = 64 MiB.
- [x] Tests: roundtrip, torn tail on truncated payload, clean EOF, over-cap len reads Torn.
- [x] Commit.

---

### Task 5: Branch log reader

- [x] `log.rs`: `HeaderRecord` (kind root/subagent/compact + parent fields + `created_at` + `inherited_seq`), `SegmentRefRecord {hash, kind, ts}`, `CompactMapRecord {start, end, summary_hash, ts}`, `SideEffectRecord {seq, tool_call_id, before_hash, after_hash, path, ts}`, `LogRecord` enum, payload encode/decode by tag.
- [x] `branch.rs`: `Branch::open` single pass — header must be first and unique; compact-map records only in compact logs; torn tail truncated in place; `records()` with end offsets; `create_log_with_header`.
- [x] Tests: header roundtrip; header+appended ref read back; duplicate header → error; compact-map in non-compact log → error.
- [x] Commit.

---

### Task 6: BranchWriter

- [x] `writer.rs`: `open` = `Branch::open` (torn-recover) + append handle; `position` = log_len; seq = max(inherited, own). `append_segment` puts object + writes segment-ref. `fork()` creates subagent log (parent = self@position, inherited seq), HEAD untouched. `compact(mappings)` validates `start < end`, puts summaries, writes compact-map records; HEAD moves iff compacting the HEAD branch. `append_side_effect` content-addresses before/after, seq += 1. `snapshot()` writes HEAD at current position. `flush` = `sync_all`.
- [x] Tests: append writes ref + object, position monotonic; fork writes header with parent position, HEAD untouched; compact writes maps + moves HEAD (main line) but not for a subagent branch.
- [x] Commit.

---

### Task 7: View

- [x] `view.rs`: `materialize` walks the chain (visited-set cyclic guard), materializes parent prefix up to `parent_position`, applies compact maps per-slot (later wins, consecutive same-hash collapse), then own segment refs. Returns `Vec<ViewItem { segment, hash }>`. `fetch_originals` via `summary.sources`.
- [x] Tests: `materialize_plain` replays refs in order; map range validation (`end` beyond parent view → error).
- [x] Commit.

---

### Task 8: Compaction test

- [x] Test: append 4 segments, compact [1,3) with summary (sources = hashes of segs 1/2) → view = [seg0, S, seg3]; `fetch_originals` returns segs 1/2. Adjacent ranges → adjacent summaries. Overlapping maps → later wins. Chained compaction (compact of compacted view). Undo: materialize parent log → original 4 segments intact.
- [x] Commit.

---

### Task 9: refs — HEAD, rollback, resume

- [x] `refs.rs`: `create_session` (manifest + root log + HEAD), `read_head`/`write_head` (tmp+rename), `open_current` (HEAD → torn-recover → writer at end), `resume` (truncate to HEAD checkpoint → writer), `rollback` (explicit head → truncate + set HEAD).
- [x] Tests: snapshot → append → rollback → view without the appended segment; `resume` reopens at checkpoint after further appends; `open_current` keeps post-checkpoint records.
- [x] Commit.

---

### Task 10: Side-effect WAL test

- [x] Test: append two side-effects (first `before_hash` None), seq monotonic from 1; fork → child seq continues; compact → seq continues; read back records.
- [x] Commit.

---

### Task 11: Crash recovery test

- [x] Tests: append good record then garbage tail → `Branch::open` truncates to good length; `open_current` after garbage recovers and appends cleanly. Concurrent `put` of same content → same hash, one file.
- [x] Commit.

---

### Task 12: Integration test

- [x] Test full lifecycle: create_session → root append 3 → subagent fork (inherits 3, appends 2) → parent compact [1,3) → append on compacted → chained compact → undo via parent log materialization → snapshot → append → rollback → resume → append. Verify view lengths/segment types at each stage; verify parent log never mutated by compaction.
- [x] `cargo test -p aaos-session` → all pass. `cargo test --workspace` → no regressions. `cargo clippy -p aaos-session -- -D warnings` → clean.
- [x] Commit.

---

## Self-Review

1. **Spec coverage**: object store (3), append-only logs (5/6), fork (6), compaction-as-fork + HEAD (6/8/9), view recursion (7), provenance (2/8), side-effects (10), crash recovery vs resume distinction (9/11). GC and trigger policy explicitly out of scope per decisions.

2. **No placeholders**: no TBD/TODO. No CRC, no base64 — torn-tail length-based, payloads content-addressed, canonical JSON via serde_json default key sorting.

3. **Type consistency**: `ViewItem { segment, hash }`; `compact(mappings: Vec<(Range<u64>, SummarySegment)>)`; `SessionHead { log_relpath, position }`; `HeaderRecord.inherited_seq` — consistent across tasks.

4. **Dependency hygiene**: blake3, serde, serde_json, thiserror, tokio, tempfile only. No pi-agent-core import. `Branch::open` single-pass.

5. **Scope discipline**: no GC, no compaction policy, no WAL undo/redo, no SQLite index, no view caching, no `AgentSession` wiring. Storage layer data structures + operations only.
