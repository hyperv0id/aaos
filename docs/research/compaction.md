# aaos 压缩（Compaction）：触发链路、机制与架构设计计划

> 状态：~~设计计划（draft）~~ 已实施但机制偏离本文 · 原始日期：2026-08-31
> 依据：`.scratch/compaction-reference.md`（pi / oh-my-pi / grok-build / deepseek-harness 四实现调研）、`.scratch/compaction-aaos-survey.md`（aaos 现状普查，所有 `文件:行号` 与之核对一致）、根 `CONTEXT.md`（词汇）、ADR-0001/0002/0003。
> 实施结果（commit `048e79e` "feat: context compaction with deterministic transcript (#70)"）：压缩摘要为**确定性转录**，无任何模型调用——LLM 摘要子系统已删除，见 [ADR-0007](../adr/0007-compaction-deterministic-transcript-paging.md) 与 `crates/aaos-session/src/compaction.rs::build_transcript`。
> 以下机制节自此仅作历史记录（读者须知其已否决）：D3（独立 Agent 实例）、D5 的 turn-prefix 摘要、§2.3 的 `summaryMaxTokens` 行、§3.2（摘要生成路径）、§3.3（摘要消息形式与投影）、P1.4（摘要生成）。
> 词汇：本文严格使用「压缩（派生）」「视图」「资产」「书签」「头指针」；「overlay / summarization」不作为结构义（摘要只是压缩的生成方式），「当前会话」不指头指针。

---

## 0. 决策摘要

| # | 决策 | 理由 |
|---|---|---|
| D1 | 第一期**先修 provider usage 解析**，再以 usage 锚定触发计量；本地估计仅兜底 | 现状 usage 恒零（anthropic_messages.rs:320-324），纯估计会把「usage 锚 + 增量」这个四实现通用模式推迟到二期，且触发点代码要返工 |
| D2 | 压缩后通过 `AgentSession::resume(compacted_id)` 切换节点；**不写 head**（ADR-0003），不引入任何指针簿记 | 压缩天然是派生——映射即投影，`firstKeptEntryId` 簿记被「映射 + 视图折叠」取代（db.rs:471-487），与 ADR-0002/0003 完全一致 |
| D3 | 摘要生成用**独立 `Agent` 实例**（不 attach 持久化 listener），复用会话前缀保持 KV cache；摘要以 user 角色 + 包裹标记经 `transform_context` 注入 | 复用当前 Agent 实例会把摘要写进当前节点（agent_session.rs:74-94 的 MessageEnd listener），独立实例是唯一不需要改 pi-agent-core 的路径 |
| D4 | 触发检查点收敛为**两处自动检查点 + 一处手动入口**：`prepare_next_turn` hook（turn 末，主链路）+ `transform_context` hook（每次 LLM 请求前，含 mid-turn 工具循环）+ CLI `/compact`（手动） | pi 的对齐目标：请求前检查点（agent-session.ts:543-586）+ turn 末 `_checkCompaction`（2126-2180）+ 手动 compact |
| D5 | 切点选择：从尾向头累计保留 `keepRecentTokens`（默认 20000），切点不落在 toolResult 上；**允许切在 turn 中间**，前缀另生成 turn-prefix 摘要 | 对齐 pi findCutPoint（compaction.ts:403-480）与 isSplitTurn（721-742）；DSH 允许同样规则 |
| D6 | overflow 恢复只试一次；压缩后仍超限不做二次压缩（由下次阈值检查自然接管） | 对齐 pi `_overflowRecoveryAttempted`（2164-2178）与「无二次处理」边界（2403-2417） |
| D7 | 压缩节点创建后校验「投影变小」；空压缩（保留段覆盖全部）拒绝 | 对齐 grok sanitize/validate 门（compaction.rs:1701-1721）与 pi "Nothing to compact"（1960-1963） |

---

## 1. 目标与范围

### 1.1 第一期做什么

端到端最小链路（对齐 pi 主线）：

1. **触发**：每次 LLM 请求前与 turn 末两个自动检查点 + 手动 `/compact`，阈值 = 上下文 token > `context_window − reserveTokens`（默认 16384）。
2. **计量**：provider usage 锚定 + 本地估计兜底（先修 usage 解析，D1）。
3. **机制**：切点选择（保留尾 20000 token、toolResult 配对约束、turn 中间切割）、结构化摘要 prompt（增量更新）、摘要消息注入、派生压缩节点、压缩后节点切换。
4. **持久化**：走既有 `SessionStore::compact`（db.rs:376-431）——区间替换映射派生、原文可寻址、undo = 从父派生（store_compaction.rs:63-69 已验证契约）。

### 1.2 明确不做（第一期）

- 不做 omp 的推测式压缩、idle 压缩、snapcompact、mid-turn 工具循环内压缩的**推测/死胡同检测**部分（触发点本身覆盖 mid-turn 的 LLM 请求前检查）；不做 grok 的 full-replace；不做 DSH 的插件化能力缝/`surfaceOp` 事件面。这些只在对齐论证中引用（见 §6）。
- 不做摘要的独立压缩模型（沿用会话目标模型，对齐 DSH `summarizationModel` 默认继承）。
- 不做 toolResult 修剪（omp `pruneToolOutputs` / DSH `toolResultPruner`）。
- 不做 `retainedTail` 落库：aaos 的保留段是**运行时投影**（切点每次重新计算，映射只记录区间），与 pi 的 `firstKeptEntryId` 簿记不同——见 §3.4。
- 不做 TUI；`--json` 模式下 `/compact` 输出约束在 stderr。

---

## 2. 触发链路

### 2.1 检查点设计

对齐 pi 的三个入口（agent-session.ts:543-586 请求前检查点；2126-2180 turn 末 `_checkCompaction`；1940-2078 手动 `compact()`），映射到 aaos 既有 seam（现状：survey §三候选挂载点）：

| 检查点 | aaos 挂载位置 | 对齐来源 | 说明 |
|---|---|---|---|
| 请求前阈值检查 | `Agent.transform_context` hook（agent.rs:275；执行于 agent_loop.rs:570-578） | pi `_compactBeforeNextAssistantResponse`（agent-session.ts:543-586） | 每次 LLM 请求前（含工具循环内每次续跑）检查阈值，命中则先压缩再继续本次请求 |
| turn 末检查 | `prepare_next_turn` hook（agent_loop.rs:421-455，类型 types.rs:698-705） | pi turn 末 `_checkCompaction`（agent-session.ts:2126-2180） | 携带刚定稿的 assistant 消息（含 usage 字段）与完整上下文，是「turn 末决定压缩」的天然位置 |
| overflow 恢复 | 同一 hook 内的错误分支（assistant 消息 `stop_reason == Error` 且错误信息匹配 overflow 模式） | pi `isContextOverflow`（overflow.ts:134-173）+ `_overflowRecoveryAttempted`（agent-session.ts:2164-2178） | 只试一次（D6）；aborted 消息默认跳过检查 |
| 手动 | CLI `run_repl` 输入循环（main.rs:259-273，输入 trim 于 :266，`prompt()` 于 :270）加 `/compact [text]` 前缀分发 | pi 交互式 `/compact [text]`（interactive-mode.ts:3082-3083） | 前置检查：最后一条已是压缩 → "Already compacted"；无可压缩内容 → "Nothing to compact"（pi 1960-1963） |

**挂载方式**：CLI `build_agent`（main.rs:146-200）设置 `transform_context` 与 `prepare_next_turn`，闭包捕获一个「压缩协调器」（见 §4.4 的 `CompactionCoordinator`）；`run_repl` 循环在 `session.agent_mut().prompt(input)` 前拦截 `/` 前缀命令（main.rs:266-270 之间插入分发）。

**mid-turn 覆盖**：`transform_context` 每个 LLM 请求都执行，工具循环内的每次续跑天然被覆盖——这是 pi 请求前检查点的直接等价（pi 本身也无独立 mid-turn 检查点；omp 的 `maintainContextMidRun` 是增强，不在第一期）。

### 2.2 token 计量：usage 锚 + 本地估计兜底

**先决修复项（D1）**：现状 provider 适配器不解析流中 usage（anthropic_messages.rs:320-324 明确注释；`final_seed` 恒零 :266-272；openai-completions / google-genai / cohere-chat 同）。`Usage` 结构两侧齐备且逐字段互转（convert.rs:116-125/242-251），缺口只在适配器解析。

**决策：第一期先修 usage 解析，再锚定**（理由见下）。

计量方案（对齐 pi compaction.ts:166-169/183-214/280-350）：

1. **usage 锚**：取 `agent.state.messages` 中最后一条 `usage.total_tokens > 0` 的 assistant 消息；`tokens = usage.total_tokens + 其后各消息的 estimateTokens 之和`（pi: `calculateContextTokens(usage)` + 尾部估计）。usage 失真（全零/529 错误）时退回纯估计防死锁（pi compaction.ts:280-350 注释）。
2. **本地估计兜底**（无任何可用 usage 时全量）：文本 `ceil(chars/4)`；图片固定常数（对齐 pi `ESTIMATED_IMAGE_CHARS=4800` ≈ 1200 token）；assistant 含 thinking 与 tool_call 的 name+arguments；toolResult 计内容文本。
3. **context_window**：`agent.state.model.context_window`（types.rs:45，运行期可读，survey §1.6）；catalog 侧 `CatalogModel.context_window`（catalog.rs:138）已透传（:165-191）。

**选择理由（D1）**：若第一期纯本地估计，则第二期改锚定时必须重写触发代码与测试，而 usage 解析本身是独立、可单独验收的小改动（四个适配器 + 现有 provider 测试惯例）；且 pi 主线的计量语义就是「usage 锚 + 增量估计」，纯估计方案与对齐目标偏差最大。风险：usage 解析涉及四个适配器的 SSE 终态（`message_stop` / `finish_reason` / `usage` 块，sse.rs:198-206），是第一期最大的改动面——但它是**独立前置**，可先落地并单独验收（见 §5 分期）。

### 2.3 阈值与配置项

对齐 pi `CompactionSettings`（compaction.ts:148-161；settings-manager.ts:830-850）：

| 配置 | 默认 | 语义 | 对齐 |
|---|---|---|---|
| `compaction.enabled` | true | 自动触发总开关（手动 `/compact` 不受其控） | pi |
| `compaction.reserveTokens` | 16384 | 阈值 = `contextTokens > context_window − reserveTokens`（compaction.ts:235-249） | pi |
| `compaction.keepRecentTokens` | 20000 | 保留尾预算，切点从尾向头累计（compaction.ts:436-447） | pi |
| `compaction.summaryMaxTokens` | `min(floor(0.8 × reserveTokens), model.max_tokens)` | 摘要输出上限（compaction.ts:503-560 预算） | pi |
| `compaction.overflowRetry` | 1（只试一次） | overflow 恢复次数上限 | pi（2164-2178）/ DSH `maxOverflowRetries=1` |

不做 omp 的 `thresholdPercent` / `thresholdTokens` / `midTurnEnabled` / `idleEnabled` / `methodOrder`；不做 grok 的 `threshold_percent`（85）——第一期统一 reserve 式（D4 收敛）。配置载体：第一期 CLI 环境变量（如 `AAOS_COMPACTION_*`，沿用 CLI 现状无配置文件的事实），硬编码默认值落于协调器常量。

**不做什么**：不做「usage 源在压缩前」的陈旧性检查（pi agent-session.ts:2196-2224 防压缩后立刻误触发）——aaos 每次触发前都重算 usage 锚与全量估计，且 `transform_context` 检查点紧邻请求，陈旧窗口极小；列为风险 R4 观察。

---

## 3. 机制

### 3.1 切点选择

对齐 pi `findCutPoint`（compaction.ts:403-480）+ `findValidCutPoints`（373-402）+ `prepareCompaction`（616-720）：

1. 对视图消息序列（`agent.state.messages`）从尾向头累计 `estimateTokens`，直到 ≥ `keepRecentTokens`（默认 20000）→ 得到「首个保留索引」候选。
2. 合法切点集合：消息 role ∈ {user, assistant, toolResult（配对完整）}——**toolResult 不可作切点**（否则悬空 tool_call）。具体规则（对齐 pi 373-402 + DSH `toolPairingBalancedBefore/After` region.ts:126-137）：
   - 切点前一条若为 assistant 且含未配对的 `tool_call`，或切点本身是 toolResult → 向前移动到最近合法边界。
   - 只检查配对，**不检查 turn 边界**。
3. 压缩映射 = `[(0, first_kept_index)]`（单个连续区间；aaos 映射支持多区间与相邻折叠，db.rs:492-525，但 pi 主线为单区间 tail-keep，第一期不主动生成多区间）。
4. **切在 turn 中间时（isSplitTurn）**：被切 turn 的前缀（turn 起始到保留段起点）单独生成 turn-prefix 摘要，与历史摘要拼接（对齐 pi TURN_PREFIX_SUMMARIZATION_PROMPT，compaction.ts:721-742）。判断方式与 pi 一致：切点前一条消息不是 user 消息即视为 split turn。
5. 保留段内 tool_call/toolResult 配对天然完整（切点在配对之外）。

### 3.2 摘要生成路径（独立 Agent 实例）

**关键约束（D3）**：不能复用当前 Agent 实例——其 `MessageEnd` listener 会把摘要消息追加进当前节点（agent_session.rs:74-94）。设计：

- 新建独立 `Agent` 实例（同一 provider/model/catalog 解析，复用 `build_agent` 的模型装配，main.rs:146-200），**不 attach 任何持久化 listener**；驱动方式：`agent.prompt(summary_prompt)` 或直接 `agent_loop`（pi-agent-core 公开，agent_loop.rs:124-165），取返回消息的文本。
- **KV cache 保活**：摘要请求的上下文 = 会话自己的 system prompt + 前缀消息逐字重放 + 摘要指令作为最后一条 user 消息（对齐 DSH summarizer.ts:29-75 复用最近路由请求的可缓存前缀；grok 的 verbatim 阶梯同思路）。第一期以「与主链路同 provider/model、同 system prompt、前缀复用」实现，不做 DSH 的逐字重放优化细节——列为二期候选。
- 摘要 prompt（对齐 pi compaction.ts:503-560 结构化模板）：
  - system：`SUMMARIZATION_SYSTEM_PROMPT`（"You are a context summarization assistant… ONLY output the structured summary."）。
  - user：`<conversation>\n{serializeConversation(压缩区间内消息)}\n</conversation>` + 可选 `<previous-summary>…</previous-summary>`（增量更新，对齐 pi UPDATE_SUMMARIZATION_PROMPT：保留旧信息、合并新进展）+ 结构化模板（Goal / Constraints / Progress / Key Decisions / Next Steps / Critical Context；要求精确保留文件路径、函数名、错误信息）。
  - 输出上限：`summaryMaxTokens`（§2.3）。
- 序列化：`serializeConversation` 把区间内消息压成紧凑文本（对齐 pi utils.ts；第一期不含 `extractFileOperations` 文件操作提取，列为二期候选）。

### 3.3 摘要消息形式与投影

- **落库**：`Segment::summary(content, sources)`（segment.rs:62-64）+ `SummarySegment.generated_by(model)`（segment.rs:113-116，补上唯一生产写入方）。
- **`SummarySegment.sources` 哈希获取（现状缺口 #6）**：调用方须用 `store.materialize(id)`（db.rs:213-220，带 hash）取被压缩区间的哈希，构造 `sources`；不得走 `materialize_plain`（agent_session.rs:149，丢 hash）。对齐 pi 压缩 entry 的 provenance 与 ADR-0001 出处双轨（结构级 `fetch_originals` + 内容级 `sources`）。
- **视图投影（读路径已实现）**：`materialize_messages`（agent_session.rs:161-175）把 `Summary` 渲染为 user 消息 `"[compacted summary] {content}"`。**第一期不改变此格式**——pi 的 `<summary>` 包裹与「history before this point was compacted」前导（messages.ts:4-10/142-160）属于 LLM 请求面，见下。
- **LLM 请求面注入**：`transform_context` hook（agent_loop.rs:570-578）在触发压缩后把摘要消息转换为 pi 形式注入本次请求：`The conversation history before this point was compacted into the following summary: <summary>…</summary>`（role: user，对齐 pi messages.ts:142-160 `convertToLlm`；grok summary.rs:117-126 前导、DSH CHECKPOINT_PREAMBLE 同思路）。压缩后 `agent.state.messages` 直接替换为压缩节点视图（对齐 pi agent-session.ts:2363）。

### 3.4 压缩后节点切换

**对齐 pi 与 aaos 结构优势的结合（D2）**：

- pi 的上下文重建 = 最新 compaction entry + `firstKeptEntryId` 起保留段 + 其后条目（session-manager.ts:461-489），`firstKeptEntryId` 是簿记字段。
- aaos 不需要该簿记：压缩是**派生**——`SessionStore::compact(parent_id, mappings, summary)`（db.rs:376-431）落 `sessions` 行（kind=compact）+ `compactions` 映射行，视图折叠在 `view_hashes`（db.rs:471-487）按映射应用，保留段由**运行时切点重算**表达（§2.3 的 keepRecentTokens 每次重算），不是落库时冻结的指针。映射即投影。
- 流程：
  1. 协调器用 `materialize(current_id)` 取视图 + 哈希，选切点，生成摘要。
  2. `store.compact(current_id, &[(0, first_kept)], &summary)` → `compacted_id`（不动 head，ADR-0003）。
  3. `session.resume(&compacted_id)`（agent_session.rs:148-156）——整体替换 `agent.state.messages` 并切换当前节点，后续 `MessageEnd` 追加自动重定向（agent_session.rs:74-94 共享锁机制）。
  4. undo = 从父节点 `fork`（无 Summary，store_compaction.rs:63-69 已验证）；书签/分叉语义天然可用（lifecycle.rs:86-102 已验证压缩线上的 bookmark/fork_at）。
- **head 语义**：压缩节点不成为 head（ADR-0003「compact/fork/create_root 都不动头指针」）；用户默认下次启动仍从 head 派生新线，压缩线通过显式 `--session compacted_id` 或书签继续。这是 aaos 与 pi 的刻意差异（pi 无 head 概念，树导航即恢复路径）——对齐 ADR-0003 的进程隔离语义。

### 3.5 失败分类与边界

对齐 pi compaction.ts:588-596 + agent-session.ts:2164-2184/2403-2417：

| 情形 | 行为 | 对齐 |
|---|---|---|
| 摘要失败（stopReason=Error） | 自动路径：发 `compaction_end {errorMessage}` + `session_compact_failed` 事件，**不落压缩节点**，原始错误不吞；手动路径：stderr 报 "Compaction failed: …" | pi 588-596 / 2071 |
| 摘要 abort | 信号传导，不落节点 | pi 596 |
| 压缩后仍超限 | **无自动二次压缩**，仅记录；由下次请求前阈值检查自然接管 | pi 2403-2417 |
| overflow 恢复 | 只试一次（`overflow_retry_attempted` 位）；恢复前把最后一条失败/截断的 assistant 消息从 agent state 移除（保留在会话历史，排除出重试上下文）；再 overflow 直接失败并提示 | pi 2164-2184, 2384-2393；DSH maxOverflowRetries=1 |
| 空压缩 | 保留段覆盖全部视图（无可压缩内容）→ 拒绝，报 "Nothing to compact"；最后一条已是压缩 → "Already compacted" | pi 1960-1963 |
| 投影校验 | 压缩后视图 token 估计 ≥ 压缩前 → 拒绝该压缩（防退化） | grok sanitize/validate 门（1701-1721）+ omp snapcompact 投影检查 |
| 切点无合法边界 | 压缩区间为空/无法满足配对约束 → 确定性失败，不静默 | pi findValidCutPoints 空集即 Nothing to compact |

---

## 4. 架构

### 4.1 逐 crate 归属

| crate | 改动 | 接缝依据 |
|---|---|---|
| `aaos-providers` | **usage 解析（先决，D1）**：四个适配器（openai-completions / anthropic-messages / google-genai / cohere-chat）在 SSE 终态解析 usage 块并写入 assistant 消息；anthropic_messages.rs:320-324 注释移除，`final_seed`（:266-272）填真实值 | 适配器是唯一能拿到流中 usage 的层；`Usage` 结构两侧齐备（types.rs:152-160 ↔ segment.rs:153-161），解析后 convert.rs:116-125 自动搬运 |
| `pi-agent-core` | **不改动**（一期）。`transform_context` / `prepare_next_turn` / `TurnEnd` 事件均已存在（agent.rs:270-275、agent_loop.rs:415-419/421-455/570-578），无需新增 seam | ADR-0002 决定 pi-agent-core 保持 Pi Agent 忠实移植的无状态设计，领域逻辑不泄漏进核心 |
| `aaos-session` | `AgentSession` 新增：① `store()` 访问器或等价方法（现状 store 私有，agent_session.rs:34-45，CLI 拿不到句柄——survey 缺口 #4）；② 压缩协调方法（或由 CLI 侧组合：`materialize` → `compact` → `resume`，接口最小化优先）；③ 视需要暴露「最后 usage 锚」读取 | ADR-0002：`aaos-session` 是唯一集成入口，CLI 不直接触 store |
| `aaos-cli` | ① `run_repl` 输入循环加 `/compact [text]` 前缀分发（main.rs:266-270 之间）；② `build_agent`（main.rs:146-200）设置 `transform_context` / `prepare_next_turn` 闭包；③ 新增压缩协调器（`CompactionCoordinator`：阈值判定、切点、摘要调用、`compact`+`resume` 编排、失败分类、事件输出）；④ `Cli` 结构体（main.rs:24-45）加 `--compact` 单发参数（可选，一期以 REPL `/compact` 为准） | 协调器持 AgentSession 句柄与模型装配（build_session 同级，main.rs:99-111） |
| `aaos-tools` | 不改动 | 无 token 计量职责（survey §1.8） |

**协调器位置**：`aaos-cli`（一期唯一前端；ADR-0002「吸收进 aaos-session 为多前端预留同一入口」的考量下，若后续 TUI 需要，再把编排下沉——一期不预设）。协调器的**纯逻辑部分**（阈值、切点、序列化、prompt 构造）放 `aaos-session` 内新模块（可测、不依赖 CLI），CLI 只做装配与事件输出。

### 4.2 与 ADR-0002/0003 一致性

- **ADR-0002**：摘要生成不走当前 Agent 的 MessageEnd 通道（D3）；所有持久化仍经 `aaos-session` 单通道；`Segment`↔`Message` 转换不新增变体（Summary 仍无 agent 侧消息类型，读路径渲染兜底不变，agent_session.rs:161-175）。
- **ADR-0003**：压缩派生不动 head；`resume(compacted_id)` 切换只是进程内节点重定向，不写 head；头指针唯一性（meta 表唯一可变行）不受影响。
- **词汇**：全程「压缩 = 派生」，「视图 = 重放投影」，「原文仍可寻址」（`fetch_originals` db.rs:436-457 + `sources` 双轨）。

### 4.3 pi 对齐 vs aaos 结构优势（§3.4 展开）

pi 的机制依赖 JSONL 追加 + `buildContextEntries` 投影 + `firstKeptEntryId` 簿记（session-manager.ts:1098-1120）；aaos 的「派生 + 区间映射」把同样语义**归约为一次 `compact` 调用**：

- 无 `firstKeptEntryId`：保留段边界由切点计算表达，映射区间即边界。
- 无「重建上下文」步骤：视图折叠在 `view_hashes`/`materialize` 读路径（db.rs:471-487），resume 即重建。
- 多压缩链：`chained_compaction_resolves_by_order`（store_compaction.rs:117-135）已验证「第二次压缩的索引以折叠视图为坐标系」——pi 的「最新 compaction entry + 保留段」是同一语义的特例。
- undo/书签/分叉：结构层已有（lifecycle.rs:86-108），压缩无需新增恢复路径。

---

## 5. 分期实施步骤

### 第一期：端到端最小链路（pi 对齐）

| 步骤 | 改动面 | 验收 |
|---|---|---|
| P1.1 usage 解析（先决，D1） | `aaos-providers/src/formats/*` 四适配器 + 现有 provider 测试 | provider 测试断言 assistant 消息 usage 非零且与 SSE usage 块一致（现有测试惯例：`crates/aaos-providers/tests/`） |
| P1.2 计量模块 | `aaos-session` 新模块（usage 锚 + 本地估计 + 阈值判定） | 单元测试：纯估计 / usage 锚 / 锚失效回退三路 |
| P1.3 切点选择 | 同模块（find_cut_point + 配对约束 + split-turn 判定） | 单元测试：toolResult 不可切、turn 中间切割、空压缩拒绝 |
| P1.4 摘要生成 | `aaos-cli` 协调器（独立 Agent 实例 + prompt 构造 + 序列化） | 手动 `/compact` 冒烟：摘要落库（`SummarySegment` 含 sources 哈希与 model） |
| P1.5 触发与切换 | `aaos-cli`：`build_agent` 设 hook + `run_repl` 命令分发 + `resume(compacted_id)` | 见下「端到端验收」 |
| P1.6 契约测试 | `crates/aaos-session/tests/`（沿用 store_compaction.rs 风格，见 store_compaction.rs:22-77） | 压缩后视图 = 保留段 + 摘要；sources 哈希可解析；undo = fork 父；head 不变 |

**端到端验收（第一期的完成标准）**：REPL 中「阈值触发（或 `/compact`）→ 生成摘要 → 派生压缩节点 → 切换续写 → resume 后视图正确」全链路跑通：

1. 造一个超过阈值的会话（大 prompt / 多轮），观察请求前 `transform_context` 检查点自动压缩；
2. `/compact` 手动路径：输出压缩节点 id，后续输入追加到该节点；
3. `--session <compacted_id>` 重开：视图 = 摘要 + 保留段（`"[compacted summary]"` 前缀渲染），续写正常；
4. `fetch_originals` / `sources` 双轨可取回原文；undo 从父节点派生无 Summary；
5. head 未动（默认启动仍从原 head 派生）。

### 第二期（后续可选）

- 摘要 KV cache 保活优化（DSH 逐字前缀重放细节）、`extractFileOperations` 文件操作清单（pi utils.ts）、`thresholdPercent` 等配置扩展（omp）、推测式压缩（omp speculation-lead）、snapcompact（omp）、full-replace 模式（grok）、插件化能力缝（DSH `ctx.compaction`）、toolResult 修剪（omp pruneToolOutputs / DSH pruner）。

---

## 6. 风险与未决问题

| # | 风险/未决 | 缓解 |
|---|---|---|
| R1 | usage 解析是四适配器大改，SSE 终态各 provider 字段差异大（anthropic `message_delta.usage` vs openai `chunk.usage`） | P1.1 独立前置 + 独立验收；估计兜底保触发不依赖 usage 正确性 |
| R2 | `transform_context` 内执行压缩是 await 内嵌（hook 返回 Result<Vec<Message>, String>），压缩失败只报错不阻断请求 | 对齐 DSH pre-step「失败仅 warn 不阻断」；错误事件进 stderr |
| R3 | 独立摘要 Agent 与主 Agent 并发/复用 provider 连接、模型不一致 | 摘要实例串行执行（压缩期间新输入排队——REPL 单线程天然串行）；模型沿用会话目标模型（DSH 默认继承） |
| R4 | 陈旧 usage 锚（压缩后误触发） | 每次触发前重算锚 + 全量估计；若实测误触发再引入 pi 的陈旧性检查（agent-session.ts:2196-2224） |
| R5 | 压缩后仍超限的「无二次压缩」在超大单轮下可能反复触发 | 对齐 pi 行为（仅记录）；omp mid-turn 死胡同检测列为二期 |
| R6 | `SummarySegment.sources` 的哈希规模（长会话哈希列表膨胀） | 第一期如实记录（对齐 pi provenance）；若膨胀实测可见，二期可只记区间坐标 |
| R7 | 未决：`/compact` 后 REPL 是否打印提示/如何呈现（对齐 pi 无此行为，aaos 特有 UX 决定） | 一期：stderr 打印压缩节点 id + 摘要 token 账目 |
| R8 | 未决：摘要消息注入格式是否同步修改 `materialize_messages` 的 `"[compacted summary]"` 前缀（一期不改，读路径稳定优先） | 若二期引入 pi 式 `<summary>` 包裹，需同时更新读路径渲染与契约测试 |

---

## 附：对齐引用索引

- pi：`agent-session.ts:543-586`（请求前检查点）、`:2126-2180`（turn 末 `_checkCompaction`）、`:2164-2184`（overflow 单次恢复）、`:2196-2224`（usage 陈旧性）、`:2363`（state 替换）、`:2403-2417`（失败事件）、`:1940-2078`（手动 compact）、`:1960-1963`（空压缩/已压缩拒绝）；`compaction.ts:148-161`（设置默认）、`:166-169`（usage 计算）、`:183-214`（usage 锚）、`:235-249`（阈值）、`:280-350`（本地估计）、`:373-402`（合法切点）、`:403-480`（findCutPoint）、`:436-447`（保留尾累计）、`:503-560`（摘要 prompt）、`:588-596`（摘要失败分类）、`:616-720`（prepareCompaction）、`:721-742`（turn-prefix 摘要）、`:830-842`（文件操作）；`messages.ts:4-10`（摘要前缀）、`:47-62`（compactionSummary 消息）、`:142-160`（LLM 转换）；`session-manager.ts:461-489`（上下文重建）、`:1098-1120`（firstKeptEntryId）；`interactive-mode.ts:3082-3083`（`/compact`）；`overflow.ts:134-173`（overflow 识别）、`:175-179`（length 恢复）。
- omp：`session-maintenance.ts:1221-1330`（推测式）、`:1647-1763`（mid-turn）、`compaction-methods.ts:6-52`（方法链）、`snapcompact.ts`（位图压缩）——仅 §6 引用。
- grok：`compaction.rs:1701-1721`（装配后校验）、`:1929-1989`（pre-sampling）、`code_compaction/compact.rs:3-20`（full-replace）——仅 §6 引用。
- DSH：`compaction-basic/src/index.ts:150-260`（自动触发）、`:272-290`（pruner）、`summarizer.ts:29-75`（前缀重放 + 标签）、`region.ts:126-137`（配对平衡）、`token-meter`（usage 锚）——对齐引用见正文，插件化缝仅 §6。
