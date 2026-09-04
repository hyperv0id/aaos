# 0007 · 压缩摘要 = 确定性转录：块是页、压缩是装卸载、对象路径是可回读地址（否决 LLM 摘要）

`docs/research/compaction.md`（2026-08-31，draft）的摘要设计是 LLM 生成的：独立 `Agent` 实例、结构化 prompt、增量更新、`summaryMaxTokens` 上界（D3、D5 turn-prefix 摘要、§3.2、§3.3、P1.4）。该子系统**已删除（owner 决策，commit 048e79e "feat: context compaction with deterministic transcript (#70)"）**：压缩后的 Summary 内容不再来自任何模型调用，而是 `aaos_session::compaction::build_transcript` 拼出的确定性转录。owner 的心智模型是 OS 虚拟内存：**消息块 = 页；压缩 = 换出；内容变成可回读的绝对路径；agent 需要时用 `read` 工具按需换入**。ADR-0006 已把对象做成块粒度的裸自读字节——正是这个分页模型的 backing store；压缩节点内嵌的转录就是对换出页的描述，对象路径就是可回读的虚拟地址。本 ADR 一并记录随之定下的陈旧锚规则与单一渲染。

## 决定细节

- **分页模型词汇**：块（content block）= 页；压缩（compaction 派生）= 换出/卸载，保留段 = 换入/驻留；转录与保留段内引用的对象绝对路径 = 可回读地址；agent 经 `read` 工具按需读回（换入）。`build_transcript` 产出的转录本身就是 `Segment::Summary` 的内容（`compaction.rs` 模块文档明言 "No model involvement — compaction never calls an LLM"）。
- **对象路径按块粒度重算**：`build_transcript` 引用路径来自对段自身字节的哈希重算（`hash_hex(block_bytes(block))`，与写路径共用 `db::block_bytes` 的编码规则），不读 `entry_blocks`。内容寻址下 store 出来的段 encode∘decode 为恒等，重算即精确——调用方契约：传 materialize 过的段，引用路径必然存在。
- **转录渲染规则**（`build_transcript`，逐条对应实现）：
  - 首行 `TRANSCRIPT_PREAMBLE`：声明历史已被压缩成该转录、完整输出与参数在引用的绝对路径上、可用 `read` 工具读取；
  - user/assistant 对话文本逐字内联：`[User] {text}` / `[Assistant] {text}`；
  - **thinking 块整体丢弃**（不是换出——见下）；
  - tool_call 序列化参数 ≤ 100 字符内联为 `[Tool call] name({args})`；更长者截 100 字符预览并附对象路径：`[Tool call] name({preview}…) — full arguments at {path}`（按字符计：canonical JSON 可含多字节 UTF-8）；
  - 图片为 `[Image] at {path}`（路径指向存图片裸字节的对象）；
  - tool_result 每个内容块一行 `[Tool result] full output at {path}`，`details` 非空另加 `[Tool result] details at {path}`；两者皆空渲染 `[Tool result] (empty)`；
  - 区间内若嵌有 `Segment::Summary`，其内容原样内联——它本身已是带路径的转录，再压缩保持可传递（transitive）。
- **thinking 丢弃的理由**：thinking 是**过程不是内容**（`compaction.rs`："Thinking is process, not content: dropped, not unloaded"）；原文不因丢弃而失——压缩节点的 `fetch_originals` 照常取回被压缩区间的全部原文段（含 thinking），取证路径未断。丢弃不写在转录里，转录里只保留"供续写消费"的内容。
- **陈旧锚规则**：从压缩派生出的视图（`resume` / 注入视图）**不携带任何压缩前的 usage 锚**——段 → 消息视图边界处清空 assistant 的 `usage`（仅内存；持久化 `entries.usage` 保持忠实原值，计量回退到 chars/4 估计，直到下一条 assistant 响应落地新锚）。理由：`usage.total_tokens` 描述的是**压缩前**的上下文；不清零既会让触发误判（陈旧锚 + 新追加的估计立刻超阈值，压缩后立刻再压缩），也破坏"投影必须变小"校验（压缩后视图带上压缩前的大锚，`after_tokens ≥ before_tokens` 恒拒）。
- **单一渲染**：Summary 处处渲染为**裸 user 消息**——`view_messages` 已是如此（`Err(ConvertError::Summary) => Message::User(UserMessage::new(content))`），resume 路径的 `"[compacted summary] "` 前缀**删除**，与 `view_messages` 收敛为一种形式。转录首行 `TRANSCRIPT_PREAMBLE` 已自我说明（"The conversation history before this point was compacted into this transcript…"），前缀冗余。
- **出处维持 ADR-0006 的结构单轨**：Summary 段不带 `sources`（代码里没有、也不会复活）；原文取回唯一走 `fetch_originals`。转录内的块路径是内容寻址引用，不是 provenance 记录。
- **切点**：`find_cut_point` 从尾向头累计保留尾（`keep_recent_tokens` 默认 20000），候选前移到最近合法边界（不可切在 toolResult 上、切点前不可有悬空 tool_call 配对）；保留尾覆盖全部 → `Nothing to compact`。不建模 pi 的 `isSplitTurn`：转录嵌入整个被压缩区间，turn 中间切割无需 turn-prefix 摘要。压缩映射为单区间 `[(0, first_kept)]`。
- **投影校验**：压缩后视图（转录 + 保留尾）估计 token ≥ 压缩前 → 拒绝（"compaction would not reduce context"），不落节点。
- **no-LLM 边界**：协调器只保留活模型的 `context_window` 供触发检查，从不重新解析模型、从不调用模型（`compaction_coordinator.rs` 设计注）。`aaos_session::compaction` 为纯逻辑模块（计量、切点、转录），契约测试在 `crates/aaos-session/tests/compaction_contract.rs`。

## Considered Options

- **LLM 摘要（`docs/research/compaction.md` §3.2–3.3、D5 turn-prefix、P1.4）**：独立 `Agent` 实例（不 attach 持久化 listener、复用会话前缀保 KV cache）、结构化摘要 prompt（`SUMMARIZATION_SYSTEM_PROMPT` + `<conversation>…</conversation>` 序列化 + `<previous-summary>` 增量更新 + Goal/Constraints/… 模板）、输出以 `summaryMaxTokens`（`min(floor(0.8 × reserveTokens), model.max_tokens)`）上界。**否决**：成本与延迟（每次压缩一次完整模型调用）、非确定性（同输入不同摘要，压缩节点内容不可复现）、摘要失败路径（失败/中止/降级时压缩如何落库——`summaryMaxTokens` 只是软预算，模型可超发），且与 owner 的分页心智模型根本不合——摘要模型"理解并改写"历史，分页模型只要求历史可回读、上下文可换出。诚实记录其唯一真实优势：**稳态体积有界**——转录把对话文本钉死内联，跨多次压缩线性增长（无 LLM 上界）；LLM 摘要则每次把旧文本折叠成有界摘要。此代价被接受，理由：对话文本是**指令流**，必须驻留（`[User]`/`[Assistant]` 逐字内联），截断它等于截断指令；体积大头是 thinking 与工具输出，thinking 已整体丢弃、工具输出已换出为路径引用——转录体积的增长项只剩指令文本本身，按字符计费、缓慢且可预期。
- **thinking 渲染为路径引用**（丢弃之外的另一候选）：thinking 原文在对象存储中自足可读，转录里可以只写 `at {path}` 让 agent 按需读回，与工具输出同等处理。**否决（owner 选择整体丢弃）**：thinking 是过程不是内容，续写不需要它；若真需要，`fetch_originals` 一条命令即可取证，不必为过程内容保留常驻引用、白白占据上下文预算（按路径引用仍要花 token 才能决定是否读）。
- （继承自 ADR-0006 的区间替换映射 + 原文可寻址结构不重开；`sources` 不复活。）

## Consequences

- **转录体积随对话文本线性增长**，无 LLM 摘要上界：多次压缩后转录内嵌早先的 `[User]`/`[Assistant]` 文本仍逐字在列。缓解：每次投影校验保证单次压缩确实变小；对话文本是全部消息里增速最慢、语义密度最高的一部分（每字符都可能是指令）。
- 压缩后视图的计量暂时回到纯估计（陈旧锚规则）：`context_tokens` 无锚可用即全量 chars/4 估计，触发阈值可能偏松/偏紧直到下一条 assistant 响应落地——一次交换内的临时偏差，代价低于误触发。
- `docs/research/compaction.md` 的机制节（D3、D5 turn-prefix、§3.2、§3.3、P1.4、§2.3 `summaryMaxTokens`）自此**仅作历史记录**（状态横幅已改注）。
- 压缩节点上 `fetch_originals` 是原文唯一取证路径（结构轨）；thinking 丢弃后无转录内引用，取证依赖它。
- 分页模型后，研究文档 §1.2 已 defer 的逐块老化/常态置换方向（toolResult 修剪）是二期候选：本 ADR 只定"换出整段前缀"，块级置换（把仍超阈值视图里的旧 toolResult 块再换出）留待那期。
- 单一渲染落地后测试断言如实反映：摘要消息的渲染文本**精确等于** summary content（裸 user 消息，无前缀）——`compaction_contract.rs` 断言 `text == "first turn summarized"`（bare summary content），`main.rs` 的 `happy_path_creates_node_and_resumes` 与 `recompaction_embeds_previous_transcript` 均断言 `first_text(messages[0]) == summary.content`；`TRANSCRIPT_PREAMBLE` 首行属转录内容的自带属性，仅以 `contains` 作内容健全性检查，不作渲染判别式。
