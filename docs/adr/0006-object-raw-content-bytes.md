# 0006 · 资产 = 裸内容字节，结构与元信息收编 DB（出处单轨）

对象文件此前是 `Segment` 的 canonical JSON 信封（`serde_json::to_vec(segment)`），打开是 `{"type":"assistant","content":[…]}`，不是内容本身——与 CONTEXT.md 的资产定义（不可变内容）背离，并阻断已定目标：对话内容（人类/AI/工具调用/调用结果）作为对话管理对象（OS 意义的内存块）并作为文件让 LLM 直接可重读，JSON 包一层做不到。本 ADR 定下新编码：**对象 = 本身即可读的内容字节；结构与元信息 = DB**，判别式是字节离开 DB 后是否自足可读——是 → 对象，否 → DB 列；编码按**块**粒度（assistant 消息天然多块：thinking + text + 多个 tool_call；消息粒度必须发明分隔/转义，即换皮信封）；并对 [ADR-0001](0001-sqlite-structural-source-of-truth.md) 做一处修订：出处双轨 → 仅结构轨，删 `SummarySegment.sources`。决定性理由：块粒度是唯一不发明分隔/转义的编码，让每块字节离开 DB 即自足可读（text 原文、图片本体、canonical JSON 都可直接由 LLM 重读）；块粒度后消息无单一哈希，sources 无直接后继，而生产路径仅 `compact()`、结构轨（compactions 区间映射）全覆盖，删 sources 顺带消除长会话的哈希膨胀。

## 决定细节

- **判别式**：字节离开 DB 后是否自足可读。是 → 对象（进 ObjectStore，按内容哈希寻址）；否（如块归属、mime_type、tool_call id/name、时间戳）→ DB 列。对象本身不再携带任何结构与元信息。
- **对象编码按块粒度**：assistant 消息天然多块（thinking + text + 多个 tool_call），块粒度免去消息级的分隔/转义发明；各块编码：
  - text / thinking → UTF-8 原文；
  - image → 图片字节本体（mime_type 入 DB）；
  - tool_call → arguments 的 canonical JSON（id / name 入 DB）；
  - tool_result details → canonical JSON；
  - summary → 摘要文本。
- **DB 收编**：`entries` 加列 `created_at`（修复时间戳丢失：`convert.rs` 文档声称 ts 在 log records，ADR-0001 废日志链后无处可依，读路径只能以 `now()` 冒充）、`stop_reason`、`model`、`provider`、`api`、`usage`、`error_message`、`is_error`、`added_tool_names`；新表 `entry_blocks(session_id, seq, idx, kind, hash, mime_type, tool_call_id, tool_name)` 记每条消息的块清单；`compactions` 加 `model` 列（摘要生成方），`summary_hash` 语义变为摘要文本对象。
- **出处双轨 → 仅结构轨（对 ADR-0001 的修订）**：删除 `SummarySegment.sources`（含 `compaction_coordinator` 的 sources 构造链，`aaos-cli/src/main.rs:394-455` 一带）；原文取回唯一走 `fetch_originals`。理由：块粒度后消息无单一哈希，sources 无直接后继；生产路径仅 `compact()`，结构轨全覆盖；顺带消除长会话哈希膨胀（`docs/research/compaction.md` R6）。
- **Segment 保留为内存货币**：`append_segment` / `materialize` 签名不变，内部分解 / 重组；死掉的只是磁盘编码。[ADR-0002](0002-session-absorbs-agent-integration.md) 的 Message↔Segment 同构不受影响。

## Considered Options

- **消息粒度编码**（消息整体串成对象字节）：assistant 消息天然多块，消息粒度必须发明分隔/转义才能把 thinking/text/tool_call 装进一个字节串——换皮信封，与现 JSON 信封同病。否决：块粒度。每块自足可读，块归属由 `entry_blocks` 清单承担。
- **保留双轨出处**（sources + 结构轨并存，沿用 ADR-0001 原状）：块粒度后消息无单一哈希可记，sources 无直接后继；sources 还随压缩嵌套膨胀（长会话哈希膨胀，`docs/research/compaction.md` R6）；结构轨经 `fetch_originals` 已全覆盖生产路径。否决：出处单轨，删 `SummarySegment.sources`。

## Consequences

- canonical 编码重定义 → 现存对象哈希全部失效；迁移决定：wipe `~/.config/aaos`（项目未发布、本机数据，不写迁移脚本）。
- `ObjectStore` 退为纯字节库（typed `put`/`get` 删除）。
- view fold 从哈希列表坐标化为 `(session_id, seq)` 列表。
- `preserve_order` NOTE 仍适用（arguments / details 仍走 Value → canonical JSON）。
- 内存块文件面（导出 / 渲染、LLM 重读视图）另立 issue，本决策只解除存储层阻断。
- 词汇：CONTEXT.md「资产 (Object)」词条随本 ADR 更新为块粒度裸字节定义；ADR-0001「出处双轨」条款被本 ADR 取代，见其「## 修订」条目。
