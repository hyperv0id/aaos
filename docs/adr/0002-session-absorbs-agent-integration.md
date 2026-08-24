# 0002 · aaos-session 吸收 Agent 集成层，段落按 MessageEnd 持久化

`aaos-session-store` 升级为 `aaos-session`，吸收此前缺失的 Agent↔store 集成层：直接依赖 `pi-agent-core`，持有同构的 `Message`↔`Segment` 转换，并对外暴露 turn 级（追加一轮对话、恢复到 `AgentState`、按工具执行捕获副作用）操作。CLI REPL 以会话节点为入口、向叶子追加子节点。决定性理由：gap 是结构性的——store 有完整会话模型却无任何路径接进 Agent 循环，CLI 因此只能单发；转换经核实为同构（字段一一对应，仅 `timestamp` 与 store-native `SummarySegment` 两侧不对齐），集成成本是 `From`/`Into` impl，不值得再开一个桥接 crate；存储内部已按工具执行时记副作用设计，段落持久化应对齐 Agent 自身的状态提交点（`MessageEnd` 时 `state.messages.push`）以获逐条消息崩溃恢复，无需额外回合边界刷新机制。

## 决定细节

- **升级吸收集成层**：`aaos-session-store` → `aaos-session`。新 crate 依赖 `pi-agent-core`，承担：`Message`→`Segment`（写）与 `Segment`→`Message`（读）的同构转换、turn 级追加、resume 到 `AgentState`、按工具执行的副作用捕获。不再有独立桥接 crate。
- **会话 = 节点树，对话 = 一条路径**：现有 `sessions` 表即节点树（`id` 是节点，`parent_id`/`parent_position` 是边）。resume = 从某个节点开始；续写 = 向该节点追加子节点；`fork`/`fork_at` = 派生子节点。叶子是默认续写目标（`latest_session` 查询），但模型本身不分叶子/根——任何节点都能 resume。
- **段落按 MessageEnd 持久化（单通道）**：Agent 在 `AgentEvent::MessageEnd` 把消息 push 进 `state.messages`（`agent.rs:644-646`），`aaos-session` 在同一边界追加对应 `Segment`。经规格验证核实：**一切**入态消息——user prompt、steering、assistant、tool result——都经 MessageEnd 提交；tool result 在定稿时由 `emit_tool_result_message`（tool_engine.rs:439）发射 MessageEnd，`TurnEnd.tool_results` 是同一批消息的重复携带，持久化只认 MessageEnd 一次，无角色分派、无二次追加。副作用在 `AgentEvent::ToolExecutionEnd` 经 `append_side_effect` 记录（store 已有设计；before/after 数据来自 `before_tool_call` 钩子捕获，见规格 §5.2）。回合 = 一串段落追加，无独立回合边界刷新；崩溃恢复粒度到单条消息——但中断会留下悬空 `tool_call`，resume 侧负责配对修复（规格 §3.4）。
- **同构转换的事实基础**：`Segment` 各变体与 `Message` 各变体字段一一对应；`ContentBlock`/`ToolCall`/`Usage`/`Cost`/`StopReason`/`ImageSource` 两侧行状一致（store 侧 `Serialize/Deserialize`，agent 侧内存）。差异仅：`timestamp`（agent 有、store 丢——对象去重要求，blob-vs-commit 切分，见 `segment.rs` 文档）、`SummarySegment`（store-native，压缩出处，agent 无对应）。

## Considered Options

- **独立桥接 crate（`aaos-agent-session`）扇入两个核心 crate**：`segment.rs` 文档预设此路（"a later bridge crate only moves fields"）。被否决：转换同构到零成本，桥接 crate 是空壳；用户明确要"升级吸收"。桥接的接缝价值（可测、不污染双方）由 `aaos-session` 内部模块边界保留即可，不必跨 crate。
- **集成层放 `aaos-cli`（当前唯一消费方）**：更窄的接缝，但押注单一前端；非目标已点名 TUI，吸收进 `aaos-session` 为多前端预留同一入口。决策时未考察此选项，补记。
- **`pi-agent-core` 内可选 `persistence: Option<SessionBackend>` trait**：让 agent 感知 session 概念。否决：违背其作为 Pi `Agent` 类忠实移植的无状态设计，泄漏领域概念进核心。
- **段落按 TurnEnd / AgentEnd 持久化**：粗粒度、更简单。否决：丢失逐消息崩溃恢复；与 store 既有副作用按工具执行记录的设计��奏不一致。
- **event-driven 逐事件追加（`ToolExecutionEnd` 即写 `ToolResultSegment`）**：最细粒度。否决（理由修正，2026-08-23）：初稿理由"顺序与 `state.messages` 提交点解耦"**前提不成立**——tool result 正常路径本就经 MessageEnd 提交（`emit_tool_result_message`），并未与提交点解耦。保留否决的正确理由：MessageEnd 单通道已在同一时点覆盖全部段落，另开一条写路径只会引入双投递去重；`ToolExecutionEnd` 边界留给副作用轴。

## Consequences

- `aaos-session-store` crate 重命名 + 新增 `pi-agent-core` 依赖；`aaos-session-store` 的既有 import 路径全量改名。
- 新增集成模块（转换 + turn 级操作 + resume）；`Segment`↔`Message` 的 `From`/`Into` impl 居此。
- `aaos-cli` 升级为 REPL：以会话节点为入口（`latest_session` 为默认续写目标），`loop { read → prompt/continue_run → print }`，`MessageEnd` 处追加段落，`ToolExecutionEnd` 处记副作用。
- `aaos-session-store` 的六个 scratch issues（01–06）全部是 store 内部机制，不触及此集成；集成是新工作面，需另开 ticket。
- 词汇：根 `CONTEXT.md`（会话、派生、分叉、压缩、书签、视图、副作用、资产）；机制：`docs/adr/0001-sqlite-structural-source-of-truth.md`。
