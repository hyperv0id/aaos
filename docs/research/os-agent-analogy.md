# OS 视角的 Agent 系统调研报告

> 调研日期：2026-08-30 · 范围：操作系统范式 ↔ agent 系统的问题、痛点与方案 · 目的：为 aaos 的架构演进提供问题地图与方案参照
> 方法：一手论文（AIOS / MemGPT / PagedAttention / SGLang / CaMeL）+ 工业界技术报告（Anthropic / Chroma）+ 协议规范（MCP / A2A）。刻意停留在架构与设计层，不展开 vendor API 细节。

## 0. 类比的支点：无状态执行体 + 外化状态

整个类比成立的关键不在"agent 像进程"，而在一个更精确的事实：

**LLM 是无状态纯函数 `f(context) → next token`；一个 agent 进程的全部状态都外化在上下文里。**

传统 OS 里进程状态散落于寄存器、内核栈、地址空间，调度器要做复杂的现场保存；agent 的"现场"只有一份——上下文。由此得到本项目最重要的一个推论：

**一切 agent 系统管理问题——调度、隔离、恢复、迁移、分叉——最终都退化为上下文管理问题。谁拥有上下文，谁就拥有这个进程。**

这正是 aaos 以会话为中心、把会话做成内容寻址派生链的正确性来源：它把"进程状态"变成了一等可管理对象。Karpathy 的 "LLM as CPU, context window as RAM" 论述是同一直觉的大众化表达。

| OS | Agent 系统 |
|---|---|
| CPU（执行体） | LLM（无状态，逐步预测） |
| 进程 / 地址空间 | 会话 / 上下文 |
| 内核 | harness（agent 运行时） |
| syscall | 工具调用 |
| 设备驱动 | 工具实现 / MCP server |
| RAM（物理内存） | 上下文窗口 |
| 磁盘 / swap | 外部存储：文件、检索、资产库 |
| fork + COW | 会话派生（前缀共享） |
| 信号 / 中断 | steering / 人机介入 |
| journal / WAL | append-only 记录 + 副作用载荷 |

以下按问题域展开：每个域给出痛点、OS 已有解法、agent 世界的对应实践（含已验证效果）。

## 1. 内存管理：上下文是稀缺资源

**痛点（已有实证）。** 上下文不是"越大越好"的免费资源：

- Chroma《Context Rot》（2025-07）测 18 个主流模型：性能随输入变长**非均匀、梯度式退化**，连机械复制文本这类零推理任务也在退化；NIAH 类基准只测词面检索，系统性高估真实长上下文能力。
- Lost in the Middle（arXiv:2307.03172）：模型对上下文首尾敏感、中部丢失。
- Anthropic（2025-09）把上下文定性为"边际收益递减的有限资源"，存在类似工作记忆上限的 attention budget。

**结论：上下文必须被"管理"，正如 RAM 必须被管理。** OS 在内存管理上发明的机制，几乎都能在 agent 世界找到对应物：

- **虚拟内存 + demand paging → 虚拟上下文管理**。MemGPT（arXiv:2310.08560，已产品化为 Letta）是自觉的实现：上下文窗口 = 主存，外部存储 = 磁盘，由模型自己决定何时换入换出（self-directed paging），并用 interrupt 管理控制流。aaos 的技能"system prompt 只注入索引、全文按需 read"已是同一模式的局部实例。
- **分页 / 页表 → PagedAttention**（vLLM，SOSP 2023）：把虚拟内存分页**字面地**用于推理层 KV cache——按需分配块、跨请求共享，做到 KV cache 近零浪费、吞吐 2–4×。这是"OS 类比能产出顶级系统成果"的最强证据。
- **路径名缓存 / radix tree → RadixAttention**（SGLang）：用 radix tree 做 KV 前缀复用 + cache-aware 调度，多轮/共享前缀场景吞吐至 6.4×。设计要点与 dcache 相同：让"前缀"成为可复用资产。
- **缓存层级思维 → prompt caching**。对 agent 架构的设计含义只有一条但很关键：**缓存命中是上下文布局的属性，不是一个开关**——稳定内容（系统提示、工具定义）前置，易变内容后置，追求前缀稳定性，等价于程序追求 cache locality。
- **swap / 存储分层 → KV cache 分层与解耦**：Mooncake（arXiv:2407.00079）、LMCache 等把 KV 在 HBM↔RAM↔SSD 间分层换出、prefill/decode 分离部署，就是 swap 与按需调页的推理层重演。
- **GC / 内存压缩 → compaction**。Anthropic 总结的长程任务三策略——压缩、结构化笔记（显式换出到"磁盘"）、子 agent（独立地址空间 + 只回传摘要，本质是消息传递替代共享内存）——全是内存管理动作。aaos 的压缩派生（区间替换映射、原文仍可寻址）对应"压缩但不丢出处"。
- **OOM killer → 当前空白**。上下文溢出时"牺牲谁、保什么"在多数系统里是报错或硬截断，没有策略层。这是一个明确的、尚未被认领的设计空间。

## 2. 进程与调度

- **进程 = 会话，fork + COW = 派生**。aaos 已实现：派生链共享前缀资产、写时才分化——语义上就是 copy-on-write。
- **上下文切换 = 上下文序列化**。AIOS（arXiv:2403.16971，COLM 2025）把调度/内存/访问控制收进 agent 内核，其 context manager 做挂起/恢复（snapshot/restore）让多个 agent 分时复用 LLM，整体提速至 2.1×。注意：因为状态全在上下文里，"切进程"和"存会话"是同一件事——aaos 的会话恢复天然就是上下文切换机制。
- **调度**：推理侧 continuous batching（Orca，OSDI 2022）把请求级调度细化到迭代级，就是时间片的引入。agent 侧调度（多会话并发、优先级、公平性）目前普遍原始：FIFO 或干脆没有。
- **配额与回收：rlimit / cgroups / OOM killer ↔ token 预算、步数上限、runaway 检测**。失控 agent 烧 token 循环就是没人 wait() 的僵尸进程。现状是各 harness 各自塞一个 max-turns 参数，没有统一的配额层与耗尽语义——又一块空白。
- **信号 / 中断 = steering**。MemGPT 用 interrupt 管理自身与用户间的控制流；aaos 的"Ctrl+C 打断但不中止 run"已是信号语义的雏形。完整的信号模型（待处理集、屏蔽、处理时机）对 steering 排队/丢弃语义有直接借鉴价值。
- **容错：supervision tree（Erlang/OTP）↔ orchestrator-worker**。let-it-crash 成立的前提是状态可恢复——于是容错问题再次落回持久化设计。

## 3. IPC：多 agent 通信

- **共享内存 vs 消息传递之争正在 agent 世界重演**。共享黑板/共享可变上下文 = 共享内存：快，但带来一致性与上下文污染问题，且目前基本没有同步纪律；A2A 式消息传递 = RPC：有边界、可审计，代价是序列化信息损失。OS 五十年的经验结论值得直接继承：**默认消息传递；共享状态是例外，且必须配同步纪律**。
- **协议分层已形成**：MCP = 外设模型（官方自比 USB-C；tools/resources/prompts 三类原语，解决 agent↔工具/数据源的集成碎片化）；A2A（Google 发起，已捐给 Linux Foundation，v1.0）= 进程间协议，Agent Card 做服务发现、Task 承载作业生命周期，且刻意 opaque（不暴露内部记忆与逻辑）。官方定位互补：MCP 管 agent↔tool，A2A 管 agent↔agent。对照 OS：一个是设备/驱动层，一个是 socket/RPC 层。
- **Unix 哲学复兴**：小工具 + 组合在 agent 工具设计上重新成为主流实践（工具即命令、输出可串联）。

## 4. 安全与隔离

**核心同构：prompt injection = 缓冲区溢出。** 冯诺依曼架构"数据与代码同处一个地址空间"的困境在 agent 上重演：指令与不可信数据混在同一份上下文里，模型自己区分——这正是栈溢出能被利用的同一结构原因。

OS 安全的机制谱系几乎逐条可映射，CaMeL（arXiv:2503.18813，Google，2025）是最完整的自觉实现：

- **W^X / NX（数据页不可执行）→ 指令/数据通道分离**：CaMeL 从可信用户查询中显式提取控制流与数据流，保证不可信数据**永远不能影响控制流**。
- **reference monitor → 外围系统层强制**：安全检查在 LLM 之外的系统层执行，不信任模型自觉——如同权限检查在内核而非用户程序。
- **capability-based security（KeyKOS / seL4）→ 工具调用授权**：每次工具调用检查"这次数据流是否持有相应 capability"，防止私有数据沿未授权路径外泄。
- **信息流控制 / taint tracking（HiStar / Flume）→ 不可信来源打标**，在 sink（工具调用点）做策略检查。
- **sandbox / namespace / seccomp → 工具执行沙箱**；**sudo 提权 → 危险操作需人类批准**（各 coding agent 的权限提示即此）。

效果参照：AgentDojo 基准上 CaMeL 以**可证明安全**完成 77% 任务，无防御基线为 84%——安全性与效用的交换比第一次有了量化数字。对 aaos 的相关性：`side_effects` 已是审计日志的底子，加上来源信任级标注即可为 taint 式策略层留位。

## 5. 持久化、恢复与可重现性

这是 aaos 投入最深、与 OS 对应最整齐的域：

- **journaling / WAL + 事务 → append-only 记录 + SQLite 事务**（ADR-0001）：崩溃一致性由事务保证，framing/torn-tail 整套机制消失——正是当年文件系统从日志到事务的演进路径。
- **fsync 边界 = 提交点设计**（ADR-0002）：MessageEnd 单通道持久化，崩溃恢复粒度到单条消息——"在何处提交"就是 fsync 边界放在哪的同一决策。
- **checkpoint / restore（CRIU）→ agent 状态挂起恢复**：LangGraph checkpointer、Temporal durable execution（事件溯源 + 重放重建状态 = 日志重放）。
- **record / replay（rr）→ agent trace 确定性重放调试**：目前几乎空白，而派生链 + 副作用载荷恰好是实现它所需的全部原料——aaos 在这个方向上有结构优势。
- **fork 炸弹防护 = 派生的深度/扇出/预算限制**：配合 §2 的配额层。

## 6. 可观测性

- **/proc 自省接口 ↔ agent 运行时暴露自身状态**：当前 agent 系统普遍只有对外日志，没有一个"内核视角"的自省面（当前跑哪些会话、各占多少上下文/预算、阻塞在哪次工具调用）。
- **strace / dtrace ↔ agent trace**：OpenTelemetry 的 GenAI 语义约定正在把 trace 标准化，等价于给 agent 世界立 syscall trace 的格式。
- **core dump ↔ 崩溃会话的事后解剖**：aaos 的派生链 + 单消息恢复粒度，使"崩溃现场"天然完整可解剖——多数系统崩溃即丢失。

## 7. 类比的边界（设计层面必须记住）

类比有用，但有三处结构性断裂，照搬会出错：

1. **硬件确定，模型随机**。OS 机制提供硬保证（隔离、互斥、配额）；agent 的"执行体"本身是随机的，机制可借、**保证要重证**（CaMeL 把检查移出模型就是这个道理的正面应用）。
2. **局部性原理弱化**。LRU 有效是因为程序有强时间局部性；上下文中"什么还相关"是语义问题而非时间问题——淘汰/压缩策略必须语义感知，不能照搬 LRU。
3. **抢占粒度**。进程可在任意指令边界抢占且现场保存廉价；LLM 只能在 token 边界让出，"现场"（KV cache）的保存/恢复有真实成本——调度器设计要以此为约束。

类比的正确用法是提问框架：**"OS 在这里发明了什么机制、维护什么不变式？"**——机制映射过来，实现重新设计。

## 8. aaos 的位置与机会点

**已经走在 OS 路上的**（多数是先做对了、后发现同构）：

| aaos 机制 | OS 同构 |
|---|---|
| 内容寻址资产（BLAKE3、写一次） | git object / inode |
| 派生（分叉/压缩统一） | fork + COW；压缩 = GC |
| 头指针（meta 表唯一可变行） | runqueue 头 / HEAD 引用 |
| MessageEnd 单通道持久化 | commit / fsync 边界 |
| side_effects（before/after 载荷） | WAL / 审计日志 |
| 技能索引 + 按需 read | demand paging + exec |
| Ctrl+C 不打断 run | 信号语义雏形 |

**机会点**（按本文问题域排序，均为设计层）：

1. **配额层**：token/步数/派生深度的统一 rlimit 语义 + 耗尽时的 OOM killer 策略（§1、§2 的共同空白）。
2. **信号语义补全**：steering 的排队/屏蔽/处理时机，对照信号模型设计（§2）。
3. **内部 URI 泛化为 VFS**：`skill://` 之外扩到 `asset://`、`session://` 等，"一切皆文件"的统一寻址（§3）。
4. **资产回收策略**：内容寻址对象的生命周期管理 = GC 问题，需要引用可达性定义（§1）。
5. **自省接口**：/proc 式的运行时状态面，供 CLI/调试消费（§6）。
6. **信任级标注**：side_effects 记录来源信任级，为 capability/taint 式策略层预留位置（§4）。
7. **确定性重放**：派生链 + 副作用载荷已具备 record/replay 原料，可做 agent 版 rr（§5）。

## 参考来源

- AIOS: LLM Agent Operating System — [arXiv:2403.16971](https://arxiv.org/abs/2403.16971)（COLM 2025）
- MemGPT: Towards LLMs as Operating Systems — [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)
- vLLM / PagedAttention — [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)（SOSP 2023）
- SGLang / RadixAttention — [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)
- CaMeL: Defeating Prompt Injections by Design — [arXiv:2503.18813](https://arxiv.org/abs/2503.18813)
- Chroma: Context Rot — [trychroma.com/research/context-rot](https://www.trychroma.com/research/context-rot)（2025-07）
- Anthropic: Effective context engineering for AI agents — [anthropic.com/engineering/effective-context-engineering-for-ai-agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)（2025-09）
- Lost in the Middle — [arXiv:2307.03172](https://arxiv.org/abs/2307.03172)
- Mooncake（KV cache 分层/解耦） — [arXiv:2407.00079](https://arxiv.org/abs/2407.00079)
- MCP — [modelcontextprotocol.io](https://modelcontextprotocol.io/introduction)；A2A — [a2a-protocol.org](https://a2a-protocol.org/latest/)（Linux Foundation）
