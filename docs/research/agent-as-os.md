# 从 OS 视角看 agent：问题、痛点与设计映射

调研报告 · 2026-08-30。回答一个问题：**用操作系统的概念框架审视 LLM agent，能发现哪些共同模式、哪些真痛点、OS 的哪些成熟解法可以直接映射为 agent 的设计**。本文写于架构/设计层，不深入具体 API 参数；涉及现状的论断均给一手出处。

## 0 · 方法论：为什么 OS 是好的类比来源

OS 与 agent 面对的是**同四类根本问题**，只是资源类型不同（OS 管理硅与内存，agent 管理注意力与 token）：

1. **有限快速资源上的工作集管理**——OS：内存层级、缓存、swap；agent：上下文窗口、KV/prompt 缓存、压缩。
2. **隔离计算单元的生命周期、切换与复用**——OS：进程、调度、信号；agent：agent loop、会话、多 agent 编排。
3. **隔离单元之间交换信息而不共享内部状态**——OS：IPC、管道、socket、命名；agent：子 agent 通信、MCP、共享工作区。
4. **失效恢复与对不可信代码的资源保护**——OS：WAL、事务、capability、沙箱；agent：会话恢复、副作用回滚、工具权限。

OS 在这四类问题上有 60 年被验证的答案与失败的坑。agent 生态正在**独立重演**这些问题——MemGPT 显式以 OS 自居（虚拟内存分页管理 LLM 上下文，[arXiv 2310.08560]），Claude Code 的沙箱直接落在 OS 原语上。重演时可以抄答案，也会踩 OS 踩过的坑。aaos 自身的演化已经在印证：ADR-0001 否决「overlays 原地覆盖行」、选择 append-only 派生，正是 OS「日志结构 + 事务重放」战胜「原地打补丁」的重演。

### 映射总表

| OS 概念 | agent 对应物 | 共同本质问题 | aaos 现状 | 缺口 |
|---|---|---|---|---|
| 内存层级 / cache | 上下文窗口 / KV·prompt 缓存 | 工作集 > 快速资源；前缀失效 | append-only 资产天然前缀稳定 | 缓存感知的上下文策略未成文 |
| demand paging | 技能索引注入 + 按需 read | 按需调入，别预装 | **已实现**（skill://） | prefetch / 常驻集估计 |
| swap / GC | 压缩（compaction） | 回收空间但不破坏可达性 | **已实现**（区间替换映射，原文可寻址） | 压缩时机与缓存击穿的调度 |
| COW fork | 会话分叉 | 复制昂贵，共享只读前缀 | **已实现** | — |
| 进程 / PC | 会话 + agent loop | 执行上下文与生命周期 | 部分实现 | 运行时状态机未显式建模 |
| 抢占 / 信号 | steering / abort | 中断与优雅关闭 | steering 已有 | SIGTERM 级优雅中止缺位 |
| 僵尸收割 | 悬空 tool_call 修复 | 中断后资源回收 | **已实现**（resume 配对） | — |
| checkpoint/restore | 会话恢复 | 快照与恢复的粒度 | **已实现**（消息级 + 头指针） | — |
| pipe / RPC | 子 agent 返回值 | 有损/无损耗信息传递 | 无子 agent | 结构化返回协议 |
| 共享内存 | 共享工作区 | 无同步则竞态 | 单会话，无并发防护 | 锁 / 租约惯例 |
| socket / 设备驱动 | MCP | 标准化的外设接口 | 未接入 | MCP 客户端 |
| mount namespace | 内部 URI（skill://） | per-process 命名空间 | 第一步已落 | 随会话挂载更多 scheme |
| cgroups / rlimit | token / 成本 / 并发配额 | 资源配额与限额 | 无 | 配额抽象 |
| capability / ACL | 工具权限 | 最小权限 | 无 | 能力句柄式权限 |
| 沙箱（seccomp 等） | 命令隔离 | 不可信代码 | 无（可借 OS 件） | OS 沙箱集成 |
| 审计日志 | 副作用 before/after | 事后追责与回溯 | **已实现**（记录） | 无消费方（undo 未建） |
| WAL / undo logging | 工具副作用回滚 | 崩溃时世界与日志一致 | 内部状态已解决 | 外部副作用不可回滚 |
| 快照 / 层叠 fs | workspace 回滚 | 世界级回滚 | 无 | fs 快照 / git 事务 |
| saga / 补偿 | 不可逆操作的补偿 | 不可逆操作的语义 | 无 | 补偿式工具设计 |

---

## 1 · 内存层级与缓存：上下文是「按前缀寻址的主存」

**共同问题**：工作集大于快速资源；快速资源（页缓存 / KV cache）按地址（前缀）组织，失效沿地址向下传播。

### agent 现状与痛点

- 上下文窗口 = 主存；provider 侧的 prompt/KV 缓存 = 对**只追加前缀**的内容寻址缓存。三家 provider 的缓存语义在架构上同构：命中要求前缀**逐字节一致**，失效是前缀级的——改了前缀中任何字节，其后所有缓存作废（[Anthropic][pc-anthropic] / [OpenAI][pc-openai] / [Gemini][pc-gemini]）。这不是实现细节，而是**设计约束**：agent 的上下文组织方式直接决定缓存命中率，进而决定成本与延迟的数量级。
- **痛点 1 · 动态内容毒化前缀**：时间戳、随机 id、每次都变的工具清单放进 system prompt 或历史前部，即毁掉全部命中。OS 对应物是「把热数据写进 cache 专用页 vs 混入脏页」的布局问题。
- **痛点 2 · 压缩击穿缓存**：压缩 = 改写历史 = 前缀突变 = 缓存全 miss + 一笔「重新预热」的成本尖峰；OpenAI 文档明确将「总结、压缩、截断」列为破坏复用的首要因素 [pc-openai]。这是 swap 语义与 GC 语义的混用：**swap 搬移页而不改写内容，GC 才改写**。实践中 Claude Code 触发 auto-compact 的时机（接近窗口上限时）恰好是最坏的时机——在任务中途最需要缓存连续性的时候击穿它（[社区分析][compact-analysis]，官方 [compaction 文档][cc-compaction]）。
- **痛点 3 · 层级缺失**：什么常驻（system prompt、技能索引）、什么按需（文件全文、检索结果）、什么外置（摘要库、向量库）没有系统化的分层策略；「lost in the middle」类研究说明窗口中部的注意力衰减让层级设计不是省钱问题而是正确性问题 [litm]。
- **痛点 4 · OOM 处理粗糙**：超窗时的选项（硬截断 / 压缩 / 拒绝 / 丢弃旧工具结果——Claude Code 的 tool-result clearing 即「优先丢可再生页」[cc-context-eng]）对应 OS OOM killer 的「牺牲低价值进程」，但 agent 侧普遍缺少「页价值」的显式模型。

### OS 答案 → 设计原则

1. **上下文组织为「不可变前缀 + 追加尾部」**：稳定内容（身份、技能索引、长期事实）前置且字节级不变；易变内容一律后置。这是可直接写进设计守则的不变量。
2. **分层与按需调入**：L1 = 常驻集（索引），L2 = 按需页（文件/技能全文），L3 = 外存（aaos 资产库）。aaos 的技能机制（system prompt 只注入索引、全文按需 read）**已经是 demand paging 的正确实现**，且内容寻址资产让「页框」天然去重。
3. **压缩要区分搬移与改写**：aaos 的压缩是区间替换映射 + 原文仍可寻址——即「GC 保留指向被移对象的指针」，在缓存语义上意味着压缩后可以**恢复原文重建旧前缀**，这是多数 agent 做不到的架构红利，应保持并显式利用。
4. **压缩时机调度**：在任务边界 / 缓存已过期的空闲期压缩，而不是窗口将满时被迫压缩——对应 OS 把 reclaim 放到低负载期。

---

## 2 · 进程抽象与调度：切换太贵，决定了调度必须粗粒度

**共同问题**：计算单元的生命周期管理、中断与恢复、切换代价与调度策略的匹配。

### agent 现状与痛点

- **agent = 进程**：agent loop 的 next-token 循环 ≈ 取指-执行；messages ≈ 地址空间；窗口上限 ≈ 地址空间上限。会话（aaos 语义）≈ 可持久化、可派生的执行上下文。
- **痛点 1 · 上下文切换昂贵三个数量级**：OS 进程切换微秒级；agent 切换 = 重建执行现场（KV cache miss、重喂历史、重新加载技能），秒级且按 token 计费。OS 的调度器可以频繁抢占轮转；agent 的调度必须**粗粒度、少切换、高亲和**——同一任务粘住同一会话直到完成。这反过来是 aaos「以会话为中心」的调度学依据。
- **痛点 2 · 抢占点稀疏**：agent 只能在轮次/工具边界被抢占；工具执行中不可中断（一个跑 10 分钟的 bash 无法 SIGSTOP）。aaos 的 steering 是「用户态中断」——消息在下一个安全点注入（[pi-agent-core 的 steering hook][pa-agent]），这正是 OS 「信号只在中断点投递」的语义。
- **痛点 3 · 只有 SIGKILL，没有 SIGTERM**：现有的 abort 是立即中止，留下悬空 tool_call（aaos 用 resume 配对修复，即 **reaping**——把僵尸收割做进了恢复路径，这是正确的）。缺的是**优雅中止**：通知 agent 收尾当前工具、持久化边界状态、然后退出。对应 OS：Ctrl-C 默认 SIGTERM（进程可安装 handler），SIGKILL 是最后手段。
- **痛点 4 · 子 agent = fork 后只回传自由文本**：Claude Code 的 subagent 拥有独立上下文、不共享父对话、完成后**只把摘要返回主对话**（[官方文档][cc-subagents]）；OpenAI Agents SDK 的 handoffs 同样以「控制权移交」为中心（[handoffs 文档][oai-handoffs]）。设计问题：fork 的返回值是**有损压缩**的，且格式是自由文本而非结构化 artifact——OS 里 fork 返回 exit code、exec 前约定好管道协议；agent 生态尚未沉淀「返回协议」标准。OS 对应物还有另一层：子 agent 并发上限与嵌套深度限制（Claude Code 默认并发 20、深度 3）本质是**进程表与递归 fork 的资源守恒**。
- **痛点 5 · 生命周期状态机未显式建模**：OS 有 running/ready/blocked/zombie 的清晰状态机；agent 的「运行中 / 排队 / 等审批 / 等用户 / 已结束 / 悬空」散落在实现里，没有统一抽象。等审批 = blocked on IO，可能永久阻塞——需要超时与唤醒机制（watchdog），这是「优先级反转」的 agent 版：一个低优先级任务挂着审批，阻塞整条流水线。
- **痛点 6 · checkpoint/restore 已收敛，粒度是战场**：OS 侧 CRIU [criu] 做进程级快照恢复；agent 侧 aaos 已有消息级持久化 + 头指针恢复（[ADR-0002][adr2]），Claude Code 每条用户提示自动快照、代码与对话可分别回滚（[checkpointing 文档][cc-checkpointing]）。**快照频率与切换代价的权衡**是共同的设计问题：快照太频繁 = 写放大，太稀疏 = 丢失恢复点。

### OS 答案 → 设计原则

1. **切换昂贵 → 会话亲和、批处理、少 spawn**；把「spawn 一个子 agent」当成 fork 一台虚拟机的代价来对待。
2. **抢占点显式化**：在轮次边界、工具调用之间定义「安全点」；中断信号只在安全点投递（aaos steering 已符合）。
3. **补齐 SIGTERM**：优雅中止 = 「完成当前工具 + 提交边界 + 退出」，SIGKILL 只留给超时。看门狗（工具级超时）对应 OS 定时信号。
4. **子 agent 返回协议结构化**：exit code（成败）+ 管道（结构化 artifact，走内容寻址资产）+ 摘要（给人看）。aaos 的资产 ID 天然是「fd」——返回 artifact 的引用而非内容本身，主 agent 按需 read（又回到 demand paging）。
5. **显式生命周期状态机**：会话应有声明的状态（running / waiting-approval / idle / reaped），调度器据此决策；aaos 的书签（snapshot）是状态锚点的现成机制。

---

## 3 · 通信与命名：一切资源皆 URI

**共同问题**：隔离单元之间交换信息而不共享内部状态；命名空间决定可见性。

### agent 现状与痛点

- **MCP = 总线/设备驱动层**：官方定位「AI 应用的 USB-C 接口」（[MCP intro][mcp-intro]）。架构上是 host 内多个 client 对多个 server 的连接（一连接一 client），数据层 JSON-RPC + 三类 server 原语——tools（动作）、resources（上下文数据）、prompts（模板）——发现与能力协商先行，传输层 stdio（本地）或 Streamable HTTP（远程 + OAuth）（[MCP architecture][mcp-arch]）。OS 类比：stdio 传输 ≈ 管道对；MCP 连接 ≈ 打开的设备文件；`*/list` ≈ 设备枚举。协议仍在快速演化（2026-07-28 版本将 sampling 降级、新增 durable Tasks 扩展）——**agent 的「驱动生态」一年一换代，接入层必须隔离这种易变性**。
- **A2A = 跨机 IPC**：agent 间协作的开放标准，Google 发起、已捐 Linux Foundation，v1.0（[A2A][a2a]）。设计上强调 agent 保持**不透明**——不共享内部记忆/工具，只通过声明的接口交换任务与工件。OS 类比：不是共享内存而是消息传递 + 接口发现；「opaque agent」正是 OS 强隔离进程间「只能走协议」的原则。
- **痛点 1 · 共享内存没有同步原语**：多 agent / 多会话写同一工作区 = 两个进程写同一片共享内存，而 agent 生态几乎没有普及锁的惯例（git 的 index.lock 是唯一广泛存在的例子）。竞态后果：互相覆盖、读到半成品状态。
- **痛点 2 · 命名即权限**：技能发现（aaos：项目级压用户级）、MCP server 列表、tool 清单——**谁能给 agent 注入什么名字，就是谁能决定 agent 看见什么世界**。OS 的教训：PATH 劫持、命名空间逃逸。Claude Code 沙箱明确禁止写入自身配置与 shell 启动文件以防「自我提权」（[sandboxing 文档][cc-sandbox]）——防的正是命名层投毒。
- **方向 · per-process 命名空间**：Plan 9 的 per-process namespace（[Plan 9][plan9]）与 aaos 内部 URI（`skill://`）同向：**每个会话可以挂载不同的命名空间视图**。内部 URI 是第一步；终局是会话 = 进程 = 有自己的资源视图（skill://、file://（受限）、mcp://、session://），read/write 是统一 syscall。「everything is a file / URI」让权限、审计、缓存都收敛到一个寻址层。

### OS 答案 → 设计原则

1. **接入层隔离协议易变性**：MCP/A2A 都是「外设」，挂在自己的总线抽象后面，驱动（adapter）可插拔——aaos providers 的方言 adapter 是同一设计思路的复用。
2. **共享工作区需要同步惯例**：文件锁（index.lock 模式）、租约（带超时的锁）、乐观并发（改前校验内容寻址 ID——aaos 资产天然支持 CAS 式「期望旧值」检查）。多会话并发写是未来必答题。
3. **命名空间按会话挂载**：内部 URI 扩展为 scheme 集合；会话级挂载表决定可见集，权限与审计挂在命名解析上。

---

## 4 · 资源管理与保护：配额、能力与沙箱

**共同问题**：不可信代码在共享资源上运行；需要配额、最小权限与隔离。

### agent 现状与痛点

- **痛点 1 · 配额抽象缺位**：OS 有 rlimit/cgroups/ioctl 配额；agent 生态的 token 预算、成本上限、并发工具数、MCP 连接数散落在各产品的开关里，没有统一的「资源计量 + 限额 + 优先级」抽象。一个会话烧掉无限 token，与一个进程吃光内存没有 OOM killer 制止，是同构的问题。
- **痛点 2 · 权限模型太粗**：现状普遍是「全有全无 + 确认疲劳」——允许 bash ≈ 允许一切。OS 的答案是 **capability**：权限是「打开的句柄」而非「全局开关」——fd 本身就是不可伪造的能力。映射到 agent：路径白名单、只读视图、网络域名白名单应该是**会话开始时授予的能力对象**，工具只能通过能力访问资源，而不是每次调用时人肉裁决。
- **痛点 3 · 沙箱应该借 OS 的，不该自造**：Claude Code 的沙箱就是 OS 原语——macOS Seatbelt、Linux bubblewrap，网络走沙箱外代理做域名白名单；并给出了一个干净的分界：**权限系统决定「是否运行」，沙箱决定「运行后能碰什么」**（[sandboxing 文档][cc-sandbox]）。aaos 用 Rust 实现，无需重造——进程级隔离交给 OS 件即可。
- **痛点 4 · 审计记录有了，消费方没有**：aaos 的副作用（工具执行的 before/after 载荷、沿链继承、内容寻址）在结构上等价于 **syscall 审计日志**，且因 append-only + 内容寻址而**agent 自身不可篡改**——这是多数 agent 框架没有的性质。但记录尚未被消费：没有 undo、没有 diff 报告、没有「谁在什么时候改了什么」的查询面。

### OS 答案 → 设计原则

1. **资源接口收敛为「计量 + 限额 + 优先级」三元组**（cgroup 三件套）：token/成本/并发都走同一抽象，会话与会话组（进程组！）可以树形分配预算。
2. **权限 = 能力句柄**：授权发生在「打开」时（会话配置、approval 时），不是每次「调用」时；approval 的结果应该沉淀为可复用的能力而非一次性放行。
3. **隔离借 OS**：bwrap/seccomp/Seatbelt + 网络代理白名单；「权限管是否运行、沙箱管运行后」的分界直接采纳。
4. **审计与安全同源**：副作用日志即审计日志即 undo 日志——一份记录，三个消费方。

---

## 5 · 持久化与副作用事务：会话回滚 ≠ 世界回滚

**共同问题**：计算易失而世界持久；崩溃时如何让「日志」与「世界」一致。

### agent 现状与痛点

- **内部状态已收敛**：会话状态的崩溃恢复各家做法趋同（append-only + 提交点 + 恢复修复）。aaos 的方案（SQLite WAL + 内容寻址 append-only 资产 + MessageEnd 消息级提交 + 悬空 tool_call 配对修复，[ADR-0001][adr1]/[ADR-0002][adr2]）在粒度与出处上领先——压缩保留原文可寻址，等价于 GC 保留指向被回收对象的引用。
- **痛点 1 · 外部副作用不可回滚**：Claude Code 的检查点文档**明确声明**：不追踪 bash 命令造成的文件改动（rm/mv/cp 无法 rewind）、不追踪会话外改动、多数 subagent 编辑不恢复，长期回滚交给 git（[checkpointing 文档][cc-checkpointing]）。也就是说，主流 coding agent 的「世界回滚」只覆盖**自己那支笔**写过的文件；shell 一执行，事务性就断了。「会话回滚了但世界没回滚」造成状态分叉，是当前 agent 工具链最深的坑之一。
- **痛点 2 · 工具重试不幂等**：at-least-once 重试（网络抖动、超时重发）遇到非幂等工具（发消息、建 PR）会放大副作用。OS/分布式答案是幂等键——aaos 的内容寻址资产 ID 天然可以充当「（会话, 位置, 工具, 参数哈希）→ 结果」的去重键。
- **痛点 3 · 不可逆操作无二阶段语义**：Terraform 的 plan/apply（[两阶段][terraform]）、saga 的补偿事务（[Sagas, Garcia-Molina & Salem 1987][sagas]）在 agent 工具设计里尚未成为惯例；不可逆操作目前只靠「执行前问一下」这一种 UX。

### OS 答案 → 设计原则（副作用按可撤销性分级）

| 级别 | 工具类 | OS 机制 | agent 方案 |
|---|---|---|---|
| 可 undo | 文件写/编辑 | undo logging（before 镜像） | **aaos 副作用载荷已在记录 before/after**——补上「按会话回滚 = 逆序应用 before 载荷」即可 |
| 可快照 | 整个工作区 | COW fs 快照 / 层叠 fs | btrfs/ZFS 快照、overlayfs 每步一层、或 git auto-commit 事务（Claude Code 官方兜底建议） |
| 只能补偿 | 网络副作用 | saga 补偿事务 | 工具 schema 声明补偿动作（send → rescind）；无法补偿的（rm 已发布的产物）前置为「危险操作需显式能力 + 确认」 |

- **undo 的正确性边界**：盲目还原 before 载荷不安全——若 after 之后文件被外部修改，还原会踩掉别人的改动。解法是乐观并发：undo 前校验当前内容哈希是否仍等于 after 哈希（aaos 的 BLAKE3 寻址直接支持），不匹配则升级为冲突处理而非静默覆盖。
- **原子提交**：多步文件操作走「staging → 单点提交」（对应暂存区 / 原子 rename(2) 的事务式用法 [rename(2)][rename]），把「改了 3 个文件崩在第 2 个」从恢复问题降级为丢弃问题。

---

## 6 · 结论：借自 OS 的十条设计原则（按对 aaos 的价值排序）

1. **上下文 = 不可变前缀 + 追加尾部**，立为不变量：动态内容永不进前缀；技能索引注入保持字节级稳定。（守缓存命中——aaos append-only 架构天然满足，值得成文守护。）
2. **压缩 = 有出处、可回溯的 GC**：保持区间替换映射 + 原文寻址；把压缩时机当作调度问题（任务边界、缓存空闲期）而非内存告急的应急。
3. **切换昂贵 → 会话亲和调度**：任务粘会话、少 spawn、子 agent 当虚拟机级 fork 对待。
4. **子 agent 返回协议结构化**：exit code + artifact 引用（资产 ID = fd）+ 人读摘要；禁止裸文本承载唯一结果。
5. **补齐 SIGTERM 与看门狗**：优雅中止（完成当前工具、提交边界）与工具级超时；abort 保持为最后手段。
6. **显式生命周期状态机**：running / waiting / idle / reaped，调度与恢复据此决策。
7. **副作用按可撤销性分级并写进工具 schema**：undo 级（before 载荷 + 哈希校验回滚，记录已在）、快照级（fs/git 事务）、补偿级（saga）；一份副作用日志同时服务 undo、审计、追责。
8. **权限 = 能力句柄**：会话级授予（路径、只读、网络白名单），approval 沉淀为能力；沙箱借 OS 件（bwrap/Seatbelt + 代理白名单），不自造。
9. **配额三元组抽象**（计量/限额/优先级）覆盖 token、成本、并发、连接数——agent 的 cgroup。
10. **命名空间按会话挂载**：内部 URI 扩展为 scheme 家族（skill:// → file://、mcp://、session://），可见性、权限、缓存收敛到统一寻址层。

### 可立项方向（issue 候选，未开）

- **A · 副作用回滚 v0**：文件类工具的 undo（消费已有 before/after 载荷 + BLAKE3 校验）——aaos 已有数据，缺消费方，性价比最高。
- **B · 优雅中止 + 工具超时**：SIGTERM 语义 + watchdog（pi-agent-core 层）。
- **C · 缓存感知上下文策略**：稳定前缀守则 + 压缩时机调度（CLI/agent-loop 层）。
- **D · 资源配额抽象**：token/成本/并发限额（provider 调用层）。
- **E · 子 agent + 结构化返回协议**（最大的一块，依赖 A/B/C 前置程度低）。
- **F · MCP 接入**（总线层，adapter 模式复用 providers 经验）。

---

## 参考来源

**Agent 侧官方文档**（均为 2026-08 抓取核实）：

- [pc-anthropic]: https://platform.claude.com/docs/en/build-with-claude/prompt-caching
- [pc-openai]: https://developers.openai.com/api/docs/guides/prompt-caching
- [pc-gemini]: https://ai.google.dev/gemini-api/docs/caching
- [cc-compaction]: https://platform.claude.com/docs/en/build-with-claude/compaction
- [cc-context-eng]: https://platform.claude.com/cookbook/tool-use-context-engineering-context-engineering-tools
- [compact-analysis]: https://hyperdev.matsuoka.com/p/how-claude-code-got-better-by-protecting
- [cc-subagents]: https://code.claude.com/docs/en/sub-agents
- [oai-handoffs]: https://openai.github.io/openai-agents-python/handoffs/
- [mcp-intro]: https://modelcontextprotocol.io/docs/getting-started/intro
- [mcp-arch]: https://modelcontextprotocol.io/docs/2026-07-28/learn/architecture
- [a2a]: https://a2a-protocol.org/latest/
- [cc-sandbox]: https://code.claude.com/docs/en/sandboxing
- [cc-checkpointing]: https://code.claude.com/docs/en/checkpointing

**论文与教材**：

- MemGPT: Towards LLMs as Operating Systems — arXiv 2310.08560（agent 即 OS 的直接先例）
- Efficient Memory Management for LLM Serving with PagedAttention — arXiv 2309.06180（KV cache 的虚拟内存化）
- Lost in the Middle — arXiv 2307.03172（位置衰减）
- Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving — arXiv 2406.17585
- Sagas — Garcia-Molina & Salem, SIGMOD 1987（补偿事务）
- Crash-Only Software — Candea & Fox, HotOS 2003
- OSTEP（Operating Systems: Three Easy Pieces），Remzi & Andrea Arpaci-Dusseau——进程、调度、swap 章节
- The Use of Name Spaces in Plan 9（per-process namespace）
- rename(2)、signal(7)、cgroups(v2) Linux 手册
- CRIU（checkpoint/restore in userspace）: https://criu.org

**aaos 内部**：`CONTEXT.md`（词汇表）、`docs/adr/0001-sqlite-structural-source-of-truth.md`、`docs/adr/0002-session-absorbs-agent-integration.md`、`docs/adr/0004-skills-internal-uri.md`、`crates/pi-agent-core/src/agent.rs`（steering/abort API 面）。
