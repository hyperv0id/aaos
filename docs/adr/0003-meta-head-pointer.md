# 0003 · meta 表头指针：默认恢复回到最后写入的线

ADR-0001 的「id 即指针」（无 HEAD 文件、无 current 列）在 CLI 默认恢复路径上站不住：`latest_session()`（`ORDER BY created_at DESC` 全局查询）被拿来近似"当前会话"，但正常对话只 append `entries`，不产生 `sessions` 行——该查询既不随对话推进，也不属于任何进程。默认 resume 目标被冻结在全局最新创建的会话上（issue #61：退出提示 id 三天不变），并发进程还会互相拿到同一节点交叉写入。

## 决定

- **头指针落库**：`meta(key, value)` 表存 `head` → 会话节点 id。这是结构层唯一的可变行，其余仍然 insert-only；库连接设 `busy_timeout`。
- **head 跟随追加**：`append_segment` 在同一 IMMEDIATE 事务里把 head 移到被追加的节点——head = 最后写入的线。`create_root` / `fork` / `compact` 不动 head；resume 也不动。
- **默认恢复 = 从 head 派生新线**：无 `--session` 时对 head 做 fork 并续写派生。派生继承完整视图（对话内容连续），但每个进程写自己的节点——n 个并发进程不交叉写入同一会话，退出提示返回本进程真实节点。旧库（无 head）回退 `latest_created_session()`，空库建新 root。
- **显式 `--session` 原地续写**：从指定节点继续，不被默认逻辑改写；`--fork` 仍表示从解析目标派生；指定节点不存在时直接报错，不静默改走默认路径。
- **退出提示只在本进程真正持久化过内容时打印**，打印的是本进程当前节点。

## 并发分析（n 实例同时运行）

- **隔离**：默认解析各 fork 各的节点——派生行互不冲突，对话各写各的线；head 是 last-writer-wins，翻转指针不移动任何人的写入。
- **写锁**：所有写语句短于毫秒级；`busy_timeout(5s)` 让并发写者排队而非立即 SQLITE_BUSY（rusqlite 默认超时为 0，撞锁即失败丢段）。
- **append 事务**：seq 读取 + entries 插入 + head 更新在一个 `BEGIN IMMEDIATE` 事务里。DEFERRED 的先读后写在 WAL 下被并发提交会得 BUSY_SNAPSHOT——busy handler 对它不重试，只能换 IMMEDIATE。附带收益：显式共享同一节点的两个进程 append 时，`MAX(seq)+1` 在写锁内读取已提交前驱，不再撞 `(session_id, seq)` 主键。
- **id 唯一性**：id = 毫秒时间戳 + **每进程 pid**（`std::process::id`）+ 进程内计数器。并发启动的进程 pid 互不相同；无进程项时 n 个实例同一毫秒启动会生成相同 id、撞 `sessions` 主键。
- **读路径**：WAL 下任意多进程并发读（resume / materialize）不阻塞。对象层 `.tmp-*` + rename 原子，并发同内容写安全。

## Considered Options

- **resume 原地续写 head**：语义最线性（`--session 旧节点` 能看到全部后续），但并发进程必然交叉写入同一节点，缺陷不解；放弃。
- **进程租约 / 声明协议**：解决同一问题的另一路，需要心跳与崩溃回收，复杂度与收益不成比例；fork 派生天然达到同样效果。
- **HEAD 独立小文件**（ADR-0001 已否）：多进程写者下原子性不足，且与库事务脱节；落库 `meta` 表后由 SQLite 事务覆盖。
- **update `sessions.created_at` 冒充推进**：破坏 append-only 且仍无进程归属；否决。

## Consequences

- 结构层新增唯一可变状态：`meta` 表。entries 与 head 同事务写入，指针不会指向未提交的内容。
- 每次 `aaos` 默认启动产生一个派生行（即使用户未发一言、静默退出）；空派生物化代价可忽略，静默退出也不再打印 resume 提示。
- 显式多进程共享同一 `--session` 节点：对话按轮次交错进同一条血统——用户的显式选择，修后只交错、不丢数据不报错。
