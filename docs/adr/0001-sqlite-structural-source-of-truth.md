# 0001 · SQLite 为结构事实源，会话结构统一为 append-only 派生

会话存储分两层：内容层是 BLAKE3 内容寻址、写一次的对象文件（已在 feat/session-store 实现，原样保留）；结构层决定放弃追加日志链 + HEAD 引用文件，改为 SQLite 单库事实源。决定性理由：查询面是近期刚需（aaos-cli 的会话列表/检索，及后续把会话中 agent 工作流程沉淀为可复用 skill 的 ResourceRef 方向——跨会话结构查询）；多进程访问不能承诺单进程（agent 运行中 CLI 只读查询），SQLite WAL 原生覆盖；崩溃一致性由事务保证，framing / torn-tail / 单遍日志扫描机制整体消失。

## 决定细节

- **统一派生**：一切结构变更 = 新会话行 `sessions(id, parent_id, parent_position, kind root|fork|compact)` + append-only 记录（entries 追加 / compactions 区间映射 / side_effects 副作用）。分叉（纯追加）与压缩（首批记录为 `[start, end)` 区间替换映射）是**同一通用操作的两种记录构成**，不是两套语义；链序即优先级，无 per-index 冲突规则。
- **id 即指针**：无 HEAD 文件、无 current 列；latest = `ORDER BY created_at DESC` 查询；resume 按会话 id 打开链、尾部续写。
- **append-only，无删行回退**：回退与撤销压缩 = 从 (会话, 位置) 派生新会话；snapshots 降级为纯书签（git tag），永不自动恢复。
- **出处双轨**：`SummarySegment.sources`（内容级，跨 fork 稳定）+ compactions 区间的 seq 范围查询（结构级取回原文）。

## Considered Options

- **追加日志链 + HEAD 引用（原 plan，已实现于 feat/session-store）**：领域语义被本 ADR 继承，宿主被替换——跨会话查询要目录遍历 + 全量重放，多进程写者下 HEAD 原子小文件不够用。实现作废为参考。
- **overlays 原地覆盖行（issue #4 初稿）**：per-index newest-wins 覆盖引入区间重叠/链式冲突的补丁式规则，且与 fork 并存形成两套结构语义。否决。
- **sqlx**：async 原生但重，连接池模型与 SQLite 单写者相性差（SQLITE_BUSY 锁饥饿）；rusqlite bundled 本机实测冷构建约 6s（debug）/ 34s（release），二进制净增约 2MB，可接受。

## Consequences

- workspace 引入首个数据库依赖：`rusqlite`（bundled）+ `tokio-rusqlite`（单后台线程持有连接，DB 调用统一走其克隆句柄，不手搓 Mutex + spawn_blocking）。
- feat/session-store 结构层六模块（framing / log / branch / refs / writer / view）重写为 DB 实现；资产层三模块（object_store / segment / canon）与既有测试场景平移复用。
- `docs/superpowers/plans/2026-08-22-session-storage.md` 标 superseded；issue #4 schema 按本 ADR 修订；词汇表见根 `CONTEXT.md`。
