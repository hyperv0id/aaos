# aaos — agent 指引

以会话为中心的 LLM agent 系统：SQLite 结构事实源 + BLAKE3 内容寻址的会话存储，事件驱动
agent loop，4 种 provider API格式 adapter。用 OS 设计（GC/swap、COW fork、信号、WAL）
重新审视 agent harness 的上下文管理、多 agent 协作与失败恢复。

本文是目录页：知识在 docs/ 里按需读，这里只放不变量与验证命令。

## Crates

| crate | 职责 |
|---|---|
| `pi-agent-core` | agent loop 核心：事件驱动、工具调用、hook 生命周期、turn 重试 |
| `aaos-providers` | 模型目录（models.dev）+ API格式 adapter（SSE）+ provider HTTP 重试 |
| `aaos-tools` | 工具实现：bash / edit / write / read（含 `skill://`）/ skills / prompt |
| `aaos-session` | 会话存储：SQLite 结构层 + BLAKE3 对象层，insert-only，结构变更只经派生 |
| `aaos-cli` | `aaos` 命令入口：目录装配、REPL、compaction 编排 |

依赖单向：`pi-agent-core` ← tools / providers / session ← cli。

## 验证命令（不变量；提交前必过，CI 同款）

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace --locked --all-targets --all-features
cargo test --workspace --locked --doc --all-features
cargo deny check
```

文档构建：`RUSTDOCFLAGS=-D warnings cargo doc --no-deps --document-private-items --all-features --workspace`。

## 不变量

- 用词遵循 [`CONTEXT.md`](CONTEXT.md) 词汇表；各词条 _Avoid_ 列表是硬约束。
- 生产代码禁止 `panic!`/`unreachable!`/`todo!`/`unimplemented!`/`unwrap`/`expect`；
  例外必须用带理由的 `#[expect]` 在原位登记。
- 会话存储 insert-only：任何结构变更只经派生（fork / compaction），原文永不改写。
- 提交信息用 Conventional Commits（type/scope 见 [`CONTRIBUTING.md`](CONTRIBUTING.md)）；
  squash-merge 下 PR 标题即最终提交标题，PR 标题有 CI 门禁。

## 文档地图

| 何时读 | 位置 |
|---|---|
| 领域词汇（资产、会话、派生、压缩、书签、视图…） | [`CONTEXT.md`](CONTEXT.md) |
| 架构决策记录 | [`docs/adr/`](docs/adr/) |
| 贡献流程：风格、测试、提交规范、分支与 PR | [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| 仓库概览、快速开始、crate 表 | [`README.md`](README.md) |
| agent 侧 GitHub 操作（issue、triage 标签、领域文档维护） | [`docs/agents/`](docs/agents/) |
| 架构图规格（archify JSON，改后 pages.yml 自动重渲染） | [`docs/arch/`](docs/arch/) |
| 历史实现计划（多数已收官） | [`docs/superpowers/plans/`](docs/superpowers/plans/) |
| 技术调研（快照性质，结论可能过时，各文有日期横幅） | [`docs/research/`](docs/research/) |
| harness 化审计与改造计划 | [`docs/plans/`](docs/plans/) |

## Agent skills

- **Issue tracker**：issue 在 `hyperv0id/aaos` 的 GitHub Issues（经 `gh`）。见
  [`docs/agents/issue-tracker.md`](docs/agents/issue-tracker.md)。
- **Triage labels**：canonical 角色 1:1 映射 `needs-triage`、`needs-info`、
  `ready-for-agent`、`ready-for-human`、`wontfix`。见
  [`docs/agents/triage-labels.md`](docs/agents/triage-labels.md)。
- **Domain docs**：单上下文（根 `CONTEXT.md` + `docs/adr/`）。
  见 [`docs/agents/domain.md`](docs/agents/domain.md)。
- 技能锁文件 `skills-lock.json` 由安装器 `.agents/skills/` 维护；`.claude/skills/`
  是其镜像。
