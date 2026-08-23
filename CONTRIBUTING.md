# Contributing to aaos

感谢参与 aaos 开发。本文档是贡献者需要知道的全部约定。

## 项目简介

aaos 是以会话为中心的 agent 系统，用操作系统设计（GC/swap、COW fork、信号、WAL、进程组）重新审视 LLM agent harness 中的上下文管理、多 agent 协作与失败恢复问题。

仓库结构：

```
crates/
├── pi-agent-core/      agent loop 核心
├── aaos-providers/     provider 域：模型注册表 + 方言 adapter
├── aaos-tools/         工具实现（bash 等）
├── aaos-session-store/ 会话存储（SQLite）
└── aaos-cli/           CLI 入口（aaos 命令）
```

## 开发环境

需要 Rust stable 工具链（`rustup default stable`）。项目无 nightly 依赖，`rustfmt.toml` 只用 stable 选项。

```bash
git clone git@github.com:hyperv0id/aaos.git
cd aaos
cargo build
```

## 常用命令

```bash
# 测试
cargo test --workspace

# 格式检查
cargo fmt --all -- --check

# Lint
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings

# 文档
cargo doc --no-deps --document-private-items --workspace

# 供应链检查
cargo install --locked cargo-deny
cargo deny check
```

CI 会在每个 PR 上跑全部上述检查。本地提交前建议先过一遍。

## 代码风格

### rustfmt

`rustfmt.toml` 配置 `reorder_imports = true`（stable）。直接用 `cargo fmt --all` 格式化，无需额外参数。

### Clippy

workspace 级 lint 配置在根 `Cargo.toml` 的 `[workspace.lints.clippy]`：

- `clippy::all` = warn（priority -1）
- `clippy::pedantic` = warn（priority -1）

所有 crate 通过 `[lints] workspace = true` 继承。CI 用 `-D warnings` 将 warn 升级为 deny。

pedantic 组会产出大量告警，逐个用 `#[allow(clippy::xxx)]` + 理由注释处理。不要全局关闭 pedantic。

不要整组开启 `restriction` 或 `nursery`。如需某个 restriction lint（如 `unwrap_used`、`panic`、`indexing_slicing`），在 workspace lints 中单独声明：

```toml
[workspace.lints.clippy]
all = { level = "warn", priority = -1 }
pedantic = { level = "warn", priority = -1 }
unwrap_used = "warn"
```

## 测试要求

- 新增功能必须有对应测试。
- 测试覆盖行为和边界，而非实现细节。
- 集成测试放在 crate 的 `tests/` 目录，单元测试与代码同文件 `#[cfg(test)] mod tests`。
- 领域概念命名遵循 [`CONTEXT.md`](./CONTEXT.md) 词汇表。

## 提交规范

采用 [Conventional Commits 1.0.0](https://www.conventionalcommits.org/)。

格式：

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

### Type

| Type | 用途 |
|------|------|
| `feat` | 新功能 |
| `fix` | bug 修复 |
| `chore` | 构建、依赖、杂项 |
| `docs` | 文档变更 |
| `refactor` | 重构（不改变行为） |
| `test` | 测试相关 |
| `perf` | 性能优化 |
| `ci` | CI 配置 |
| `style` | 代码风格（格式化、命名） |
| `revert` | 回滚某次提交 |

### Scope

用 crate 名或领域名词：

`core`、`cli`、`providers`、`tools`、`session-store`、`ci`、`docs`、`workspace`、`lints`

### 示例

```
feat(session-store): SQLite structural source of truth (ADR-0001)
fix(pi-agent-core): continue validation, addedToolNames, Off→None
chore(deps): bump tokio from 1.45 to 1.46
docs(research): CI/CD 调研报告
```

### Breaking Change

在 type 后加 `!` 或在 footer 写 `BREAKING CHANGE: <说明>`：

```
feat(core)!: rename AgentRun to AgentHandle
```

## 分支与 PR

### 分支命名

| 前缀 | 用途 |
|------|------|
| `feat/` | 新功能 |
| `fix/` | bug 修复 |
| `issue/` | issue 驱动开发 |
| `docs/` | 纯文档变更 |

禁止直接 push 到 `master`。所有变更通过 PR。

### PR 流程

1. 从 `master` 切分支。
2. 开发并测试通过。
3. 提 PR，PR 标题用 Conventional Commits 格式。
4. 确保 PR checklist 全部勾选。
5. Squash merge 到 `master`——squash commit 标题即 PR 标题（Conventional Commits 格式）。

## 领域文档

- [`CONTEXT.md`](./CONTEXT.md) — 领域词汇表，用词以此为准。
- [`docs/adr/`](./docs/adr/) — 架构决策记录，修改涉及已有 ADR 的部分时先读对应 ADR。

## MSRV

当前未承诺最低支持 Rust 版本。首次 crates.io 发布前会确定并在 `Cargo.toml` 各 crate 中声明 `rust-version`。

## 供应链安全

[`deny.toml`](./deny.toml) 配置 cargo-deny 检查：

- `advisories`：RustSec 漏洞数据库
- `bans`：禁止重复版本和 wildcard 依赖
- `licenses`：许可证白名单（MIT/Apache-2.0/BSD/ISC 等）
- `sources`：仅允许 crates.io

[Dependabot](https://github.com/hyperv0id/aaos/blob/master/.github/dependabot.yml) 每周检查 cargo 依赖和 GitHub Actions 版本更新，自动提交 PR。
