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
├── aaos-session/ 会话存储（SQLite）
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

# Lint gate
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

**目标**：门禁只拦截"共识问题"——真实 bug（Clippy 默认的 `all` 组：correctness / style / complexity / perf）和团队显式强制的规则。有主观争议的风格问题不纳入门禁：大量无关告警会淹没真正的问题。

**现状**：全部显式配置集中在根 `Cargo.toml` 的 `[workspace.lints.clippy]`，所有 crate 通过 `[lints] workspace = true` 继承；CI 以 `-D warnings` 运行 Clippy，把默认组告警和下面这几条显式规则统一升级为错误——任何新增告警都会阻断 PR：

```toml
[workspace.lints.clippy]
dbg_macro = "deny"        # 禁止 debug!() 调试残留
print_stderr = "warn"     # 禁止 eprintln!() 打印
unwrap_used = "warn"      # 生产代码禁止 unwrap
expect_used = "warn"      # 生产代码禁止 expect
```

**为什么不开 pedantic**：pedantic 是 Clippy 官方定义的 power-user 组——lint 本身不是 bug，但较严格、偶有误报，且组内条目随版本增减。整组开启只有两种结局：被大量（多为主观风格）告警淹没，或为压掉告警堆满局部 `#[allow]` 从而降低可读性。因此不启用；将来确需某一条（如 `cast_lossless`）时，单独加进上面的表即可，同受 `-D warnings` 门禁。

**更严规则怎么加**：`restriction` 与 `nursery` 组同样不整组开——前者官方明确说整组开启自身会产生警告，后者仍未稳定。需要强制某条规则（如禁止直接 `panic!`）时，把它单独加进表里：

```toml
[workspace.lints.clippy]
panic = "deny"             # 禁止直接 panic!
indexing_slicing = "warn"  # 禁止直接索引，改用 get
```

**个别例外**：确需豁免的位置，用带理由的局部 `#[expect(clippy::xxx)]`（首推：lint 不再触发时 expect 自身会告警，避免例外腐烂）或 `#[allow(clippy::xxx)]`；不要全局放开。

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
