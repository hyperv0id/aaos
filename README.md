# aaos

以会话为中心的 LLM agent 系统。用操作系统设计（GC/swap、COW fork、信号、WAL、进程组）重新审视 vibe-coding 工具链中上下文管理、多 agent 协作与失败恢复的问题，并收敛 agent 资源接口与会话存储的设计。

## Crates

| Crate | 说明 |
|---|---|
| [`pi-agent-core`](crates/pi-agent-core) | Agent loop 核心：事件驱动、工具调用、hook 生命周期 |
| [`aaos-providers`](crates/aaos-providers) | provider 域：模型注册表（models.dev）+ 方言 adapter 与 dispatch |
| [`aaos-tools`](crates/aaos-tools) | 工具实现（bash 等） |
| [`aaos-session`](crates/aaos-session) | 会话存储：SQLite 结构事实源 + BLAKE3 内容寻址 |
| [`aaos-cli`](crates/aaos-cli) | CLI 入口（`aaos` 命令） |

## 快速开始

```bash
git clone git@github.com:hyperv0id/aaos.git
cd aaos
cargo build
cargo test --workspace
```

运行 CLI：

```bash
cargo run -p aaos-cli -- --help
```

## 文档

| 文档 | 内容 |
|---|---|
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南：代码风格、测试、提交规范、分支策略 |
| [CONTEXT.md](CONTEXT.md) | 领域词汇表（会话、资产、派生、分叉、压缩、书签、视图、副作用） |
| [docs/adr/](docs/adr/) | 架构决策记录 |
| [docs/research/](docs/research/) | 技术调研报告（CI/CD、代码风格、GitHub 自动化） |

## 开发

CI 在每个 PR 上运行 `cargo fmt --check`、`cargo clippy -D warnings`、`cargo test`、`cargo doc` 和 `cargo-deny`。提交前本地确认：

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo test --workspace
```

详见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 许可证

[MIT](LICENSE)
