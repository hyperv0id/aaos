# aaos 代码风格与贡献规范调研报告

> 调研日期：2026-08-22 · 范围：rustfmt / clippy / CONTRIBUTING / MSRV / 供应链 / 提交规范
> 方法：一手文档（rustfmt Configurations.md、clippy 官方 lints、cargo manifest/rust-version/CI 官方文档、edition guide、rustup overrides、cargo-deny、RustSec、Conventional Commits）+ 知名仓库实际配置（rust-lang/rust、tokio-rs/tokio、serde-rs/serde、bevyengine/bevy、BurntSushi/ripgrep）

## 1. rustfmt.toml 配置

**结论：aaos 当前 rustfmt.toml 含两个 nightly-only 选项（`imports_granularity`、`group_imports`），stable 工具链不会生效。两条路：删掉 nightly-only 选项回归 stable，或为 fmt 检查单独配 pinned nightly。**

rustfmt 官方 [Configurations.md](https://github.com/rust-lang/rustfmt/blob/master/Configurations.md) 逐项标注稳定性：

| 选项 | Stable | 来源 |
|------|--------|------|
| `reorder_imports` | Yes | Configurations.md |
| `style_edition` | Yes | Configurations.md |
| `use_small_heuristics` | Yes | Configurations.md |
| `merge_derives` | Yes | Configurations.md |
| `use_field_init_shorthand` | Yes | Configurations.md |
| `max_width` | Yes | Configurations.md |
| `imports_granularity` | **No** | [#4991](https://github.com/rust-lang/rustfmt/issues/4991) 未关闭 |
| `group_imports` | **No** | [#5083](https://github.com/rust-lang/rustfmt/issues/5083) 未关闭 |
| `unstable_features` | **No** | [#3387](https://github.com/rust-lang/rustfmt/issues/3387) |

aaos 当前 rustfmt.toml：`reorder_imports = true`（stable，生效）+ `imports_granularity = "Crate"` + `group_imports = "StdExternalCrate"`（均 nightly-only，stable 上被忽略并警告）。CI 用 stable，故两项不生效——本地 nightly 格式化结果与 CI stable 检查不一致。

**参考仓库做法**：
- [rust-lang/rust rustfmt.toml](https://github.com/rust-lang/rust/blob/master/rustfmt.toml)：`style_edition = "2024"`、`use_small_heuristics = "Max"`、`merge_derives = false`、`group_imports = "StdExternalCrate"`、`imports_granularity = "Module"`——全在 nightly 语境下用（rust 仓库 CI 用 nightly fmt）。
- [tokio-rs/tokio](https://github.com/tokio-rs/tokio)：根目录**无 rustfmt.toml、无 rust-toolchain.toml**，CI fmt job 用 stable rustfmt `--check --edition 2021`。tokio 选择只用 stable 支持的选项。
- [BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep)：有 rustfmt.toml，CI fmt job 用 nightly。

**建议**：两条路均可——
1. **最简**：删掉 `imports_granularity` 和 `group_imports`，只留 stable 选项，CI fmt 用 stable（tokio 模式）。
2. **保留 import 规范**：为 fmt 检查单独配 pinned nightly job（ripgrep/rust 模式，见 ci-cd 报告 Q1）。

## 2. clippy 配置

**结论：workspace 级 `all` + `pedantic` 作为基线合理；不要整组开 `restriction` 或 `nursery`，按需 cherry-pick 单个 restriction lint。**

Clippy 官方 [lints 页面](https://doc.rust-lang.org/clippy/lints.html) 明确各组语义：
- `correctness`：deny-by-default，指出错误代码。
- `pedantic`：power-user 组，开启后会大量告警，需大量局部 `#[allow]`。
- `restriction`：**不建议整组开**（官方文档明确说整组属性会自身产生 warning）；应 cherry-pick 单个 lint。
- `nursery`：仍处开发中，可能 buggy，不建议整组开。
- `cargo`：只检查 Cargo.toml 元数据（如 missing license/description）。

**restriction 组值得 cherry-pick 的 lint**（源码确认均为 restriction 组）：
- `unwrap_used` / `expect_used`：防止生产代码 panic。
- `panic`：禁止直接 panic!。
- `indexing_slicing`：禁止直接索引（用 `get` 替代）。
- `as_conversions`：限制 `as` 类型转换。
- `dbg_macro` / `print_stdout`（CLI crate 适合）：禁止 debug 打印残留。

全部用局部 `#[allow(...)] + 理由注释`，不要全局放开。

**Cargo [lints] 官方文档**：[manifest#the-lints-section](https://doc.rust-lang.org/cargo/reference/manifest.html#the-lints-section)。`level` = `forbid`/`deny`/`warn`/`allow`，`priority` 负值使组级配置（priority -1）被具体 lint（默认 0）覆盖。aaos 当前 `[workspace.lints.clippy]` 的 `all` + `pedantic` 均为 `warn`、`priority = -1`，是标准写法。

**CI 策略**：aaos CI 已用 `-- -D warnings`；cargo book 建议 [CI 门禁用 warnings-clean、本地宽松](https://doc.rust-lang.org/cargo/guide/continuous-integration.html#checking-for-warnings)。tokio CI 用 `RUSTFLAGS=-Dwarnings` + 固定 clippy 版本。建议保留 `-D warnings`（或升级为 `CARGO_BUILD_WARNINGS: deny`，见 ci-cd 报告）。

## 3. CONTRIBUTING.md 内容

**结论：参考 tokio 分章方式，aaos CONTRIBUTING 应覆盖以下章节。**

一手样本结构：
- [Rust CONTRIBUTING.md](https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.md)：帮助渠道、dev guide 指引、bug report 流程、LLM policy。
- [Tokio CONTRIBUTING.md](https://github.com/tokio-rs/tokio/blob/master/CONTRIBUTING.md) + [docs/contributing/README.md](https://github.com/tokio-rs/tokio/blob/master/docs/contributing/README.md) + [pull-requests.md](https://github.com/tokio-rs/tokio/blob/master/docs/contributing/pull-requests.md)：issue/PR/review 流程、环境命令、测试/doc/bench 要求、commit 规范、squash 策略。
- [Serde CONTRIBUTING.md](https://github.com/serde-rs/serde/blob/master/CONTRIBUTING.md)：MCVE 要求、测试要求、CoC。

**aaos CONTRIBUTING 应含**：
1. 项目简介与范围
2. 开发环境搭建（工具链安装、`cargo build`）
3. 常用命令（`cargo test`、`cargo fmt`、`cargo clippy`、`cargo doc`）
4. 代码风格（rustfmt + clippy 规则，指向 workspace lints 配置）
5. 测试要求
6. 提交规范（Conventional Commits，见 Q7）
7. 分支与 PR 策略（见 Q7）
8. MSRV 政策（见 Q4）
9. 供应链安全（见 Q5）
10. 领域文档指引（指向 `CONTEXT.md` + `docs/adr/`）

## 4. MSRV（Minimum Supported Rust Version）

**结论：aaos 未发布，可暂不承诺 MSRV；若承诺，用 `rust-version` 字段 + CI cargo-hack 验证。**

- Cargo [`rust-version`](https://doc.rust-lang.org/cargo/reference/rust-version.html) 字段自 Cargo/Rust 1.56 起尊重；不兼容工具链报错，`--ignore-rust-version` 可绕过。
- Rust 2021 edition 发布于 [1.56.0](https://doc.rust-lang.org/edition-guide/rust-2021/index.html)。
- tokio 政策：至少支持 6 个月前的编译器，CI `rust_min = 1.71` + [cargo-hack](https://github.com/taiki-e/cargo-hack) 验证。
- **建议**：aaos 首次发布前定 MSRV（建议 5 个 stable 版本前的版本），每 crate `Cargo.toml` 加 `rust-version`，CI 加 `cargo hack check --rust-version --workspace --all-targets --ignore-private`。不要把 edition 下限直接当 MSRV 政策。

## 5. cargo-deny / cargo-audit

**结论：推荐 cargo-deny（覆盖面更广），至少配 advisories 检查。**

- [cargo-deny](https://embarkstudios.github.io/cargo-deny/)：`cargo deny init` / `cargo deny check`；检查 `licenses`（许可证白名单）、`bans`（重复依赖/wildcard 依赖）、`advisories`（RustSec 漏洞/unmaintained/yanked）、`sources`（未知 registry/git 源）。
- [cargo-audit](https://github.com/RustSec/cargo-audit)：仅审计 `Cargo.lock` against RustSec advisory DB。
- [Bevy 实际 deny.toml](https://github.com/bevyengine/bevy/blob/main/deny.toml)：advisories ignore+reason、license allowlist、bans multiple-versions/wildcards、sources crates.io-only。
- **建议**：aaos 加 cargo-deny，起步只配 `advisories`，成熟后开 `bans`/`licenses`/`sources`。定时 CI job 更新 advisory DB（见 ci-cd 报告 Q4）。

## 6. commit message 规范

**结论：采用 Conventional Commits 1.0.0；aaos 现有提交历史已隐含该格式，正式写入 CONTRIBUTING。**

[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/) 规范：
- 格式：`<type>[scope]: description` + optional body/footer。
- `fix` = PATCH，`feat` = MINOR，`BREAKING CHANGE` footer 或 `type/scope!` = MAJOR。
- scope 为括号内的名词，标识影响范围。
- 推荐 type：`feat`/`fix`/`chore`/`docs`/`refactor`/`test`/`perf`/`ci`/`build`/`style`。
- footer token（如 `Fixes: #123`）与 `BREAKING CHANGE:` 规则。

tokio 额外要求：首行 imperative mood、lowercase、模块前缀、≤72 字符、空第二行、正文 wrap 72、`Fixes:`/`Refs:` trailers。

**aaos scopes**（基于现有 crate 结构）：
- `core`（pi-agent-core）、`cli`（aaos-cli）、`openai`（aaos-openai）、`tools`（aaos-tools）、`catalog`（aaos-catalog）、`session`（aaos-session）、`session-store`（aaos-session-store）
- 跨 crate：`ci`、`docs`、`workspace`、`lints`

**aaos 现有提交**已隐含此格式：`feat(lints):`、`fix(pi-agent-core):`、`chore(pi-agent-core):`、`test(pi-agent-core):`、`docs(...)`。正式写入规范即可。

## 7. 分支策略

**结论：沿用现有 feat/fix/issue 前缀 + PR squash merge。**

- [tokio PR workflow](https://github.com/tokio-rs/tokio/blob/master/docs/contributing/pull-requests.md)：禁止直接 push master，feature branch 提 PR；评审中不要 rebase（避免丢失 review 上下文）；落地主干可按逻辑变更 squash。
- aaos 现有分支：`master`（主干）、`feat/*`（功能）、`issue/*`（issue 驱动）。建议保留此约定。
- PR title 用 Conventional Commits 格式（`feat(scope): description`），squash merge 保持主干简洁——每个 squash commit 即一个完整 PR 的逻辑变更。
- CI 当前 `push: branches: [main, master]` + `pull_request:` 触发，与该策略一致。

## 针对 aaos 的具体建议

| # | 动作 | 依据 |
|---|------|------|
| 1 | 决定 rustfmt 策略：删 nightly-only 或配 nightly fmt job | Configurations.md #4991/#5083 |
| 2 | workspace lints 保持 `all` + `pedantic` 基线；按需 cherry-pick restriction lint | clippy lints 官方页 |
| 3 | 写 CONTRIBUTING.md（10 章节，见 Q3） | tokio/rust/serde 样本 |
| 4 | 暂不承诺 MSRV；首次发布前加 `rust-version` + cargo-hack | cargo rust-version 文档 |
| 5 | 加 cargo-deny（起步 advisories） | cargo-deny 文档 |
| 6 | 正式采用 Conventional Commits，写入 CONTRIBUTING | conventionalcommits.org |
| 7 | 沿用 feat/fix/issue 分支 + squash merge | tokio PR workflow |

## 参考资料

- rustfmt Configurations.md：https://github.com/rust-lang/rustfmt/blob/master/Configurations.md
- rust-lang/rust rustfmt.toml：https://github.com/rust-lang/rust/blob/master/rustfmt.toml
- tokio-rs/tokio（无 rustfmt.toml）：https://github.com/tokio-rs/tokio
- Clippy 官方 lints：https://doc.rust-lang.org/clippy/lints.html
- Clippy book — lints.md：https://github.com/rust-lang/rust-clippy/blob/master/book/src/lints.md
- Cargo [lints] section：https://doc.rust-lang.org/cargo/reference/manifest.html#the-lints-section
- Cargo CI 官方文档：https://doc.rust-lang.org/cargo/guide/continuous-integration.html
- Rust CONTRIBUTING.md：https://github.com/rust-lang/rust/blob/master/CONTRIBUTING.md
- Tokio CONTRIBUTING.md：https://github.com/tokio-rs/tokio/blob/master/CONTRIBUTING.md
- Tokio docs/contributing/：https://github.com/tokio-rs/tokio/blob/master/docs/contributing/README.md
- Serde CONTRIBUTING.md：https://github.com/serde-rs/serde/blob/master/CONTRIBUTING.md
- Cargo rust-version：https://doc.rust-lang.org/cargo/reference/rust-version.html
- Rust 2021 Edition Guide：https://doc.rust-lang.org/edition-guide/rust-2021/index.html
- Rustup overrides：https://rust-lang.github.io/rustup/overrides.html
- cargo-deny：https://embarkstudios.github.io/cargo-deny/
- cargo-audit README：https://github.com/RustSec/cargo-audit
- Bevy deny.toml：https://github.com/bevyengine/bevy/blob/main/deny.toml
- Conventional Commits 1.0.0：https://www.conventionalcommits.org/en/v1.0.0/
- Tokio pull-requests.md：https://github.com/tokio-rs/tokio/blob/master/docs/contributing/pull-requests.md
