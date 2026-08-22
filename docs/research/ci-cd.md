# aaos CI/CD 调研报告

> 调研日期：2026-08-22 · 范围：Rust workspace CI/CD 最佳实践 · 目的：为 aaos 升级 `.github/workflows/ci.yml` 提供事实依据
> 方法：一手文档（cargo / clippy / rustup / rustfmt 官方文档）+ 知名仓库 CI 配置（rust-lang/rust、tokio-rs/tokio、BurntSushi/ripgrep）+ 工具文档（cargo-audit、cargo-deny、cargo-tarpaulin、cargo-llvm-cov、cargo-release、release-plz、Swatinem/rust-cache、sccache）

## 0. 现状盘点（已核实）

- 现有 [`ci.yml`](../../.github/workflows/ci.yml) 两个 job：`test`（`cargo test --workspace --locked --all-targets`）、`clippy`（`cargo clippy --workspace --all-targets --locked -- -D warnings`），均 `ubuntu-latest` + `dtolnay/rust-toolchain@stable` + `Swatinem/rust-cache@v2`，已有 `permissions: contents: read`（最小权限，与 ripgrep CI 一致）。
- [`rustfmt.toml`](../../rustfmt.toml)：`reorder_imports = true`（**stable**）、`imports_granularity = "Crate"`（**unstable**）、`group_imports = "StdExternalCrate"`（**unstable**）。
- workspace：7 个 crate，均 `edition 2021`、`version 0.1.0`，**无 `[features]`、无 `rust-version`、无 license/description 等发布元数据**；`Cargo.lock` 已提交；`rusqlite = { features = ["bundled"] }` 在 `aaos-session-store`（需编译 C 代码）；无 `rust-toolchain.toml`。

## 1. fmt 检查

**结论：必须加 `fmt` job；且由于 rustfmt.toml 含 unstable 选项，必须用 nightly toolchain（建议 pinned 日期）。**

- CI 命令形式：`cargo fmt --all -- --check`（`--` 之后是传给 rustfmt 的选项；ripgrep CI 用等价形式 `cargo fmt --all --check`）。证据：[ripgrep/.github/workflows/ci.yml](https://github.com/BurntSushi/ripgrep/blob/master/.github/workflows/ci.yml) 的 `rustfmt` job。
- **关键事实**：rustfmt 官方 [Configurations.md](https://github.com/rust-lang/rustfmt/blob/main/Configurations.md) 明确：*"Each configuration option is either stable or unstable. Stable options can always be used, while unstable options are only available on a nightly toolchain and must be opted into."* 且 `imports_granularity` 标注 **Stable: No**（跟踪 [rustfmt#4991](https://github.com/rust-lang/rustfmt/issues/4991)）、`group_imports` 标注 **Stable: No**（跟踪 [rustfmt#5083](https://github.com/rust-lang/rustfmt/issues/5083)），两者**至今未稳定**。也就是说 stable 工具链上的 `cargo fmt --check` 无法真正执行 aaos 的 rustfmt.toml 约定（unstable 选项在 stable 上不可用），CI 会得到与本地 nightly 格式化不一致的结果。
- **单独 job 还是并入 clippy**：两个知名仓库都设独立 job（tokio 的 `fmt` job、ripgrep 的 `rustfmt` job）；fmt 秒级完成，独立 job 失败定位清晰、天然并行；并入 clippy 可省一次 runner 启动，但业界惯例是独立。**建议独立 job**。fmt 不构建代码，不需要 rust-cache（ripgrep 的 fmt job 没有）。
- nightly 选择：rust-cache 文档明确 nightly 缓存每天作废（*"Using it with Nightly Rust is less effective as it will throw away the cache every day, unless a specific nightly build is being pinned"*），tokio 也 pin 日期（`rust_nightly: nightly-2025-10-12`）。**建议 pin `nightly-YYYY-MM-DD`**，避免 nightly 每日变动造成 CI 随机红。

```yaml
  fmt:
    name: fmt
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: false
      - name: Install Rust toolchain (nightly, pinned)
        uses: dtolnay/rust-toolchain@nightly
        with:
          toolchain: nightly-2026-08-01  # 与本地开发用的 nightly 一致，定期手动升级
          components: rustfmt
      - name: Check formatting
        run: cargo fmt --all -- --check
```

替代方案：删掉 rustfmt.toml 里两个 unstable 选项、回归 stable rustfmt——不推荐，刚定的 import 规范会丢。

## 2. clippy `--all-features`

**结论：加 `--all-features`，同时考虑 `--keep-going` 与 `CARGO_BUILD_WARNINGS=deny`。**

- 官方 clippy book 的 GitHub Actions 示例就是 `cargo clippy --all-targets --all-features`（配 `RUSTFLAGS: "-Dwarnings"`）：[rust-clippy/book/src/continuous_integration/github_actions.md](https://github.com/rust-lang/rust-clippy/blob/master/book/src/continuous_integration/github_actions.md)；cargo book 的 warnings 示例是 `CARGO_BUILD_WARNINGS=deny` + `cargo clippy --all-targets --all-features --keep-going`：[cargo book CI](https://doc.rust-lang.org/cargo/guide/continuous-integration.html#checking-for-warnings)。
- 对覆盖的意义：`--all-features` 保证 feature 组合下的代码（含 `#[cfg(feature = ...)]` 分支）也被检查；`--all-targets` 补上 tests/benches/examples。**aaos 现状：7 个 crate 都没有 `[features]`，`--all-features` 现阶段是 no-op（零成本）**，但满足 issue #5 验收，且以后加 feature 自动被覆盖。
- 潜在问题：`--all-features` 会开启平台相关 feature，可能在某些 target 失败——tokio 在跨平台 job 里特意不用它，注释即证据：*"We use `--features $TOKIO_STABLE_FEATURES` instead of `--all-features` since `--all-features` includes `io_uring` and `taskdump`, which is not available on all targets"*（[tokio/.github/workflows/ci.yml](https://github.com/tokio-rs/tokio/blob/master/.github/workflows/ci.yml)）。aaos 只在 `ubuntu-latest` 单平台跑，无此风险；将来加多平台矩阵时按 feature 分组即可。feature 冲突由 manifest 设计负责，`--all-features` 只会暴露问题而非制造问题。
- 顺带升级：clippy README 的 CI 说明指出，Cargo 1.97 之前惯例是 `-D warnings`，之后推荐 `CARGO_BUILD_WARNINGS=deny`，*"has the benefit of not invalidating build caches, and is thus to be preferred going forward"*（[rust-clippy README](https://github.com/rust-lang/rust-clippy)）。当前 stable 已 ≥1.97，可将 `-- -D warnings` 换成 env `CARGO_BUILD_WARNINGS: deny`（同时覆盖 rustc 警告）；clippy book 的 CI 章节也建议用与编译相同的 toolchain（stable），不要用 nightly clippy 做门禁。

```yaml
  clippy:
    name: clippy
    runs-on: ubuntu-latest
    env:
      CARGO_BUILD_WARNINGS: deny
    steps:
      - uses: actions/checkout@v4
        with:
          persist-credentials: false
      - name: Install Rust toolchain
        uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - name: Cache cargo
        uses: Swatinem/rust-cache@v2
      - name: Clippy
        run: cargo clippy --workspace --all-targets --all-features --locked --keep-going
```

## 3. CI job 矩阵（must-have vs nice-to-have）

基于三个仓库实际配置与官方文档：

**Must-have（aaos 现在就应具备）**
- `test`：现有 job 达标（`--workspace --locked --all-targets`）。cargo book 入门示例即 build + test：[cargo book CI](https://doc.rust-lang.org/cargo/guide/continuous-integration.html)。
- `clippy`：现有，补 `--all-features`（见 Q2）。
- `fmt`：缺失，必须加（见 Q1）。
- `docs`：建议加。ripgrep 与 tokio 都有独立 docs job，模式为 `RUSTDOCFLAGS: -D warnings` + `cargo doc --no-deps --document-private-items --workspace`（[ripgrep ci.yml](https://github.com/BurntSushi/ripgrep/blob/master/.github/workflows/ci.yml) 的 `docs` job）。对库 crate 尤其值得，能抓住 doc comment 里的坏链接/代码块错误。

**Nice-to-have（按需/将来）**
- MSRV check：cargo book 给出现成方案 `cargo hack check --rust-version --workspace --all-targets --ignore-private`（[cargo book CI # Verifying rust-version](https://doc.rust-lang.org/cargo/guide/continuous-integration.html#verifying-rust-version)）。**aaos 目前没有任何 crate 声明 `rust-version`，暂不适用**；首次发布前定好 MSRV 再加。
- beta/多通道矩阵：cargo book 入门示例跑 stable/beta/nightly 三通道并要求每通道独立 job；ripgrep 矩阵是 pinned + stable + beta + nightly。aaos 单 stable 合理，可选加 beta（latest-deps 的 `continue-on-error` 变体见 cargo book）。
- miri：tokio 用 pinned nightly + `MIRIFLAGS` 跑 miri 三件套（lib/test/doc），rust-lang/rust 也有 miri job。对 unsafe 密集代码收益大，aaos 无明显 unsafe 压力，列为将来选项。
- security audit：见 Q4，成本低，建议直接上。
- coverage：见 Q5，nice-to-have。
- semver 检查（tokio 用 cargo-semver-checks）、minimal-versions、latest-deps：tokio/cargo book 提供的高级项，0.1.0 阶段价值低，不加。
- 通用现代化项（tokio/rust 都有）：`concurrency: { group: ..., cancel-in-progress: true }` 取消旧运行；`--keep-going` 看全错误。可选。

## 4. security audit：cargo-audit vs cargo-deny

**结论：推荐 cargo-deny（单工具覆盖 audit + license + bans），配 schedule；想最轻量则 cargo-audit 也够。**

**cargo-audit**（RustSec 官方）：只查已知漏洞（RustSec Advisory Database）。RustSec 自己的仓库就这么跑——`on: pull_request paths: Cargo.lock` + `push main paths: Cargo.lock` + `schedule: "0 0 * * *"`，用 `actions-rs/audit-check`：[rustsec/.github/workflows/security_audit.yml](https://github.com/rustsec/rustsec/blob/main/.github/workflows/security_audit.yml)；cargo-audit README 明确 GitHub Actions 用 [audit-check action](https://github.com/rustsec/audit-check)。忽略项用 `--ignore RUSTSEC-xxxx` 或 `audit.toml`（[cargo-audit README](https://github.com/rustsec/rustsec/blob/main/cargo-audit/README.md)）。

**cargo-deny**：超集——`advisories`（漏洞/unmaintained/yanked）+ `bans`（多版本、wildcard）+ `licenses`（白名单）+ `sources`（未知 registry/git 源）。Quickstart：`cargo install --locked cargo-deny && cargo deny init && cargo deny check`；GitHub Action 一段式接入 `EmbarkStudios/cargo-deny-action`（[cargo-deny book](https://embarkstudios.github.io/cargo-deny/)）。

**CI 怎么跑**（cargo-deny-action README 推荐模式，[README](https://github.com/EmbarkStudios/cargo-deny-action)）：
- `advisories`：**定时 schedule**（矩阵写法 + `continue-on-error: true`，*"Prevent sudden announcement of a new advisory from failing ci"*）——新漏洞公告不该让日常 CI 突然全红，而是定时暴露。
- `bans licenses sources`：只在依赖变更时跑（`paths: ['**/Cargo.lock', '**/Cargo.toml', '**/deny.toml']` 过滤）。

**deny.toml 怎么配**：参考 cargo-deny 自己的 [deny.toml](https://github.com/EmbarkStudios/cargo-deny/blob/master/deny.toml)：`[graph] all-features = true`、`[advisories] unmaintained = "workspace"`、`[bans] multiple-versions = "deny"`、`[licenses] allow = ["Apache-2.0", "MIT", ...]`、`[sources] unknown-registry = "deny"`（`unmaintained = 'workspace'` 语义见 [cargo-deny advisories](https://embarkstudios.github.io/cargo-deny/checks/advisories/index.html)）。aaos 的 rusqlite bundled / blake3 / tokio 等依赖多为 MIT/Apache，白名单即可。

```yaml
  cargo-deny:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        checks: [advisories, 'bans licenses sources']
    continue-on-error: ${{ matrix.checks == 'advisories' }}
    steps:
      - uses: actions/checkout@v4
      - uses: EmbarkStudios/cargo-deny-action@v2
        with:
          command: check ${{ matrix.checks }}
```

## 5. coverage：cargo-tarpaulin vs cargo-llvm-cov

**结论：推荐 cargo-llvm-cov；上传 codecov 可选。**

- **cargo-tarpaulin**：Linux x86_64 上默认 ptrace 引擎（README：*"Tarpaulin's default tracing backend is still Ptrace and will only work on x86_64 processors"*），可 `--engine llvm`；自带 coveralls/codecov 上报、`--fail-under` 门槛、`tarpaulin.toml` 配置、`--workspace`/`--all-features`。llvm 引擎已知坑：测试非零退出码时该测试的覆盖数据丢失，fork 类 syscall 处理不了（[tarpaulin README](https://github.com/xd009642/tarpaulin)）。
- **cargo-llvm-cov**：包装 rustc 原生 `-C instrument-coverage`，行/区域覆盖精准，branch 覆盖需 nightly；输出 `--lcov`/`--codecov`/`--cobertura`；自带 `--fail-under-lines`/`--fail-under-functions` 门槛；**默认排除依赖与 vendored 代码**，也可 `--include-ffi` 覆盖链接的 C/C++（对 rusqlite bundled 的 C 不是必需——那是依赖代码）；安装用 `taiki-e/install-action@cargo-llvm-cov`（[cargo-llvm-cov README](https://github.com/taiki-e/cargo-llvm-cov)）。
- **CI 配法**（README 官方示例）：`cargo llvm-cov --all-features --workspace --lcov --output-path lcov.info` + `codecov/codecov-action@v5`，`token: ${{ secrets.CODECOV_TOKEN }}`（*"required for private repos or protected branches"*）、`fail_ci_if_error: true`；想要区域覆盖换 `--codecov` 输出。
- **要不要上传**：aaos 是本地私有项目，上传 codecov 需要额外 token/账号；不上传也能用：job 里 `--fail-under-lines 80` 类门禁 + `--html` 生成本地报告。上传与否不是覆盖率的必要条件。

## 6. release workflow

**结论：0.1.0 阶段暂不配完整 release 流程；首次发布前再上。工具推荐 cargo-release（本地驱动）或 release-plz（CI 全自动）。**

- **cargo-release 怎么用**：默认 **dry-run**，预览后 `--execute` 真执行（README：*"By default, cargo-release runs in dry-run mode... Once you are ready, pass the `--execute` flag"*）；流程 = version bump（`[LEVEL|VERSION]`：patch/minor/major/alpha/beta/rc/release）→ commit → tag → push → `cargo publish`；完整支持 workspace（`--workspace`/`--exclude`、`release.toml` 或 `[workspace.metadata.release]` 配置：`allow-branch`、`tag-name = "{{prefix}}v{{version}}"`、`publish`、`push-remote` 等）：[cargo-release README](https://github.com/crate-ci/cargo-release)、[cargo-release reference](https://github.com/crate-ci/cargo-release/blob/master/docs/reference.md)。注意 `release` 级别是去预发布号（0.1.0-alpha.1 → 0.1.0），适合 0.x 预发布节奏。
- **crates.io 发布**：cargo book 强调发布基本不可撤销（*"a publish is generally permanent"*），发布前先 `cargo publish --dry-run`；首个发布前必须补齐 license/description/repository/readme 元数据（[cargo book publishing](https://doc.rust-lang.org/cargo/reference/publishing.html)）。token 用 `cargo login`，CI 里用 env `CARGO_REGISTRIES_CRATES_IO_TOKEN`。
- **GitHub Release 自动创建**：cargo-release 只负责 tag/push/publish，**不建 GitHub Release**（README 功能列表无此项）；release-plz 会基于 git tag 自动发布 GitHub Release + crates.io + git-cliff changelog + cargo-semver-checks（[release-plz README](https://github.com/release-plz/release-plz)）。所以"tag 驱动自动建 Release"要么 release-plz，要么 workflow 里 `gh release create`（tag 触发）。
- **0.1.0 要不要现在配**：**不要**。理由：发布是不可逆的一次性动作（cargo book），内部迭代期配 release 自动化是纯维护负担（YAGNI）；且 aaos crates 连 license/description 都没有，发布前提未到。届时二选一：cargo-release 本地命令或 release-plz（GitHub Action）。

## 7. 缓存策略

**结论：继续用 Swatinem/rust-cache@v2（已在正确位置），可选微调 `save-if`；sccache 暂不需要。**

- rust-cache 核心机制（[README](https://github.com/Swatinem/rust-cache)）：缓存 `~/.cargo` + `./target`（只含依赖产物，workspace 自身 crate 不缓存，*"workspace crates themselves are not cached since doing so is generally not effective"*）；cache key = job id + rustc 版本 + RUST* 等 env 前缀 + Cargo.lock/Cargo.toml/.cargo/config hash；**必须放在 toolchain 安装之后**（aaos 现有顺序正确）；自动 `CARGO_INCREMENTAL=0`（aaos 手动设的是冗余但无害）；GitHub 缓存 10GB 上限，超限逐出旧缓存。
- 可选改进：`save-if: ${{ github.ref == 'refs/heads/main' }}`——PR 只恢复不写入（README 官方示例：*"To only cache runs from master"*）；`workspaces` 参数（多 workspace 时用）aaos 单 workspace 不需要。
- **sccache 要不要加**：不加。sccache 是跨 job/runner 的编译缓存（rustc + C/C++ 都可，[README](https://github.com/mozilla/sccache)），rust-lang/rust 用 sccache + S3 是因为几千个 job；aaos 规模（2 job、单 OS、免费 runner）下 rust-cache 缓存 target 目录已把 rusqlite 的 C 编译产物一起缓存，sccache 收益抵不上运维成本。

## 8. rust-toolchain.toml

**结论：现阶段不建议加；CI 用显式逐 job toolchain（当前写法即业界主流，tokio/ripgrep 模式）。**

- 语义（[rustup book overrides](https://rust-lang.github.io/rustup/overrides.html)）：`rust-toolchain.toml` 格式 `[toolchain] channel = ... / components = [...] / targets = [...] / profile = "minimal"`，适合入库；工具链选择优先级：`+toolchain` 简写 > `RUSTUP_TOOLCHAIN` > 目录 override > `rust-toolchain.toml` > default。
- 与 CI 动作的交互（[dtolnay/rust-toolchain action.yml](https://github.com/dtolnay/rust-toolchain/blob/master/action.yml)）：动作做 `rustup toolchain install ...` + `rustup default <toolchain>`——default 优先级低于仓库内 `rust-toolchain.toml`，**加了文件后 dtolnay 选的 channel 会被文件覆盖**，想用别的 channel 必须 `cargo +nightly ...` 或设 `RUSTUP_TOOLCHAIN`。
- aaos 需求是 **stable 构建 + nightly fmt**。`rust-toolchain.toml` 只能表达一个 channel，两方案：
  1. **不加文件，CI 逐 job 显式指定**（推荐）：tokio 用 env 变量、ripgrep 用矩阵 + `dtolnay/rust-toolchain` 的 `toolchain` 输入。零配置最灵活。
  2. 加 `rust-toolchain.toml`（`channel = "stable"`）统一本地体验，CI fmt job 改用 `cargo +nightly fmt --all -- --check`（`+` 简写优先级最高）；代价是 CI 所有 job 都要用 `+` 形式显式绕开，复杂度上升。

## 9. 针对 aaos 的具体建议（按优先级）

| # | 动作 | Job | 依据 |
|---|------|-----|------|
| 1 | **新增 fmt job**：pinned nightly + rustfmt component + `cargo fmt --all -- --check` | 新增 `fmt` | rustfmt Configurations.md（#4991/#5083）、ripgrep/tokio CI |
| 2 | **clippy 加 `--all-features`（+ `--keep-going`）**，满足 issue #5 验收；可选把 `-D warnings` 换成 `CARGO_BUILD_WARNINGS: deny` | 改 `clippy` | clippy book、cargo book、clippy README |
| 3 | **新增 docs job**：stable + `RUSTDOCFLAGS: -D warnings` + `cargo doc --no-deps --document-private-items --workspace` | 新增 `docs` | ripgrep/tokio CI |
| 4 | **新增 cargo-deny**：`deny.toml`，advisories 走 schedule（continue-on-error），bans/licenses/sources 走依赖变更 | 新增 `deny` | cargo-deny book、cargo-deny-action README |
| 5 | （可选）rust-cache 加 `save-if: ${{ github.ref == 'refs/heads/main' }}` | 改 test/clippy | rust-cache README |
| 6 | （可选）`concurrency: cancel-in-progress` | 全局 | tokio/rust CI |
| 7 | coverage（cargo-llvm-cov + `--fail-under-lines`，上传 codecov 可选） | nice-to-have | cargo-llvm-cov README |
| 8 | release workflow、MSRV、miri、sccache | **暂缓** | 0.1.0 阶段，无 rust-version、无 unsafe 压力、规模小 |

**特别提醒**：改 clippy 命令若现状代码有 feature 相关配置问题会立刻暴露（当前无 feature，预期直接绿）；fmt job 上线前先本地用 pinned nightly `cargo fmt --all -- --check` 过一次，确认仓库当前格式与所选 nightly rustfmt 匹配（rustfmt 版本差异可能导致首次大面积 diff）。

## 参考资料

- Cargo Book — Continuous Integration：https://doc.rust-lang.org/cargo/guide/continuous-integration.html
- Cargo Book — Publishing：https://doc.rust-lang.org/cargo/reference/publishing.html
- Rustup Book — Overrides：https://rust-lang.github.io/rustup/overrides.html
- Rustfmt — Configurations.md：https://github.com/rust-lang/rustfmt/blob/main/Configurations.md · #4991 / #5083
- Clippy Book — CI (GitHub Actions)：https://github.com/rust-lang/rust-clippy/blob/master/book/src/continuous_integration/github_actions.md
- Clippy README（CARGO_BUILD_WARNINGS）：https://github.com/rust-lang/rust-clippy
- ripgrep CI：https://github.com/BurntSushi/ripgrep/blob/master/.github/workflows/ci.yml
- tokio CI：https://github.com/tokio-rs/tokio/blob/master/.github/workflows/ci.yml
- rust-lang/rust CI：https://github.com/rust-lang/rust/blob/master/.github/workflows/ci.yml
- Swatinem/rust-cache：https://github.com/Swatinem/rust-cache
- mozilla/sccache：https://github.com/mozilla/sccache
- cargo-audit / RustSec：https://github.com/rustsec/rustsec
- cargo-deny book：https://embarkstudios.github.io/cargo-deny/
- cargo-deny-action：https://github.com/EmbarkStudios/cargo-deny-action
- cargo-tarpaulin：https://github.com/xd009642/tarpaulin
- cargo-llvm-cov：https://github.com/taiki-e/cargo-llvm-cov
- cargo-release：https://github.com/crate-ci/cargo-release
- release-plz：https://github.com/release-plz/release-plz
- dtolnay/rust-toolchain：https://github.com/dtolnay/rust-toolchain
