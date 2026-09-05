# Harness 化改造 · 问题清单与解决计划

依据 [OpenAI Harness Engineering](https://openai.com/index/harness-engineering/) 最佳实践对本仓库的全面扫描（2026-09-05，分支 `harnessing`，基线 commit `0b43aa5`）。

**最佳实践核心**：仓库是系统事实源；AGENTS.md 是 ~100 行目录页而非百科全书（渐进披露）；只写不变量 + 验证命令，不微管理实现；架构不变量机械化执行；定期垃圾回收防熵增。

**扫描基线**：`cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` 与 `cargo test --workspace --locked` 均干净通过。三路只读扫描：代码质量 / 文档体系 / 门禁覆盖。

---

## A. 代码质量（crates/）

| # | 严重度 | 问题 | 证据 | 修复方向 |
|---|--------|------|------|----------|
| A1 | Important | `Agent::reset()` 用 `panic!` 处理可恢复错误，与同前提的 `prompt()`（返回 `Err(AlreadyProcessing)`）不一致；是核心 crate 唯一生产 panic | `crates/pi-agent-core/src/agent.rs:415-424` | reset 改为返回 `Result<(), AgentError>`，调用点（`run_prompt_messages` reset 路径与文档化用法）同步迁移 |
| A2 | Important | workspace lints 不覆盖 `panic!`/`unreachable!`/`todo!`/`unimplemented!` 宏族，4 处生产宏全部逃过 `-D warnings` 门禁 | 根 `Cargo.toml` `[workspace.lints.clippy]` 仅 deny `dbg_macro`、warn `unwrap_used`/`expect_used` | 增 `panic`/`unreachable`/`todo`/`unimplemented` 四条 deny——注意 `clippy::panic` 只覆盖 `panic!`/`todo!`/`unimplemented!`，**不**覆盖 `unreachable!`（后者需独立的 `clippy::unreachable`，实施时已按此办理）。存量 `unreachable!` 两处（`retry.rs:93,101`、`skills.rs:208`）加带理由 `#[expect]` 登记 |
| A3 | Important | `aaos-cli/src/compaction_coordinator.rs` 无自有测试：`CompactionSettings::from_env`、`pre_request_hook`、`post_turn_hook`、`compact` 及 overflow/silent 分支仅经 main.rs tests 间接覆盖 | `crates/aaos-cli/src/compaction_coordinator.rs`（全文件无 `#[cfg(test)]`） | 补直接单元测试：from_env 解析矩阵、hook 分支、overflow 判定 |
| A4 | Minor | `wait_aborted`/`abortable_sleep` 三处复制，跨 3 个文件 2 个 crate | `pi-agent-core/src/agent_loop.rs:206-225`、`aaos-providers/src/retry.rs:109-123`、`aaos-providers/src/formats/sse.rs:26` | 从 pi-agent-core 导出公共 helper，providers 两个文件改为复用 |
| A5 | Minor | 4 个 provider adapter 的 `call()` transport 样板复制（api_key 解析 + body 构建 + url + channel 装配） | `anthropic_messages.rs:260-264`、`cohere_chat.rs:257-260`、`google_genai.rs:185-188`、`openai_completions.rs:230-233` | transport 下沉到 `formats/sse.rs` 共享执行函数 |
| A6 | Minor | 5 crate 均未声明 `rust-version`（MSRV）；CI 只钉 @stable | 各 crate `Cargo.toml` | 统一声明 `rust-version = "1.85"`（edition 2024 下限），CI 锁定后可验 |
| A7 | Minor | 4 个 format 测试模块的裸 TCP SSE 桩 `serve(...)` 逐字节重复 | `anthropic_messages.rs:769-803`、`openai_completions.rs:619-663` 等 | 抽共享测试 helper（随 A5 顺手收敛，不单独立项） |

**确认干净**：无 TODO/FIXME/XXX 注释、无生产 unwrap/expect（仅测试模块内且带显式 allow）、无硬编码 secret/IP、无死代码；5 crate 均已声明 `[lints] workspace = true`；pi-agent-core/aaos-session 有 `tests/` 集成目录，providers/tools 靠 inline 测试 + cli wiremock 端到端，覆盖充分暂不补。

## B. 文档体系（根文档 + docs/）

| B1 | HIGH | AGENTS.md 仅 395B，只有 3 条技能指针：缺仓库概述、验证命令、文档地图、提交/CI 规范指针——违背"目录页"最佳实践 | `AGENTS.md`（14 行） | 重写为 ~60-100 行目录页：一句话定位 + crate 地图 + 验证命令 + 文档地图（何时读什么）+ 提交规范指针 |
| B2 | ~~HIGH~~ 误报 | ~~`CLAUDE.md` 为 0 字节空文件~~ 经核实 `CLAUDE.md` 是指向 `AGENTS.md` 的符号链接（git mode `120000`），内容天然同步，扫描工具把 symlink stat 成 0B 导致误判。无需处理 | `ls -la CLAUDE.md` → `CLAUDE.md -> AGENTS.md` | 无；注意向 CLAUDE.md 写入会穿符号链接覆盖 AGENTS.md 本体 |
| B3 | MED | ADR-0004 两处引用不存在的 `0003-static-instruction-chain.md`；真实 0003 是 meta-head-pointer，语义无关——编号引用笔误 | `docs/adr/0004-skills-internal-uri.md:7,12` | 改为不带链接的文字引用（发现边界与归属逻辑是 ADR-0004 自身决定，不依赖 0003 语义） |
| B4 | MED | README 文档表缺 `docs/agents/`、`docs/arch/`、`docs/superpowers/`；docs/arch（6 个 archify JSON 规格仅 pages.yml 消费）在根文档零说明 | `README.md:33-37` | 文档表补三行；一行说明 arch JSON 的更新路径 |
| B5 | MED | README providers 行用旧词「模型注册表 + 方言 adapter」，与 CONTEXT.md 词汇表冲突（规范词：目录/Catalog、API格式） | `README.md:10`；`CONTEXT.md` _Avoid: 注册表、方言_ | 换新词汇对齐词汇表 |
| B6 | MED | CONTRIBUTING scope 表缺 `deps`，但自家示例与 dependabot 都用 `chore(deps)` | `CONTRIBUTING.md:124` vs `:131`、`.github/dependabot.yml` | scope 表补 `deps` |
| B7 | MED | docs/research/{ci-cd,commit-convention}.md 的「现状」快照过时（7 crate/edition 2021/2 job → 实际 5 crate/2024/5 job；9 type → 10） | `docs/research/ci-cd.md:8-10`、`commit-convention.md:50` | 文首加状态横幅注明调研日期与已过时（compaction.md 有先例） |
| B8 | LOW | CONTRIBUTING 供应链描述过严（bans multiple-versions 实为 warn 非 deny） | `CONTRIBUTING.md:178` vs `deny.toml:18` | 措辞修正为「警告重复版本、禁止 wildcard」 |
| B9 | LOW | triage-labels.md 未覆盖仓库实际标签全集（bug/enhancement/dependencies/ci），feature_request 不自动带 needs-triage，triage 入口不对称 | `docs/agents/triage-labels.md`；`.github/ISSUE_TEMPLATE/feature_request.yml` | 文档补全标签映射；feature_request.yml 补 `needs-triage` 标签 |
| B10 | LOW | skills-lock.json 无维护说明，AGENTS.md 未指向 | `skills-lock.json` | AGENTS.md 文档地图一行带出 |
| B11 | LOW | CONTEXT.md 范围声明偏窄（声称覆盖两域，实际含 Agent 域词条） | `CONTEXT.md:3` | 范围声明改为三域 |

## C. 门禁（.github/ + 配置）

| # | 严重度 | 问题 | 证据 | 修复方向 |
|---|--------|------|------|----------|
| C1 | Important | test job 永不执行 doctest：`--all-targets` ≠ `--doc`，仓库有 2 处真实 doctest | `.github/workflows/ci.yml:67` | test job 增加独立 `cargo test --workspace --locked --doc` step |
| C2 | Important | test/docs job 缺 `--all-features`（clippy 有），未来 features 盲区 | `ci.yml:67,86` | 两处补 `--all-features` |
| C3 | Minor | ci.yml 无 concurrency 取消，过期 run 白耗 runner | `ci.yml` | 加 `concurrency: group=${{ github.workflow }}-${{ github.ref }}`, cancel-in-progress |
| C4 | Minor | deny job 未用 rust-cache，每次全量编译慢；deny.toml `unmaintained` 注释与行为不符（注释说 warn，`workspace` 行为是仅 workspace 依赖告警） | `ci.yml:88-97`、`deny.toml:7-9` | deny job 加 cache；deny.toml 注释修正 + yanked 显式声明 |
| C5 | 确认无需 | MSRV 门禁、cargo-semver-checks、本地 commit-msg hook、RUSTDOCFLAGS 断链 lint | CONTRIBUTING.md:169-171 明确未承诺 MSRV；0.1.0 未发布；squash 流下 pr-title 已覆盖提交信息；`-D warnings` 已含 broken_intra_doc_links | 不加，避免过度工程 |

---

## 实施记录（2026-09-05 完成，分支 `harnessing`）

| # | 内容 | 提交 | 实施偏差 |
|---|------|------|----------|
| C1-C4 + B9 标签 | 门禁补齐：doctest step、test/docs 补 --all-features、concurrency 取消、deny 加 cache、deny.toml 注释修正 + yanked deny、feature_request 补 needs-triage | `a22b489` ci | yanked 归属 [advisories]（官方 schema），非计划原写的 [bans] |
| B1 | AGENTS.md 目录页改造 | `7de06dc` docs(agents) | — |
| B2 | CLAUDE.md | 随 `7de06dc` | 实施中核实为 AGENTS.md 的 symlink（误报），无需处理 |
| B3-B9、B11 | 文档对齐：坏链、词汇、文档表、scope、research 横幅、triage 标签集、范围声明 | `b494aa5` docs | — |
| B10 | skills-lock.json 维护说明并入 AGENTS.md 文档地图末段 | 随 `7de06dc` docs(agents) | — |
| A1 | reset() 返回 Result 对齐 prompt | `9e97ce5` refactor(core)! | — |
| A2 | lints deny panic 宏族 + 存量登记 | `54b5974` chore(lints)! | 拦截面 ~9 倍于预估（36+ 处）；生产 unreachable 两处（retry.rs/skills.rs）带理由 #[expect]，panic 族宏全部位于测试模块（模块头一次 #![expect]；acceptance.rs 两处 panic! 改写为 matches! 断言故无需豁免；compaction_coordinator 新测试无 panic 族宏故无豁免） |
| A3 | compaction_coordinator 直接测试 25 个 | `1941a97` test(cli) | 提交信息误称登记了 #![expect(clippy::panic)]——实际该测试无 panic 族宏，仅有 unwrap/expect 豁免；已核实无功能影响 |
| A6 | MSRV 声明 | `59c0dea` chore(workspace) | 声明 1.88 而非 1.85（传递依赖 icu_* 实际顶高） |
| A4/A5/A7 | providers 三组重复收敛 | `17be2fd` refactor(providers) | 净 -180 行，公开 API 零变化 |

全量门禁收口：`cargo fmt --check` + `cargo clippy -D warnings` + `cargo test --workspace`（含 doctest）+ `cargo deny check`（0.20.2 四段全过）全部通过。
