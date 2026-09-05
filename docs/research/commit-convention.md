# aaos 提交信息规范调研报告

> 调研日期：2026-08-23 · 范围：Conventional Commits 1.0.0 规范要点、类型集来源、强制工具链（GitHub Action / 本地 hook / 全历史检查）、aaos 本地历史合规度取证
> 方法：primary source 6 篇（Conventional Commits 1.0.0、Angular CONTRIBUTING.md、action-semantic-pull-request README、gitlint docs、cocogitto README、GitHub changelog）+ 本地 `git log` 全量分析（53 个提交逐条核对）

> [!WARNING]
> 本文为 2026-08-23 的调研快照。文中提到的 9 种 type 现已扩为 10 种（补了 `revert`），
> pr-title.yml 强制门禁已落地（§1 的建议 4/5 均已实施）。现状以 [`CONTRIBUTING.md`](../../CONTRIBUTING.md) 为准。

## 0. 结论先行

**规范不是"要不要定"——已经定了，缺的是强制；"乱七八糟"的是 6 个早期导入提交，此后全部合规。**

1. `CONTRIBUTING.md` §提交规范已完整采用 Conventional Commits 1.0.0（格式、9 种 type、scope 建议、分支命名、squash 流程），2026-08-22 随 #7 落地。
2. 取证：53 个提交中 47 个合规（89%）；不合规的 6 个全部落在 2026-08-20~22 初始导入期（无 type 前缀 5 个 + GitHub merge commit 1 个）。**#7 之后合规率 100%。**
3. 真正的缺口是**强制**：CI（fmt/clippy/test/docs/deny）不检查提交信息，PR 模板只有手动 checklist 项，纯自愿。
4. 推荐强制点 = **PR 标题**（本项目 squash merge，PR 标题即为最终提交标题）：`amannn/action-semantic-pull-request@v6`，一个 GitHub Action，零本地工具，无 Node/Python 依赖。背书：Electron、Vite、Excalidraw、Apache Pulsar、Vercel、Microsoft SynapseML、Firebase、AWS。
5. 需人工补一个仓库设置：Pull requests → "Default to PR title for squash merge commits"（GitHub 2022-05-11 引入）。否则单 commit PR 时 GitHub 会建议用 commit message 代替 PR 标题，绕过检查。
6. 历史 6 个提交**不改写**——改写会改 SHA，收益为零（CC 规范 FAQ 明确：不合规提交只是被工具"跳过"，不是事故）。
7. 发布阶段（目前未打版本，YAGNI）：可选 cocogitto（Rust 单二进制，`cog bump`/`cog changelog`，按 CC 历史驱动 SemVer）或 git-cliff。

## 1. 规范本身：Conventional Commits 1.0.0

来源（primary）：[conventionalcommits.org/en/v1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)

格式：`<type>[optional scope]: <description>` + optional body + optional footer(s)。

**规范强制（MUST）的少，自由的多：**

| 要素 | 规范态度 |
|------|---------|
| `type` + 冒号空格 + description | MUST（强制） |
| `feat` | MUST 用于新功能（↔ SemVer MINOR） |
| `fix` | MUST 用于 bug 修复（↔ SemVer PATCH） |
| `BREAKING CHANGE` footer 或 `type!:` | 表示破坏性变更（↔ SemVer MAJOR），任何 type 都可带 |
| scope | MAY，括号内名词 |
| 其他 type | MAY——规范明确 "types other than feat and fix are allowed"，推荐集来自 [commitlint config-conventional / Angular](https://github.com/conventional-changelog/commitlint/tree/master/@commitlint/config-conventional) |
| footer | MAY，git trailer 格式；token 用 `-` 连字（`Acked-by`），`BREAKING CHANGE` 例外 |
| 大小写 | 除 `BREAKING CHANGE` 必须大写外一律不敏感 |

**FAQ 关键三条（原文要点）：**

- **"贡献者都要学会吗？"——不用。** 原文："If you use a squash based workflow ... lead maintainers can clean up the commit messages as they're merged—adding no workload to casual committers." 本项目正是这种 flow：贡献者随手写，squash 时定稿。
- **"初始开发阶段怎么办？"——按已发布标准执行。** 这一点 aaos 已做到。
- **"用错/非规范 type 怎么办？"** 合并前用 `git rebase -i` 改；合并/发布后不合规提交只是被工具跳过，"not the end of the world"。
- revert 推荐：`revert:` type + `Refs:` footer 引用被 revert 的 SHA。

## 2. 类型集的两条来源线

| 来源 | 类型集 |
|------|--------|
| **Angular 官方**（[commit-message-guidelines.md](https://github.com/angular/angular/blob/main/contributing-docs/commit-message-guidelines.md)，primary） | `build` `ci` `docs` `feat` `fix` `perf` `refactor` `test`（8 种；无 chore/style；build 涵盖构建系统与外部依赖） |
| **commitlint config-conventional**（CC 规范点名推荐） | Angular 8 种 + `chore` `revert` `style`（`chore` 覆盖构建/依赖/杂项） |
| **aaos CONTRIBUTING.md 现状** | `feat` `fix` `chore` `docs` `refactor` `test` `perf` `ci` `style`（9 种，融合两线：用 chore 承接 Angular 的 build 职责，另加 ci/style） |
| **aaos 实际使用** | `feat` `fix` `chore` `docs` `refactor` `test` `perf`——与文档一致，无越界 type |

结论：aaos 的 type 集是合理融合，无需改。建议补一个 `revert`（Angular/commitlint 都有，成本零）。

## 3. aaos 现状取证

### 3.1 历史合规度（`git log` 全量 53 条逐条核对）

| 指标 | 数值 |
|------|------|
| 总提交 | 53 |
| 合规 | 47（89%） |
| 不合规 | 6（11%）——全部在 2026-08-20 ~ 08-22 初始导入期，早于 #7（08-22）规范落地 |

不合规清单（原因：无 type 前缀）：

```
54ca2eb 2026-08-22 Redesign CLI output: human tool rendering, message-level JSON stream (#3)
4407ae4 2026-08-22 Merge pull request #1 from hyperv0id/feat/tool-schema-session
ad9d8f5 2026-08-20 Satisfy clippy -D warnings so CI clippy job is green.
6b2cfb0 2026-08-20 Add GitHub Actions CI for workspace tests and clippy.
8002894 2026-08-20 Add agent skills config for GitHub issues and domain docs.
603b327 2026-08-20 Add models.dev catalog, OpenAI Completions streaming, and aaos CLI.
```

### 3.2 现有强制缺口

- CI（`.github/workflows/ci.yml`）5 个 job（fmt/clippy/test/docs/deny）均与提交信息无关。
- PR 模板 checklist 有"提交信息遵循 Conventional Commits"——手动项，无门禁。
- Dependabot 已配 `commit-message.prefix: chore(deps)` / `chore(ci)`（`.github/dependabot.yml`），机器人 PR 标题天然合规 ✅。

### 3.3 scope 的文档与实践偏差（重要）

- CONTRIBUTING.md 建议 scope：`core` `cli` `providers` `tools` `session-store` `ci` `docs` `workspace` `lints`。
- 实际提交用：`pi-agent-core` `aaos-providers` `aaos-cli` `session-store` `naming` `adr` `deps` `skills` `workspace` `ci` `docs`——crate 全名是主流。

结论：**若未来在 Action 里配 `scopes` 白名单，会立即拦掉现状一半的提交风格。** 初期只强制 type，不强制 scope（action 默认亦不强制）。scope 语义留待发布阶段（changelog 生成）再统一。

## 4. 强制手段对比

| 手段 | 工具 | 依赖 | 强制时点 | 评价 |
|------|------|------|---------|------|
| **PR 标题 lint** | [amannn/action-semantic-pull-request@v6](https://github.com/amannn/action-semantic-pull-request) | 无（GitHub Action） | merge 前，不可绕过 | ✅ 推荐。squash flow 下 PR 标题=最终提交标题，一处检查两端生效；`pull_request_target` 触发对 fork PR 安全；`with.types` 白名单与 CONTRIBUTING 对齐；`validateSingleCommit` 可兜底 GitHub 用 commit message 代替 PR 标题的 case（根治手段是仓库设置） |
| **本地 commit-msg hook** | gitlint（Python）/ commitlint（Node）/ cocogitto `cog check`（Rust） | 需装工具 | 提交时，`--no-verify` 可绕过 | 可选。单人作者体验辅助；纯 Rust 仓库引入 Python/Node 只为 hook 不值得；cocogitto 是 Rust 单二进制但当前无版本发布需求 |
| **提交后全历史检查** | gitlint（CI 模式）/ cocogitto-action | 同上 | push/PR 后 | 不推荐。与 squash 流程冗余；gitlint 默认规则（标题长度、trailer 等）会立刻打红 6 个 legacy 提交，需要额外配置豁免，复杂度不值 |

**选型关键事实（primary sources）：**

- amannn action README 要求/建议：PR 标题单行，breaking change 只能用 `!` 表示；配合仓库设置 "Default to PR title for squash merge commits" [（GitHub changelog 2022-05-11）](https://github.blog/changelog/2022-05-11-default-to-pr-titles-for-squash-merge-commit-messages/) 使用；`pull_request_target` 下配置取自 master——首落地需先合入 master。默认 types 来自 commitizen/conventional-commit-types 全集，可 `with.types` 收紧为 aaos 的 9 种。
- gitlint（[jorisroovers.com/gitlint](https://jorisroovers.com/gitlint/)）：Python，支持 commit-msg hook / pre-commit / CI 脚本，全 unicode（中文提交信息 OK）。
- cocogitto（[cocogitto/cocogitto](https://github.com/cocogitto/cocogitto)）：Rust 单二进制（仅依赖 libgit2），`cog commit/check/bump/changelog`，含 GitHub action 与 bot；monorepo 版本支持——若将来 aaos workspace 需自动版本，这是最顺手的。

## 5. 历史提交处理建议

**不改写。** 依据：

- CC FAQ：初始阶段的历史不合规提交"被工具跳过"即可，不是事故。
- 改写需要 `git filter-repo`/`rebase -i` + force-push，6 个 SHA 全部变化；当前无外部贡献者、无基于历史的工具依赖，改动收益为零。
- 若强迫症发作：一次性 `git filter-repo --msg-filter` 给 6 个提交加 type 前缀，只影响这 6 个的 SHA，无需在 master 之外任何分支处理。**不推荐，除非有 changelog 回填需求。**

## 6. 建议落地清单

| # | 动作 | 成本 | 备注 |
|---|------|------|------|
| 1 | 新增 `.github/workflows/pr-title.yml`：`pull_request_target`（opened/reopened/edited/synchronize）+ `amannn/action-semantic-pull-request@v6`，`with.types` = aaos 9 种 | 1 个文件 | 唯一新配置；fork PR 安全 |
| 2 | GitHub 仓库设置勾选 **Default to PR title for squash merge commits** | 人工一次 | 根治 GitHub "单 commit PR 用 commit message" 的建议路径 |
| 3 | CONTRIBUTING.md 补 `revert` type | 一行 | 与 Angular/commitlint 对齐 |
| 4 | 历史不改写 | 0 | §5 |
| 5 | 发布阶段：cocogitto（changelog + SemVer） | 届时再说 | 现在 YAGNI；scope 统一也放到那时 |

## 7. 参考资料

- Conventional Commits 1.0.0（spec，primary）：https://www.conventionalcommits.org/en/v1.0.0/
- Angular commit message guidelines：https://github.com/angular/angular/blob/main/contributing-docs/commit-message-guidelines.md
- amannn/action-semantic-pull-request（README）：https://github.com/amannn/action-semantic-pull-request
- GitHub changelog — Default to PR titles for squash merge commits：https://github.blog/changelog/2022-05-11-default-to-pr-titles-for-squash-merge-commit-messages/
- gitlint：https://jorisroovers.com/gitlint/ · https://github.com/jorisroovers/gitlint
- cocogitto：https://github.com/cocogitto/cocogitto
- 本地取证：`git log --format='%h %ad %s' --date=short`（53 条逐条核对）、`.github/workflows/ci.yml`、`.github/pull_request_template.md`、`.github/dependabot.yml`、`CONTRIBUTING.md`
