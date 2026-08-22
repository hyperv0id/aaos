# aaos GitHub 自动化调研报告

> 调研日期：2026-08-22 · 范围：Issue 模板、PR 模板、CODEOWNERS、Dependabot、Conventional Commits
> 方法：GitHub 官方文档 5 篇 + 知名 Rust 仓库实际配置（rust-lang/rust、tokio-rs/tokio、BurntSushi/ripgrep、helix-editor/helix、firecracker-microvm/firecracker、clap-rs/clap）+ Conventional Commits 1.0.0 + aaos 本地现状

## 1. Issue 模板格式：Issue Forms vs Markdown 模板

**结论：GitHub 现在推荐 Issue Forms（YAML `.yml`），它提供结构化字段输入；但 markdown 模板（`.md`）仍完全支持且更灵活。知名 Rust 仓库两种都有用。**

GitHub 官方文档：
- [About issue and pull request templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates)：模板引导用户在开 issue/PR 时提供有用信息。
- [Configuring issue templates](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository)：可创建多个模板，配 `config.yml` 控制行为；Issue Forms（YAML schema）提供表单字段（dropdown、checkbox），markdown 模板只提供预填文本。

**参考仓库**：
| 仓库 | 格式 | 文件 |
|------|------|------|
| rust-lang/rust | markdown + YAML 混合 | `.github/ISSUE_TEMPLATE/bug_report.md`、`diagnostics.yaml` |
| tokio-rs/tokio | markdown | `.github/ISSUE_TEMPLATE/bug_report.md` |
| BurntSushi/ripgrep | Issue Form (YAML) | `.github/ISSUE_TEMPLATE/bug_report.yml` |

**Issue Forms 优势**：结构化输入（dropdown 选择、必填校验）、自动打标签（`labels` 字段）、GitHub UI 渲染为表单。**Markdown 模板优势**：更灵活、支持任意 markdown、兼容性广。

**建议**：aaos 用 Issue Forms（YAML）做 bug report 和 feature request——结构化输入对 agent 驱动的 issue 处理更友好（字段可预测），且可自动打 `bug`/`enhancement` 标签。

## 2. Issue 模板内容

**结论：配 3 个模板——bug report、feature request、config.yml。**

**Bug report 模板字段**（参考 ripgrep bug_report.yml + tokio bug_report.md）：
- 描述（textarea，必填）
- 复现步骤（textarea，必填）
- 预期行为（textarea）
- 实际行为（textarea）
- 版本/环境（input：aaos 版本、OS、Rust 版本）
- 自动标签：`bug`、`needs-triage`

**Feature request 模板字段**（参考 ripgrep feature_request.md）：
- 问题陈述（textarea：要解决什么问题）
- 期望方案（textarea：想要什么功能）
- 替代方案（textarea：考虑过什么替代）
- 自动标签：`enhancement`

**config.yml**：
GitHub 官方 [config.yml 文档](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository#configuring-template-keepers)：控制 issue template 行为。

```yaml
blank_issues_enabled: false
contact_links:
  - name: 讨论与提问
    url: https://github.com/hyperv0id/aaos/discussions
    about: 使用 Discussions 进行问答和讨论，不要开 issue
```

`blank_issues_enabled: false` 阻止空白 issue（强制用户选模板）；`contact_links` 引导非 bug/feature 的提问去 Discussions（若开启）。

## 3. config.yml 作用

**结论：控制模板列表是否展示、是否允许空白 issue、添加外部链接。**

如上。aaos 建议 `blank_issues_enabled: false`（强制模板）+ 可选 contact_links。

## 4. PR 模板

**结论：用 `.github/pull_request_template.md`，含 checklist。**

[GitHub 官方文档](https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository)：PR 模板在创建 PR 时自动填充 body。

**参考仓库**：
- [tokio PR 模板](https://github.com/tokio-rs/tokio/blob/master/.github/PULL_REQUEST_TEMPLATE.md)：简短 checklist + 指向 contributing guide。
- [rust-lang/rust PR 模板](https://github.com/rust-lang/rust/blob/main/.github/pull_request_template.md)：含 A- 标签指引、review 指南。
- [firecracker PR 模板](https://github.com/firecracker-microvm/firecracker/blob/main/.github/pull_request_template.md)：详细 checklist（测试、文档、license）。

**aaos PR 模板建议**：

```markdown
## 变更说明

<!-- 简述此 PR 做了什么、为什么 -->

## 关联 Issue

<!-- Closes #N / Refs #N -->

## Checklist

- [ ] 提交信息遵循 [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] `cargo fmt --all -- --check` 通过
- [ ] `cargo clippy --workspace --all-targets --all-features --locked -- -D warnings` 通过
- [ ] `cargo test --workspace` 通过
- [ ] 新增功能有对应测试
- [ ] 领域概念用词遵循 [CONTEXT.md](../CONTEXT.md) 词汇表
```

## 5. CODEOWNERS

**结论：单人维护者仓库暂不需要 CODEOWNERS。**

[GitHub 官方 CODEOWNERS 文档](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners)：CODEOWNERS 指定谁自动被请求 review 某些路径的 PR。格式：`path/pattern @username` 或 `path/pattern @team`。

**关键事实**：GitHub [protected branches 文档](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)明确——PR 作者**不能批准自己的 PR**。如果开启 branch protection + required reviews + CODEOWNERS，单人维护者会被自锁：自己的 PR 永远无法自己批准，导致无法合并。

**参考仓库**：
- tokio：无 CODEOWNERS（多维护者通过 team 管理）。
- firecracker：有 CODEOWNERS（团队项目，按子系统分配）。

**建议**：aaos 是单人维护者（hyperv0id），不配 CODEOWNERS、不开 required reviews。将来有多人贡献者时再配。

## 6. Dependabot

**结论：配 cargo + github-actions 双 ecosystem。**

[Dependabot 配置文档](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file)：`.github/dependabot.yml` 控制 Dependabot 版本更新。

**参考仓库**：
- [helix dependabot.yml](https://github.com/helix-editor/helix/blob/master/.github/dependabot.yml)：cargo + github-actions 双 ecosystem，`interval: weekly`——**helix 是 Rust workspace + cargo dependabot 的好实例**。
- [clap dependabot.yml](https://github.com/clap-rs/clap/blob/main/.github/dependabot.yml)：cargo ecosystem。
- [tokio dependabot.yml](https://github.com/tokio-rs/tokio/blob/master/.github/dependabot.yml)：cargo ecosystem。

**aaos 建议**：

```yaml
version: 2
updates:
  - package-ecosystem: "cargo"
    directory: "/"
    schedule:
      interval: "weekly"
    open-pull-requests-limit: 10
    labels:
      - "dependencies"
    commit-message:
      prefix: "chore(deps)"

  - package-ecosystem: "github-actions"
    directory: "/"
    schedule:
      interval: "weekly"
    labels:
      - "dependencies"
      - "ci"
    commit-message:
      prefix: "chore(ci)"
```

`interval: weekly` 是主流选择（helix/tokio/clap 均用周级）。cargo ecosystem 自动检测 `Cargo.toml` + `Cargo.lock` 变更。`commit-message.prefix` 让 Dependabot PR 也遵循 Conventional Commits。

## 7. Conventional Commits

**结论：aaos 现有提交历史已隐含此格式，正式写入 CONTRIBUTING 即可。**

[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/)：
- 格式：`<type>[scope]: <description>` + 空行 + optional body + 空行 + optional footer。
- `fix` = PATCH，`feat` = MINOR，`BREAKING CHANGE` footer 或 `type!:` = MAJOR。
- scope 为括号内名词，标识影响范围。
- 推荐 type：`feat`/`fix`/`chore`/`docs`/`refactor`/`test`/`perf`/`ci`/`build`/`style`。
- footer：`BREAKING CHANGE: <说明>` 或 `Fixes: #123` / `Refs: #123`。

**aaos 现有提交历史**（`git log --oneline`）已隐含此格式：
```
feat(lints): adopt workspace-level clippy config and rustfmt.toml
feat(session-store): SQLite structural source of truth (ADR-0001)
fix(pi-agent-core): continue validation, addedToolNames, Off→None
chore(pi-agent-core): cleanup pass — drop ActiveRun abort_tx dead field
test(pi-agent-core): pin empty-transcript continue error to exact variant
```

**aaos scopes**：`core`、`cli`、`openai`、`tools`、`catalog`、`session`、`session-store`、`ci`、`docs`、`workspace`、`lints`。

正式写入 CONTRIBUTING.md 即可，无需额外工具或 CI 强制（0.1.0 阶段 YAGNI）。

## 针对 aaos 的具体建议

| # | 动作 | 依据 |
|---|------|------|
| 1 | 配 2 个 Issue Forms（bug_report.yml + feature_request.yml）+ config.yml | ripgrep issue form、GitHub 官方文档 |
| 2 | 配 `.github/pull_request_template.md`（含 checklist） | tokio/rust/firecracker PR 模板 |
| 3 | **不配 CODEOWNERS**（单人维护者会自锁） | GitHub protected branches 文档 |
| 4 | 配 `.github/dependabot.yml`（cargo + github-actions，weekly） | helix/tokio/clap dependabot |
| 5 | 正式采用 Conventional Commits，写入 CONTRIBUTING | conventionalcommits.org + 现有提交历史 |
| 6 | 可选：PR 标题用 CC 格式（squash merge 时标题即 commit message） | tokio PR workflow |

## 参考资料

- GitHub — Issue/PR templates 概述：https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/about-issue-and-pull-request-templates
- GitHub — 配置 issue 模板：https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/configuring-issue-templates-for-your-repository
- GitHub — 创建 PR 模板：https://docs.github.com/en/communities/using-templates-to-encourage-useful-issues-and-pull-requests/creating-a-pull-request-template-for-your-repository
- GitHub — About CODEOWNERS：https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-code-owners
- GitHub — Protected branches：https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches
- GitHub — Dependabot 配置：https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/configuration-options-for-the-dependabot.yml-file
- rust-lang/rust ISSUE_TEMPLATE：bug_report.md、config.yml、diagnostics.yaml
- tokio-rs/tokio：PULL_REQUEST_TEMPLATE.md、dependabot.yml、ISSUE_TEMPLATE/bug_report.md
- BurntSushi/ripgrep：ISSUE_TEMPLATE/bug_report.yml、config.yml、feature_request.md
- helix-editor/helix：dependabot.yml（cargo + github-actions）
- clap-rs/clap：dependabot.yml
- firecracker-microvm/firecracker：CODEOWNERS、pull_request_template.md、dependabot.yml
- Conventional Commits 1.0.0：https://www.conventionalcommits.org/en/v1.0.0/
