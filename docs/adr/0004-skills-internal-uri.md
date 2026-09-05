# 0004 · skills 经内部 URI（skill://）按需读取

read 的 path 参数引入 scheme 分派：无 scheme 走文件系统（行为不变），`skill://<name>[/path]` 解析到单层发现的技能目录——name 直指该技能的 SKILL.md（可省略），/path 访问目录内文件。system prompt 只注入纯索引（name、description、`skill://` URI，**不含真实路径**），技能全文与捆绑资源由模型按需 read；read 结果附解析出的真实 sourcePath，模型据此按 SKILL.md 的相对路径指引推算脚本位置交 bash。决定性理由：统一寻址让"能力"成为可寻址内容（与 oh-my-pi 的 internal URL 体系同源，本方案是其 `skill://` 语义的移植）；索引零路径泄漏保持抽象完整，而真实路径在 read 时惰性披露、恰在模型需要它的时刻可得；文件系统存储使技能脚本天然可被 bash 执行，无需 materialize。

## 决定细节

- **发现**：用户级 `~/.agents/skills/` + project_root 的 `.agents/skills/`，单层发现（嵌套不生效），同名项目级压用户级。SKILL.md frontmatter 兼容既有约定：`name` 可省略（默认目录名）、`description` 必填、其余字段忽略。
- **索引**：`<skills>` 块逐条 `- name: description` + `skill://name`，配指令句"匹配到技能时先 read `skill://<name>`"；仅 read 工具存在时注入；不含任何真实路径。
- **read 分派语义**：`skill://name[/SKILL.md]` ≡ SKILL.md；`skill://name/` ≡ `skill://name`；`/path` 指向目录内文件，指向目录返回列举（仅 skill:// 有列举，fs 目录维持报错）；路径段 percent-decode 后拒绝绝对路径与 `..`，canonicalize 必须落在技能目录内；未知技能报错并列出可用名。
- **bash**：本决策不含 bash 的 URI 改写——另立 issue 做执行前改写（命令内 `skill://` → shell-escaped 真实路径，oh-my-pi `bash-skill-urls.ts` 为参考）；落地后模型回归纯 URI，索引无需变动，read 仍附 sourcePath。
- **trust**：无门控，项目级与用户级同待遇（理由见 Considered Options）；未来门控点两处：索引注入时过滤、read 分派时拒绝。
- **归属**：发现、解析、索引注入均属 `aaos-tools`（系统提示词组装与工具职责）；scheme 解析按 handler-per-scheme 结构留位，不发明通用 VFS；会话侧零改动（read 结果照常进会话段，内容寻址免费得）。

## Considered Options

- **技能存入内容寻址对象存储**（skill 即资产，hash 寻址）：bash 执行需真实路径，须 materialize 且要发明"安装技能"的导入流程；且对象存储在领域模型里装的是会话产物（对话段、摘要、副作用载荷），技能是输入资源不是会话产物，塞入会污染词汇表。否决：纯文件系统存储。内容寻址可作未来分发格式，不进 v1。
- **bash 同期做 URI 改写**：一次性做完难以验收、开发困难。否决：read-only 先行，bash 改写另立 issue，每片独立可验收。
- **索引携带双字段**（URI + 真实 baseDir）：第一天就打穿统一寻址抽象，且 baseDir 常驻每次请求的 system prompt。被取代：read 结果附 sourcePath（oh-my-pi 的 `InternalResource.sourcePath` 同款），索引零泄漏、路径在需要时才披露；模型本就须先 read 技能再谈执行（"MUST read first"），无额外往返。
- **trust 门控**（项目级技能加载前确认）：pi 对 skills 本就不门控（其 trust 只管加载期执行的项目级扩展）；门控对 AI 工作量大、收益低，难以审计，人类又会直接选"完全信任"；skill 注入风险与 AGENTS.md 同构，后者已在 ADR-0003 显式接受。否决：v1 无门控。
- **私有目录 `.aaos/skills`**：`.agents/skills` 是跨工具既有约定，用户级与项目级均有现成技能即日可用。否决。

## Consequences

- CONTEXT.md 新增"技能""内部 URI"两词条。
- bash URI 改写 issue 为纯增量：只加改写，无索引字段要删。
- skill 正文可含脚本执行指令，其注入风险与 AGENTS.md 同级，已显式接受；trust 到来时的两个门控点已在"决定细节"定位。
- 技能 read 结果进会话段后按资产不可变模型固化；技能目录内容在磁盘上的变化只影响之后的 read，与 ADR-0003 对指令文件"磁盘最新态生效"的取舍不同——此处是工具结果快照，属正常。
