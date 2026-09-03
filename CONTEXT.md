# aaos

以会话为中心的 agent 系统。本词汇表覆盖会话存储域（`aaos-session`）与 provider 域（`aaos-providers`）。

## Language

**资产 (Object)**:
裸内容字节：离开 DB 即自足可读的内容（判别式）——按内容哈希寻址、写一次、全局一份、不可变；结构与元信息（块清单、mime_type、tool_call id/name、时间戳等）属 DB，不是资产。内容单位是对话内容块。
_Avoid_: blob、文件（作内容义）

**会话 (Session)**:
派生结构中的一个节点，由唯一 id 标识；根无父节点，其余由（父会话， 位置）派生而来。其内容 = 从根沿派生链到该节点重放的全部资产。
_Avoid_: 分支、日志（作会话结构义）

**派生 (Derivation)**:
从（父会话， 位置）建立新会话的操作，一切结构变更的唯一形式；有两种形状：分叉与压缩。
_Avoid_: 复制、fork（泛指派生时）

**分叉 (Fork)**:
不替换父内容的派生：子会话视图 = 父视图截到指定位置的前缀，此后只追加。
_Avoid_: branch、子分支

**压缩 (Compaction)**:
首批记录为区间替换映射的派生；被替换的原文仍然可寻址。
_Avoid_: overlay、覆盖、summarization（摘要如何生成是另一回事）

**书签 (Snapshot)**:
指向（会话， 位置）的纯标记；永不自动恢复，仅作派生的锚点。
_Avoid_: checkpoint、检查点、HEAD

**头指针 (Head)**:
meta 表中唯一的可变行，存放最后被追加的会话 id；相当于 git 的 HEAD 引用。追加在事务中原子前移它；默认恢复（无 --session）从它派生新会话，追加只写入派生出的会话。
_Avoid_: 当前会话（那是进程内绑定）、latest

**视图 (View)**:
沿派生链重放并应用压缩映射后得到的会话当前内容。
_Avoid_: replay、materialize（作名词）

**副作用 (Side Effect)**:
工具执行留下的 before/after 载荷记录，属于执行它的会话并沿链继承。
_Avoid_: WAL（实现词）

### Provider

**API格式 (API format)**:
一种 HTTP 请求/响应的形状，挂在**模型**上，不挂在提供商上。同一提供商的不同模型可以是不同 API格式。
_Avoid_: 方言、dialect、接口（作线形义）

**提供商 (Provider)**:
对外提供模型服务的一方。覆盖名单按提供商列。共用一种 API格式时，格式细节仍按提供商（或该提供商下的模型）记录。
_Avoid_: vendor、厂商（作目录身份时）

**目录 (Catalog)**:
aaos 能解析并选用的模型表。
_Avoid_: 注册表、registry

**重试 (Retry)**:
分两层：provider HTTP 重试（`aaos-providers`，透明重试单次请求，默认关闭）与 agent turn 重试（`pi-agent-core` agent_loop，turn 失败后退避重跑，默认开启）。
_Avoid_: 回退、降级（作重试义）

### Agent

**技能 (Skill)**:
以 SKILL.md 为入口文件的目录；frontmatter 给出 name（缺省为目录名）与 description。发现于用户级 `~/.agents/skills/` 与 project_root 的 `.agents/skills/`（单层，同名时项目级压用户级）；system prompt 只注入索引（name、description、内部 URI），全文由模型按需 read，read 结果附解析出的真实路径。
_Avoid_: 插件、扩展（作技能义）

**内部 URI (Internal URI)**:
read 的 path 参数接受的 `scheme://` 寻址形式；现仅 `skill://<name>[/path]`：name 直指该技能的 SKILL.md，/path 访问技能目录内文件（指向目录返回列举）。path 无 scheme 时一律按文件系统路径，行为不变。
_Avoid_: 链接、协议（作寻址义）
