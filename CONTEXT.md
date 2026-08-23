# aaos

以会话为中心的 agent 系统。本词汇表覆盖会话存储域（`aaos-session-store`）与 provider 域（`aaos-providers`）。

## Language

**会话 (Session)**:
一条对话血统的名字；其内容 = 从根会话沿派生链到它的全部资产。
_Avoid_: 分支、日志（作会话结构义）

**资产 (Object)**:
按内容哈希寻址、写一次、全局一份的不可变内容（对话段、摘要、副作用载荷）。
_Avoid_: blob、文件（作内容义）

**派生 (Derivation)**:
从（父会话， 位置）建立新会话的操作；一切结构变更的唯一形式。
_Avoid_: 复制、fork（泛指派生时）

**分叉 (Fork)**:
只携带追加记录的派生；子会话与父会话共享前缀。
_Avoid_: branch、子分支

**压缩 (Compaction)**:
首批记录为区间替换映射的派生；被替换的原文仍然可寻址。
_Avoid_: overlay、覆盖、summarization（摘要如何生成是另一回事）

**书签 (Snapshot)**:
指向（会话， 位置）的纯标记；永不自动恢复，仅作派生的锚点。
_Avoid_: checkpoint、检查点、HEAD

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
