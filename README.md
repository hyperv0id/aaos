# aaos

面向 LLM agent harness 的架构设计仓库。用操作系统设计（GC/swap、COW fork、信号、WAL、进程组）重新审视 vibe-coding 工具链中上下文管理、多 agent 协作与失败恢复的问题，并收敛 agent 资源接口与会话存储的设计。

## 文档

| 文档 | 内容 |
|---|---|
| [DESIGN.md](DESIGN.md) | OS 类比下的 harness 架构重思：compaction、COW fork、checkpoint/WAL 与信号、多 agent 协作模式 |
| [RESOURCE-PROTOCOL-DESIGN.md](RESOURCE-PROTOCOL-DESIGN.md) | agent 资源接口：opaque ResourceRef、统一 `read(ref)`、descriptor/children/parts、错误语义与权限边界 |
| [SESSION-STORAGE.md](SESSION-STORAGE.md) | 会话存储：内容寻址对象库 + 每分支追加日志；压缩可逆、分叉零复制 |
| [.scratch/resource-protocol/](.scratch/resource-protocol/) | 资源协议的设计决策记录（issues 01–07）与收敛地图 |

## 状态

设计文档阶段，无代码。资源协议 v1 的范围与「不做」清单见 RESOURCE-PROTOCOL-DESIGN.md 第 9 节。
