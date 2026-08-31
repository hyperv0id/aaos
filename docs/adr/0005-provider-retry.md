# ADR-0005: Provider 重试

## 状态

已接受

## 上下文

provider 调用失败（429/5xx/网络错误）直接导致 agent 运行终止，用户体验差。需要自动重试机制。

## 决策

实现两层重试（参考 Pi）：

1. **Provider HTTP 重试**：透明重试单次 HTTP 请求。重试条件：HTTP 408/409/429/≥500 或无状态码（传输错误）。退避策略：jittered exponential `min(0.5×2^idx, 8)s × (1−rand×0.25)`，遵循 `retry-after-ms`/`retry-after` 头。默认关闭（max_retries=0），匹配 Pi 的 `provider.maxRetries: 0`。

2. **Agent turn 重试**：turn 失败后弹出错误 assistant message，指数退避 `base_delay_ms × 2^(n-1)`，重新调用。默认开启（enabled=true, max_retries=3, base_delay_ms=2000）。配额/计费错误不可重试。

### 错误分类

使用正则匹配错误消息（与 Pi 的 `isRetryableAssistantError` 一致）：
- 先检查不可重试模式（配额/计费）
- 再检查可重试模式（transient provider errors）

### 排除

- Context overflow：由 compaction 处理（AAOS 尚无 compaction）
- Provider 轮换/fallback：不在本次范围

## 后果

- `pi-agent-core` 新增 `retry` 模块（`RetryConfig`, `is_retryable_error`）
- `aaos-providers` 新增 `retry` 模块（`ProviderRetryConfig`, `retry_provider_call`, `RetryingStreamFn`）
- `AgentLoopConfig` 新增 `retry: RetryConfig` 字段
- `StreamFnOptions` 新增 `provider_retry_max_retries` 和 `provider_retry_max_delay_ms` 字段
- CLI 默认使用 `stream_fn_for_with_retry`
