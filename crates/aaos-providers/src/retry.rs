//! Provider HTTP retry: transparent retry of single HTTP requests for
//! transient failures (429, 5xx, network errors) with jittered exponential
//! backoff and abortable sleep.
//!
//! Mirrors Pi's `retryProviderRequest`: the request factory is re-invoked
//! on each retry; `retry-after-ms` / `retry-after` response headers are
//! honored; server-requested delays above `max_retry_delay_ms` fail
//! immediately. Default `max_retries = 0` (disabled).

use std::sync::Arc;

use async_trait::async_trait;
use pi_agent_core::retry::abortable_sleep;
use pi_agent_core::types::{AssistantEventStream, LlmContext, Model, StreamFn, StreamFnOptions};
use tokio::sync::watch;

/// Provider HTTP retry configuration.
#[derive(Debug, Clone)]
pub struct ProviderRetryConfig {
    /// Maximum retry attempts. Default `0` (no retry).
    pub max_retries: u32,
    /// Maximum server-requested delay (ms) before failing. Default `60000`.
    /// Set to `0` to disable the limit.
    pub max_retry_delay_ms: u64,
}

impl Default for ProviderRetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 0,
            max_retry_delay_ms: 60_000,
        }
    }
}

/// Extract HTTP status code from an error string.
///
/// Matches the format produced by sse.rs `run_stream`: `"HTTP {status}: {text}"`.
fn parse_http_status(error_msg: &str) -> Option<u16> {
    error_msg
        .strip_prefix("HTTP ")?
        .split(':')
        .next()?
        .parse()
        .ok()
}

/// Whether the error string represents a retryable provider error.
///
/// Retryable: 408, 409, 429, >=500, or no status (transport error).
fn is_retryable_provider_error(error_msg: &str) -> bool {
    match parse_http_status(error_msg) {
        Some(408 | 409 | 429) => true,
        Some(s) if s >= 500 => true,
        Some(_) => false,
        None => true,
    }
}

/// Parse `retry-after-ms` or `retry-after` value from error text.
///
/// sse.rs errors include the response body text after the status code.
/// We look for `retry-after-ms: <ms>` or `retry-after: <seconds>` patterns.
fn parse_retry_after_ms(error_msg: &str) -> Option<u64> {
    // Try retry-after-ms first
    if let Some(pos) = error_msg.find("retry-after-ms:") {
        let rest = error_msg[pos + 15..].trim_start();
        if let Some(end) = rest.find(|c: char| !c.is_ascii_digit()) {
            if let Ok(ms) = rest[..end].parse::<u64>() {
                return Some(ms);
            }
        } else if let Ok(ms) = rest.parse::<u64>() {
            return Some(ms);
        }
    }
    // Try retry-after (seconds)
    if let Some(pos) = error_msg.find("retry-after:") {
        let rest = error_msg[pos + 12..].trim_start();
        if let Some(end) = rest.find(|c: char| !c.is_ascii_digit() && c != '.') {
            if let Ok(secs) = rest[..end].parse::<f64>() {
                return Some((secs * 1000.0) as u64);
            }
        } else if let Ok(secs) = rest.parse::<f64>() {
            return Some((secs * 1000.0) as u64);
        }
    }
    None
}

/// Jittered exponential backoff delay in milliseconds.
///
/// Formula: min(0.5 * 2^idx, 8) * 1000 * (1 - rand * 0.25)
/// Matches Pi's getRetryDelayMs fallback.
fn backoff_delay_ms(retry_index: u32) -> u64 {
    let base = (0.5f64 * 2f64.powi(retry_index as i32)).min(8.0) * 1000.0;
    let jitter = 1.0 - simple_random() * 0.25;
    (base * jitter) as u64
}

/// Simple pseudo-random value in [0, 1) without a rand crate.
/// Uses system time nanoseconds for jitter — sufficient for retry delays.
fn simple_random() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    (nanos % 1000) as f64 / 1000.0
}

/// Retry a provider call with bounded retries and jittered backoff.
///
/// `make_request` is a factory closure invoked on each attempt.
/// Returns the last error when retries are exhausted or the error is not retryable.
pub async fn retry_provider_call<T, F, Fut>(
    config: &ProviderRetryConfig,
    mut abort: watch::Receiver<bool>,
    mut make_request: F,
) -> Result<T, String>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, String>>,
{
    let mut retries_remaining = config.max_retries;

    loop {
        match make_request().await {
            Ok(v) => return Ok(v),
            Err(e) => {
                if *abort.borrow() {
                    return Err("Request aborted".into());
                }
                if retries_remaining == 0 || !is_retryable_provider_error(&e) {
                    return Err(e);
                }
                retries_remaining -= 1;
                let retry_index = config.max_retries - retries_remaining - 1;

                // Honor server-requested delay, capped by max_retry_delay_ms.
                let delay_ms = match parse_retry_after_ms(&e) {
                    Some(server_ms) => {
                        if config.max_retry_delay_ms > 0 && server_ms > config.max_retry_delay_ms {
                            return Err(format!(
                                "Server requested {}s retry delay (max: {}s). {}",
                                server_ms / 1000,
                                config.max_retry_delay_ms / 1000,
                                e
                            ));
                        }
                        server_ms
                    }
                    None => backoff_delay_ms(retry_index),
                };

                if abortable_sleep(delay_ms, &mut abort).await {
                    return Err("Request aborted".into());
                }
            }
        }
    }
}

/// A [`StreamFn`] wrapper that retries the inner stream function on
/// transient provider errors.
pub struct RetryingStreamFn {
    inner: Arc<dyn StreamFn>,
    retry_config: ProviderRetryConfig,
}

impl RetryingStreamFn {
    pub fn new(inner: Arc<dyn StreamFn>, retry_config: ProviderRetryConfig) -> Self {
        Self {
            inner,
            retry_config,
        }
    }
}

#[async_trait]
impl StreamFn for RetryingStreamFn {
    async fn call(
        &self,
        model: Model,
        context: LlmContext,
        options: StreamFnOptions,
        abort: watch::Receiver<bool>,
    ) -> Result<Box<dyn AssistantEventStream>, String> {
        let inner = self.inner.clone();
        let config = self.retry_config.clone();
        retry_provider_call(&config, abort.clone(), || {
            inner.call(
                model.clone(),
                context.clone(),
                options.clone(),
                abort.clone(),
            )
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_http_status_works() {
        assert_eq!(parse_http_status("HTTP 429: rate limited"), Some(429));
        assert_eq!(parse_http_status("HTTP 503: unavailable"), Some(503));
        assert_eq!(parse_http_status("HTTP 200: ok"), Some(200));
        assert_eq!(parse_http_status("network error"), None);
        assert_eq!(parse_http_status("HTTP abc: bad"), None);
    }

    #[test]
    fn retryable_classification() {
        assert!(is_retryable_provider_error("HTTP 429: rate limited"));
        assert!(is_retryable_provider_error("HTTP 503: unavailable"));
        assert!(is_retryable_provider_error("HTTP 500: server error"));
        assert!(is_retryable_provider_error("HTTP 408: timeout"));
        assert!(is_retryable_provider_error("HTTP 409: conflict"));
        assert!(!is_retryable_provider_error("HTTP 400: bad request"));
        assert!(!is_retryable_provider_error("HTTP 401: unauthorized"));
        assert!(is_retryable_provider_error("connection refused"));
    }

    #[test]
    fn backoff_delay_bounded() {
        let d0 = backoff_delay_ms(0);
        let d1 = backoff_delay_ms(1);
        let d2 = backoff_delay_ms(2);
        let d3 = backoff_delay_ms(3);
        // All within [base*0.75, base]
        assert!((375..=500).contains(&d0));
        assert!((750..=1000).contains(&d1));
        assert!((1500..=2000).contains(&d2));
        assert!((3000..=4000).contains(&d3));
        // Capped at 8000
        let d10 = backoff_delay_ms(10);
        assert!(d10 <= 8000);
    }

    #[tokio::test]
    async fn retry_succeeds_after_transient_failure() {
        let config = ProviderRetryConfig {
            max_retries: 2,
            max_retry_delay_ms: 60000,
        };
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = call_count.clone();
        let (_abort_tx, abort_rx) = watch::channel(false);

        let result = retry_provider_call(&config, abort_rx, move || {
            let c = count.clone();
            async move {
                let n = c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if n < 2 {
                    Err("HTTP 503: temporarily unavailable".to_string())
                } else {
                    Ok("success")
                }
            }
        })
        .await;

        assert_eq!(result, Ok("success"));
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_gives_up_on_non_retryable() {
        let config = ProviderRetryConfig {
            max_retries: 3,
            max_retry_delay_ms: 60000,
        };
        let (_abort_tx, abort_rx) = watch::channel(false);

        let result = retry_provider_call(&config, abort_rx, || async {
            Err::<String, _>("HTTP 400: bad request".to_string())
        })
        .await;

        assert_eq!(result, Err("HTTP 400: bad request".to_string()));
    }

    #[tokio::test]
    async fn retry_respects_max_retries() {
        let config = ProviderRetryConfig {
            max_retries: 2,
            max_retry_delay_ms: 60000,
        };
        let call_count = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));
        let count = call_count.clone();
        let (_abort_tx, abort_rx) = watch::channel(false);

        let result = retry_provider_call(&config, abort_rx, move || {
            let c = count.clone();
            async move {
                c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Err::<String, _>("HTTP 503: unavailable".to_string())
            }
        })
        .await;

        assert!(result.is_err());
        assert_eq!(call_count.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_aborts_during_sleep() {
        let config = ProviderRetryConfig {
            max_retries: 5,
            max_retry_delay_ms: 60000,
        };
        let (abort_tx, abort_rx) = watch::channel(false);

        // Abort after a short delay
        let tx = abort_tx.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            let _ = tx.send(true);
        });

        let result = retry_provider_call(&config, abort_rx, || async {
            Err::<String, _>("HTTP 503: unavailable".to_string())
        })
        .await;

        assert_eq!(result, Err("Request aborted".to_string()));
    }

    #[tokio::test]
    async fn no_retry_when_max_retries_zero() {
        let config = ProviderRetryConfig::default();
        let (_abort_tx, abort_rx) = watch::channel(false);

        let result = retry_provider_call(&config, abort_rx, || async {
            Err::<String, _>("HTTP 503: unavailable".to_string())
        })
        .await;

        assert_eq!(result, Err("HTTP 503: unavailable".to_string()));
    }
}
