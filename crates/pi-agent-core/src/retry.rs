//! Agent turn retry configuration and error classification, plus the
//! abort-watching primitives shared by the loop and by provider adapters.
//!
//! Mirrors Pi's `isRetryableAssistantError`: NON_RETRYABLE patterns checked
//! first (quota/billing), then RETRYABLE patterns (transient provider errors).

use std::sync::LazyLock;

use regex::Regex;
use tokio::sync::watch;

/// Park until the abort flag flips; returns immediately if already set.
///
/// Also returns once the channel closes (sender dropped), treating that as
/// "no more abort signals" — the caller's terminal path runs either way.
pub async fn wait_aborted(abort: &mut watch::Receiver<bool>) {
    if *abort.borrow() {
        return;
    }
    while abort.changed().await.is_ok() {
        if *abort.borrow() {
            return;
        }
    }
}

/// Interruptible sleep. Returns `true` if aborted before the delay elapsed.
pub async fn abortable_sleep(delay_ms: u64, abort: &mut watch::Receiver<bool>) -> bool {
    tokio::select! {
        biased;
        _ = wait_aborted(abort) => true,
        _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => false,
    }
}

/// Agent-level turn retry configuration.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Whether agent turn retry is enabled. Default `true`.
    pub enabled: bool,
    /// Maximum retry attempts. Default `3`.
    pub max_retries: u32,
    /// Base delay in milliseconds for exponential backoff.
    /// Attempt n delay = base_delay_ms * 2^(n-1). Default `2000`.
    pub base_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            base_delay_ms: 2000,
        }
    }
}

/// Patterns matching transient provider errors (Pi's RETRYABLE_PROVIDER_ERROR_PATTERN).
const RETRYABLE_PATTERNS: &[&str] = &[
    "overloaded",
    "rate.?limit",
    "too many requests",
    "429",
    "500",
    "502",
    "503",
    "504",
    "524",
    "service.?unavailable",
    "server.?error",
    "internal.?error",
    "provider.?returned.?error",
    "exceeded request buffer limit while retrying upstream",
    "network.?error",
    "connection.?error",
    "connection.?refused",
    "connection.?lost",
    "other side closed",
    "fetch failed",
    "getaddrinfo",
    "ENOTFOUND",
    "EAI_AGAIN",
    "upstream.?connect",
    "reset before headers",
    "socket hang up",
    "socket connection was closed",
    "timed? out",
    "timeout",
    "terminated",
    "websocket.?closed",
    "websocket.?error",
    "ended without",
    "stream ended before message_stop",
    "stream ended before a terminal response event",
    "http2 request did not get a response",
    "retry delay",
    "you can retry your request",
    "try your request again",
    "please retry your request",
    "ResourceExhausted",
];

/// Patterns matching non-retryable quota/billing errors (Pi's NON_RETRYABLE_PROVIDER_LIMIT_ERROR_PATTERN).
const NON_RETRYABLE_PATTERNS: &[&str] = &[
    "GoUsageLimitError",
    "FreeUsageLimitError",
    "Monthly usage limit reached",
    "available balance",
    "insufficient_quota",
    "out of budget",
    "quota exceeded",
    "billing",
];

static RETRYABLE_RE: LazyLock<Regex> =
    LazyLock::new(
        || match Regex::new(&format!("(?i){}", RETRYABLE_PATTERNS.join("|"))) {
            Ok(re) => re,
            // 模式为字面常量正则，编译期已验证合法，运行时该分支不可达
            #[expect(clippy::unreachable)]
            Err(e) => unreachable!("static retryable patterns must be valid regexes: {e}"),
        },
    );

static NON_RETRYABLE_RE: LazyLock<Regex> =
    LazyLock::new(
        || match Regex::new(&format!("(?i){}", NON_RETRYABLE_PATTERNS.join("|"))) {
            Ok(re) => re,
            // 模式为字面常量正则，编译期已验证合法，运行时该分支不可达
            #[expect(clippy::unreachable)]
            Err(e) => unreachable!("static non-retryable patterns must be valid regexes: {e}"),
        },
    );

/// Whether an error message describes a transient provider error worth retrying.
///
/// Returns `false` for quota/billing errors (checked first), `true` for
/// transient provider errors, `false` for anything else.
pub fn is_retryable_error(msg: &str) -> bool {
    if NON_RETRYABLE_RE.is_match(msg) {
        return false;
    }
    RETRYABLE_RE.is_match(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_patterns_match() {
        assert!(is_retryable_error("overloaded"));
        assert!(is_retryable_error("rate limit exceeded"));
        assert!(is_retryable_error("HTTP 429: too many requests"));
        assert!(is_retryable_error("HTTP 503: service unavailable"));
        assert!(is_retryable_error("connection refused"));
        assert!(is_retryable_error("socket hang up"));
        assert!(is_retryable_error("timeout"));
        assert!(is_retryable_error("ResourceExhausted"));
        assert!(is_retryable_error("stream ended before message_stop"));
    }

    #[test]
    fn non_retryable_patterns_match() {
        assert!(!is_retryable_error("insufficient_quota"));
        assert!(!is_retryable_error("billing required"));
        assert!(!is_retryable_error("quota exceeded"));
        assert!(!is_retryable_error("Monthly usage limit reached"));
    }

    #[test]
    fn unknown_error_not_retryable() {
        assert!(!is_retryable_error("invalid api key"));
        assert!(!is_retryable_error("bad request"));
        assert!(!is_retryable_error(""));
    }

    #[test]
    fn case_insensitive() {
        assert!(is_retryable_error("OVERLOADED"));
        assert!(is_retryable_error("Rate Limit"));
    }
}
