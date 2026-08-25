#![allow(clippy::unwrap_used, clippy::expect_used)]
use std::fs;
use std::process::Command;
use tempfile::TempDir;

fn bin() -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_aaos"));
    cmd.env_remove("CCHUB_API_KEY");
    cmd.env_remove("DEEPSEEK_API_KEY");
    cmd
}

fn registry() -> String {
    serde_json::json!({
        "deepseek": {
            "id": "deepseek",
            "env": ["DEEPSEEK_API_KEY"],
            "npm": "@ai-sdk/openai-compatible",
            "api": "https://api.deepseek.com",
            "models": {
                "deepseek-v4-flash": {
                    "id": "deepseek-v4-flash",
                    "name": "DeepSeek V4 Flash",
                    "reasoning": true,
                    "tool_call": true,
                    "limit": { "context": 1000000, "output": 384000 },
                    "cost": { "input": 0.14, "output": 0.28 }
                }
            }
        }
    })
    .to_string()
}

/// Write `models.json` redirecting deepseek to `base_url` via CCHUB_API_KEY.
fn write_config(tmp: &TempDir, base_url: &str) {
    fs::write(
        tmp.path().join("models.json"),
        format!(
            r#"{{"providers":{{"deepseek":{{"baseUrl":"{base_url}","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}}}}"#
        ),
    )
    .unwrap();
}

/// Start a wiremock server serving `registry()` at `/api.json`.
async fn mock_registry() -> wiremock::MockServer {
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .and(wiremock::matchers::path("/api.json"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(registry()))
        .mount(&server)
        .await;
    server
}

/// Spawn a mock SSE server that responds with a fixed two-delta "Hi!" stream.
/// Returns the bound address the CLI should point at.
fn mock_sse_server() -> std::net::SocketAddr {
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = read_http_request(&mut sock);
            let sse = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n\n",
                "data: {\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\n",
                "data: [DONE]\n\n"
            );
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                sse.len()
            );
            let _ = sock.write_all(header.as_bytes());
            let _ = sock.write_all(sse.as_bytes());
        }
    });
    addr
}

/// single-delta "ok" stream, capturing each request body (JSON payload) into a
/// shared vector in arrival order. Returns the bound address and the capture.
fn mock_sse_server_capturing(
    n: usize,
) -> (
    std::net::SocketAddr,
    std::sync::Arc<std::sync::Mutex<Vec<String>>>,
) {
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::Arc;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured_thread = captured.clone();
    thread::spawn(move || {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            sse.len()
        );
        for _ in 0..n {
            let Ok((mut sock, _)) = listener.accept() else {
                break;
            };
            let request = read_http_request(&mut sock);
            if let Some((_, body)) = request.split_once("\r\n\r\n") {
                captured_thread.lock().unwrap().push(body.to_string());
            }
            let _ = sock.write_all(header.as_bytes());
            let _ = sock.write_all(sse.as_bytes());
        }
    });
    (addr, captured)
}

/// single-delta "ok" stream. Returns the bound address the CLI should point at.
fn mock_sse_server_n(n: usize) -> std::net::SocketAddr {
    use std::io::Write;
    use std::net::TcpListener;
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    thread::spawn(move || {
        let sse = concat!(
            "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n"
        );
        let header = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            sse.len()
        );
        for _ in 0..n {
            let Ok((mut sock, _)) = listener.accept() else {
                break;
            };
            let _ = read_http_request(&mut sock);
            let _ = sock.write_all(header.as_bytes());
            let _ = sock.write_all(sse.as_bytes());
        }
    });
    addr
}

#[tokio::test]
async fn json_prompt_streams_text_and_done() {
    let addr = mock_sse_server();
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry().await;

    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .args(["--json", "hello"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout} {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("\"type\":\"message_end\""), "{stdout}");
    assert!(stdout.contains("\"stop_reason\":\"stop\""), "{stdout}");
    assert!(stdout.contains("\"content\":\"Hi!\""), "{stdout}");
    assert!(stdout.contains("\"type\":\"done\""), "{stdout}");
    assert!(
        !stdout.contains("\"type\":\"text_delta\""),
        "token-level deltas must not appear in json mode: {stdout}"
    );
}

#[test]
fn missing_api_key_exits_nonzero() {
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, "http://127.0.0.1:9");
    let server = mock_registry_url();
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("AAOS_MODELS_URL", format!("{}/api.json", server))
        .args(["hello"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("CCHUB_API_KEY"), "{err}");
}

#[test]
fn missing_model_exits_nonzero() {
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, "http://127.0.0.1:9");
    let server = mock_registry_url();
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "k")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server))
        .args(["--model", "nope", "hello"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("model not found"));
}

#[test]
fn json_flag_after_prompt() {
    let addr = mock_sse_server();
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry_url();

    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server))
        .args(["hello", "--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout} {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        stdout.contains("\"type\":\"message_end\""),
        "--json after prompt must enable JSON mode: {stdout}"
    );
    assert!(
        stdout.contains("\"type\":\"done\""),
        "--json after prompt must emit done event: {stdout}"
    );
}

#[tokio::test]
async fn repl_eof_exits_clean() {
    let addr = mock_sse_server();
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry().await;

    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    // EOF exit prints the session resume hint to stderr.
    assert!(stderr.contains("Session saved. Resume with:"), "{stderr}");
    assert!(stderr.contains("aaos --session "), "{stderr}");
}

#[tokio::test]
async fn repl_keeps_history() {
    use std::io::Write;
    use std::process::Stdio;

    let (addr, captured) = mock_sse_server_capturing(2);
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry().await;

    let mut child = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"first\nsecond\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout} {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // Two REPL turns ⇒ two LLM requests, each echoed once.
    assert_eq!(stdout.matches("ok").count(), 2, "{stdout}");

    // Context accumulation: the second request must carry the first turn's
    // assistant reply in its message history.
    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 2, "one LLM request per REPL turn");
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let messages = second["messages"].as_array().unwrap();
    let assistant = messages
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap_or_else(|| panic!("second request carries no assistant message: {second}"));
    assert!(
        assistant["content"].as_str().unwrap().contains("ok"),
        "second request must contain the first turn's assistant reply: {second}"
    );
}

#[tokio::test]
async fn repl_json_emits_events() {
    use std::io::Write;
    use std::process::Stdio;

    let addr = mock_sse_server_n(1);
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry().await;

    let mut child = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .args(["--json"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"hello\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout} {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("\"type\":\"message_end\""), "{stdout}");
    assert!(stdout.contains("\"type\":\"done\""), "{stdout}");
}

/// In --json mode with no input (immediate EOF): the resume hint still goes
/// to stderr, but stdout must stay pure JSON (no hint there).
#[tokio::test]
async fn repl_json_eof_hint_on_stderr_only() {
    let addr = mock_sse_server();
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry().await;

    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .args(["--json"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout} {stderr}");
    // The hint lives on stderr even in --json mode.
    assert!(stderr.contains("aaos --session "), "{stderr}");
    // Stdout must not contain the human resume hint.
    assert!(!stdout.contains("Session saved"), "{stdout}");
    assert!(!stdout.contains("aaos --session"), "{stdout}");
}

/// Ctrl+C is deliberately unbound: SIGINT must be swallowed (the REPL
/// survives it, idle or mid-run) and the REPL must keep serving turns
/// afterward. Sends a real SIGINT via `kill -INT` once startup has settled,
/// then drives one turn and exits via EOF.
#[tokio::test]
async fn sigint_is_swallowed_and_repl_keeps_working() {
    use std::io::Write;
    use std::process::Stdio;
    use std::thread::sleep;
    use std::time::Duration;

    let addr = mock_sse_server();
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry().await;

    let mut child = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    // Let startup (catalog load + store open + listener registration) settle.
    sleep(Duration::from_millis(1500));
    assert!(
        child.try_wait().unwrap().is_none(),
        "REPL exited before SIGINT"
    );

    // Simulated Ctrl+C: the process must survive it.
    let pid = child.id().to_string();
    let kill_status = Command::new("kill").args(["-INT", &pid]).status().unwrap();
    assert!(kill_status.success(), "kill -INT failed");
    sleep(Duration::from_millis(500));
    assert!(
        child.try_wait().unwrap().is_none(),
        "SIGINT killed the REPL; expected it to be swallowed"
    );

    // The swallowed signal must not wedge the loop: one more turn runs,
    // then EOF exits cleanly with the resume hint.
    child.stdin.take().unwrap().write_all(b"hello\n").unwrap();
    let output = child.wait_with_output().unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stdout} {stderr}");
    assert!(stdout.contains("Hi!"), "{stdout}");
    assert!(stderr.contains("aaos --session "), "{stderr}");
}

#[tokio::test]
async fn persists_segments() {
    let (addr, _captured) = mock_sse_server_capturing(1);
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    // Pre-create a root session so the single shot resumes a known node.
    let store = aaos_session::SessionStore::open(tmp.path()).await.unwrap();
    let root = store.create_root().await.unwrap();
    drop(store);
    let server = mock_registry().await;

    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .args(["hello"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let store = aaos_session::SessionStore::open(tmp.path()).await.unwrap();
    let segments = store.materialize_plain(&root).await.unwrap();
    assert_eq!(segments.len(), 2, "user + assistant must persist");
    assert_eq!(segments[0].kind(), "user");
    assert_eq!(user_text(&segments[0]), "hello");
    assert_eq!(segments[1].kind(), "assistant");
}

#[tokio::test]
async fn carries_context() {
    let (addr, captured) = mock_sse_server_capturing(2);
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry().await;

    for _ in 0..2 {
        let output = bin()
            .env("AAOS_CONFIG_DIR", tmp.path())
            .env("CCHUB_API_KEY", "test-key")
            .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
            .args(["hello"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let bodies = captured.lock().unwrap();
    assert_eq!(bodies.len(), 2, "one LLM request per single-shot run");
    let second: serde_json::Value = serde_json::from_str(&bodies[1]).unwrap();
    let messages = second["messages"].as_array().unwrap();
    let assistant = messages
        .iter()
        .find(|m| m["role"] == "assistant")
        .unwrap_or_else(|| panic!("second request carries no assistant message: {second}"));
    assert!(
        assistant["content"].as_str().unwrap().contains("ok"),
        "second request must carry the first run's assistant reply: {second}"
    );
}

#[tokio::test]
async fn json_persists_segments() {
    let addr = mock_sse_server();
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry().await;

    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .args(["--json", "hello"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout} {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let store = aaos_session::SessionStore::open(tmp.path()).await.unwrap();
    let latest = store
        .latest_session()
        .await
        .unwrap()
        .expect("single shot must have created/updated a session");
    let segments = store.materialize_plain(&latest).await.unwrap();
    assert_eq!(
        segments.len(),
        2,
        "user + assistant must persist in json mode"
    );
    assert_eq!(segments[0].kind(), "user");
    assert_eq!(segments[1].kind(), "assistant");
}

#[tokio::test]
async fn empty_store_creates_root() {
    let addr = mock_sse_server();
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry().await;

    assert!(
        !tmp.path().join("store.db").exists(),
        "precondition: empty store"
    );
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .args(["hello"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let store = aaos_session::SessionStore::open(tmp.path()).await.unwrap();
    let latest = store
        .latest_session()
        .await
        .unwrap()
        .expect("create_root fallback must have created a session");
    let segments = store.materialize_plain(&latest).await.unwrap();
    assert_eq!(segments.len(), 2, "user + assistant must persist");
    assert_eq!(user_text(&segments[0]), "hello");
}

/// Extract the first text block of a user segment.
fn user_text(segment: &aaos_session::Segment) -> &str {
    match segment {
        aaos_session::Segment::User(user) => match &user.content[0] {
            aaos_session::ContentBlock::Text { text } => text,
            other => panic!("expected text block, got {other:?}"),
        },
        other => panic!("expected user segment, got {other:?}"),
    }
}

fn read_http_request(sock: &mut impl std::io::Read) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 4096];
    loop {
        let n = match std::io::Read::read(sock, &mut tmp) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        buf.extend_from_slice(&tmp[..n]);
        let Some(header_end) = buf.windows(4).position(|w| w == b"\r\n\r\n") else {
            continue;
        };
        let headers = String::from_utf8_lossy(&buf[..header_end]);
        let content_length = headers.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            if name.eq_ignore_ascii_case("content-length") {
                value.trim().parse::<usize>().ok()
            } else {
                None
            }
        });
        match content_length {
            Some(len) if buf.len() >= header_end + 4 + len => {
                buf.truncate(header_end + 4 + len);
                break;
            }
            Some(_) => continue,
            None => break,
        }
    }
    String::from_utf8_lossy(&buf).into_owned()
}

#[test]
fn invalid_config_exits_nonzero() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("models.json"), "{not json").unwrap();
    let server = mock_registry_url();
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("AAOS_MODELS_URL", format!("{}/api.json", server))
        .args(["hello"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(
        err.contains("invalid config") || err.contains("not json") || err.contains("expected"),
        "{err}"
    );
}

#[test]
fn thinking_flags_reach_request() {
    use std::io::Write;
    use std::net::TcpListener;
    use std::sync::{Arc, Mutex};
    use std::thread;

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let captured = Arc::new(Mutex::new(String::new()));
    let cap = captured.clone();
    thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            *cap.lock().unwrap() = read_http_request(&mut sock);
            let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n";
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                sse.len()
            );
            let _ = sock.write_all(header.as_bytes());
            let _ = sock.write_all(sse.as_bytes());
        }
    });

    let tmp = TempDir::new().unwrap();
    write_config(&tmp, &format!("http://{addr}"));
    let server = mock_registry_url();
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "flag-key")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server))
        .args([
            "--provider",
            "deepseek",
            "--model",
            "deepseek-v4-flash",
            "--thinking",
            "high",
            "hello",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("ok"), "{stdout}");
    let raw = captured.lock().unwrap().clone();
    // Directory/adapter seam: the request line must hit the appended tail
    // segment, never a raw base URL.
    assert_eq!(
        raw.lines().next().unwrap_or(""),
        "POST /chat/completions HTTP/1.1",
        "{raw}"
    );
    assert!(
        raw.to_ascii_lowercase().contains("bearer flag-key"),
        "{raw}"
    );
    let body = raw.split("\r\n\r\n").nth(1).unwrap_or("");
    let json: serde_json::Value = serde_json::from_str(body).unwrap();
    assert_eq!(json["model"], "deepseek-v4-flash");
    assert_eq!(json["reasoning_effort"], "high");
    let names: Vec<&str> = json["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["function"]["name"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["read", "bash", "edit", "write"]);
    let read = &json["tools"][0]["function"]["parameters"];
    assert_eq!(read["required"], serde_json::json!(["path"]));
    let sys = json["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| m["role"] == "system")
        .unwrap();
    assert!(
        sys["content"]
            .as_str()
            .unwrap()
            .contains("Available tools:")
    );
}

#[test]
fn network_error_exits_nonzero() {
    let tmp = TempDir::new().unwrap();
    write_config(&tmp, "http://127.0.0.1:1");
    let server = mock_registry_url();
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "k")
        .env("AAOS_MODELS_URL", format!("{}/api.json", server))
        .args(["hello"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(!err.trim().is_empty(), "{err}");
}

/// Synchronous one-shot HTTP server serving `registry()` at `/api.json`.
/// For `#[test]` functions that cannot use the async wiremock helper.
fn mock_registry_url() -> String {
    use std::io::Write;
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let body = registry();
    std::thread::spawn(move || {
        if let Ok((mut sock, _)) = listener.accept() {
            let _ = read_http_request(&mut sock);
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            let _ = sock.write_all(header.as_bytes());
            let _ = sock.write_all(body.as_bytes());
        }
    });
    format!("http://{addr}")
}
