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

#[tokio::test]
async fn models_refresh_lists_flash_from_override_config() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("models.json"),
        r#"{
          "providers": {
            "deepseek": {
              "baseUrl": "https://cchub.example/v1",
              "api": "openai-completions",
              "apiKey": "$CCHUB_API_KEY"
            }
          }
        }"#,
    )
    .unwrap();
    let server = wiremock::MockServer::start().await;
    wiremock::Mock::given(wiremock::matchers::method("GET"))
        .respond_with(wiremock::ResponseTemplate::new(200).set_body_string(registry()))
        .mount(&server)
        .await;
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("AAOS_MODELS_URL", format!("{}/api.json", server.uri()))
        .args(["models", "refresh"])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success(),
        "{stdout} {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(stdout.contains("deepseek/deepseek-v4-flash"));
    assert!(stdout.contains("reasoning=true"));
    assert!(stdout.contains("tool_call=true"));
    assert!(stdout.contains("0.14"));
    assert!(tmp.path().join("catalog-cache.json").exists());
}

#[test]
fn missing_api_key_exits_nonzero() {
    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("models.json"),
        r#"{"providers":{"deepseek":{"baseUrl":"http://127.0.0.1:9","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}"#,
    )
    .unwrap();
    fs::write(
        tmp.path().join("catalog-cache.json"),
        serde_json::json!({
            "fetched_at_unix": 4_000_000_000u64,
            "warning": null,
            "models": [{
                "id": "deepseek-v4-flash",
                "name": "flash",
                "api": "openai-completions",
                "provider": "deepseek",
                "base_url": "http://127.0.0.1:9",
                "reasoning": true,
                "tool_call": true,
                "input": ["text"],
                "cost": {"input": 0.14, "output": 0.28, "cache_read": 0.0, "cache_write": 0.0},
                "context_window": 1000,
                "max_tokens": 100,
                "api_key_env": "CCHUB_API_KEY"
            }]
        })
        .to_string(),
    )
    .unwrap();
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
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
    fs::write(
        tmp.path().join("catalog-cache.json"),
        serde_json::json!({
            "fetched_at_unix": 4_000_000_000u64,
            "warning": null,
            "models": []
        })
        .to_string(),
    )
    .unwrap();
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .args(["--model", "nope", "hello"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("model not found"));
}

#[test]
fn json_prompt_streams_text_and_done() {
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

    let tmp = TempDir::new().unwrap();
    fs::write(
        tmp.path().join("catalog-cache.json"),
        serde_json::json!({
            "fetched_at_unix": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs(),
            "warning": null,
            "models": [{
                "id": "deepseek-v4-flash",
                "name": "flash",
                "api": "openai-completions",
                "provider": "deepseek",
                "base_url": format!("http://{addr}"),
                "reasoning": true,
                "tool_call": true,
                "input": ["text"],
                "cost": {"input": 0.14, "output": 0.28, "cache_read": 0.0, "cache_write": 0.0},
                "context_window": 1000,
                "max_tokens": 100,
                "api_key_env": "CCHUB_API_KEY"
            }]
        })
        .to_string(),
    )
    .unwrap();

    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "test-key")
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

fn write_cache(tmp: &TempDir, base_url: &str) {
    fs::write(
        tmp.path().join("catalog-cache.json"),
        serde_json::json!({
            "fetched_at_unix": 4_000_000_000u64,
            "warning": null,
            "models": [{
                "id": "deepseek-v4-flash",
                "name": "flash",
                "api": "openai-completions",
                "provider": "deepseek",
                "base_url": base_url,
                "reasoning": true,
                "tool_call": true,
                "input": ["text"],
                "cost": {"input": 0.14, "output": 0.28, "cache_read": 0.0, "cache_write": 0.0},
                "context_window": 1000,
                "max_tokens": 100,
                "api_key_env": "CCHUB_API_KEY"
            }]
        })
        .to_string(),
    )
    .unwrap();
}

#[test]
fn invalid_config_exits_nonzero() {
    let tmp = TempDir::new().unwrap();
    fs::write(tmp.path().join("models.json"), "{not json").unwrap();
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("AAOS_MODELS_URL", "http://127.0.0.1:1/missing")
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
fn provider_model_thinking_flags_reach_request() {
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
            let sse = concat!(
                "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n",
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

    let tmp = TempDir::new().unwrap();
    write_cache(&tmp, &format!("http://{addr}"));
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "flag-key")
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
    write_cache(&tmp, "http://127.0.0.1:1");
    let output = bin()
        .env("AAOS_CONFIG_DIR", tmp.path())
        .env("CCHUB_API_KEY", "k")
        .args(["hello"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(!err.trim().is_empty(), "{err}");
}
