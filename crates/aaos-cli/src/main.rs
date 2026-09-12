use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::Arc;

use aaos_providers::{DEFAULT_MODEL_LIST_URL, Paths};
use aaos_runtime::compaction::CompactionSettings;
use aaos_runtime::event::{EventSink, SessionEvent};
use aaos_runtime::model::{AgentConfig, EnvConfig};
use aaos_runtime::session::{SessionConfig, SessionHandle, SessionNodeConfig, create_session};
use clap::Parser;
use pi_agent_core::types::{
    AgentEvent, AgentToolResult, AssistantMessage, AssistantMessageEvent, ContentBlock, StopReason,
};
use serde_json::{Value, json};

/// DeepSeek product defaults; the provider crate stays product-agnostic.
const DEFAULT_PROVIDER: &str = "deepseek";
const DEFAULT_MODEL_ID: &str = "deepseek-v4-flash";
const DEFAULT_THINKING: &str = "high";

#[derive(Parser, Debug, Default)]
#[command(name = "aaos", about = "Minimal aaos CLI for CCHUB/DeepSeek prompts")]
struct Cli {
    #[arg(long)]
    provider: Option<String>,
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    thinking: Option<String>,
    #[arg(long)]
    json: bool,
    /// Session node id to resume in place; without it a fresh session is
    /// derived from the head pointer.
    #[arg(long = "session")]
    session_id: Option<String>,
    /// With `--session`, resume a fork of that node instead of the node
    /// itself (the default path already derives a fresh session).
    #[arg(long)]
    fork: bool,
    /// Prompt text.
    prompt: Vec<String>,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => code,
        Err(err) => {
            let _ = writeln!(io::stderr(), "{err}");
            ExitCode::from(1)
        }
    }
}

async fn run() -> Result<ExitCode, String> {
    let cli = Cli::parse();
    let paths = paths_from_env();
    swallow_sigint();
    if cli.prompt.is_empty() {
        run_repl(&cli, &paths).await
    } else {
        run_prompt(cli, paths).await
    }
}

/// Swallow SIGINT (Ctrl+C): deliberately unbound — it neither aborts an
/// active run nor exits the process. The REPL ends on EOF (Ctrl+D) or a
/// stdin read error; the one-shot path exits when its prompt completes.
/// Once the spawned listener is first polled it takes over SIGINT from the
/// OS default disposition, which would otherwise terminate the process on
/// every signal.
fn swallow_sigint() {
    tokio::spawn(async {
        loop {
            let _ = tokio::signal::ctrl_c().await;
        }
    });
}

fn paths_from_env() -> Paths {
    if let Ok(dir) = std::env::var("AAOS_CONFIG_DIR") {
        Paths::from_config_dir(dir)
    } else {
        Paths::default_user()
    }
}

fn model_list_url_override() -> String {
    std::env::var("AAOS_MODELS_URL").unwrap_or_else(|_| DEFAULT_MODEL_LIST_URL.to_string())
}

/// Build the session config from CLI args and the host environment: product
/// defaults (DeepSeek provider/model, High thinking) are filled in here —
/// they stay in the CLI, not the runtime. cwd and the skills directory are
/// resolved once, explicitly, for the runtime.
fn build_session_config(cli: &Cli, paths: &Paths) -> Result<SessionConfig, String> {
    Ok(SessionConfig {
        session_node: SessionNodeConfig {
            session_id: cli.session_id.clone(),
            fork: cli.fork,
        },
        agent: AgentConfig {
            provider: cli.provider.clone().or(Some(DEFAULT_PROVIDER.to_string())),
            model: cli.model.clone().or(Some(DEFAULT_MODEL_ID.to_string())),
            thinking: cli.thinking.clone().or(Some(DEFAULT_THINKING.to_string())),
        },
        env: EnvConfig {
            paths: paths.clone(),
            model_list_url: model_list_url_override(),
            api_key_resolver: Arc::new(|k| std::env::var(k).ok()),
        },
        compaction: compaction_settings_from_env(),
        cwd: std::env::current_dir().map_err(|e| e.to_string())?,
        user_skills_dir: std::env::home_dir()
            .map(|h| h.join(".agents/skills"))
            .unwrap_or_default(),
    })
}

/// Read the compaction settings from the environment (the env reads stay in
/// the CLI; the pure parse lives in `aaos_runtime::compaction`).
fn compaction_settings_from_env() -> CompactionSettings {
    CompactionSettings::from_env_values(
        std::env::var("AAOS_COMPACTION_ENABLED").ok().as_deref(),
        std::env::var("AAOS_COMPACTION_RESERVE_TOKENS")
            .ok()
            .as_deref(),
        std::env::var("AAOS_COMPACTION_KEEP_RECENT_TOKENS")
            .ok()
            .as_deref(),
    )
}

/// The CLI's event sink: kernel agent events go to the renderer; compaction
/// hook failures print the original stderr text. The CLI no longer
/// subscribes the agent directly — every event arrives through the sink.
struct CliEventSink {
    json_mode: bool,
}

/// The shared session prologue for both entry modes: capture `--json` into
/// the CLI event sink (the renderer reads `cli.json` directly) and assemble
/// the session.
async fn prepare_session(cli: &Cli, paths: &Paths) -> Result<SessionHandle, String> {
    let sink = Arc::new(CliEventSink {
        json_mode: cli.json,
    });
    create_session(&build_session_config(cli, paths)?, sink).await
}

impl EventSink for CliEventSink {
    fn on_event(&self, event: SessionEvent) {
        match event {
            SessionEvent::Agent(event) => print_agent_event(&event, self.json_mode),
            SessionEvent::CompactionFailed { error } => {
                let _ = writeln!(io::stderr(), "compaction failed: {error}");
            }
        }
    }
}

async fn run_prompt(cli: Cli, paths: Paths) -> Result<ExitCode, String> {
    let prompt = cli.prompt.join(" ");
    if prompt.trim().is_empty() {
        return Err("missing prompt".into());
    }
    let mut session = prepare_session(&cli, &paths).await?;
    let json_mode = cli.json;

    let outcome = session.run_turn(&prompt).await?;

    if !json_mode {
        let mut stdout = io::stdout();
        let _ = writeln!(stdout);
    }

    match outcome.stop_reason {
        Some(StopReason::Aborted) => {
            if !json_mode {
                let _ = writeln!(io::stderr(), "aborted");
            }
            Ok(ExitCode::from(130))
        }
        Some(StopReason::Error) => {
            let _ = writeln!(
                io::stderr(),
                "{}",
                outcome
                    .error_message
                    .unwrap_or_else(|| "provider error".into())
            );
            Ok(ExitCode::from(1))
        }
        _ => Ok(ExitCode::SUCCESS),
    }
}

async fn run_repl(cli: &Cli, paths: &Paths) -> Result<ExitCode, String> {
    let mut session = prepare_session(cli, paths).await?;
    let json_mode = cli.json;

    let stdin = io::stdin();
    for line in stdin.lines() {
        let input = match line {
            // A stdin read error ends the loop the same way EOF does; the
            // session stays persisted either way.
            Ok(input) => input,
            Err(_) => break,
        };
        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        // Manual `/compact` path: `/compact` compacts the current node.
        // Slash commands never reach the model.
        if let Some(rest) = input.strip_prefix("/compact") {
            if !rest.is_empty() {
                // `/compactX…` is not the compact command — fall through to
                // the unknown-command hint below.
                let _ = writeln!(io::stderr(), "unknown command: {input}");
                continue;
            }
            match session.compact_now().await {
                Ok(outcome) => {
                    let _ = writeln!(
                        io::stderr(),
                        "Compacted into {} ({} → {} tokens)",
                        outcome.compacted_id,
                        outcome.before_tokens,
                        outcome.after_tokens
                    );
                }
                Err(err) => {
                    let _ = writeln!(io::stderr(), "{err}");
                }
            }
            continue;
        }
        if input.starts_with('/') {
            let _ = writeln!(io::stderr(), "unknown command: {input}");
            continue;
        }
        let outcome = match session.run_turn(input).await {
            Ok(outcome) => outcome,
            Err(err) => {
                let _ = writeln!(io::stderr(), "{err}");
                continue;
            }
        };
        if !json_mode {
            let mut stdout = io::stdout();
            let _ = writeln!(stdout);
        }
        match outcome.stop_reason {
            Some(StopReason::Aborted) if !json_mode => {
                let _ = writeln!(io::stderr(), "aborted");
            }
            Some(StopReason::Error) if !json_mode => {
                let _ = writeln!(
                    io::stderr(),
                    "{}",
                    outcome
                        .error_message
                        .unwrap_or_else(|| "provider error".into())
                );
            }
            _ => {}
        }
    }
    // EOF (Ctrl+D) or a read error ends the REPL. Only claim a save when
    // this run actually persisted something, and print this process's own
    // node — the session it derived and wrote, never a global latest guess.
    // Always to stderr — `--json` only requires stdout to stay pure JSON.
    if session.has_persisted_segments() {
        let session_id = session.current_session_id().await;
        let _ = writeln!(
            io::stderr(),
            "\nSession saved. Resume with:\n  aaos --session {session_id}"
        );
    }
    Ok(ExitCode::SUCCESS)
}

fn print_agent_event(event: &AgentEvent, json_mode: bool) {
    match event {
        AgentEvent::MessageUpdate {
            assistant_event, ..
        } => {
            if !json_mode {
                match assistant_event.as_ref() {
                    AssistantMessageEvent::TextDelta { delta, .. } => {
                        print!("{delta}");
                        let _ = io::stdout().flush();
                    }
                    AssistantMessageEvent::ToolCallEnd { tool_call, .. } => {
                        println!(
                            "● {}({})",
                            tool_call.name,
                            summarize_args(&tool_call.name, &tool_call.arguments)
                        );
                    }
                    _ => {}
                }
            }
        }
        AgentEvent::MessageEnd { message } if json_mode => {
            let Some(assistant) = message.as_assistant() else {
                return;
            };
            match assistant.stop_reason {
                StopReason::Error | StopReason::Aborted => {
                    println!(
                        "{}",
                        json!({
                            "type": "error",
                            "reason": assistant.stop_reason.to_string(),
                            "message": assistant.error_message
                        })
                    );
                }
                _ => {
                    println!("{}", message_end_json(assistant));
                }
            }
        }
        AgentEvent::ToolExecutionStart {
            tool_call_id,
            tool_name,
            args,
        } if json_mode => {
            println!(
                "{}",
                json!({
                    "type": "tool_execution_start",
                    "tool_call_id": tool_call_id,
                    "name": tool_name,
                    "args": args
                })
            );
        }
        AgentEvent::ToolExecutionEnd {
            tool_call_id,
            tool_name,
            result,
            is_error,
        } => {
            if json_mode {
                println!(
                    "{}",
                    json!({
                        "type": "tool_execution_end",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "result": summarize_result_text(result),
                        "is_error": is_error
                    })
                );
            } else {
                println!("  → {}", summarize_result_text(result));
            }
        }
        AgentEvent::AgentEnd { messages } if json_mode => {
            let reason = messages
                .iter()
                .rev()
                .find_map(|m| m.as_assistant())
                .map(|a| a.stop_reason.to_string())
                .unwrap_or_else(|| "stop".into());
            println!("{}", json!({"type": "done", "reason": reason}));
        }
        _ => {}
    }
}

/// Serialize an assistant message into a `message_end` JSON event.
fn message_end_json(assistant: &AssistantMessage) -> Value {
    let mut text = String::new();
    let mut tool_calls = Vec::new();
    for block in &assistant.content {
        match block {
            ContentBlock::Text { text: t } => text.push_str(t),
            ContentBlock::ToolCall(tc) => tool_calls.push(json!({
                "id": tc.id,
                "name": tc.name,
                "arguments": tc.arguments
            })),
            _ => {}
        }
    }
    json!({
        "type": "message_end",
        "role": "assistant",
        "stop_reason": assistant.stop_reason.to_string(),
        "content": text,
        "tool_calls": tool_calls
    })
}

/// Extract the first text block from a tool result, truncated to 200 chars.
fn summarize_result_text(result: &AgentToolResult) -> String {
    let text = result
        .content
        .iter()
        .find_map(|c| match c {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .unwrap_or("(no text)");
    truncate_str(text, 200)
}

/// Produce a single-line argument summary for a tool call.
///
/// Picks the tool's primary argument so the human rendering stays compact:
/// `read`/`edit`/`write` → `path` (with optional offset/limit for read),
/// `bash` → `command`. Unknown tools fall back to compact JSON.
fn summarize_args(tool_name: &str, args: &Value) -> String {
    match tool_name {
        "read" => {
            let path = args.get("path").and_then(Value::as_str).unwrap_or("?");
            let offset = args.get("offset").and_then(Value::as_u64);
            let limit = args.get("limit").and_then(Value::as_u64);
            match (offset, limit) {
                (Some(o), Some(l)) => format!("{path}:{o}-{l}"),
                _ => path.to_string(),
            }
        }
        "bash" => args
            .get("command")
            .and_then(Value::as_str)
            .map(|s| truncate_str(s, 60))
            .unwrap_or_else(|| "?".into()),
        "edit" | "write" => args
            .get("path")
            .and_then(Value::as_str)
            .unwrap_or("?")
            .to_string(),
        _ => truncate_str(&args.to_string(), 60),
    }
}

/// Truncate a string to `max` chars, appending `…` if truncated.
fn truncate_str(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max).collect();
        format!("{truncated}…")
    }
}

#[cfg(test)]
mod tests {
    // Test-support expects and unwraps are the test idiom here; the
    // production paths above stay panic-free.
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::sync::{Arc, Mutex};

    use aaos_tools::{SkillIndex, build_system_prompt, create_coding_tools};
    use pi_agent_core::agent::Agent;
    use serde_json::json;

    use pi_agent_core::stream::{MockAssistantStream, mock_stream_fn};
    use pi_agent_core::types::{
        AssistantMessage, ContentBlock, LlmContext, Model, StopReason, ThinkingLevel,
    };

    #[tokio::test]
    async fn prompt_runs_read_tool_and_sends_schema() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("note.txt"), "hello from file").unwrap();
        let captured_ctx: Arc<Mutex<Option<LlmContext>>> = Arc::new(Mutex::new(None));
        let captured_ctx_for_stream = captured_ctx.clone();
        let llm_calls = Arc::new(AtomicUsize::new(0));
        let llm_calls_for_stream = llm_calls.clone();
        let stream_fn = mock_stream_fn(move |_model, ctx, _stream_options| {
            let call_index = llm_calls_for_stream.fetch_add(1, Ordering::SeqCst);
            if call_index == 0 {
                *captured_ctx_for_stream.lock().unwrap() = Some(ctx);
                let msg = AssistantMessage {
                    content: vec![ContentBlock::tool_call(
                        "c1",
                        "read",
                        json!({"path": "note.txt"}),
                    )],
                    stop_reason: StopReason::ToolUse,
                    ..Default::default()
                };
                Box::new(MockAssistantStream::new(msg))
            } else {
                Box::new(MockAssistantStream::new(AssistantMessage::text("done")))
            }
        });
        let cwd = tmp.path().to_path_buf();
        let skills = Arc::new(SkillIndex::discover(
            &cwd.join(".agents/skills"),
            &cwd.join(".agents/skills"),
        ));
        let tools = create_coding_tools(&cwd, skills.clone());
        let system_prompt = build_system_prompt(&cwd, &tools, &skills);
        let mut agent = Agent::new(stream_fn);
        agent.state.model = Model {
            id: "test".into(),
            ..Model::unknown()
        };
        agent.state.thinking_level = ThinkingLevel::Off;
        agent.state.tools = tools;
        agent.state.system_prompt = system_prompt;
        agent.stream_fn_options.api_key = None;
        agent.prompt("read the note").await.unwrap();
        let ctx = captured_ctx
            .lock()
            .unwrap()
            .clone()
            .expect("first llm call");
        let names: Vec<_> = ctx.tools.iter().map(|t| t.name().to_string()).collect();
        assert_eq!(names, ["read", "bash", "edit", "write"]);
        let read = ctx.tools.iter().find(|t| t.name() == "read").unwrap();
        assert_eq!(read.parameters()["required"], json!(["path"]));
        assert!(ctx.system_prompt.contains("Available tools:"));
        let cwd = tmp.path().display().to_string().replace('\\', "/");
        assert!(
            ctx.system_prompt
                .contains(&format!("Current working directory: {cwd}")),
            "{}",
            ctx.system_prompt
        );
        let tool_text: String = agent
            .state
            .messages
            .iter()
            .filter_map(|m| m.as_tool_result())
            .flat_map(|t| t.content.iter())
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert!(tool_text.contains("hello from file"), "{tool_text}");
        assert!(llm_calls.load(Ordering::SeqCst) >= 2);
    }
}
