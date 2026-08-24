use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::Arc;

use aaos_providers::{
    DEFAULT_REGISTRY_URL, Paths, load_catalog, parse_thinking, resolve_model, stream_fn_for,
};
use aaos_session::{AgentSession, SessionStore};
use aaos_tools::{build_system_prompt, create_coding_tools};
use clap::Parser;
use pi_agent_core::agent::Agent;
use pi_agent_core::types::{
    AgentEvent, AgentToolResult, AssistantMessage, AssistantMessageEvent, ContentBlock, StopReason,
    ThinkingLevel,
};
use serde_json::{Value, json};

/// DeepSeek product defaults; the provider crate stays product-agnostic.
const DEFAULT_PROVIDER: &str = "deepseek";
const DEFAULT_MODEL_ID: &str = "deepseek-v4-flash";
const DEFAULT_THINKING: ThinkingLevel = ThinkingLevel::High;

#[derive(Parser, Debug)]
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
    /// Session node id to resume instead of the latest session.
    #[arg(long = "session")]
    session_id: Option<String>,
    /// Fork the resolved session and resume the new node.
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
    if cli.prompt.is_empty() {
        run_repl(&cli, &paths).await
    } else {
        run_prompt(cli, paths).await
    }
}

fn paths_from_env() -> Paths {
    if let Ok(dir) = std::env::var("AAOS_CONFIG_DIR") {
        Paths::from_config_dir(dir)
    } else {
        Paths::default_user()
    }
}

fn registry_url_override() -> String {
    std::env::var("AAOS_MODELS_URL").unwrap_or_else(|_| DEFAULT_REGISTRY_URL.to_string())
}

/// Build the session for both entry modes: open the store, resolve the node
/// to continue, build the agent, bind it via `AgentSession::new` (MessageEnd
/// → append_segment listener) and `resume` its view into `state.messages`
/// (replacing it, with dangling tool-call repair).
async fn build_session(cli: &Cli, paths: &Paths) -> Result<AgentSession, String> {
    let store = SessionStore::open(&paths.config_dir)
        .await
        .map_err(|e| e.to_string())?;
    let session_id = resolve_session(&store, cli).await?;
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let mut session = AgentSession::new(store, build_agent(cli, paths).await?, &session_id, cwd);
    session
        .resume(&session_id)
        .await
        .map_err(|e| e.to_string())?;
    Ok(session)
}

/// Resolve the session node to continue: an explicit `--session` wins, else
/// the latest session, else a fresh root. `--fork` forks a pre-existing
/// target only; a just-created root is resumed as-is.
async fn resolve_session(store: &SessionStore, cli: &Cli) -> Result<String, String> {
    let target = if let Some(id) = cli.session_id.as_deref() {
        id.to_string()
    } else if let Some(latest) = store.latest_session().await.map_err(|e| e.to_string())? {
        latest
    } else {
        return store.create_root().await.map_err(|e| e.to_string());
    };
    if cli.fork {
        store.fork(&target).await.map_err(|e| e.to_string())
    } else {
        Ok(target)
    }
}

async fn build_agent(cli: &Cli, paths: &Paths) -> Result<Agent, String> {
    let thinking = match cli.thinking.as_deref() {
        Some(s) => parse_thinking(s)?,
        None => DEFAULT_THINKING,
    };
    let provider_id = cli
        .provider
        .as_deref()
        .unwrap_or(DEFAULT_PROVIDER)
        .to_string();
    let model_id = cli.model.as_deref().unwrap_or(DEFAULT_MODEL_ID).to_string();
    let spec = if model_id.contains('/') {
        model_id.clone()
    } else {
        format!("{provider_id}/{model_id}")
    };

    let models = load_catalog(paths, &registry_url_override())
        .await
        .map_err(|e| e.to_string())?;
    let catalog_model = resolve_model(&models, &spec).map_err(|e| e.to_string())?;
    let api_key = catalog_model
        .resolve_api_key(|k| std::env::var(k).ok())
        .map_err(|e| e.to_string())?;
    let mut model = catalog_model.to_model();
    model.api = catalog_model.api.clone();

    let provider = stream_fn_for(&model).map_err(|e| e.to_string())?;
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    let tools = create_coding_tools(&cwd);
    let system_prompt = build_system_prompt(&cwd, &tools);
    let mut agent = Agent::new(provider);
    agent.state.model = model;
    agent.state.thinking_level = thinking;
    agent.state.tools = tools;
    agent.state.system_prompt = system_prompt;
    agent.stream_fn_options.api_key = Some(api_key);

    let json_mode = cli.json;
    let _ = agent.subscribe(Arc::new(move |event, _signal| {
        Box::pin(async move {
            print_agent_event(&event, json_mode);
        })
    }));

    Ok(agent)
}

async fn run_prompt(cli: Cli, paths: Paths) -> Result<ExitCode, String> {
    let prompt = cli.prompt.join(" ");
    if prompt.trim().is_empty() {
        return Err("missing prompt".into());
    }

    let mut session = build_session(&cli, &paths).await?;
    let json_mode = cli.json;

    session
        .agent_mut()
        .prompt(prompt)
        .await
        .map_err(|e| e.to_string())?;

    if !json_mode {
        let mut stdout = io::stdout();
        let _ = writeln!(stdout);
    }

    let state = session.state();
    let last = state.messages.iter().rev().find_map(|m| m.as_assistant());
    let stop_reason = last.map(|m| m.stop_reason);
    let error_message = last
        .and_then(|m| m.error_message.clone())
        .or_else(|| state.error_message.clone());
    match stop_reason {
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
                error_message.unwrap_or_else(|| "provider error".into())
            );
            Ok(ExitCode::from(1))
        }
        _ => Ok(ExitCode::SUCCESS),
    }
}

async fn run_repl(cli: &Cli, paths: &Paths) -> Result<ExitCode, String> {
    let mut session = build_session(cli, paths).await?;
    let json_mode = cli.json;

    let stdin = io::stdin();
    for line in stdin.lines() {
        let input = line.map_err(|e| e.to_string())?;
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Re-entrancy errors cannot occur in a sequential read loop; surface
        // them and keep the session alive either way.
        if let Err(err) = session.agent_mut().prompt(input).await {
            let _ = writeln!(io::stderr(), "{err}");
            continue;
        }
        if !json_mode {
            let mut stdout = io::stdout();
            let _ = writeln!(stdout);
        }

        let state = session.state();
        let last = state.messages.iter().rev().find_map(|m| m.as_assistant());
        match last.map(|m| m.stop_reason) {
            Some(StopReason::Aborted) if !json_mode => {
                let _ = writeln!(io::stderr(), "aborted");
            }
            Some(StopReason::Error) if !json_mode => {
                let error_message = last
                    .and_then(|m| m.error_message.clone())
                    .or_else(|| state.error_message.clone())
                    .unwrap_or_else(|| "provider error".into());
                let _ = writeln!(io::stderr(), "{error_message}");
            }
            _ => {}
        }
    }
    // EOF (Ctrl+D) or a read error ends the REPL. The session is already
    // persisted; print the resume command so the user can pick it back up.
    // `--json` keeps stdout pure JSON; the hint goes to stderr either way.
    if !json_mode {
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use pi_agent_core::stream::{MockAssistantStream, mock_stream_fn};
    use pi_agent_core::types::{
        AssistantMessage, ContentBlock, LlmContext, Model, StopReason, ThinkingLevel,
    };

    use super::*;

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
        let tools = create_coding_tools(&cwd);
        let system_prompt = build_system_prompt(&cwd, &tools);
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
