use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::SystemTime;

use aaos_catalog::{
    CACHE_TTL, DEFAULT_MODEL_ID, DEFAULT_PROVIDER, DEFAULT_REGISTRY_URL, DEFAULT_THINKING, Paths,
    format_model_line, load_catalog, parse_thinking, refresh_catalog,
};
use aaos_openai::OpenAiCompletionsProvider;
use clap::{Parser, Subcommand};
use pi_agent_core::types::{
    AgentEvent, AgentToolResult, AssistantMessage, AssistantMessageEvent, ContentBlock, StopReason,
    StreamFn,
};
use serde_json::{Value, json};

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
    #[command(subcommand)]
    command: Option<Command>,
    /// Prompt text when no subcommand is given.
    #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
    prompt: Vec<String>,
}

#[derive(Subcommand, Debug)]
enum Command {
    Models {
        #[command(subcommand)]
        command: ModelsCommand,
    },
}

#[derive(Subcommand, Debug)]
enum ModelsCommand {
    Refresh,
}

#[tokio::main]
async fn main() -> ExitCode {
    match run().await {
        Ok(code) => code,
        Err(err) => {
            eprintln!("{err}");
            ExitCode::from(1)
        }
    }
}

async fn run() -> Result<ExitCode, String> {
    let cli = Cli::parse();
    let paths = paths_from_env();
    match cli.command {
        Some(Command::Models {
            command: ModelsCommand::Refresh,
        }) => {
            let outcome = refresh_catalog(&paths, &registry_url_override(), SystemTime::now())
                .await
                .map_err(|e| e.to_string())?;
            if let Some(warning) = &outcome.catalog.warning {
                eprintln!("warning: {warning}");
            }
            for model in &outcome.catalog.models {
                println!("{}", format_model_line(model));
            }
            Ok(ExitCode::SUCCESS)
        }
        None => run_prompt(cli, paths).await,
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

async fn run_prompt(cli: Cli, paths: Paths) -> Result<ExitCode, String> {
    let prompt = cli.prompt.join(" ");
    if prompt.trim().is_empty() {
        return Err("missing prompt".into());
    }
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

    let outcome = load_catalog(
        &paths,
        &registry_url_override(),
        SystemTime::now(),
        CACHE_TTL,
    )
    .await
    .map_err(|e| e.to_string())?;
    if let Some(warning) = &outcome.catalog.warning {
        eprintln!("warning: {warning}");
    }
    let catalog_model = outcome.catalog.resolve(&spec).map_err(|e| e.to_string())?;
    let api_key = catalog_model
        .resolve_api_key(|k| std::env::var(k).ok())
        .map_err(|e| e.to_string())?;
    let mut model = catalog_model.to_model();
    model.api = catalog_model.api.clone();

    let provider: Arc<dyn StreamFn> = Arc::new(OpenAiCompletionsProvider::new());
    let mut session = aaos_session::AgentSession::new(aaos_session::SessionOptions {
        cwd: std::env::current_dir().map_err(|e| e.to_string())?,
        model,
        stream_fn: provider,
        thinking_level: thinking,
        api_key: Some(api_key),
    });

    let json_mode = cli.json;
    let _unsub = session.subscribe(Arc::new(move |event, _signal| {
        Box::pin(async move {
            print_agent_event(&event, json_mode);
        })
    }));

    let handle = session.handle();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        handle.abort();
    });

    session.prompt(prompt).await.map_err(|e| e.to_string())?;

    if !json_mode {
        let mut stdout = io::stdout();
        let _ = writeln!(stdout);
    }

    let state = &session.agent().state;
    let last = state.messages.iter().rev().find_map(|m| m.as_assistant());
    let stop_reason = last.map(|m| m.stop_reason);
    let error_message = last
        .and_then(|m| m.error_message.clone())
        .or_else(|| state.error_message.clone());
    match stop_reason {
        Some(StopReason::Aborted) => {
            if !json_mode {
                eprintln!("aborted");
            }
            Ok(ExitCode::from(130))
        }
        Some(StopReason::Error) => {
            eprintln!(
                "{}",
                error_message.unwrap_or_else(|| "provider error".into())
            );
            Ok(ExitCode::from(1))
        }
        _ => Ok(ExitCode::SUCCESS),
    }
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
