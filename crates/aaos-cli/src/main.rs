use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::SystemTime;

use aaos_catalog::{
    format_model_line, load_catalog, parse_thinking, refresh_catalog, Paths, CACHE_TTL,
    DEFAULT_MODEL_ID, DEFAULT_PROVIDER, DEFAULT_REGISTRY_URL, DEFAULT_THINKING,
};
use aaos_openai::OpenAiCompletionsProvider;
use clap::{Parser, Subcommand};
use pi_agent_core::types::{AgentEvent, AssistantMessageEvent, StopReason, StreamFn};
use serde_json::{json, Value};

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
            if json_mode {
                println!("{}", event_json(assistant_event));
            } else if let AssistantMessageEvent::TextDelta { delta, .. } = assistant_event.as_ref()
            {
                print!("{delta}");
                let _ = io::Write::flush(&mut io::stdout());
            }
        }
        AgentEvent::MessageEnd { message } if json_mode => {
            let Some(assistant) = message.as_assistant() else {
                return;
            };
            let event = match assistant.stop_reason {
                StopReason::Error | StopReason::Aborted => AssistantMessageEvent::Error {
                    reason: assistant.stop_reason,
                    error: assistant.clone(),
                },
                StopReason::ToolUse | StopReason::Pending => return,
                reason => AssistantMessageEvent::Done {
                    reason,
                    message: assistant.clone(),
                },
            };
            println!("{}", event_json(&event));
        }
        _ => {}
    }
}

fn event_json(event: &AssistantMessageEvent) -> Value {
    match event {
        AssistantMessageEvent::Start { .. } => json!({"type": "start"}),
        AssistantMessageEvent::TextStart { content_index, .. } => {
            json!({"type": "text_start", "content_index": content_index})
        }
        AssistantMessageEvent::TextDelta {
            content_index,
            delta,
            ..
        } => json!({"type": "text_delta", "content_index": content_index, "delta": delta}),
        AssistantMessageEvent::TextEnd {
            content_index,
            content,
            ..
        } => json!({"type": "text_end", "content_index": content_index, "content": content}),
        AssistantMessageEvent::ThinkingStart { content_index, .. } => {
            json!({"type": "thinking_start", "content_index": content_index})
        }
        AssistantMessageEvent::ThinkingDelta {
            content_index,
            delta,
            ..
        } => json!({"type": "thinking_delta", "content_index": content_index, "delta": delta}),
        AssistantMessageEvent::ThinkingEnd {
            content_index,
            content,
            ..
        } => json!({"type": "thinking_end", "content_index": content_index, "content": content}),
        AssistantMessageEvent::ToolCallStart { content_index, .. } => {
            json!({"type": "toolcall_start", "content_index": content_index})
        }
        AssistantMessageEvent::ToolCallDelta {
            content_index,
            delta,
            ..
        } => json!({"type": "toolcall_delta", "content_index": content_index, "delta": delta}),
        AssistantMessageEvent::ToolCallEnd {
            content_index,
            tool_call,
            ..
        } => json!({
            "type": "toolcall_end",
            "content_index": content_index,
            "id": tool_call.id,
            "name": tool_call.name,
            "arguments": tool_call.arguments
        }),
        AssistantMessageEvent::Done { reason, .. } => {
            json!({"type": "done", "reason": reason.to_string()})
        }
        AssistantMessageEvent::Error { reason, error } => json!({
            "type": if *reason == StopReason::Aborted { "aborted" } else { "error" },
            "reason": reason.to_string(),
            "message": error.error_message
        }),
    }
}
