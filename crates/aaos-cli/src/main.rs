use std::io::{self, Write};
use std::process::ExitCode;
use std::time::SystemTime;

use aaos_catalog::{
    format_model_line, load_catalog, parse_thinking, refresh_catalog, Paths, CACHE_TTL,
    DEFAULT_MODEL_ID, DEFAULT_PROVIDER, DEFAULT_REGISTRY_URL, DEFAULT_THINKING,
};
use aaos_openai::OpenAiCompletionsProvider;
use clap::{Parser, Subcommand};
use pi_agent_core::types::{
    AssistantMessageEvent, LlmContext, Message, StopReason, StreamFn, StreamFnOptions, UserMessage,
};
use serde_json::{json, Value};
use tokio::sync::watch;

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

    let provider = OpenAiCompletionsProvider::new();
    let (abort_tx, abort_rx) = watch::channel(false);
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        let _ = abort_tx.send(true);
    });

    let mut stream = provider
        .call(
            model,
            LlmContext {
                system_prompt: String::new(),
                messages: vec![Message::User(UserMessage::new(prompt))],
                tools: vec![],
            },
            StreamFnOptions {
                api_key: Some(api_key),
                thinking_level: Some(thinking),
                ..Default::default()
            },
            abort_rx,
        )
        .await
        .map_err(|e| e)?;

    let json_mode = cli.json;
    let mut stdout = io::stdout();
    let mut terminal: Option<StopReason> = None;
    while let Some(event) = stream.next_event().await {
        if json_mode {
            writeln!(stdout, "{}", event_json(&event)).map_err(|e| e.to_string())?;
        } else if let AssistantMessageEvent::TextDelta { delta, .. } = &event {
            write!(stdout, "{delta}").map_err(|e| e.to_string())?;
            let _ = stdout.flush();
        }
        match &event {
            AssistantMessageEvent::Done { reason, .. } => terminal = Some(*reason),
            AssistantMessageEvent::Error { reason, .. } => {
                terminal = Some(*reason);
            }
            _ => {}
        }
    }
    let final_msg = stream.result().await;
    if !json_mode {
        writeln!(stdout).ok();
    }
    match terminal.or(Some(final_msg.stop_reason)) {
        Some(StopReason::Aborted) => {
            if !json_mode {
                eprintln!("aborted");
            }
            Ok(ExitCode::from(130))
        }
        Some(StopReason::Error) => {
            eprintln!(
                "{}",
                final_msg
                    .error_message
                    .unwrap_or_else(|| "provider error".into())
            );
            Ok(ExitCode::from(1))
        }
        _ => Ok(ExitCode::SUCCESS),
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

#[cfg(test)]
mod event_json_tests {
    use super::*;
    use pi_agent_core::types::{AssistantMessage, ToolCall};

    #[test]
    fn jsonl_covers_required_event_types() {
        let partial = AssistantMessage::default();
        for (ev, ty) in [
            (
                AssistantMessageEvent::Start {
                    partial: partial.clone(),
                },
                "start",
            ),
            (
                AssistantMessageEvent::Done {
                    reason: StopReason::Stop,
                    message: partial.clone(),
                },
                "done",
            ),
            (
                AssistantMessageEvent::Error {
                    reason: StopReason::Error,
                    error: partial.clone(),
                },
                "error",
            ),
            (
                AssistantMessageEvent::Error {
                    reason: StopReason::Aborted,
                    error: partial.clone(),
                },
                "aborted",
            ),
        ] {
            assert_eq!(event_json(&ev)["type"], ty);
        }
        let tool = AssistantMessageEvent::ToolCallEnd {
            content_index: 0,
            tool_call: ToolCall {
                id: "1".into(),
                name: "echo".into(),
                arguments: json!({"x": 1}),
            },
            partial,
        };
        assert_eq!(event_json(&tool)["type"], "toolcall_end");
    }
}
