use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::Arc;

use aaos_providers::{
    DEFAULT_MODEL_LIST_URL, Paths, parse_thinking, resolve_catalog_model, stream_fn_for,
};
use aaos_session::{AgentSession, SessionStore};
use aaos_tools::{SkillIndex, build_system_prompt, create_coding_tools};
use clap::Parser;
use pi_agent_core::agent::Agent;
use pi_agent_core::types::{
    AgentEvent, AgentState, AgentToolResult, AssistantMessage, AssistantMessageEvent, ContentBlock,
    StopReason, ThinkingLevel,
};
use serde_json::{Value, json};

/// DeepSeek product defaults; the provider crate stays product-agnostic.
const DEFAULT_PROVIDER: &str = "deepseek";
const DEFAULT_MODEL_ID: &str = "deepseek-v4-flash";
const DEFAULT_THINKING: ThinkingLevel = ThinkingLevel::High;

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

/// Resolve the session node to continue. An explicit `--session` wins and
/// resumes that node in place (`--fork` derives a new session from it
/// instead); an unknown id errors rather than silently starting a session
/// nothing points at. The default continues the user's head session — the persisted
/// head pointer (the node last appended to; `latest_created_session` for
/// stores that predate the pointer) — as a fresh derivation: the derivation
/// inherits the full view, while each process appends to its own node, so n
/// concurrent runs never cross-write one session. An empty store gets a
/// fresh root.
async fn resolve_session(store: &SessionStore, cli: &Cli) -> Result<String, String> {
    if let Some(id) = cli.session_id.as_deref() {
        if cli.fork {
            return store.fork(id).await.map_err(|e| e.to_string());
        }
        if !store.session_exists(id).await.map_err(|e| e.to_string())? {
            return Err(format!("session not found: {id}"));
        }
        return Ok(id.to_string());
    }
    let target = match store.head().await.map_err(|e| e.to_string())? {
        Some(id) => id,
        None => match store
            .latest_created_session()
            .await
            .map_err(|e| e.to_string())?
        {
            Some(id) => id,
            None => return store.create_root().await.map_err(|e| e.to_string()),
        },
    };
    store.fork(&target).await.map_err(|e| e.to_string())
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

    let catalog_model = resolve_catalog_model(paths, &model_list_url_override(), &spec)
        .await
        .map_err(|e| e.to_string())?;
    let api_key = catalog_model
        .resolve_api_key(|k| std::env::var(k).ok())
        .map_err(|e| e.to_string())?;
    let model = catalog_model.to_model();

    let provider = stream_fn_for(&model).map_err(|e| e.to_string())?;
    let cwd = std::env::current_dir().map_err(|e| e.to_string())?;
    // Discover skills once at startup (frozen for the process lifetime):
    // user-level `~/.agents/skills/` plus project-level `<cwd>/.agents/skills/`.
    let user_skills_dir = std::env::home_dir()
        .map(|h| h.join(".agents/skills"))
        .unwrap_or_default();
    let skills = Arc::new(SkillIndex::discover(
        &user_skills_dir,
        &cwd.join(".agents/skills"),
    ));
    let tools = create_coding_tools(&cwd, skills.clone());
    let system_prompt = build_system_prompt(&cwd, &tools, &skills);
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

/// Resolve the outcome of a finished turn: the last assistant message's stop
/// reason, plus its error message falling back to the session-level error.
fn turn_outcome(state: &AgentState) -> (Option<StopReason>, Option<String>) {
    let last = state.messages.iter().rev().find_map(|m| m.as_assistant());
    let stop_reason = last.map(|m| m.stop_reason);
    let error_message = last
        .and_then(|m| m.error_message.clone())
        .or_else(|| state.error_message.clone());
    (stop_reason, error_message)
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
    let (stop_reason, error_message) = turn_outcome(state);
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
        if let Err(err) = session.agent_mut().prompt(input).await {
            let _ = writeln!(io::stderr(), "{err}");
            continue;
        }
        if !json_mode {
            let mut stdout = io::stdout();
            let _ = writeln!(stdout);
        }
        let state = session.state();
        let (stop_reason, error_message) = turn_outcome(state);
        match stop_reason {
            Some(StopReason::Aborted) if !json_mode => {
                let _ = writeln!(io::stderr(), "aborted");
            }
            Some(StopReason::Error) if !json_mode => {
                let _ = writeln!(
                    io::stderr(),
                    "{}",
                    error_message.unwrap_or_else(|| "provider error".into())
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
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use std::sync::{Arc, Mutex};

    use aaos_session::Segment;
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

    /// The `resolve_session` decision rules; names state one rule each.
    mod resolve_session {
        use super::*;

        /// Issue #61: the default run continues the head session as a fresh
        /// derivation — its own node, the head's full view — and the head only
        /// moves when something is actually appended.
        #[tokio::test]
        async fn default_derives_own_line() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            store
                .append_segment(&root, &Segment::user_text("q"))
                .await
                .unwrap();

            let resolved = resolve_session(&store, &Cli::default()).await.unwrap();
            assert_ne!(
                resolved, root,
                "the default run continues on its own session"
            );
            assert_eq!(
                store.materialize_plain(&resolved).await.unwrap(),
                vec![Segment::user_text("q")],
                "the derivation inherits the head's view"
            );
            assert_eq!(
                store.head().await.unwrap().as_deref(),
                Some(root.as_str()),
                "head follows appends, not derivations"
            );
        }

        #[tokio::test]
        async fn explicit_session_resumes_in_place() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            store
                .append_segment(&root, &Segment::user_text("q"))
                .await
                .unwrap();

            let cli = Cli {
                session_id: Some(root.clone()),
                ..Default::default()
            };
            assert_eq!(
                resolve_session(&store, &cli).await.unwrap(),
                root,
                "--session resumes the node itself"
            );

            let forked = resolve_session(
                &store,
                &Cli {
                    session_id: Some(root.clone()),
                    fork: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap();
            assert_ne!(forked, root);
            assert_eq!(
                store.materialize_plain(&forked).await.unwrap(),
                vec![Segment::user_text("q")]
            );
        }

        /// `--session` must fail loudly on an unknown node instead of silently
        /// starting from a session nothing points at; the `--fork` path is checked
        /// by the store's own lookup.
        #[tokio::test]
        async fn unknown_id_errors() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();

            let err = resolve_session(
                &store,
                &Cli {
                    session_id: Some("nope".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
            assert!(err.contains("nope"), "{err}");

            let err = resolve_session(
                &store,
                &Cli {
                    session_id: Some("nope".into()),
                    fork: true,
                    ..Default::default()
                },
            )
            .await
            .unwrap_err();
            assert!(err.contains("nope"), "{err}");
        }

        #[tokio::test]
        async fn empty_store_creates_root() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();

            let root = resolve_session(&store, &Cli::default()).await.unwrap();
            assert!(store.materialize_plain(&root).await.unwrap().is_empty());
            assert_eq!(store.head().await.unwrap(), None, "no appends, no head");
        }

        /// A store written before the head pointer existed: the fallback picks
        /// the newest created session and derives from it.
        #[tokio::test]
        async fn legacy_store_still_resumes() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            let child = store.fork(&root).await.unwrap();
            assert_eq!(store.head().await.unwrap(), None);

            let resolved = resolve_session(&store, &Cli::default()).await.unwrap();
            assert_ne!(resolved, child);
            assert!(store.materialize_plain(&resolved).await.unwrap().is_empty());
        }
    }
}
