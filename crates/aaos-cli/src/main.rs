use std::io::{self, Write};
use std::process::ExitCode;
use std::sync::Arc;

use aaos_providers::{
    DEFAULT_MODEL_LIST_URL, Paths, ProviderRetryConfig, parse_thinking, resolve_catalog_model,
    stream_fn_for_with_retry,
};
use aaos_session::{AgentSession, SessionStore};
use aaos_tools::{SkillIndex, build_system_prompt, create_coding_tools};
use clap::Parser;
use pi_agent_core::agent::Agent;
use pi_agent_core::types::{
    AgentContext, AgentEvent, AgentLoopTurnUpdate, AgentState, AgentToolResult, AssistantMessage,
    AssistantMessageEvent, ContentBlock, Model, StopReason, StreamFn, ThinkingLevel,
};
use serde_json::{Value, json};
mod compaction_coordinator;

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
    let (model, provider, api_key) = resolve_model_provider(cli, paths).await?;
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
    agent.stream_fn_options.provider_retry_max_retries = 0;
    agent.stream_fn_options.provider_retry_max_delay_ms = 60000;
    let json_mode = cli.json;
    let _ = agent.subscribe(Arc::new(move |event, _signal| {
        Box::pin(async move {
            print_agent_event(&event, json_mode);
        })
    }));

    Ok(agent)
}

/// Resolve the provider/model from CLI args and the model catalog, returning
/// the runtime model, its provider stream (with retry layer), and the API key.
/// The session agent owns the resolved model; the compaction coordinator
/// reads its `context_window` from `agent.state` (no second resolution).
async fn resolve_model_provider(
    cli: &Cli,
    paths: &Paths,
) -> Result<(Model, Arc<dyn StreamFn>, String), String> {
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
    let provider = stream_fn_for_with_retry(&model, ProviderRetryConfig::default())
        .map_err(|e| e.to_string())?;
    Ok((model, provider, api_key))
}
/// Build the compaction coordinator from the already-resolved live model
/// (the session agent's `state.model`) — its context window drives the
/// auto-trigger checks. No model re-resolution happens here.
fn build_compaction_coordinator(
    model: &Model,
    store: &SessionStore,
) -> Arc<compaction_coordinator::CompactionCoordinator> {
    Arc::new(compaction_coordinator::CompactionCoordinator::new(
        store.clone(),
        compaction_coordinator::CompactionSettings::from_env(),
        model,
    ))
}

/// Install the auto-trigger compaction hooks on `agent`:
/// - `transform_context` compacts pre-request when the outgoing context
///   exceeds the window; on success the injected view replaces the messages
///   for this request and the append target switches to the compacted node.
/// - `prepare_next_turn` compacts post-turn on threshold overshoot or
///   context overflow; on success the injected view replaces the context for
///   subsequent in-run turns. A second overflow per run fails the run.
///
/// Both hooks share the coordinator's per-run state; `node_handle` is the
/// session's node-id lock (same lock the persist listener reads), so the
/// append-target switch is atomic with respect to post-compaction appends.
fn install_compaction_hooks(
    agent: &mut Agent,
    coordinator: &Arc<compaction_coordinator::CompactionCoordinator>,
    node_handle: Arc<tokio::sync::RwLock<String>>,
) {
    let transform_coordinator = coordinator.clone();
    let transform_handle = node_handle.clone();
    agent.transform_context = Some(Arc::new(move |messages, _abort| {
        let coordinator = transform_coordinator.clone();
        let node_handle = transform_handle.clone();
        Box::pin(async move {
            let current_id = node_handle.read().await.clone();
            match coordinator.pre_request_hook(&messages, &current_id).await {
                Some(outcome) => {
                    *node_handle.write().await = outcome.compacted_id.clone();
                    Ok(outcome.injected_view)
                }
                None => Ok(messages),
            }
        })
    }));

    let prepare_coordinator = coordinator.clone();
    let prepare_handle = node_handle.clone();
    agent.prepare_next_turn = Some(Arc::new(move |ctx, _abort| {
        let coordinator = prepare_coordinator.clone();
        let node_handle = prepare_handle.clone();
        Box::pin(async move {
            let current_id = node_handle.read().await.clone();
            match coordinator
                .post_turn_hook(&ctx.message, &ctx.context, &current_id)
                .await
            {
                Ok(Some(outcome)) => {
                    *node_handle.write().await = outcome.compacted_id.clone();
                    Ok(Some(AgentLoopTurnUpdate {
                        context: Some(AgentContext {
                            system_prompt: ctx.context.system_prompt.clone(),
                            messages: outcome.injected_view,
                            tools: ctx.context.tools.clone(),
                        }),
                        model: None,
                        thinking_level: None,
                    }))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(e),
            }
        })
    }));
}

/// Resync the session's in-memory view after a run if a compaction committed
/// mid-run (auto or manual): `take_pending_resync` yields the compacted node.
async fn resync_after_run(
    session: &mut AgentSession,
    coordinator: &Arc<compaction_coordinator::CompactionCoordinator>,
) -> Result<(), String> {
    if let Some(id) = coordinator.take_pending_resync() {
        session.resume(&id).await.map_err(|e| e.to_string())?;
    }
    Ok(())
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
    let coordinator = build_compaction_coordinator(&session.agent().state.model, session.store());
    let node_handle = session.session_id_lock();
    install_compaction_hooks(session.agent_mut(), &coordinator, node_handle);

    coordinator.begin_run();
    session
        .agent_mut()
        .prompt(prompt)
        .await
        .map_err(|e| e.to_string())?;
    resync_after_run(&mut session, &coordinator).await?;

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

    let coordinator = build_compaction_coordinator(&session.agent().state.model, session.store());
    let node_handle = session.session_id_lock();
    install_compaction_hooks(session.agent_mut(), &coordinator, node_handle);

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
            let current_id = session.current_session_id().await;
            match coordinator.compact(&current_id).await {
                Ok(outcome) => {
                    if let Err(e) = session.resume(&outcome.compacted_id).await {
                        let _ = writeln!(
                            io::stderr(),
                            "Compaction failed: resume onto {} failed: {e}",
                            outcome.compacted_id
                        );
                        continue;
                    }
                    // Manual compact resumes immediately; consume the pending
                    // resync the coordinator recorded so the post-run resync
                    // doesn't re-resume the same node.
                    let _ = coordinator.take_pending_resync();
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
        coordinator.begin_run();
        if let Err(err) = session.agent_mut().prompt(input).await {
            let _ = writeln!(io::stderr(), "{err}");
            continue;
        }
        if let Err(err) = resync_after_run(&mut session, &coordinator).await {
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
        AgentContext, AssistantMessage, ContentBlock, LlmContext, Model, StopReason, ThinkingLevel,
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
    /// Manual `/compact` path through the compaction coordinator. Tests
    /// construct `CompactionSettings` directly (not `from_env`) so they are
    /// hermetic against process-env leakage.
    mod compact {
        use super::*;
        use aaos_session::compaction::{
            DEFAULT_KEEP_RECENT_TOKENS, DEFAULT_RESERVE_TOKENS, TRANSCRIPT_PREAMBLE,
        };
        use aaos_session::{
            AgentSession, AssistantSegment, ContentBlock as StoreBlock, Segment, SessionStore,
            StopReason as StoreStopReason, ToolCall as StoreToolCall, Usage as StoreUsage,
        };
        use pi_agent_core::agent::Agent;
        use pi_agent_core::stream::simple_text_response;
        use pi_agent_core::types::Message;

        use super::compaction_coordinator::{
            CompactionCoordinator, CompactionError, CompactionSettings,
        };

        fn test_model() -> Model {
            Model {
                id: "test".into(),
                ..Model::unknown()
            }
        }

        fn first_text(msg: &Message) -> String {
            let content = match msg {
                Message::User(u) => &u.content,
                Message::Assistant(a) => &a.content,
                Message::ToolResult(t) => &t.content,
            };
            content
                .iter()
                .find_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.clone()),
                    _ => None,
                })
                .unwrap_or_default()
        }

        /// Seed `n` user/assistant text turns, each ~`chars` chars long.
        async fn seed_turns(store: &SessionStore, root: &str, n: usize, chars: usize) {
            for i in 0..n {
                store
                    .append_segment(
                        root,
                        &Segment::user_text(format!("u{i}-{}", "x".repeat(chars))),
                    )
                    .await
                    .unwrap();
                store
                    .append_segment(
                        root,
                        &Segment::assistant_text(format!("a{i}-{}", "y".repeat(chars))),
                    )
                    .await
                    .unwrap();
            }
        }

        /// Seed one tool round-trip: an assistant tool call + a `result_chars`-
        /// long tool result. Tool results are the context bulk compaction
        /// replaces with a path, so they make the projection strictly smaller.
        async fn seed_tool_turn(
            store: &SessionStore,
            root: &str,
            call_id: &str,
            name: &str,
            args: serde_json::Value,
            result_chars: usize,
        ) {
            store
                .append_segment(
                    root,
                    &Segment::Assistant(AssistantSegment {
                        content: vec![StoreBlock::ToolCall(StoreToolCall {
                            id: call_id.into(),
                            name: name.into(),
                            arguments: args,
                        })],
                        stop_reason: StoreStopReason::ToolUse,
                        model: "test".into(),
                        provider: "test".into(),
                        api: "test".into(),
                        usage: StoreUsage::default(),
                        error_message: None,
                    }),
                )
                .await
                .unwrap();
            store
                .append_segment(
                    root,
                    &Segment::tool_result_text(call_id, "R".repeat(result_chars)),
                )
                .await
                .unwrap();
        }

        /// Absolute object paths referenced by a transcript's path lines:
        /// the `[Tool result] {path}` payload, `[Tool call] … — full
        /// arguments at {path}`, and `[Image] at {path}` (block-granular
        /// objects, ADR-0006).
        fn transcript_paths(transcript: &str) -> Vec<&str> {
            transcript
                .lines()
                .filter_map(|line| {
                    let path = match line.strip_prefix("[Tool result] ") {
                        Some(rest) if rest != "(empty)" => rest,
                        _ => line.rsplit_once(" at ").map(|(_, path)| path.trim())?,
                    };
                    Some(path.trim()).filter(|path| path.starts_with('/'))
                })
                .collect()
        }

        /// (a) Happy path: enough content → compact creates a node whose
        /// summary segment is a deterministic transcript (seeded texts inline,
        /// tool results replaced by absolute paths that resolve), with model None (deterministic, no
        /// generating model; ADR-0006 structural provenance); resuming onto the compacted node
        /// yields the transcript user-message + retained tail.
        #[tokio::test]
        async fn happy_path_creates_node_and_resumes() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            // Tool round-trip FIRST (its 4000-char result is the context bulk
            // the compaction replaces with a path), then 5 text turns; the cut
            // lands inside the text turns, so the result is compacted away.
            seed_tool_turn(
                &store,
                &root,
                "c1",
                "read",
                json!({"path": "/tmp/note.txt"}),
                4000,
            )
            .await;
            seed_turns(&store, &root, 5, 100).await;

            let settings = CompactionSettings {
                enabled: true,
                reserve_tokens: 16_384,
                keep_recent_tokens: 60,
            };
            let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

            let outcome = coordinator.compact(&root).await.expect("compact ok");
            assert_ne!(outcome.compacted_id, root);
            assert!(
                outcome.before_tokens > outcome.after_tokens,
                "{} -> {}",
                outcome.before_tokens,
                outcome.after_tokens
            );

            // Transcript content: preamble, seeded texts inline, tool result
            // at the absolute path, tool call inline.
            let summary = &first_text(&outcome.injected_view[0]);
            assert!(summary.starts_with(TRANSCRIPT_PREAMBLE), "{summary}");
            assert!(summary.contains("[User] u0-"), "{summary}");
            assert!(summary.contains("[Assistant] a0-"), "{summary}");
            let paths = transcript_paths(summary);
            let result_path = summary
                .lines()
                .find_map(|line| line.strip_prefix("[Tool result] "))
                .expect("transcript references the result object");
            assert!(
                summary.contains(&format!("[Tool result] {result_path}")),
                "{summary}"
            );
            assert!(
                summary.contains(r#"[Tool call] read({"path":"/tmp/note.txt"})"#),
                "{summary}"
            );
            assert!(!summary.contains("RRRR"), "result text is not inlined");
            assert!(
                !summary.contains("<summary>"),
                "no pi-style summary wrapper"
            );

            // Every referenced object path resolves on disk; the result
            // object holds the raw output bytes (block-granular, ADR-0006).
            assert!(
                !paths.is_empty(),
                "transcript references at least one object"
            );
            for path in &paths {
                assert!(std::fs::exists(path).unwrap(), "object exists: {path}");
            }
            assert_eq!(
                std::fs::read_to_string(result_path).unwrap(),
                "R".repeat(4000),
                "result object holds the raw output bytes"
            );

            // Summary segment persisted: provenance is
            // structural (ADR-0006): `fetch_originals` covers the prefix.
            let view = store
                .materialize_plain(&outcome.compacted_id)
                .await
                .unwrap();
            let Segment::Summary(s) = &view[0] else {
                panic!("first segment must be a summary");
            };
            assert_eq!(s.content, *summary);
            let originals = store.fetch_originals(&outcome.compacted_id).await.unwrap();
            assert_eq!(originals.len(), 1, "one compaction map");
            assert!(
                originals[0].originals.len() > 1,
                "the covered prefix is retrievable: {:?}",
                originals[0]
            );

            // Resume onto the compacted node: transcript user-message + tail.
            let mut session = AgentSession::new(
                store.clone(),
                Agent::new(simple_text_response("ok")),
                &root,
                dir.path(),
            );
            session.resume(&outcome.compacted_id).await.unwrap();
            let messages = &session.state().messages;
            // Summary renders as a bare user message: exactly the summary
            // content (no provenance prefix — the preamble is
            // self-describing).
            assert_eq!(
                first_text(&messages[0]),
                *summary,
                "summary renders bare as a user message"
            );
            assert!(first_text(&messages[0]).contains(TRANSCRIPT_PREAMBLE));
            assert!(
                messages.len() < 12,
                "tail must be shorter than the full transcript"
            );
            assert_eq!(
                session.current_session_id().await,
                outcome.compacted_id,
                "resume switched the append target"
            );
            // Head unchanged: compaction derives but does not append.
            assert_eq!(store.head().await.unwrap().as_deref(), Some(root.as_str()));
        }

        /// (b) Nothing to compact: a tiny session refuses without creating a node.
        #[tokio::test]
        async fn tiny_session_refuses_nothing_to_compact() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            store
                .append_segment(&root, &Segment::user_text("hi"))
                .await
                .unwrap();

            let settings = CompactionSettings::default();
            let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

            let err = coordinator.compact(&root).await.unwrap_err();
            assert_eq!(err, CompactionError::NothingToCompact);
            // No node created: root's view is unchanged, head unchanged.
            assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 1);
            assert_eq!(store.head().await.unwrap().as_deref(), Some(root.as_str()));
        }

        /// (c) Re-compaction: a compacted node compacts again naturally. The
        /// second node's transcript embeds the first transcript's text, the
        /// first node's object paths still resolve, and the view is correct.
        #[tokio::test]
        async fn recompaction_embeds_previous_transcript() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            seed_tool_turn(&store, &root, "c1", "read", json!({"path": "/a"}), 4000).await;
            seed_turns(&store, &root, 4, 100).await;

            let settings = CompactionSettings {
                enabled: true,
                reserve_tokens: 16_384,
                keep_recent_tokens: 60,
            };
            let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

            let first = coordinator.compact(&root).await.expect("first compact ok");
            // Keep compacting on the compacted node.
            let first_view = store.materialize_plain(&first.compacted_id).await.unwrap();
            let Segment::Summary(first_summary) = &first_view[0] else {
                panic!("first node starts with a summary");
            };
            let first_transcript = first_summary.content.clone();
            let first_paths: Vec<String> = transcript_paths(&first_transcript)
                .iter()
                .map(|p| p.to_string())
                .collect();
            assert!(
                !first_paths.is_empty(),
                "first transcript references object paths"
            );

            // Append a second tool round-trip and enough turns after it that
            // the second cut lands in the text tail and the new result is
            // compacted away; the old transcript is embedded verbatim.
            seed_tool_turn(
                &store,
                &first.compacted_id,
                "c2",
                "bash",
                json!({"command": "ls"}),
                4000,
            )
            .await;
            seed_turns(&store, &first.compacted_id, 4, 100).await;
            let second = coordinator
                .compact(&first.compacted_id)
                .await
                .expect("second compact ok");
            assert_ne!(second.compacted_id, first.compacted_id);

            // Second node: the old transcript is embedded verbatim (it is
            // already a transcript with paths — transitive), and the old
            // object paths still resolve.
            let second_view = store.materialize_plain(&second.compacted_id).await.unwrap();
            let Segment::Summary(second_summary) = &second_view[0] else {
                panic!("second node starts with a summary");
            };
            assert!(
                second_summary.content.contains(&first_transcript),
                "first transcript embedded: {}",
                second_summary.content
            );
            assert!(
                second_summary.content.contains("[User] u0-"),
                "new turns rendered inline: {}",
                second_summary.content
            );
            for path in &first_paths {
                assert!(
                    std::fs::exists(path).unwrap(),
                    "old object path still resolves: {path}"
                );
            }
            for path in transcript_paths(&second_summary.content) {
                assert!(
                    std::fs::exists(path).unwrap(),
                    "second transcript's paths resolve: {path}"
                );
            }

            // Resume onto the second node: transcript + retained tail.
            let mut session = AgentSession::new(
                store.clone(),
                Agent::new(simple_text_response("ok")),
                &root,
                dir.path(),
            );
            session.resume(&second.compacted_id).await.unwrap();
            let messages = &session.state().messages;
            // Summary renders as a bare user message (no provenance prefix).
            assert_eq!(
                first_text(&messages[0]),
                second_summary.content,
                "second summary renders bare as a user message"
            );
            assert!(
                first_text(&messages[0]).contains(TRANSCRIPT_PREAMBLE),
                "second summary keeps the preamble"
            );
            // Only the embedded first transcript may mention the old prefix;
            // the retained tail must not.
            assert!(
                messages[1..].iter().all(|m| !first_text(m).contains("u0-")),
                "old prefix not in the tail"
            );
            assert_eq!(
                session.current_session_id().await,
                second.compacted_id,
                "resume switched to the second node"
            );
        }

        /// (d) Degenerate projection: a transcript larger than the prefix it
        /// replaces is rejected — no node is created. Constructed with a
        /// tool-free dialogue where the transcript preamble dominates.
        #[tokio::test]
        async fn degenerate_projection_rejected() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            // Text-only dialogue: no tool results to shed, so the transcript
            // (preamble + role labels) is never smaller than the prefix it
            // replaces.
            seed_turns(&store, &root, 6, 100).await;

            let settings = CompactionSettings {
                enabled: true,
                reserve_tokens: 16_384,
                keep_recent_tokens: 60,
            };
            let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

            let err = coordinator.compact(&root).await.unwrap_err();
            match &err {
                CompactionError::Failed(m) => {
                    assert!(m.contains("reduce"), "expected reduce, got {m}");
                }
                other => panic!("expected Failed, got {other:?}"),
            }
            assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 12);
        }

        /// Stale usage anchors: a pre-compaction assistant in the retained
        /// tail carries a `usage.total_tokens` describing the PRE-compaction
        /// context. The projected view contains a summary by construction,
        /// so its assistant usage must be zeroed before metering —
        /// `after_tokens` falls back to pure estimation instead of
        /// re-anchoring on the stale total, which refuses every compaction
        /// with "would not reduce context".
        #[tokio::test]
        async fn stale_tail_anchor_does_not_refuse_compaction() {
            let dir = tempfile::tempdir().unwrap();
            let store = SessionStore::open(dir.path()).await.unwrap();
            let root = store.create_root().await.unwrap();
            // Tool round-trip for the compactable bulk, then text turns; the
            // LAST assistant carries a large pre-compaction usage total (the
            // normal end of a live turn) and lands in the retained tail.
            seed_tool_turn(&store, &root, "c1", "read", json!({"path": "/a"}), 4000).await;
            seed_turns(&store, &root, 4, 100).await;
            store
                .append_segment(
                    &root,
                    &Segment::Assistant(AssistantSegment {
                        content: vec![StoreBlock::Text {
                            text: "stale anchor".into(),
                        }],
                        stop_reason: StoreStopReason::Stop,
                        model: "test".into(),
                        provider: "test".into(),
                        api: "test".into(),
                        usage: StoreUsage {
                            total_tokens: 100_000,
                            ..StoreUsage::default()
                        },
                        error_message: None,
                    }),
                )
                .await
                .unwrap();

            let settings = CompactionSettings {
                enabled: true,
                reserve_tokens: 16_384,
                keep_recent_tokens: 60,
            };
            let coordinator = CompactionCoordinator::new(store.clone(), settings, &test_model());

            let outcome = coordinator
                .compact(&root)
                .await
                .expect("compaction must not be refused by the stale tail anchor");
            assert!(
                outcome.after_tokens < outcome.before_tokens,
                "{} -> {}",
                outcome.before_tokens,
                outcome.after_tokens
            );
            // The injected view carries a summary, so no assistant in it may
            // hold a pre-compaction usage anchor.
            for message in &outcome.injected_view {
                if let Message::Assistant(a) = message {
                    assert_eq!(
                        a.usage.total_tokens, 0,
                        "injected assistant must carry zero usage"
                    );
                }
            }
        }

        /// Auto-trigger hooks: transform_context + prepare_next_turn wired
        /// through `install_compaction_hooks`. All settings constructed
        /// directly (never from_env) so tests are hermetic.
        mod hooks {
            use super::*;

            /// A session agent with a recording fake stream and compaction hooks
            /// installed, bound to a seeded session.
            async fn hooked_session(
                dir: &tempfile::TempDir,
                store: &SessionStore,
                root: &str,
                settings: CompactionSettings,
                model: &Model,
                record: Arc<Mutex<Vec<String>>>,
            ) -> (AgentSession, Arc<CompactionCoordinator>, Arc<Mutex<usize>>) {
                let llm_calls = Arc::new(Mutex::new(0usize));
                let llm_calls_for_stream = llm_calls.clone();
                let record_for_stream = record.clone();
                let stream_fn = mock_stream_fn(move |_model, ctx, _opts| {
                    *llm_calls_for_stream.lock().unwrap() += 1;
                    let texts: Vec<String> = ctx.messages.iter().map(first_text).collect();
                    record_for_stream
                        .lock()
                        .unwrap()
                        .push(texts.join("\n---\n"));
                    Box::new(pi_agent_core::stream::MockAssistantStream::new(
                        AssistantMessage::text("ok"),
                    ))
                });
                let mut agent = Agent::new(stream_fn);
                agent.state.model = model.clone();
                agent.state.system_prompt = "sys".to_string();
                agent.stream_fn_options.api_key = None;
                let mut session =
                    AgentSession::new(store.clone(), agent, root.to_string(), dir.path());
                // Load the seeded transcript so the prompt's context is the full
                // conversation (which is what the hooks see and measure).
                session.resume(root).await.unwrap();
                let coordinator =
                    Arc::new(CompactionCoordinator::new(store.clone(), settings, model));
                let node_handle = session.session_id_lock();
                install_compaction_hooks(session.agent_mut(), &coordinator, node_handle);
                (session, coordinator, llm_calls)
            }

            /// (a) transform_context threshold trigger: a big seeded session
            /// compacts during prompt(); the request the fake stream receives is
            /// the transcript message, not the full old prefix; after the run
            /// the session resyncs onto the compacted node; appends land on
            /// the compacted node.
            #[tokio::test]
            async fn transform_context_threshold_compacts_and_resyncs() {
                let dir = tempfile::tempdir().unwrap();
                let store = SessionStore::open(dir.path()).await.unwrap();
                let root = store.create_root().await.unwrap();
                // ~1100 tokens total with a 4000-char tool result in the prefix.
                seed_tool_turn(&store, &root, "c1", "read", json!({"path": "/a"}), 4000).await;
                seed_turns(&store, &root, 6, 100).await;

                let model = Model {
                    id: "test".into(),
                    context_window: 100,
                    ..Model::unknown()
                };
                let settings = CompactionSettings {
                    enabled: true,
                    reserve_tokens: 50, // threshold: 100-50 = 50 → way above
                    keep_recent_tokens: 60,
                };
                let record: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
                let (mut session, coordinator, llm_calls) =
                    hooked_session(&dir, &store, &root, settings, &model, record.clone()).await;

                coordinator.begin_run();
                session.agent_mut().prompt("hello").await.unwrap();
                let resynced = coordinator.take_pending_resync();
                if let Some(id) = resynced {
                    session.resume(&id).await.unwrap();
                }

                // The request the fake stream saw: transcript message, not the
                // old prefix. (Record locked in a scoped block so no guard
                // crosses the store awaits below.)
                {
                    let calls = record.lock().unwrap();
                    assert_eq!(calls.len(), 1, "one session request");
                    assert!(
                        calls[0].contains(TRANSCRIPT_PREAMBLE),
                        "transcript in request: {}",
                        calls[0]
                    );
                    assert!(
                        !calls[0].contains(&"R".repeat(4000)),
                        "tool-result bulk must be gone from the request: {}",
                        calls[0]
                    );
                }

                // Session resynced onto the compacted node: state = transcript + tail.
                let messages = &session.state().messages;
                assert!(
                    first_text(&messages[0]).contains(TRANSCRIPT_PREAMBLE),
                    "first message is the transcript"
                );
                assert!(
                    messages
                        .iter()
                        .all(|m| !first_text(m).contains(&"R".repeat(4000))),
                    "no tool-result bulk in state"
                );
                let last = messages.last().unwrap();
                assert_eq!(
                    first_text(last),
                    "ok",
                    "the run's assistant landed in state"
                );

                // Appends landed on the compacted node: head moved to it.
                let head = store.head().await.unwrap().unwrap();
                assert_ne!(head, root, "head moved off root");
                let view = store.materialize_plain(&head).await.unwrap();
                let Segment::Summary(s) = &view[0] else {
                    panic!("compacted node starts with a summary");
                };
                assert!(s.content.starts_with(TRANSCRIPT_PREAMBLE));
                assert!(*llm_calls.lock().unwrap() >= 1);
            }

            /// (b) no double compaction: threshold still exceeded after the
            /// compaction → only one compact node per run.
            #[tokio::test]
            async fn no_double_compaction_per_run() {
                let dir = tempfile::tempdir().unwrap();
                let store = SessionStore::open(dir.path()).await.unwrap();
                let root = store.create_root().await.unwrap();
                // Tool result in the compacted prefix; the retained tail (4
                // turns ≈ 100 tokens) still exceeds the threshold after.
                seed_tool_turn(&store, &root, "c1", "bash", json!({"command": "ls"}), 4000).await;
                seed_turns(&store, &root, 12, 100).await; // ~300 tokens

                let model = Model {
                    id: "test".into(),
                    context_window: 100,
                    ..Model::unknown()
                };
                let settings = CompactionSettings {
                    enabled: true,
                    reserve_tokens: 50,      // threshold 50
                    keep_recent_tokens: 100, // retained tail ~100 tokens > 50
                };
                let record: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
                let (mut session, coordinator, _) =
                    hooked_session(&dir, &store, &root, settings, &model, record.clone()).await;

                coordinator.begin_run();
                session.agent_mut().prompt("hello").await.unwrap();
                if let Some(id) = coordinator.take_pending_resync() {
                    session.resume(&id).await.unwrap();
                }

                // Exactly one compact derivation along the chain.
                let head = store.head().await.unwrap().unwrap();
                assert_ne!(head, root);
                let originals = store.fetch_originals(&head).await.unwrap();
                assert_eq!(originals.len(), 1, "one compaction: {originals:?}");
                let view = store.materialize_plain(&head).await.unwrap();
                let summaries = view
                    .iter()
                    .filter(|seg| matches!(seg, Segment::Summary(_)))
                    .count();
                assert_eq!(summaries, 1, "exactly one summary segment");
            }

            /// context window triggers one compaction; a second overflow in the
            /// same run fails it (the run ends with an Error stop reason).
            #[tokio::test]
            async fn overflow_recovery_once_then_fails() {
                let dir = tempfile::tempdir().unwrap();
                let store = SessionStore::open(dir.path()).await.unwrap();
                let root = store.create_root().await.unwrap();
                // Small session — below threshold, so no pre-request compaction.
                seed_tool_turn(&store, &root, "c1", "bash", json!({"command": "ls"}), 4000).await;
                seed_turns(&store, &root, 2, 100).await;

                let model = Model {
                    id: "test".into(),
                    context_window: 100,
                    ..Model::unknown()
                };
                let settings = CompactionSettings {
                    enabled: true,
                    reserve_tokens: 50,
                    keep_recent_tokens: 60,
                };
                // Scripted session stream: turn 1 requests an unknown tool (so a
                // second turn happens) with a huge usage; turn 2 silently
                // overflows again.
                let calls = Arc::new(Mutex::new(0usize));
                let calls_for_stream = calls.clone();
                let session_stream = mock_stream_fn(move |_model, _ctx, _opts| {
                    let mut count = calls_for_stream.lock().unwrap();
                    let call = *count;
                    *count += 1;
                    let msg = if call == 0 {
                        AssistantMessage {
                            content: vec![ContentBlock::tool_call("c1", "absent_tool", json!({}))],
                            stop_reason: StopReason::ToolUse,
                            usage: pi_agent_core::types::Usage {
                                input: 10_000,
                                ..Default::default()
                            },
                            ..Default::default()
                        }
                    } else {
                        AssistantMessage {
                            content: vec![ContentBlock::text("big response")],
                            usage: pi_agent_core::types::Usage {
                                input: 10_000,
                                ..Default::default()
                            },
                            ..Default::default()
                        }
                    };
                    Box::new(pi_agent_core::stream::MockAssistantStream::new(msg))
                });
                let mut agent = Agent::new(session_stream);
                agent.state.model = model.clone();
                agent.state.system_prompt = "sys".to_string();
                agent.stream_fn_options.api_key = None;
                let mut session = AgentSession::new(store.clone(), agent, root.clone(), dir.path());
                session.resume(&root).await.unwrap();
                let coordinator =
                    Arc::new(CompactionCoordinator::new(store.clone(), settings, &model));
                let node_handle = session.session_id_lock();
                install_compaction_hooks(session.agent_mut(), &coordinator, node_handle);

                // One run: first overflow compacts, second overflow fails the run.
                coordinator.begin_run();
                session.agent_mut().prompt("hello").await.unwrap();
                assert_eq!(*calls.lock().unwrap(), 2, "two turns happened");
                let state = session.state();
                let (stop_reason, error_message) = turn_outcome(state);
                assert_eq!(stop_reason, Some(StopReason::Error));
                assert!(
                    error_message
                        .as_deref()
                        .unwrap_or_default()
                        .contains("overflow"),
                    "overflow surfaced: {error_message:?}"
                );
                // One compaction happened during the run.
                let head = store.head().await.unwrap().unwrap();
                assert_ne!(head, root);
                let view = store.materialize_plain(&head).await.unwrap();
                assert!(matches!(view[0], Segment::Summary(_)));
            }

            /// (f) Overflow recovery excludes the failed/truncated assistant
            /// from the retry context (issue #70 §3.5) while the store keeps
            /// it in session history. Driven through `post_turn_hook`
            /// directly: an Error-stop assistant ends the run, so the retry
            /// context is the hook's injected view, consumed by the caller.
            #[tokio::test]
            async fn overflow_recovery_strips_failed_assistant_from_retry_context() {
                let dir = tempfile::tempdir().unwrap();
                let store = SessionStore::open(dir.path()).await.unwrap();
                let root = store.create_root().await.unwrap();
                // Seed enough context that the recovery compaction succeeds,
                // and persist the failed/truncated assistant as the last
                // segment of the session history (the message the overflow
                // branch strips from the injected view).
                seed_tool_turn(&store, &root, "c1", "bash", json!({"command": "ls"}), 4000).await;
                seed_turns(&store, &root, 5, 100).await;
                let failed_text = "FAILED MESSAGE TEXT";
                store
                    .append_segment(
                        &root,
                        &Segment::Assistant(AssistantSegment {
                            content: vec![StoreBlock::Text {
                                text: failed_text.into(),
                            }],
                            stop_reason: StoreStopReason::Error,
                            model: "test".into(),
                            provider: "test".into(),
                            api: "test".into(),
                            usage: StoreUsage::default(),
                            error_message: Some(
                                "provider: prompt is too long for context window".into(),
                            ),
                        }),
                    )
                    .await
                    .unwrap();

                let model = Model {
                    id: "test".into(),
                    context_window: 100,
                    ..Model::unknown()
                };
                let settings = CompactionSettings {
                    enabled: true,
                    reserve_tokens: 50,
                    keep_recent_tokens: 60,
                };
                let coordinator = CompactionCoordinator::new(store.clone(), settings, &model);
                coordinator.begin_run();

                let failed = AssistantMessage {
                    content: vec![ContentBlock::text(failed_text)],
                    stop_reason: StopReason::Error,
                    error_message: Some("provider: prompt is too long for context window".into()),
                    ..Default::default()
                };
                let context = AgentContext {
                    system_prompt: "sys".into(),
                    messages: vec![],
                    tools: vec![],
                };
                let outcome = coordinator
                    .post_turn_hook(&failed, &context, &root)
                    .await
                    .expect("overflow recovery succeeds")
                    .expect("a compaction committed");

                // Injected view: transcript first, failed assistant excluded.
                assert!(
                    first_text(&outcome.injected_view[0]).contains(TRANSCRIPT_PREAMBLE),
                    "injected view starts with the transcript: {:?}",
                    outcome.injected_view[0]
                );
                assert!(
                    !outcome
                        .injected_view
                        .iter()
                        .any(|m| first_text(m) == failed_text),
                    "failed assistant excluded from the injected view: {:?}",
                    outcome.injected_view
                );

                // The failed message is still persisted in session history
                // (compact derives; the segment stays in the store).
                let head = store.head().await.unwrap().unwrap();
                let view = store.materialize_plain(&head).await.unwrap();
                assert!(
                    view.iter().any(|seg| matches!(
                        seg,
                        Segment::Assistant(a)
                            if a.stop_reason == StoreStopReason::Error
                                && a.content.iter().any(|b| matches!(
                                    b,
                                    StoreBlock::Text { text }
                                        if text == failed_text
                                ))
                    )),
                    "failed assistant stays in the store"
                );
            }

            /// (d) disabled via settings: no auto compaction fires; the full
            /// context goes to the model and no compact node is created.
            #[tokio::test]
            async fn disabled_no_auto_compaction() {
                let dir = tempfile::tempdir().unwrap();
                let store = SessionStore::open(dir.path()).await.unwrap();
                let root = store.create_root().await.unwrap();
                seed_turns(&store, &root, 6, 200).await;

                let model = Model {
                    id: "test".into(),
                    context_window: 100,
                    ..Model::unknown()
                };
                let settings = CompactionSettings {
                    enabled: false,
                    reserve_tokens: 50,
                    keep_recent_tokens: 60,
                };
                let record: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
                let (mut session, coordinator, _) =
                    hooked_session(&dir, &store, &root, settings, &model, record.clone()).await;

                coordinator.begin_run();
                session.agent_mut().prompt("hello").await.unwrap();

                // No pending resync, no compact node.
                assert!(coordinator.take_pending_resync().is_none());
                let head = store.head().await.unwrap().unwrap();
                assert_eq!(head, root, "head stays on root when disabled");
                // The request carried the full original prefix.
                let calls = record.lock().unwrap();
                assert!(calls[0].contains("u0-"), "full context sent: {}", calls[0]);
                assert!(
                    !calls[0].contains(TRANSCRIPT_PREAMBLE),
                    "no transcript injected"
                );
            }

            /// (e) AAOS_COMPACTION_ENABLED=0 does not block manual /compact —
            /// `enabled=false` settings still compact on explicit request.
            #[tokio::test]
            async fn manual_compact_ignores_disabled() {
                let dir = tempfile::tempdir().unwrap();
                let store = SessionStore::open(dir.path()).await.unwrap();
                let root = store.create_root().await.unwrap();
                seed_tool_turn(&store, &root, "c1", "read", json!({"path": "/a"}), 4000).await;
                seed_turns(&store, &root, 5, 100).await;

                let settings = CompactionSettings {
                    enabled: false,
                    reserve_tokens: 50,
                    keep_recent_tokens: 60,
                };
                let coordinator =
                    CompactionCoordinator::new(store.clone(), settings, &test_model());
                let outcome = coordinator.compact(&root).await.expect("manual compact ok");
                assert_ne!(outcome.compacted_id, root);
                let view = store
                    .materialize_plain(&outcome.compacted_id)
                    .await
                    .unwrap();
                let Segment::Summary(s) = &view[0] else {
                    panic!("manual compact created a summary");
                };
                assert!(s.content.starts_with(TRANSCRIPT_PREAMBLE));
            }

            /// `CompactionSettings` env parsing: defaults when unset, overrides
            /// parse, invalid values fall back to defaults. Exercises the
            /// pure `from_env_values` core — the workspace denies
            /// `unsafe_code`, so tests cannot mutate process env.
            #[test]
            fn from_env_defaults_overrides_and_fallbacks() {
                let defaults = CompactionSettings::from_env_values(None, None, None);
                assert!(defaults.enabled);
                assert_eq!(defaults.reserve_tokens, DEFAULT_RESERVE_TOKENS);
                assert_eq!(defaults.keep_recent_tokens, DEFAULT_KEEP_RECENT_TOKENS);

                let parsed =
                    CompactionSettings::from_env_values(Some("false"), Some("4096"), Some("12345"));
                assert!(!parsed.enabled, "false disables");
                assert_eq!(parsed.reserve_tokens, 4096);
                assert_eq!(parsed.keep_recent_tokens, 12345);

                // Invalid values fall back to defaults; unrecognized strings
                // count as enabled.
                let fallback =
                    CompactionSettings::from_env_values(Some("bogus"), Some("nan"), Some("-7"));
                assert!(fallback.enabled);
                assert_eq!(fallback.reserve_tokens, DEFAULT_RESERVE_TOKENS);
                assert_eq!(fallback.keep_recent_tokens, DEFAULT_KEEP_RECENT_TOKENS);

                // Disabled strings: "0", "false", "no" (case-insensitive).
                for v in ["0", "FALSE", "No"] {
                    assert!(
                        !CompactionSettings::from_env_values(Some(v), None, None).enabled,
                        "{v}"
                    );
                }
            }
        }
    }
}
