//! Thin product-layer composition around the kernel [`Agent`].
//!
//! Session cwd is turned into coding tools and a system prompt; catalog and
//! OpenAI stay injected as [`Model`] + [`StreamFn`].

use std::path::PathBuf;
use std::sync::Arc;

use aaos_tools::{build_system_prompt, create_coding_tools};
use pi_agent_core::agent::{Agent, AgentError, AgentHandle, Listener};
use pi_agent_core::types::{Model, StreamFn, ThinkingLevel};

/// Inputs needed to construct an [`AgentSession`].
pub struct SessionOptions {
    pub cwd: PathBuf,
    pub model: Model,
    pub stream_fn: Arc<dyn StreamFn>,
    pub thinking_level: ThinkingLevel,
    pub api_key: Option<String>,
}

/// Owns a kernel [`Agent`] wired with this product's default coding tools.
pub struct AgentSession {
    agent: Agent,
}

impl AgentSession {
    pub fn new(opts: SessionOptions) -> Self {
        let tools = create_coding_tools(&opts.cwd);
        let system_prompt = build_system_prompt(&opts.cwd, &tools);
        let mut agent = Agent::new(opts.stream_fn);
        agent.state.model = opts.model;
        agent.state.thinking_level = opts.thinking_level;
        agent.state.tools = tools;
        agent.state.system_prompt = system_prompt;
        agent.stream_fn_options.api_key = opts.api_key;
        Self { agent }
    }

    pub fn subscribe(&self, listener: Listener) -> impl FnOnce() {
        self.agent.subscribe(listener)
    }

    pub async fn prompt(&mut self, text: impl Into<String>) -> Result<(), AgentError> {
        self.agent.prompt(text.into()).await
    }

    pub fn abort(&self) {
        self.agent.abort();
    }

    pub fn handle(&self) -> AgentHandle {
        self.agent.handle()
    }

    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    pub fn agent_mut(&mut self) -> &mut Agent {
        &mut self.agent
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use serde_json::json;

    use pi_agent_core::stream::{mock_stream_fn, MockAssistantStream};
    use pi_agent_core::types::{
        AssistantMessage, ContentBlock, LlmContext, Model, StopReason, ThinkingLevel,
    };

    use super::*;

    #[tokio::test]
    async fn prompt_runs_read_tool_and_sends_schema() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let tmp = tempfile::TempDir::new().unwrap();
        std::fs::write(tmp.path().join("note.txt"), "hello from file").unwrap();
        let captured: Arc<Mutex<Option<LlmContext>>> = Arc::new(Mutex::new(None));
        let cap = captured.clone();
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = n.clone();
        let stream_fn = mock_stream_fn(move |_model, ctx, _opts| {
            let i = n2.fetch_add(1, Ordering::SeqCst);
            if i == 0 {
                *cap.lock().unwrap() = Some(ctx);
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
        let mut session = AgentSession::new(SessionOptions {
            cwd: tmp.path().to_path_buf(),
            model: Model {
                id: "t".into(),
                ..Model::unknown()
            },
            stream_fn,
            thinking_level: ThinkingLevel::Off,
            api_key: None,
        });
        session.prompt("read the note").await.unwrap();
        let ctx = captured.lock().unwrap().clone().expect("first llm call");
        let names: Vec<_> = ctx.tools.iter().map(|t| t.name().to_string()).collect();
        assert_eq!(names, ["read", "bash", "edit", "write"]);
        let read = ctx.tools.iter().find(|t| t.name() == "read").unwrap();
        assert_eq!(read.parameters()["required"], json!(["path"]));
        assert!(ctx.system_prompt.contains("Available tools:"));
        assert!(
            ctx.system_prompt
                .contains(&tmp.path().display().to_string().replace('\\', "/"))
                || ctx.system_prompt.contains("Current working directory:")
        );
        let tool_text: String = session
            .agent()
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
        assert!(n.load(Ordering::SeqCst) >= 2);
    }
}
