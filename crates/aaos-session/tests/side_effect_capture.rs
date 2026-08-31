//! Spec step 6 — side-effect capture through the integration layer.
//!
//! A `write` tool call (file pre-existing → before bytes) and a `bash` tool
//! call are run through `AgentSession`; the `side_effects` table must hold:
//! - write: before + after bytes, path = the file path
//! - bash: no before/after, path = the command
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::sync::Arc;

use aaos_session::{AgentSession, SessionStore};
use aaos_tools::{SkillIndex, build_system_prompt, create_coding_tools};
use pi_agent_core::stream::{MockAssistantStream, mock_stream_fn};
use pi_agent_core::types::{
    AssistantMessage, ContentBlock, Model, StopReason, StreamFn, ThinkingLevel,
};
use serde_json::json;

fn make_agent(stream_fn: Arc<dyn StreamFn>, cwd: &std::path::Path) -> pi_agent_core::agent::Agent {
    let skills = Arc::new(SkillIndex::discover(
        std::path::Path::new("/nonexistent"),
        &cwd.join(".agents/skills"),
    ));
    let tools = create_coding_tools(cwd, skills.clone());
    let system_prompt = build_system_prompt(cwd, &tools, &skills);
    let mut agent = pi_agent_core::agent::Agent::new(stream_fn);
    agent.state.model = Model {
        id: "test".into(),
        ..Model::unknown()
    };
    agent.state.thinking_level = ThinkingLevel::Off;
    agent.state.tools = tools;
    agent.state.system_prompt = system_prompt;
    agent.stream_fn_options.api_key = None;
    agent
}

/// Build a mock stream that emits the given tool calls in sequence, then "done".
fn stream_with_calls(specs: &[(&str, &str, serde_json::Value)]) -> Arc<dyn StreamFn> {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let call = Arc::new(AtomicUsize::new(0));
    let specs: Vec<(String, String, serde_json::Value)> = specs
        .iter()
        .map(|(id, tool, args)| (id.to_string(), tool.to_string(), args.clone()))
        .collect();
    mock_stream_fn(move |_model, _ctx, _opts| {
        let n = call.fetch_add(1, Ordering::SeqCst);
        if n < specs.len() {
            let (id, tool, args) = &specs[n];
            let msg = AssistantMessage {
                content: vec![ContentBlock::tool_call(
                    id.as_str(),
                    tool.as_str(),
                    args.clone(),
                )],
                stop_reason: StopReason::ToolUse,
                ..Default::default()
            };
            Box::new(MockAssistantStream::new(msg))
        } else {
            Box::new(MockAssistantStream::new(AssistantMessage::text("done")))
        }
    })
}

/// Set up a session, run a prompt, and return the store + session_id for inspection.
async fn run_capture(
    cwd: &std::path::Path,
    stream_fn: Arc<dyn StreamFn>,
) -> (SessionStore, String) {
    let store = SessionStore::open(cwd).await.unwrap();
    let session_id = store.create_root().await.unwrap();
    let agent = make_agent(stream_fn, cwd);
    let mut session = AgentSession::new(store.clone(), agent, &session_id, cwd.to_path_buf());
    session.resume(&session_id).await.unwrap();
    session.agent_mut().prompt("go").await.unwrap();
    (store, session_id)
}

#[tokio::test]
async fn records_write_and_bash_side_effects() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    // Pre-existing file so the write tool captures before bytes.
    std::fs::write(cwd.join("target.txt"), "old content").unwrap();

    let stream_fn = stream_with_calls(&[
        (
            "c1",
            "write",
            json!({"path": "target.txt", "content": "new content"}),
        ),
        ("c2", "bash", json!({"command": "echo hi"})),
    ]);
    let (store, session_id) = run_capture(&cwd, stream_fn).await;

    let effects = store.side_effects(&session_id).await.unwrap();
    assert_eq!(effects.len(), 2, "one side effect per tool call");

    // write: before + after bytes, path = "target.txt"
    let write_effect = &effects[0];
    assert_eq!(write_effect.tool_call_id, "c1");
    assert_eq!(write_effect.path, "target.txt");
    assert!(
        write_effect.before_hash.is_some(),
        "write captures before bytes"
    );
    assert!(
        write_effect.after_hash.is_some(),
        "write captures after bytes"
    );

    // Resolve before/after payloads from the object store.
    let before = write_effect.before_hash.as_ref().unwrap();
    let after = write_effect.after_hash.as_ref().unwrap();
    assert_eq!(
        store.objects().get_bytes(before).await.unwrap(),
        b"old content",
        "before bytes = pre-write content"
    );
    assert_eq!(
        store.objects().get_bytes(after).await.unwrap(),
        b"new content",
        "after bytes = post-write content"
    );

    // bash: no before/after, path = the command
    let bash_effect = &effects[1];
    assert_eq!(bash_effect.tool_call_id, "c2");
    assert_eq!(bash_effect.path, "echo hi");
    assert!(
        bash_effect.before_hash.is_none(),
        "bash has no before bytes"
    );
    assert!(bash_effect.after_hash.is_none(), "bash has no after bytes");

    // seq is monotonic across the two side effects.
    assert!(write_effect.seq < bash_effect.seq);
}

#[tokio::test]
async fn new_file_write_records_no_before_bytes() {
    let dir = tempfile::tempdir().unwrap();
    let cwd = dir.path().to_path_buf();
    // No pre-existing file → before bytes = None (new file).

    let stream_fn = stream_with_calls(&[(
        "c1",
        "write",
        json!({"path": "fresh.txt", "content": "brand new"}),
    )]);
    let (store, session_id) = run_capture(&cwd, stream_fn).await;

    let effects = store.side_effects(&session_id).await.unwrap();
    assert_eq!(effects.len(), 1);
    let eff = &effects[0];
    assert_eq!(eff.path, "fresh.txt");
    assert!(eff.before_hash.is_none(), "new file has no before bytes");
    assert!(eff.after_hash.is_some(), "after bytes captured");
    assert_eq!(
        store
            .objects()
            .get_bytes(eff.after_hash.as_ref().unwrap())
            .await
            .unwrap(),
        b"brand new"
    );
}
