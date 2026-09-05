//! Ticket 70 — P1.6 contract tests for the compaction feature end to end:
//! `SessionStore::compact` + `AgentSession::resume` over the compacted node,
//! with the pure `compaction` module driving the cut point and building the
//! deterministic transcript.
//!
//! Contracts asserted (ADR-0006 edition — provenance is structural only):
//! 1. After `compact(parent, [(0, first_kept)], summary)`, the compacted
//!    node's materialized view = summary + retained tail segments.
//! 2. The transcript's referenced object paths resolve on disk and hold the
//!    raw block bytes (content track); `fetch_originals` returns the covered
//!    prefix (structural track — `SummarySegment` carries no sources).
//! 3. Undo: forking the parent yields a view with no Summary segment.
//! 4. Head unchanged by compact.
//! 5. `AgentSession::resume` onto the compacted node replaces agent messages
//!    with summary-rendered-as-bare-user-message plus the retained tail, and
//!    switches the current session id to the compacted node so subsequent
//!    appends land there.
//! 6. After a resume onto a compacted node, assistant `usage` is zeroed in
//!    memory (stale pre-compaction anchors overstate the view); the
//!    persisted `entries.usage` column stays faithful.
#![allow(clippy::unwrap_used, clippy::expect_used)]
#![expect(clippy::panic)]

mod common;

use std::sync::Arc;

use aaos_session::compaction::{build_transcript, find_cut_point};
use aaos_session::{
    AgentSession, AssistantSegment, ContentBlock as StoreBlock, Segment, SessionStore,
    StopReason as StoreStopReason, ToolCall as StoreToolCall, Usage as StoreUsage,
};
use common::store_with;
use pi_agent_core::agent::Agent;
use pi_agent_core::stream::simple_text_response;
use pi_agent_core::types::{ContentBlock, Message, Model, StreamFn};

/// Build a transcript of user/assistant text turns, each `chars` chars long.
async fn seed_text_turns(store: &SessionStore, root: &str, turns: &[(&str, usize)]) {
    for (label, chars) in turns {
        let text = format!("{label}-{}", "x".repeat(*chars));
        store
            .append_segment(root, &Segment::user_text(text.clone()))
            .await
            .unwrap();
        store
            .append_segment(
                root,
                &Segment::assistant_text(format!("a-{label}-{}", "y".repeat(*chars))),
            )
            .await
            .unwrap();
    }
}

/// Absolute object paths referenced by a transcript's path lines: the
/// `[Tool result] {path}` payload, `[Tool call] … — full arguments at
/// {path}`, and `[Image] at {path}`.
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

fn make_agent(stream_fn: Arc<dyn StreamFn>) -> Agent {
    let mut agent = Agent::new(stream_fn);
    agent.state.model = Model {
        id: "test".into(),
        ..Model::unknown()
    };
    agent
}

#[tokio::test]
async fn compacted_view_is_summary_plus_retained_tail() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();

    // 4 turns of 40-char prompts + 40-char answers.
    seed_text_turns(
        &store,
        &root,
        &[("q1", 40), ("q2", 40), ("q3", 40), ("q4", 40)],
    )
    .await;
    let segments = store.materialize_plain(&root).await.unwrap();
    assert_eq!(segments.len(), 8);

    // Pick a cut point with the pure module: a small budget forces a real cut
    // (the default 20000 would keep everything → None).
    let messages: Vec<Message> = segments
        .iter()
        .map(|seg| Message::try_from(seg.clone()).expect("text segments convert to messages"))
        .collect();
    let first_kept = find_cut_point(&messages, 60).expect("cut point with small budget");
    assert!(first_kept > 0 && first_kept < messages.len());

    // Build the deterministic transcript of exactly the replaced prefix
    // [0, first_kept) — what the coordinator stores as the summary content.
    let summary_text = build_transcript(&segments[..first_kept], store.objects()).unwrap();
    assert!(
        summary_text.contains("q1-"),
        "seeded user texts are inlined: {summary_text}"
    );
    let summary = Segment::summary(summary_text.clone());

    let compacted = store
        .compact(&root, &[(0, first_kept as u64)], &summary)
        .await
        .unwrap();

    let compacted_view = store.materialize_plain(&compacted).await.unwrap();
    assert_eq!(compacted_view.len(), 1 + (messages.len() - first_kept));
    assert_eq!(compacted_view[0], summary);
    for (i, kept) in messages[first_kept..].iter().enumerate() {
        assert_eq!(
            compacted_view[i + 1].kind(),
            kept.role(),
            "retained tail segment {}",
            i + 1
        );
    }
}

#[tokio::test]
async fn transcript_paths_resolve_and_originals_retrievable() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();

    // A tool round-trip (result = raw content object) plus text turns.
    store
        .append_segment(
            &root,
            &Segment::Assistant(AssistantSegment {
                content: vec![StoreBlock::ToolCall(StoreToolCall {
                    id: "c1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path": "/tmp/note.txt"}),
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
    let result_text = "R".repeat(4000);
    store
        .append_segment(&root, &Segment::tool_result_text("c1", &result_text))
        .await
        .unwrap();
    seed_text_turns(&store, &root, &[("q1", 40), ("q2", 40)]).await;

    // Compact the whole prefix except the last turn.
    let segments = store.materialize_plain(&root).await.unwrap();
    let summary_text = build_transcript(&segments[..4], store.objects()).unwrap();
    let compacted = store
        .compact(&root, &[(0, 4)], &Segment::summary(summary_text.clone()))
        .await
        .unwrap();

    // Content track: every path the transcript references exists on disk and
    // the result object holds the raw output bytes.
    let paths = transcript_paths(&summary_text);
    assert!(!paths.is_empty(), "transcript references objects");
    for path in &paths {
        assert!(std::fs::exists(path).unwrap(), "object exists: {path}");
    }
    let result_path = summary_text
        .lines()
        .find_map(|line| line.strip_prefix("[Tool result] "))
        .expect("result path rendered");
    assert_eq!(std::fs::read_to_string(result_path).unwrap(), result_text);

    // Structural track: `fetch_originals` returns the covered prefix
    // verbatim (the summary segment itself carries no sources, ADR-0006).
    let originals = store.fetch_originals(&compacted).await.unwrap();
    assert_eq!(originals.len(), 1);
    assert_eq!(originals[0].start, 0);
    assert_eq!(originals[0].end, 4);
    assert_eq!(originals[0].originals, segments[..4]);
}

#[tokio::test]
async fn undo_forks_parent_with_no_summary() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    seed_text_turns(&store, &root, &[("q1", 40), ("q2", 40), ("q3", 40)]).await;

    let summary = Segment::summary("first turn");
    let compacted = store.compact(&root, &[(0, 2)], &summary).await.unwrap();
    assert!(!compacted.is_empty());

    let undo = store.fork(&root).await.unwrap();
    let undone = store.materialize_plain(&undo).await.unwrap();
    assert_eq!(undone.len(), 6);
    assert!(undone.iter().all(|s| s.kind() != "summary"));
}

#[tokio::test]
async fn head_unchanged_by_compact() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    store
        .append_segment(&root, &Segment::user_text("q1"))
        .await
        .unwrap();
    store
        .append_segment(&root, &Segment::assistant_text("a1"))
        .await
        .unwrap();
    store
        .append_segment(&root, &Segment::user_text("q2"))
        .await
        .unwrap();
    // head follows appends (ADR-0003): root is last appended to.
    assert_eq!(store.head().await.unwrap(), Some(root.clone()));

    let summary = Segment::summary("prefix");
    let _compacted = store.compact(&root, &[(0, 2)], &summary).await.unwrap();

    // compact is a derivation — it must not move the head pointer.
    assert_eq!(store.head().await.unwrap(), Some(root));
}

#[tokio::test]
async fn resume_onto_compacted_node_renders_summary_and_redirects_appends() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    // Turn 1 (compact range [0, 2)); turn 2's assistant carries a large
    // pre-compaction usage total that must not survive the resume.
    seed_text_turns(&store, &root, &[("q1", 40)]).await;
    store
        .append_segment(&root, &Segment::user_text("q2-x".repeat(10)))
        .await
        .unwrap();
    let mut tail_assistant = Segment::assistant_text(format!("a-q2-{}", "y".repeat(40)));
    if let Segment::Assistant(a) = &mut tail_assistant {
        a.usage = StoreUsage {
            total_tokens: 50_000,
            ..StoreUsage::default()
        };
    }
    store.append_segment(&root, &tail_assistant).await.unwrap();
    let summary = Segment::summary("first turn summarized");
    let compacted = store.compact(&root, &[(0, 2)], &summary).await.unwrap();

    let mut session = AgentSession::new(
        store.clone(),
        make_agent(simple_text_response("ok")),
        root.clone(),
        "/tmp",
    );
    let loaded = session.resume(&compacted).await.unwrap();
    // View = summary (as user message) + kept tail (user q2 + assistant a2).
    assert_eq!(loaded, 3);
    assert_eq!(session.current_session_id().await, compacted.clone());

    let messages = &session.agent().state.messages;
    assert_eq!(messages.len(), 3);
    // Summary rendered as a bare user message — exactly the summary
    // content, no provenance prefix (the transcript's preamble is
    // self-describing).
    let Message::User(first) = &messages[0] else {
        panic!("first message should be a user-rendered summary");
    };
    let ContentBlock::Text { text } = &first.content[0] else {
        panic!("summary must be a text block, got {:?}", first.content[0]);
    };
    assert_eq!(text, "first turn summarized", "bare summary content");
    // Retained tail is verbatim (user q2 + assistant a2), not summarized,
    // and carries NO stale pre-compaction usage anchors.
    assert!(matches!(&messages[1], Message::User(_)));
    let Message::Assistant(resumed_assistant) = &messages[2] else {
        panic!("retained tail assistant");
    };
    assert_eq!(
        resumed_assistant.usage.total_tokens, 0,
        "resume onto a compacted node zeroes assistant usage"
    );

    // A subsequent append lands on the compacted node (the shared lock
    // redirects the persist listener), not on the root.
    session.agent_mut().prompt("follow-up").await.unwrap();
    assert_eq!(session.current_session_id().await, compacted.clone());
    let appended = store.materialize_plain(&compacted).await.unwrap();
    assert_eq!(appended.len(), 5);
    assert_eq!(appended[3].kind(), "user");
    assert_eq!(appended[4].kind(), "assistant");
    // The root line is untouched.
    assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 4);
}
