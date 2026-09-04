//! Ticket 01 — 根会话追加与视图：SQLite 结构层的最小纵切。
//! Seam: `SessionStore` 公共接口（open / create_root / append_segment /
//! materialize / materialize_plain）。
#![allow(clippy::unwrap_used, clippy::expect_used)]

use aaos_session::{
    AssistantSegment, ContentBlock, Cost, ImageSource, Segment, SessionStore, StopReason, ToolCall,
    ToolResultSegment, Usage, UserSegment,
};

#[tokio::test]
async fn root_append_materialize_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();
    let session = store.create_root().await.unwrap();
    let segs = vec![
        Segment::user_text("hello"),
        Segment::assistant_text("hi there"),
        Segment::tool_result_text("call-1", "42"),
    ];
    for seg in &segs {
        store.append_segment(&session, seg).await.unwrap();
    }
    assert_eq!(store.materialize_plain(&session).await.unwrap(), segs);
}

#[tokio::test]
async fn summary_segment_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();
    let session = store.create_root().await.unwrap();

    let summary = Segment::summary("compacted");
    store.append_segment(&session, &summary).await.unwrap();

    let view = store.materialize_plain(&session).await.unwrap();
    assert_eq!(view, vec![summary]);
    assert_eq!(view[0].kind(), "summary");
}

/// ADR-0006 multi-block round-trip: assistant messages with thinking +
/// text + tool_call blocks, tool results with details, and image blocks
/// survive the block decomposition / re-assembly byte-exact.
#[tokio::test]
async fn multiblock_segments_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();
    let session = store.create_root().await.unwrap();

    let assistant = Segment::Assistant(AssistantSegment {
        content: vec![
            ContentBlock::Thinking {
                text: "hidden".into(),
            },
            ContentBlock::Text {
                text: "visible".into(),
            },
            ContentBlock::ToolCall(ToolCall {
                id: "call-9".into(),
                name: "read_file".into(),
                // Keys inserted non-sorted on purpose: the canonical JSON
                // object must come back in the same Value, and the object
                // bytes are canonical (sorted), see the raw-content test.
                arguments: serde_json::json!({"z": 1, "a": 2}),
            }),
        ],
        stop_reason: StopReason::ToolUse,
        model: "m".into(),
        provider: "p".into(),
        api: "a".into(),
        usage: Usage {
            input: 1,
            output: 2,
            cache_read: 3,
            cache_write: 4,
            total_tokens: 5,
            cost: Cost {
                input: 0.1,
                output: 0.2,
                cache_read: 0.0,
                cache_write: 0.0,
                total: 0.3,
            },
        },
        error_message: Some("boom".into()),
    });
    let result = Segment::ToolResult(ToolResultSegment {
        tool_call_id: "call-9".into(),
        tool_name: "read_file".into(),
        content: vec![ContentBlock::Text {
            text: "the file".into(),
        }],
        details: serde_json::json!({"bytes": 9}),
        usage: None,
        added_tool_names: Some(vec!["extra".into()]),
        is_error: false,
    });
    let image = Segment::User(UserSegment {
        content: vec![ContentBlock::Image {
            source: ImageSource {
                mime_type: "image/png".into(),
                bytes: vec![0x89, 0x50, 0x4e, 0x47],
            },
        }],
    });

    let segs = vec![assistant, result, image];
    for seg in &segs {
        store.append_segment(&session, seg).await.unwrap();
    }
    assert_eq!(store.materialize_plain(&session).await.unwrap(), segs);
}

/// ADR-0006 storage contract: an appended segment is decomposed into
/// `entry_blocks` rows pointing at raw-content objects — the bytes on disk
/// are the content itself (UTF-8 text, canonical JSON), never an envelope.
/// Also pins the `preserve_order` invariant: `Value::Object` keys land
/// sorted in the canonical JSON object.
#[tokio::test]
async fn block_objects_are_raw_content() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();
    let session = store.create_root().await.unwrap();
    store
        .append_segment(&session, &Segment::assistant_text("plain text"))
        .await
        .unwrap();
    store
        .append_segment(
            &session,
            &Segment::Assistant(AssistantSegment {
                content: vec![
                    ContentBlock::Thinking {
                        text: "think".into(),
                    },
                    ContentBlock::ToolCall(ToolCall {
                        id: "call-9".into(),
                        name: "read_file".into(),
                        arguments: serde_json::json!({"z": 1, "a": 2}),
                    }),
                ],
                stop_reason: StopReason::ToolUse,
                model: "m".into(),
                provider: "p".into(),
                api: "a".into(),
                usage: Usage::default(),
                error_message: None,
            }),
        )
        .await
        .unwrap();

    // Row-level view of what the appends wrote.
    let conn = rusqlite::Connection::open(dir.path().join("store.db")).unwrap();
    let (kind, hash): (String, String) = conn
        .query_row(
            "SELECT kind, hash FROM entry_blocks WHERE session_id = ?1 AND seq = 0 AND idx = 0",
            [&session],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(kind, "text");
    assert_eq!(
        store.objects().get_bytes(&hash).await.unwrap(),
        b"plain text",
        "the text block object is the raw UTF-8 content, not an envelope"
    );

    let (kind, hash, tool_call_id, tool_name): (String, String, String, String) = conn
        .query_row(
            "SELECT kind, hash, tool_call_id, tool_name FROM entry_blocks
             WHERE session_id = ?1 AND seq = 1 AND idx = 1",
            [&session],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert_eq!(kind, "tool_call");
    assert_eq!(tool_call_id, "call-9");
    assert_eq!(tool_name, "read_file");
    assert_eq!(
        store.objects().get_bytes(&hash).await.unwrap(),
        br#"{"a":2,"z":1}"#,
        "tool_call object = canonical arguments JSON with sorted keys"
    );
}

#[tokio::test]
async fn reopen_persists_sessions() {
    let dir = tempfile::tempdir().unwrap();
    let session = {
        let store = SessionStore::open(dir.path()).await.unwrap();
        let session = store.create_root().await.unwrap();
        store
            .append_segment(&session, &Segment::user_text("persisted"))
            .await
            .unwrap();
        session
        // store dropped here: dedicated DB thread shuts down
    };

    let store = SessionStore::open(dir.path()).await.unwrap();
    let view = store.materialize_plain(&session).await.unwrap();
    assert_eq!(view, vec![Segment::user_text("persisted")]);
}

#[tokio::test]
async fn append_missing_session_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let store = SessionStore::open(dir.path()).await.unwrap();

    let err = store
        .append_segment("no-such-session", &Segment::user_text("x"))
        .await
        .unwrap_err();
    assert!(err.to_string().contains("not found"), "got: {err}");
}

#[tokio::test]
async fn concurrent_handle_sees_appends() {
    let dir = tempfile::tempdir().unwrap();
    let writer = SessionStore::open(dir.path()).await.unwrap();
    let session = writer.create_root().await.unwrap();

    // WAL: a second handle (same-process stand-in for the CLI reader process)
    // materializes while the writer is still open.
    let reader = SessionStore::open(dir.path()).await.unwrap();
    assert_eq!(reader.materialize_plain(&session).await.unwrap(), vec![]);

    writer
        .append_segment(&session, &Segment::user_text("seen"))
        .await
        .unwrap();
    let view = reader.materialize_plain(&session).await.unwrap();
    assert_eq!(view, vec![Segment::user_text("seen")]);
}

// --- Ticket 02: fork — derivation and chain view ---

mod common;

use common::store_with;

#[tokio::test]
async fn fork_inherits_prefix_extends_tail() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    let segs = vec![
        Segment::user_text("q1"),
        Segment::assistant_text("a1"),
        Segment::user_text("q2"),
    ];
    for seg in &segs {
        store.append_segment(&root, seg).await.unwrap();
    }

    let child = store.fork(&root).await.unwrap();
    assert_eq!(store.materialize_plain(&child).await.unwrap(), segs);

    let own = vec![Segment::assistant_text("a2"), Segment::user_text("q3")];
    for seg in &own {
        store.append_segment(&child, seg).await.unwrap();
    }

    let mut want = segs.clone();
    want.extend(own);
    assert_eq!(store.materialize_plain(&child).await.unwrap(), want);
    // Parent is immutable under the child's appends.
    assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 3);
}

#[tokio::test]
async fn fork_at_position_inherits_prefix() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    let segs = vec![
        Segment::user_text("1"),
        Segment::user_text("2"),
        Segment::user_text("3"),
        Segment::user_text("4"),
    ];
    for seg in &segs {
        store.append_segment(&root, seg).await.unwrap();
    }

    let child = store.fork_at(&root, 2).await.unwrap();
    assert_eq!(
        store.materialize_plain(&child).await.unwrap(),
        vec![Segment::user_text("1"), Segment::user_text("2")]
    );
    store
        .append_segment(&child, &Segment::user_text("child-only"))
        .await
        .unwrap();
    assert_eq!(store.materialize_plain(&child).await.unwrap().len(), 3);
    assert_eq!(store.materialize_plain(&root).await.unwrap().len(), 4);
}

#[tokio::test]
async fn fork_beyond_parent_view_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    store
        .append_segment(&root, &Segment::user_text("only"))
        .await
        .unwrap();

    let err = store.fork_at(&root, 5).await.unwrap_err();
    assert!(
        err.to_string().contains("position"),
        "expected position error, got: {err}"
    );
}

#[tokio::test]
async fn grandchild_materializes_chain() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    store
        .append_segment(&root, &Segment::user_text("r1"))
        .await
        .unwrap();
    store
        .append_segment(&root, &Segment::user_text("r2"))
        .await
        .unwrap();

    let child = store.fork(&root).await.unwrap();
    store
        .append_segment(&child, &Segment::user_text("c1"))
        .await
        .unwrap();

    let grandchild = store.fork(&child).await.unwrap();
    store
        .append_segment(&grandchild, &Segment::user_text("g1"))
        .await
        .unwrap();

    let view = store.materialize_plain(&grandchild).await.unwrap();
    assert_eq!(
        view,
        vec![
            Segment::user_text("r1"),
            Segment::user_text("r2"),
            Segment::user_text("c1"),
            Segment::user_text("g1"),
        ]
    );
}

#[tokio::test]
async fn latest_created_session_is_most_recent() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    let root = store.create_root().await.unwrap();
    store
        .append_segment(&root, &Segment::user_text("q"))
        .await
        .unwrap();
    assert_eq!(
        store.latest_created_session().await.unwrap(),
        Some(root.clone())
    );

    let child = store.fork(&root).await.unwrap();
    assert_eq!(store.latest_created_session().await.unwrap(), Some(child));
}

/// ADR-0003: the head pointer is the session last written — appends move it,
/// derivations do not, and it survives a reopen.
#[tokio::test]
async fn head_follows_appends() {
    let dir = tempfile::tempdir().unwrap();
    let store = store_with(dir.path()).await;
    assert_eq!(store.head().await.unwrap(), None, "fresh store");

    let root = store.create_root().await.unwrap();
    assert_eq!(
        store.head().await.unwrap(),
        None,
        "creating a session moves nothing"
    );

    store
        .append_segment(&root, &Segment::user_text("q"))
        .await
        .unwrap();
    assert_eq!(store.head().await.unwrap(), Some(root.clone()));

    let child = store.fork(&root).await.unwrap();
    assert_eq!(
        store.head().await.unwrap(),
        Some(root.clone()),
        "deriving moves nothing"
    );

    store
        .append_segment(&child, &Segment::user_text("c"))
        .await
        .unwrap();
    assert_eq!(store.head().await.unwrap(), Some(child.clone()));

    drop(store);
    let store = store_with(dir.path()).await;
    assert_eq!(store.head().await.unwrap(), Some(child));
}
