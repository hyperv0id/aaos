use crate::agent_loop::AgentRun;
use crate::types::{AgentEvent, AgentState, AgentToolResult, ContentBlock, Message};

/// A normalized trace entry for behavioral equivalence checks.
///
/// Timestamps and provider object identity are deliberately omitted.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceEntry {
    Event {
        event_type: String,
    },
    MessageStart {
        role: String,
    },
    MessageEnd {
        role: String,
        stop_reason: Option<String>,
    },
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        args: serde_json::Value,
    },
    ToolExecutionEnd {
        tool_call_id: String,
        tool_name: String,
        result_summary: String,
        is_error: bool,
    },
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        is_error: bool,
    },
    TurnEnd {
        tool_result_ids: Vec<String>,
    },
    StateSnapshot {
        is_streaming: bool,
        pending: Vec<String>,
    },
}

#[derive(Debug, Clone, Default)]
pub struct TraceCollector {
    entries: Vec<TraceEntry>,
}

impl TraceCollector {
    pub fn new() -> Self {
        Self::default()
    }

    /// Consume all events from a low-level run and return its produced
    /// messages.
    ///
    /// # Errors
    ///
    /// Returns the spawn [`tokio::task::JoinError`] if the low-level loop
    /// panicked or was cancelled.
    pub async fn collect_run(
        &mut self,
        run: &mut AgentRun,
    ) -> Result<Vec<Message>, crate::agent_loop::LoopError> {
        while let Some(event) = run.next_event().await {
            self.observe_event(&event);
        }
        run.result().await
    }

    pub fn observe_event(&mut self, event: &AgentEvent) {
        match event {
            AgentEvent::AgentStart => self.push_event("agent_start"),
            AgentEvent::AgentEnd { .. } => self.push_event("agent_end"),
            AgentEvent::TurnStart => self.push_event("turn_start"),
            AgentEvent::TurnEnd { tool_results, .. } => {
                self.entries.push(TraceEntry::TurnEnd {
                    tool_result_ids: tool_results
                        .iter()
                        .map(|result| result.tool_call_id.clone())
                        .collect(),
                });
            }
            AgentEvent::MessageStart { message } => {
                self.entries.push(TraceEntry::MessageStart {
                    role: message.role().into(),
                });
            }
            AgentEvent::MessageUpdate { .. } | AgentEvent::ToolExecutionUpdate { .. } => {
                // Streaming updates are intentionally secondary trace detail.
            }
            AgentEvent::MessageEnd { message } => {
                self.entries.push(TraceEntry::MessageEnd {
                    role: message.role().into(),
                    stop_reason: message
                        .as_assistant()
                        .map(|assistant| assistant.stop_reason.to_string()),
                });
                if let Message::ToolResult(result) = message {
                    self.entries.push(TraceEntry::ToolResult {
                        tool_call_id: result.tool_call_id.clone(),
                        tool_name: result.tool_name.clone(),
                        is_error: result.is_error,
                    });
                }
            }
            AgentEvent::ToolExecutionStart {
                tool_call_id,
                tool_name,
                args,
            } => {
                self.entries.push(TraceEntry::ToolExecutionStart {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    args: args.clone(),
                });
            }
            AgentEvent::ToolExecutionEnd {
                tool_call_id,
                tool_name,
                result,
                is_error,
            } => {
                self.entries.push(TraceEntry::ToolExecutionEnd {
                    tool_call_id: tool_call_id.clone(),
                    tool_name: tool_name.clone(),
                    result_summary: summarize_result(result, *is_error),
                    is_error: *is_error,
                });
            }
        }
    }

    pub fn snapshot_state(&mut self, state: &AgentState) {
        let mut pending: Vec<String> = state.pending_tool_calls.iter().cloned().collect();
        pending.sort();
        self.entries.push(TraceEntry::StateSnapshot {
            is_streaming: state.is_streaming,
            pending,
        });
    }

    pub fn entries(&self) -> &[TraceEntry] {
        &self.entries
    }

    pub fn into_entries(self) -> Vec<TraceEntry> {
        self.entries
    }

    fn push_event(&mut self, event_type: &str) {
        self.entries.push(TraceEntry::Event {
            event_type: event_type.into(),
        });
    }
}

fn summarize_result(result: &AgentToolResult, is_error: bool) -> String {
    if is_error {
        return "error".into();
    }
    result
        .content
        .iter()
        .find_map(|content| match content {
            ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "(no text)".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AssistantMessage, UserMessage};

    #[test]
    fn message_events_normalize_roles() {
        let mut trace = TraceCollector::new();
        trace.observe_event(&AgentEvent::MessageStart {
            message: Message::User(UserMessage::new("hi")),
        });
        trace.observe_event(&AgentEvent::MessageEnd {
            message: Message::Assistant(AssistantMessage::text("ok")),
        });

        assert_eq!(
            trace.entries(),
            &[
                TraceEntry::MessageStart {
                    role: "user".into()
                },
                TraceEntry::MessageEnd {
                    role: "assistant".into(),
                    stop_reason: Some("stop".into()),
                },
            ]
        );
    }

    #[test]
    fn state_snapshot_captures_pending_and_streaming() {
        let state = AgentState {
            is_streaming: true,
            pending_tool_calls: ["c1".into(), "c2".into()].into_iter().collect(),
            ..Default::default()
        };

        let mut trace = TraceCollector::new();
        trace.snapshot_state(&state);

        assert_eq!(
            trace.entries()[0],
            TraceEntry::StateSnapshot {
                is_streaming: true,
                pending: vec!["c1".into(), "c2".into()],
            }
        );
    }

    #[test]
    fn tool_execution_start_keeps_argument_snapshot() {
        let mut trace = TraceCollector::new();
        trace.observe_event(&AgentEvent::ToolExecutionStart {
            tool_call_id: "c1".into(),
            tool_name: "echo".into(),
            args: serde_json::json!({"text": "hello"}),
        });

        assert_eq!(
            trace.entries()[0],
            TraceEntry::ToolExecutionStart {
                tool_call_id: "c1".into(),
                tool_name: "echo".into(),
                args: serde_json::json!({"text": "hello"}),
            }
        );
    }

    #[test]
    fn error_result_uses_compact_summary() {
        let mut trace = TraceCollector::new();
        trace.observe_event(&AgentEvent::ToolExecutionEnd {
            tool_call_id: "c1".into(),
            tool_name: "echo".into(),
            result: AgentToolResult::text("ignored"),
            is_error: true,
        });

        assert_eq!(
            trace.entries()[0],
            TraceEntry::ToolExecutionEnd {
                tool_call_id: "c1".into(),
                tool_name: "echo".into(),
                result_summary: "error".into(),
                is_error: true,
            }
        );
    }
}
