//! Model resolution and agent assembly: the model-catalog resolution chain,
//! the construction of a fully equipped `Agent`, and the compaction
//! coordinator assembly.

use std::path::Path;
use std::sync::Arc;

use aaos_providers::{
    ProviderRetryConfig, parse_thinking, resolve_catalog_model, stream_fn_for_with_retry,
};
use aaos_session::SessionStore;
use aaos_tools::{SkillIndex, build_system_prompt, create_coding_tools};
use pi_agent_core::agent::Agent;
use pi_agent_core::types::{Model, StreamFn};

use crate::compaction::{CompactionCoordinator, CompactionSettings};

/// Agent assembly config: model resolution + tools + system prompt inputs.
/// `None` fields mean "unset" — the composing frontend fills in its product
/// defaults before assembly (the runtime stays product-agnostic).
#[derive(Debug, Clone, Default)]
pub struct AgentConfig {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub thinking: Option<String>,
}

/// Host environment adaptation: the frontend injects `Paths`, the model
/// list URL, and the API-key env resolver; the runtime reads no environment
/// variables itself.
#[derive(Clone)]
pub struct EnvConfig {
    pub paths: aaos_providers::Paths,
    pub model_list_url: String,
    pub api_key_resolver: Arc<dyn Fn(&str) -> Option<String> + Send + Sync>,
}

impl std::fmt::Debug for EnvConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnvConfig")
            .field("paths", &self.paths)
            .field("model_list_url", &self.model_list_url)
            .finish_non_exhaustive()
    }
}

/// Resolve the model from [`AgentConfig`] and the host [`EnvConfig`]: spec
/// concatenation → `resolve_catalog_model` → api key resolution → `to_model`
/// → `stream_fn_for_with_retry`. Returns the live model (whose
/// `context_window` feeds compaction), the assembled stream, and the API key.
pub async fn resolve_model(
    cfg: &AgentConfig,
    env: &EnvConfig,
) -> Result<(Model, Arc<dyn StreamFn>, String), String> {
    let provider_id = cfg
        .provider
        .as_deref()
        .ok_or("no provider specified")?
        .to_string();
    let model_id = cfg
        .model
        .as_deref()
        .ok_or("no model specified")?
        .to_string();
    let spec = if model_id.contains('/') {
        model_id.clone()
    } else {
        format!("{provider_id}/{model_id}")
    };

    let catalog_model = resolve_catalog_model(&env.paths, &env.model_list_url, &spec)
        .await
        .map_err(|e| e.to_string())?;
    let api_key = catalog_model
        .resolve_api_key(|k| (env.api_key_resolver)(k))
        .map_err(|e| e.to_string())?;
    let model = catalog_model.to_model();
    let provider = stream_fn_for_with_retry(&model, ProviderRetryConfig::default())
        .map_err(|e| e.to_string())?;
    Ok((model, provider, api_key))
}

/// Assemble the agent: thinking parse → model resolution → skill discovery
/// (frozen at assembly time) → tools → system prompt → agent state.
///
/// Event routing is deliberately absent here: frontends route events through
/// their own sink handling (or the agent's `subscribe` as the underlying
/// mechanism), keeping assembly reusable across UIs.
pub async fn build_agent(
    cfg: &AgentConfig,
    env: &EnvConfig,
    cwd: &Path,
    user_skills_dir: &Path,
) -> Result<Agent, String> {
    let thinking = match cfg.thinking.as_deref() {
        Some(s) => parse_thinking(s)?,
        None => return Err("no thinking level specified".into()),
    };
    let (model, provider, api_key) = resolve_model(cfg, env).await?;
    // Discover skills once at assembly (frozen for the process lifetime):
    // user-level skills plus project-level `<cwd>/.agents/skills/`.
    let skills = Arc::new(SkillIndex::discover(
        user_skills_dir,
        &cwd.join(".agents/skills"),
    ));
    let tools = create_coding_tools(cwd, skills.clone());
    let system_prompt = build_system_prompt(cwd, &tools, &skills);
    let mut agent = Agent::new(provider);
    agent.state.model = model;
    agent.state.thinking_level = thinking;
    agent.state.tools = tools;
    agent.state.system_prompt = system_prompt;
    agent.stream_fn_options.api_key = Some(api_key);
    agent.stream_fn_options.provider_retry_max_retries = 0;
    agent.stream_fn_options.provider_retry_max_delay_ms = 60000;

    Ok(agent)
}

/// Build the compaction coordinator from the already-resolved live model —
/// its context window drives the auto-trigger checks. No model
/// re-resolution happens here.
pub fn build_coordinator(
    model: &Model,
    store: &SessionStore,
    settings: CompactionSettings,
) -> Arc<CompactionCoordinator> {
    Arc::new(CompactionCoordinator::new(store.clone(), settings, model))
}
