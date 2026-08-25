//! Model registry: models.dev fetch, user `models.json` overrides,
//! credential resolution, and catalog lookups.
//!
//! Fetching the registry and serving the conversation are orthogonal: the
//! registry layer is persisted to `registry-cache.json` in the config
//! directory and refreshed in the background once per process start
//! (awaited inline only on cold start); the `models.json` override layer
//! is never cached.

use std::collections::HashMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use pi_agent_core::types::{Model, ModelCost, ModelInput, ThinkingLevel};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::formats::anthropic_messages;
use crate::formats::cohere_chat;
use crate::formats::google_genai;
use crate::formats::openai_completions;

/// Default models.dev registry endpoint returning the canonical provider/model JSON.
pub const DEFAULT_REGISTRY_URL: &str = "https://models.dev/api.json";

const CONFIG_FILE: &str = "models.json";
const CACHE_FILE: &str = "registry-cache.json";
const CACHE_FILE_TMP: &str = "registry-cache.json.tmp";

#[derive(Debug, Error)]
pub enum CatalogError {
    #[error("failed to read config {path}: {source}")]
    ConfigIo {
        path: String,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid config {path}: {source}")]
    ConfigParse {
        path: String,
        #[source]
        source: serde_json::Error,
    },
    #[error("models.dev request failed: {0}")]
    Fetch(String),
    #[error("models.dev returned invalid JSON: {0}")]
    RegistryParse(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("API key environment variable {0} is not set")]
    MissingApiKey(String),
}

/// Filesystem location for the user `models.json` overrides.
#[derive(Debug, Clone)]
pub struct Paths {
    /// Directory holding `models.json`.
    pub config_dir: PathBuf,
}

impl Paths {
    /// Build a [`Paths`] rooted at an explicit config directory.
    pub fn from_config_dir(config_dir: impl Into<PathBuf>) -> Self {
        Self {
            config_dir: config_dir.into(),
        }
    }

    /// Default user config path: `$HOME/.config/aaos`.
    pub fn default_user() -> Self {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/"));
        Self {
            config_dir: home.join(".config").join("aaos"),
        }
    }

    /// Path to the user `models.json` override file.
    pub fn models_json(&self) -> PathBuf {
        self.config_dir.join(CONFIG_FILE)
    }
    /// Path to the persisted registry cache holding the raw models.dev JSON.
    pub fn registry_cache(&self) -> PathBuf {
        self.config_dir.join(CACHE_FILE)
    }
}

/// Deserialized `models.json`: provider-level and model-level overrides.
///
/// Model-level overrides (keyed by `provider/model`) take precedence over
/// provider-level overrides, which take precedence over npm-derived defaults.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct UserConfig {
    /// Provider-level overrides, keyed by provider id.
    #[serde(default)]
    pub providers: HashMap<String, ProviderOverride>,
    /// 模型级覆盖，键为 `provider/model` 限定 id，优先级高于提供商级覆盖与 npm 推导。
    #[serde(default)]
    pub models: HashMap<String, ProviderOverride>,
}

/// A single override applied to a provider or model.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderOverride {
    /// Override the resolved base URL (must include the version path).
    pub base_url: Option<String>,
    /// Override the API format dispatch key (e.g. `openai-completions`).
    pub api: Option<String>,
    /// Override the API key, either as a raw string or a `$ENV_VAR` reference.
    pub api_key: Option<String>,
}

/// A model entry resolved from models.dev + user overrides, ready for catalog lookup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CatalogModel {
    /// Model id within the provider (e.g. `gpt-4o`).
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// API format dispatch key (e.g. `openai-completions`, `anthropic-messages`).
    pub api: String,
    /// Provider id (e.g. `openai`, `anthropic`).
    pub provider: String,
    /// Fully-qualified base URL including version path.
    pub base_url: String,
    /// Whether the model supports extended thinking/reasoning.
    pub reasoning: bool,
    /// Supported input modalities (`"text"`, `"image"`).
    pub input: Vec<String>,
    /// Per-token pricing in USD.
    pub cost: ModelCostDto,
    /// Maximum total context window in tokens.
    pub context_window: u64,
    /// Maximum output tokens.
    pub max_tokens: u64,
    /// Environment variable name holding the API key.
    pub api_key_env: String,
}

/// Per-token pricing for a catalog model, in USD per million tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ModelCostDto {
    /// Cost per million input tokens.
    pub input: f64,
    /// Cost per million output tokens.
    pub output: f64,
    /// Cost per million cached input tokens (read).
    pub cache_read: f64,
    /// Cost per million tokens written to the prompt cache.
    pub cache_write: f64,
}

impl CatalogModel {
    /// Return the `provider/model` qualified id used in catalog specs and overrides.
    pub fn qualified_id(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }

    /// Convert to a runtime [`Model`], mapping DTO modalities to [`ModelInput`].
    pub fn to_model(&self) -> Model {
        Model {
            id: self.id.clone(),
            name: self.name.clone(),
            api: self.api.clone(),
            provider: self.provider.clone(),
            base_url: self.base_url.clone(),
            reasoning: self.reasoning,
            input: self
                .input
                .iter()
                .filter_map(|m| match m.as_str() {
                    "text" => Some(ModelInput::Text),
                    "image" => Some(ModelInput::Image),
                    _ => None,
                })
                .collect(),
            cost: ModelCost {
                input: self.cost.input,
                output: self.cost.output,
                cache_read: self.cost.cache_read,
                cache_write: self.cost.cache_write,
            },
            context_window: self.context_window,
            max_tokens: self.max_tokens,
        }
    }

    /// Resolve the API key via `getenv`, returning [`CatalogError::MissingApiKey`] if unset or empty.
    ///
    /// `getenv` is injected so callers can mock the environment in tests.
    pub fn resolve_api_key(
        &self,
        getenv: impl Fn(&str) -> Option<String>,
    ) -> Result<String, CatalogError> {
        getenv(&self.api_key_env)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| CatalogError::MissingApiKey(self.api_key_env.clone()))
    }
}

/// Resolve a `provider/model` spec against a model list.
///
/// Bare model ids are rejected: a fallback provider is a product decision
/// that does not live here.
pub fn resolve_model<'a>(
    models: &'a [CatalogModel],
    spec: &str,
) -> Result<&'a CatalogModel, CatalogError> {
    let Some((provider, id)) = spec.split_once('/') else {
        return Err(CatalogError::ModelNotFound(spec.to_string()));
    };
    models
        .iter()
        .find(|m| m.provider == provider && m.id == id)
        .ok_or_else(|| CatalogError::ModelNotFound(spec.to_string()))
}

/// Load user `models.json` overrides from `path`, returning an empty config if the file is absent.
///
/// # Errors
///
/// Returns [`CatalogError::ConfigIo`] if the file exists but cannot be read,
/// or [`CatalogError::ConfigParse`] if the JSON is invalid.
fn load_user_config(path: &Path) -> Result<UserConfig, CatalogError> {
    if !path.exists() {
        return Ok(UserConfig::default());
    }
    let text = fs::read_to_string(path).map_err(|source| CatalogError::ConfigIo {
        path: path.display().to_string(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|source| CatalogError::ConfigParse {
        path: path.display().to_string(),
        source,
    })
}

/// Parse a thinking-level token (case-insensitive) into a [`ThinkingLevel`].
///
/// Accepts `off`/`none`, `minimal`, `low`, `medium`, `high`, `xhigh`/`x-high`,
/// and `max`. Any other string yields an error naming the unknown value.
///
/// # Examples
///
/// ```
/// use aaos_providers::parse_thinking;
/// use pi_agent_core::types::ThinkingLevel;
///
/// assert_eq!(parse_thinking("high"), Ok(ThinkingLevel::High));
/// // Case-insensitive.
/// assert_eq!(parse_thinking("OFF"), Ok(ThinkingLevel::Off));
/// assert_eq!(parse_thinking("x-high"), Ok(ThinkingLevel::XHigh));
/// assert!(parse_thinking("bogus").is_err());
/// ```
pub fn parse_thinking(s: &str) -> Result<ThinkingLevel, String> {
    match s.to_ascii_lowercase().as_str() {
        "off" | "none" => Ok(ThinkingLevel::Off),
        "minimal" => Ok(ThinkingLevel::Minimal),
        "low" => Ok(ThinkingLevel::Low),
        "medium" => Ok(ThinkingLevel::Medium),
        "high" => Ok(ThinkingLevel::High),
        "xhigh" | "x-high" => Ok(ThinkingLevel::XHigh),
        "max" => Ok(ThinkingLevel::Max),
        other => Err(format!("unknown thinking level: {other}")),
    }
}

#[derive(Debug, Deserialize)]
struct RegistryProvider {
    id: Option<String>,
    api: Option<String>,
    npm: Option<String>,
    env: Option<Vec<String>>,
    models: Option<HashMap<String, RegistryModel>>,
}

#[derive(Debug, Deserialize)]
struct RegistryModel {
    id: Option<String>,
    name: Option<String>,
    reasoning: Option<bool>,
    modalities: Option<Modalities>,
    limit: Option<Limit>,
    cost: Option<RegistryCost>,
}

#[derive(Debug, Deserialize)]
struct Modalities {
    input: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct Limit {
    context: Option<u64>,
    output: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct RegistryCost {
    input: Option<f64>,
    output: Option<f64>,
    cache_read: Option<f64>,
    cache_write: Option<f64>,
}
/// models.dev `npm` 字段 → API格式 键名映射（规格 §3）。
///
/// openai-completions 复用 [`openai_completions::API`]；anthropic-messages
/// 复用 [`anthropic_messages::API`]（issue 08）；google-genai 复用
/// [`google_genai::API`]（issue 09）；cohere-chat 复用
/// [`cohere_chat::API`]（issue 10）。openai-responses 的 adapter 尚未落地，
/// 故本表无对应行——其分发键仅预留于 `stream_fn_for`，未挂载目录。
/// 云托管（bedrock/vertex/azure/gateway/vercel）、社区包与未知 npm 不在表中：
/// 对应提供商整体跳过，绝不静默回退到 openai-completions。
const NPM_API_FORMATS: &[(&str, &str)] = &[
    ("@ai-sdk/openai-compatible", openai_completions::API),
    ("@ai-sdk/openai", openai_completions::API),
    ("@ai-sdk/xai", openai_completions::API),
    ("@ai-sdk/anthropic", anthropic_messages::API),
    ("@ai-sdk/google", google_genai::API),
    ("@ai-sdk/cohere", cohere_chat::API),
    ("@ai-sdk/mistral", openai_completions::API),
    ("@ai-sdk/groq", openai_completions::API),
    ("@ai-sdk/perplexity", openai_completions::API),
    ("@ai-sdk/togetherai", openai_completions::API),
    ("@ai-sdk/cerebras", openai_completions::API),
    ("@ai-sdk/deepinfra", openai_completions::API),
    ("@ai-sdk/deepseek", openai_completions::API),
    ("@ai-sdk/moonshotai", openai_completions::API),
    ("@ai-sdk/alibaba", openai_completions::API),
    ("@ai-sdk/minimax", openai_completions::API),
    ("@ai-sdk/fireworks", openai_completions::API),
    ("@ai-sdk/huggingface", openai_completions::API),
    ("@ai-sdk/baseten", openai_completions::API),
    ("@ai-sdk/gmicloud", openai_completions::API),
];

/// models.dev 中无 `api` URL 字段的 canonical 提供商默认 base URL（规格 §3）。
/// 与 [`NPM_API_FORMATS`] 同处维护。URL 含完整版本路径，与对应 `@ai-sdk/*`
/// 包的 `baseURL` 一致（规格 §2「base URL 与端点拼接约定」）：adapter 只追加
/// 尾段（`/chat/completions`、`/messages`、`/chat`、`/models/{id}:…`），
/// 绝不重拼版本路径。故 cohere 保留 `/v2`、deepinfra 保留 `/openai` 后缀。
const DEFAULT_BASE_URLS: &[(&str, &str)] = &[
    ("openai", "https://api.openai.com/v1"),
    ("anthropic", "https://api.anthropic.com/v1"),
    ("google", "https://generativelanguage.googleapis.com/v1beta"),
    ("cohere", "https://api.cohere.com/v2"),
    ("xai", "https://api.x.ai/v1"),
    ("mistral", "https://api.mistral.ai/v1"),
    ("groq", "https://api.groq.com/openai/v1"),
    ("perplexity", "https://api.perplexity.ai"),
    ("togetherai", "https://api.together.xyz/v1"),
    ("cerebras", "https://api.cerebras.ai/v1"),
    ("deepinfra", "https://api.deepinfra.com/v1/openai"),
];

fn npm_api_format(npm: &str) -> Option<&'static str> {
    NPM_API_FORMATS
        .iter()
        .find(|(pkg, _)| *pkg == npm)
        .map(|(_, api)| *api)
}

fn default_base_url(provider_id: &str) -> Option<&'static str> {
    DEFAULT_BASE_URLS
        .iter()
        .find(|(id, _)| *id == provider_id)
        .map(|(_, url)| *url)
}

fn api_key_env_from_ref(raw: &str) -> String {
    raw.trim().trim_start_matches('$').trim().to_string()
}

/// Parse models.dev registry JSON + user overrides into a sorted list of [`CatalogModel`]s.
///
/// Providers whose `npm` package is not in the API-format mapping are skipped
/// entirely (cloud-hosted/community/unknown). Models missing `base_url` or
/// `api_key_env` after all override layers are also skipped.
///
/// # Errors
///
/// Returns [`CatalogError::RegistryParse`] if the JSON cannot be deserialized.
fn build_catalog(
    registry_json: &str,
    config: &UserConfig,
) -> Result<Vec<CatalogModel>, CatalogError> {
    let providers: HashMap<String, RegistryProvider> = serde_json::from_str(registry_json)
        .map_err(|e| CatalogError::RegistryParse(e.to_string()))?;

    let mut out = Vec::new();
    for (provider_key, provider) in providers {
        let provider_id = provider.id.clone().unwrap_or(provider_key);
        let r#override = config.providers.get(&provider_id);
        let npm = provider.npm.as_deref();
        let registry_api = provider.api.as_deref().unwrap_or_default();

        // npm → API格式 推导；云托管/社区/未知 npm 不在表中，整个提供商跳过。
        // 该门禁先于 provider 覆盖：覆盖不能挽救未映射的 npm。
        let Some(derived_api) = npm.and_then(npm_api_format) else {
            continue;
        };
        let api = r#override
            .and_then(|ov| ov.api.as_deref().filter(|s| !s.is_empty()))
            .unwrap_or(derived_api);

        let base_url = r#override
            .and_then(|ov| ov.base_url.as_deref().filter(|s| !s.is_empty()))
            .or_else(|| {
                if registry_api.starts_with("http") {
                    Some(registry_api)
                } else {
                    None
                }
            })
            .or_else(|| default_base_url(&provider_id))
            .unwrap_or_default();

        let api_key_env = r#override
            .and_then(|o| o.api_key.as_deref().filter(|s| !s.is_empty()))
            .map(api_key_env_from_ref)
            .or_else(|| {
                provider
                    .env
                    .as_ref()
                    .and_then(|e| e.first())
                    .filter(|s| !s.is_empty())
                    .cloned()
            })
            .unwrap_or_default();

        // base_url/api_key 的空值门禁在模型循环内检查：模型级覆盖可补齐缺失值。

        for (model_key, model) in provider.models.unwrap_or_default() {
            let id = model.id.unwrap_or(model_key);
            // 模型级覆盖按 `provider/model` 限定 id 查找，优先级：模型级 > 提供商级 > npm 推导。
            let model_override = config.models.get(&format!("{provider_id}/{id}"));
            let api = model_override
                .and_then(|mo| mo.api.as_deref().filter(|s| !s.is_empty()))
                .unwrap_or(api);
            let base_url = model_override
                .and_then(|mo| mo.base_url.as_deref().filter(|s| !s.is_empty()))
                .unwrap_or(base_url);
            let api_key_env = model_override
                .and_then(|mo| mo.api_key.as_deref().filter(|s| !s.is_empty()))
                .map(api_key_env_from_ref)
                .unwrap_or_else(|| api_key_env.clone());
            if base_url.is_empty() || api_key_env.is_empty() {
                continue;
            }
            let cost = model.cost.unwrap_or(RegistryCost {
                input: None,
                output: None,
                cache_read: None,
                cache_write: None,
            });
            let limit = model.limit.unwrap_or(Limit {
                context: None,
                output: None,
            });
            let inputs = model
                .modalities
                .and_then(|m| m.input)
                .unwrap_or_else(|| vec!["text".into()]);
            out.push(CatalogModel {
                id,
                name: model.name.unwrap_or_default(),
                api: api.to_string(),
                provider: provider_id.clone(),
                base_url: base_url.to_string(),
                reasoning: model.reasoning.unwrap_or(false),
                input: inputs,
                cost: ModelCostDto {
                    input: cost.input.unwrap_or(0.0),
                    output: cost.output.unwrap_or(0.0),
                    cache_read: cost.cache_read.unwrap_or(0.0),
                    cache_write: cost.cache_write.unwrap_or(0.0),
                },
                context_window: limit.context.unwrap_or(0),
                max_tokens: limit.output.unwrap_or(0),
                api_key_env: api_key_env.clone(),
            });
        }
    }
    out.sort_by_key(|a| a.qualified_id());
    Ok(out)
}

/// Fetch the raw models.dev registry JSON from `url`.
///
/// The client carries a 10-second total timeout so a hanging registry can
/// never block conversation startup; timeout and transport failures both
/// surface as [`CatalogError::Fetch`] and take the existing cold-start fallback path; with a valid cache no fetch happens at all.
///
/// # Errors
///
/// Returns [`CatalogError::Fetch`] on transport failure, timeout, or
/// non-2xx status.
async fn fetch_registry(url: &str) -> Result<String, CatalogError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .map_err(|e| CatalogError::Fetch(e.to_string()))?;
    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| CatalogError::Fetch(e.to_string()))?;
    if !response.status().is_success() {
        return Err(CatalogError::Fetch(format!("HTTP {}", response.status())));
    }
    response
        .text()
        .await
        .map_err(|e| CatalogError::Fetch(e.to_string()))
}

/// Whether a provider-level override can serve any model id under it: api,
/// base_url, and api_key must all be present and non-empty.
fn provider_override_complete(ov: &ProviderOverride) -> bool {
    let non_empty =
        |field: &Option<String>| field.as_deref().is_some_and(|value| !value.is_empty());
    non_empty(&ov.api) && non_empty(&ov.base_url) && non_empty(&ov.api_key)
}

/// Build a catalog entry purely from user overrides, with no registry input.
///
/// Model-level overrides (`config.models["provider/id"]`, keyed by qualified
/// id) take precedence over provider-level overrides
/// (`config.providers[provider]`) on a per-field basis. Returns `None` when
/// the resolved override set lacks a non-empty `api`, `base_url`, or
/// `api_key` — those three are mandatory for a synthesized model. The rest
/// take benign defaults: `name` = the model id, `reasoning` = false,
/// text-only input, and zeroed cost/limits.
fn synthesize_model(config: &UserConfig, provider: &str, id: &str) -> Option<CatalogModel> {
    let model_override = config.models.get(&format!("{provider}/{id}"));
    let provider_override = config.providers.get(provider);
    // One field at a time: model-level non-empty value wins, then the
    // provider-level one; a missing/empty result fails synthesis.
    let field = |get: fn(&ProviderOverride) -> &Option<String>| {
        model_override
            .and_then(|o| get(o).as_deref().filter(|s| !s.is_empty()))
            .or_else(|| provider_override.and_then(|o| get(o).as_deref().filter(|s| !s.is_empty())))
    };
    let api = field(|o| &o.api)?;
    let base_url = field(|o| &o.base_url)?;
    let api_key = field(|o| &o.api_key)?;
    Some(CatalogModel {
        id: id.to_string(),
        name: id.to_string(),
        api: api.to_string(),
        provider: provider.to_string(),
        base_url: base_url.to_string(),
        reasoning: false,
        input: vec!["text".into()],
        cost: ModelCostDto::default(),
        context_window: 0,
        max_tokens: 0,
        api_key_env: api_key_env_from_ref(api_key),
    })
}

/// Catalog entries that stand without the registry: one per `provider/model`
/// key in `config.models`, synthesized purely from overrides.
fn standalone_models(config: &UserConfig) -> Vec<CatalogModel> {
    let mut out = Vec::new();
    for key in config.models.keys() {
        let Some((provider, id)) = key.split_once('/') else {
            continue;
        };
        if provider.is_empty() || id.is_empty() {
            continue;
        }
        if let Some(model) = synthesize_model(config, provider, id) {
            out.push(model);
        }
    }
    out
}

/// Atomically persist the raw registry JSON to `registry-cache.json`.
///
/// Writes to `registry-cache.json.tmp` in the same directory, then renames
/// it over the cache file, so a crashed write can never leave a corrupt
/// cache (the previous cache remains intact until the rename). Failures are
/// swallowed — the cache is an optimization and must never fail the
/// conversation.
async fn write_registry_cache(paths: &Paths, json: &str) {
    let _ = tokio::fs::create_dir_all(&paths.config_dir).await;
    let tmp = paths.config_dir.join(CACHE_FILE_TMP);
    if tokio::fs::write(&tmp, json).await.is_ok() {
        let _ = tokio::fs::rename(&tmp, paths.registry_cache()).await;
    }
}

/// Append standalone override entries not already present (registry entries
/// win collisions) and re-sort by qualified id.
fn merge_standalone(models: &mut Vec<CatalogModel>, config: &UserConfig) {
    for standalone in standalone_models(config) {
        if !models
            .iter()
            .any(|m| m.qualified_id() == standalone.qualified_id())
        {
            models.push(standalone);
        }
    }
    models.sort_by_key(|m| m.qualified_id());
}

/// Build the merged catalog from raw registry JSON plus fresh overrides.
///
/// Applies `build_catalog` to the registry JSON, then layers standalone
/// override entries that are not already present, re-sorted by qualified id.
fn merged_catalog(
    registry_json: &str,
    config: &UserConfig,
) -> Result<Vec<CatalogModel>, CatalogError> {
    let mut models = build_catalog(registry_json, config)?;
    merge_standalone(&mut models, config);
    Ok(models)
}

/// Read `registry-cache.json` and build its merged catalog.
///
/// Returns `None` when the cache is missing or corrupt. A corrupt cache is
/// treated exactly like a missing one: the caller falls through to the
/// cold-start inline fetch, whose success rewrites the cache.
fn load_cached_registry(paths: &Paths, config: &UserConfig) -> Option<Vec<CatalogModel>> {
    let text = fs::read_to_string(paths.registry_cache()).ok()?;
    merged_catalog(&text, config).ok()
}

/// Spawn a silent, fire-and-forget background refresh of the registry cache.
///
/// Fetches the registry (with the same 10-second timeout used everywhere
/// else) and atomically rewrites the cache on success; on failure it does
/// nothing — no warning, no stderr. If the process exits before the task
/// finishes, the update is simply lost and the cache keeps serving the
/// previous run's data. This task never affects the current conversation.
fn spawn_registry_refresh(paths: Paths, registry_url: String) {
    tokio::spawn(async move {
        if let Ok(json) = fetch_registry(&registry_url).await {
            write_registry_cache(&paths, &json).await;
        }
    });
}

/// Load the user config once and build the merged catalog, returning both.
///
/// The catalog is always built from persisted registry state plus fresh
/// `models.json` overrides. When the cache is present and parseable it is
/// served directly while a silent background refresh is spawned to update
/// the persisted copy for the *next* run. A missing or corrupt cache causes
/// a cold-start inline fetch (the only time a fetch is awaited); on success
/// the cache is atomically rewritten, and on failure the catalog falls back
/// to standalone entries with a warning on stderr.
async fn load_catalog_with_config(
    paths: &Paths,
    registry_url: &str,
) -> Result<(UserConfig, Vec<CatalogModel>), CatalogError> {
    let config = load_user_config(&paths.models_json())?;
    let models = match load_cached_registry(paths, &config) {
        Some(models) => {
            spawn_registry_refresh(paths.clone(), registry_url.to_string());
            models
        }
        None => match fetch_registry(registry_url).await {
            Ok(json) => {
                write_registry_cache(paths, &json).await;
                merged_catalog(&json, &config)?
            }
            Err(err) => {
                let standalone = standalone_models(&config);
                // Usable = model-level overrides can stand alone in the catalog, or a
                // complete provider-level override lets resolution synthesize on demand.
                let usable = !standalone.is_empty()
                    || config.providers.values().any(provider_override_complete);
                let notice = if usable {
                    "falling back to models.json overrides"
                } else {
                    "no models loaded (models.json has no usable overrides)"
                };
                let _ = writeln!(
                    io::stderr(),
                    "warning: models.dev fetch failed ({err}); {notice}"
                );
                standalone
            }
        },
    };
    Ok((config, models))
}

/// Load the merged model catalog, applying `models.json` overrides fresh.
///
/// The registry layer is persisted to `registry-cache.json` and refreshed
/// once per process start in the background, so fetch success, failure, or
/// slowness never affects the current conversation: a cache hit serves the
/// persisted registry while a silent background refresh updates the cache
/// for the next run. The cache holds the raw registry JSON, so
/// `models.json` overrides are re-applied — and re-read — on every call
/// and never serve stale edits. Only a cold start (missing or corrupt
/// cache) awaits the fetch inline, persisting the result on success; on
/// that fetch's failure, a warning is printed to stderr and the catalog
/// falls back to the standalone entries built purely from `models.json`
/// overrides (which may be empty).
///
/// # Errors
///
/// Returns [`CatalogError::ConfigIo`] / [`CatalogError::ConfigParse`] if the
/// user config is unreadable, or [`CatalogError::RegistryParse`] if the
/// cached or freshly fetched registry JSON cannot be deserialized. A network
/// failure is logged (cold start only) and yields a fallback catalog rather
/// than an error.
pub async fn load_catalog(
    paths: &Paths,
    registry_url: &str,
) -> Result<Vec<CatalogModel>, CatalogError> {
    load_catalog_with_config(paths, registry_url)
        .await
        .map(|(_, models)| models)
}

/// Resolve a `provider/model` spec into a ready-to-use [`CatalogModel`],
/// loading the merged catalog (cached registry + fresh `models.json`
/// overrides) and falling back to purely override-driven synthesis when the
/// spec's provider is absent.
///
/// Resolution rules, in order:
///
/// * The spec matches a merged catalog entry — that entry is returned.
/// * The spec's provider exists in the catalog but the model id is wrong —
///   the original [`CatalogError::ModelNotFound`] is returned unchanged, so
///   typos surface instead of being silently synthesized.
/// * The spec's provider is absent (cold-start fetch failed, or the provider
///   is unknown to the persisted registry) — a model is synthesized from
///   `models.json` overrides; if the overrides cannot supply non-empty
///   `api`, `base_url`, and `api_key`, the original
///   [`CatalogError::ModelNotFound`] is returned.
///
/// A bare model id (no `/`), or an empty provider or model id (e.g.
/// `deepseek/`), never synthesizes: there is no usable `provider/model`
/// split to override.
///
/// # Errors
///
/// Returns [`CatalogError::ConfigIo`] / [`CatalogError::ConfigParse`] if the
/// user config is unreadable, [`CatalogError::RegistryParse`] if the
/// cached or freshly fetched registry JSON cannot be deserialized, or
/// [`CatalogError::ModelNotFound`] per the resolution rules above. A network
/// failure is logged (cold start only) and never errors; it only narrows the
/// catalog to override-driven entries.
pub async fn resolve_catalog_model(
    paths: &Paths,
    registry_url: &str,
    spec: &str,
) -> Result<CatalogModel, CatalogError> {
    let (config, models) = load_catalog_with_config(paths, registry_url).await?;
    match resolve_model(&models, spec) {
        Ok(model) => Ok(model.clone()),
        Err(err @ CatalogError::ModelNotFound(_)) => {
            let Some((provider, id)) = spec.split_once('/') else {
                return Err(err);
            };
            if provider.is_empty() || id.is_empty() {
                return Err(err);
            }
            if models.iter().any(|m| m.provider == provider) {
                return Err(err);
            }
            synthesize_model(&config, provider, id).ok_or(err)
        }
        Err(other) => Err(other),
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]
    use super::*;
    use rstest::rstest;
    use std::io::Write;
    use tempfile::TempDir;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn fixture_registry() -> String {
        serde_json::json!({
            "deepseek": {
                "id": "deepseek",
                "env": ["DEEPSEEK_API_KEY"],
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://api.deepseek.com",
                "models": {
                    "deepseek-v4-flash": {
                        "id": "deepseek-v4-flash",
                        "name": "DeepSeek V4 Flash",
                        "reasoning": true,
                        "tool_call": true,
                        "modalities": { "input": ["text"] },
                        "limit": { "context": 1000000, "output": 384000 },
                        "cost": { "input": 0.14, "output": 0.28, "cache_read": 0.0028 }
                    }
                }
            },
            "anthropic": {
                "id": "anthropic",
                "env": ["ANTHROPIC_API_KEY"],
                "npm": "@ai-sdk/anthropic",
                "models": {
                    "claude": { "id": "claude", "name": "Claude", "reasoning": true, "tool_call": true }
                }
            },
            "openai": {
                "id": "openai",
                "env": ["OPENAI_API_KEY"],
                "npm": "@ai-sdk/openai",
                "models": {
                    "gpt-5": {
                        "id": "gpt-5",
                        "name": "GPT-5",
                        "limit": { "context": 400000, "output": 128000 },
                        "cost": { "input": 1.25, "output": 10.0 }
                    }
                }
            }
        })
        .to_string()
    }

    mod override_ {
        use super::*;

        #[test]
        fn rewrites_provider_keeps_model() {
            let config = UserConfig {
                providers: HashMap::from([(
                    "deepseek".into(),
                    ProviderOverride {
                        base_url: Some("https://cchub.example/v1".into()),
                        api: Some("openai-completions".into()),
                        api_key: Some("$CCHUB_API_KEY".into()),
                    },
                )]),
                ..UserConfig::default()
            };
            let models = build_catalog(&fixture_registry(), &config).unwrap();
            assert_eq!(models.len(), 3);
            let m = resolve_model(&models, "deepseek/deepseek-v4-flash").unwrap();
            assert_eq!(m.provider, "deepseek");
            assert_eq!(m.id, "deepseek-v4-flash");
            assert_eq!(m.qualified_id(), "deepseek/deepseek-v4-flash");
            assert_eq!(m.base_url, "https://cchub.example/v1");
            assert_eq!(m.api, "openai-completions");
            assert_eq!(m.api_key_env, "CCHUB_API_KEY");
            assert!(m.reasoning);
            assert_eq!(m.context_window, 1_000_000);
            assert_eq!(m.max_tokens, 384_000);
            assert_eq!(m.cost.input, 0.14);
            assert_eq!(m.cost.output, 0.28);
            let key = m
                .resolve_api_key(|k| {
                    if k == "CCHUB_API_KEY" {
                        Some("secret".into())
                    } else {
                        None
                    }
                })
                .unwrap();
            assert_eq!(key, "secret");
            assert!(m.resolve_api_key(|_| None).is_err());
        }

        #[test]
        fn without_api_keeps_npm_format() {
            let config = UserConfig {
                providers: HashMap::from([(
                    "deepseek".into(),
                    ProviderOverride {
                        base_url: Some("https://cchub.example/v1".into()),
                        api: None,
                        api_key: Some("$CCHUB_API_KEY".into()),
                    },
                )]),
                ..UserConfig::default()
            };
            let models = build_catalog(&fixture_registry(), &config).unwrap();
            let m = models.iter().find(|m| m.provider == "deepseek").unwrap();
            assert_eq!(m.api, "openai-completions");
            assert_eq!(m.base_url, "https://cchub.example/v1");
            assert_eq!(m.api_key_env, "CCHUB_API_KEY");
        }

        #[test]
        fn provider_level_applies_to_all() {
            let config = UserConfig {
                providers: HashMap::from([(
                    "openai".into(),
                    ProviderOverride {
                        base_url: None,
                        api: Some("openai-responses".into()),
                        api_key: None,
                    },
                )]),
                ..UserConfig::default()
            };
            let models = build_catalog(&fixture_registry(), &config).unwrap();
            let m = models.iter().find(|m| m.provider == "openai").unwrap();
            assert_eq!(m.id, "gpt-5");
            assert_eq!(m.api, "openai-responses");
            assert_eq!(m.base_url, "https://api.openai.com/v1");
        }

        #[test]
        fn model_level_beats_provider_and_npm() {
            let config = UserConfig {
                providers: HashMap::from([(
                    "openai".into(),
                    ProviderOverride {
                        base_url: Some("https://provider.example/v1".into()),
                        api: Some("openai-completions".into()),
                        api_key: Some("$PROVIDER_KEY".into()),
                    },
                )]),
                models: HashMap::from([(
                    "openai/gpt-5".into(),
                    ProviderOverride {
                        base_url: Some("https://model.example/v1".into()),
                        api: Some("openai-responses".into()),
                        api_key: Some("$MODEL_KEY".into()),
                    },
                )]),
            };
            let models = build_catalog(&fixture_registry(), &config).unwrap();
            assert_eq!(models.len(), 3);
            let m = models.iter().find(|m| m.provider == "openai").unwrap();
            assert_eq!(m.api, "openai-responses");
            assert_eq!(m.base_url, "https://model.example/v1");
            assert_eq!(m.api_key_env, "MODEL_KEY");
        }

        #[test]
        fn model_level_partial_merges_provider() {
            let config = UserConfig {
                providers: HashMap::from([(
                    "openai".into(),
                    ProviderOverride {
                        base_url: Some("https://provider.example/v1".into()),
                        api: Some("openai-completions".into()),
                        api_key: Some("$PROVIDER_KEY".into()),
                    },
                )]),
                models: HashMap::from([(
                    "openai/gpt-5".into(),
                    ProviderOverride {
                        base_url: None,
                        api: Some("openai-responses".into()),
                        api_key: None,
                    },
                )]),
            };
            let models = build_catalog(&fixture_registry(), &config).unwrap();
            let m = models.iter().find(|m| m.provider == "openai").unwrap();
            assert_eq!(m.api, "openai-responses");
            assert_eq!(m.base_url, "https://provider.example/v1");
            assert_eq!(m.api_key_env, "PROVIDER_KEY");
        }

        #[test]
        fn model_base_url_rescues_provider() {
            let registry = serde_json::json!({
                "cchub": {
                    "id": "cchub",
                    "env": ["CCHUB_API_KEY"],
                    "npm": "@ai-sdk/openai-compatible",
                    "models": { "m": { "id": "m" } }
                }
            })
            .to_string();
            let config = UserConfig {
                models: HashMap::from([(
                    "cchub/m".into(),
                    ProviderOverride {
                        base_url: Some("https://cchub.example/v1".into()),
                        api: None,
                        api_key: None,
                    },
                )]),
                ..UserConfig::default()
            };
            let models = build_catalog(&registry, &config).unwrap();
            assert_eq!(models.len(), 1);
            assert_eq!(models[0].base_url, "https://cchub.example/v1");
            assert_eq!(models[0].api, "openai-completions");
            assert_eq!(models[0].api_key_env, "CCHUB_API_KEY");
        }

        #[test]
        fn empty_string_falls_through() {
            let config = UserConfig {
                providers: HashMap::from([(
                    "openai".into(),
                    ProviderOverride {
                        base_url: Some("https://provider.example/v1".into()),
                        api: Some("".into()),
                        api_key: Some("$PROVIDER_KEY".into()),
                    },
                )]),
                models: HashMap::from([(
                    "openai/gpt-5".into(),
                    ProviderOverride {
                        base_url: Some("".into()),
                        api: Some("".into()),
                        api_key: None,
                    },
                )]),
            };
            let models = build_catalog(&fixture_registry(), &config).unwrap();
            let m = models.iter().find(|m| m.provider == "openai").unwrap();
            assert_eq!(m.api, "openai-completions");
            assert_eq!(m.base_url, "https://provider.example/v1");
            assert_eq!(m.api_key_env, "PROVIDER_KEY");
        }

        /// A provider-level override is "complete" (usable for the fetch-failure
        /// warning branch) only when api, base_url, and api_key are all
        /// non-empty: such an override lets `resolve_catalog_model` synthesize
        /// any requested model id under the provider.
        #[test]
        fn provider_override_complete_requires_all_fields() {
            let complete = ProviderOverride {
                api: Some("openai-completions".into()),
                base_url: Some("https://provider.example/v1".into()),
                api_key: Some("$PROVIDER_KEY".into()),
            };
            assert!(provider_override_complete(&complete));
            assert!(!provider_override_complete(&ProviderOverride {
                api: None,
                ..complete.clone()
            }));
            assert!(!provider_override_complete(&ProviderOverride {
                base_url: None,
                ..complete.clone()
            }));
            assert!(!provider_override_complete(&ProviderOverride {
                api_key: None,
                ..complete.clone()
            }));
            // Empty strings are not usable either.
            assert!(!provider_override_complete(&ProviderOverride {
                api: Some("".into()),
                ..complete.clone()
            }));
            assert!(!provider_override_complete(&ProviderOverride {
                base_url: Some("".into()),
                ..complete.clone()
            }));
            assert!(!provider_override_complete(&ProviderOverride {
                api_key: Some("".into()),
                ..complete.clone()
            }));
        }
    }

    mod canonical {
        use super::*;

        #[test]
        fn mounts_npm_format_and_base_url() {
            let models = build_catalog(&fixture_registry(), &UserConfig::default()).unwrap();
            assert_eq!(models.len(), 3);
            assert_eq!(models[0].provider, "anthropic");

            let anthropic = &models[0];
            assert_eq!(anthropic.id, "claude");
            assert_eq!(anthropic.api, "anthropic-messages");
            assert_eq!(anthropic.base_url, "https://api.anthropic.com/v1");
            assert_eq!(anthropic.api_key_env, "ANTHROPIC_API_KEY");

            let openai = models.iter().find(|m| m.provider == "openai").unwrap();
            assert_eq!(openai.id, "gpt-5");
            assert_eq!(openai.api, "openai-completions");
            assert_eq!(openai.base_url, "https://api.openai.com/v1");
            assert_eq!(openai.api_key_env, "OPENAI_API_KEY");

            let deepseek = models.iter().find(|m| m.provider == "deepseek").unwrap();
            assert_eq!(deepseek.api, "openai-completions");
            assert_eq!(deepseek.base_url, "https://api.deepseek.com");
            assert_eq!(deepseek.api_key_env, "DEEPSEEK_API_KEY");
        }

        #[test]
        fn cloud_and_unknown_npm_skipped() {
            let registry = serde_json::json!({
                "amazon-bedrock": {
                    "id": "amazon-bedrock",
                    "env": ["BEDROCK_API_KEY"],
                    "npm": "@ai-sdk/amazon-bedrock",
                    "api": "https://bedrock.example",
                    "models": { "claude-x": { "id": "claude-x" } }
                },
                "qvac": {
                    "id": "qvac",
                    "env": ["QVAC_API_KEY"],
                    "npm": "@qvac/ai-sdk-provider",
                    "api": "https://qvac.example",
                    "models": { "m1": { "id": "m1" } }
                },
                "madeup": {
                    "id": "madeup",
                    "env": ["MADEUP_API_KEY"],
                    "npm": "@ai-sdk/madeup",
                    "api": "https://madeup.example",
                    "models": { "m1": { "id": "m1" } }
                }
            })
            .to_string();
            let models = build_catalog(&registry, &UserConfig::default()).unwrap();
            assert!(models.is_empty());

            // The npm gate applies before provider overrides: an override cannot
            // rescue an unmapped npm package.
            let config = UserConfig {
                providers: HashMap::from([(
                    "madeup".into(),
                    ProviderOverride {
                        base_url: Some("https://override.example".into()),
                        api: Some("openai-completions".into()),
                        api_key: Some("$MADEUP_API_KEY".into()),
                    },
                )]),
                ..UserConfig::default()
            };
            let models = build_catalog(&registry, &config).unwrap();
            assert!(models.is_empty());
        }

        #[rstest]
        #[case::openai(
            "openai",
            "@ai-sdk/openai",
            "openai-completions",
            "https://api.openai.com/v1"
        )]
        #[case::anthropic(
            "anthropic",
            "@ai-sdk/anthropic",
            "anthropic-messages",
            "https://api.anthropic.com/v1"
        )]
        #[case::google(
            "google",
            "@ai-sdk/google",
            "google-genai",
            "https://generativelanguage.googleapis.com/v1beta"
        )]
        #[case::cohere("cohere", "@ai-sdk/cohere", "cohere-chat", "https://api.cohere.com/v2")]
        #[case::xai("xai", "@ai-sdk/xai", "openai-completions", "https://api.x.ai/v1")]
        #[case::mistral(
            "mistral",
            "@ai-sdk/mistral",
            "openai-completions",
            "https://api.mistral.ai/v1"
        )]
        #[case::groq(
            "groq",
            "@ai-sdk/groq",
            "openai-completions",
            "https://api.groq.com/openai/v1"
        )]
        #[case::perplexity(
            "perplexity",
            "@ai-sdk/perplexity",
            "openai-completions",
            "https://api.perplexity.ai"
        )]
        #[case::togetherai(
            "togetherai",
            "@ai-sdk/togetherai",
            "openai-completions",
            "https://api.together.xyz/v1"
        )]
        #[case::cerebras(
            "cerebras",
            "@ai-sdk/cerebras",
            "openai-completions",
            "https://api.cerebras.ai/v1"
        )]
        #[case::deepinfra(
            "deepinfra",
            "@ai-sdk/deepinfra",
            "openai-completions",
            "https://api.deepinfra.com/v1/openai"
        )]
        fn default_base_url_mounts_provider(
            #[case] id: &str,
            #[case] npm: &str,
            #[case] expected_api: &str,
            #[case] expected_url: &str,
        ) {
            // Hardcoded expected formats (spec §3), NOT derived from
            // NPM_API_FORMATS: rewiring e.g. @ai-sdk/google to
            // openai-completions must fail here. Each provider runs as its
            // own isolated case, so a single regression pinpoints the culprit.
            let env_key = format!("{id}_API_KEY");
            let registry = serde_json::json!({
                id: {
                    "id": id,
                    "env": [env_key],
                    "npm": npm,
                    "models": { "m": { "id": "m" } }
                }
            })
            .to_string();
            let models = build_catalog(&registry, &UserConfig::default()).unwrap();
            assert_eq!(
                models.len(),
                1,
                "provider {id} must mount exactly one model"
            );
            assert_eq!(models[0].base_url, expected_url, "provider {id} base_url");
            assert_eq!(
                models[0].api, expected_api,
                "provider {id} derived the wrong api"
            );
        }

        #[test]
        fn every_npm_api_format_mounts() {
            // Spec §3 pins exactly these 20 npm → api_format mappings. Hardcode
            // the list: deleting an entry, adding one, or rewiring a mapping fails
            // the equality assert even though the mount loop walks the hardcoded
            // list. The loop itself proves every entry is actually wired through
            // build_catalog (the 8 official-package entries would otherwise have
            // zero coverage).
            const EXPECTED: &[(&str, &str)] = &[
                ("@ai-sdk/openai-compatible", "openai-completions"),
                ("@ai-sdk/openai", "openai-completions"),
                ("@ai-sdk/xai", "openai-completions"),
                ("@ai-sdk/anthropic", "anthropic-messages"),
                ("@ai-sdk/google", "google-genai"),
                ("@ai-sdk/cohere", "cohere-chat"),
                ("@ai-sdk/mistral", "openai-completions"),
                ("@ai-sdk/groq", "openai-completions"),
                ("@ai-sdk/perplexity", "openai-completions"),
                ("@ai-sdk/togetherai", "openai-completions"),
                ("@ai-sdk/cerebras", "openai-completions"),
                ("@ai-sdk/deepinfra", "openai-completions"),
                ("@ai-sdk/deepseek", "openai-completions"),
                ("@ai-sdk/moonshotai", "openai-completions"),
                ("@ai-sdk/alibaba", "openai-completions"),
                ("@ai-sdk/minimax", "openai-completions"),
                ("@ai-sdk/fireworks", "openai-completions"),
                ("@ai-sdk/huggingface", "openai-completions"),
                ("@ai-sdk/baseten", "openai-completions"),
                ("@ai-sdk/gmicloud", "openai-completions"),
            ];
            assert_eq!(
                NPM_API_FORMATS, EXPECTED,
                "spec §3 npm→api_format table drifted"
            );
            for (i, &(npm, api)) in EXPECTED.iter().enumerate() {
                let id = format!("p{i}");
                let registry = serde_json::json!({
                    (id.clone()): {
                        "id": id,
                        "env": ["PROVIDER_API_KEY"],
                        "npm": npm,
                        "api": "https://provider.example/v1",
                        "models": { "m": { "id": "m" } }
                    }
                })
                .to_string();
                let models = build_catalog(&registry, &UserConfig::default()).unwrap();
                assert_eq!(models.len(), 1, "{npm} must mount exactly one model");
                assert_eq!(models[0].api, api, "{npm} derived the wrong format");
                assert_eq!(models[0].base_url, "https://provider.example/v1");
            }
        }

        #[test]
        fn resolve_requires_qualified_spec() {
            let models = build_catalog(&fixture_registry(), &UserConfig::default()).unwrap();
            let m = resolve_model(&models, "deepseek/deepseek-v4-flash").unwrap();
            assert_eq!(m.id, "deepseek-v4-flash");
            assert_eq!(m.qualified_id(), "deepseek/deepseek-v4-flash");
            assert!(resolve_model(&models, "deepseek/nope").is_err());
            assert!(resolve_model(&models, "bare-model-id").is_err());
            assert_eq!(parse_thinking("high").unwrap(), ThinkingLevel::High);
        }
    }

    mod load {
        use super::*;
        use std::time::Instant;

        /// A fresh tempdir has no cache, so `load_catalog` cold-starts against
        /// the registry, serves the catalog, and persists the raw registry JSON
        /// to `registry-cache.json`.
        #[tokio::test]
        async fn cold_start_writes_cache() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(models.len(), 3);
            assert!(
                paths.registry_cache().exists(),
                "cache file must exist after cold start"
            );
            let cached = fs::read_to_string(paths.registry_cache()).unwrap();
            assert_eq!(
                cached,
                fixture_registry(),
                "cache must hold the raw registry JSON"
            );
        }

        /// After a successful cold start, a broken registry does not empty the
        /// catalog: the persisted cache is served instead.
        #[tokio::test]
        async fn second_call_uses_cache_when_fetch_fails() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(models.len(), 3);

            server.reset().await;
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(500))
                .mount(&server)
                .await;
            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(models.len(), 3, "cache must survive a registry failure");
            let m = resolve_model(&models, "deepseek/deepseek-v4-flash").unwrap();
            assert_eq!(m.id, "deepseek-v4-flash");
        }

        /// A background refresh spawned on cache hit updates the persisted
        /// raw JSON for the *next* run; the current conversation stays on
        /// the old cache until then.
        #[tokio::test]
        async fn background_refresh_updates_cache_for_next_run() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(models.len(), 3, "cold-start catalog");

            // Registry now offers a new provider. The next call must still
            // serve the old cache while a background refresh starts.
            server.reset().await;
            let mut updated: serde_json::Value = serde_json::from_str(&fixture_registry()).unwrap();
            updated["cchub"] = serde_json::json!({
                "id": "cchub",
                "env": ["CCHUB_API_KEY"],
                "npm": "@ai-sdk/openai-compatible",
                "api": "https://cchub.example/v1",
                "models": { "steve": { "id": "steve", "name": "Steve" } }
            });
            let updated_json = updated.to_string();
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(updated_json))
                .mount(&server)
                .await;

            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(models.len(), 3, "cache-hit must serve old registry");
            assert!(resolve_model(&models, "cchub/steve").is_err());

            // Poll the cache file until the refresh lands; never use a fixed
            // sleep. If the runtime dropped us the spawn would never run.
            let cache_path = paths.registry_cache();
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                if fs::read_to_string(&cache_path)
                    .map(|text| text.contains("cchub"))
                    .unwrap_or(false)
                {
                    break;
                }
                assert!(
                    Instant::now() < deadline,
                    "background refresh never updated the cache"
                );
                tokio::time::sleep(Duration::from_millis(50)).await;
            }

            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(models.len(), 4, "subsequent run sees refreshed cache");
            assert!(resolve_model(&models, "cchub/steve").is_ok());
        }

        /// With a populated cache, editing `models.json` takes effect on the
        /// very next call — overrides are never served stale.
        #[tokio::test]
        async fn models_json_edits_apply_with_cache_present() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(
                paths.models_json(),
                r#"{"providers":{"deepseek":{"baseUrl":"https://cchub.example/v1","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}"#,
            )
            .unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let models = load_catalog(&paths, &url).await.unwrap();
            let m = resolve_model(&models, "deepseek/deepseek-v4-flash").unwrap();
            assert_eq!(m.base_url, "https://cchub.example/v1");
            assert!(paths.registry_cache().exists());

            fs::write(
                paths.models_json(),
                r#"{"providers":{"deepseek":{"baseUrl":"https://edited.example/v1","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}"#,
            )
            .unwrap();
            let models = load_catalog(&paths, &url).await.unwrap();
            let m = resolve_model(&models, "deepseek/deepseek-v4-flash").unwrap();
            assert_eq!(m.base_url, "https://edited.example/v1");
        }

        /// A corrupt cache is treated exactly like a missing one: the cold-start
        /// inline fetch recovers, the catalog is built, and the cache is rewritten
        /// with valid JSON.
        #[tokio::test]
        async fn corrupt_cache_treated_as_missing() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(paths.registry_cache(), b"\x00\xff not json {{").unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(
                models.len(),
                3,
                "cold-start fetch must recover from corrupt cache"
            );
            let cached = fs::read_to_string(paths.registry_cache()).unwrap();
            assert_eq!(
                cached,
                fixture_registry(),
                "corrupt cache must be rewritten with raw JSON"
            );
        }

        /// A model-level override for a provider absent from the registry must
        /// still land in the merged catalog on a successful fetch.
        #[tokio::test]
        async fn merges_standalone_models_on_fetch_success() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(
                paths.models_json(),
                r#"{"models":{"cchub/steve":{"baseUrl":"http://cchub.example/v1","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}"#,
            )
            .unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(models.len(), 4, "3 registry entries + 1 standalone");
            let m = resolve_model(&models, "cchub/steve").unwrap();
            assert_eq!(m.provider, "cchub");
            assert_eq!(m.id, "steve");
            assert_eq!(m.base_url, "http://cchub.example/v1");
            assert_eq!(m.api, "openai-completions");
            assert_eq!(m.api_key_env, "CCHUB_API_KEY");
        }

        /// A model-level override for a registry-present model must not
        /// duplicate the entry; registry-derived metadata (name/cost/limits)
        /// wins, with override base_url/api/api_key_env layered on top.
        #[tokio::test]
        async fn registry_model_entry_not_duplicated() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(
                paths.models_json(),
                r#"{"models":{"deepseek/deepseek-v4-flash":{"baseUrl":"http://alt.example/v1","api":"openai-completions","apiKey":"$ALT_KEY"}}}"#,
            )
            .unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(models.len(), 3, "override must not duplicate the entry");
            let m = resolve_model(&models, "deepseek/deepseek-v4-flash").unwrap();
            // Registry-derived metadata wins.
            assert_eq!(m.name, "DeepSeek V4 Flash");
            assert!(m.reasoning);
            assert_eq!(m.context_window, 1_000_000);
            assert_eq!(m.max_tokens, 384_000);
            assert_eq!(m.cost.input, 0.14);
            // Override fields still layer onto the registry entry.
            assert_eq!(m.base_url, "http://alt.example/v1");
            assert_eq!(m.api_key_env, "ALT_KEY");
        }

        /// On fetch failure the catalog is exactly the standalone entries built
        /// from model-level overrides — no registry data, no duplicates.
        #[tokio::test]
        async fn fetch_failure_returns_standalone_models() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(
                paths.models_json(),
                r#"{"models":{"cchub/steve":{"baseUrl":"http://cchub.example/v1","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}"#,
            )
            .unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(500))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let models = load_catalog(&paths, &url).await.unwrap();
            assert_eq!(models.len(), 1);
            let m = resolve_model(&models, "cchub/steve").unwrap();
            assert_eq!(m.base_url, "http://cchub.example/v1");
            assert_eq!(m.api_key_env, "CCHUB_API_KEY");
            assert!(
                resolve_model(&models, "deepseek/deepseek-v4-flash").is_err(),
                "registry entries must be absent after fetch failure"
            );
        }
    }

    mod resolve_catalog_model {
        use super::*;

        /// Resolve via a purely provider-level override when the registry fetch
        /// fails — the issue #59 reporter's exact config shape (no `models` key).
        #[tokio::test]
        async fn fetch_failure_synthesizes_provider_level_override() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(
                paths.models_json(),
                r#"{"providers":{"deepseek":{"baseUrl":"http://localhost:23000/v1","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}"#,
            )
            .unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(500))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let m = resolve_catalog_model(&paths, &url, "deepseek/deepseek-v4-flash")
                .await
                .unwrap();
            assert_eq!(m.provider, "deepseek");
            assert_eq!(m.id, "deepseek-v4-flash");
            assert_eq!(m.base_url, "http://localhost:23000/v1");
            assert_eq!(m.api, "openai-completions");
            assert_eq!(m.api_key_env, "CCHUB_API_KEY");
            assert!(!m.reasoning);
            assert_eq!(m.max_tokens, 0);
        }

        /// A provider override without `apiKey` cannot synthesize a model, so
        /// the original `ModelNotFound` is returned unchanged.
        #[tokio::test]
        async fn fetch_failure_missing_api_key_is_not_found() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(
                paths.models_json(),
                r#"{"providers":{"deepseek":{"baseUrl":"http://localhost:23000/v1","api":"openai-completions"}}}"#,
            )
            .unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(500))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let err = resolve_catalog_model(&paths, &url, "deepseek/deepseek-v4-flash")
                .await
                .unwrap_err();
            assert!(matches!(err, CatalogError::ModelNotFound(_)), "{err}");
            assert_eq!(
                err.to_string(),
                "model not found: deepseek/deepseek-v4-flash"
            );
        }

        /// Typo protection: the provider exists in the fetched catalog, so a
        /// wrong model id must stay `ModelNotFound` even when a complete
        /// provider override is present — no synthesis on the typo path.
        #[tokio::test]
        async fn known_provider_wrong_model_stays_not_found() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(
                paths.models_json(),
                r#"{"providers":{"deepseek":{"baseUrl":"http://cchub.example/v1","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}"#,
            )
            .unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let err = resolve_catalog_model(&paths, &url, "deepseek/not-a-model")
                .await
                .unwrap_err();
            assert!(matches!(err, CatalogError::ModelNotFound(_)), "{err}");
            assert_eq!(err.to_string(), "model not found: deepseek/not-a-model");
        }

        /// Provider absent from the fetched registry but fully configured at the
        /// provider level: synthesize the model and resolve it.
        #[tokio::test]
        async fn provider_absent_from_registry_synthesizes_override() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(
                paths.models_json(),
                r#"{"providers":{"cchub":{"baseUrl":"http://cchub.example/v1","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}"#,
            )
            .unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .and(path("/api.json"))
                .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let m = resolve_catalog_model(&paths, &url, "cchub/steve")
                .await
                .unwrap();
            assert_eq!(m.provider, "cchub");
            assert_eq!(m.id, "steve");
            assert_eq!(m.base_url, "http://cchub.example/v1");
            assert_eq!(m.api, "openai-completions");
            assert_eq!(m.api_key_env, "CCHUB_API_KEY");
        }

        /// A trailing slash (empty model id) must not synthesize an empty-id
        /// model: `deepseek/` stays `ModelNotFound` even with a complete
        /// provider-level override present.
        #[tokio::test]
        async fn empty_model_id_stays_not_found() {
            let tmp = TempDir::new().unwrap();
            let paths = Paths::from_config_dir(tmp.path());
            fs::write(
                paths.models_json(),
                r#"{"providers":{"deepseek":{"baseUrl":"http://localhost:23000/v1","api":"openai-completions","apiKey":"$CCHUB_API_KEY"}}}"#,
            )
            .unwrap();
            let server = MockServer::start().await;
            Mock::given(method("GET"))
                .respond_with(ResponseTemplate::new(500))
                .mount(&server)
                .await;

            let url = format!("{}/api.json", server.uri());
            let err = resolve_catalog_model(&paths, &url, "deepseek/")
                .await
                .unwrap_err();
            assert!(matches!(err, CatalogError::ModelNotFound(_)), "{err}");
            assert_eq!(err.to_string(), "model not found: deepseek/");
        }
    }

    mod config_parse {
        use super::*;

        #[test]
        fn invalid_config_is_error() {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("models.json");
            let mut f = fs::File::create(&path).unwrap();
            f.write_all(b"{not json").unwrap();
            assert!(load_user_config(&path).is_err());
        }

        #[test]
        fn without_models_key_parses() {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("models.json");
            fs::write(
                &path,
                r#"{ "providers": { "openai": { "baseUrl": "https://x.example/v1" } } }"#,
            )
            .unwrap();
            let config = load_user_config(&path).unwrap();
            assert!(config.models.is_empty());
            assert_eq!(
                config.providers["openai"].base_url.as_deref(),
                Some("https://x.example/v1")
            );
        }

        #[test]
        fn model_key_deserializes_camel_case() {
            let tmp = TempDir::new().unwrap();
            let path = tmp.path().join("models.json");
            fs::write(
                &path,
                r#"{ "models": { "openai/gpt-5": { "api": "openai-responses", "baseUrl": "https://m.example/v1", "apiKey": "$MODEL_KEY" } } }"#,
            )
            .unwrap();
            let config = load_user_config(&path).unwrap();
            let ov = &config.models["openai/gpt-5"];
            assert_eq!(ov.api.as_deref(), Some("openai-responses"));
            assert_eq!(ov.base_url.as_deref(), Some("https://m.example/v1"));
            assert_eq!(ov.api_key.as_deref(), Some("$MODEL_KEY"));
        }
    }
}
