//! Model registry: models.dev fetch/cache, user `models.json` overrides,
//! credential resolution, and catalog lookups.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pi_agent_core::types::{Model, ModelCost, ModelInput, ThinkingLevel};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::formats::openai_completions;
use crate::formats::anthropic_messages;

pub const DEFAULT_REGISTRY_URL: &str = "https://models.dev/api.json";
pub const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);

const CACHE_FILE: &str = "catalog-cache.json";
const CONFIG_FILE: &str = "models.json";

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
    #[error("no catalog cache at {0} and models.dev is unreachable")]
    NoCache(String),
    #[error("model not found: {0}")]
    ModelNotFound(String),
    #[error("API key environment variable {0} is not set")]
    MissingApiKey(String),
}

#[derive(Debug, Clone)]
pub struct Paths {
    pub config_dir: PathBuf,
}

impl Paths {
    pub fn from_config_dir(config_dir: impl Into<PathBuf>) -> Self {
        Self {
            config_dir: config_dir.into(),
        }
    }

    pub fn default_user() -> Self {
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("/"));
        Self {
            config_dir: home.join(".config").join("aaos"),
        }
    }

    pub fn models_json(&self) -> PathBuf {
        self.config_dir.join(CONFIG_FILE)
    }

    pub fn cache_json(&self) -> PathBuf {
        self.config_dir.join(CACHE_FILE)
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct UserConfig {
    #[serde(default)]
    pub providers: HashMap<String, ProviderOverride>,
    /// 模型级覆盖，键为 `provider/model` 限定 id，优先级高于提供商级覆盖与 npm 推导。
    #[serde(default)]
    pub models: HashMap<String, ProviderOverride>,
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ProviderOverride {
    pub base_url: Option<String>,
    pub api: Option<String>,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CatalogModel {
    pub id: String,
    pub name: String,
    pub api: String,
    pub provider: String,
    pub base_url: String,
    pub reasoning: bool,
    pub tool_call: bool,
    pub input: Vec<String>,
    pub cost: ModelCostDto,
    pub context_window: u64,
    pub max_tokens: u64,
    pub api_key_env: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ModelCostDto {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}

impl CatalogModel {
    pub fn qualified_id(&self) -> String {
        format!("{}/{}", self.provider, self.id)
    }

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

    pub fn resolve_api_key(
        &self,
        getenv: impl Fn(&str) -> Option<String>,
    ) -> Result<String, CatalogError> {
        getenv(&self.api_key_env)
            .filter(|s| !s.is_empty())
            .ok_or_else(|| CatalogError::MissingApiKey(self.api_key_env.clone()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CachedCatalog {
    pub fetched_at_unix: u64,
    pub warning: Option<String>,
    pub models: Vec<CatalogModel>,
}

impl CachedCatalog {
    pub fn is_fresh(&self, now: SystemTime, ttl: Duration) -> bool {
        let fetched = UNIX_EPOCH + Duration::from_secs(self.fetched_at_unix);
        now.duration_since(fetched)
            .map(|age| age < ttl)
            .unwrap_or(true)
    }

    pub fn get(&self, provider: &str, model_id: &str) -> Option<&CatalogModel> {
        self.models
            .iter()
            .find(|m| m.provider == provider && m.id == model_id)
    }

    /// Resolve a `provider/model` spec. Bare model ids are rejected: a
    /// fallback provider is a product decision that does not live here.
    pub fn resolve(&self, spec: &str) -> Result<&CatalogModel, CatalogError> {
        let Some((provider, id)) = spec.split_once('/') else {
            return Err(CatalogError::ModelNotFound(spec.to_string()));
        };
        self.get(provider, id)
            .ok_or_else(|| CatalogError::ModelNotFound(spec.to_string()))
    }
}

pub fn load_user_config(path: &Path) -> Result<UserConfig, CatalogError> {
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

pub fn read_cache(path: &Path) -> Option<CachedCatalog> {
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

pub fn write_cache(path: &Path, catalog: &CachedCatalog) -> Result<(), CatalogError> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|source| CatalogError::ConfigIo {
            path: parent.display().to_string(),
            source,
        })?;
    }
    let text = serde_json::to_string_pretty(catalog).expect("catalog serializes");
    fs::write(path, text).map_err(|source| CatalogError::ConfigIo {
        path: path.display().to_string(),
        source,
    })
}

pub fn format_model_line(model: &CatalogModel) -> String {
    format!(
        "{}  provider={}  context={}  max_tokens={}  reasoning={}  tool_call={}  cost.input={}  cost.output={}",
        model.qualified_id(),
        model.provider,
        model.context_window,
        model.max_tokens,
        model.reasoning,
        model.tool_call,
        model.cost.input,
        model.cost.output
    )
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
    tool_call: Option<bool>,
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
/// 复用 [`anthropic_messages::API`]（issue 08 已落地）。其余三种格式
/// （google-genai/cohere-chat/openai-responses）的 adapter 尚未落地，
/// 由 issue 09–10 各自声明 `API` const，届时替换字符串字面量。
/// 云托管（bedrock/vertex/azure/gateway/vercel）、社区包与未知 npm 不在表中：
/// 对应提供商整体跳过，绝不静默回退到 openai-completions。
const NPM_API_FORMATS: &[(&str, &str)] = &[
    ("@ai-sdk/openai-compatible", openai_completions::API),
    ("@ai-sdk/openai", openai_completions::API),
    ("@ai-sdk/xai", openai_completions::API),
    ("@ai-sdk/anthropic", anthropic_messages::API),
    ("@ai-sdk/google", "google-genai"),
    ("@ai-sdk/cohere", "cohere-chat"),
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

pub fn build_catalog(
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
                tool_call: model.tool_call.unwrap_or(false),
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

pub async fn fetch_registry(url: &str) -> Result<String, CatalogError> {
    let response = reqwest::get(url)
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

pub struct RefreshOutcome {
    pub catalog: CachedCatalog,
    pub used_cache: bool,
}

pub async fn refresh_catalog(
    paths: &Paths,
    registry_url: &str,
    now: SystemTime,
) -> Result<RefreshOutcome, CatalogError> {
    let config = load_user_config(&paths.models_json())?;
    match fetch_registry(registry_url).await {
        Ok(json) => {
            let models = build_catalog(&json, &config)?;
            let catalog = CachedCatalog {
                fetched_at_unix: now.duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                warning: None,
                models,
            };
            write_cache(&paths.cache_json(), &catalog)?;
            Ok(RefreshOutcome {
                catalog,
                used_cache: false,
            })
        }
        Err(err) => {
            if let Some(old) = read_cache(&paths.cache_json()) {
                let mut catalog = old;
                catalog.warning = Some(format!(
                    "models.dev refresh failed ({err}); keeping cached catalog"
                ));
                Ok(RefreshOutcome {
                    catalog,
                    used_cache: true,
                })
            } else {
                Err(CatalogError::NoCache(
                    paths.cache_json().display().to_string(),
                ))
            }
        }
    }
}

pub async fn load_catalog(
    paths: &Paths,
    registry_url: &str,
    now: SystemTime,
    ttl: Duration,
) -> Result<RefreshOutcome, CatalogError> {
    if let Some(cache) = read_cache(&paths.cache_json())
        && cache.is_fresh(now, ttl)
    {
        return Ok(RefreshOutcome {
            catalog: cache,
            used_cache: true,
        });
    }
    refresh_catalog(paths, registry_url, now).await
}

#[cfg(test)]
mod tests {
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

    #[test]
    fn override_rewrites_deepseek_to_cchub_keeps_model_metadata() {
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
        let cat = CachedCatalog {
            fetched_at_unix: 1,
            warning: None,
            models,
        };
        let m = cat.resolve("deepseek/deepseek-v4-flash").unwrap();
        assert_eq!(m.provider, "deepseek");
        assert_eq!(m.id, "deepseek-v4-flash");
        assert_eq!(m.qualified_id(), "deepseek/deepseek-v4-flash");
        assert_eq!(m.base_url, "https://cchub.example/v1");
        assert_eq!(m.api, "openai-completions");
        assert_eq!(m.api_key_env, "CCHUB_API_KEY");
        assert!(m.reasoning);
        assert!(m.tool_call);
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
    fn canonical_providers_mount_npm_derived_format_and_default_base_url() {
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
    fn resolve_requires_qualified_spec() {
        let models = build_catalog(&fixture_registry(), &UserConfig::default()).unwrap();
        let cat = CachedCatalog {
            fetched_at_unix: 1,
            warning: None,
            models,
        };
        let m = cat.resolve("deepseek/deepseek-v4-flash").unwrap();
        assert_eq!(m.id, "deepseek-v4-flash");
        assert_eq!(m.qualified_id(), "deepseek/deepseek-v4-flash");
        assert!(cat.resolve("deepseek/nope").is_err());
        assert!(cat.resolve("bare-model-id").is_err());
        assert_eq!(parse_thinking("high").unwrap(), ThinkingLevel::High);
    }

    #[tokio::test]
    async fn refresh_writes_cache_and_keeps_old_on_network_failure() {
        let tmp = TempDir::new().unwrap();
        let paths = Paths::from_config_dir(tmp.path());
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api.json"))
            .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
            .mount(&server)
            .await;

        let url = format!("{}/api.json", server.uri());
        let now = UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let first = refresh_catalog(&paths, &url, now).await.unwrap();
        assert!(!first.used_cache);
        assert!(paths.cache_json().exists());
        assert_eq!(first.catalog.models[0].id, "claude");
        assert_eq!(first.catalog.models.len(), 3);

        let line = format_model_line(first.catalog.resolve("deepseek/deepseek-v4-flash").unwrap());
        assert!(line.contains("deepseek/deepseek-v4-flash"));
        assert!(line.contains("reasoning=true"));
        assert!(line.contains("tool_call=true"));

        server.reset().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .mount(&server)
            .await;
        let second = refresh_catalog(&paths, &url, now + Duration::from_secs(10))
            .await
            .unwrap();
        assert!(second.used_cache);
        assert!(
            second
                .catalog
                .warning
                .unwrap()
                .contains("keeping cached catalog")
        );
        assert_eq!(second.catalog.models.len(), 3);
    }

    #[tokio::test]
    async fn startup_uses_fresh_cache_without_fetch() {
        let tmp = TempDir::new().unwrap();
        let paths = Paths::from_config_dir(tmp.path());
        let catalog = CachedCatalog {
            fetched_at_unix: 1_700_000_000,
            warning: None,
            models: build_catalog(&fixture_registry(), &UserConfig::default()).unwrap(),
        };
        write_cache(&paths.cache_json(), &catalog).unwrap();
        let now = UNIX_EPOCH + Duration::from_secs(1_700_000_000 + 60);
        let loaded = load_catalog(&paths, "http://127.0.0.1:1/missing", now, CACHE_TTL)
            .await
            .unwrap();
        assert!(loaded.used_cache);
        assert_eq!(loaded.catalog.models.len(), 3);
    }

    #[test]
    fn invalid_config_is_error() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("models.json");
        let mut f = fs::File::create(&path).unwrap();
        f.write_all(b"{not json").unwrap();
        assert!(load_user_config(&path).is_err());
    }

    #[tokio::test]
    async fn stale_cache_refreshes_from_registry() {
        let tmp = TempDir::new().unwrap();
        let paths = Paths::from_config_dir(tmp.path());
        let stale = CachedCatalog {
            fetched_at_unix: 1,
            warning: None,
            models: vec![],
        };
        write_cache(&paths.cache_json(), &stale).unwrap();
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_string(fixture_registry()))
            .mount(&server)
            .await;
        let now = UNIX_EPOCH + Duration::from_secs(1 + CACHE_TTL.as_secs() + 1);
        let loaded = load_catalog(
            &paths,
            &format!("{}/api.json", server.uri()),
            now,
            CACHE_TTL,
        )
        .await
        .unwrap();
        assert!(!loaded.used_cache);
        assert_eq!(loaded.catalog.models[0].id, "claude");
        assert_eq!(loaded.catalog.models.len(), 3);
    }

    #[test]
    fn cloud_hosted_and_unknown_npm_providers_are_skipped() {
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

    #[test]
    fn provider_override_without_api_keeps_npm_derived_format() {
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
    fn provider_level_api_override_applies_to_all_models() {
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
    fn model_level_override_beats_provider_level_and_npm_derivation() {
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
    fn model_level_partial_override_merges_with_provider_level() {
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
    fn model_level_base_url_rescues_provider_without_any_base() {
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
    fn empty_string_overrides_fall_through_to_next_priority() {
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
    fn every_default_base_url_mounts_canonical_provider(
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
    fn every_npm_api_format_entry_mounts() {
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
    fn models_json_without_models_key_still_parses() {
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
    fn models_json_model_key_deserializes_camel_case() {
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
