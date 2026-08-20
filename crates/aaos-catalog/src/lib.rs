use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use pi_agent_core::types::{Model, ModelCost, ModelInput, ThinkingLevel};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub const DEFAULT_PROVIDER: &str = "deepseek";
pub const DEFAULT_MODEL_ID: &str = "deepseek-v4-flash";
pub const DEFAULT_MODEL_REF: &str = "deepseek/deepseek-v4-flash";
pub const DEFAULT_THINKING: ThinkingLevel = ThinkingLevel::High;
pub const DEFAULT_REGISTRY_URL: &str = "https://models.dev/api.json";
pub const CACHE_TTL: Duration = Duration::from_secs(24 * 60 * 60);
pub const OPENAI_COMPLETIONS_API: &str = "openai-completions";

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

    pub fn resolve(&self, spec: &str) -> Result<&CatalogModel, CatalogError> {
        if let Some((provider, id)) = spec.split_once('/') {
            self.get(provider, id)
                .ok_or_else(|| CatalogError::ModelNotFound(spec.to_string()))
        } else {
            let matches: Vec<_> = self.models.iter().filter(|m| m.id == spec).collect();
            match matches.as_slice() {
                [one] => Ok(*one),
                _ => self
                    .get(DEFAULT_PROVIDER, spec)
                    .ok_or_else(|| CatalogError::ModelNotFound(spec.to_string())),
            }
        }
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

fn maps_to_openai_completions(api: &str, npm: Option<&str>) -> bool {
    let api_l = api.to_ascii_lowercase();
    if api_l == OPENAI_COMPLETIONS_API
        || api_l == "openai-compatible"
        || api_l == "openai-completions"
        || api_l.contains("openai")
    {
        return true;
    }
    matches!(
        npm,
        Some("@ai-sdk/openai-compatible" | "@ai-sdk/openai" | "@ai-sdk/azure")
    )
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
        let registry_api = provider.api.clone().unwrap_or_default();

        let (base_url, api) = if let Some(ov) = r#override {
            let base = ov
                .base_url
                .clone()
                .or_else(|| {
                    if registry_api.starts_with("http") {
                        Some(registry_api.clone())
                    } else {
                        None
                    }
                })
                .unwrap_or_default();
            let api = ov
                .api
                .clone()
                .unwrap_or_else(|| OPENAI_COMPLETIONS_API.to_string());
            (base, api)
        } else {
            let api = if maps_to_openai_completions(&registry_api, npm) {
                OPENAI_COMPLETIONS_API.to_string()
            } else {
                continue;
            };
            let base = if registry_api.starts_with("http") {
                registry_api
            } else {
                continue;
            };
            (base, api)
        };

        if !maps_to_openai_completions(&api, npm) && api != OPENAI_COMPLETIONS_API {
            continue;
        }
        if base_url.is_empty() {
            continue;
        }

        let api_key_env = if let Some(ov) = r#override.and_then(|o| o.api_key.as_deref()) {
            api_key_env_from_ref(ov)
        } else {
            provider
                .env
                .as_ref()
                .and_then(|e| e.first())
                .cloned()
                .unwrap_or_default()
        };
        if api_key_env.is_empty() {
            continue;
        }

        for (model_key, model) in provider.models.unwrap_or_default() {
            let id = model.id.unwrap_or(model_key);
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
                api: api.clone(),
                provider: provider_id.clone(),
                base_url: base_url.clone(),
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
    out.sort_by(|a, b| a.qualified_id().cmp(&b.qualified_id()));
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
    if let Some(cache) = read_cache(&paths.cache_json()) {
        if cache.is_fresh(now, ttl) {
            return Ok(RefreshOutcome {
                catalog: cache,
                used_cache: true,
            });
        }
    }
    refresh_catalog(paths, registry_url, now).await
}

#[cfg(test)]
mod tests {
    use super::*;
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
                "api": "https://api.anthropic.com",
                "models": {
                    "claude": { "id": "claude", "name": "Claude", "reasoning": true, "tool_call": true }
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
        };
        let models = build_catalog(&fixture_registry(), &config).unwrap();
        assert_eq!(models.len(), 1);
        let m = &models[0];
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
    fn unmapped_providers_are_skipped() {
        let models = build_catalog(&fixture_registry(), &UserConfig::default()).unwrap();
        assert!(models.iter().all(|m| m.provider != "anthropic"));
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].api_key_env, "DEEPSEEK_API_KEY");
        assert_eq!(models[0].base_url, "https://api.deepseek.com");
    }

    #[test]
    fn default_model_ref_resolves() {
        let models = build_catalog(&fixture_registry(), &UserConfig::default()).unwrap();
        let cat = CachedCatalog {
            fetched_at_unix: 1,
            warning: None,
            models,
        };
        let m = cat.resolve(DEFAULT_MODEL_REF).unwrap();
        assert_eq!(m.id, DEFAULT_MODEL_ID);
        assert_eq!(parse_thinking("high").unwrap(), DEFAULT_THINKING);
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
        assert_eq!(first.catalog.models[0].id, "deepseek-v4-flash");

        let line = format_model_line(&first.catalog.models[0]);
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
        assert!(second
            .catalog
            .warning
            .unwrap()
            .contains("keeping cached catalog"));
        assert_eq!(second.catalog.models.len(), 1);
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
        assert_eq!(loaded.catalog.models.len(), 1);
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
        assert_eq!(loaded.catalog.models[0].id, "deepseek-v4-flash");
    }
}
