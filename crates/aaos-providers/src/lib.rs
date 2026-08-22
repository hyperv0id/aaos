//! Provider domain in a single crate: the model registry (models.dev
//! fetch/cache, user `models.json` overrides, credential resolution) and
//! wire-dialect adapters dispatched by [`Model::api`].
//!
//! Callers resolve a [`StreamFn`] through [`stream_fn_for`] instead of
//! hardcoding a dialect adapter; product defaults (which model, which
//! provider, which thinking level) live in the composing application.

pub mod dialects;
pub mod registry;

use std::sync::Arc;

use pi_agent_core::types::{Model, StreamFn};
use thiserror::Error;

pub use dialects::openai_completions::OpenAiCompletionsProvider;
pub use registry::{
    CACHE_TTL, CachedCatalog, CatalogError, CatalogModel, DEFAULT_REGISTRY_URL, Paths,
    RefreshOutcome, format_model_line, load_catalog, parse_thinking, refresh_catalog,
};

/// No adapter is registered for the model's `api` dialect.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("no provider adapter for model api {0:?}")]
    UnknownApi(String),
}

/// Resolve the wire dialect for `model` from its `api` field.
///
/// Registering a dialect means adding a module under `dialects` that
/// declares its `API` key plus one arm here; callers never name adapters.
pub fn stream_fn_for(model: &Model) -> Result<Arc<dyn StreamFn>, ProviderError> {
    match model.api.as_str() {
        dialects::openai_completions::API => Ok(Arc::new(OpenAiCompletionsProvider::new())),
        other => Err(ProviderError::UnknownApi(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with_api(api: &str) -> Model {
        Model {
            api: api.into(),
            ..Model::unknown()
        }
    }

    #[test]
    fn stream_fn_for_known_api_resolves_adapter() {
        let model = model_with_api(dialects::openai_completions::API);
        assert!(stream_fn_for(&model).is_ok());
    }

    #[test]
    fn stream_fn_for_unknown_api_is_error() {
        let err = match stream_fn_for(&model_with_api("anthropic-messages")) {
            Ok(_) => panic!("unknown api must not resolve an adapter"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("anthropic-messages"), "{err}");
    }
}
