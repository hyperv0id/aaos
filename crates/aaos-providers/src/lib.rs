//! Provider domain in a single crate: the model registry (models.dev
//! fetch/cache, user `models.json` overrides, credential resolution) and
//! wire-format adapters dispatched by [`Model::api`].
//!
//! Callers resolve a [`StreamFn`] through [`stream_fn_for`] instead of
//! hardcoding a format adapter; product defaults (which model, which
//! provider, which thinking level) live in the composing application.

pub mod formats;
pub mod registry;

use std::sync::Arc;

use pi_agent_core::types::{Model, StreamFn};
use thiserror::Error;

pub use formats::openai_completions::OpenAiCompletionsProvider;
pub use registry::{
    CACHE_TTL, CachedCatalog, CatalogError, CatalogModel, DEFAULT_REGISTRY_URL, Paths,
    RefreshOutcome, format_model_line, load_catalog, parse_thinking, refresh_catalog,
};

/// No adapter is registered for the model's `api` format.
#[derive(Debug, Error)]
pub enum ProviderError {
    #[error("no provider adapter for model api {0:?}")]
    UnknownApi(String),
}

/// Resolve the wire format adapter for `model` from its `api` field.
///
/// Registering a format means adding a module under `formats` that
/// declares its `API` key plus one arm here; callers never name adapters.
///
/// # Examples
///
/// ```
/// use aaos_providers::stream_fn_for;
/// use pi_agent_core::types::Model;
///
/// let mut model = Model::unknown();
/// model.api = "openai-completions".into();
/// assert!(stream_fn_for(&model).is_ok());
///
/// model.api = "bogus-format".into();
/// assert!(stream_fn_for(&model).is_err());
/// ```
pub fn stream_fn_for(model: &Model) -> Result<Arc<dyn StreamFn>, ProviderError> {
    match model.api.as_str() {
        formats::openai_completions::API => Ok(Arc::new(OpenAiCompletionsProvider::new())),
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
        let model = model_with_api(formats::openai_completions::API);
        assert!(stream_fn_for(&model).is_ok());
    }

    #[test]
    fn stream_fn_for_unknown_api_is_error() {
        let err = match stream_fn_for(&model_with_api("bogus-format")) {
            Ok(_) => panic!("unknown api must not resolve an adapter"),
            Err(e) => e.to_string(),
        };
        assert!(err.contains("bogus-format"), "{err}");
    }

    #[test]
    fn dispatch_key_is_api_not_provider() {
        // Spec §4: `model.provider` never enters the dispatch key. A provider
        // whose own format differs (anthropic → anthropic-messages) must still
        // resolve by `api` alone.
        let mut model = model_with_api(formats::openai_completions::API);
        model.provider = "anthropic".into();
        assert!(stream_fn_for(&model).is_ok());

        let mut model = model_with_api("anthropic-messages");
        model.provider = "openai".into();
        assert!(stream_fn_for(&model).is_err());
    }
}
