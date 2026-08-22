//! Error type for the session store.

#[derive(Debug, thiserror::Error)]
pub enum StoreError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid hash: {0}")]
    InvalidHash(String),
    #[error("not found: {0}")]
    NotFound(String),
    #[error("encode: {0}")]
    Encode(String),
    #[error("decode: {0}")]
    Decode(String),
    #[error("invalid derivation {context}: {reason}")]
    InvalidDerivation { context: String, reason: String },
    #[error("cyclic derivation chain at {0}")]
    CyclicChain(String),
    #[error("db: {0}")]
    Db(#[from] tokio_rusqlite::Error),
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

pub type Result<T> = std::result::Result<T, StoreError>;
