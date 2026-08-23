//! Wire-format adapters. Each format module implements the kernel
//! `StreamFn` trait for one HTTP shape and declares its `API` key (the
//! value of `Model::api`); dispatch lives in [`crate::stream_fn_for`].

pub mod openai_completions;
pub mod anthropic_messages;
