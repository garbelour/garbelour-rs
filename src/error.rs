//! Crate-wide error helpers.
//!
//! Garbelour uses `anyhow::Error` end-to-end. This module exists as a single
//! place to grow domain-specific error types if any module ever needs to
//! discriminate between failure modes (e.g. a missing `GITHUB_TOKEN` vs an
//! HTTP 4xx vs an event-payload parse failure). For v1, a `Result` alias is
//! enough.

pub type Result<T> = anyhow::Result<T>;
