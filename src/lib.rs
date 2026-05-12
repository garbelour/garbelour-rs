//! Garbelour: classify PR diff hunks by reviewer attention.
//!
//! Every hunk in a pull request gets one of three levels: `review` (read
//! carefully), `skim` (glance), or `skip` (collapse). Heuristics handle the
//! mechanical cases (lockfiles, generated code, comment-only edits, import
//! reordering, large changes); an optional LLM pass triages the rest.

pub mod ast;
pub mod classifiers;
pub mod classify;
pub mod cli;
pub mod config;
pub mod consolidate;
pub mod diff;
pub mod error;
pub mod github;
pub mod lang;
pub mod llm;
pub mod render;
pub mod run;

pub use classify::{
    Category, Classification, Classified, Classifier, FocusLines, Level, Pipeline, PipelineConfig,
    Side, Source, Unclassified,
};
pub use diff::{Diff, FileDiff, FileStatus, Hunk, HunkId, LineRange};
pub use lang::Language;
pub use run::run_cli;
