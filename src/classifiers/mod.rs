//! Concrete classifiers. Each module implements `Classifier` for one rule.

pub mod comment_only;
pub mod control_flow;
pub mod error_handling;
pub mod generated;
pub mod import_reorder;
pub mod lockfile;
pub mod public_api;
pub mod size_threshold;

pub use comment_only::CommentOnly;
pub use control_flow::ControlFlow;
pub use error_handling::ErrorHandlingDeleted;
pub use generated::Generated;
pub use import_reorder::ImportReorder;
pub use lockfile::Lockfile;
pub use public_api::PublicApi;
pub use size_threshold::SizeThreshold;
