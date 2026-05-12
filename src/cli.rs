//! Command-line argument parsing.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "garbelour",
    version,
    about = "Classify PR diffs by reviewer attention: review, skim, or skip"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Classify hunks in a PR diff and render the report.
    Review(ReviewArgs),
}

#[derive(clap::Args, Debug)]
pub struct ReviewArgs {
    /// Base git ref. Required outside a GitHub Actions PR event.
    #[arg(long)]
    pub base: Option<String>,

    /// Head git ref.
    #[arg(long, default_value = "HEAD")]
    pub head: String,

    /// Path to git repository.
    #[arg(long, default_value = ".")]
    pub repo: PathBuf,

    /// Output format. `auto` picks markdown for --post-comment, human for a
    /// TTY, json otherwise.
    #[arg(long, value_enum, default_value_t = FormatChoice::Auto)]
    pub format: FormatChoice,

    /// Color output.
    #[arg(long, value_enum, default_value_t = ColorChoice::Auto)]
    pub color: ColorChoice,

    /// Post or update the sticky comment on the GitHub PR.
    #[arg(long)]
    pub post_comment: bool,

    /// GitHub repo owner. Defaults to the value from the event payload.
    #[arg(long)]
    pub owner: Option<String>,

    /// GitHub repo name. Defaults to the value from the event payload.
    #[arg(long = "repo-name")]
    pub repo_name: Option<String>,

    /// PR number. Defaults to the value from the event payload.
    #[arg(long)]
    pub pr: Option<u64>,

    /// Send unclassified hunks to an LLM for triage.
    #[arg(long)]
    pub llm: bool,

    /// LLM provider. Auto-detected from the API key environment variable.
    #[arg(long)]
    pub llm_provider: Option<String>,

    /// LLM model. Defaults to the provider's default.
    #[arg(long)]
    pub llm_model: Option<String>,

    /// Override the LLM API base URL.
    #[arg(long)]
    pub llm_base_url: Option<String>,

    /// Path to garbelour.toml.
    #[arg(long, default_value = "garbelour.toml")]
    pub config: PathBuf,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum FormatChoice {
    Auto,
    Human,
    Markdown,
    Json,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
pub enum ColorChoice {
    Auto,
    Always,
    Never,
}

/// The resolved output format after applying `auto` rules.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Format {
    Human,
    Markdown,
    Json,
}
