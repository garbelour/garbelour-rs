use std::process::ExitCode;

use clap::Parser;
use garbelour::classify::{
    Category, Classification, Classified, Level, Pipeline, PipelineConfig, Source,
};
use garbelour::cli::{Cli, ColorChoice, Command, Format, FormatChoice, ReviewArgs};
use garbelour::config::Config;
use garbelour::{classifiers, diff, github, llm, render};

fn build_pipeline_config(args: &ReviewArgs, config: &Config) -> PipelineConfig {
    let mut generated_globs = config.classify.generated_globs.clone();
    let mut lockfile_names = config.classify.lockfile_names.clone();
    // Allow the config to extend; CLI controls only size_threshold for now.
    generated_globs.sort();
    generated_globs.dedup();
    lockfile_names.sort();
    lockfile_names.dedup();
    let generated_paths = classifiers::generated::read_gitattributes_generated(&args.repo);
    PipelineConfig {
        size_threshold: config
            .classify
            .size_threshold
            .unwrap_or(args.size_threshold),
        generated_globs,
        generated_paths,
        lockfile_names,
    }
}

fn resolve_format(args: &ReviewArgs) -> Format {
    match args.format {
        FormatChoice::Human => Format::Human,
        FormatChoice::Markdown => Format::Markdown,
        FormatChoice::Json => Format::Json,
        FormatChoice::Auto => {
            if args.post_comment {
                Format::Markdown
            } else if render::stdout_is_tty() {
                Format::Human
            } else {
                Format::Json
            }
        }
    }
}

fn resolve_color(args: &ReviewArgs) -> bool {
    match args.color {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => render::stdout_is_tty() && std::env::var_os("NO_COLOR").is_none(),
    }
}

fn build_repo_ref(args: &ReviewArgs, event: Option<&github::PrEvent>) -> Option<render::RepoRef> {
    let owner = args
        .owner
        .clone()
        .or_else(|| event.map(|e| e.owner.clone()))?;
    let repo = args
        .repo_name
        .clone()
        .or_else(|| event.map(|e| e.repo.clone()))?;
    let pr = args.pr.or_else(|| event.map(|e| e.pr_number))?;
    Some(render::RepoRef {
        host: "https://github.com".to_string(),
        owner,
        repo,
        pr,
    })
}

fn review(args: ReviewArgs) -> anyhow::Result<u8> {
    let config = Config::load(&args.config)?;

    // Resolve base/head: prefer CLI, then GitHub event payload (when
    // available), then error.
    let event = if args.base.is_none()
        || args.owner.is_none()
        || args.repo_name.is_none()
        || args.pr.is_none()
    {
        github::parse_event().ok()
    } else {
        None
    };

    let base = args
        .base
        .clone()
        .or_else(|| event.as_ref().map(|e| e.base_sha.clone()))
        .ok_or_else(|| {
            anyhow::anyhow!("--base is required (and no GITHUB_EVENT_PATH was found)")
        })?;
    let head = if args.head == "HEAD" {
        event
            .as_ref()
            .map(|e| e.head_sha.clone())
            .unwrap_or_else(|| args.head.clone())
    } else {
        args.head.clone()
    };

    // 1. Extract diff into Diff struct
    let mut diff = diff::extract(&args.repo, &base, &head)?;

    // 2. Build PipelineConfig and run heuristic pipeline.
    let pipeline_config = build_pipeline_config(&args, &config);
    let pipeline = Pipeline::standard(&pipeline_config)?;
    let (mut classified, unclassified) = pipeline.run(&mut diff);

    // println!("{:?}", classified);

    // 3. Optional LLM pass.
    if args.llm && !unclassified.is_empty() {
        let llm_config = llm::detect_provider(
            args.llm_provider
                .as_deref()
                .or(config.llm.provider.as_deref()),
            args.llm_model.as_deref().or(config.llm.model.as_deref()),
            args.llm_base_url.as_deref(),
        )?;
        let llm_results = llm::classify_hunks(&unclassified, &llm_config)?;
        classified.extend(llm_results);
    }

    // 4. Default any still-unclassified hunks to Review (heuristics didn't
    // claim them and the LLM either ran but missed or wasn't invoked).
    for u in &unclassified {
        if !classified.iter().any(|c| c.hunk_id == u.hunk_id) {
            classified.push(Classified {
                hunk_id: u.hunk_id.clone(),
                file_path: u.file_path.clone(),
                new_range: u.new_range.clone(),
                classification: Classification {
                    level: Level::Review,
                    category: Category::LlmAssessed,
                    rationale: "no heuristic match and LLM not run — defaulting to review".into(),
                    source: Source::Heuristic {
                        name: "default".into(),
                    },
                    focus_lines: None,
                },
            });
        }
    }

    // 5. Resolve format and color.
    let format = resolve_format(&args);
    let use_color = resolve_color(&args);

    // 6. Render + output.
    match format {
        Format::Human => {
            eprintln!("{}", render::summary_line(&classified));
            print!("{}", render::human(&diff, &classified, use_color));
        }
        Format::Markdown => {
            let repo_ref = build_repo_ref(&args, event.as_ref());
            let body = render::markdown(&diff, &classified, repo_ref.as_ref());
            if args.post_comment {
                let client = github::GitHubClient::from_env()?;
                let pr = repo_ref.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("--post-comment requires owner/repo/pr (CLI or event payload)")
                })?;
                github::upsert_sticky_comment(&client, &pr.owner, &pr.repo, pr.pr, &body)?;
                eprintln!("garbelour: posted review map to PR #{}", pr.pr);
            } else {
                print!("{body}");
            }
        }
        Format::Json => {
            let json = render::json(&diff, &classified)?;
            println!("{json}");
        }
    }

    Ok(0)
}

fn run(cli: Cli) -> anyhow::Result<u8> {
    match cli.command {
        Command::Review(args) => review(args),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => ExitCode::from(code),
        Err(e) => {
            eprintln!("garbelour: {:#}", e);
            ExitCode::from(2_u8)
        }
    }
}
