//! Output rendering: human (terminal), markdown (sticky comment), json.

use std::collections::BTreeMap;
use std::io::IsTerminal;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::classify::{Category, Classified, FocusLines, Level, Side};
use crate::diff::Diff;

/// Counts of classified hunks at each level.
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub total: usize,
    pub review: usize,
    pub skim: usize,
    pub skip: usize,
}

impl Summary {
    pub fn from_classified(classified: &[Classified]) -> Self {
        let mut s = Summary::default();
        for c in classified {
            s.total += 1;
            match c.classification.level {
                Level::Review => s.review += 1,
                Level::Skim => s.skim += 1,
                Level::Skip => s.skip += 1,
            }
        }
        s
    }
}

/// Single-line summary printed to stderr in human-mode runs.
pub fn summary_line(classified: &[Classified]) -> String {
    let s = Summary::from_classified(classified);
    format!(
        "garbelour: {} of {} hunks need review, {} worth skimming, {} mechanical",
        s.review, s.total, s.skim, s.skip
    )
}

// --- human ---------------------------------------------------------------

/// Terminal-friendly report. Sections: Review (always), Skim (if any), Skip
/// (grouped by category). When `use_color` is true, ANSI escape codes color
/// the section headers and file:line columns.
pub fn human(_diff: &Diff, classified: &[Classified], use_color: bool) -> String {
    let mut review = Vec::new();
    let mut skim = Vec::new();
    let mut skip: BTreeMap<Category, Vec<String>> = BTreeMap::new();
    for c in classified {
        match c.classification.level {
            Level::Review => review.push(c),
            Level::Skim => skim.push(c),
            Level::Skip => skip
                .entry(c.classification.category)
                .or_default()
                .push(c.file_path.display().to_string()),
        }
    }
    let mut out = String::new();

    if !review.is_empty() {
        push_section(&mut out, "Review", &review, use_color, Color::Red);
    }
    if !skim.is_empty() {
        push_section(&mut out, "Skim", &skim, use_color, Color::Yellow);
    }
    if !skip.is_empty() {
        let count: usize = skip.values().map(Vec::len).sum();
        push_color(
            &mut out,
            &format!("  Skip ({})\n", count),
            use_color,
            Color::DimBold,
        );
        for (category, files) in &skip {
            let label = category_label(*category);
            let mut deduped: Vec<&String> = files.iter().collect();
            deduped.sort();
            deduped.dedup();
            let preview = preview_paths(&deduped);
            let line = format!("    {} ({})    {}\n", label, files.len(), preview);
            push_color(&mut out, &line, use_color, Color::Dim);
        }
    }
    out
}

fn push_section(
    out: &mut String,
    title: &str,
    items: &[&Classified],
    use_color: bool,
    section_color: Color,
) {
    let header_color = match section_color {
        Color::Red => Color::BoldRed,
        Color::Yellow => Color::BoldYellow,
        c => c,
    };
    push_color(
        out,
        &format!("  {} ({})\n", title, items.len()),
        use_color,
        header_color,
    );
    let max_locator = items.iter().map(|c| locator(c).len()).max().unwrap_or(0);
    for c in items {
        let loc = locator(c);
        let pad = " ".repeat(max_locator.saturating_sub(loc.len()) + 4);
        let mut left = format!("    {loc}");
        if use_color {
            left = colorize(&left, section_color);
        }
        out.push_str(&left);
        out.push_str(&pad);
        out.push_str(&c.classification.rationale);
        out.push('\n');
    }
}

fn locator(c: &Classified) -> String {
    let path = c.file_path.display();
    match &c.classification.focus_lines {
        Some(FocusLines { start, end, side }) => {
            let suffix = match side {
                Side::Old => " (old)",
                Side::New => "",
            };
            if start == end {
                format!("{}:{}{}", path, start, suffix)
            } else {
                format!("{}:{}–{}{}", path, start, end, suffix)
            }
        }
        None => format!("{}:{}", path, c.new_range.start),
    }
}

fn preview_paths(paths: &[&String]) -> String {
    const MAX: usize = 3;
    if paths.len() <= MAX {
        paths
            .iter()
            .map(|s| s.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    } else {
        format!(
            "{}, ...",
            paths
                .iter()
                .take(MAX)
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )
    }
}

fn category_label(c: Category) -> &'static str {
    match c {
        Category::Generated => "generated",
        Category::Lockfile => "lockfile",
        Category::FormatterOnly => "formatter-only",
        Category::CommentOnly => "comment-only",
        Category::ImportReorder => "import-reorder",
        Category::PureRename => "rename",
        Category::TestFixture => "test-fixture",
        Category::PublicApiChange => "public-api",
        Category::ControlFlow => "control-flow",
        Category::ErrorHandlingDeleted => "error-handling-deleted",
        Category::LargeChange => "large-change",
        Category::LlmAssessed => "llm",
    }
}

#[derive(Copy, Clone, Debug)]
enum Color {
    Red,
    BoldRed,
    Yellow,
    BoldYellow,
    Dim,
    DimBold,
}

fn colorize(s: &str, color: Color) -> String {
    let code = match color {
        Color::Red => "\x1b[31m",
        Color::BoldRed => "\x1b[1;31m",
        Color::Yellow => "\x1b[33m",
        Color::BoldYellow => "\x1b[1;33m",
        Color::Dim => "\x1b[2m",
        Color::DimBold => "\x1b[1;2m",
    };
    format!("{}{}\x1b[0m", code, s)
}

fn push_color(out: &mut String, s: &str, use_color: bool, color: Color) {
    if use_color {
        out.push_str(&colorize(s, color));
    } else {
        out.push_str(s);
    }
}

// --- markdown ------------------------------------------------------------

pub const STICKY_MARKER: &str = "<!-- garbelour:sticky -->";

/// Build the GitHub sticky-comment body. `repo_ref` carries the owner/repo/pr
/// triple needed for deep links; if `None`, links are omitted (so the
/// rendered markdown is still readable when invoked outside GitHub).
pub fn markdown(_diff: &Diff, classified: &[Classified], repo_ref: Option<&RepoRef>) -> String {
    let s = Summary::from_classified(classified);

    let mut review = Vec::new();
    let mut skim = Vec::new();
    let mut skip: BTreeMap<Category, Vec<String>> = BTreeMap::new();
    for c in classified {
        match c.classification.level {
            Level::Review => review.push(c),
            Level::Skim => skim.push(c),
            Level::Skip => skip
                .entry(c.classification.category)
                .or_default()
                .push(c.file_path.display().to_string()),
        }
    }

    let mut out = String::new();
    out.push_str(STICKY_MARKER);
    out.push_str("\n## Garbelour\n\n");
    out.push_str(&format!(
        "**{} of {} hunks need review.** {} worth skimming. {} mechanical.\n",
        s.review, s.total, s.skim, s.skip
    ));

    if !review.is_empty() {
        out.push_str(&format!("\n### Review ({})\n\n", review.len()));
        for c in &review {
            out.push_str(&markdown_item(c, repo_ref));
        }
    }
    if !skim.is_empty() {
        out.push_str(&format!("\n### Skim ({})\n\n", skim.len()));
        out.push_str("<details>\n<summary>Click to expand</summary>\n\n");
        for c in &skim {
            out.push_str(&markdown_item(c, repo_ref));
        }
        out.push_str("\n</details>\n");
    }
    if !skip.is_empty() {
        let total: usize = skip.values().map(Vec::len).sum();
        out.push_str(&format!("\n### Skip ({})\n\n", total));
        out.push_str("<details>\n<summary>Click to expand</summary>\n\n");
        for (category, files) in &skip {
            let mut deduped: Vec<&String> = files.iter().collect();
            deduped.sort();
            deduped.dedup();
            let preview = preview_paths(&deduped);
            out.push_str(&format!(
                "- **{}** ({}): {}\n",
                category_label(*category),
                files.len(),
                preview
            ));
        }
        out.push_str("\n</details>\n");
    }

    out
}

fn markdown_item(c: &Classified, repo_ref: Option<&RepoRef>) -> String {
    let (display_line, target_line, side) = match &c.classification.focus_lines {
        Some(f) => {
            let display = if f.start == f.end {
                format!("{}", f.start)
            } else {
                format!("{}–{}", f.start, f.end)
            };
            (display, f.start, f.side)
        }
        None => (c.new_range.start.to_string(), c.new_range.start, Side::New),
    };
    let label = format!("`{}:{}`", c.file_path.display(), display_line);
    let link_text = match repo_ref {
        Some(r) => format!(
            "[{}]({})",
            label,
            deep_link(r, &c.file_path.to_string_lossy(), target_line, side)
        ),
        None => label,
    };
    format!("- {} — {}\n", link_text, c.classification.rationale)
}

#[derive(Clone, Debug)]
pub struct RepoRef {
    pub host: String, // e.g. "https://github.com"
    pub owner: String,
    pub repo: String,
    pub pr: u64,
}

fn deep_link(r: &RepoRef, path: &str, line: u32, side: Side) -> String {
    format!(
        "{}/{}/{}/pull/{}/files#{}",
        r.host,
        r.owner,
        r.repo,
        r.pr,
        diff_anchor(path, line, side)
    )
}

pub fn diff_anchor(path: &str, line: u32, side: Side) -> String {
    let hash = hex::encode(Sha256::digest(path.as_bytes()));
    let prefix = match side {
        Side::New => 'R',
        Side::Old => 'L',
    };
    format!("diff-{}{}{}", hash, prefix, line)
}

// --- json ----------------------------------------------------------------

#[derive(Serialize)]
struct JsonReport<'a> {
    base_sha: &'a str,
    head_sha: &'a str,
    hunks: Vec<JsonHunk<'a>>,
    summary: JsonSummary,
}

#[derive(Serialize)]
struct JsonHunk<'a> {
    hunk_id: &'a str,
    file: String,
    line: u32,
    level: Level,
    category: Category,
    rationale: &'a str,
    focus_lines: &'a Option<FocusLines>,
    source: &'a crate::classify::Source,
}

#[derive(Serialize)]
struct JsonSummary {
    total: usize,
    review: usize,
    skim: usize,
    skip: usize,
}

pub fn json(diff: &Diff, classified: &[Classified]) -> anyhow::Result<String> {
    let summary = Summary::from_classified(classified);
    let hunks: Vec<JsonHunk> = classified
        .iter()
        .map(|c| JsonHunk {
            hunk_id: &c.hunk_id.0,
            file: c.file_path.display().to_string(),
            line: c.new_range.start,
            level: c.classification.level,
            category: c.classification.category,
            rationale: &c.classification.rationale,
            focus_lines: &c.classification.focus_lines,
            source: &c.classification.source,
        })
        .collect();
    let report = JsonReport {
        base_sha: &diff.base_sha,
        head_sha: &diff.head_sha,
        hunks,
        summary: JsonSummary {
            total: summary.total,
            review: summary.review,
            skim: summary.skim,
            skip: summary.skip,
        },
    };
    Ok(serde_json::to_string_pretty(&report)?)
}

// --- format auto-detect helpers -----------------------------------------

/// Whether stdout is connected to a TTY. Wraps `io::stdout().is_terminal()`
/// so callers don't need to import the trait.
pub fn stdout_is_tty() -> bool {
    std::io::stdout().is_terminal()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::classify::{Classification, Source};
    use crate::diff::{Diff, HunkId, LineRange};

    fn diff() -> Diff {
        Diff {
            base_sha: "a".repeat(40),
            head_sha: "b".repeat(40),
            files: Vec::new(),
        }
    }

    fn classified(level: Level, category: Category, file: &str, line: u32) -> Classified {
        Classified {
            hunk_id: HunkId(format!("{file}:{line}")),
            file_path: PathBuf::from(file),
            new_range: LineRange {
                start: line,
                count: 1,
            },
            classification: Classification {
                level,
                category,
                rationale: "test rationale".into(),
                source: Source::Heuristic {
                    name: "test".into(),
                },
                focus_lines: None,
            },
        }
    }

    #[test]
    fn anchor_uses_sha256_hex_and_side_prefix() {
        let anchor = diff_anchor("src/index.js", 42, Side::New);
        // The known SHA-256 of "src/index.js":
        let expected_hash = hex::encode(Sha256::digest("src/index.js".as_bytes()));
        assert_eq!(anchor, format!("diff-{}R42", expected_hash));
        assert!(diff_anchor("src/index.js", 42, Side::Old).contains('L'));
    }

    #[test]
    fn summary_counts_levels() {
        let cs = vec![
            classified(Level::Review, Category::PublicApiChange, "a", 1),
            classified(Level::Review, Category::ControlFlow, "b", 1),
            classified(Level::Skim, Category::LlmAssessed, "c", 1),
            classified(Level::Skip, Category::Lockfile, "Cargo.lock", 1),
        ];
        let s = Summary::from_classified(&cs);
        assert_eq!(s.total, 4);
        assert_eq!(s.review, 2);
        assert_eq!(s.skim, 1);
        assert_eq!(s.skip, 1);
    }

    #[test]
    fn human_renders_review_section() {
        let cs = vec![classified(
            Level::Review,
            Category::PublicApiChange,
            "src/x.rs",
            42,
        )];
        let out = human(&diff(), &cs, false);
        assert!(out.contains("Review (1)"));
        assert!(out.contains("src/x.rs:42"));
        assert!(out.contains("test rationale"));
        // No ANSI when use_color=false.
        assert!(!out.contains("\x1b["));
    }

    #[test]
    fn human_renders_with_color() {
        let cs = vec![classified(Level::Review, Category::ControlFlow, "x", 1)];
        let out = human(&diff(), &cs, true);
        assert!(out.contains("\x1b[1;31m"), "should contain bold red ANSI");
    }

    #[test]
    fn human_groups_skip_by_category() {
        let cs = vec![
            classified(Level::Skip, Category::Generated, "dist/a.js", 1),
            classified(Level::Skip, Category::Generated, "dist/b.js", 1),
            classified(Level::Skip, Category::Lockfile, "Cargo.lock", 1),
        ];
        let out = human(&diff(), &cs, false);
        assert!(out.contains("generated (2)"));
        assert!(out.contains("lockfile (1)"));
    }

    #[test]
    fn markdown_starts_with_sticky_marker() {
        let cs = vec![classified(Level::Review, Category::ControlFlow, "x", 1)];
        let out = markdown(&diff(), &cs, None);
        assert!(out.starts_with(STICKY_MARKER));
        assert!(out.contains("## Garbelour"));
    }

    #[test]
    fn markdown_review_items_have_no_link_without_repo_ref() {
        let cs = vec![classified(
            Level::Review,
            Category::ControlFlow,
            "src/x.rs",
            42,
        )];
        let out = markdown(&diff(), &cs, None);
        assert!(out.contains("`src/x.rs:42`"));
        assert!(!out.contains("https://"));
    }

    #[test]
    fn markdown_review_items_have_link_with_repo_ref() {
        let r = RepoRef {
            host: "https://github.com".into(),
            owner: "acme".into(),
            repo: "widget".into(),
            pr: 42,
        };
        let cs = vec![classified(
            Level::Review,
            Category::ControlFlow,
            "src/x.rs",
            7,
        )];
        let out = markdown(&diff(), &cs, Some(&r));
        assert!(out.contains("https://github.com/acme/widget/pull/42/files#diff-"));
        assert!(out.contains("R7"));
    }

    #[test]
    fn markdown_skim_section_is_collapsed() {
        let cs = vec![classified(Level::Skim, Category::LlmAssessed, "x", 1)];
        let out = markdown(&diff(), &cs, None);
        assert!(out.contains("<details>"));
        assert!(out.contains("Skim (1)"));
    }

    #[test]
    fn json_renders_valid_schema() {
        let cs = vec![classified(
            Level::Review,
            Category::PublicApiChange,
            "src/x.rs",
            142,
        )];
        let out = json(&diff(), &cs).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["base_sha"].as_str().unwrap().len(), 40);
        assert_eq!(v["summary"]["total"], 1);
        assert_eq!(v["summary"]["review"], 1);
        assert_eq!(v["hunks"][0]["file"], "src/x.rs");
        assert_eq!(v["hunks"][0]["line"], 142);
        assert_eq!(v["hunks"][0]["level"], "review");
        assert_eq!(v["hunks"][0]["category"], "public_api_change");
    }
}
