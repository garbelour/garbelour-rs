//! Output rendering: human (terminal), markdown (sticky comment), json.
//!
//! All three renderers operate on `&[ConsolidatedItem]`, where each item
//! carries one or more `Location`s. Multi-location items are rendered with
//! a comma-separated locator list and a single shared rationale.

use std::collections::BTreeMap;
use std::io::IsTerminal;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::classify::{Category, FocusLines, Level, Side, Source};
use crate::consolidate::{ConsolidatedItem, Location};
use crate::diff::Diff;

/// Counts at each level. `total` is the number of consolidated items (after
/// dedup/merge), not the underlying hunk count.
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub total: usize,
    pub review: usize,
    pub skim: usize,
    pub skip: usize,
}

impl Summary {
    pub fn from_items(items: &[ConsolidatedItem]) -> Self {
        let mut s = Summary::default();
        for c in items {
            s.total += 1;
            match c.level {
                Level::Review => s.review += 1,
                Level::Skim => s.skim += 1,
                Level::Skip => s.skip += 1,
            }
        }
        s
    }
}

pub fn summary_line(items: &[ConsolidatedItem]) -> String {
    let s = Summary::from_items(items);
    format!(
        "garbelour: {} of {} hunks need review, {} worth skimming, {} mechanical",
        s.review, s.total, s.skim, s.skip
    )
}

// --- human ---------------------------------------------------------------

/// Terminal-friendly report. Sections: Review (always), Skim (if any), Skip
/// (grouped by category). When `use_color` is true, ANSI escape codes color
/// the section headers and file:line columns.
pub fn human(_diff: &Diff, items: &[ConsolidatedItem], use_color: bool) -> String {
    let mut review = Vec::new();
    let mut skim = Vec::new();
    let mut skip: BTreeMap<Category, Vec<String>> = BTreeMap::new();
    for c in items {
        match c.level {
            Level::Review => review.push(c),
            Level::Skim => skim.push(c),
            Level::Skip => {
                let entry = skip.entry(c.category).or_default();
                for loc in &c.locations {
                    entry.push(loc.file_path.display().to_string());
                }
            }
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
    items: &[&ConsolidatedItem],
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
    let locators: Vec<String> = items.iter().map(|c| human_locator(c)).collect();
    let max = locators.iter().map(|s| s.len()).max().unwrap_or(0);
    for (c, loc) in items.iter().zip(locators.iter()) {
        let pad = " ".repeat(max.saturating_sub(loc.len()) + 4);
        let mut left = format!("    {loc}");
        if use_color {
            left = colorize(&left, section_color);
        }
        out.push_str(&left);
        out.push_str(&pad);
        out.push_str(&c.rationale);
        out.push('\n');
    }
}

/// Build a single locator string for the human renderer. Multi-location
/// items render as `file:L1, file:L2, ...`, falling back to bare path
/// elision (`file:L1, :L2`) when all locations share the same file.
fn human_locator(c: &ConsolidatedItem) -> String {
    let mut parts: Vec<String> = Vec::new();
    let first_path = c.primary().file_path.display().to_string();
    for (i, loc) in c.locations.iter().enumerate() {
        let path_str = loc.file_path.display().to_string();
        let path_part = if i > 0 && path_str == first_path {
            String::new()
        } else {
            path_str
        };
        parts.push(format!("{}{}", path_part, line_suffix(loc)));
    }
    parts.join(", ")
}

fn line_suffix(loc: &Location) -> String {
    match &loc.focus_lines {
        Some(FocusLines { start, end, side }) => {
            let s = match side {
                Side::Old => " (old)",
                Side::New => "",
            };
            if start == end {
                format!(":{}{}", start, s)
            } else {
                format!(":{}–{}{}", start, end, s)
            }
        }
        None => format!(":{}", loc.new_range.start),
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
        Category::NumericalCalc => "numerical-calc",
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
pub fn markdown(_diff: &Diff, items: &[ConsolidatedItem], repo_ref: Option<&RepoRef>) -> String {
    let s = Summary::from_items(items);

    let mut review = Vec::new();
    let mut skim = Vec::new();
    let mut skip: BTreeMap<Category, Vec<String>> = BTreeMap::new();
    for c in items {
        match c.level {
            Level::Review => review.push(c),
            Level::Skim => skim.push(c),
            Level::Skip => {
                let entry = skip.entry(c.category).or_default();
                for loc in &c.locations {
                    entry.push(loc.file_path.display().to_string());
                }
            }
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

fn markdown_item(c: &ConsolidatedItem, repo_ref: Option<&RepoRef>) -> String {
    let primary_path = c.primary().file_path.display().to_string();
    let parts: Vec<String> = c
        .locations
        .iter()
        .enumerate()
        .map(|(i, loc)| markdown_locator(loc, repo_ref, i, &primary_path))
        .collect();
    format!("- {}: {}\n", parts.join(", "), c.rationale)
}

fn markdown_locator(
    loc: &Location,
    repo_ref: Option<&RepoRef>,
    idx: usize,
    primary_path: &str,
) -> String {
    let (display_line, target_line, side) = match &loc.focus_lines {
        Some(f) => {
            let display = if f.start == f.end {
                format!("{}", f.start)
            } else {
                format!("{}–{}", f.start, f.end)
            };
            (display, f.start, f.side)
        }
        None => (
            loc.new_range.start.to_string(),
            loc.new_range.start,
            Side::New,
        ),
    };
    let path_str = loc.file_path.display().to_string();
    let shown_path = if idx > 0 && path_str == primary_path {
        String::new()
    } else {
        path_str.clone()
    };
    let label = format!("`{}:{}`", shown_path, display_line);
    match repo_ref {
        Some(r) => format!(
            "[{}]({})",
            label,
            deep_link(r, &path_str, target_line, side)
        ),
        None => label,
    }
}

#[derive(Clone, Debug)]
pub struct RepoRef {
    pub host: String,
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
    items: Vec<JsonItem<'a>>,
    summary: JsonSummary,
}

#[derive(Serialize)]
struct JsonItem<'a> {
    level: Level,
    category: Category,
    rationale: &'a str,
    source: &'a Source,
    locations: Vec<JsonLocation<'a>>,
}

#[derive(Serialize)]
struct JsonLocation<'a> {
    hunk_id: &'a str,
    file: String,
    line: u32,
    focus_lines: &'a Option<FocusLines>,
}

#[derive(Serialize)]
struct JsonSummary {
    total: usize,
    review: usize,
    skim: usize,
    skip: usize,
}

pub fn json(diff: &Diff, items: &[ConsolidatedItem]) -> anyhow::Result<String> {
    let summary = Summary::from_items(items);
    let json_items: Vec<JsonItem> = items
        .iter()
        .map(|c| JsonItem {
            level: c.level,
            category: c.category,
            rationale: &c.rationale,
            source: &c.source,
            locations: c
                .locations
                .iter()
                .map(|loc| JsonLocation {
                    hunk_id: &loc.hunk_id.0,
                    file: loc.file_path.display().to_string(),
                    line: loc.new_range.start,
                    focus_lines: &loc.focus_lines,
                })
                .collect(),
        })
        .collect();
    let report = JsonReport {
        base_sha: &diff.base_sha,
        head_sha: &diff.head_sha,
        items: json_items,
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

pub fn stdout_is_tty() -> bool {
    std::io::stdout().is_terminal()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::classify::{Classification, Classified, Source};
    use crate::consolidate::consolidate_exact;
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

    fn item(level: Level, category: Category, file: &str, line: u32) -> ConsolidatedItem {
        ConsolidatedItem::from(classified(level, category, file, line))
    }

    #[test]
    fn anchor_uses_sha256_hex_and_side_prefix() {
        let anchor = diff_anchor("src/index.js", 42, Side::New);
        let expected_hash = hex::encode(Sha256::digest("src/index.js".as_bytes()));
        assert_eq!(anchor, format!("diff-{}R42", expected_hash));
        assert!(diff_anchor("src/index.js", 42, Side::Old).contains('L'));
    }

    #[test]
    fn summary_counts_levels() {
        let cs = vec![
            item(Level::Review, Category::PublicApiChange, "a", 1),
            item(Level::Review, Category::ControlFlow, "b", 1),
            item(Level::Skim, Category::LlmAssessed, "c", 1),
            item(Level::Skip, Category::Lockfile, "Cargo.lock", 1),
        ];
        let s = Summary::from_items(&cs);
        assert_eq!(s.total, 4);
        assert_eq!(s.review, 2);
        assert_eq!(s.skim, 1);
        assert_eq!(s.skip, 1);
    }

    #[test]
    fn human_renders_review_section() {
        let cs = vec![item(
            Level::Review,
            Category::PublicApiChange,
            "src/x.rs",
            42,
        )];
        let out = human(&diff(), &cs, false);
        assert!(out.contains("Review (1)"));
        assert!(out.contains("src/x.rs:42"));
        assert!(out.contains("test rationale"));
        assert!(!out.contains("\x1b["));
    }

    #[test]
    fn human_groups_skip_by_category() {
        let cs = vec![
            item(Level::Skip, Category::Generated, "dist/a.js", 1),
            item(Level::Skip, Category::Generated, "dist/b.js", 1),
            item(Level::Skip, Category::Lockfile, "Cargo.lock", 1),
        ];
        let out = human(&diff(), &cs, false);
        assert!(out.contains("generated (2)"));
        assert!(out.contains("lockfile (1)"));
    }

    #[test]
    fn markdown_starts_with_sticky_marker() {
        let cs = vec![item(Level::Review, Category::ControlFlow, "x", 1)];
        let out = markdown(&diff(), &cs, None);
        assert!(out.starts_with(STICKY_MARKER));
        assert!(out.contains("## Garbelour"));
    }

    #[test]
    fn markdown_review_items_have_no_link_without_repo_ref() {
        let cs = vec![item(Level::Review, Category::ControlFlow, "src/x.rs", 42)];
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
        let cs = vec![item(Level::Review, Category::ControlFlow, "src/x.rs", 7)];
        let out = markdown(&diff(), &cs, Some(&r));
        assert!(out.contains("https://github.com/acme/widget/pull/42/files#diff-"));
        assert!(out.contains("R7"));
    }

    #[test]
    fn markdown_multi_location_item_lists_each_location() {
        let inputs = vec![
            {
                let mut c = classified(Level::Review, Category::PublicApiChange, "src/x.rs", 30);
                c.classification.rationale = "public enum signature changed".into();
                c.classification.focus_lines = Some(FocusLines {
                    start: 28,
                    end: 156,
                    side: Side::New,
                });
                c
            },
            {
                let mut c = classified(Level::Review, Category::PublicApiChange, "src/x.rs", 90);
                c.classification.rationale = "public enum signature changed".into();
                c.classification.focus_lines = Some(FocusLines {
                    start: 28,
                    end: 156,
                    side: Side::New,
                });
                c
            },
        ];
        let consolidated = consolidate_exact(inputs);
        assert_eq!(consolidated.len(), 1);
        let out = markdown(&diff(), &consolidated, None);
        // One bullet, two locators, comma-separated; second locator path elided.
        let bullet = out.lines().find(|l| l.starts_with("- ")).unwrap();
        assert!(bullet.contains("src/x.rs:28–156"));
        assert!(
            bullet.contains("`:28–156`"),
            "second locator should elide path"
        );
        assert!(bullet.contains("public enum signature changed"));
    }

    #[test]
    fn json_renders_locations_array() {
        let cs = vec![item(
            Level::Review,
            Category::PublicApiChange,
            "src/x.rs",
            142,
        )];
        let out = json(&diff(), &cs).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["base_sha"].as_str().unwrap().len(), 40);
        assert_eq!(v["summary"]["total"], 1);
        assert_eq!(v["items"][0]["locations"][0]["file"], "src/x.rs");
        assert_eq!(v["items"][0]["locations"][0]["line"], 142);
        assert_eq!(v["items"][0]["level"], "review");
        assert_eq!(v["items"][0]["category"], "public_api_change");
    }
}
