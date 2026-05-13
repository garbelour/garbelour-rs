//! Output rendering: human (terminal), markdown (sticky comment), json.
//!
//! All three renderers operate on `&[Item]`. Each item is a finding cluster
//! anchored at one (file, side, range) and may contain multiple findings,
//! each with its own level/category/rationale. The renderer's job is to
//! present one anchor per item plus a per-finding sub-list.

use std::collections::BTreeMap;
use std::io::IsTerminal;

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::classify::{Category, Finding, FocusLines, Level, Side, Source};
use crate::consolidate::Item;
use crate::diff::Diff;

/// Counts at each level. `total` is the number of items (after consolidation),
/// not the underlying finding or hunk count.
#[derive(Clone, Copy, Debug, Default)]
pub struct Summary {
    pub total: usize,
    pub review: usize,
    pub skim: usize,
    pub skip: usize,
}

impl Summary {
    pub fn from_items(items: &[Item]) -> Self {
        let mut s = Summary::default();
        for c in items {
            s.total += 1;
            match c.level() {
                Level::Review => s.review += 1,
                Level::Skim => s.skim += 1,
                Level::Skip => s.skip += 1,
            }
        }
        s
    }
}

pub fn summary_line(items: &[Item]) -> String {
    let s = Summary::from_items(items);
    format!(
        "garbelour: {} of {} items need review, {} worth skimming, {} mechanical",
        s.review, s.total, s.skim, s.skip
    )
}

// --- human ---------------------------------------------------------------

/// Terminal-friendly report. Sections: Review (always), Skim (if any), Skip
/// (grouped by category). When `use_color` is true, ANSI escape codes color
/// the section headers and file:line columns. Items in Review/Skim that
/// contain multiple findings render as a header line + indented sub-bullets,
/// one per additional finding.
pub fn human(_diff: &Diff, items: &[Item], use_color: bool) -> String {
    let mut review = Vec::new();
    let mut skim = Vec::new();
    let mut skip: BTreeMap<Category, Vec<String>> = BTreeMap::new();
    for c in items {
        match c.level() {
            Level::Review => review.push(c),
            Level::Skim => skim.push(c),
            Level::Skip => {
                // Each Skip item rolls into the per-category file-count
                // summary, once per location. A single Skip item may carry
                // multiple Skip findings from different categories (e.g.
                // Generated + Lockfile on the same hunk) — count each so
                // non-headline categories aren't dropped from the breakdown.
                for f in &c.findings {
                    let entry = skip.entry(f.category).or_default();
                    for _loc in &c.locations {
                        entry.push(c.file_path.display().to_string());
                    }
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
    items: &[&Item],
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
        let findings = sorted_findings(c);
        // Defensive: skip an externally-constructed Item with no findings
        // rather than panic on findings[0]. The internal pipeline always
        // produces non-empty findings.
        let Some(head) = findings.first() else {
            continue;
        };
        let pad = " ".repeat(max.saturating_sub(loc.len()) + 4);
        let mut left = format!("    {loc}");
        if use_color {
            left = colorize(&left, section_color);
        }
        out.push_str(&left);
        out.push_str(&pad);
        out.push_str(&head.rationale);
        out.push('\n');
        // Sub-bullets for any additional findings.
        let indent = " ".repeat(max + 4 + 4);
        for f in &findings[1..] {
            let chip = level_chip(f.level);
            let label = category_label(f.category);
            let chip_color = level_color(f.level);
            let mut line = format!("↳ [{chip}] {label}: {}", f.rationale);
            if use_color {
                line = colorize(&line, chip_color);
            }
            out.push_str(&indent);
            out.push_str(&line);
            out.push('\n');
        }
    }
}

/// Findings sorted for display: descending level (Review first), then by
/// focus_lines.start ascending, then heuristic before LLM.
fn sorted_findings(item: &Item) -> Vec<&Finding> {
    let mut v: Vec<&Finding> = item.findings.iter().collect();
    v.sort_by(|a, b| {
        b.level
            .cmp(&a.level)
            .then_with(|| focus_start(a).cmp(&focus_start(b)))
            .then_with(|| source_rank(&a.source).cmp(&source_rank(&b.source)))
    });
    v
}

fn focus_start(f: &Finding) -> u32 {
    f.focus_lines.as_ref().map(|x| x.start).unwrap_or(u32::MAX)
}

fn source_rank(s: &Source) -> u8 {
    match s {
        Source::Heuristic { .. } => 0,
        Source::Llm { .. } => 1,
    }
}

fn level_chip(level: Level) -> &'static str {
    match level {
        Level::Review => "REVIEW",
        Level::Skim => "SKIM",
        Level::Skip => "SKIP",
    }
}

fn level_color(level: Level) -> Color {
    match level {
        Level::Review => Color::Red,
        Level::Skim => Color::Yellow,
        Level::Skip => Color::Dim,
    }
}

/// Build a single locator string for the human renderer. The item supplies
/// its merged range; if there are additional locations (Stage-B merge), they
/// append as `:Lx` suffixes with the path elided when it matches the item's
/// own file.
fn human_locator(c: &Item) -> String {
    let path_str = c.file_path.display().to_string();
    let side_suffix = match c.side {
        Side::Old => " (old)",
        Side::New => "",
    };
    let primary = if c.range.start == c.range.end {
        format!("{}:{}{}", path_str, c.range.start, side_suffix)
    } else {
        format!(
            "{}:{}–{}{}",
            path_str, c.range.start, c.range.end, side_suffix
        )
    };
    if c.locations.len() <= 1 {
        return primary;
    }
    // Stage-B-merged items: append extra-location starts (path elided).
    // Use the item's side to pick old vs new line, otherwise old-side
    // items would emit post-image line numbers.
    let mut parts = vec![primary];
    for loc in c.locations.iter().skip(1) {
        parts.push(format!(":{}", location_line(loc, c.side)));
    }
    parts.join(", ")
}

/// Pick the line number from a `Location` matching the item's side.
fn location_line(loc: &crate::consolidate::Location, side: Side) -> u32 {
    match side {
        Side::New => loc.new_range.start,
        Side::Old => loc.old_range.start,
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
pub fn markdown(_diff: &Diff, items: &[Item], repo_ref: Option<&RepoRef>) -> String {
    let s = Summary::from_items(items);

    let mut review = Vec::new();
    let mut skim = Vec::new();
    let mut skip: BTreeMap<Category, Vec<String>> = BTreeMap::new();
    for c in items {
        match c.level() {
            Level::Review => review.push(c),
            Level::Skim => skim.push(c),
            Level::Skip => {
                // See `human()` — count each Skip finding under its own
                // category so multi-category Skip items are visible in the
                // breakdown.
                for f in &c.findings {
                    let entry = skip.entry(f.category).or_default();
                    for _loc in &c.locations {
                        entry.push(c.file_path.display().to_string());
                    }
                }
            }
        }
    }

    let mut out = String::new();
    out.push_str(STICKY_MARKER);
    out.push_str("\n## Garbelour\n\n");
    out.push_str(&format!(
        "**{} of {} items need review.** {} worth skimming. {} mechanical.\n",
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

fn markdown_item(c: &Item, repo_ref: Option<&RepoRef>) -> String {
    let findings = sorted_findings(c);
    let Some(head) = findings.first() else {
        // Externally-constructed Item with no findings — emit nothing
        // rather than panic on findings[0].
        return String::new();
    };
    let anchor = markdown_anchor(c, repo_ref);
    let mut out = String::new();
    if findings.len() == 1 {
        out.push_str(&format!("- {}: {}\n", anchor, head.rationale));
    } else {
        out.push_str(&format!("- {}:\n", anchor));
        for f in &findings {
            out.push_str(&format!(
                "  - **{}** *({})*: {}\n",
                level_chip(f.level),
                category_label(f.category),
                f.rationale
            ));
        }
    }
    out
}

fn markdown_anchor(c: &Item, repo_ref: Option<&RepoRef>) -> String {
    let path_str = c.file_path.display().to_string();
    let display_line = if c.range.start == c.range.end {
        format!("{}", c.range.start)
    } else {
        format!("{}–{}", c.range.start, c.range.end)
    };
    let mut primary_label = format!("`{}:{}`", path_str, display_line);
    if let Some(r) = repo_ref {
        primary_label = format!(
            "[{}]({})",
            primary_label,
            deep_link(r, &path_str, c.range.start, c.side)
        );
    }
    if c.locations.len() <= 1 {
        return primary_label;
    }
    let mut parts = vec![primary_label];
    for loc in c.locations.iter().skip(1) {
        let line = location_line(loc, c.side);
        let label = format!("`:{}`", line);
        let part = if let Some(r) = repo_ref {
            format!("[{}]({})", label, deep_link(r, &path_str, line, c.side))
        } else {
            label
        };
        parts.push(part);
    }
    parts.join(", ")
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
    schema_version: u32,
    base_sha: &'a str,
    head_sha: &'a str,
    items: Vec<JsonItem<'a>>,
    summary: JsonSummary,
}

#[derive(Serialize)]
struct JsonItem<'a> {
    level: Level,
    file: String,
    side: Side,
    range: JsonRange,
    findings: Vec<JsonFinding<'a>>,
    locations: Vec<JsonLocation<'a>>,
}

#[derive(Serialize)]
struct JsonRange {
    start: u32,
    end: u32,
}

#[derive(Serialize)]
struct JsonFinding<'a> {
    level: Level,
    category: Category,
    rationale: &'a str,
    source: &'a Source,
    focus_lines: &'a Option<FocusLines>,
}

#[derive(Serialize)]
struct JsonLocation<'a> {
    hunk_id: &'a str,
    new_line: u32,
    old_line: u32,
}

#[derive(Serialize)]
struct JsonSummary {
    total: usize,
    review: usize,
    skim: usize,
    skip: usize,
}

pub fn json(diff: &Diff, items: &[Item]) -> anyhow::Result<String> {
    let summary = Summary::from_items(items);
    let json_items: Vec<JsonItem> = items
        .iter()
        .map(|c| JsonItem {
            level: c.level(),
            file: c.file_path.display().to_string(),
            side: c.side,
            range: JsonRange {
                start: c.range.start,
                end: c.range.end,
            },
            findings: c
                .findings
                .iter()
                .map(|f| JsonFinding {
                    level: f.level,
                    category: f.category,
                    rationale: &f.rationale,
                    source: &f.source,
                    focus_lines: &f.focus_lines,
                })
                .collect(),
            locations: c
                .locations
                .iter()
                .map(|loc| JsonLocation {
                    hunk_id: &loc.hunk_id.0,
                    new_line: loc.new_range.start,
                    old_line: loc.old_range.start,
                })
                .collect(),
        })
        .collect();
    let report = JsonReport {
        schema_version: 2,
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
    use crate::classify::{Finding, HunkFindings, Source};
    use crate::consolidate::consolidate_exact;
    use crate::diff::{Diff, HunkId, LineRange};

    fn diff() -> Diff {
        Diff {
            base_sha: "a".repeat(40),
            head_sha: "b".repeat(40),
            files: Vec::new(),
        }
    }

    fn finding(
        level: Level,
        category: Category,
        rationale: &str,
        focus: Option<(u32, u32, Side)>,
    ) -> Finding {
        Finding {
            level,
            category,
            rationale: rationale.into(),
            source: Source::Heuristic {
                name: "test".into(),
            },
            focus_lines: focus.map(|(s, e, side)| FocusLines {
                start: s,
                end: e,
                side,
            }),
        }
    }

    /// Build a single-finding item the way `consolidate_exact` would, so
    /// renderer tests exercise the real pipeline shape.
    fn single_finding_item(level: Level, category: Category, file: &str, line: u32) -> Item {
        let hf = HunkFindings {
            hunk_id: HunkId(format!("{file}:{line}")),
            file_path: PathBuf::from(file),
            old_range: LineRange {
                start: line,
                count: 1,
            },
            new_range: LineRange {
                start: line,
                count: 1,
            },
            findings: vec![finding(level, category, "test rationale", None)],
        };
        consolidate_exact(vec![hf]).into_iter().next().unwrap()
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
            single_finding_item(Level::Review, Category::PublicApiChange, "a", 1),
            single_finding_item(Level::Review, Category::ControlFlow, "b", 1),
            single_finding_item(Level::Skim, Category::LlmAssessed, "c", 1),
            single_finding_item(Level::Skip, Category::Lockfile, "Cargo.lock", 1),
        ];
        let s = Summary::from_items(&cs);
        assert_eq!(s.total, 4);
        assert_eq!(s.review, 2);
        assert_eq!(s.skim, 1);
        assert_eq!(s.skip, 1);
    }

    #[test]
    fn human_renders_review_section() {
        let cs = vec![single_finding_item(
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

    /// A Skip item carrying both a Generated finding AND a Lockfile finding
    /// shows up under both categories — neither is dropped.
    #[test]
    fn human_skip_summary_counts_each_finding_category() {
        let hf = HunkFindings {
            hunk_id: HunkId("Cargo.lock:1".into()),
            file_path: PathBuf::from("Cargo.lock"),
            old_range: LineRange { start: 1, count: 1 },
            new_range: LineRange { start: 1, count: 1 },
            findings: vec![
                finding(Level::Skip, Category::Generated, "generated", None),
                finding(Level::Skip, Category::Lockfile, "lockfile", None),
            ],
        };
        let items = consolidate_exact(vec![hf]);
        let out = human(&diff(), &items, false);
        assert!(out.contains("generated (1)"));
        assert!(out.contains("lockfile (1)"));
    }

    #[test]
    fn human_groups_skip_by_category() {
        let cs = vec![
            single_finding_item(Level::Skip, Category::Generated, "dist/a.js", 1),
            single_finding_item(Level::Skip, Category::Generated, "dist/b.js", 1),
            single_finding_item(Level::Skip, Category::Lockfile, "Cargo.lock", 1),
        ];
        let out = human(&diff(), &cs, false);
        assert!(out.contains("generated (2)"));
        assert!(out.contains("lockfile (1)"));
    }

    #[test]
    fn markdown_starts_with_sticky_marker() {
        let cs = vec![single_finding_item(
            Level::Review,
            Category::ControlFlow,
            "x",
            1,
        )];
        let out = markdown(&diff(), &cs, None);
        assert!(out.starts_with(STICKY_MARKER));
        assert!(out.contains("## Garbelour"));
    }

    #[test]
    fn markdown_review_items_have_no_link_without_repo_ref() {
        let cs = vec![single_finding_item(
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
        let cs = vec![single_finding_item(
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
    fn markdown_multi_finding_item_lists_each_finding_as_nested_bullet() {
        let hf = HunkFindings {
            hunk_id: HunkId("src/x.rs:30".into()),
            file_path: PathBuf::from("src/x.rs"),
            old_range: LineRange {
                start: 30,
                count: 10,
            },
            new_range: LineRange {
                start: 30,
                count: 10,
            },
            findings: vec![
                finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "public enum signature changed",
                    Some((30, 40, Side::New)),
                ),
                finding(
                    Level::Skim,
                    Category::ControlFlow,
                    "branch added",
                    Some((35, 45, Side::New)),
                ),
            ],
        };
        let items = consolidate_exact(vec![hf]);
        assert_eq!(items.len(), 1);
        let out = markdown(&diff(), &items, None);
        assert!(out.contains("`src/x.rs:30–45`"));
        assert!(out.contains("**REVIEW**"));
        assert!(out.contains("**SKIM**"));
        assert!(out.contains("public enum signature changed"));
        assert!(out.contains("branch added"));
    }

    #[test]
    fn json_renders_findings_and_locations_arrays() {
        let cs = vec![single_finding_item(
            Level::Review,
            Category::PublicApiChange,
            "src/x.rs",
            142,
        )];
        let out = json(&diff(), &cs).unwrap();
        let v: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["schema_version"], 2);
        assert_eq!(v["base_sha"].as_str().unwrap().len(), 40);
        assert_eq!(v["summary"]["total"], 1);
        assert_eq!(v["items"][0]["file"], "src/x.rs");
        assert_eq!(v["items"][0]["level"], "review");
        assert_eq!(
            v["items"][0]["findings"][0]["category"],
            "public_api_change"
        );
        assert_eq!(v["items"][0]["locations"][0]["hunk_id"], "src/x.rs:142");
        assert_eq!(v["items"][0]["range"]["start"], 142);
    }
}
