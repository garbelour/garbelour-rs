//! Consolidation: turn per-hunk findings into render-ready items grouped
//! by line range.
//!
//! Three operations, run in order:
//!
//! 1. **Stage A1 — exact finding dedup** (`consolidate_exact`): mechanical.
//!    Findings with identical `(file_path, focus_lines, level, category,
//!    rationale)` collapse into one, merging their hunk provenance. Catches
//!    the case where one big `pub enum` spans several hunks that each
//!    classify against the same enclosing signature.
//!
//! 2. **Stage A2 — range-grouping into items** (`group_by_range`): findings
//!    on the same `(file, side)` whose focus ranges overlap collapse into
//!    one `Item`. One item = one PR deep-link, but may carry many findings
//!    rendered as a nested bullet list. Touching-but-not-overlapping ranges
//!    (10–15 and 16–20) stay separate items.
//!
//! 3. **Stage B — LLM cross-location grouping** (`consolidate_llm`,
//!    optional): asks the model which items describe "the same change
//!    repeated at different locations" within one file, and unions their
//!    findings under one item (plus a synthesized summary finding from the
//!    model's merged rationale).

use std::collections::HashSet;
use std::path::PathBuf;

use serde_json::{json, Value};

use crate::classify::{Category, Finding, HunkFindings, Level, Side, Source};
use crate::diff::{HunkId, LineRange};
use crate::llm::{self, LlmConfig, LlmProvider};

/// One place in the diff a `Item` points at. Items carry 1..N of these so
/// JSON output can list every contributing hunk_id; the visible PR anchor
/// is derived from the item's merged range, not from any single location.
#[derive(Clone, Debug)]
pub struct Location {
    pub hunk_id: HunkId,
    pub new_range: LineRange,
    pub old_range: LineRange,
}

/// Inclusive line range. Two ranges overlap iff
/// `a.start <= b.end && b.start <= a.end`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InclusiveLineRange {
    pub start: u32,
    pub end: u32,
}

impl InclusiveLineRange {
    pub fn overlaps(self, other: Self) -> bool {
        self.start <= other.end && other.start <= self.end
    }
}

/// A render-ready unit: one or more findings sharing an overlapping focus
/// range on the same (file, side). The item is the unit of PR-link
/// anchoring; findings inside it render as a nested list.
#[derive(Clone, Debug)]
pub struct Item {
    pub file_path: PathBuf,
    pub side: Side,
    pub range: InclusiveLineRange,
    pub locations: Vec<Location>,
    pub findings: Vec<Finding>,
}

impl Item {
    /// Highest severity of any contained finding. Drives section bucketing
    /// and anchor color.
    pub fn level(&self) -> Level {
        self.findings
            .iter()
            .map(|f| f.level)
            .reduce(Level::max)
            .unwrap_or(Level::Skim)
    }

    /// Highest-severity finding (tie-break: first occurrence). Returns
    /// `None` for an `Item` with no findings — fields are public, so an
    /// external constructor could leave them empty. The standard
    /// consolidation pipeline always produces non-empty findings.
    pub fn headline_finding(&self) -> Option<&Finding> {
        if self.findings.is_empty() {
            return None;
        }
        let max_level = self.level();
        self.findings.iter().find(|f| f.level == max_level)
    }
}

/// Per-finding provenance used by Stage A1 (exact dedup) before items are
/// formed. Internal-only.
#[derive(Clone, Debug)]
struct FindingWithProvenance {
    file_path: PathBuf,
    hunk_id: HunkId,
    new_range: LineRange,
    old_range: LineRange,
    finding: Finding,
    /// Additional hunk-level provenance picked up during dedup. Empty for
    /// "fresh" findings; grows when two findings with identical metadata
    /// across different hunks collapse.
    merged: Vec<(HunkId, LineRange, LineRange)>,
}

/// Flatten the per-hunk findings into a list keyed by finding, carrying
/// hunk provenance with each.
fn flatten(input: Vec<HunkFindings>) -> Vec<FindingWithProvenance> {
    let mut out = Vec::with_capacity(input.iter().map(|h| h.findings.len()).sum());
    for hf in input {
        for f in hf.findings {
            out.push(FindingWithProvenance {
                file_path: hf.file_path.clone(),
                hunk_id: hf.hunk_id.clone(),
                new_range: hf.new_range.clone(),
                old_range: hf.old_range.clone(),
                finding: f,
                merged: Vec::new(),
            });
        }
    }
    out
}

/// Stage A1: collapse findings whose metadata is exactly equal. Surviving
/// finding accumulates all contributing hunks in `merged`.
pub fn consolidate_exact(input: Vec<HunkFindings>) -> Vec<Item> {
    let flat = flatten(input);
    let mut deduped: Vec<FindingWithProvenance> = Vec::new();
    'outer: for inc in flat {
        for existing in deduped.iter_mut() {
            if exact_match(existing, &inc) {
                existing.merged.push((
                    inc.hunk_id.clone(),
                    inc.new_range.clone(),
                    inc.old_range.clone(),
                ));
                continue 'outer;
            }
        }
        deduped.push(inc);
    }
    group_by_range(deduped)
}

fn exact_match(a: &FindingWithProvenance, b: &FindingWithProvenance) -> bool {
    if a.file_path != b.file_path
        || a.finding.level != b.finding.level
        || a.finding.category != b.finding.category
        || a.finding.rationale != b.finding.rationale
    {
        return false;
    }
    match (&a.finding.focus_lines, &b.finding.focus_lines) {
        (Some(af), Some(bf)) => af.start == bf.start && af.end == bf.end && af.side == bf.side,
        (None, None) => a.new_range.start == b.new_range.start,
        _ => false,
    }
}

/// Stage A2: group deduped findings into items by overlapping focus ranges
/// on the same `(file, side)`. Findings with no `focus_lines` fall back to
/// the hunk's `new_range` (side `New`).
fn group_by_range(input: Vec<FindingWithProvenance>) -> Vec<Item> {
    // Resolve each finding to (range, side) + carry provenance.
    let mut resolved: Vec<(InclusiveLineRange, Side, FindingWithProvenance)> = input
        .into_iter()
        .map(|f| {
            let (range, side) = effective_range(&f);
            (range, side, f)
        })
        .collect();

    // Sort by (file, side, range.start, range.end) so overlap detection is a
    // single sweep within each (file, side) group.
    resolved.sort_by(|a, b| {
        a.2.file_path
            .cmp(&b.2.file_path)
            .then((a.1 as u8).cmp(&(b.1 as u8)))
            .then(a.0.start.cmp(&b.0.start))
            .then(a.0.end.cmp(&b.0.end))
    });

    let mut out: Vec<Item> = Vec::new();
    let mut current: Option<Item> = None;
    for (range, side, fwp) in resolved {
        let can_extend = current.as_ref().is_some_and(|cur| {
            cur.file_path == fwp.file_path && cur.side == side && cur.range.overlaps(range)
        });
        if can_extend {
            let cur = current.as_mut().unwrap();
            cur.range.end = cur.range.end.max(range.end);
            push_finding_into(cur, fwp);
        } else {
            if let Some(item) = current.take() {
                out.push(item);
            }
            current = Some(new_item(range, side, fwp));
        }
    }
    if let Some(item) = current {
        out.push(item);
    }
    out
}

fn new_item(range: InclusiveLineRange, side: Side, fwp: FindingWithProvenance) -> Item {
    let mut item = Item {
        file_path: fwp.file_path.clone(),
        side,
        range,
        locations: Vec::new(),
        findings: Vec::new(),
    };
    push_finding_into(&mut item, fwp);
    item
}

fn push_finding_into(item: &mut Item, fwp: FindingWithProvenance) {
    // Locations are deduped by hunk_id so a single hunk contributing many
    // findings to the same item still produces only one location.
    let primary_loc = Location {
        hunk_id: fwp.hunk_id.clone(),
        new_range: fwp.new_range.clone(),
        old_range: fwp.old_range.clone(),
    };
    if !item
        .locations
        .iter()
        .any(|l| l.hunk_id == primary_loc.hunk_id)
    {
        item.locations.push(primary_loc);
    }
    for (h, nr, or_) in fwp.merged {
        if !item.locations.iter().any(|l| l.hunk_id == h) {
            item.locations.push(Location {
                hunk_id: h,
                new_range: nr,
                old_range: or_,
            });
        }
    }
    item.findings.push(fwp.finding);
}

/// Effective range for grouping. If a finding has explicit `focus_lines`
/// use them; otherwise fall back to the hunk's `new_range` on `Side::New`.
fn effective_range(fwp: &FindingWithProvenance) -> (InclusiveLineRange, Side) {
    if let Some(f) = &fwp.finding.focus_lines {
        let end = f.end.max(f.start);
        return (
            InclusiveLineRange {
                start: f.start,
                end,
            },
            f.side,
        );
    }
    let start = fwp.new_range.start;
    let count = fwp.new_range.count;
    let end = if count == 0 { start } else { start + count - 1 };
    (InclusiveLineRange { start, end }, Side::New)
}

/// Stage B: ask the LLM which items describe the same change repeated at
/// different locations within the same file, then merge them.
///
/// Only items whose `level()` is `Review`/`Skim` participate; `Skip`-only
/// items pass through unchanged. Groups are filtered to same-file.
pub fn consolidate_llm(items: Vec<Item>, config: &LlmConfig) -> anyhow::Result<Vec<Item>> {
    let eligible: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c.level(), Level::Review | Level::Skim))
        .map(|(i, _)| i)
        .collect();
    if eligible.len() < 2 {
        return Ok(items);
    }

    let prompt = build_consolidation_prompt(&items, &eligible);
    let raw = call_llm(&prompt, config)?;
    let groups = parse_consolidation_response(&raw, &eligible, &items);
    Ok(apply_groups(items, groups, config))
}

const CONSOLIDATION_SYSTEM_PROMPT: &str =
    "You are consolidating duplicate findings from a code review tool. Each finding has an \
     index, a file:line range, a diff side (NEW = post-image, OLD = pre-image), and one or \
     more one-sentence rationales. Identify groups of findings that describe substantively \
     the same kind of change repeated at different locations in the same file (e.g., the same \
     modification pattern applied to a LowPass and a HighPass struct). Do not group findings \
     that merely share a topic but describe different changes. Only group findings from the \
     same file AND the same side — never mix NEW-side and OLD-side items. Items in no group \
     remain as-is. Respond with strict JSON only.";

fn build_consolidation_prompt(items: &[Item], eligible: &[usize]) -> String {
    let mut out = String::new();
    out.push_str("Findings:\n");
    for &idx in eligible {
        let c = &items[idx];
        let side_label = match c.side {
            Side::New => "NEW",
            Side::Old => "OLD",
        };
        let loc = format!(
            "{}:{}-{} [{side_label}]",
            c.file_path.display(),
            c.range.start,
            c.range.end
        );
        out.push_str(&format!("[{idx}] {loc}:\n"));
        for f in &c.findings {
            // Cap each rationale to keep prompt size bounded.
            let r: String = f.rationale.chars().take(240).collect();
            out.push_str(&format!("    - {r}\n"));
        }
    }
    out.push_str(concat!(
        "\nRespond ONLY with valid JSON:\n",
        "{\n  \"groups\": [\n    {\n",
        "      \"ids\": [<indices, 2+>],\n",
        "      \"merged_rationale\": \"one sentence covering all grouped locations\"\n",
        "    }\n  ]\n}\n",
        "Each id must appear in at most one group. All ids within a group must share the ",
        "same file AND the same side. Findings not listed in any group stay as-is.\n",
    ));
    out
}

fn call_llm(user: &str, config: &LlmConfig) -> anyhow::Result<String> {
    let body = match &config.provider {
        LlmProvider::Anthropic => json!({
            "model": config.model,
            "max_tokens": 4096,
            "system": CONSOLIDATION_SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": user}]
        }),
        LlmProvider::OpenAI | LlmProvider::Ollama => json!({
            "model": config.model,
            "messages": [
                {"role": "system", "content": CONSOLIDATION_SYSTEM_PROMPT},
                {"role": "user", "content": user}
            ]
        }),
    };
    let url = match &config.provider {
        LlmProvider::Anthropic => format!("{}/v1/messages", config.base_url),
        LlmProvider::OpenAI | LlmProvider::Ollama => {
            format!("{}/v1/chat/completions", config.base_url)
        }
    };
    let req = match &config.provider {
        LlmProvider::Anthropic => ureq::post(&url)
            .header("x-api-key", &config.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json"),
        LlmProvider::OpenAI | LlmProvider::Ollama => ureq::post(&url)
            .header("Authorization", &format!("Bearer {}", config.api_key))
            .header("content-type", "application/json"),
    };
    let resp = req
        .send_json(&body)
        .map_err(|e| anyhow::anyhow!("consolidation LLM request failed: {}", e))?;
    let text = resp
        .into_body()
        .read_to_string()
        .map_err(|e| anyhow::anyhow!("failed to read consolidation response: {}", e))?;
    let parsed: Value = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse consolidation response JSON: {}", e))?;
    Ok(match &config.provider {
        LlmProvider::Anthropic => parsed["content"][0]["text"]
            .as_str()
            .unwrap_or("")
            .to_string(),
        LlmProvider::OpenAI | LlmProvider::Ollama => parsed["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string(),
    })
}

/// Decoded group from the LLM response. `ids` are indices into the original
/// `items` slice passed to `consolidate_llm`.
#[derive(Debug)]
pub(crate) struct Group {
    pub(crate) ids: Vec<usize>,
    pub(crate) rationale: String,
}

pub(crate) fn parse_consolidation_response(
    raw: &str,
    eligible: &[usize],
    items: &[Item],
) -> Vec<Group> {
    let json_str = llm::extract_json_for_consolidation(raw);
    let parsed: Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("garbelour: consolidation response parse error: {e}");
            return Vec::new();
        }
    };
    let arr = match parsed["groups"].as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };
    let eligible_set: HashSet<usize> = eligible.iter().copied().collect();
    let mut claimed: HashSet<usize> = HashSet::new();
    let mut out = Vec::new();
    for g in arr {
        let ids: Vec<usize> = match g["ids"].as_array() {
            Some(a) => a
                .iter()
                .filter_map(|v| v.as_u64().map(|n| n as usize))
                .collect(),
            None => continue,
        };
        if ids.len() < 2 {
            continue;
        }
        if !ids
            .iter()
            .all(|i| eligible_set.contains(i) && !claimed.contains(i))
        {
            continue;
        }
        // Enforce single-file, single-side grouping. Cross-side merges
        // would inherit the primary's side and produce incorrect anchors
        // (e.g. an old-side item silently absorbed under a new-side primary
        // would link to a non-existent post-image line).
        let first_file = &items[ids[0]].file_path;
        let first_side = items[ids[0]].side;
        if !ids
            .iter()
            .all(|&i| &items[i].file_path == first_file && items[i].side == first_side)
        {
            continue;
        }
        let rationale = g["merged_rationale"].as_str().unwrap_or("").to_string();
        if rationale.is_empty() {
            continue;
        }
        for &i in &ids {
            claimed.insert(i);
        }
        out.push(Group { ids, rationale });
    }
    out
}

fn apply_groups(items: Vec<Item>, groups: Vec<Group>, config: &LlmConfig) -> Vec<Item> {
    if groups.is_empty() {
        return items;
    }
    let claimed: HashSet<usize> = groups.iter().flat_map(|g| g.ids.iter().copied()).collect();
    let mut slots: Vec<Option<Item>> = items.into_iter().map(Some).collect();
    let mut out: Vec<Item> = Vec::new();
    let mut emitted: HashSet<usize> = HashSet::new();

    for idx in 0..slots.len() {
        if !claimed.contains(&idx) {
            if let Some(item) = slots[idx].take() {
                out.push(item);
            }
            continue;
        }
        let group_idx = groups.iter().position(|g| g.ids.contains(&idx)).unwrap();
        if !emitted.insert(group_idx) {
            continue;
        }
        let group = &groups[group_idx];

        // Walk in declared id order. The first surviving member fixes the
        // (file_path, side) the rest of the group must agree with. The
        // merged item's `range` is the *union* (min start, max end) across
        // all matched members — a single member's range under-represents
        // the span of a multi-location group. Defense-in-depth: even though
        // `parse_consolidation_response` enforces same-file + same-side,
        // any member whose (file, side) doesn't match is preserved as a
        // standalone item rather than silently merged.
        let mut primary_id_and_side: Option<(PathBuf, Side)> = None;
        for &gi in &group.ids {
            if let Some(item) = slots[gi].as_ref() {
                primary_id_and_side = Some((item.file_path.clone(), item.side));
                break;
            }
        }
        let Some((file_path, side)) = primary_id_and_side else {
            continue;
        };

        let mut merged_findings: Vec<Finding> = Vec::new();
        let mut locations: Vec<Location> = Vec::new();
        let mut matched_members: usize = 0;
        let mut range: Option<InclusiveLineRange> = None;
        for &gi in &group.ids {
            if let Some(item) = slots[gi].take() {
                if item.file_path != file_path || item.side != side {
                    out.push(item);
                    continue;
                }
                matched_members += 1;
                range = Some(match range {
                    None => item.range,
                    Some(r) => InclusiveLineRange {
                        start: r.start.min(item.range.start),
                        end: r.end.max(item.range.end),
                    },
                });
                for loc in item.locations {
                    if !locations.iter().any(|l| l.hunk_id == loc.hunk_id) {
                        locations.push(loc);
                    }
                }
                merged_findings.extend(item.findings);
            }
        }
        let Some(range) = range else {
            continue;
        };

        // If fewer than two members actually agreed on (file, side), don't
        // synthesize a summary — emit the lone primary as a standalone item
        // so the LLM's "these N describe the same change" rationale isn't
        // attached to a single-member result.
        if matched_members < 2 {
            out.push(Item {
                file_path,
                side,
                range,
                locations,
                findings: merged_findings,
            });
            continue;
        }

        // Synthesized summary finding from the model's merged rationale,
        // prepended so it renders first. Level = max of contained findings.
        let max_level = merged_findings
            .iter()
            .map(|f| f.level)
            .reduce(Level::max)
            .unwrap_or(Level::Skim);
        let summary = Finding {
            level: max_level,
            category: Category::LlmAssessed,
            rationale: group.rationale.clone(),
            source: Source::Llm {
                provider: config.provider.name().to_string(),
                model: config.model.clone(),
            },
            focus_lines: None,
        };
        let mut findings = Vec::with_capacity(merged_findings.len() + 1);
        findings.push(summary);
        findings.extend(merged_findings);

        out.push(Item {
            file_path,
            side,
            range,
            locations,
            findings,
        });
    }

    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::classify::{FocusLines, Side, Source};
    use crate::diff::{HunkId, LineRange};

    fn hunk_findings(file: &str, hunk_line: u32, findings: Vec<Finding>) -> HunkFindings {
        HunkFindings {
            hunk_id: HunkId(format!("{file}:{hunk_line}")),
            file_path: PathBuf::from(file),
            old_range: LineRange {
                start: hunk_line,
                count: 1,
            },
            new_range: LineRange {
                start: hunk_line,
                count: 1,
            },
            findings,
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

    #[test]
    fn exact_dedup_collapses_findings_with_identical_metadata() {
        // Two hunks each producing a "public enum signature changed" finding
        // at the same focus range collapse into one item with one finding
        // and two locations.
        let h1 = hunk_findings(
            "src/a.rs",
            30,
            vec![finding(
                Level::Review,
                Category::PublicApiChange,
                "public enum signature changed",
                Some((28, 156, Side::New)),
            )],
        );
        let h2 = hunk_findings(
            "src/a.rs",
            90,
            vec![finding(
                Level::Review,
                Category::PublicApiChange,
                "public enum signature changed",
                Some((28, 156, Side::New)),
            )],
        );
        let out = consolidate_exact(vec![h1, h2]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].findings.len(), 1);
        assert_eq!(out[0].locations.len(), 2);
    }

    #[test]
    fn exact_dedup_keeps_different_rationale_separate() {
        let h1 = hunk_findings(
            "src/a.rs",
            30,
            vec![finding(
                Level::Review,
                Category::PublicApiChange,
                "a",
                Some((28, 156, Side::New)),
            )],
        );
        let h2 = hunk_findings(
            "src/a.rs",
            31,
            vec![finding(
                Level::Review,
                Category::PublicApiChange,
                "b",
                Some((28, 156, Side::New)),
            )],
        );
        // Same focus range so they group into ONE item, but as TWO findings.
        let out = consolidate_exact(vec![h1, h2]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].findings.len(), 2);
    }

    #[test]
    fn range_grouping_merges_overlapping_focus_into_one_item() {
        let h1 = hunk_findings(
            "src/a.rs",
            30,
            vec![finding(
                Level::Review,
                Category::PublicApiChange,
                "sig change",
                Some((30, 40, Side::New)),
            )],
        );
        let h2 = hunk_findings(
            "src/a.rs",
            35,
            vec![finding(
                Level::Review,
                Category::ControlFlow,
                "branch added",
                Some((35, 50, Side::New)),
            )],
        );
        let out = consolidate_exact(vec![h1, h2]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].range.start, 30);
        assert_eq!(out[0].range.end, 50);
        assert_eq!(out[0].findings.len(), 2);
        assert_eq!(out[0].locations.len(), 2);
    }

    #[test]
    fn range_grouping_keeps_adjacent_but_non_overlapping_ranges_separate() {
        let h1 = hunk_findings(
            "src/a.rs",
            10,
            vec![finding(
                Level::Review,
                Category::PublicApiChange,
                "a",
                Some((10, 15, Side::New)),
            )],
        );
        let h2 = hunk_findings(
            "src/a.rs",
            16,
            vec![finding(
                Level::Review,
                Category::PublicApiChange,
                "b",
                Some((16, 20, Side::New)),
            )],
        );
        let out = consolidate_exact(vec![h1, h2]);
        assert_eq!(
            out.len(),
            2,
            "ranges 10-15 and 16-20 are adjacent, not overlapping"
        );
    }

    #[test]
    fn range_grouping_keeps_opposite_sides_separate() {
        let h1 = hunk_findings(
            "src/a.rs",
            10,
            vec![finding(
                Level::Review,
                Category::PublicApiChange,
                "new",
                Some((30, 40, Side::New)),
            )],
        );
        let h2 = hunk_findings(
            "src/a.rs",
            10,
            vec![finding(
                Level::Review,
                Category::ErrorHandlingDeleted,
                "old",
                Some((30, 40, Side::Old)),
            )],
        );
        let out = consolidate_exact(vec![h1, h2]);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn one_hunks_overlapping_findings_collapse_locations() {
        // One hunk emits two findings with overlapping ranges → one item
        // with both findings but only one location (deduped by hunk_id).
        let h = hunk_findings(
            "src/a.rs",
            30,
            vec![
                finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "sig",
                    Some((30, 40, Side::New)),
                ),
                finding(
                    Level::Skim,
                    Category::ControlFlow,
                    "branch",
                    Some((35, 45, Side::New)),
                ),
            ],
        );
        let out = consolidate_exact(vec![h]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].findings.len(), 2);
        assert_eq!(out[0].locations.len(), 1);
    }

    #[test]
    fn item_level_is_max_severity_of_findings() {
        let h = hunk_findings(
            "src/a.rs",
            30,
            vec![
                finding(
                    Level::Skim,
                    Category::ControlFlow,
                    "skim",
                    Some((30, 40, Side::New)),
                ),
                finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "review",
                    Some((30, 40, Side::New)),
                ),
            ],
        );
        let out = consolidate_exact(vec![h]);
        assert_eq!(out[0].level(), Level::Review);
    }

    #[test]
    fn finding_with_no_focus_falls_back_to_hunk_new_range_on_new_side() {
        let h = hunk_findings(
            "src/a.rs",
            42,
            vec![finding(Level::Skip, Category::Generated, "g", None)],
        );
        let out = consolidate_exact(vec![h]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].side, Side::New);
        assert_eq!(out[0].range.start, 42);
    }

    #[test]
    fn parse_consolidation_response_rejects_cross_file_groups() {
        let items = vec![
            Item {
                file_path: PathBuf::from("src/a.rs"),
                side: Side::New,
                range: InclusiveLineRange { start: 1, end: 5 },
                locations: vec![Location {
                    hunk_id: HunkId("a:1".into()),
                    new_range: LineRange { start: 1, count: 5 },
                    old_range: LineRange { start: 1, count: 5 },
                }],
                findings: vec![finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "ra",
                    Some((1, 5, Side::New)),
                )],
            },
            Item {
                file_path: PathBuf::from("src/b.rs"),
                side: Side::New,
                range: InclusiveLineRange { start: 1, end: 5 },
                locations: vec![Location {
                    hunk_id: HunkId("b:1".into()),
                    new_range: LineRange { start: 1, count: 5 },
                    old_range: LineRange { start: 1, count: 5 },
                }],
                findings: vec![finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "rb",
                    Some((1, 5, Side::New)),
                )],
            },
        ];
        let raw = r#"{"groups":[{"ids":[0,1],"merged_rationale":"both"}]}"#;
        let groups = parse_consolidation_response(raw, &[0, 1], &items);
        assert!(groups.is_empty(), "cross-file group should be rejected");
    }

    /// Multi-location Stage-B groups expose a merged range that spans
    /// every matched member, not just the primary's slice.
    #[test]
    fn apply_groups_unions_member_ranges_into_merged_range() {
        let cfg = LlmConfig {
            provider: LlmProvider::Anthropic,
            model: "test".into(),
            api_key: "x".into(),
            base_url: "http://localhost".into(),
        };
        let items = vec![
            Item {
                file_path: PathBuf::from("src/a.rs"),
                side: Side::New,
                range: InclusiveLineRange { start: 30, end: 40 },
                locations: vec![Location {
                    hunk_id: HunkId("a:30".into()),
                    new_range: LineRange {
                        start: 30,
                        count: 10,
                    },
                    old_range: LineRange {
                        start: 30,
                        count: 10,
                    },
                }],
                findings: vec![finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "lowpass",
                    Some((30, 40, Side::New)),
                )],
            },
            Item {
                file_path: PathBuf::from("src/a.rs"),
                side: Side::New,
                range: InclusiveLineRange {
                    start: 200,
                    end: 220,
                },
                locations: vec![Location {
                    hunk_id: HunkId("a:200".into()),
                    new_range: LineRange {
                        start: 200,
                        count: 21,
                    },
                    old_range: LineRange {
                        start: 200,
                        count: 21,
                    },
                }],
                findings: vec![finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "highpass",
                    Some((200, 220, Side::New)),
                )],
            },
        ];
        let groups = vec![Group {
            ids: vec![0, 1],
            rationale: "same change applied to LowPass and HighPass".into(),
        }];
        let out = apply_groups(items, groups, &cfg);
        assert_eq!(out.len(), 1);
        // Union of (30..=40) and (200..=220).
        assert_eq!(out[0].range.start, 30);
        assert_eq!(out[0].range.end, 220);
        assert_eq!(out[0].locations.len(), 2);
        // Summary + 2 member findings.
        assert_eq!(out[0].findings.len(), 3);
    }

    /// `apply_groups` is defense-in-depth: even if a malformed group with
    /// mixed sides slips past `parse_consolidation_response`, the mismatched
    /// member is preserved as a standalone item rather than absorbed under
    /// the primary's anchor.
    #[test]
    fn apply_groups_preserves_off_side_members_as_standalone() {
        let cfg = LlmConfig {
            provider: LlmProvider::Anthropic,
            model: "test".into(),
            api_key: "x".into(),
            base_url: "http://localhost".into(),
        };
        let items = vec![
            // primary: new-side
            Item {
                file_path: PathBuf::from("src/a.rs"),
                side: Side::New,
                range: InclusiveLineRange { start: 1, end: 5 },
                locations: vec![Location {
                    hunk_id: HunkId("a:1".into()),
                    new_range: LineRange { start: 1, count: 5 },
                    old_range: LineRange { start: 1, count: 5 },
                }],
                findings: vec![finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "primary",
                    Some((1, 5, Side::New)),
                )],
            },
            // off-side member: old-side
            Item {
                file_path: PathBuf::from("src/a.rs"),
                side: Side::Old,
                range: InclusiveLineRange { start: 10, end: 15 },
                locations: vec![Location {
                    hunk_id: HunkId("a:10".into()),
                    new_range: LineRange {
                        start: 10,
                        count: 5,
                    },
                    old_range: LineRange {
                        start: 10,
                        count: 5,
                    },
                }],
                findings: vec![finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "off-side",
                    Some((10, 15, Side::Old)),
                )],
            },
        ];
        let groups = vec![Group {
            ids: vec![0, 1],
            rationale: "should be rejected as mixed-side".into(),
        }];
        let out = apply_groups(items, groups, &cfg);
        assert_eq!(out.len(), 2);
        // The off-side member is preserved with its original side and
        // rationale intact.
        let off = out.iter().find(|i| i.side == Side::Old).unwrap();
        assert_eq!(off.findings.len(), 1);
        assert_eq!(off.findings[0].rationale, "off-side");
        // The primary stayed on the new side, single-member (no synthesized
        // summary finding from the LLM).
        let prim = out.iter().find(|i| i.side == Side::New).unwrap();
        assert_eq!(prim.findings.len(), 1);
        assert_eq!(prim.findings[0].rationale, "primary");
    }

    #[test]
    fn parse_consolidation_response_rejects_cross_side_groups() {
        let items = vec![
            Item {
                file_path: PathBuf::from("src/a.rs"),
                side: Side::New,
                range: InclusiveLineRange { start: 1, end: 5 },
                locations: vec![Location {
                    hunk_id: HunkId("a:1".into()),
                    new_range: LineRange { start: 1, count: 5 },
                    old_range: LineRange { start: 1, count: 5 },
                }],
                findings: vec![finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "new",
                    Some((1, 5, Side::New)),
                )],
            },
            Item {
                file_path: PathBuf::from("src/a.rs"),
                side: Side::Old,
                range: InclusiveLineRange { start: 1, end: 5 },
                locations: vec![Location {
                    hunk_id: HunkId("a:1".into()),
                    new_range: LineRange { start: 1, count: 5 },
                    old_range: LineRange { start: 1, count: 5 },
                }],
                findings: vec![finding(
                    Level::Review,
                    Category::PublicApiChange,
                    "old",
                    Some((1, 5, Side::Old)),
                )],
            },
        ];
        let raw = r#"{"groups":[{"ids":[0,1],"merged_rationale":"both"}]}"#;
        let groups = parse_consolidation_response(raw, &[0, 1], &items);
        assert!(
            groups.is_empty(),
            "cross-side group (same file) should be rejected"
        );
    }
}
