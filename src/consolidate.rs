//! Consolidation: collapse multiple `Classified` hunks into single display
//! items.
//!
//! Two stages, run in order:
//!
//! 1. **Exact** (`consolidate_exact`): mechanical dedup. Items with the same
//!    `(file_path, focus_lines, side, level, category, rationale)` are merged
//!    into one `ConsolidatedItem` carrying multiple `Location`s. Catches the
//!    common case where one big `pub enum` or `class` spans many lines and
//!    several hunks edit different variants within it — each hunk classifies
//!    against the *enclosing* declaration's signature, producing identical
//!    rationales.
//!
//! 2. **LLM** (`consolidate_llm`, optional): groups items the model judges
//!    "the same change repeated at different locations" — restricted to the
//!    same file. The model returns groupings; this module merges each group
//!    into one item with a merged rationale.

use std::collections::HashSet;
use std::path::PathBuf;

use serde_json::{json, Value};

use crate::classify::{Category, Classification, Classified, FocusLines, Level, Source};
use crate::diff::{HunkId, LineRange};
use crate::llm::{self, LlmConfig, LlmProvider};

/// One place in the diff that a `ConsolidatedItem` points at. A
/// consolidated item carries 1..N of these; rendering shows one locator per
/// location.
#[derive(Clone, Debug)]
pub struct Location {
    pub hunk_id: HunkId,
    pub file_path: PathBuf,
    pub new_range: LineRange,
    pub focus_lines: Option<FocusLines>,
}

/// A render-ready unit. Each item carries one or more `Location`s that share
/// the same classification verdict and rationale.
#[derive(Clone, Debug)]
pub struct ConsolidatedItem {
    pub locations: Vec<Location>,
    pub level: Level,
    pub category: Category,
    pub rationale: String,
    pub source: Source,
}

impl ConsolidatedItem {
    /// The first location — used as the "anchor" for sorting and for the
    /// first locator in the rendered link list.
    pub fn primary(&self) -> &Location {
        &self.locations[0]
    }
}

impl From<Classified> for ConsolidatedItem {
    fn from(c: Classified) -> Self {
        let Classified {
            hunk_id,
            file_path,
            new_range,
            classification,
        } = c;
        let Classification {
            level,
            category,
            rationale,
            source,
            focus_lines,
        } = classification;
        ConsolidatedItem {
            locations: vec![Location {
                hunk_id,
                file_path,
                new_range,
                focus_lines,
            }],
            level,
            category,
            rationale,
            source,
        }
    }
}

/// Stage A: collapse items whose `(file_path, focus_lines, level, category,
/// rationale)` are exactly equal. Always safe — no semantic decision.
pub fn consolidate_exact(items: Vec<Classified>) -> Vec<ConsolidatedItem> {
    let mut out: Vec<ConsolidatedItem> = Vec::new();
    'outer: for c in items {
        let incoming = ConsolidatedItem::from(c);
        for existing in out.iter_mut() {
            if can_merge_exact(existing, &incoming) {
                existing.locations.extend(incoming.locations);
                continue 'outer;
            }
        }
        out.push(incoming);
    }
    out
}

fn can_merge_exact(a: &ConsolidatedItem, b: &ConsolidatedItem) -> bool {
    if a.level != b.level || a.category != b.category || a.rationale != b.rationale {
        return false;
    }
    // All locations in `a` share the same focus by construction; comparing
    // against the primary is sufficient.
    let al = a.primary();
    let bl = b.primary();
    if al.file_path != bl.file_path {
        return false;
    }
    match (&al.focus_lines, &bl.focus_lines) {
        (Some(af), Some(bf)) => af.start == bf.start && af.end == bf.end && af.side == bf.side,
        (None, None) => al.new_range.start == bl.new_range.start,
        _ => false,
    }
}

/// Stage B: ask the LLM which items describe the same change repeated at
/// different locations within the same file, then merge them.
///
/// Only `Review`/`Skim` items are sent to the model; `Skip` items pass
/// through unchanged. Groups are filtered to same-file before applying.
pub fn consolidate_llm(
    items: Vec<ConsolidatedItem>,
    config: &LlmConfig,
) -> anyhow::Result<Vec<ConsolidatedItem>> {
    // Partition: index into `items` for those eligible to be grouped.
    let eligible: Vec<usize> = items
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c.level, Level::Review | Level::Skim))
        .map(|(i, _)| i)
        .collect();
    if eligible.len() < 2 {
        return Ok(items);
    }

    let prompt = build_consolidation_prompt(&items, &eligible);
    let raw = call_llm(&prompt, config)?;
    let groups = parse_consolidation_response(&raw, &eligible, &items);
    Ok(apply_groups(items, groups))
}

const CONSOLIDATION_SYSTEM_PROMPT: &str =
    "You are consolidating duplicate findings from a code review tool. Each finding has an \
     index, a file:line range, and a one-sentence rationale. Identify groups of findings \
     that describe substantively the same kind of change repeated at different locations \
     in the same file (e.g., the same modification pattern applied to a LowPass and a \
     HighPass struct). Do not group findings that merely share a topic but describe \
     different changes. Only group findings from the same file. Items in no group remain \
     as-is. Respond with strict JSON only.";

fn build_consolidation_prompt(items: &[ConsolidatedItem], eligible: &[usize]) -> String {
    let mut out = String::new();
    out.push_str("Findings:\n");
    for &idx in eligible {
        let c = &items[idx];
        let p = c.primary();
        let loc = match &p.focus_lines {
            Some(f) if f.start == f.end => format!("{}:{}", p.file_path.display(), f.start),
            Some(f) => format!("{}:{}–{}", p.file_path.display(), f.start, f.end),
            None => format!("{}:{}", p.file_path.display(), p.new_range.start),
        };
        out.push_str(&format!("[{idx}] {loc}: {}\n", c.rationale));
    }
    out.push_str(concat!(
        "\nRespond ONLY with valid JSON:\n",
        "{\n  \"groups\": [\n    {\n",
        "      \"ids\": [<indices, 2+>],\n",
        "      \"merged_rationale\": \"one sentence covering all grouped locations\"\n",
        "    }\n  ]\n}\n",
        "Each id must appear in at most one group. Findings not listed in any group ",
        "stay as-is.\n",
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
    items: &[ConsolidatedItem],
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
        // Enforce single-file grouping.
        let first_file = &items[ids[0]].primary().file_path;
        if !ids
            .iter()
            .all(|&i| &items[i].primary().file_path == first_file)
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

fn apply_groups(items: Vec<ConsolidatedItem>, groups: Vec<Group>) -> Vec<ConsolidatedItem> {
    if groups.is_empty() {
        return items;
    }
    let claimed: HashSet<usize> = groups.iter().flat_map(|g| g.ids.iter().copied()).collect();
    let mut slots: Vec<Option<ConsolidatedItem>> = items.into_iter().map(Some).collect();
    let mut out: Vec<ConsolidatedItem> = Vec::new();
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
        let mut locations = Vec::new();
        let mut level = Level::Skim;
        let mut category = Category::LlmAssessed;
        let mut source = Source::Heuristic {
            name: "consolidate".into(),
        };
        // Walk in the group's declared id order so locators render naturally.
        for &gi in &group.ids {
            if let Some(item) = slots[gi].take() {
                // Promote to Review if any member was Review.
                if matches!(item.level, Level::Review) {
                    level = Level::Review;
                }
                category = item.category;
                source = item.source;
                locations.extend(item.locations);
            }
        }
        out.push(ConsolidatedItem {
            locations,
            level,
            category,
            rationale: group.rationale.clone(),
            source,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::classify::{Side, Source};
    use crate::diff::{HunkId, LineRange};

    fn make(file: &str, line: u32, focus: Option<(u32, u32)>, rationale: &str) -> Classified {
        Classified {
            hunk_id: HunkId(format!("{file}:{line}")),
            file_path: PathBuf::from(file),
            new_range: LineRange {
                start: line,
                count: 1,
            },
            classification: Classification {
                level: Level::Review,
                category: Category::PublicApiChange,
                rationale: rationale.into(),
                source: Source::Heuristic {
                    name: "public_api".into(),
                },
                focus_lines: focus.map(|(s, e)| FocusLines {
                    start: s,
                    end: e,
                    side: Side::New,
                }),
            },
        }
    }

    #[test]
    fn exact_dedup_collapses_identical_focus_and_rationale() {
        let items = vec![
            make(
                "src/a.rs",
                30,
                Some((28, 156)),
                "public enum signature changed",
            ),
            make(
                "src/a.rs",
                90,
                Some((28, 156)),
                "public enum signature changed",
            ),
        ];
        let out = consolidate_exact(items);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].locations.len(), 2);
        assert_eq!(out[0].rationale, "public enum signature changed");
    }

    #[test]
    fn exact_dedup_keeps_different_rationale_separate() {
        let items = vec![
            make(
                "src/a.rs",
                30,
                Some((28, 156)),
                "public enum signature changed",
            ),
            make("src/a.rs", 30, Some((28, 156)), "control flow changed"),
        ];
        let out = consolidate_exact(items);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn exact_dedup_keeps_different_focus_separate() {
        let items = vec![
            make("src/a.rs", 30, Some((28, 156)), "x"),
            make("src/a.rs", 30, Some((200, 220)), "x"),
        ];
        let out = consolidate_exact(items);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn exact_dedup_keeps_different_files_separate() {
        let items = vec![
            make("src/a.rs", 30, Some((28, 156)), "x"),
            make("src/b.rs", 30, Some((28, 156)), "x"),
        ];
        let out = consolidate_exact(items);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn apply_groups_merges_listed_ids_into_one_item() {
        let items: Vec<ConsolidatedItem> = vec![
            make("src/x.rs", 119, Some((119, 122)), "LowPass change").into(),
            make("src/x.rs", 200, Some((200, 204)), "unrelated").into(),
            make("src/x.rs", 245, Some((245, 248)), "HighPass change").into(),
        ];
        let groups = vec![Group {
            ids: vec![0, 2],
            rationale: "Removing poles field storage from filter constructors.".into(),
        }];
        let out = apply_groups(items, groups);
        assert_eq!(out.len(), 2);
        // First item is the merged group (group's first id is 0, so it appears first).
        assert_eq!(out[0].locations.len(), 2);
        assert!(out[0].rationale.starts_with("Removing poles"));
        // Second item is the unrelated one, preserved.
        assert_eq!(out[1].rationale, "unrelated");
    }

    #[test]
    fn parse_consolidation_response_rejects_cross_file_groups() {
        let items: Vec<ConsolidatedItem> = vec![
            make("src/a.rs", 1, Some((1, 5)), "ra").into(),
            make("src/b.rs", 1, Some((1, 5)), "rb").into(),
        ];
        let raw = r#"{"groups":[{"ids":[0,1],"merged_rationale":"both"}]}"#;
        let groups = parse_consolidation_response(raw, &[0, 1], &items);
        assert!(groups.is_empty(), "cross-file group should be rejected");
    }

    #[test]
    fn parse_consolidation_response_rejects_overlapping_groups() {
        let items: Vec<ConsolidatedItem> = vec![
            make("src/a.rs", 1, Some((1, 5)), "a").into(),
            make("src/a.rs", 6, Some((6, 10)), "b").into(),
            make("src/a.rs", 11, Some((11, 15)), "c").into(),
        ];
        let raw = r#"{"groups":[
            {"ids":[0,1],"merged_rationale":"first"},
            {"ids":[1,2],"merged_rationale":"second"}
        ]}"#;
        let groups = parse_consolidation_response(raw, &[0, 1, 2], &items);
        assert_eq!(groups.len(), 1, "second group claims an already-claimed id");
    }
}
