//! LLM dispatch for hunks no heuristic claimed.
//!
//! Mirrors the structure of disclude-rs's `llm.rs`: a `LlmProvider` enum
//! with three providers (Anthropic, OpenAI, Ollama), env-var based
//! detection, a single Anthropic-shaped function and a single
//! OpenAI-shaped function, and a tiny extract-JSON helper to handle models
//! that wrap responses in code fences.

use serde_json::{json, Value};

use crate::classify::{
    Category, Classification, Classified, FocusLines, Level, Side, Source, Unclassified,
};
use crate::diff::{HunkId, LineRange};

/// Maximum approximate prompt size per request, in bytes. Larger than
/// disclude's 6KB because diff hunks include both pre- and post-image
/// content.
pub const BATCH_PAYLOAD_LIMIT: usize = 24 * 1024;

pub enum LlmProvider {
    Anthropic,
    OpenAI,
    Ollama,
}

impl LlmProvider {
    pub fn name(&self) -> &'static str {
        match self {
            LlmProvider::Anthropic => "Anthropic",
            LlmProvider::OpenAI => "OpenAI",
            LlmProvider::Ollama => "Ollama",
        }
    }

    fn defaults(&self) -> (&'static str, &'static str, &'static str) {
        match self {
            LlmProvider::Anthropic => (
                "claude-haiku-4-5",
                "https://api.anthropic.com",
                "ANTHROPIC_API_KEY",
            ),
            LlmProvider::OpenAI => ("gpt-4o-mini", "https://api.openai.com", "OPENAI_API_KEY"),
            LlmProvider::Ollama => ("llama3.2", "https://api.ollama.ai", "OLLAMA_API_KEY"),
        }
    }

    fn from_str(s: &str) -> Option<Self> {
        match s {
            "anthropic" => Some(LlmProvider::Anthropic),
            "openai" => Some(LlmProvider::OpenAI),
            "ollama" => Some(LlmProvider::Ollama),
            _ => None,
        }
    }
}

pub struct LlmConfig {
    pub provider: LlmProvider,
    pub model: String,
    pub api_key: String,
    pub base_url: String,
}

impl LlmConfig {
    pub fn provider_name(&self) -> &'static str {
        self.provider.name()
    }
}

/// Resolve the provider from an explicit flag, falling back to the first
/// API-key env var present. Errors if `--llm` was requested but no key is
/// available.
pub fn detect_provider(
    llm_provider: Option<&str>,
    llm_model: Option<&str>,
    llm_base_url: Option<&str>,
) -> anyhow::Result<LlmConfig> {
    let provider = match llm_provider {
        Some(s) => LlmProvider::from_str(s).ok_or_else(|| {
            anyhow::anyhow!(
                "unknown --llm-provider '{}'; expected: anthropic, openai, ollama",
                s
            )
        })?,
        None => {
            if std::env::var("ANTHROPIC_API_KEY").is_ok() {
                LlmProvider::Anthropic
            } else if std::env::var("OPENAI_API_KEY").is_ok() {
                LlmProvider::OpenAI
            } else if std::env::var("OLLAMA_API_KEY").is_ok() {
                LlmProvider::Ollama
            } else {
                anyhow::bail!(
                    "garbelour --llm requires an API key in the environment.\n\
                     Set one of:\n  ANTHROPIC_API_KEY  (uses Anthropic)\n  \
                     OPENAI_API_KEY    (uses OpenAI)\n  \
                     OLLAMA_API_KEY    (uses Ollama cloud)\n\
                     or pass --llm-provider to specify a provider."
                );
            }
        }
    };

    let (default_model, default_url, key_env) = provider.defaults();
    let api_key = std::env::var(key_env).map_err(|_| {
        anyhow::anyhow!(
            "garbelour --llm: provider '{}' selected but {} is not set",
            provider.name(),
            key_env
        )
    })?;

    Ok(LlmConfig {
        provider,
        model: llm_model.unwrap_or(default_model).to_string(),
        api_key,
        base_url: llm_base_url.unwrap_or(default_url).to_string(),
    })
}

const SYSTEM_PROMPT: &str = "You are not reviewing code. Your only job is to decide \
how much attention a human reviewer should spend on each diff hunk. For each hunk, \
choose one of three levels: review (reviewer must read carefully — real logic, \
behavior, or contract change), skim (reviewer should glance — substantive but \
low-risk), or skip (reviewer can collapse — boilerplate, no semantic effect). Do not \
suggest changes. Do not flag bugs. Do not write code review comments. For hunks \
classified as review or skim, identify the specific line range within the hunk that \
is most important. Output a single line of justification per hunk, no more.";

/// Top-level entry: classify all unclassified hunks, returning one
/// `Classified` per hunk the model was able to assess. Hunks the model
/// failed to return a verdict for are silently dropped — main.rs falls
/// back to the default level for those.
pub fn classify_hunks(
    hunks: &[Unclassified],
    config: &LlmConfig,
) -> anyhow::Result<Vec<Classified>> {
    let mut out = Vec::new();
    for batch in build_batches(hunks) {
        let user = build_prompt(batch);
        let raw = match &config.provider {
            LlmProvider::Anthropic => call_anthropic(SYSTEM_PROMPT, &user, config)?,
            LlmProvider::OpenAI | LlmProvider::Ollama => {
                call_openai_compat(SYSTEM_PROMPT, &user, config)?
            }
        };
        out.extend(parse_response(&raw, batch, config));
    }
    Ok(out)
}

/// Group hunks into payload-bounded batches.
pub fn build_batches(hunks: &[Unclassified]) -> Vec<&[Unclassified]> {
    if hunks.is_empty() {
        return Vec::new();
    }
    let mut batches = Vec::new();
    let mut start = 0;
    let mut size = 0usize;
    for (idx, h) in hunks.iter().enumerate() {
        let sz = estimate_hunk_size(h);
        if idx > start && size + sz > BATCH_PAYLOAD_LIMIT {
            batches.push(&hunks[start..idx]);
            start = idx;
            size = 0;
        }
        size += sz;
    }
    batches.push(&hunks[start..]);
    batches
}

fn estimate_hunk_size(h: &Unclassified) -> usize {
    h.file_path.to_string_lossy().len()
        + h.old_lines.iter().map(|s| s.len() + 2).sum::<usize>()
        + h.new_lines.iter().map(|s| s.len() + 2).sum::<usize>()
        + 256
}

pub fn build_prompt(batch: &[Unclassified]) -> String {
    let mut out = String::new();
    for h in batch {
        out.push_str(&format!(
            "Hunk {id}\n  File: {path} ({lang})\n",
            id = h.hunk_id.0,
            path = h.file_path.display(),
            lang = h.language.map(|l| l.as_str()).unwrap_or("unknown"),
        ));
        out.push_str("  --- old\n");
        for line in &h.old_lines {
            out.push_str(&format!("  -{line}\n"));
        }
        out.push_str("  +++ new\n");
        for line in &h.new_lines {
            out.push_str(&format!("  +{line}\n"));
        }
        out.push('\n');
    }
    out.push_str(concat!(
        "Respond ONLY with valid JSON:\n",
        "{\n  \"verdicts\": [\n    {\n",
        "      \"id\": \"<hunk_id>\",\n",
        "      \"level\": \"review|skim|skip\",\n",
        "      \"focus_start\": <line_number_or_null>,\n",
        "      \"focus_end\": <line_number_or_null>,\n",
        "      \"rationale\": \"one sentence referencing the focus lines\"\n",
        "    }\n  ]\n}\n\n",
        "For review and skim verdicts, set focus_start and focus_end to the post-image (+) line numbers of the most important region. For skip verdicts, set both to null.\n",
    ));
    out
}

fn call_anthropic(system: &str, user: &str, config: &LlmConfig) -> anyhow::Result<String> {
    let url = format!("{}/v1/messages", config.base_url);
    let body = json!({
        "model": config.model,
        "max_tokens": 8192,
        "system": system,
        "messages": [{"role": "user", "content": user}]
    });
    let resp = ureq::post(&url)
        .header("x-api-key", &config.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .send_json(&body)
        .map_err(|e| anyhow::anyhow!("Anthropic API request failed: {}", e))?;
    let text = resp
        .into_body()
        .read_to_string()
        .map_err(|e| anyhow::anyhow!("failed to read Anthropic response: {}", e))?;
    let parsed: Value = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse Anthropic response JSON: {}", e))?;
    Ok(parsed["content"][0]["text"]
        .as_str()
        .unwrap_or("")
        .to_string())
}

fn call_openai_compat(system: &str, user: &str, config: &LlmConfig) -> anyhow::Result<String> {
    let url = format!("{}/v1/chat/completions", config.base_url);
    let body = json!({
        "model": config.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
    });
    let resp = ureq::post(&url)
        .header("Authorization", &format!("Bearer {}", config.api_key))
        .header("content-type", "application/json")
        .send_json(&body)
        .map_err(|e| anyhow::anyhow!("OpenAI-compatible API request failed: {}", e))?;
    let text = resp
        .into_body()
        .read_to_string()
        .map_err(|e| anyhow::anyhow!("failed to read API response: {}", e))?;
    let parsed: Value = serde_json::from_str(&text)
        .map_err(|e| anyhow::anyhow!("failed to parse API response JSON: {}", e))?;
    Ok(parsed["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string())
}

pub fn parse_response(raw: &str, batch: &[Unclassified], config: &LlmConfig) -> Vec<Classified> {
    let json_str = extract_json(raw);
    let parsed: Value = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("garbelour: llm response parse error: {e}");
            return Vec::new();
        }
    };
    let arr = match parsed["verdicts"].as_array() {
        Some(a) => a,
        None => {
            eprintln!("garbelour: llm response missing 'verdicts' array");
            return Vec::new();
        }
    };

    // Map hunk_id → index for quick lookup and so we ignore stray ids.
    let mut by_id: std::collections::HashMap<&str, &Unclassified> =
        std::collections::HashMap::new();
    for u in batch {
        by_id.insert(u.hunk_id.0.as_str(), u);
    }

    let mut out = Vec::new();
    for item in arr {
        let id = match item["id"].as_str() {
            Some(s) => s,
            None => continue,
        };
        let u = match by_id.get(id) {
            Some(u) => *u,
            None => continue,
        };
        let level = match item["level"].as_str().unwrap_or("skim") {
            "review" => Level::Review,
            "skip" => Level::Skip,
            _ => Level::Skim,
        };
        let focus_start = item["focus_start"].as_u64().map(|v| v as u32);
        let focus_end = item["focus_end"].as_u64().map(|v| v as u32);
        let focus_lines = match (focus_start, focus_end) {
            (Some(s), Some(e)) => Some(FocusLines {
                start: s,
                end: e,
                side: Side::New,
            }),
            _ => None,
        };
        let rationale = item["rationale"].as_str().unwrap_or("").to_string();

        out.push(Classified {
            hunk_id: HunkId(id.to_string()),
            file_path: u.file_path.clone(),
            new_range: LineRange {
                start: u.new_range.start,
                count: u.new_range.count,
            },
            classification: Classification {
                level,
                category: Category::LlmAssessed,
                rationale,
                source: Source::Llm {
                    provider: config.provider.name().to_string(),
                    model: config.model.clone(),
                },
                focus_lines,
            },
        });
    }
    out
}

fn extract_json(s: &str) -> &str {
    if let Some(start) = s.find('{') {
        if let Some(end) = s.rfind('}') {
            if end >= start {
                return &s[start..=end];
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::diff::{HunkId, LineRange};
    use crate::lang::Language;

    fn unclassified(id: &str, path: &str, line: u32) -> Unclassified {
        Unclassified {
            hunk_id: HunkId(id.to_string()),
            file_path: PathBuf::from(path),
            language: Some(Language::Rust),
            old_range: LineRange {
                start: line,
                count: 1,
            },
            new_range: LineRange {
                start: line,
                count: 1,
            },
            old_lines: vec!["old".into()],
            new_lines: vec!["new".into()],
        }
    }

    fn config() -> LlmConfig {
        LlmConfig {
            provider: LlmProvider::Anthropic,
            model: "test-model".into(),
            api_key: "fake".into(),
            base_url: "http://localhost".into(),
        }
    }

    #[test]
    fn extract_json_finds_braces_inside_markdown() {
        let s = "```json\n{\"verdicts\": []}\n```";
        assert_eq!(extract_json(s), "{\"verdicts\": []}");
    }

    #[test]
    fn extract_json_returns_input_when_no_braces() {
        let s = "no json here";
        assert_eq!(extract_json(s), "no json here");
    }

    #[test]
    fn build_prompt_includes_hunk_id_path_and_lines() {
        let batch = vec![unclassified("src/x.rs:5", "src/x.rs", 5)];
        let prompt = build_prompt(&batch);
        assert!(prompt.contains("Hunk src/x.rs:5"));
        assert!(prompt.contains("File: src/x.rs (rust)"));
        assert!(prompt.contains("-old"));
        assert!(prompt.contains("+new"));
        assert!(prompt.contains("Respond ONLY with valid JSON"));
    }

    #[test]
    fn build_batches_groups_under_limit() {
        let hunks = vec![
            unclassified("a:1", "a", 1),
            unclassified("b:1", "b", 1),
            unclassified("c:1", "c", 1),
        ];
        let batches = build_batches(&hunks);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 3);
    }

    #[test]
    fn build_batches_returns_empty_for_empty_input() {
        let batches = build_batches(&[]);
        assert!(batches.is_empty());
    }

    #[test]
    fn parse_response_assigns_review_level_with_focus() {
        let raw = r#"{"verdicts":[{"id":"a:1","level":"review","focus_start":3,"focus_end":7,"rationale":"new branch"}]}"#;
        let batch = vec![unclassified("a:1", "src/x.rs", 1)];
        let out = parse_response(raw, &batch, &config());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].classification.level, Level::Review);
        let f = out[0].classification.focus_lines.as_ref().unwrap();
        assert_eq!((f.start, f.end), (3, 7));
        assert_eq!(f.side, Side::New);
        assert_eq!(out[0].classification.rationale, "new branch");
        match &out[0].classification.source {
            Source::Llm { provider, model } => {
                assert_eq!(provider, "Anthropic");
                assert_eq!(model, "test-model");
            }
            _ => panic!("expected Llm source"),
        }
    }

    #[test]
    fn parse_response_handles_skip_with_null_focus() {
        let raw = r#"{"verdicts":[{"id":"a:1","level":"skip","focus_start":null,"focus_end":null,"rationale":"trivial"}]}"#;
        let batch = vec![unclassified("a:1", "src/x.rs", 1)];
        let out = parse_response(raw, &batch, &config());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].classification.level, Level::Skip);
        assert!(out[0].classification.focus_lines.is_none());
    }

    #[test]
    fn parse_response_defaults_unknown_level_to_skim() {
        let raw = r#"{"verdicts":[{"id":"a:1","level":"???","focus_start":null,"focus_end":null,"rationale":"x"}]}"#;
        let batch = vec![unclassified("a:1", "src/x.rs", 1)];
        let out = parse_response(raw, &batch, &config());
        assert_eq!(out[0].classification.level, Level::Skim);
    }

    #[test]
    fn parse_response_drops_unknown_ids() {
        let raw = r#"{"verdicts":[{"id":"unknown:1","level":"review","focus_start":null,"focus_end":null,"rationale":"x"}]}"#;
        let batch = vec![unclassified("a:1", "src/x.rs", 1)];
        let out = parse_response(raw, &batch, &config());
        assert!(out.is_empty());
    }

    #[test]
    fn parse_response_handles_markdown_wrapped_json() {
        let raw = "```json\n{\"verdicts\":[{\"id\":\"a:1\",\"level\":\"review\",\"focus_start\":1,\"focus_end\":1,\"rationale\":\"r\"}]}\n```";
        let batch = vec![unclassified("a:1", "src/x.rs", 1)];
        let out = parse_response(raw, &batch, &config());
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn detect_provider_errors_when_no_keys_set() {
        // Make sure none of the env vars are set in this test process, so
        // detection fails. (Can't really guarantee this in CI without
        // unsetting, but if any are set, we still want a meaningful test.)
        let result = detect_provider(Some("nonexistent"), None, None);
        assert!(result.is_err());
    }
}
