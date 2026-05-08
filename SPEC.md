# Garbelour — Implementation Specification

Garbelour classifies every hunk in a GitHub pull request diff by how much reviewer attention it deserves: **review**, **skim**, or **skip**. It posts a single sticky comment on the PR with deep links to the hunks that matter, collapsing everything else.

The name comes from Anglo-Norman *garbelour*, the medieval official who sifted impurities from imported spices. The tool does the same thing to diffs.

## Repository

`garbelour-rs` — a Rust CLI and library, published to crates.io as `garbelour`.

---

## 1. Project layout

```
garbelour-rs/
├── Cargo.toml
├── README.md
├── LICENSE                     # MIT
├── .github/
│   └── workflows/
│       └── ci.yml
├── src/
│   ├── main.rs                 # CLI entry point
│   ├── lib.rs                  # Public API re-exports
│   ├── cli.rs                  # clap argument definitions
│   ├── config.rs               # garbelour.toml loading
│   ├── error.rs                # Crate-wide error types
│   ├── diff.rs                 # Diff, File, Hunk types and extraction
│   ├── lang.rs                 # Language enum, detection from path/shebang
│   ├── classify.rs             # Classifier trait, Level/Category enums, pipeline
│   ├── classifiers/
│   │   ├── mod.rs              # Re-exports all classifiers
│   │   ├── generated.rs        # Generated-file detection
│   │   ├── lockfile.rs         # Lockfile detection
│   │   ├── comment_only.rs     # Comment/docstring-only changes
│   │   ├── import_reorder.rs   # Import reordering detection
│   │   ├── public_api.rs       # Public API surface changes
│   │   ├── control_flow.rs     # Control flow changes
│   │   ├── error_handling.rs   # Deleted error handling
│   │   └── size_threshold.rs   # Auto-elevate large hunks
│   ├── llm.rs                  # LLM provider dispatch (Anthropic, OpenAI, Ollama)
│   ├── github.rs               # GitHub API client (sticky comment)
│   └── render.rs               # Output rendering (human, markdown, json)
├── tests/
│   ├── fixtures/               # Sample diffs with expected classifications
│   │   ├── rust/
│   │   ├── python/
│   │   └── typescript/
│   └── integration/
│       ├── classify_test.rs
│       └── render_test.rs
```

Follow the conventions established in [disclude-rs](https://github.com/disclude-io/disclude-rs):

- Flat `src/` module layout (no deep nesting), except `classifiers/` which groups the individual classifier implementations.
- `lib.rs` + `main.rs` dual-target: the library is usable independently of the CLI.
- `[lib]` and `[[bin]]` sections in `Cargo.toml` both named `garbelour`.
- Integration tests under `tests/integration/`.

---

## 2. Cargo.toml

```toml
[package]
name = "garbelour"
version = "0.1.0"
edition = "2021"
license = "MIT"
repository = "https://github.com/garbelour-io/garbelour-rs"
readme = "README.md"
keywords = ["code-review", "github", "diff", "triage"]
description = "Classify PR diff hunks by reviewer attention: review, skim, or skip"

[lib]
name = "garbelour"
path = "src/lib.rs"

[[bin]]
name = "garbelour"
path = "src/main.rs"

[dependencies]
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
anyhow = "1"
ureq = { version = "3", features = ["json"] }
sha2 = "0.10"
hex = "0.4"
tree-sitter = "0.26"
tree-sitter-rust = "0.24"
tree-sitter-python = "0.25"
tree-sitter-typescript = "0.23"
ignore = "0.4"
toml = "0.8"

[profile.release]
lto = "thin"
```

Pin `ureq` to a specific version (e.g. `=3.3.0`) once the initial lockfile is generated, matching disclude-rs practice.

---

## 3. Core types

### 3.1 diff.rs

These types represent the parsed diff. All types own their data (no lifetimes). PR diffs are bounded in size and the process is short-lived, so cloning is acceptable.

```rust
use std::path::PathBuf;

/// A complete diff between two git refs.
pub struct Diff {
    pub base_sha: String,
    pub head_sha: String,
    pub files: Vec<FileDiff>,
}

/// A single file's changes.
pub struct FileDiff {
    pub path: PathBuf,
    pub old_path: Option<PathBuf>,      // Some when renamed
    pub status: FileStatus,
    pub language: Option<Language>,      // Detected by lang::detect()
    pub hunks: Vec<Hunk>,
    pub old_content: Option<String>,     // Full pre-image, lazily loaded
    pub new_content: Option<String>,     // Full post-image, lazily loaded
}

pub enum FileStatus {
    Added,
    Modified,
    Deleted,
    Renamed { similarity: u8 },         // 0–100
}

/// A single contiguous change within a file.
pub struct Hunk {
    pub id: HunkId,
    pub old_range: LineRange,
    pub new_range: LineRange,
    pub old_lines: Vec<String>,
    pub new_lines: Vec<String>,
}

/// Stable identifier: "{file_path}:{new_range.start}".
/// Used to correlate heuristic output with LLM output and rendered links.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct HunkId(pub String);

pub struct LineRange {
    pub start: u32,
    pub count: u32,
}
```

**Diff extraction:** shell out to `git diff --no-color -U3 {base_sha}..{head_sha}` and parse the unified diff output. Use the `git diff -M -C` flags to detect renames. This is simpler than using `gix` or `git2` as a library and mirrors what the GitHub Action environment provides. Parse the unified diff format into `FileDiff` and `Hunk` structs.

For each `FileDiff`, populate `old_content` and `new_content` by running `git show {sha}:{path}` when a classifier requests it (lazy — only the classifiers that need tree-sitter parsing will trigger this). Implement a method `FileDiff::ensure_content(&mut self, repo_path: &Path)` that populates these fields on first access.

### 3.2 lang.rs

```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
}

impl Language {
    /// Detect from file extension. Returns None for unsupported languages.
    pub fn detect(path: &Path) -> Option<Language> { ... }

    pub fn as_str(&self) -> &'static str { ... }

    /// File extension patterns for this language.
    pub fn extensions(&self) -> &'static [&'static str] { ... }
}
```

Extension mapping:

| Language   | Extensions                          |
|------------|-------------------------------------|
| Rust       | `.rs`                               |
| Python     | `.py`, `.pyi`                       |
| TypeScript | `.ts`, `.tsx`, `.mts`, `.cts`       |
| JavaScript | `.js`, `.jsx`, `.mjs`, `.cjs`       |

Unsupported languages (Go, C, Java, etc.) return `None`. Hunks in files with `language: None` can still be classified by path-based heuristics (generated, lockfile) but skip AST-based classifiers.

---

## 4. Classification system

### 4.1 Level and Category

```rust
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Review,     // Reviewer should read carefully
    Skim,       // Reviewer should glance
    Skip,       // Reviewer can ignore
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    // Skip categories (mechanical noise)
    Generated,
    Lockfile,
    FormatterOnly,
    CommentOnly,
    ImportReorder,
    PureRename,
    TestFixture,

    // Review categories (load-bearing)
    PublicApiChange,
    ControlFlow,
    ErrorHandlingDeleted,
    LargeChange,

    // LLM-assessed (any level)
    LlmAssessed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Heuristic { name: String },
    Llm { provider: String, model: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Classification {
    pub level: Level,
    pub category: Category,
    pub rationale: String,
    pub source: Source,
    /// Optional line range within the hunk that triggered the classification.
    /// For AST-based classifiers, this is the specific lines the tree-sitter
    /// query matched (e.g. the pub fn signature, the added if-block, the
    /// removed `?` operator). For path-based classifiers (Generated, Lockfile),
    /// this is None — the entire hunk is covered uniformly.
    /// When present, the renderer uses this to tell the reviewer exactly where
    /// to look within a hunk, and the deep link targets this line rather than
    /// the hunk start.
    pub focus_lines: Option<FocusLines>,
}

/// A contiguous range of lines within a hunk that are the reason for its
/// classification. Uses post-image (new file) line numbers for most
/// categories; uses pre-image (old file) line numbers only for deletions
/// (e.g. ErrorHandlingDeleted).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FocusLines {
    pub start: u32,
    pub end: u32,          // inclusive
    pub side: Side,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Old,
    New,
}
```

### 4.2 Classifier trait

```rust
/// A single deterministic classification rule.
///
/// Returns `Some(Classification)` when this classifier claims the hunk.
/// Returns `None` to defer to the next classifier in the pipeline.
///
/// **Line-range rationales:** AST-based classifiers SHOULD populate
/// `Classification.focus_lines` with the specific line range that triggered
/// the classification. This tells the reviewer exactly where to look within
/// a hunk, rather than forcing them to scan the entire hunk. Path-based
/// classifiers (Generated, Lockfile) set `focus_lines: None` because the
/// entire hunk is uniformly mechanical.
pub trait Classifier: Send + Sync {
    /// Stable identifier for logging and source attribution.
    fn name(&self) -> &'static str;

    /// Lower values run first. Path-based classifiers (Generated, Lockfile)
    /// should use 0–99. AST-based classifiers should use 100–199.
    /// Size-threshold (auto-elevate) should use 200+.
    fn priority(&self) -> i32;

    /// Classify a hunk in the context of its containing file.
    /// `file` is mutable to allow lazy content loading via `ensure_content()`.
    fn classify(&self, file: &mut FileDiff, hunk: &Hunk) -> Option<Classification>;
}
```

### 4.3 Pipeline

```rust
pub struct Pipeline {
    classifiers: Vec<Box<dyn Classifier>>,
}

/// A hunk that no heuristic claimed. Passed to the LLM.
pub struct Unclassified<'a> {
    pub file: &'a FileDiff,
    pub hunk: &'a Hunk,
}

/// A hunk with its classification decision.
pub struct Classified {
    pub hunk_id: HunkId,
    pub file_path: PathBuf,
    pub new_range: LineRange,
    pub classification: Classification,
}

impl Pipeline {
    /// Build the standard pipeline with all heuristic classifiers,
    /// configured from the project's garbelour.toml.
    pub fn standard(config: &Config) -> Self { ... }

    /// Run all classifiers against every hunk in the diff.
    /// Returns (classified_hunks, unclassified_hunks).
    /// Classifiers run in priority order; first match wins.
    pub fn run<'a>(&self, diff: &'a mut Diff) -> (Vec<Classified>, Vec<Unclassified<'a>>) { ... }
}
```

---

## 5. Classifiers

Each classifier is a struct in `src/classifiers/` implementing the `Classifier` trait. Listed here in priority order.

### 5.1 Generated (priority: 0)

Path-based. No parsing.

Checks:
1. File path matches a glob in the `generated_globs` config list.
2. File has `linguist-generated=true` in `.gitattributes` (read once, cached).

Default globs: `*.lock`, `package-lock.json`, `*_pb2.py`, `*.pb.go`, `dist/**`, `build/**`, `vendor/**`, `*.min.js`, `*.min.css`, `*.generated.*`.

Returns: `Classification { level: Skip, category: Generated, focus_lines: None, ... }`.

### 5.2 Lockfile (priority: 1)

Path-based. No parsing.

Matches exact filenames: `Cargo.lock`, `package-lock.json`, `yarn.lock`, `pnpm-lock.yaml`, `Gemfile.lock`, `poetry.lock`, `Pipfile.lock`, `composer.lock`, `go.sum`.

Returns: `Classification { level: Skip, category: Lockfile, focus_lines: None, ... }`.

### 5.3 CommentOnly (priority: 100)

AST-based. Requires tree-sitter.

For each hunk:
1. Parse the pre-image and post-image of the enclosing file using tree-sitter.
2. Walk the AST and collect all comment and docstring/doc-comment nodes with their byte ranges.
3. Check whether every changed line in the hunk falls entirely within a comment or docstring node.
4. If yes, the hunk is comment-only.

For files with `language: None`, skip (return `None`).

Returns: `Classification { level: Skip, category: CommentOnly, focus_lines: None, ... }`.

`focus_lines` is `None` because the entire hunk is comment content — there is no specific subrange to highlight.

### 5.4 ImportReorder (priority: 110)

AST-based. Requires tree-sitter.

For each hunk:
1. Parse pre-image and post-image.
2. Extract import/use nodes from both (language-specific: `use_declaration` in Rust, `import_statement`/`import_from_statement` in Python, `import_statement`/`import_clause` in TypeScript/JavaScript).
3. Collect the imported identifiers/paths as a set.
4. If the sets are equal (same imports, different order), it's a reorder.

This only fires when the hunk contains *exclusively* import lines. If there are any non-import changes mixed in, defer.

Returns: `Classification { level: Skip, category: ImportReorder, focus_lines: None, ... }`.

### 5.5 PublicApi (priority: 120)

AST-based. Requires tree-sitter.

Detects changes to the public API surface of a file. Language-specific rules:

**Rust:** any change to a node that is a direct child of a `pub` visibility modifier: `fn_item`, `struct_item`, `enum_item`, `trait_item`, `type_item`, `const_item`, `static_item`. Specifically: if the hunk's changed lines overlap a `pub` declaration's signature (not just its body), classify as `PublicApiChange`.

**Python:** any change to a module-level `def` or `class` statement whose name does not start with `_`. Signature changes (parameters, return type annotation, decorators) count; body-only changes do not trigger this classifier (they'll be caught by ControlFlow or deferred to LLM).

**TypeScript/JavaScript:** any change to an `export_statement` or `export default` node's signature.

Returns: `Classification { level: Review, category: PublicApiChange, ... }`.

`focus_lines` MUST be populated with the line range of the changed public declaration's signature. For example, if lines 142–145 contain a changed `pub fn apply(...)` signature within a hunk spanning lines 140–160, set `focus_lines: Some(FocusLines { start: 142, end: 145, side: Side::New })`. The rationale should reference these lines, e.g. "public fn signature changed at lines 142–145".

### 5.6 ControlFlow (priority: 130)

AST-based. Requires tree-sitter.

Detects added, removed, or structurally modified control-flow nodes. This means:

- Added/removed: `if_expression`, `match_expression`, `for_expression`, `while_expression`, `loop_expression`, `return_statement` (Rust names; equivalent for Python/TS/JS).
- Modified: a control-flow node exists at the same position in both images but its condition or branch structure changed.

Implementation approach: diff the AST node types at the hunk's line range. If control-flow node counts or types differ between pre and post image, classify.

Returns: `Classification { level: Review, category: ControlFlow, ... }`.

`focus_lines` MUST be populated with the line range of the added, removed, or modified control-flow node. Use `Side::New` for added/modified nodes, `Side::Old` for removed nodes. The rationale should reference the specific lines, e.g. "new if-branch at lines 201–208".

### 5.7 ErrorHandlingDeleted (priority: 140)

AST-based. Requires tree-sitter.

Detects *removal* of error handling patterns:

**Rust:** removed `?` operator, removed `match` arm on `Err(...)`, removed `.unwrap_or(...)` / `.unwrap_or_else(...)`, removed `if let Err(...)`.

**Python:** removed `try`/`except` block, removed `except` clause.

**TypeScript/JavaScript:** removed `try`/`catch` block, removed `catch` clause.

Only fires on deletions (lines present in `old_lines` but absent in `new_lines`). Additions of error handling are benign and should not be flagged.

Returns: `Classification { level: Review, category: ErrorHandlingDeleted, ... }`.

`focus_lines` MUST be populated with the line range of the deleted error-handling construct, using `Side::Old` since these lines exist only in the pre-image. The rationale should reference the specific lines, e.g. "removed try/except block at old lines 88–95".

### 5.8 SizeThreshold (priority: 200)

No parsing. Pure line count.

If a hunk has more than 150 changed lines (sum of added + removed), auto-classify as `Review` with `LargeChange`. Large changes deserve human attention by default, and they're too expensive to send to the LLM.

The threshold is configurable via `garbelour.toml`.

Returns: `Classification { level: Review, category: LargeChange, focus_lines: None, ... }`.

`focus_lines` is `None` because the entire hunk is large — the reviewer needs to read all of it.

---

## 6. LLM integration

Follow the pattern established in [disclude-rs `src/llm.rs`](https://github.com/disclude-io/disclude-rs/blob/main/src/llm.rs).

### 6.1 Provider dispatch

```rust
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
            LlmProvider::OpenAI => (
                "gpt-4o-mini",
                "https://api.openai.com",
                "OPENAI_API_KEY",
            ),
            LlmProvider::Ollama => (
                "llama3.2",
                "https://api.ollama.ai",
                "OLLAMA_API_KEY",
            ),
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
```

### 6.2 Provider detection

Same fall-through logic as disclude-rs:

1. If `--llm-provider` is passed, use that.
2. Otherwise, auto-detect from which API key environment variable is set: `ANTHROPIC_API_KEY` → Anthropic, `OPENAI_API_KEY` → OpenAI, `OLLAMA_API_KEY` → Ollama.
3. If no key is set and `--llm` was requested, error with a message listing the expected env vars.
4. `--llm-model` and `--llm-base-url` override the defaults from the provider.

```rust
pub fn detect_provider(
    llm_provider: Option<&str>,
    llm_model: Option<&str>,
    llm_base_url: Option<&str>,
) -> anyhow::Result<LlmConfig> { ... }
```

### 6.3 API calls

Two functions, matching disclude-rs exactly:

```rust
/// Anthropic Messages API.
fn call_anthropic(system: &str, user: &str, config: &LlmConfig) -> anyhow::Result<String> {
    // POST {base_url}/v1/messages
    // Headers: x-api-key, anthropic-version: "2023-06-01", content-type: application/json
    // Body: { model, max_tokens: 8192, system, messages: [{role: "user", content: user}] }
    // Extract: response["content"][0]["text"]
}

/// OpenAI-compatible chat completions (also used for Ollama).
fn call_openai_compat(system: &str, user: &str, config: &LlmConfig) -> anyhow::Result<String> {
    // POST {base_url}/v1/chat/completions
    // Headers: Authorization: Bearer {api_key}, content-type: application/json
    // Body: { model, messages: [{role: "system", content: system}, {role: "user", content: user}] }
    // Extract: response["choices"][0]["message"]["content"]
}
```

Use `ureq` for both. Match the disclude-rs calling convention exactly (header names, JSON shapes, response extraction).

### 6.4 Batching

```rust
const BATCH_PAYLOAD_LIMIT: usize = 24 * 1024;  // 24KB — diff hunks are larger than disclude findings
```

Build batches by accumulating hunks until the estimated prompt size exceeds `BATCH_PAYLOAD_LIMIT`. Estimate size as: file path length + old_lines total length + new_lines total length + 256 bytes overhead per hunk.

### 6.5 System prompt

```
You are not reviewing code. Your only job is to decide how much attention a
human reviewer should spend on each diff hunk. For each hunk, choose one of
three levels: review (reviewer must read carefully — real logic, behavior,
or contract change), skim (reviewer should glance — substantive but
low-risk), or skip (reviewer can collapse — boilerplate, no semantic
effect). Do not suggest changes. Do not flag bugs. Do not write code review
comments. For hunks classified as review or skim, identify the specific
line range within the hunk that is most important. Output a single line of
justification per hunk, no more.
```

### 6.6 User prompt

For each hunk in the batch, format as:

```
Hunk {hunk_id}
  File: {relative_path} ({language})
  --- old
  {old_lines joined by newline, prefixed with -}
  +++ new
  {new_lines joined by newline, prefixed with +}

```

End with the response format instruction:

```
Respond ONLY with valid JSON:
{
  "verdicts": [
    {
      "id": "<hunk_id>",
      "level": "review|skim|skip",
      "focus_start": <line_number_or_null>,
      "focus_end": <line_number_or_null>,
      "rationale": "one sentence referencing the focus lines"
    }
  ]
}

For review and skim verdicts, set focus_start and focus_end to the post-image
(+) line numbers of the most important region. For skip verdicts, set both
to null.
```

### 6.7 Response parsing

Use the same `extract_json` pattern from disclude-rs: find the first `{` and last `}` in the raw response string and parse that slice. This handles models that wrap JSON in markdown code fences or add preamble text.

Parse each verdict into a `Classified` with `Source::Llm { provider, model }` and `Category::LlmAssessed`. Map the `level` string to the `Level` enum; default to `Level::Skim` for unrecognized values. If `focus_start` and `focus_end` are both present and non-null, construct `FocusLines { start, end, side: Side::New }`. Otherwise set `focus_lines: None`.

### 6.8 Top-level entry

```rust
pub fn classify_hunks(
    hunks: &[Unclassified<'_>],
    config: &LlmConfig,
) -> anyhow::Result<Vec<Classified>> { ... }
```

---

## 7. GitHub integration

### 7.1 github.rs

Minimal ureq-based client for two GitHub API endpoints:

```rust
pub struct GitHubClient {
    token: String,
    base_url: String,   // https://api.github.com; configurable for testing
    agent: ureq::Agent,
}

impl GitHubClient {
    /// Build from GITHUB_TOKEN env var.
    pub fn from_env() -> anyhow::Result<Self> { ... }

    /// List comments on an issue/PR.
    /// GET /repos/{owner}/{repo}/issues/{issue_number}/comments
    pub fn list_issue_comments(
        &self, owner: &str, repo: &str, issue: u64,
    ) -> anyhow::Result<Vec<IssueComment>> { ... }

    /// Create a comment.
    /// POST /repos/{owner}/{repo}/issues/{issue_number}/comments
    pub fn create_issue_comment(
        &self, owner: &str, repo: &str, issue: u64, body: &str,
    ) -> anyhow::Result<IssueComment> { ... }

    /// Update an existing comment.
    /// PATCH /repos/{owner}/{repo}/issues/comments/{comment_id}
    pub fn update_issue_comment(
        &self, owner: &str, repo: &str, comment_id: u64, body: &str,
    ) -> anyhow::Result<IssueComment> { ... }
}

#[derive(Deserialize)]
pub struct IssueComment {
    pub id: u64,
    pub body: String,
}
```

### 7.2 Sticky comment

The sticky comment is identified by a hidden HTML marker:

```rust
pub const STICKY_MARKER: &str = "<!-- garbelour:sticky -->";
```

The upsert flow:
1. List all comments on the PR.
2. Find the first comment whose `body` contains `STICKY_MARKER`.
3. If found, PATCH it with the new body.
4. If not found, POST a new comment with the body.

```rust
pub fn upsert_sticky_comment(
    client: &GitHubClient,
    owner: &str,
    repo: &str,
    issue: u64,
    body: &str,
) -> anyhow::Result<IssueComment> { ... }
```

The body MUST begin with `STICKY_MARKER` (on its own line) so the find step works.

### 7.3 Event parsing

In a GitHub Action, the event payload is at `GITHUB_EVENT_PATH`. Parse it to extract:
- `pull_request.number`
- `pull_request.base.sha`
- `pull_request.head.sha`
- `repository.owner.login`
- `repository.name`

```rust
pub struct PrEvent {
    pub owner: String,
    pub repo: String,
    pub pr_number: u64,
    pub base_sha: String,
    pub head_sha: String,
}

pub fn parse_event() -> anyhow::Result<PrEvent> {
    // Read GITHUB_EVENT_PATH, deserialize the JSON, extract fields.
}
```

---

## 8. Rendering

### 8.1 Deep links

GitHub PR file diffs use anchors of the form:

```
https://github.com/{owner}/{repo}/pull/{n}/files#diff-{file_hash}R{line}
```

Where `{file_hash}` is the hex-encoded SHA-256 of the file path as it appears in the diff. `R` prefix = post-image (right side); `L` prefix = pre-image (left side). Use `R` for most findings; use `L` only for `ErrorHandlingDeleted` where the relevant line was removed.

```rust
fn diff_anchor(path: &str, line: u32, side: Side) -> String {
    let hash = hex::encode(sha2::Sha256::digest(path.as_bytes()));
    let prefix = match side { Side::New => 'R', Side::Old => 'L' };
    format!("diff-{}{}{}", hash, prefix, line)
}
```

When `focus_lines` is present, the deep link targets `focus_lines.start` on the appropriate side rather than the hunk's `new_range.start`. This lands the reviewer directly on the important line within the hunk, not just the hunk's beginning.

For renamed files, use the new path. For deleted files, use the old path.

### 8.2 Human renderer (terminal)

The default output for interactive local use. Designed for scanning in a terminal — no markdown, no deep links, no collapsed sections. Matches the `human` format in disclude-rs.

**Example output (with color):**

```
garbelour: 3 of 47 hunks need review, 5 worth skimming, 39 mechanical

  Review
    src/engine.rs:145–148    public fn signature changed
    src/engine.rs:201–208    new branch in retry loop, no termination on Pending
    src/store.rs:88–95       removed ? propagation (old lines)

  Skim
    src/engine.rs:50–62      refactored helper, behavior preserved
    src/config.rs:12–14      default value changed
    src/config.rs:30–33      new config field added
    lib/utils.ts:8–12        type annotation added
    lib/utils.ts:40–44       renamed local variable

  Skip (39 hunks)
    lockfile (1)          Cargo.lock
    generated (15)        proto/*.pb.go, dist/bundle.js, ...
    comment-only (3)      src/engine.rs, src/store.rs, README.md
    import-reorder (8)    src/engine.rs, src/store.rs, ...
    test-fixture (12)     tests/fixtures/*.snap
```

**Layout rules:**

- **Summary line** first, printed to stderr (so it's visible even when stdout is piped). Format: `garbelour: {review} of {total} hunks need review, {skim} worth skimming, {skip} mechanical`.
- **Review section** always shown. Each item on its own line. Left column: file path with focus line range (or hunk start if no focus). Right column: rationale. Columns are tab-aligned to the longest path in the section. When color is enabled, the section header and left column are bold red.
- **Skim section** always shown (unless zero items). Same layout as Review. When color is enabled, section header and left column are yellow.
- **Skip section** shown as a grouped summary, not individual hunks. Each category on its own line: `{category} ({count})` left-aligned, then a truncated list of affected file paths. When color is enabled, section header and left column are dim/gray.
- If a section has zero items, omit it entirely (including the header).
- **Focus line display:** When `focus_lines` is present, show `{file}:{start}–{end}`. When `focus_lines` is `None`, show `{file}:{hunk_start}`. For `Side::Old` focus lines, append `(old)` to signal these are pre-image line numbers.
- **Max width:** truncate rationale text to fit within terminal width (query from `COLUMNS` env var or default to 120). Do not wrap — truncate with `…`.

**Color scheme (ANSI):**

| Element             | ANSI code          | Appearance        |
|---------------------|--------------------|--------------------|
| Review header       | `\x1b[1;31m`      | Bold red           |
| Review file:line    | `\x1b[31m`        | Red                |
| Skim header         | `\x1b[1;33m`      | Bold yellow        |
| Skim file:line      | `\x1b[33m`        | Yellow             |
| Skip header         | `\x1b[1;2m`       | Bold dim           |
| Skip category       | `\x1b[2m`         | Dim                |
| Rationale text      | (no code)          | Default            |
| Reset               | `\x1b[0m`         | Reset              |

Do not use any external crate for color output. Raw ANSI escape codes are sufficient and avoid a dependency. Gate all escape codes behind the resolved color flag (see CLI section 9).

### 8.3 Markdown renderer

Produces the sticky comment body. Used when `--post-comment` is set or `--format markdown` is explicit. Structure:

```markdown
<!-- garbelour:sticky -->
## Garbelour

**{review_count} of {total_count} hunks need review.** {skim_count} worth skimming. {skip_count} mechanical.

### Review ({review_count})

- [`{file}:{focus_or_hunk_line}`]({deep_link}) — {rationale}
- ...

### Skim ({skim_count})

<details>
<summary>Click to expand</summary>

- [`{file}:{focus_or_hunk_line}`]({deep_link}) — {rationale}
- ...

</details>

### Skip ({skip_count})

<details>
<summary>Click to expand</summary>

- **{category}** ({count}): {file_list}
- ...

</details>
```

Rules:
- **Review** items are always expanded, listed individually with deep links and rationales.
- **Skim** items are in a collapsed `<details>` block, listed individually with deep links and rationales.
- **Skip** items are in a collapsed `<details>` block, grouped by category (e.g. "Generated (15): `proto/*.pb.go`, `dist/bundle.js`, ..."). No individual deep links — these are noise and linking to them defeats the purpose.
- If there are zero items in a section, omit the section header entirely.
- File paths should be relative to the repo root.
- **Focus line rendering:** When `focus_lines` is present, the displayed link text uses the focus range: `` `src/engine.rs:145–148` `` and the deep link anchors to `focus_lines.start`. When `focus_lines` is `None`, fall back to the hunk start line. The rationale from the classifier or LLM already references the specific lines, so the link text and rationale reinforce each other.
- End with a footer: `*[garbelour](https://github.com/garbelour-io/garbelour-rs) · {provider}/{model}*` (only mention the model if the LLM was invoked).

### 8.4 JSON renderer

Produces structured JSON to stdout. Used for piped output, CI pipelines, downstream tooling, and programmatic consumption (e.g. Claude Code parsing the output). This is the default when stdout is not a TTY.

```json
{
  "base_sha": "abc123",
  "head_sha": "def456",
  "hunks": [
    {
      "hunk_id": "src/engine.rs:142",
      "file": "src/engine.rs",
      "line": 142,
      "level": "review",
      "category": "public_api_change",
      "rationale": "public fn signature changed at lines 145–148",
      "focus_lines": { "start": 145, "end": 148, "side": "new" },
      "source": { "heuristic": { "name": "public_api" } }
    },
    {
      "hunk_id": "src/store.rs:80",
      "file": "src/store.rs",
      "line": 80,
      "level": "review",
      "category": "error_handling_deleted",
      "rationale": "removed try/except block at old lines 88–95",
      "focus_lines": { "start": 88, "end": 95, "side": "old" },
      "source": { "heuristic": { "name": "error_handling" } }
    },
    {
      "hunk_id": "Cargo.lock:1",
      "file": "Cargo.lock",
      "line": 1,
      "level": "skip",
      "category": "lockfile",
      "rationale": "lockfile update",
      "focus_lines": null,
      "source": { "heuristic": { "name": "lockfile" } }
    }
  ],
  "summary": {
    "total": 47,
    "review": 3,
    "skim": 5,
    "skip": 39
  }
}
```

---

## 9. CLI

```
garbelour review [OPTIONS]
```

| Flag                 | Default               | Description                                              |
|----------------------|-----------------------|----------------------------------------------------------|
| `--base <sha>`       | (required in local)   | Base git ref                                             |
| `--head <sha>`       | `HEAD`                | Head git ref                                             |
| `--repo <path>`      | `.`                   | Path to git repository                                   |
| `--format`           | `auto`                | Output format: `human`, `markdown`, `json`, `auto`       |
| `--color`            | `auto`                | Color output: `always`, `never`, `auto`                  |
| `--post-comment`     | `false`               | Post/update sticky comment on GitHub PR                  |
| `--owner <owner>`    | from event payload    | GitHub repo owner                                        |
| `--repo-name <name>` | from event payload    | GitHub repo name                                         |
| `--pr <number>`      | from event payload    | PR number                                                |
| `--llm`              | `false`               | Enable LLM classification of unclassified hunks          |
| `--llm-provider`     | auto-detect from env  | LLM provider: `anthropic`, `openai`, `ollama`            |
| `--llm-model`        | provider default      | Model to use                                             |
| `--llm-base-url`     | provider default      | Override API base URL                                    |
| `--no-llm`           | `false`               | Explicitly disable LLM even if keys are set              |
| `--config <path>`    | `garbelour.toml`      | Path to config file                                      |
| `--size-threshold`   | `150`                 | Lines changed above which a hunk auto-classifies as Review |

### Format auto-detection

When `--format` is `auto` (the default), the format is resolved as follows:

1. If `--post-comment` is set → `markdown` (the sticky comment body).
2. If stdout is a TTY (interactive terminal) → `human`.
3. If stdout is not a TTY (piped or redirected) → `json`.

This means `garbelour review --base main` in a terminal shows the human-readable
report, `garbelour review --base main | jq .summary` gets structured JSON, and
the GitHub Action workflow sets `--post-comment` and gets markdown — all without
the user having to think about format flags.

Use `std::io::stdout().is_terminal()` (stable since Rust 1.70) for the TTY check.

### Color auto-detection

When `--color` is `auto` (the default):

1. If stdout is a TTY → color enabled.
2. If stdout is not a TTY → color disabled.
3. Respect `NO_COLOR` environment variable (see https://no-color.org): if set, disable color regardless of TTY status.

`--color always` forces color on (useful for `garbelour review --base main --color always | less -R`).
`--color never` forces color off.

When `--post-comment` is set, the tool reads GitHub context from either CLI flags or the `GITHUB_EVENT_PATH` environment variable (auto-detected in Actions). `GITHUB_TOKEN` is required for posting.

When `--post-comment` is not set, the rendered output goes to stdout.

---

## 10. Configuration file

`garbelour.toml` at the repository root. All fields optional.

```toml
[classify]
# Additional globs for generated files (merged with defaults)
generated_globs = ["generated/**", "*.auto.ts"]

# Additional lockfile names (merged with defaults)
lockfile_names = ["shrinkwrap.json"]

# Lines-changed threshold for auto-elevating to Review
size_threshold = 150

[llm]
# Provider override (otherwise auto-detected from env)
provider = "anthropic"
model = "claude-haiku-4-5"

[github]
# Override base URL for GitHub Enterprise
base_url = "https://github.example.com/api/v3"
```

---

## 11. Main dispatch

```rust
// src/main.rs

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let config = Config::load(&cli.config)?;

    // 1. Extract diff
    let mut diff = diff::extract(&cli.repo, &cli.base, &cli.head)?;

    // 2. Run heuristic pipeline
    let pipeline = Pipeline::standard(&config);
    let (mut classified, unclassified) = pipeline.run(&mut diff);

    // 3. Optional LLM pass
    if cli.llm && !cli.no_llm && !unclassified.is_empty() {
        let llm_config = llm::detect_provider(
            cli.llm_provider.as_deref(),
            cli.llm_model.as_deref(),
            cli.llm_base_url.as_deref(),
        )?;
        let llm_results = llm::classify_hunks(&unclassified, &llm_config)?;
        classified.extend(llm_results);
    }

    // 4. Any hunks still unclassified default to Skim
    for u in &unclassified {
        if !classified.iter().any(|c| c.hunk_id == u.hunk.id) {
            classified.push(Classified {
                hunk_id: u.hunk.id.clone(),
                file_path: u.file.path.clone(),
                new_range: u.hunk.new_range.clone(),
                classification: Classification {
                    level: Level::Skim,
                    category: Category::LlmAssessed,
                    rationale: "unclassified — defaulting to skim".into(),
                    source: Source::Heuristic { name: "default".into() },
                    focus_lines: None,
                },
            });
        }
    }

    // 5. Resolve output format
    let format = resolve_format(&cli);
    let use_color = resolve_color(&cli);

    // 6. Render and output
    match format {
        Format::Human => {
            let output = render::human(&diff, &classified, use_color)?;
            // Summary line to stderr (visible even when piped)
            eprintln!("{}", render::summary_line(&classified));
            println!("{output}");
        }
        Format::Markdown => {
            let body = render::markdown(&diff, &classified, &cli)?;
            if cli.post_comment {
                let gh = GitHubClient::from_env()?;
                let event = github::parse_event()?;
                github::upsert_sticky_comment(
                    &gh, &event.owner, &event.repo, event.pr_number, &body,
                )?;
                eprintln!(
                    "garbelour: posted review map to PR #{}",
                    event.pr_number
                );
            } else {
                println!("{body}");
            }
        }
        Format::Json => {
            let json = render::json(&diff, &classified)?;
            println!("{json}");
        }
    }

    Ok(())
}

/// Resolve the output format from CLI flags and environment.
fn resolve_format(cli: &Cli) -> Format {
    match cli.format {
        Some(f) => f,               // Explicit flag wins
        None if cli.post_comment => Format::Markdown,
        None if std::io::stdout().is_terminal() => Format::Human,
        None => Format::Json,
    }
}

/// Resolve whether to use color from CLI flags and environment.
fn resolve_color(cli: &Cli) -> bool {
    match cli.color {
        ColorChoice::Always => true,
        ColorChoice::Never => false,
        ColorChoice::Auto => {
            std::io::stdout().is_terminal()
                && std::env::var_os("NO_COLOR").is_none()
        }
    }
}
```

---

## 12. Build order

Implement in this order. Each step should be independently testable before moving to the next.

1. **Scaffolding:** `Cargo.toml`, `main.rs`, `lib.rs`, `cli.rs`, `error.rs`. Verify `cargo build` and `cargo test` pass with a no-op `main`.

2. **Diff extraction:** `diff.rs`. Shell out to `git diff`, parse unified diff into `Diff`/`FileDiff`/`Hunk`. Write tests against fixture diffs in `tests/fixtures/`.

3. **Language detection:** `lang.rs`. Pure function from `Path → Option<Language>`. Trivial to test.

4. **Classification core:** `classify.rs` with the `Classifier` trait, `Pipeline`, `Level`, `Category`, `Source`, `Classification`, `Classified`, `Unclassified` types.

5. **First classifier — Lockfile:** `classifiers/lockfile.rs`. Pure path matching, easiest to implement and test. Proves the trait and pipeline work end-to-end.

6. **Second classifier — Generated:** `classifiers/generated.rs`. Path matching + `.gitattributes` reading.

7. **Third classifier — SizeThreshold:** `classifiers/size_threshold.rs`. Pure line counting.

8. **AST classifiers:** `classifiers/comment_only.rs`, `classifiers/import_reorder.rs`, `classifiers/public_api.rs`, `classifiers/control_flow.rs`, `classifiers/error_handling.rs`. Implement one language at a time (Rust first, then Python, then TypeScript). Each needs tree-sitter parsing of fixture files.

9. **Rendering:** `render.rs`. Three renderers: human (terminal), markdown (GitHub comment), JSON (structured output). Implement `human` first since it's the most useful for local testing during development. Test human output against expected strings (strip ANSI codes for comparison). Test markdown for section headers, deep-link URLs, and collapsed sections. Test JSON against expected structure.

10. **LLM integration:** `llm.rs`. Provider enum, detection, API calls, batching, prompt construction, response parsing. Test with a local HTTP server that returns canned responses (use `--llm-base-url` pointed at `http://localhost:{port}`).

11. **GitHub integration:** `github.rs`. Event parsing, comment listing, upsert. Test with canned JSON fixtures for the event payload.

12. **Main dispatch:** Wire everything together in `main.rs`.

---

## 13. Testing strategy

- **Unit tests** for each classifier using small fixture diffs. Each test creates a `FileDiff` + `Hunk` in memory and asserts the classifier returns the expected `Classification` (including correct `focus_lines`) or `None`.

- **Integration tests** in `tests/integration/` that run the full pipeline against multi-file fixture diffs and assert the correct partition into Review/Skim/Skip.

- **LLM tests** using `--llm-base-url` pointed at a simple HTTP server (in-test or via a helper binary) that returns canned JSON responses. Asserts correct prompt construction and response parsing, including `focus_start`/`focus_end` extraction.

- **Render tests** for all three formats:
  - **Human:** assert the output contains the expected section headers, file:line-range columns, and rationale text. Strip ANSI codes before string comparison. Test with color enabled and disabled.
  - **Markdown:** assert the output contains `STICKY_MARKER`, correct deep-link URLs (using focus lines when present), collapsed `<details>` sections, and the grouped skip summary.
  - **JSON:** assert the output parses as valid JSON and matches the expected schema, including `focus_lines` objects and null values.

- **Format auto-detection tests** that verify `resolve_format` returns `Human` when stdout is a TTY, `Json` when piped, and `Markdown` when `--post-comment` is set. These may need to be integration tests that invoke the binary as a subprocess with different pipe configurations.

- **No network in CI by default.** All tests that would hit external services (GitHub API, LLM APIs) use local canned responses. Real API tests are behind a `#[ignore]` attribute and run manually.

---

## 14. Non-goals for v1

- **No inline PR review comments.** The tool posts a single sticky summary comment. Inline comments are a different UX and create noise.
- **No CI gating.** The tool does not fail the build based on classifications. It's informational only.
- **No caching.** Each run re-analyzes the full diff. Caching across pushes is a v2 concern.
- **No formatter invocation.** The `FormatterOnly` classifier is deferred to v2. In v1, comment-only and import-reorder cover the most common formatter-adjacent noise.
- **No GitHub App.** The tool runs as a GitHub Action only.
- **No support for languages beyond Rust, Python, TypeScript, JavaScript.** Additional languages are new tree-sitter grammar dependencies and new classifier implementations, added incrementally.
- **No sub-hunk classification.** The classification unit is the hunk, not individual lines. However, classifiers populate `focus_lines` to pinpoint the specific lines within a hunk that triggered the classification. This gives reviewers line-level guidance without the architectural complexity of splitting hunks into independently-classified line ranges. If real-world usage shows that mixed hunks (where different lines deserve different levels) are common, sub-hunk classification becomes a v2 candidate — the `focus_lines` data from v1 provides the signal to measure this.
