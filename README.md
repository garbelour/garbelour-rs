# garbelour

Classify every hunk in a git diff (or a GitHub pull request) by how much reviewer
attention it deserves: **review**, **skim**, or **skip**.

## What it does

Garbelour walks the diff between two git refs, runs each hunk through a
pipeline of deterministic classifiers, and (optionally) sends whatever the
heuristics didn't classify to an LLM for enrichment. The output is a single
report in one of three formats:

- **human** (terminal): grouped by level, color-coded, with file:line
  pointers to the lines that matter.
- **markdown** (GitHub sticky comment): collapsible Skim/Skip sections
  and deep links to the exact lines of each Review item.
- **json** (machine-readable): one record per classified hunk plus a
  summary; suitable for piping into other tools.

The format defaults to `auto`: markdown when posting to GitHub, human in
a TTY, json otherwise.

## Built-in classifiers

| Classifier             | Level   | What it catches                                                          |
|------------------------|---------|--------------------------------------------------------------------------|
| `generated`            | Skip    | Files matching globs (`*.lock`, `dist/**`, `*.min.js`, …) or `.gitattributes` `linguist-generated` |
| `lockfile`             | Skip    | Cargo.lock, package-lock.json, yarn.lock, pnpm-lock.yaml, go.sum, …      |
| `comment_only`         | Skip    | Hunks where every changed line is inside a comment / docstring node      |
| `import_reorder`       | Skip    | Same set of imports, different order                                     |
| `public_api`           | Review  | `pub` items in Rust, `export` in TS/JS, module-level `def`/`class` in Python |
| `control_flow`         | Review  | Added / removed / modified `if`, `match`/`switch`, `for`, `while`, `loop`, `return` |
| `numerical_calc`       | Review  | Non-trivial arithmetic (≥ 2 operators) or calls into `Math.*` / `np.*` / `f64::*` / … |
| `error_handling`       | Review  | Removed `?`, `try`/`except`, `try`/`catch`, `except` clauses             |

AST-based classifiers use tree-sitter for Rust, Python, TypeScript, and
JavaScript. Files in unsupported languages still go through the path-based
classifiers (generated, lockfile).

Large hunks aren't auto-elevated. Size alone is a weak signal; instead, big
hunks that no other heuristic claims flow through `--llm` for a content-aware
verdict, or fall back to **review** when the LLM is off.

## Install

To install the CLI via Python pip:

```sh
pip install garbelour
```

To instsall the CLI via Rust cargo:

```sh
cargo install garbelour
```


## Usage

### Locally, against any two refs

```sh
garbelour review --base main
```

Pretty terminal output by default. Pipe to `jq` to get JSON instead:

```sh
garbelour review --base main | jq '.summary'
```

Force a specific format:

```sh
garbelour review --base main --format markdown
garbelour review --base main --format json
```

Force color on through a pager:

```sh
garbelour review --base main --color always | less -R
```

### In a GitHub Action

Garbelour reads the PR event payload from `GITHUB_EVENT_PATH` and posts a
single sticky comment, updating it on each push.

```yaml
# .github/workflows/garbelour.yml
on:
  pull_request:
    types: [opened, synchronize, reopened]
permissions:
  contents: read
  pull-requests: write
jobs:
  review-map:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: '3.14'
      - run: pip install garbelour
      - run: garbelour review --post-comment --llm
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

`fetch-depth: 0` matters: `git diff` needs both base and head refs in
the local repo.


### LLM triage (optional)

`--llm` sends hunks no heuristic claimed to an LLM, which assigns each a
level (review/skim/skip), a focus line range, and a one-line rationale.

The provider is auto-detected from whichever API-key env var is present:

| Env var               | Provider  | Default model        |
|-----------------------|-----------|----------------------|
| `ANTHROPIC_API_KEY`   | Anthropic | `claude-haiku-4-5`   |
| `OPENAI_API_KEY`      | OpenAI    | `gpt-4o-mini`        |
| `OLLAMA_API_KEY`      | Ollama    | `llama3.2`           |

Override with `--llm-provider`, `--llm-model`, or `--llm-base-url`.

Without `--llm`, hunks no heuristic claimed default to **review**: the
safe option, since the tool hasn't actually evaluated them.


## Configuration

`garbelour.toml` at the repository root is loaded automatically. All
fields are optional.

```toml
[classify]
generated_globs = ["generated/**", "*.auto.ts"]   # merged with defaults
lockfile_names = ["shrinkwrap.json"]              # merged with defaults

[llm]
provider = "anthropic"
model = "claude-haiku-4-5"

[github]
base_url = "https://github.example.com/api/v3"     # GitHub Enterprise
```

CLI flags override config; config overrides built-in defaults.

`.gitattributes` is also consulted: any path marked `linguist-generated`
(or `linguist-generated=true`) is treated as generated. Glob patterns in
`.gitattributes` are not interpreted; fall back to
`[classify].generated_globs` for those.


## Output examples

**Human:**

```
garbelour: 3 of 47 hunks need review, 5 worth skimming, 39 mechanical
  Review (3)
    src/engine.rs:145–148    public fn signature changed
    src/engine.rs:201–208    new branch in retry loop
    src/store.rs:88–95 (old) removed try/except block
  Skim (5)
    …
  Skip (39)
    generated (15)        proto/*.pb.go, dist/bundle.js, ...
    lockfile (1)          Cargo.lock
    comment-only (3)      src/engine.rs, src/store.rs, README.md
    import-reorder (8)    src/engine.rs, src/store.rs, ...
    test-fixture (12)     tests/fixtures/*.snap
```

**JSON:**

```json
{
  "base_sha": "...",
  "head_sha": "...",
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
    }
  ],
  "summary": { "total": 47, "review": 3, "skim": 5, "skip": 39 }
}
```

## Library use

Garbelour is dual-target: a `garbelour` binary and a `garbelour` library
crate. The pipeline, classifiers, and renderers are public API.

```rust
use garbelour::{diff, classify::{Pipeline, PipelineConfig}};

let mut d = diff::extract(std::path::Path::new("."), "main", "HEAD")?;
let pipeline = Pipeline::standard(&PipelineConfig::default())?;
let (classified, unclassified) = pipeline.run(&mut d);
```

## Design notes

- One subprocess to `git diff --raw -z -M -C` for file statuses (rename
  detection), one to `git diff -U3 -M -C` for hunks (parsed via the
  `patch` crate). File content for AST classifiers is loaded lazily via
  `git show`.
- Heuristics emit only `Skip` or `Review`. `Skim` requires positive
  evidence, currently only the LLM emits it.
- `focus_lines` pinpoints the specific line range that triggered each
  classification, so the markdown deep link lands on the right line, not
  just the start of the hunk.



## What is New in Garbelour

### 0.3.0

Improved classifier prompt.

Added new classifier for numerical calculations.

### 0.2.0

Extensions to the public interface.

### 0.1.0

Initial release.

