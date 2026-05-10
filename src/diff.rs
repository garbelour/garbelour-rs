//! Diff extraction and parsing.
//!
//! We invoke `git` as a subprocess for two reasons: (a) the GitHub Action
//! environment already has git on `PATH`, so we avoid pulling in `gix` or
//! `libgit2`; (b) the user's local repo state is the authoritative source.
//!
//! Two passes:
//!   1. `git diff --raw -z -M -C base..head` → file statuses (A/M/D/R/C with
//!      similarity scores). The `--raw` form is the only reliable place to get
//!      rename detection in machine-readable form; the unified-diff body
//!      strips that information.
//!   2. `git diff --no-color -U3 -M -C base..head` → hunks, parsed in
//!      `parse_unified_diff`.
//!
//! We then join the two by file path.
//!
//! ## Why hand-rolled diff parsing
//!
//! The `patch` crate (0.7.0, last touched 2022) panics on `\ No newline at
//! end of file` markers when they appear between `-` and `+` lines, which
//! is what git emits whenever the pre-image had no trailing newline. The
//! parser hits an `assert!()` and aborts the process. Our needs are minimal
//! — just hunk headers (`@@ -a,b +c,d @@`) and per-line `+`/`-`/` `
//! prefixes — so a small purpose-built parser is more robust than depending
//! on an unmaintained one.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{anyhow, bail, Context};

use crate::lang::Language;

/// A complete diff between two git refs.
#[derive(Debug)]
pub struct Diff {
    pub base_sha: String,
    pub head_sha: String,
    pub files: Vec<FileDiff>,
}

/// A single file's changes.
#[derive(Debug)]
pub struct FileDiff {
    pub path: PathBuf,
    pub old_path: Option<PathBuf>,
    pub status: FileStatus,
    pub language: Option<Language>,
    pub hunks: Vec<Hunk>,
    pub old_content: Option<String>,
    pub new_content: Option<String>,
    /// Where this file lives on disk. Stored on the FileDiff so classifiers
    /// can call `ensure_content()` without the pipeline needing to thread a
    /// `repo_path` argument through the `Classifier` trait. Empty for
    /// `FileDiff::for_test`, which makes `ensure_content` a no-op.
    repo_path: PathBuf,
    base_sha: String,
    head_sha: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileStatus {
    Added,
    Modified,
    Deleted,
    Renamed { similarity: u8 },
}

impl FileStatus {
    pub fn is_added(&self) -> bool {
        matches!(self, FileStatus::Added)
    }
    pub fn is_deleted(&self) -> bool {
        matches!(self, FileStatus::Deleted)
    }
}

/// A single contiguous change within a file.
#[derive(Clone, Debug)]
pub struct Hunk {
    pub id: HunkId,
    pub old_range: LineRange,
    pub new_range: LineRange,
    /// Pre-image lines: removed-prefix lines plus context lines, in file order.
    pub old_lines: Vec<String>,
    /// Post-image lines: added-prefix lines plus context lines, in file order.
    pub new_lines: Vec<String>,
    /// Count of `+` lines (additions only, excluding context).
    pub added: u32,
    /// Count of `-` lines (deletions only, excluding context).
    pub removed: u32,
    /// Post-image line numbers (1-indexed) of the `+` lines.
    pub added_lines: Vec<u32>,
    /// Pre-image line numbers (1-indexed) of the `-` lines.
    pub removed_lines: Vec<u32>,
}

/// Stable identifier: `{file_path}:{new_range.start}`.
#[derive(Clone, Eq, PartialEq, Hash, Debug)]
pub struct HunkId(pub String);

impl HunkId {
    pub fn for_hunk(path: &Path, new_start: u32) -> Self {
        HunkId(format!("{}:{}", path.display(), new_start))
    }
}

#[derive(Clone, Debug)]
pub struct LineRange {
    pub start: u32,
    pub count: u32,
}

impl FileDiff {
    /// Build a FileDiff that isn't backed by a real repository. Used by
    /// classifier tests that construct synthetic hunks directly. Calling
    /// `ensure_content()` on the result is a no-op (the SHAs are empty).
    pub fn for_test(
        path: PathBuf,
        status: FileStatus,
        language: Option<Language>,
        hunks: Vec<Hunk>,
    ) -> Self {
        FileDiff {
            path,
            old_path: None,
            status,
            language,
            hunks,
            old_content: None,
            new_content: None,
            repo_path: PathBuf::new(),
            base_sha: String::new(),
            head_sha: String::new(),
        }
    }

    /// Lazily populate `old_content` and `new_content` by running `git show`.
    /// Idempotent. Skips the side that doesn't exist (added → no old, deleted
    /// → no new). No-op for FileDiffs built via `for_test` (empty SHAs).
    /// Errors are non-fatal: classifiers that need content can check whether
    /// it loaded.
    pub fn ensure_content(&mut self) -> anyhow::Result<()> {
        if self.base_sha.is_empty() {
            return Ok(());
        }
        if !self.status.is_added() && self.old_content.is_none() {
            let p = self.old_path.as_deref().unwrap_or(&self.path);
            self.old_content = git_show(&self.repo_path, &self.base_sha, p).ok();
        }
        if !self.status.is_deleted() && self.new_content.is_none() {
            self.new_content = git_show(&self.repo_path, &self.head_sha, &self.path).ok();
        }
        Ok(())
    }
}

/// Extract a diff between two git refs in the repository at `repo_path`.
pub fn extract(repo_path: &Path, base: &str, head: &str) -> anyhow::Result<Diff> {
    let base_sha =
        rev_parse(repo_path, base).with_context(|| format!("resolving base ref `{}`", base))?;
    let head_sha =
        rev_parse(repo_path, head).with_context(|| format!("resolving head ref `{}`", head))?;

    let raw_entries = run_diff_raw(repo_path, &base_sha, &head_sha)?;
    let unified = run_diff_unified(repo_path, &base_sha, &head_sha)?;

    let mut hunks_by_key: HashMap<PathBuf, Vec<Hunk>> = HashMap::new();
    if !unified.trim().is_empty() {
        for parsed in parse_unified_diff(&unified) {
            hunks_by_key.insert(parsed.lookup_key(), parsed.hunks);
        }
    }

    let mut files = Vec::with_capacity(raw_entries.len());
    for entry in raw_entries {
        let language = Language::detect(&entry.path);
        let hunks = hunks_by_key.remove(&entry.path).unwrap_or_default();
        files.push(FileDiff {
            language,
            hunks,
            old_path: entry.old_path,
            status: entry.status,
            old_content: None,
            new_content: None,
            repo_path: repo_path.to_path_buf(),
            base_sha: base_sha.clone(),
            head_sha: head_sha.clone(),
            path: entry.path,
        });
    }

    Ok(Diff {
        base_sha,
        head_sha,
        files,
    })
}

// --- raw-output parsing --------------------------------------------------

#[derive(Debug)]
struct RawEntry {
    status: FileStatus,
    /// The current file path (= new path for non-deleted files, old path for
    /// deleted files). For non-deleted files, this is also the key we use to
    /// look up hunks from the unified-diff parse.
    path: PathBuf,
    /// Only set for renames and copies: the previous path.
    old_path: Option<PathBuf>,
}

fn run_diff_raw(repo: &Path, base: &str, head: &str) -> anyhow::Result<Vec<RawEntry>> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["diff", "--no-color", "--raw", "-M", "-C", "-z"])
        .arg(format!("{}..{}", base, head))
        .output()
        .context("invoking `git diff --raw`")?;
    if !output.status.success() {
        bail!(
            "git diff --raw failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout =
        String::from_utf8(output.stdout).context("git diff --raw produced non-utf8 output")?;
    parse_raw_entries(&stdout)
}

fn parse_raw_entries(s: &str) -> anyhow::Result<Vec<RawEntry>> {
    // With -z, the format per record is:
    //   :<srcmode> <dstmode> <srcsha> <dstsha> <status>\0<path1>\0[<path2>\0]
    // Header fields are space-separated; paths follow as separate NUL-terminated
    // tokens. The number of paths is 1, except for R (rename) and C (copy)
    // statuses, which have 2.
    let mut entries = Vec::new();
    let mut tokens = s.split('\0').filter(|t| !t.is_empty()).peekable();

    while let Some(header) = tokens.next() {
        if !header.starts_with(':') {
            bail!("unexpected `git diff --raw` token: {:?}", header);
        }
        let parts: Vec<&str> = header.split_whitespace().collect();
        let status_code = parts
            .last()
            .copied()
            .ok_or_else(|| anyhow!("malformed --raw header: {:?}", header))?;
        let (status, paths_to_consume) = parse_status_code(status_code)?;
        let path1 = tokens
            .next()
            .ok_or_else(|| anyhow!("--raw entry missing path"))?;
        let entry = if paths_to_consume == 2 {
            let path2 = tokens
                .next()
                .ok_or_else(|| anyhow!("--raw rename entry missing new path"))?;
            RawEntry {
                status,
                path: PathBuf::from(path2),
                old_path: Some(PathBuf::from(path1)),
            }
        } else {
            RawEntry {
                status,
                path: PathBuf::from(path1),
                old_path: None,
            }
        };
        entries.push(entry);
    }
    Ok(entries)
}

fn parse_status_code(code: &str) -> anyhow::Result<(FileStatus, usize)> {
    let mut chars = code.chars();
    let first = chars
        .next()
        .ok_or_else(|| anyhow!("empty --raw status code"))?;
    let similarity_str: String = chars.collect();
    let similarity: u8 = if similarity_str.is_empty() {
        0
    } else {
        similarity_str.parse().unwrap_or(0)
    };
    match first {
        'A' => Ok((FileStatus::Added, 1)),
        'M' => Ok((FileStatus::Modified, 1)),
        'D' => Ok((FileStatus::Deleted, 1)),
        // Type-changes (regular file ↔ symlink, etc.) — treat as modification.
        'T' => Ok((FileStatus::Modified, 1)),
        // Treat copy as rename for v1; both signal a moved-or-derived file
        // and the classifiers don't distinguish them.
        'R' | 'C' => Ok((FileStatus::Renamed { similarity }, 2)),
        _ => bail!("unknown --raw status code: {:?}", code),
    }
}

// --- unified-diff parsing ------------------------------------------------

fn run_diff_unified(repo: &Path, base: &str, head: &str) -> anyhow::Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["diff", "--no-color", "-U3", "-M", "-C"])
        .arg(format!("{}..{}", base, head))
        .output()
        .context("invoking `git diff`")?;
    if !output.status.success() {
        bail!(
            "git diff failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    String::from_utf8(output.stdout).context("git diff produced non-utf8 output")
}

/// Output of parsing one file-section of a unified diff.
struct ParsedFile {
    /// Path as it appears after `--- ` in the diff (with `a/` prefix
    /// stripped, or `/dev/null` if the file is being added).
    old_path: String,
    /// Path as it appears after `+++ ` in the diff (with `b/` prefix
    /// stripped, or `/dev/null` if the file is being deleted).
    new_path: String,
    hunks: Vec<Hunk>,
}

impl ParsedFile {
    /// The path under which we look up this file's hunks when joining
    /// against the `--raw` entries: the new path for added/modified/
    /// renamed files, the old path for deleted files.
    fn lookup_key(&self) -> PathBuf {
        let raw = if self.new_path == "/dev/null" {
            self.old_path.as_str()
        } else {
            self.new_path.as_str()
        };
        PathBuf::from(raw)
    }
}

/// Parse `git diff -U... -M -C` output into per-file hunks.
///
/// The parser is deliberately forgiving: any line it doesn't recognize
/// inside a hunk body is treated as junk and skipped. Sections that don't
/// produce any hunks (binary diffs, mode-only changes, pure renames) are
/// emitted with an empty `hunks` vec — they still need a `ParsedFile` so
/// the caller can join against `--raw` entries by path.
fn parse_unified_diff(s: &str) -> Vec<ParsedFile> {
    let mut out = Vec::new();
    for section in split_into_file_sections(s) {
        if let Some(p) = parse_file_section(section) {
            out.push(p);
        }
    }
    out
}

/// Split the diff text into per-file sections, each starting with
/// `diff --git `.
fn split_into_file_sections(s: &str) -> Vec<&str> {
    let needle = "diff --git ";
    let mut starts = Vec::new();
    if s.starts_with(needle) {
        starts.push(0);
    }
    for (idx, _) in s.match_indices("\ndiff --git ") {
        starts.push(idx + 1);
    }
    let mut sections = Vec::with_capacity(starts.len());
    for (i, &start) in starts.iter().enumerate() {
        let end = starts.get(i + 1).copied().unwrap_or(s.len());
        sections.push(&s[start..end]);
    }
    sections
}

fn parse_file_section(section: &str) -> Option<ParsedFile> {
    let mut lines = section.lines().peekable();
    let mut old_path = String::new();
    let mut new_path = String::new();
    let mut first_hunk_header: Option<String> = None;

    // Header lines: skip everything until we see `--- ` / `+++ ` / `@@ `.
    while let Some(&line) = lines.peek() {
        if let Some(rest) = line.strip_prefix("--- ") {
            old_path = first_field(rest)
                .map(strip_git_prefix)
                .unwrap_or("")
                .to_string();
            lines.next();
        } else if let Some(rest) = line.strip_prefix("+++ ") {
            new_path = first_field(rest)
                .map(strip_git_prefix)
                .unwrap_or("")
                .to_string();
            lines.next();
        } else if line.starts_with("@@ ") {
            first_hunk_header = Some(line.to_string());
            lines.next();
            break;
        } else {
            lines.next();
        }
    }

    let mut hunks = Vec::new();
    let key_for_id = if new_path == "/dev/null" {
        old_path.as_str()
    } else {
        new_path.as_str()
    };

    let mut next_header = first_hunk_header;
    while let Some(header) = next_header.take() {
        let Some((old_start, _old_count, new_start, _new_count)) = parse_hunk_header(&header)
        else {
            // Unrecognizable header — skip the rest of this section.
            break;
        };
        let mut old_lines = Vec::new();
        let mut new_lines = Vec::new();
        let mut added_lines: Vec<u32> = Vec::new();
        let mut removed_lines: Vec<u32> = Vec::new();
        let mut old_lineno = old_start;
        let mut new_lineno = new_start;
        let mut added: u32 = 0;
        let mut removed: u32 = 0;

        while let Some(&line) = lines.peek() {
            if line.starts_with("@@ ") {
                next_header = Some(line.to_string());
                lines.next();
                break;
            }
            // Defensive: should never occur because sections are split on
            // `diff --git`, but guard anyway.
            if line.starts_with("diff --git ") {
                break;
            }
            lines.next();
            if line.starts_with('\\') {
                // `\ No newline at end of file` — informational, ignore.
                continue;
            } else if let Some(rest) = line.strip_prefix('+') {
                new_lines.push(rest.to_string());
                added_lines.push(new_lineno);
                new_lineno += 1;
                added += 1;
            } else if let Some(rest) = line.strip_prefix('-') {
                old_lines.push(rest.to_string());
                removed_lines.push(old_lineno);
                old_lineno += 1;
                removed += 1;
            } else if let Some(rest) = line.strip_prefix(' ') {
                old_lines.push(rest.to_string());
                new_lines.push(rest.to_string());
                old_lineno += 1;
                new_lineno += 1;
            } else if line.is_empty() {
                // Some legacy diff producers emit a bare blank line where
                // `<space>` would be expected. Treat it as empty context.
                old_lines.push(String::new());
                new_lines.push(String::new());
                old_lineno += 1;
                new_lineno += 1;
            }
            // else: junk inside a hunk body (a stray binary marker, an
            // intermediate `index` line). Skip silently.
        }

        let path = Path::new(key_for_id);
        hunks.push(Hunk {
            id: HunkId::for_hunk(path, new_start),
            old_range: LineRange {
                start: old_start,
                count: count_for(old_start, old_lineno),
            },
            new_range: LineRange {
                start: new_start,
                count: count_for(new_start, new_lineno),
            },
            old_lines,
            new_lines,
            added,
            removed,
            added_lines,
            removed_lines,
        });
    }

    Some(ParsedFile {
        old_path,
        new_path,
        hunks,
    })
}

/// Inclusive line count: lines `start..lineno` covers `lineno - start`
/// rows. Empty hunks still report start; deleted-only hunks may have
/// `old_start == 0`, in which case the count is also 0.
fn count_for(start: u32, end_exclusive: u32) -> u32 {
    end_exclusive.saturating_sub(start)
}

/// Parse `@@ -a,b +c,d @@ optional context`. `b` and `d` default to 1
/// when omitted (e.g. `@@ -5 +5 @@` is one-line on each side).
fn parse_hunk_header(s: &str) -> Option<(u32, u32, u32, u32)> {
    let body = s.strip_prefix("@@ ")?;
    let close = body.find(" @@")?;
    let header = &body[..close];
    let mut fields = header.split_whitespace();
    let old = fields.next()?.strip_prefix('-')?;
    let new = fields.next()?.strip_prefix('+')?;
    let parse_range = |s: &str| -> Option<(u32, u32)> {
        if let Some((a, b)) = s.split_once(',') {
            Some((a.parse().ok()?, b.parse().ok()?))
        } else {
            Some((s.parse().ok()?, 1))
        }
    };
    let (a, b) = parse_range(old)?;
    let (c, d) = parse_range(new)?;
    Some((a, b, c, d))
}

/// First whitespace-separated token, or None if the input is empty.
fn first_field(s: &str) -> Option<&str> {
    s.split_whitespace().next()
}

fn strip_git_prefix(s: &str) -> &str {
    if let Some(rest) = s.strip_prefix("a/") {
        return rest;
    }
    if let Some(rest) = s.strip_prefix("b/") {
        return rest;
    }
    s
}

// --- git plumbing --------------------------------------------------------

fn rev_parse(repo: &Path, rev: &str) -> anyhow::Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["rev-parse", "--verify"])
        .arg(format!("{}^{{commit}}", rev))
        .output()
        .context("invoking `git rev-parse`")?;
    if !output.status.success() {
        bail!(
            "git rev-parse {}: {}",
            rev,
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn git_show(repo: &Path, rev: &str, path: &Path) -> anyhow::Result<String> {
    let arg = format!("{}:{}", rev, path.display());
    let output = Command::new("git")
        .arg("-C")
        .arg(repo)
        .arg("show")
        .arg(arg)
        .output()
        .context("invoking `git show`")?;
    if !output.status.success() {
        bail!(
            "git show {}:{}: {}",
            rev,
            path.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    String::from_utf8(output.stdout).context("git show produced non-utf8 output")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_simple_modified_raw_entry() {
        let raw = ":100644 100644 abc def M\0src/foo.rs\0";
        let entries = parse_raw_entries(raw).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].status, FileStatus::Modified);
        assert_eq!(entries[0].path, PathBuf::from("src/foo.rs"));
        assert!(entries[0].old_path.is_none());
    }

    #[test]
    fn parses_added_and_deleted_raw_entries() {
        let raw = ":100644 100644 abc def A\0src/new.rs\0\
                   :100644 000000 abc 000 D\0src/old.rs\0";
        let entries = parse_raw_entries(raw).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].status, FileStatus::Added);
        assert_eq!(entries[0].path, PathBuf::from("src/new.rs"));
        assert_eq!(entries[1].status, FileStatus::Deleted);
        assert_eq!(entries[1].path, PathBuf::from("src/old.rs"));
    }

    #[test]
    fn parses_rename_raw_entry_with_similarity() {
        let raw = ":100644 100644 abc def R092\0src/old.rs\0src/new.rs\0";
        let entries = parse_raw_entries(raw).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].status, FileStatus::Renamed { similarity: 92 });
        assert_eq!(entries[0].path, PathBuf::from("src/new.rs"));
        assert_eq!(entries[0].old_path, Some(PathBuf::from("src/old.rs")));
    }

    #[test]
    fn empty_raw_input_yields_empty_entries() {
        assert!(parse_raw_entries("").unwrap().is_empty());
    }

    #[test]
    fn strips_a_b_prefixes() {
        assert_eq!(strip_git_prefix("a/foo.rs"), "foo.rs");
        assert_eq!(strip_git_prefix("b/foo.rs"), "foo.rs");
        assert_eq!(strip_git_prefix("foo.rs"), "foo.rs");
        assert_eq!(strip_git_prefix("/dev/null"), "/dev/null");
    }

    #[test]
    fn splits_multi_file_diff_into_sections() {
        let s = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a\n+b\n\
                 diff --git a/y b/y\n--- a/y\n+++ b/y\n@@ -1 +1 @@\n-c\n+d\n";
        let sections = split_into_file_sections(s);
        assert_eq!(sections.len(), 2);
        assert!(sections[0].starts_with("diff --git a/x"));
        assert!(sections[1].starts_with("diff --git a/y"));
    }

    #[test]
    fn parses_simple_modify_section() {
        let s = "diff --git a/foo.rs b/foo.rs\n--- a/foo.rs\n+++ b/foo.rs\n\
                 @@ -1,3 +1,3 @@\n ctx\n-old\n+new\n more\n";
        let parsed = parse_unified_diff(s);
        assert_eq!(parsed.len(), 1);
        let p = &parsed[0];
        assert_eq!(p.new_path, "foo.rs");
        assert_eq!(p.hunks.len(), 1);
        let h = &p.hunks[0];
        assert_eq!(h.old_lines, vec!["ctx", "old", "more"]);
        assert_eq!(h.new_lines, vec!["ctx", "new", "more"]);
        assert_eq!(h.added, 1);
        assert_eq!(h.removed, 1);
        // ctx is line 1 (both), -old is old-line 2, +new is new-line 2,
        // more is line 3 (both, context).
        assert_eq!(h.added_lines, vec![2]);
        assert_eq!(h.removed_lines, vec![2]);
        assert_eq!(h.old_range.start, 1);
        assert_eq!(h.new_range.start, 1);
    }

    #[test]
    fn parser_tolerates_no_newline_marker_between_remove_and_add() {
        // This is the exact pattern that crashed the old `patch` crate:
        // a `\ No newline at end of file` marker between the `-` line and
        // the `+` lines.
        let s = "diff --git a/README.md b/README.md\n\
                 --- a/README.md\n+++ b/README.md\n\
                 @@ -1 +1,3 @@\n-old text\n\\ No newline at end of file\n+new text\n+second\n+third\n";
        let parsed = parse_unified_diff(s);
        assert_eq!(parsed.len(), 1);
        let h = &parsed[0].hunks[0];
        assert_eq!(h.old_lines, vec!["old text"]);
        assert_eq!(h.new_lines, vec!["new text", "second", "third"]);
        assert_eq!(h.added, 3);
        assert_eq!(h.removed, 1);
    }

    #[test]
    fn parser_handles_added_file_with_dev_null_old_side() {
        let s = "diff --git a/new.rs b/new.rs\n--- /dev/null\n+++ b/new.rs\n\
                 @@ -0,0 +1,2 @@\n+line one\n+line two\n";
        let parsed = parse_unified_diff(s);
        assert_eq!(parsed.len(), 1);
        let p = &parsed[0];
        assert_eq!(p.old_path, "/dev/null");
        assert_eq!(p.new_path, "new.rs");
        assert_eq!(p.lookup_key(), PathBuf::from("new.rs"));
        let h = &p.hunks[0];
        assert_eq!(h.added, 2);
        assert_eq!(h.removed, 0);
        assert_eq!(h.added_lines, vec![1, 2]);
    }

    #[test]
    fn parser_handles_deleted_file_with_dev_null_new_side() {
        let s = "diff --git a/gone.rs b/gone.rs\n--- a/gone.rs\n+++ /dev/null\n\
                 @@ -1,2 +0,0 @@\n-line one\n-line two\n";
        let parsed = parse_unified_diff(s);
        assert_eq!(parsed.len(), 1);
        let p = &parsed[0];
        assert_eq!(p.lookup_key(), PathBuf::from("gone.rs"));
        let h = &p.hunks[0];
        assert_eq!(h.removed, 2);
        assert_eq!(h.added, 0);
    }

    #[test]
    fn parser_handles_multiple_hunks_in_one_file() {
        let s = "diff --git a/x b/x\n--- a/x\n+++ b/x\n\
                 @@ -1,2 +1,2 @@\n-a\n+A\n b\n\
                 @@ -10,2 +10,2 @@\n c\n-d\n+D\n";
        let parsed = parse_unified_diff(s);
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].hunks.len(), 2);
        assert_eq!(parsed[0].hunks[0].new_range.start, 1);
        assert_eq!(parsed[0].hunks[1].new_range.start, 10);
    }

    #[test]
    fn parser_handles_single_line_hunk_header_without_count() {
        // `@@ -5 +5 @@` is shorthand for `@@ -5,1 +5,1 @@`.
        let s = "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -5 +5 @@\n-old\n+new\n";
        let parsed = parse_unified_diff(s);
        let h = &parsed[0].hunks[0];
        assert_eq!(h.old_range.start, 5);
        assert_eq!(h.new_range.start, 5);
        assert_eq!(h.added_lines, vec![5]);
        assert_eq!(h.removed_lines, vec![5]);
    }

    #[test]
    fn parser_skips_binary_file_section_gracefully() {
        let s = "diff --git a/img.png b/img.png\n\
                 index 1234..5678\n\
                 Binary files a/img.png and b/img.png differ\n";
        let parsed = parse_unified_diff(s);
        // No `@@` header → no hunks, but we still emit a ParsedFile so the
        // caller can join against --raw entries (paths still empty since
        // `--- ` / `+++ ` are absent for binaries).
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].hunks.is_empty());
    }

    #[test]
    fn parses_pure_rename_with_no_content_changes() {
        // `git diff -M` emits no body for an exact rename.
        let s = "diff --git a/old b/new\n\
                 similarity index 100%\nrename from old\nrename to new\n";
        let parsed = parse_unified_diff(s);
        assert_eq!(parsed.len(), 1);
        assert!(parsed[0].hunks.is_empty());
    }

    #[test]
    fn hunk_id_format() {
        assert_eq!(
            HunkId::for_hunk(Path::new("src/foo.rs"), 42).0,
            "src/foo.rs:42"
        );
    }

    #[test]
    fn parses_hunk_header_variants() {
        assert_eq!(parse_hunk_header("@@ -1,3 +1,3 @@"), Some((1, 3, 1, 3)));
        assert_eq!(parse_hunk_header("@@ -5 +5 @@"), Some((5, 1, 5, 1)));
        assert_eq!(
            parse_hunk_header("@@ -10,2 +12,4 @@ fn foo()"),
            Some((10, 2, 12, 4))
        );
        assert_eq!(parse_hunk_header("garbage"), None);
    }
}
