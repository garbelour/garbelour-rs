//! Detect hunks where every changed line falls inside a comment node.
//!
//! Tree-sitter calls these `comment`, `line_comment`, or `block_comment`
//! depending on the grammar. Python docstrings are *not* comment nodes —
//! they are `expression_statement > string` at the head of a module/class/
//! function body, and we treat those as comment-equivalent here.
//!
//! Blank lines (whitespace only) inside a hunk are tolerated: adding an
//! empty line between two doc-comment lines should still classify as
//! comment-only.
//!
//! Skipped for files with no detected language (returns `None` so the
//! pipeline tries the next classifier).

use tree_sitter::Tree;

use crate::ast::{collect_line_ranges, is_blank_line, lines_all_within, parse, walk};
use crate::classify::{Category, Classification, Classifier, Level, Source};
use crate::diff::{FileDiff, Hunk};
use crate::lang::Language;

pub struct CommentOnly;

impl CommentOnly {
    pub fn new() -> Self {
        CommentOnly
    }
}

impl Default for CommentOnly {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier for CommentOnly {
    fn name(&self) -> &'static str {
        "comment_only"
    }

    fn priority(&self) -> i32 {
        100
    }

    fn classify(&self, file: &mut FileDiff, hunk: &Hunk) -> Option<Classification> {
        let language = file.language?;

        // No work to do if neither side has changed lines (rare but possible
        // for context-only hunks).
        if hunk.added_lines.is_empty() && hunk.removed_lines.is_empty() {
            return None;
        }

        let _ = file.ensure_content();
        let old_content = file.old_content.as_deref()?;
        let new_content = file.new_content.as_deref()?;
        let old_tree = parse(language, old_content)?;
        let new_tree = parse(language, new_content)?;

        let old_ranges = comment_ranges(language, &old_tree);
        let new_ranges = comment_ranges(language, &new_tree);

        // Filter out blank lines from the changed-line check; they don't
        // need to be inside a comment.
        let removed_non_blank =
            non_blank_lines(&hunk.removed_lines, &hunk.old_lines, hunk.old_range.start);
        let added_non_blank =
            non_blank_lines(&hunk.added_lines, &hunk.new_lines, hunk.new_range.start);

        if !lines_all_within(&removed_non_blank, &old_ranges) {
            return None;
        }
        if !lines_all_within(&added_non_blank, &new_ranges) {
            return None;
        }

        Some(Classification {
            level: Level::Skip,
            category: Category::CommentOnly,
            rationale: "comment/docstring-only change".into(),
            source: Source::Heuristic {
                name: "comment_only".into(),
            },
            focus_lines: None,
        })
    }
}

fn comment_ranges(language: Language, tree: &Tree) -> Vec<(u32, u32)> {
    let mut ranges = collect_line_ranges(tree, |n| is_comment_node(n.kind()));
    if language == Language::Python {
        ranges.extend(python_docstring_ranges(tree));
    }
    ranges
}

fn is_comment_node(kind: &str) -> bool {
    matches!(
        kind,
        "comment" | "line_comment" | "block_comment" | "doc_comment"
    )
}

/// Filter `lines` to those whose corresponding source line is non-blank.
/// `lines` are 1-indexed file line numbers; `hunk_lines` is the slice of
/// hunk content (either old_lines or new_lines) that starts at `hunk_start`.
fn non_blank_lines(lines: &[u32], hunk_lines: &[String], hunk_start: u32) -> Vec<u32> {
    let mut out = Vec::new();
    for &lineno in lines {
        let idx = (lineno as i64 - hunk_start as i64) as usize;
        let blank = hunk_lines.get(idx).is_some_and(|s| is_blank_line(s));
        if !blank {
            out.push(lineno);
        }
    }
    out
}

/// Walk a Python AST and add the line ranges of any docstring — defined as
/// an `expression_statement` containing a single `string`, located as the
/// first named child of a `module`, `class_definition` body, or
/// `function_definition` body.
fn python_docstring_ranges(tree: &Tree) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    walk(tree.root_node(), &mut |node| {
        let body = match node.kind() {
            "module" => Some(*node),
            "class_definition" | "function_definition" => node.child_by_field_name("body"),
            _ => None,
        };
        if let Some(body) = body {
            if let Some(first) = body.named_child(0) {
                if first.kind() == "expression_statement" {
                    if let Some(inner) = first.named_child(0) {
                        if inner.kind() == "string" {
                            let (s, e) = crate::ast::node_lines(&first);
                            out.push((s, e));
                        }
                    }
                }
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::diff::{FileStatus, HunkId, LineRange};

    /// Build a FileDiff with pre/post content set, and a single hunk
    /// describing the change. All line numbers are 1-indexed.
    fn fixture(
        language: Language,
        old: &str,
        new: &str,
        added: Vec<u32>,
        removed: Vec<u32>,
    ) -> (FileDiff, Hunk) {
        let path = PathBuf::from(match language {
            Language::Rust => "src/x.rs",
            Language::Python => "src/x.py",
            Language::TypeScript => "src/x.ts",
            Language::JavaScript => "src/x.js",
        });
        let hunk = Hunk {
            id: HunkId("test:1".into()),
            old_range: LineRange {
                start: 1,
                count: old.lines().count() as u32,
            },
            new_range: LineRange {
                start: 1,
                count: new.lines().count() as u32,
            },
            old_lines: old.lines().map(|s| s.to_string()).collect(),
            new_lines: new.lines().map(|s| s.to_string()).collect(),
            added: added.len() as u32,
            removed: removed.len() as u32,
            added_lines: added,
            removed_lines: removed,
        };
        let mut file = FileDiff::for_test(
            path,
            FileStatus::Modified,
            Some(language),
            vec![hunk.clone()],
        );
        file.old_content = Some(old.to_string());
        file.new_content = Some(new.to_string());
        (file, hunk)
    }

    #[test]
    fn rust_line_comment_only() {
        let old = "// one\nfn a() {}\n";
        let new = "// two\nfn a() {}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        let result = CommentOnly::new().classify(&mut f, &h).unwrap();
        assert_eq!(result.category, Category::CommentOnly);
        assert_eq!(result.level, Level::Skip);
    }

    #[test]
    fn rust_doc_comment_only() {
        let old = "/// old doc\nfn a() {}\n";
        let new = "/// new doc\nfn a() {}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        assert!(CommentOnly::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn rust_block_comment_only() {
        let old = "/* old */\nfn a() {}\n";
        let new = "/* new */\nfn a() {}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        assert!(CommentOnly::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn rust_code_change_does_not_match() {
        let old = "fn a() { 1 }\n";
        let new = "fn a() { 2 }\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        assert!(CommentOnly::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn rust_mixed_change_does_not_match() {
        let old = "// note\nfn a() { 1 }\n";
        let new = "// updated\nfn a() { 2 }\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1, 2], vec![1, 2]);
        assert!(CommentOnly::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn rust_blank_lines_inside_comment_block_are_tolerated() {
        let old = "// one\n// two\nfn a() {}\n";
        let new = "// one\n\n// two\nfn a() {}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![]);
        // Line 2 in new is a blank line — it should be tolerated even though
        // it's not inside any comment node.
        assert!(CommentOnly::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn python_comment_only() {
        let old = "# old\ndef a():\n    return 1\n";
        let new = "# new\ndef a():\n    return 1\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![1], vec![1]);
        assert!(CommentOnly::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn python_module_docstring_only() {
        let old = "\"\"\"old docs\"\"\"\ndef a():\n    return 1\n";
        let new = "\"\"\"new docs\"\"\"\ndef a():\n    return 1\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![1], vec![1]);
        let result = CommentOnly::new().classify(&mut f, &h);
        assert!(
            result.is_some(),
            "module docstring change should classify as comment-only"
        );
    }

    #[test]
    fn python_function_docstring_only() {
        let old = "def a():\n    \"\"\"old\"\"\"\n    return 1\n";
        let new = "def a():\n    \"\"\"new\"\"\"\n    return 1\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![2], vec![2]);
        assert!(CommentOnly::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn typescript_comment_only() {
        let old = "// old\nexport const x = 1;\n";
        let new = "// new\nexport const x = 1;\n";
        let (mut f, h) = fixture(Language::TypeScript, old, new, vec![1], vec![1]);
        assert!(CommentOnly::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn javascript_block_comment_only() {
        let old = "/* old\n * note\n */\nconst x = 1;\n";
        let new = "/* new\n * note\n */\nconst x = 1;\n";
        let (mut f, h) = fixture(Language::JavaScript, old, new, vec![1], vec![1]);
        assert!(CommentOnly::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn skips_files_with_no_language() {
        let old = "anything\n";
        let new = "different\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        f.language = None;
        assert!(CommentOnly::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn skips_files_with_missing_content() {
        let old = "// one\n";
        let new = "// two\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        f.old_content = None;
        assert!(CommentOnly::new().classify(&mut f, &h).is_none());
    }
}
