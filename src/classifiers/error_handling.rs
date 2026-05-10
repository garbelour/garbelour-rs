//! Detect *removal* of error-handling constructs.
//!
//! Per language:
//!   - **Rust:** `?` operator (`try_expression`). v1 does not detect
//!     removed `match` arms on `Err`, removed `.unwrap_or*`, or removed
//!     `if let Err(...)` — those require pattern-matching against tree
//!     structures and are deferred.
//!   - **Python:** `try_statement`, `except_clause`.
//!   - **TS/JS:** `try_statement`, `catch_clause`.
//!
//! Additions of error handling are benign and never fire this rule.
//! Focus lines use `Side::Old` since the construct exists only in the
//! pre-image.

use tree_sitter::{Node, Tree};

use crate::ast::{parse, walk};
use crate::classify::{Category, Classification, Classifier, FocusLines, Level, Side, Source};
use crate::diff::{FileDiff, Hunk};
use crate::lang::Language;

pub struct ErrorHandlingDeleted;

impl ErrorHandlingDeleted {
    pub fn new() -> Self {
        ErrorHandlingDeleted
    }
}

impl Default for ErrorHandlingDeleted {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier for ErrorHandlingDeleted {
    fn name(&self) -> &'static str {
        "error_handling"
    }

    fn priority(&self) -> i32 {
        140
    }

    fn classify(&self, file: &mut FileDiff, hunk: &Hunk) -> Option<Classification> {
        let language = file.language?;
        if hunk.removed_lines.is_empty() {
            return None;
        }
        let _ = file.ensure_content();
        let old_content = file.old_content.as_deref()?;
        let old_tree = parse(language, old_content)?;

        let (label, range) = find_removed_error_handler(language, &old_tree, &hunk.removed_lines)?;
        Some(Classification {
            level: Level::Review,
            category: Category::ErrorHandlingDeleted,
            rationale: format!("removed {} at old lines {}–{}", label, range.0, range.1),
            source: Source::Heuristic { name: "error_handling".into() },
            focus_lines: Some(FocusLines {
                start: range.0,
                end: range.1,
                side: Side::Old,
            }),
        })
    }
}

fn find_removed_error_handler(
    language: Language,
    old_tree: &Tree,
    removed_lines: &[u32],
) -> Option<(&'static str, (u32, u32))> {
    let mut found: Option<(&'static str, (u32, u32))> = None;
    walk(old_tree.root_node(), &mut |node| {
        if found.is_some() {
            return;
        }
        let label = match error_handler_label(language, node.kind()) {
            Some(l) => l,
            None => return,
        };
        let start_row = node.start_position().row as u32 + 1;
        let end_row = node.end_position().row as u32 + 1;
        // For multi-line constructs (try/except blocks): fire if any
        // removed line falls within the construct's range. For single-line
        // constructs (`?` operator, which is contained in a single line):
        // fire if the start line is removed.
        let touched = if start_row == end_row {
            removed_lines.contains(&start_row)
        } else {
            removed_lines.iter().any(|&l| l >= start_row && l <= end_row)
        };
        if touched && construct_is_actually_present_on_removed_line(node, removed_lines) {
            found = Some((label, (start_row, end_row)));
        }
    });
    found
}

/// Guard against false positives where the *containing* construct still
/// exists but a body line was deleted. For multi-line constructs we only
/// fire if the head line (first row of the construct) is among the removed
/// lines — that signals the construct itself was deleted, not just an
/// unrelated body line within it. Single-line constructs always pass.
fn construct_is_actually_present_on_removed_line(node: &Node, removed_lines: &[u32]) -> bool {
    let start_row = node.start_position().row as u32 + 1;
    let end_row = node.end_position().row as u32 + 1;
    if start_row == end_row {
        true
    } else {
        removed_lines.contains(&start_row) || removed_lines.contains(&end_row)
    }
}

fn error_handler_label(language: Language, kind: &str) -> Option<&'static str> {
    match language {
        Language::Rust => match kind {
            "try_expression" => Some("? operator"),
            _ => None,
        },
        Language::Python => match kind {
            "try_statement" => Some("try/except block"),
            "except_clause" => Some("except clause"),
            _ => None,
        },
        Language::TypeScript | Language::JavaScript => match kind {
            "try_statement" => Some("try/catch block"),
            "catch_clause" => Some("catch clause"),
            _ => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::diff::{FileStatus, HunkId, LineRange};

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
            old_range: LineRange { start: 1, count: old.lines().count() as u32 },
            new_range: LineRange { start: 1, count: new.lines().count() as u32 },
            old_lines: old.lines().map(|s| s.to_string()).collect(),
            new_lines: new.lines().map(|s| s.to_string()).collect(),
            added: added.len() as u32,
            removed: removed.len() as u32,
            added_lines: added,
            removed_lines: removed,
        };
        let mut file =
            FileDiff::for_test(path, FileStatus::Modified, Some(language), vec![hunk.clone()]);
        file.old_content = Some(old.to_string());
        file.new_content = Some(new.to_string());
        (file, hunk)
    }

    #[test]
    fn rust_removed_question_mark() {
        let old = "fn f() -> Result<u32, ()> {\n    let x = parse()?;\n    Ok(x)\n}\n";
        let new = "fn f() -> u32 {\n    let x = parse();\n    x\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1, 2, 3], vec![1, 2, 3]);
        let result = ErrorHandlingDeleted::new().classify(&mut f, &h).unwrap();
        assert_eq!(result.category, Category::ErrorHandlingDeleted);
        assert_eq!(result.level, Level::Review);
        let focus = result.focus_lines.unwrap();
        assert_eq!(focus.side, Side::Old);
    }

    #[test]
    fn rust_pure_addition_does_not_fire() {
        let old = "fn f() -> u32 { 1 }\n";
        let new = "fn f() -> Result<u32, ()> { Ok(parse()?) }\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        // The `?` is added; we should not fire on additions.
        assert!(ErrorHandlingDeleted::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn python_removed_try_block() {
        let old = "def f():\n    try:\n        do()\n    except Exception:\n        pass\n";
        let new = "def f():\n    do()\n";
        let (mut f, h) = fixture(
            Language::Python,
            old,
            new,
            vec![2],
            vec![2, 3, 4, 5],
        );
        let result = ErrorHandlingDeleted::new().classify(&mut f, &h).unwrap();
        assert_eq!(result.category, Category::ErrorHandlingDeleted);
    }

    #[test]
    fn python_removed_except_clause() {
        let old = "def f():\n    try:\n        do()\n    except KeyError:\n        pass\n    except ValueError:\n        pass\n";
        let new = "def f():\n    try:\n        do()\n    except KeyError:\n        pass\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![], vec![6, 7]);
        // The `except ValueError` clause head is on the removed line.
        assert!(ErrorHandlingDeleted::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn typescript_removed_try_catch() {
        let old = "function f() {\n    try {\n        doIt();\n    } catch (e) {\n        log(e);\n    }\n}\n";
        let new = "function f() {\n    doIt();\n}\n";
        let (mut f, h) = fixture(
            Language::TypeScript,
            old,
            new,
            vec![2],
            vec![2, 3, 4, 5, 6],
        );
        assert!(ErrorHandlingDeleted::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn javascript_removed_catch_clause() {
        let old = "function f() {\n    try {\n        doIt();\n    } catch (e) {\n        log(e);\n    }\n}\n";
        let new = "function f() {\n    try {\n        doIt();\n    } finally {\n        cleanup();\n    }\n}\n";
        let (mut f, h) = fixture(
            Language::JavaScript,
            old,
            new,
            vec![4, 5],
            vec![4, 5],
        );
        assert!(ErrorHandlingDeleted::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn no_removed_lines_does_not_fire() {
        let old = "fn f() { foo(); }\n";
        let new = "fn f() { foo(); bar(); }\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![]);
        assert!(ErrorHandlingDeleted::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn removed_line_inside_try_body_only_does_not_fire() {
        let old = "def f():\n    try:\n        do_a()\n        do_b()\n    except Exception:\n        pass\n";
        let new = "def f():\n    try:\n        do_a()\n    except Exception:\n        pass\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![], vec![4]);
        // Removing a line inside the try body shouldn't trigger removal of
        // the construct itself.
        assert!(ErrorHandlingDeleted::new().classify(&mut f, &h).is_none());
    }
}
