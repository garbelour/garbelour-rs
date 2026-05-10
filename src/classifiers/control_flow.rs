//! Detect added, removed, or modified control-flow constructs.
//!
//! "Control-flow node" means an `if` / `match` (or `switch`) / `for` /
//! `while` / `loop` / `return` (and language-equivalents). The classifier
//! fires when a control-flow node's *start line* is in the hunk's added or
//! removed line set — i.e. when the user is introducing, removing, or
//! editing the head of a branch, not merely changing its body. That keeps
//! the rule focused on structural changes rather than every line inside a
//! function that contains an `if`.
//!
//! Try/catch and try/except are *not* covered here — they are the domain of
//! `ErrorHandlingDeleted`.

use std::collections::HashSet;

use tree_sitter::Tree;

use crate::ast::{parse, walk};
use crate::classify::{Category, Classification, Classifier, FocusLines, Level, Side, Source};
use crate::diff::{FileDiff, Hunk};
use crate::lang::Language;

pub struct ControlFlow;

impl ControlFlow {
    pub fn new() -> Self {
        ControlFlow
    }
}

impl Default for ControlFlow {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier for ControlFlow {
    fn name(&self) -> &'static str {
        "control_flow"
    }

    fn priority(&self) -> i32 {
        130
    }

    fn classify(&self, file: &mut FileDiff, hunk: &Hunk) -> Option<Classification> {
        let language = file.language?;
        if hunk.added_lines.is_empty() && hunk.removed_lines.is_empty() {
            return None;
        }

        let _ = file.ensure_content();

        // Added side first: if a new control-flow node starts on an added
        // line, that's the most informative case (focus_lines points at the
        // new branch).
        if !hunk.added_lines.is_empty() {
            if let Some(content) = file.new_content.as_deref() {
                if let Some(tree) = parse(language, content) {
                    let added: HashSet<u32> = hunk.added_lines.iter().copied().collect();
                    if let Some((kind, range)) = find_control_flow_node(language, &tree, &added) {
                        return Some(Classification {
                            level: Level::Review,
                            category: Category::ControlFlow,
                            rationale: format!("{} at lines {}–{}", kind, range.0, range.1),
                            source: Source::Heuristic {
                                name: "control_flow".into(),
                            },
                            focus_lines: Some(FocusLines {
                                start: range.0,
                                end: range.1,
                                side: Side::New,
                            }),
                        });
                    }
                }
            }
        }

        // Removed side: a control-flow node was deleted.
        if !hunk.removed_lines.is_empty() {
            if let Some(content) = file.old_content.as_deref() {
                if let Some(tree) = parse(language, content) {
                    let removed: HashSet<u32> = hunk.removed_lines.iter().copied().collect();
                    if let Some((kind, range)) = find_control_flow_node(language, &tree, &removed) {
                        return Some(Classification {
                            level: Level::Review,
                            category: Category::ControlFlow,
                            rationale: format!(
                                "{} removed at old lines {}–{}",
                                kind, range.0, range.1
                            ),
                            source: Source::Heuristic {
                                name: "control_flow".into(),
                            },
                            focus_lines: Some(FocusLines {
                                start: range.0,
                                end: range.1,
                                side: Side::Old,
                            }),
                        });
                    }
                }
            }
        }

        None
    }
}

/// Find the first control-flow node whose start line is in `targets`.
/// Returns the node's human-readable kind label and its full line range.
fn find_control_flow_node(
    language: Language,
    tree: &Tree,
    targets: &HashSet<u32>,
) -> Option<(&'static str, (u32, u32))> {
    let mut found: Option<(&'static str, (u32, u32))> = None;
    walk(tree.root_node(), &mut |node| {
        if found.is_some() {
            return;
        }
        let label = match control_flow_label(language, node.kind()) {
            Some(l) => l,
            None => return,
        };
        let start_row = node.start_position().row as u32 + 1;
        let end_row = node.end_position().row as u32 + 1;
        if targets.contains(&start_row) {
            found = Some((label, (start_row, end_row)));
        }
    });
    found
}

fn control_flow_label(language: Language, kind: &str) -> Option<&'static str> {
    match language {
        Language::Rust => match kind {
            "if_expression" => Some("if-branch"),
            "match_expression" => Some("match"),
            "for_expression" => Some("for-loop"),
            "while_expression" => Some("while-loop"),
            "loop_expression" => Some("loop"),
            "return_expression" => Some("return"),
            _ => None,
        },
        Language::Python => match kind {
            "if_statement" => Some("if-branch"),
            "match_statement" => Some("match"),
            "for_statement" => Some("for-loop"),
            "while_statement" => Some("while-loop"),
            "return_statement" => Some("return"),
            _ => None,
        },
        Language::TypeScript | Language::JavaScript => match kind {
            "if_statement" => Some("if-branch"),
            "switch_statement" => Some("switch"),
            "for_statement" | "for_in_statement" | "for_of_statement" => Some("for-loop"),
            "while_statement" => Some("while-loop"),
            "do_statement" => Some("do-loop"),
            "return_statement" => Some("return"),
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
    fn rust_added_if_branch() {
        let old = "fn f() {\n    foo();\n}\n";
        let new = "fn f() {\n    if x {\n        foo();\n    }\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2, 4], vec![]);
        let result = ControlFlow::new().classify(&mut f, &h).unwrap();
        assert_eq!(result.category, Category::ControlFlow);
        assert_eq!(result.level, Level::Review);
        let focus = result.focus_lines.unwrap();
        assert_eq!(focus.side, Side::New);
        assert_eq!(focus.start, 2);
    }

    #[test]
    fn rust_modified_if_condition() {
        let old = "fn f() {\n    if x {\n        foo();\n    }\n}\n";
        let new = "fn f() {\n    if x && y {\n        foo();\n    }\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![2]);
        assert!(ControlFlow::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn rust_body_only_change_does_not_fire() {
        let old = "fn f() {\n    if x {\n        foo();\n    }\n}\n";
        let new = "fn f() {\n    if x {\n        bar();\n    }\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![3], vec![3]);
        // Line 3 is the body, not the `if` head.
        assert!(ControlFlow::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn rust_removed_match() {
        let old = "fn f(x: u32) {\n    match x {\n        _ => {}\n    }\n}\n";
        let new = "fn f(_x: u32) {\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1, 2], vec![1, 2, 3, 4]);
        let result = ControlFlow::new().classify(&mut f, &h).unwrap();
        assert_eq!(result.category, Category::ControlFlow);
    }

    #[test]
    fn rust_added_return() {
        let old = "fn f(x: u32) -> u32 {\n    x + 1\n}\n";
        let new = "fn f(x: u32) -> u32 {\n    return x + 1;\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![2]);
        assert!(ControlFlow::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn python_added_if_branch() {
        let old = "def f(x):\n    return x\n";
        let new = "def f(x):\n    if x > 0:\n        return x\n    return 0\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![2, 3, 4], vec![2]);
        assert!(ControlFlow::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn python_added_for_loop() {
        let old = "def f(xs):\n    return xs\n";
        let new = "def f(xs):\n    for x in xs:\n        print(x)\n    return xs\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![2, 3], vec![]);
        assert!(ControlFlow::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn typescript_added_if() {
        let old = "function f(x: number): number {\n    return x;\n}\n";
        let new = "function f(x: number): number {\n    if (x > 0) {\n        return x;\n    }\n    return 0;\n}\n";
        let (mut f, h) = fixture(Language::TypeScript, old, new, vec![2, 3, 4, 5], vec![2]);
        assert!(ControlFlow::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn javascript_added_switch() {
        let old = "function f(x) { return x; }\n";
        let new = "function f(x) {\n    switch (x) {\n        case 1: return 1;\n    }\n    return x;\n}\n";
        let (mut f, h) = fixture(
            Language::JavaScript,
            old,
            new,
            vec![1, 2, 3, 4, 5, 6],
            vec![1],
        );
        assert!(ControlFlow::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn no_control_flow_change_does_not_fire() {
        let old = "fn f() {\n    let x = 1;\n}\n";
        let new = "fn f() {\n    let y = 2;\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![2]);
        assert!(ControlFlow::new().classify(&mut f, &h).is_none());
    }
}
