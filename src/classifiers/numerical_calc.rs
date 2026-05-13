//! Detect numerical-calculation changes.
//!
//! Numerical changes are high-stakes (off-by-one, sign flip, lost precision)
//! and a small textual change can hide a big semantic one — exactly the kind
//! of edit the LLM is least reliable about. We flag them deterministically.
//!
//! Triggers (any one fires):
//!
//! 1. **Multi-operator arithmetic**: a `binary_expression` (or, in Python,
//!    `binary_operator`) whose operator is arithmetic AND that contains at
//!    least one nested arithmetic operator. This catches real computations
//!    (`(a * b) + c`, `x*x + y*y`) while excluding trivial index math like
//!    `i + 1` or `len - 1`.
//!
//! 2. **Math-namespace call**: an added/removed line contains a call into a
//!    recognized math namespace — `Math.`, `math.`, `np.`, `numpy.`,
//!    `scipy.`, `torch.`, `f64::`, `f32::`, `libm::`, `num_traits::`. Done
//!    textually (per-line substring) because the AST shape varies too much
//!    across languages to be worth a custom matcher for each.

use std::collections::HashSet;

use tree_sitter::{Node, Tree};

use crate::ast::{parse, walk};
use crate::classify::{Category, Classifier, Finding, FocusLines, Level, Side, Source};
use crate::diff::{FileDiff, Hunk};
use crate::lang::Language;

pub struct NumericalCalc;

impl NumericalCalc {
    pub fn new() -> Self {
        NumericalCalc
    }
}

impl Default for NumericalCalc {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier for NumericalCalc {
    fn name(&self) -> &'static str {
        "numerical_calc"
    }

    fn priority(&self) -> i32 {
        // After public_api (120) and control_flow (130), before
        // error_handling. Numerical changes are usually still review-worthy
        // even when they coincide with another signal — but if a stronger
        // signal (e.g. public_api signature change) is already firing, we
        // defer.
        140
    }

    fn classify(&self, file: &mut FileDiff, hunk: &Hunk) -> Vec<Finding> {
        let Some(language) = file.language else {
            return Vec::new();
        };
        if hunk.added_lines.is_empty() && hunk.removed_lines.is_empty() {
            return Vec::new();
        }
        let _ = file.ensure_content();

        let mut findings: Vec<Finding> = Vec::new();

        if !hunk.added_lines.is_empty() {
            if let Some(content) = file.new_content.as_deref() {
                for hit in detect(language, content, &hunk.added_lines) {
                    findings.push(make(hit, Side::New));
                }
            }
        }
        if !hunk.removed_lines.is_empty() {
            if let Some(content) = file.old_content.as_deref() {
                for hit in detect(language, content, &hunk.removed_lines) {
                    findings.push(make(hit, Side::Old));
                }
            }
        }
        findings
    }
}

struct Hit {
    start: u32,
    end: u32,
    what: String,
}

fn make(hit: Hit, side: Side) -> Finding {
    let rationale = match side {
        Side::New => format!("{} at lines {}–{}", hit.what, hit.start, hit.end),
        Side::Old => format!(
            "{} removed at old lines {}–{}",
            hit.what, hit.start, hit.end
        ),
    };
    Finding {
        level: Level::Review,
        category: Category::NumericalCalc,
        rationale,
        source: Source::Heuristic {
            name: "numerical_calc".into(),
        },
        focus_lines: Some(FocusLines {
            start: hit.start,
            end: hit.end,
            side,
        }),
    }
}

/// Collect every numerical-calc signal (multi-op arithmetic node + math
/// namespace call line) that intersects `target_lines`. Deduped by start/end.
fn detect(language: Language, content: &str, target_lines: &[u32]) -> Vec<Hit> {
    let mut out: Vec<Hit> = Vec::new();
    let mut seen: HashSet<(u32, u32)> = HashSet::new();

    if let Some(tree) = parse(language, content) {
        let targets: HashSet<u32> = target_lines.iter().copied().collect();
        for hit in collect_multi_op_arithmetic(language, &tree, &targets) {
            if seen.insert((hit.start, hit.end)) {
                out.push(hit);
            }
        }
    }
    let lines: Vec<&str> = content.lines().collect();
    for &ln in target_lines {
        let idx = ln.saturating_sub(1) as usize;
        if idx >= lines.len() {
            continue;
        }
        if let Some(label) = math_namespace_in(lines[idx]) {
            if seen.insert((ln, ln)) {
                out.push(Hit {
                    start: ln,
                    end: ln,
                    what: label.to_string(),
                });
            }
        }
    }
    out
}

fn collect_multi_op_arithmetic(
    language: Language,
    tree: &Tree,
    targets: &HashSet<u32>,
) -> Vec<Hit> {
    let mut out: Vec<Hit> = Vec::new();
    walk(tree.root_node(), &mut |node| {
        if !is_arithmetic_binary(language, node) {
            return;
        }
        let total = arithmetic_op_count(language, *node);
        if total < 2 {
            return;
        }
        let s = node.start_position().row as u32 + 1;
        let e = node.end_position().row as u32 + 1;
        let intersects = (s..=e).any(|ln| targets.contains(&ln));
        if intersects {
            out.push(Hit {
                start: s,
                end: e,
                what: format!("arithmetic expression ({} operators)", total),
            });
        }
    });
    out
}

fn is_arithmetic_binary(language: Language, node: &Node) -> bool {
    let kind = match language {
        Language::Rust => "binary_expression",
        Language::Python => "binary_operator",
        Language::TypeScript | Language::JavaScript => "binary_expression",
    };
    if node.kind() != kind {
        return false;
    }
    let mut cur = node.walk();
    for child in node.children(&mut cur) {
        if !child.is_named() && is_arithmetic_op_kind(child.kind()) {
            return true;
        }
    }
    false
}

fn arithmetic_op_count(language: Language, node: Node) -> usize {
    let mut count = 0;
    walk(node, &mut |n| {
        if is_arithmetic_binary(language, n) {
            count += 1;
        }
    });
    count
}

fn is_arithmetic_op_kind(kind: &str) -> bool {
    matches!(
        kind,
        "+" | "-" | "*" | "/" | "%" | "**" | "<<" | ">>" | "^" | "&" | "|"
    )
}

fn math_namespace_in(line: &str) -> Option<&'static str> {
    // Order matters only for label preference (longer/more specific first).
    const PATTERNS: &[(&str, &str)] = &[
        ("numpy.", "numpy.* call"),
        ("scipy.", "scipy.* call"),
        ("torch.", "torch.* call"),
        ("num_traits::", "num_traits::* call"),
        ("libm::", "libm::* call"),
        ("Math.", "Math.* call"),
        ("math.", "math.* call"),
        ("np.", "np.* call"),
        ("f64::", "f64::* call"),
        ("f32::", "f32::* call"),
    ];
    for &(needle, label) in PATTERNS {
        if line.contains(needle) {
            return Some(label);
        }
    }
    None
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
    fn rust_multi_operator_arithmetic_fires() {
        let old = "fn f(a: f64, b: f64) -> f64 {\n    a\n}\n";
        let new = "fn f(a: f64, b: f64) -> f64 {\n    a * a + b * b\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![2]);
        let result = NumericalCalc::new().classify(&mut f, &h);
        assert!(!result.is_empty());
        assert_eq!(result[0].category, Category::NumericalCalc);
        assert_eq!(result[0].level, Level::Review);
        assert!(result[0].rationale.contains("arithmetic expression"));
    }

    #[test]
    fn rust_trivial_increment_does_not_fire() {
        let old = "fn f(i: usize) -> usize {\n    i\n}\n";
        let new = "fn f(i: usize) -> usize {\n    i + 1\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![2]);
        // Single arithmetic operator, no nesting — should not fire.
        assert!(NumericalCalc::new().classify(&mut f, &h).is_empty());
    }

    #[test]
    fn rust_f64_namespace_call_fires() {
        let old = "fn f(x: f64) -> f64 {\n    x\n}\n";
        let new = "fn f(x: f64) -> f64 {\n    f64::sin(x)\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![2]);
        let result = NumericalCalc::new().classify(&mut f, &h);
        assert!(!result.is_empty());
        assert_eq!(result[0].category, Category::NumericalCalc);
        assert!(result.iter().any(|r| r.rationale.contains("f64::")));
    }

    #[test]
    fn python_numpy_call_fires() {
        let old = "def f(x):\n    return x\n";
        let new = "def f(x):\n    return np.dot(x, x)\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![2], vec![2]);
        let result = NumericalCalc::new().classify(&mut f, &h);
        assert!(!result.is_empty());
        assert!(result.iter().any(|r| r.rationale.contains("np.")));
    }

    #[test]
    fn python_multi_op_fires() {
        let old = "def f(a, b):\n    return a\n";
        let new = "def f(a, b):\n    return a * a + b * b\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![2], vec![2]);
        assert!(!NumericalCalc::new().classify(&mut f, &h).is_empty());
    }

    #[test]
    fn typescript_math_call_fires() {
        let old = "function f(x: number): number { return x; }\n";
        let new = "function f(x: number): number { return Math.sqrt(x); }\n";
        let (mut f, h) = fixture(Language::TypeScript, old, new, vec![1], vec![1]);
        let result = NumericalCalc::new().classify(&mut f, &h);
        assert!(result.iter().any(|r| r.rationale.contains("Math.")));
    }

    #[test]
    fn javascript_multi_op_fires() {
        let old = "function f(a, b) { return a; }\n";
        let new = "function f(a, b) { return a * a + b * b; }\n";
        let (mut f, h) = fixture(Language::JavaScript, old, new, vec![1], vec![1]);
        assert!(!NumericalCalc::new().classify(&mut f, &h).is_empty());
    }

    #[test]
    fn removed_arithmetic_fires_on_old_side() {
        let old = "fn f(a: f64, b: f64) -> f64 {\n    a * a + b * b\n}\n";
        let new = "fn f(a: f64, _b: f64) -> f64 {\n    a\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1, 2], vec![1, 2]);
        let result = NumericalCalc::new().classify(&mut f, &h);
        let old_hit = result
            .iter()
            .find(|r| {
                r.focus_lines
                    .as_ref()
                    .is_some_and(|fl| fl.side == Side::Old)
            })
            .expect("expected an old-side hit");
        assert!(old_hit.rationale.contains("removed at old lines"));
    }

    #[test]
    fn no_numerical_change_does_not_fire() {
        let old = "fn f() -> String { String::new() }\n";
        let new = "fn f() -> String { String::from(\"hi\") }\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        assert!(NumericalCalc::new().classify(&mut f, &h).is_empty());
    }

    /// Two independent multi-op arithmetic expressions on different lines
    /// produce two findings.
    #[test]
    fn rust_two_independent_multi_op_expressions_produce_two_findings() {
        let old = "fn f(a: f64, b: f64) -> f64 { a }\n";
        let new = "fn f(a: f64, b: f64) -> f64 {\n    let x = a * a + b * b;\n    let y = (a - b) * (a + b);\n    x + y\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2, 3], vec![]);
        let result = NumericalCalc::new().classify(&mut f, &h);
        let arith_count = result
            .iter()
            .filter(|r| r.rationale.contains("arithmetic expression"))
            .count();
        assert!(
            arith_count >= 2,
            "expected >=2 arithmetic findings, got {arith_count}: {result:?}"
        );
    }
}
