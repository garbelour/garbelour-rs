//! Tree-sitter parsing and node-walk helpers shared by the AST classifiers.

use tree_sitter::{Node, Parser, Tree};

use crate::lang::Language;

/// Parse `source` as the given language. Returns `None` if the language
/// failed to load or parsing produced no tree (extremely rare —
/// tree-sitter is tolerant of syntax errors and returns a tree with
/// `has_error()` set rather than failing outright).
pub fn parse(language: Language, source: &str) -> Option<Tree> {
    let mut parser = Parser::new();
    let lang: tree_sitter::Language = match language {
        Language::Rust => tree_sitter_rust::LANGUAGE.into(),
        Language::Python => tree_sitter_python::LANGUAGE.into(),
        // TSX is a strict superset of TS — it accepts all TS plus JSX.
        // Using it for both `.ts` and `.tsx` keeps the dispatch simple.
        Language::TypeScript => tree_sitter_typescript::LANGUAGE_TSX.into(),
        Language::JavaScript => tree_sitter_javascript::LANGUAGE.into(),
    };
    parser.set_language(&lang).ok()?;
    parser.parse(source, None)
}

/// 1-indexed inclusive line range that a node spans.
pub fn node_lines(node: &Node) -> (u32, u32) {
    let start = node.start_position().row as u32 + 1;
    let end = node.end_position().row as u32 + 1;
    (start, end)
}

/// Walk every node in the subtree, calling `visit`. Pre-order traversal.
pub fn walk<F>(node: Node, visit: &mut F)
where
    F: FnMut(&Node),
{
    visit(&node);
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        walk(child, visit);
    }
}

/// Collect 1-indexed inclusive line ranges of every node satisfying `pred`.
pub fn collect_line_ranges<F>(tree: &Tree, mut pred: F) -> Vec<(u32, u32)>
where
    F: FnMut(&Node) -> bool,
{
    let mut ranges = Vec::new();
    walk(tree.root_node(), &mut |node| {
        if pred(node) {
            ranges.push(node_lines(node));
        }
    });
    ranges
}

/// True if every line in `lines` falls inside at least one of `ranges`
/// (inclusive on both ends).
pub fn lines_all_within(lines: &[u32], ranges: &[(u32, u32)]) -> bool {
    'outer: for &line in lines {
        for &(s, e) in ranges {
            if line >= s && line <= e {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// True if any line in `lines` falls inside any of `ranges`.
pub fn any_line_within(lines: &[u32], ranges: &[(u32, u32)]) -> bool {
    for &line in lines {
        for &(s, e) in ranges {
            if line >= s && line <= e {
                return true;
            }
        }
    }
    false
}

/// True if a line of source consists only of whitespace.
pub fn is_blank_line(line: &str) -> bool {
    line.chars().all(char::is_whitespace)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_rust() {
        let tree = parse(Language::Rust, "fn main() { let x = 1; }").unwrap();
        assert_eq!(tree.root_node().kind(), "source_file");
    }

    #[test]
    fn parses_python() {
        let tree = parse(Language::Python, "def f():\n    return 1\n").unwrap();
        assert_eq!(tree.root_node().kind(), "module");
    }

    #[test]
    fn parses_typescript() {
        let tree = parse(Language::TypeScript, "export const x: number = 1;\n").unwrap();
        assert_eq!(tree.root_node().kind(), "program");
    }

    #[test]
    fn parses_javascript() {
        let tree = parse(Language::JavaScript, "const x = 1;\n").unwrap();
        assert_eq!(tree.root_node().kind(), "program");
    }

    #[test]
    fn collects_node_line_ranges() {
        let src = "// one\nfn a() {}\n// two\nfn b() {}\n";
        let tree = parse(Language::Rust, src).unwrap();
        let comments = collect_line_ranges(&tree, |n| n.kind() == "line_comment");
        assert_eq!(comments, vec![(1, 1), (3, 3)]);
    }

    #[test]
    fn lines_all_within_works() {
        let ranges = vec![(1, 3), (5, 7)];
        assert!(lines_all_within(&[1, 2, 3, 5, 6, 7], &ranges));
        assert!(!lines_all_within(&[4], &ranges));
        assert!(!lines_all_within(&[1, 4], &ranges));
        assert!(lines_all_within(&[], &ranges));
    }

    #[test]
    fn any_line_within_works() {
        let ranges = vec![(1, 3), (5, 7)];
        assert!(any_line_within(&[2, 100], &ranges));
        assert!(!any_line_within(&[4, 8], &ranges));
        assert!(!any_line_within(&[], &ranges));
    }
}
