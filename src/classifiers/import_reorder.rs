//! Detect hunks that only reorder imports.
//!
//! Two conditions must hold:
//!   1. Every changed line falls inside an import-node line range (so the
//!      hunk doesn't touch any non-import code).
//!   2. The multiset of imports is unchanged between pre- and post-image,
//!      but their order differs.
//!
//! Imports are identified by tree-sitter node kind per language:
//!   - Rust:    `use_declaration`
//!   - Python:  `import_statement`, `import_from_statement`
//!   - TS/JS:   `import_statement`
//!
//! Import identity is the trimmed source text of the node. This is a
//! pragmatic choice: two imports that differ only in formatting (e.g.
//! whitespace inside braces) won't compare equal, so the classifier defers
//! to the LLM in those cases — a false negative, not a false positive.

use tree_sitter::Tree;

use crate::ast::{lines_all_within, parse, walk};
use crate::classify::{Category, Classification, Classifier, Level, Source};
use crate::diff::{FileDiff, Hunk};
use crate::lang::Language;

pub struct ImportReorder;

impl ImportReorder {
    pub fn new() -> Self {
        ImportReorder
    }
}

impl Default for ImportReorder {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier for ImportReorder {
    fn name(&self) -> &'static str {
        "import_reorder"
    }

    fn priority(&self) -> i32 {
        110
    }

    fn classify(&self, file: &mut FileDiff, hunk: &Hunk) -> Option<Classification> {
        let language = file.language?;
        if hunk.added_lines.is_empty() && hunk.removed_lines.is_empty() {
            return None;
        }

        let _ = file.ensure_content();
        let old_content = file.old_content.as_deref()?;
        let new_content = file.new_content.as_deref()?;
        let old_tree = parse(language, old_content)?;
        let new_tree = parse(language, new_content)?;

        let old_imports = collect_imports(language, &old_tree, old_content.as_bytes());
        let new_imports = collect_imports(language, &new_tree, new_content.as_bytes());

        // Early-out: if either side has no imports, this can't be a reorder.
        if old_imports.is_empty() || new_imports.is_empty() {
            return None;
        }

        let old_ranges: Vec<(u32, u32)> = old_imports.iter().map(|i| i.lines).collect();
        let new_ranges: Vec<(u32, u32)> = new_imports.iter().map(|i| i.lines).collect();

        if !lines_all_within(&hunk.removed_lines, &old_ranges) {
            return None;
        }
        if !lines_all_within(&hunk.added_lines, &new_ranges) {
            return None;
        }

        // Same multiset, different order = reorder. Comparing sorted Vecs is
        // simpler than juggling HashMap counts.
        let mut old_texts: Vec<&str> = old_imports.iter().map(|i| i.text.as_str()).collect();
        let mut new_texts: Vec<&str> = new_imports.iter().map(|i| i.text.as_str()).collect();
        let original_old = old_texts.clone();
        let original_new = new_texts.clone();
        old_texts.sort();
        new_texts.sort();

        if old_texts != new_texts {
            return None;
        }
        if original_old == original_new {
            // Imports unchanged in content AND order. Hunk must be touching
            // something else (whitespace?) — defer.
            return None;
        }

        Some(Classification {
            level: Level::Skip,
            category: Category::ImportReorder,
            rationale: "imports reordered, no semantic change".into(),
            source: Source::Heuristic {
                name: "import_reorder".into(),
            },
            focus_lines: None,
        })
    }
}

struct Import {
    text: String,
    lines: (u32, u32),
}

fn collect_imports(language: Language, tree: &Tree, source: &[u8]) -> Vec<Import> {
    let mut out = Vec::new();
    walk(tree.root_node(), &mut |node| {
        if !is_import_node(language, node.kind()) {
            return;
        }
        // Skip nested imports (e.g. dynamic `import()` calls inside
        // function bodies). Top-level imports have a `program` /
        // `source_file` / `module` parent.
        if !is_top_level(node) {
            return;
        }
        let text = node.utf8_text(source).unwrap_or("").trim().to_string();
        out.push(Import {
            text,
            lines: crate::ast::node_lines(node),
        });
    });
    out
}

fn is_import_node(language: Language, kind: &str) -> bool {
    match language {
        Language::Rust => kind == "use_declaration",
        Language::Python => matches!(kind, "import_statement" | "import_from_statement"),
        Language::TypeScript | Language::JavaScript => kind == "import_statement",
    }
}

fn is_top_level(node: &tree_sitter::Node) -> bool {
    matches!(
        node.parent().map(|p| p.kind()),
        Some("source_file") | Some("program") | Some("module")
    )
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
    fn rust_use_declarations_reordered() {
        let old = "use foo::bar;\nuse foo::baz;\nfn main() {}\n";
        let new = "use foo::baz;\nuse foo::bar;\nfn main() {}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1, 2], vec![1, 2]);
        let result = ImportReorder::new().classify(&mut f, &h).unwrap();
        assert_eq!(result.category, Category::ImportReorder);
        assert_eq!(result.level, Level::Skip);
    }

    #[test]
    fn rust_use_added_does_not_match() {
        let old = "use foo::bar;\nfn main() {}\n";
        let new = "use foo::bar;\nuse foo::baz;\nfn main() {}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![]);
        assert!(ImportReorder::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn rust_use_renamed_does_not_match() {
        let old = "use foo::bar;\nuse foo::baz;\nfn main() {}\n";
        let new = "use foo::bar;\nuse foo::qux;\nfn main() {}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![2]);
        assert!(ImportReorder::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn rust_mixed_change_does_not_match() {
        let old = "use foo::a;\nuse foo::b;\nfn main() { 1 }\n";
        let new = "use foo::b;\nuse foo::a;\nfn main() { 2 }\n";
        // Hunk touches both import lines (1, 2) AND the body change at line 3.
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1, 2, 3], vec![1, 2, 3]);
        assert!(ImportReorder::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn python_imports_reordered() {
        let old = "import os\nimport sys\n\nprint(1)\n";
        let new = "import sys\nimport os\n\nprint(1)\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![1, 2], vec![1, 2]);
        assert!(ImportReorder::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn python_from_imports_reordered() {
        let old = "from a import x\nfrom b import y\n\nprint(1)\n";
        let new = "from b import y\nfrom a import x\n\nprint(1)\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![1, 2], vec![1, 2]);
        assert!(ImportReorder::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn typescript_imports_reordered() {
        let old = "import { a } from 'a';\nimport { b } from 'b';\n\nexport const x = 1;\n";
        let new = "import { b } from 'b';\nimport { a } from 'a';\n\nexport const x = 1;\n";
        let (mut f, h) = fixture(Language::TypeScript, old, new, vec![1, 2], vec![1, 2]);
        assert!(ImportReorder::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn javascript_imports_reordered() {
        let old = "import a from 'a';\nimport b from 'b';\n\nconst x = 1;\n";
        let new = "import b from 'b';\nimport a from 'a';\n\nconst x = 1;\n";
        let (mut f, h) = fixture(Language::JavaScript, old, new, vec![1, 2], vec![1, 2]);
        assert!(ImportReorder::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn skips_files_with_no_language() {
        let old = "use foo;\n";
        let new = "use bar;\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        f.language = None;
        assert!(ImportReorder::new().classify(&mut f, &h).is_none());
    }
}
