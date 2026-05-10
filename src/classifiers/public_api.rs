//! Detect changes to a file's public API surface.
//!
//! Per language:
//!   - **Rust:** any change to a `pub` declaration's signature (function,
//!     struct, enum, trait, type alias, const, or static). Body-only
//!     changes are not flagged.
//!   - **Python:** any change to a module-level `def` or `class` whose name
//!     does not start with `_`. Signature changes only — body changes go
//!     through `ControlFlow` or the LLM.
//!   - **TS/JS:** any change inside an `export_statement` signature.
//!
//! `focus_lines` is populated with the matched signature's line range so
//! the rendered link points the reviewer at the exact declaration.

use tree_sitter::{Node, Tree};

use crate::ast::{node_lines, parse, walk};
use crate::classify::{Category, Classification, Classifier, FocusLines, Level, Side, Source};
use crate::diff::{FileDiff, Hunk};
use crate::lang::Language;

pub struct PublicApi;

impl PublicApi {
    pub fn new() -> Self {
        PublicApi
    }
}

impl Default for PublicApi {
    fn default() -> Self {
        Self::new()
    }
}

impl Classifier for PublicApi {
    fn name(&self) -> &'static str {
        "public_api"
    }

    fn priority(&self) -> i32 {
        120
    }

    fn classify(&self, file: &mut FileDiff, hunk: &Hunk) -> Option<Classification> {
        let language = file.language?;
        if hunk.added_lines.is_empty() && hunk.removed_lines.is_empty() {
            return None;
        }

        let _ = file.ensure_content();
        let new_content = file.new_content.as_deref()?;
        let new_tree = parse(language, new_content)?;
        let decls = collect_public_decls(language, &new_tree, new_content.as_bytes());

        // Find the first public-decl signature that any added line falls in.
        for decl in &decls {
            for &line in &hunk.added_lines {
                if line >= decl.sig.0 && line <= decl.sig.1 {
                    return Some(Classification {
                        level: Level::Review,
                        category: Category::PublicApiChange,
                        rationale: format!(
                            "{} signature changed at lines {}–{}",
                            decl.kind_label, decl.sig.0, decl.sig.1
                        ),
                        source: Source::Heuristic {
                            name: "public_api".into(),
                        },
                        focus_lines: Some(FocusLines {
                            start: decl.sig.0,
                            end: decl.sig.1,
                            side: Side::New,
                        }),
                    });
                }
            }
        }

        // Removed-only changes (e.g. a deleted public fn) — also worth
        // flagging. Look at old tree.
        if !hunk.removed_lines.is_empty() {
            let old_content = file.old_content.as_deref()?;
            let old_tree = parse(language, old_content)?;
            let old_decls = collect_public_decls(language, &old_tree, old_content.as_bytes());
            for decl in &old_decls {
                for &line in &hunk.removed_lines {
                    if line >= decl.sig.0 && line <= decl.sig.1 {
                        return Some(Classification {
                            level: Level::Review,
                            category: Category::PublicApiChange,
                            rationale: format!(
                                "{} removed at old lines {}–{}",
                                decl.kind_label, decl.sig.0, decl.sig.1
                            ),
                            source: Source::Heuristic {
                                name: "public_api".into(),
                            },
                            focus_lines: Some(FocusLines {
                                start: decl.sig.0,
                                end: decl.sig.1,
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

struct PublicDecl {
    /// Inclusive line range of the signature. For functions/traits this is
    /// the declaration line through the line just before the `{`. For
    /// declarations without a body, it is the entire node.
    sig: (u32, u32),
    /// Human-readable kind name for the rationale ("public fn", "exported
    /// const", etc.).
    kind_label: &'static str,
}

fn collect_public_decls(language: Language, tree: &Tree, source: &[u8]) -> Vec<PublicDecl> {
    let mut out = Vec::new();
    walk(tree.root_node(), &mut |node| {
        if let Some(decl) = inspect_node(language, node, source) {
            out.push(decl);
        }
    });
    out
}

fn inspect_node(language: Language, node: &Node, source: &[u8]) -> Option<PublicDecl> {
    match language {
        Language::Rust => rust_decl(node),
        Language::Python => python_decl(node, source),
        Language::TypeScript | Language::JavaScript => ts_js_decl(node),
    }
}

fn rust_decl(node: &Node) -> Option<PublicDecl> {
    // Body exclusion is meaningful only for function-like items where the
    // body is implementation, not API. For structs/enums (fields/variants),
    // traits (method signatures), and value items (const/static/type) the
    // entire item is the public surface.
    let (label, exclude_body) = match node.kind() {
        "function_item" => ("public fn", true),
        "struct_item" => ("public struct", false),
        "enum_item" => ("public enum", false),
        "trait_item" => ("public trait", false),
        "type_item" => ("public type", false),
        "const_item" => ("public const", false),
        "static_item" => ("public static", false),
        _ => return None,
    };
    if !has_pub_visibility(node) {
        return None;
    }
    let sig = if exclude_body {
        signature_range_excluding_body(node, "body").unwrap_or_else(|| node_lines(node))
    } else {
        node_lines(node)
    };
    Some(PublicDecl {
        sig,
        kind_label: label,
    })
}

fn python_decl(node: &Node, source: &[u8]) -> Option<PublicDecl> {
    // Body exclusion: function bodies are implementation; class bodies
    // (methods, attrs) are API surface.
    let (label, exclude_body) = match node.kind() {
        "function_definition" => ("module-level def", true),
        "class_definition" => ("module-level class", false),
        _ => return None,
    };
    if !is_module_level(node) {
        return None;
    }
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?;
    if name.starts_with('_') {
        return None;
    }
    let sig = if exclude_body {
        signature_range_excluding_body(node, "body").unwrap_or_else(|| node_lines(node))
    } else {
        node_lines(node)
    };
    Some(PublicDecl {
        sig,
        kind_label: label,
    })
}

fn ts_js_decl(node: &Node) -> Option<PublicDecl> {
    if node.kind() != "export_statement" {
        return None;
    }
    // For an `export function f() { ... }`, the signature is everything up
    // to the body of the inner declaration. We walk into the export_statement
    // and look for a function/class/etc.
    let mut sig = node_lines(node);
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(r) = signature_range_excluding_body(&child, "body") {
            // Inner declaration has a body; trim signature to before it.
            sig = (sig.0, r.1);
            break;
        }
    }
    Some(PublicDecl {
        sig,
        kind_label: "exported declaration",
    })
}

fn has_pub_visibility(node: &Node) -> bool {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "visibility_modifier" {
            // Any `pub`-prefixed visibility counts: `pub`, `pub(crate)`,
            // `pub(super)`, `pub(in path)`. They all expose the item to
            // consumers outside this file.
            return true;
        }
    }
    false
}

fn is_module_level(node: &Node) -> bool {
    let parent_kind = node.parent().map(|p| p.kind());
    if parent_kind == Some("module") {
        return true;
    }
    // Decorated functions/classes have a `decorated_definition` parent
    // whose own parent is the module.
    if parent_kind == Some("decorated_definition") {
        return node
            .parent()
            .and_then(|p| p.parent())
            .map(|gp| gp.kind() == "module")
            .unwrap_or(false);
    }
    false
}

/// Return `(start_line, end_line_just_before_body)` if `node` has a child
/// with field name `body_field`; otherwise `None`. Lines are 1-indexed.
fn signature_range_excluding_body(node: &Node, body_field: &str) -> Option<(u32, u32)> {
    let body = node.child_by_field_name(body_field)?;
    let start = node.start_position().row as u32 + 1;
    // Body's start row, minus one if the body is on its own line; if the
    // body opens on the same line as the signature (one-liner like
    // `fn f() { 1 }`), use that line.
    let body_row = body.start_position().row as u32 + 1;
    let end = if body_row > start {
        body_row - 1
    } else {
        start
    };
    Some((start, end.max(start)))
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
    fn rust_pub_fn_signature_change() {
        let old = "pub fn apply(x: u32) -> u32 {\n    x + 1\n}\n";
        let new = "pub fn apply(x: u32, y: u32) -> u32 {\n    x + y\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        let result = PublicApi::new().classify(&mut f, &h).unwrap();
        assert_eq!(result.level, Level::Review);
        assert_eq!(result.category, Category::PublicApiChange);
        let focus = result.focus_lines.unwrap();
        assert_eq!(focus.side, Side::New);
        assert!(focus.start <= 1 && focus.end >= 1);
    }

    #[test]
    fn rust_pub_fn_body_change_does_not_fire() {
        let old = "pub fn apply(x: u32) -> u32 {\n    x + 1\n}\n";
        let new = "pub fn apply(x: u32) -> u32 {\n    x + 2\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![2], vec![2]);
        // Line 2 is the body. Signature on line 1 is unchanged.
        assert!(PublicApi::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn rust_private_fn_does_not_fire() {
        let old = "fn helper(x: u32) -> u32 {\n    x + 1\n}\n";
        let new = "fn helper(x: u32, y: u32) -> u32 {\n    x + y\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        assert!(PublicApi::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn rust_pub_struct_field_change() {
        let old = "pub struct Foo {\n    pub a: u32,\n}\n";
        let new = "pub struct Foo {\n    pub a: u32,\n    pub b: u32,\n}\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![3], vec![]);
        // Struct body is part of its public surface — line 3 is added.
        assert!(PublicApi::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn rust_pub_const_change() {
        let old = "pub const MAX: u32 = 10;\n";
        let new = "pub const MAX: u32 = 20;\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        assert!(PublicApi::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn rust_pub_crate_visibility_counts_as_public() {
        let old = "pub(crate) fn apply(x: u32) -> u32 { x }\n";
        let new = "pub(crate) fn apply(x: u32, y: u32) -> u32 { x + y }\n";
        let (mut f, h) = fixture(Language::Rust, old, new, vec![1], vec![1]);
        assert!(PublicApi::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn python_module_level_def_signature_change() {
        let old = "def apply(x):\n    return x + 1\n";
        let new = "def apply(x, y):\n    return x + y\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![1], vec![1]);
        let result = PublicApi::new().classify(&mut f, &h).unwrap();
        assert_eq!(result.category, Category::PublicApiChange);
    }

    #[test]
    fn python_underscore_def_does_not_fire() {
        let old = "def _helper(x):\n    return x + 1\n";
        let new = "def _helper(x, y):\n    return x + y\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![1], vec![1]);
        assert!(PublicApi::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn python_class_signature_change() {
        let old = "class Foo:\n    pass\n";
        let new = "class Foo(Bar):\n    pass\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![1], vec![1]);
        assert!(PublicApi::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn python_function_body_only_does_not_fire() {
        let old = "def apply(x):\n    return x + 1\n";
        let new = "def apply(x):\n    return x + 2\n";
        let (mut f, h) = fixture(Language::Python, old, new, vec![2], vec![2]);
        assert!(PublicApi::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn typescript_export_function_signature_change() {
        let old = "export function apply(x: number): number {\n    return x + 1;\n}\n";
        let new = "export function apply(x: number, y: number): number {\n    return x + y;\n}\n";
        let (mut f, h) = fixture(Language::TypeScript, old, new, vec![1], vec![1]);
        assert!(PublicApi::new().classify(&mut f, &h).is_some());
    }

    #[test]
    fn typescript_non_export_does_not_fire() {
        let old = "function helper(x: number): number {\n    return x + 1;\n}\n";
        let new = "function helper(x: number, y: number): number {\n    return x + y;\n}\n";
        let (mut f, h) = fixture(Language::TypeScript, old, new, vec![1], vec![1]);
        assert!(PublicApi::new().classify(&mut f, &h).is_none());
    }

    #[test]
    fn javascript_export_const_change() {
        let old = "export const VERSION = '1.0';\n";
        let new = "export const VERSION = '2.0';\n";
        let (mut f, h) = fixture(Language::JavaScript, old, new, vec![1], vec![1]);
        assert!(PublicApi::new().classify(&mut f, &h).is_some());
    }
}
