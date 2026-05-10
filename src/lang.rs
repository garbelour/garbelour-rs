//! Language detection from file extensions.

use std::path::Path;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Language {
    Rust,
    Python,
    TypeScript,
    JavaScript,
}

impl Language {
    pub fn detect(path: &Path) -> Option<Language> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "rs" => Some(Language::Rust),
            "py" | "pyi" => Some(Language::Python),
            "ts" | "tsx" | "mts" | "cts" => Some(Language::TypeScript),
            "js" | "jsx" | "mjs" | "cjs" => Some(Language::JavaScript),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Language::Rust => "rust",
            Language::Python => "python",
            Language::TypeScript => "typescript",
            Language::JavaScript => "javascript",
        }
    }

    pub fn extensions(&self) -> &'static [&'static str] {
        match self {
            Language::Rust => &["rs"],
            Language::Python => &["py", "pyi"],
            Language::TypeScript => &["ts", "tsx", "mts", "cts"],
            Language::JavaScript => &["js", "jsx", "mjs", "cjs"],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_rust() {
        assert_eq!(
            Language::detect(Path::new("src/main.rs")),
            Some(Language::Rust)
        );
    }

    #[test]
    fn detects_python() {
        assert_eq!(
            Language::detect(Path::new("foo.py")),
            Some(Language::Python)
        );
        assert_eq!(
            Language::detect(Path::new("foo.pyi")),
            Some(Language::Python)
        );
    }

    #[test]
    fn detects_typescript() {
        assert_eq!(
            Language::detect(Path::new("a/b.ts")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            Language::detect(Path::new("a/b.tsx")),
            Some(Language::TypeScript)
        );
        assert_eq!(
            Language::detect(Path::new("a/b.mts")),
            Some(Language::TypeScript)
        );
    }

    #[test]
    fn detects_javascript() {
        assert_eq!(
            Language::detect(Path::new("a/b.js")),
            Some(Language::JavaScript)
        );
        assert_eq!(
            Language::detect(Path::new("a/b.cjs")),
            Some(Language::JavaScript)
        );
    }

    #[test]
    fn returns_none_for_unsupported() {
        assert_eq!(Language::detect(Path::new("Cargo.toml")), None);
        assert_eq!(Language::detect(Path::new("foo.go")), None);
        assert_eq!(Language::detect(Path::new("README.md")), None);
        assert_eq!(Language::detect(Path::new("no_extension")), None);
    }

    #[test]
    fn case_insensitive_extension() {
        assert_eq!(Language::detect(Path::new("Foo.RS")), Some(Language::Rust));
    }
}
