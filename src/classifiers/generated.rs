//! Match files marked as generated. Pure path lookup; no parsing.
//!
//! Two sources:
//!   1. A configurable set of globs (`[classify].generated_globs`), merged
//!      with a default set covering common artifacts (lockfiles, protobuf
//!      output, build directories, minified assets).
//!   2. The set of paths marked `linguist-generated=true` in
//!      `.gitattributes`, parsed once during pipeline construction and
//!      passed in here. The trait has no shared-state hook, so
//!      pre-computation in the pipeline constructor is the cleanest place
//!      for this.

use std::collections::HashSet;
use std::path::PathBuf;

use anyhow::Context;
use globset::{Glob, GlobSet, GlobSetBuilder};

use crate::classify::{Category, Classifier, Finding, Level, Source};
use crate::diff::{FileDiff, Hunk};

/// Default globs. Extended via `[classify].generated_globs` in `garbelour.toml`.
pub const DEFAULT_GLOBS: &[&str] = &[
    "*.lock",
    "package-lock.json",
    "*_pb2.py",
    "*.pb.go",
    "dist/**",
    "build/**",
    "vendor/**",
    "*.min.js",
    "*.min.css",
    "*.generated.*",
];

pub struct Generated {
    globset: GlobSet,
    paths: HashSet<PathBuf>,
}

impl Generated {
    pub fn new(extra_globs: Vec<String>, paths: HashSet<PathBuf>) -> anyhow::Result<Self> {
        let mut builder = GlobSetBuilder::new();
        for pattern in DEFAULT_GLOBS
            .iter()
            .map(|s| (*s).to_string())
            .chain(extra_globs)
        {
            let glob = Glob::new(&pattern)
                .with_context(|| format!("invalid generated_globs pattern: {pattern:?}"))?;
            builder.add(glob);
        }
        Ok(Self {
            globset: builder.build()?,
            paths,
        })
    }
}

impl Classifier for Generated {
    fn name(&self) -> &'static str {
        "generated"
    }

    fn priority(&self) -> i32 {
        0
    }

    fn classify(&self, file: &mut FileDiff, _hunk: &Hunk) -> Vec<Finding> {
        let matched_glob = self.globset.is_match(&file.path);
        let matched_path = self.paths.contains(&file.path);
        if matched_glob || matched_path {
            let reason = if matched_path {
                "marked linguist-generated"
            } else {
                "matches generated_globs"
            };
            vec![Finding {
                level: Level::Skip,
                category: Category::Generated,
                rationale: format!("generated file ({})", reason),
                source: Source::Heuristic {
                    name: "generated".into(),
                },
                focus_lines: None,
            }]
        } else {
            Vec::new()
        }
    }
}

/// Read `.gitattributes` from `repo_path` and return the set of paths marked
/// `linguist-generated=true`. Returns an empty set if the file is missing.
/// Only literal path patterns are honored — true glob patterns in
/// `.gitattributes` would require integrating a gitattributes parser, which
/// is out of scope for v1; users with glob-based markers can fall back to
/// `[classify].generated_globs`.
pub fn read_gitattributes_generated(repo_path: &std::path::Path) -> HashSet<PathBuf> {
    let path = repo_path.join(".gitattributes");
    let Ok(content) = std::fs::read_to_string(&path) else {
        return HashSet::new();
    };
    let mut out = HashSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let mut tokens = trimmed.split_whitespace();
        let Some(pattern) = tokens.next() else {
            continue;
        };
        let mut marked = false;
        for attr in tokens {
            // We honor only the literal `linguist-generated=true` form (and
            // its shorthand `linguist-generated`). The negation forms
            // `-linguist-generated` and `linguist-generated=false` cause the
            // pattern to NOT be added.
            if attr == "linguist-generated" || attr == "linguist-generated=true" {
                marked = true;
            }
            if attr == "-linguist-generated" || attr == "linguist-generated=false" {
                marked = false;
            }
        }
        // Only honor literal paths (no `*`, `?`, `[`). Glob patterns in
        // gitattributes would need a real parser.
        let has_glob_meta = pattern.contains(['*', '?', '[']);
        if marked && !has_glob_meta {
            out.insert(PathBuf::from(pattern));
        }
    }
    out
}

//------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::diff::{FileStatus, Hunk, HunkId, LineRange};

    fn hunk() -> Hunk {
        Hunk {
            id: HunkId("test:1".into()),
            old_range: LineRange { start: 1, count: 1 },
            new_range: LineRange { start: 1, count: 1 },
            old_lines: vec!["x".into()],
            new_lines: vec!["y".into()],
            added: 1,
            removed: 1,
            added_lines: vec![1],
            removed_lines: vec![1],
        }
    }

    fn file(path: &str) -> FileDiff {
        FileDiff::for_test(
            PathBuf::from(path),
            FileStatus::Modified,
            None,
            vec![hunk()],
        )
    }

    #[test]
    fn matches_default_globs() {
        let cls = Generated::new(vec![], HashSet::new()).unwrap();
        for path in [
            "Cargo.lock",
            "package-lock.json",
            "proto/foo_pb2.py",
            "vendor/foo/bar.go",
            "dist/bundle.js",
            "build/index.html",
            "static/main.min.js",
            "static/style.min.css",
            "src/types.generated.ts",
        ] {
            let mut f = file(path);
            assert!(
                !cls.classify(&mut f, &hunk()).is_empty(),
                "should match default glob: {path}"
            );
        }
    }

    #[test]
    fn does_not_match_arbitrary_source() {
        let cls = Generated::new(vec![], HashSet::new()).unwrap();
        let mut f = file("src/main.rs");
        assert!(cls.classify(&mut f, &hunk()).is_empty());
    }

    #[test]
    fn matches_extra_glob_from_config() {
        let cls = Generated::new(vec!["generated/**".into()], HashSet::new()).unwrap();
        let mut f = file("generated/types.ts");
        assert!(!cls.classify(&mut f, &hunk()).is_empty());
    }

    #[test]
    fn matches_path_from_gitattributes_set() {
        let mut paths = HashSet::new();
        paths.insert(PathBuf::from("docs/api.json"));
        let cls = Generated::new(vec![], paths).unwrap();
        let mut f = file("docs/api.json");
        let result = cls.classify(&mut f, &hunk());
        assert_eq!(result.len(), 1);
        assert!(result[0].rationale.contains("linguist-generated"));
    }

    #[test]
    fn parses_gitattributes_literal_paths_only() {
        let dir = tempdir();
        std::fs::write(
            dir.join(".gitattributes"),
            "# comment\n\
             docs/api.json linguist-generated\n\
             generated/manifest.json linguist-generated=true\n\
             src/* linguist-generated\n\
             vendored/lib.js linguist-generated=false\n",
        )
        .unwrap();
        let paths = read_gitattributes_generated(&dir);
        assert!(paths.contains(&PathBuf::from("docs/api.json")));
        assert!(paths.contains(&PathBuf::from("generated/manifest.json")));
        assert!(
            !paths.contains(&PathBuf::from("src/*")),
            "globs are skipped"
        );
        assert!(
            !paths.contains(&PathBuf::from("vendored/lib.js")),
            "explicitly negated"
        );
    }

    #[test]
    fn missing_gitattributes_returns_empty_set() {
        let dir = tempdir();
        let paths = read_gitattributes_generated(&dir);
        assert!(paths.is_empty());
    }

    /// Tiny throwaway temp directory; auto-cleaned at process exit. We don't
    /// pull in `tempfile` for one test.
    fn tempdir() -> PathBuf {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("garbelour-test-{nanos}"));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
