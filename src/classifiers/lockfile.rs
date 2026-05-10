//! Match well-known lockfile filenames. Pure path lookup; no parsing.

use std::collections::HashSet;

use crate::classify::{Category, Classification, Classifier, Level, Source};
use crate::diff::{FileDiff, Hunk};

/// Filenames recognized as lockfiles by default. Extended via
/// `[classify].lockfile_names` in `garbelour.toml`.
pub const DEFAULT_NAMES: &[&str] = &[
    "Cargo.lock",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "Gemfile.lock",
    "poetry.lock",
    "Pipfile.lock",
    "composer.lock",
    "go.sum",
];

pub struct Lockfile {
    names: HashSet<String>,
}

impl Lockfile {
    pub fn new(extra: Vec<String>) -> Self {
        let mut names: HashSet<String> =
            DEFAULT_NAMES.iter().map(|s| (*s).to_string()).collect();
        names.extend(extra);
        Self { names }
    }
}

impl Classifier for Lockfile {
    fn name(&self) -> &'static str {
        "lockfile"
    }

    fn priority(&self) -> i32 {
        1
    }

    fn classify(&self, file: &mut FileDiff, _hunk: &Hunk) -> Option<Classification> {
        let basename = file.path.file_name()?.to_str()?;
        if self.names.contains(basename) {
            Some(Classification {
                level: Level::Skip,
                category: Category::Lockfile,
                rationale: format!("lockfile update ({})", basename),
                source: Source::Heuristic { name: "lockfile".into() },
                focus_lines: None,
            })
        } else {
            None
        }
    }
}

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
        FileDiff::for_test(PathBuf::from(path), FileStatus::Modified, None, vec![hunk()])
    }

    #[test]
    fn matches_default_lockfile_names() {
        let cls = Lockfile::new(vec![]);
        for name in DEFAULT_NAMES {
            let mut f = file(name);
            let result = cls.classify(&mut f, &hunk()).expect(name);
            assert_eq!(result.level, Level::Skip);
            assert_eq!(result.category, Category::Lockfile);
        }
    }

    #[test]
    fn matches_lockfile_in_subdirectory() {
        let cls = Lockfile::new(vec![]);
        let mut f = file("packages/web/yarn.lock");
        assert!(cls.classify(&mut f, &hunk()).is_some());
    }

    #[test]
    fn does_not_match_arbitrary_files() {
        let cls = Lockfile::new(vec![]);
        let mut f = file("src/main.rs");
        assert!(cls.classify(&mut f, &hunk()).is_none());
    }

    #[test]
    fn matches_extra_names_from_config() {
        let cls = Lockfile::new(vec!["shrinkwrap.json".into()]);
        let mut f = file("shrinkwrap.json");
        assert!(cls.classify(&mut f, &hunk()).is_some());
    }

    #[test]
    fn rationale_includes_basename() {
        let cls = Lockfile::new(vec![]);
        let mut f = file("a/b/Cargo.lock");
        let result = cls.classify(&mut f, &hunk()).unwrap();
        assert!(result.rationale.contains("Cargo.lock"));
    }
}
