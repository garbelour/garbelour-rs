//! Auto-elevate large hunks to `Review`. Pure line counting; no parsing.
//!
//! A hunk that touches more than `threshold` lines (added + removed) gets
//! flagged for review even if no semantic classifier claimed it. Large
//! changes deserve human attention by default and are too expensive to send
//! to the LLM.

use crate::classify::{Category, Classification, Classifier, Level, Source};
use crate::diff::{FileDiff, Hunk};

pub struct SizeThreshold {
    threshold: u32,
}

impl SizeThreshold {
    pub fn new(threshold: u32) -> Self {
        Self { threshold }
    }
}

impl Classifier for SizeThreshold {
    fn name(&self) -> &'static str {
        "size_threshold"
    }

    fn priority(&self) -> i32 {
        // Run last among the heuristics. Specific classifiers (Generated,
        // Lockfile, AST-based) should claim a hunk first if they apply.
        200
    }

    fn classify(&self, _file: &mut FileDiff, hunk: &Hunk) -> Option<Classification> {
        let changed = hunk.added + hunk.removed;
        if changed > self.threshold {
            Some(Classification {
                level: Level::Review,
                category: Category::LargeChange,
                rationale: format!("{} changed lines (>{} threshold)", changed, self.threshold),
                source: Source::Heuristic {
                    name: "size_threshold".into(),
                },
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

    fn hunk(added: u32, removed: u32) -> Hunk {
        Hunk {
            id: HunkId("test:1".into()),
            old_range: LineRange {
                start: 1,
                count: removed,
            },
            new_range: LineRange {
                start: 1,
                count: added,
            },
            old_lines: (0..removed).map(|i| format!("old{i}")).collect(),
            new_lines: (0..added).map(|i| format!("new{i}")).collect(),
            added,
            removed,
            added_lines: (1..=added).collect(),
            removed_lines: (1..=removed).collect(),
        }
    }

    fn file() -> FileDiff {
        FileDiff::for_test(
            PathBuf::from("src/x.rs"),
            FileStatus::Modified,
            None,
            vec![],
        )
    }

    #[test]
    fn flags_hunks_exceeding_threshold() {
        let cls = SizeThreshold::new(150);
        let result = cls.classify(&mut file(), &hunk(100, 60)).unwrap();
        assert_eq!(result.level, Level::Review);
        assert_eq!(result.category, Category::LargeChange);
        assert!(result.rationale.contains("160"));
    }

    #[test]
    fn ignores_small_hunks() {
        let cls = SizeThreshold::new(150);
        assert!(cls.classify(&mut file(), &hunk(5, 5)).is_none());
        assert!(cls.classify(&mut file(), &hunk(75, 75)).is_none());
    }

    #[test]
    fn boundary_is_strictly_greater_than() {
        let cls = SizeThreshold::new(150);
        assert!(
            cls.classify(&mut file(), &hunk(75, 75)).is_none(),
            "exactly 150 should not fire"
        );
        assert!(
            cls.classify(&mut file(), &hunk(76, 75)).is_some(),
            "151 should fire"
        );
    }

    #[test]
    fn threshold_is_configurable() {
        let cls = SizeThreshold::new(10);
        assert!(cls.classify(&mut file(), &hunk(6, 5)).is_some());
        assert!(cls.classify(&mut file(), &hunk(5, 5)).is_none());
    }
}
