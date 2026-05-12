//! Classification core: types, the `Classifier` trait, and the pipeline.

use std::collections::HashSet;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::diff::{Diff, FileDiff, Hunk, HunkId, LineRange};
use crate::lang::Language;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "lowercase")]
pub enum Level {
    Skip,
    Skim,
    Review,
}

impl Level {
    /// Returns the higher-severity level. `Review > Skim > Skip`.
    pub fn max(self, other: Level) -> Level {
        if self >= other {
            self
        } else {
            other
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    // Skip categories (mechanical noise).
    Generated,
    Lockfile,
    FormatterOnly,
    CommentOnly,
    ImportReorder,
    PureRename,
    TestFixture,

    // Review categories (load-bearing).
    PublicApiChange,
    ControlFlow,
    ErrorHandlingDeleted,
    NumericalCalc,
    LargeChange,

    // LLM-assessed (any level).
    LlmAssessed,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    Heuristic { name: String },
    Llm { provider: String, model: String },
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    Old,
    New,
}

/// A contiguous line range within a hunk that triggered a finding.
/// Inclusive on both ends. `side` says whether the numbers refer to the
/// pre-image (old file) or post-image (new file).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FocusLines {
    pub start: u32,
    pub end: u32,
    pub side: Side,
}

/// A single classification verdict against a region of a hunk. One hunk may
/// produce many findings (one per matching classifier rule, plus N from the
/// LLM if it flags multiple concerns).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Finding {
    pub level: Level,
    pub category: Category,
    pub rationale: String,
    pub source: Source,
    pub focus_lines: Option<FocusLines>,
}

/// All findings produced for a single hunk. Both heuristics and the LLM
/// emit zero, one, or many findings per hunk.
#[derive(Clone, Debug)]
pub struct HunkFindings {
    pub hunk_id: HunkId,
    pub file_path: PathBuf,
    pub old_range: LineRange,
    pub new_range: LineRange,
    pub findings: Vec<Finding>,
}

/// A hunk that no heuristic claimed. Owned (not borrowed) so it can outlive
/// the `&mut Diff` borrow used during pipeline execution.
#[derive(Clone, Debug)]
pub struct Unclassified {
    pub hunk_id: HunkId,
    pub file_path: PathBuf,
    pub language: Option<Language>,
    pub old_range: LineRange,
    pub new_range: LineRange,
    pub old_lines: Vec<String>,
    pub new_lines: Vec<String>,
}

/// A single deterministic classification rule.
///
/// Returns every finding this rule sees on the hunk; an empty vec means
/// "no claim, defer to the next classifier." The pipeline collects findings
/// from **all** classifiers — there is no first-match-wins short-circuit.
pub trait Classifier: Send + Sync {
    /// Stable identifier for logging and `Source::Heuristic { name }`.
    fn name(&self) -> &'static str;

    /// Lower values run first. Path-based: 0–99. AST-based: 100–199.
    /// Size-threshold (auto-elevate): 200+.
    fn priority(&self) -> i32;

    fn classify(&self, file: &mut FileDiff, hunk: &Hunk) -> Vec<Finding>;
}

/// Inputs for the standard pipeline. Populated from `garbelour.toml` and CLI
/// flags. Kept here (rather than in `config.rs`) so this module can be tested
/// without pulling in TOML parsing.
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub generated_globs: Vec<String>,
    pub generated_paths: HashSet<PathBuf>,
    pub lockfile_names: Vec<String>,
    /// Maximum findings any one classifier may emit per hunk. Prevents
    /// pathological fanout (e.g. a 200-line refactor touching every
    /// control-flow node). Recommended default: 5.
    pub max_findings_per_classifier: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            generated_globs: Vec::new(),
            generated_paths: HashSet::new(),
            lockfile_names: Vec::new(),
            max_findings_per_classifier: 5,
        }
    }
}

pub struct Pipeline {
    classifiers: Vec<Box<dyn Classifier>>,
    max_findings_per_classifier: usize,
}

impl Pipeline {
    /// Construct a pipeline from an explicit list of classifiers. They are
    /// sorted by `priority()` so callers don't need to pre-order them.
    pub fn new(mut classifiers: Vec<Box<dyn Classifier>>) -> Self {
        classifiers.sort_by_key(|c| c.priority());
        Self {
            classifiers,
            max_findings_per_classifier: 5,
        }
    }

    /// The standard pipeline, built from the project's `PipelineConfig`.
    /// Includes path-based and AST-based classifiers. Hunks no heuristic
    /// claims go to the LLM (when `--llm` is on) or default to Review.
    pub fn standard(config: &PipelineConfig) -> anyhow::Result<Self> {
        use crate::classifiers::{
            CommentOnly, ControlFlow, ErrorHandlingDeleted, Generated, ImportReorder, Lockfile,
            NumericalCalc, PublicApi,
        };
        let classifiers: Vec<Box<dyn Classifier>> = vec![
            Box::new(Generated::new(
                config.generated_globs.clone(),
                config.generated_paths.clone(),
            )?),
            Box::new(Lockfile::new(config.lockfile_names.clone())),
            Box::new(CommentOnly::new()),
            Box::new(ImportReorder::new()),
            Box::new(PublicApi::new()),
            Box::new(ControlFlow::new()),
            Box::new(NumericalCalc::new()),
            Box::new(ErrorHandlingDeleted::new()),
        ];
        Ok(Pipeline {
            classifiers: {
                let mut cs = classifiers;
                cs.sort_by_key(|c| c.priority());
                cs
            },
            max_findings_per_classifier: config.max_findings_per_classifier.max(1),
        })
    }

    pub fn classifiers(&self) -> &[Box<dyn Classifier>] {
        &self.classifiers
    }

    /// Run all classifiers against every hunk in the diff. Every matching
    /// classifier contributes its findings; no short-circuit.
    /// Returns `(classified, unclassified)` where `classified` contains
    /// only hunks with at least one finding.
    pub fn run(&self, diff: &mut Diff) -> (Vec<HunkFindings>, Vec<Unclassified>) {
        let mut classified = Vec::new();
        let mut unclassified = Vec::new();
        for file_idx in 0..diff.files.len() {
            // Move hunks out so we can hold `&mut FileDiff` (for lazy content
            // loading inside classifiers) and an immutable reference to a
            // hunk at the same time. They live in FileDiff, so we can't
            // borrow both directly.
            let hunks = std::mem::take(&mut diff.files[file_idx].hunks);
            for hunk in &hunks {
                let file = &mut diff.files[file_idx];
                let mut findings: Vec<Finding> = Vec::new();
                for c in &self.classifiers {
                    let mut produced = c.classify(file, hunk);
                    if produced.len() > self.max_findings_per_classifier {
                        produced.truncate(self.max_findings_per_classifier);
                    }
                    findings.extend(produced);
                }
                if findings.is_empty() {
                    unclassified.push(Unclassified {
                        hunk_id: hunk.id.clone(),
                        file_path: file.path.clone(),
                        language: file.language,
                        old_range: hunk.old_range.clone(),
                        new_range: hunk.new_range.clone(),
                        old_lines: hunk.old_lines.clone(),
                        new_lines: hunk.new_lines.clone(),
                    });
                } else {
                    classified.push(HunkFindings {
                        hunk_id: hunk.id.clone(),
                        file_path: file.path.clone(),
                        old_range: hunk.old_range.clone(),
                        new_range: hunk.new_range.clone(),
                        findings,
                    });
                }
            }
            diff.files[file_idx].hunks = hunks;
        }
        (classified, unclassified)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use crate::diff::{FileStatus, LineRange};

    /// Test classifier: flags any hunk in a file whose path matches a name.
    struct AlwaysSkip {
        target: &'static str,
    }
    impl Classifier for AlwaysSkip {
        fn name(&self) -> &'static str {
            "always_skip"
        }
        fn priority(&self) -> i32 {
            0
        }
        fn classify(&self, file: &mut FileDiff, _hunk: &Hunk) -> Vec<Finding> {
            if file.path.to_string_lossy().contains(self.target) {
                vec![Finding {
                    level: Level::Skip,
                    category: Category::Lockfile,
                    rationale: "test".into(),
                    source: Source::Heuristic {
                        name: self.name().into(),
                    },
                    focus_lines: None,
                }]
            } else {
                Vec::new()
            }
        }
    }

    fn fake_hunk(start: u32) -> Hunk {
        Hunk {
            id: HunkId(format!("test:{start}")),
            old_range: LineRange { start, count: 1 },
            new_range: LineRange { start, count: 1 },
            old_lines: vec!["a".into()],
            new_lines: vec!["b".into()],
            added: 1,
            removed: 1,
            added_lines: vec![start],
            removed_lines: vec![start],
        }
    }

    fn fake_file(path: &str, hunks: Vec<Hunk>) -> FileDiff {
        crate::diff::FileDiff::for_test(PathBuf::from(path), FileStatus::Modified, None, hunks)
    }

    #[test]
    fn pipeline_classifies_matching_hunks_and_defers_others() {
        let pipeline = Pipeline::new(vec![Box::new(AlwaysSkip { target: "match" })]);
        let mut diff = Diff {
            base_sha: "0".repeat(40),
            head_sha: "0".repeat(40),
            files: vec![
                fake_file("src/match.rs", vec![fake_hunk(1), fake_hunk(10)]),
                fake_file("src/other.rs", vec![fake_hunk(5)]),
            ],
        };
        let (classified, unclassified) = pipeline.run(&mut diff);
        assert_eq!(classified.len(), 2);
        assert_eq!(unclassified.len(), 1);
        for c in &classified {
            assert!(c.file_path.to_string_lossy().contains("match"));
            assert_eq!(c.findings.len(), 1);
        }
        assert!(unclassified[0]
            .file_path
            .to_string_lossy()
            .contains("other"));
        // Hunks are restored after the run.
        assert_eq!(diff.files[0].hunks.len(), 2);
        assert_eq!(diff.files[1].hunks.len(), 1);
    }

    /// When two classifiers both fire on one hunk, both findings are kept
    /// (no first-match-wins).
    #[test]
    fn pipeline_accumulates_findings_from_every_matching_classifier() {
        struct Tag {
            name: &'static str,
            priority: i32,
            category: Category,
        }
        impl Classifier for Tag {
            fn name(&self) -> &'static str {
                self.name
            }
            fn priority(&self) -> i32 {
                self.priority
            }
            fn classify(&self, _f: &mut FileDiff, _h: &Hunk) -> Vec<Finding> {
                vec![Finding {
                    level: Level::Skip,
                    category: self.category,
                    rationale: self.name.into(),
                    source: Source::Heuristic {
                        name: self.name.into(),
                    },
                    focus_lines: None,
                }]
            }
        }
        let pipeline = Pipeline::new(vec![
            Box::new(Tag {
                name: "second",
                priority: 100,
                category: Category::Generated,
            }),
            Box::new(Tag {
                name: "first",
                priority: 0,
                category: Category::Lockfile,
            }),
        ]);
        let mut diff = Diff {
            base_sha: "0".repeat(40),
            head_sha: "0".repeat(40),
            files: vec![fake_file("src/x.rs", vec![fake_hunk(1)])],
        };
        let (classified, _) = pipeline.run(&mut diff);
        assert_eq!(classified.len(), 1);
        // Both classifiers fired; both findings collected, in priority order.
        assert_eq!(classified[0].findings.len(), 2);
        assert_eq!(classified[0].findings[0].category, Category::Lockfile);
        assert_eq!(classified[0].findings[1].category, Category::Generated);
    }

    #[test]
    fn pipeline_caps_findings_per_classifier() {
        struct Spammy;
        impl Classifier for Spammy {
            fn name(&self) -> &'static str {
                "spammy"
            }
            fn priority(&self) -> i32 {
                0
            }
            fn classify(&self, _f: &mut FileDiff, _h: &Hunk) -> Vec<Finding> {
                (0..20)
                    .map(|i| Finding {
                        level: Level::Skim,
                        category: Category::LlmAssessed,
                        rationale: format!("f{i}"),
                        source: Source::Heuristic {
                            name: "spammy".into(),
                        },
                        focus_lines: None,
                    })
                    .collect()
            }
        }
        let pipeline = Pipeline::new(vec![Box::new(Spammy)]);
        let mut diff = Diff {
            base_sha: "0".repeat(40),
            head_sha: "0".repeat(40),
            files: vec![fake_file("src/x.rs", vec![fake_hunk(1)])],
        };
        let (classified, _) = pipeline.run(&mut diff);
        assert_eq!(classified[0].findings.len(), 5);
    }

    #[test]
    fn standard_pipeline_classifies_lockfile() {
        let config = PipelineConfig::default();
        let pipeline = Pipeline::standard(&config).unwrap();
        let mut diff = Diff {
            base_sha: "0".repeat(40),
            head_sha: "0".repeat(40),
            files: vec![
                fake_file("Cargo.lock", vec![fake_hunk(1)]),
                fake_file("src/main.rs", vec![fake_hunk(1)]),
            ],
        };
        let (classified, unclassified) = pipeline.run(&mut diff);
        // Cargo.lock matches Generated (priority 0) and possibly Lockfile
        // (priority 1) — under the new accumulation rule both fire.
        assert_eq!(classified.len(), 1);
        assert_eq!(classified[0].file_path.to_string_lossy(), "Cargo.lock");
        assert!(classified[0]
            .findings
            .iter()
            .all(|f| f.level == Level::Skip));
        assert_eq!(unclassified.len(), 1);
        assert_eq!(unclassified[0].file_path.to_string_lossy(), "src/main.rs");
    }

    #[test]
    fn finding_round_trips_through_json() {
        let c = Finding {
            level: Level::Review,
            category: Category::PublicApiChange,
            rationale: "pub fn signature changed".into(),
            source: Source::Heuristic {
                name: "public_api".into(),
            },
            focus_lines: Some(FocusLines {
                start: 10,
                end: 14,
                side: Side::New,
            }),
        };
        let json = serde_json::to_string(&c).unwrap();
        let back: Finding = serde_json::from_str(&json).unwrap();
        assert_eq!(back.level, Level::Review);
        assert_eq!(back.category, Category::PublicApiChange);
        let f = back.focus_lines.unwrap();
        assert_eq!(f.start, 10);
        assert_eq!(f.end, 14);
        assert_eq!(f.side, Side::New);
    }

    #[test]
    fn level_max_orders_review_above_skim_above_skip() {
        assert_eq!(Level::Skip.max(Level::Skim), Level::Skim);
        assert_eq!(Level::Skim.max(Level::Skip), Level::Skim);
        assert_eq!(Level::Skim.max(Level::Review), Level::Review);
        assert_eq!(Level::Review.max(Level::Skip), Level::Review);
    }
}
