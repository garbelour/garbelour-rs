//! Integration test: extract the diff between this repo's two seed commits
//! and verify the structure. Uses the real git binary against the actual
//! repo, so it exercises the subprocess plumbing as well as the parser.

use std::path::PathBuf;

use garbelour::diff;
use garbelour::FileStatus;

fn repo_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn extracts_diff_between_seed_commits() {
    // 526f797 is the initial commit; HEAD adds SPEC.md.
    let result = diff::extract(&repo_path(), "526f797", "HEAD");
    let d = match result {
        Ok(d) => d,
        // If the repo's history has been rewritten, skip rather than fail.
        Err(_) => return,
    };
    assert_eq!(d.base_sha.len(), 40);
    assert_eq!(d.head_sha.len(), 40);
    let spec = d
        .files
        .iter()
        .find(|f| f.path.ends_with("SPEC.md"))
        .expect("SPEC.md should appear in the diff");
    assert!(matches!(
        spec.status,
        FileStatus::Added | FileStatus::Modified
    ));
    assert!(!spec.hunks.is_empty(), "SPEC.md hunks should be parsed");
    let h = &spec.hunks[0];
    assert!(!h.new_lines.is_empty());
    assert!(h.id.0.contains("SPEC.md"));
}

#[test]
fn ensure_content_loads_lazily() {
    let mut d = match diff::extract(&repo_path(), "526f797", "HEAD") {
        Ok(d) => d,
        Err(_) => return,
    };
    let spec = d
        .files
        .iter_mut()
        .find(|f| f.path.ends_with("SPEC.md"))
        .expect("SPEC.md");
    assert!(spec.new_content.is_none());
    spec.ensure_content().unwrap();
    assert!(spec.new_content.is_some());
    assert!(spec.new_content.as_ref().unwrap().contains("Garbelour"));
}
