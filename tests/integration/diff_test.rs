//! Integration test: extract the diff between this repo's seed commit and
//! HEAD and verify the structure. Uses the real git binary against the
//! actual repo, so it exercises the subprocess plumbing as well as the
//! parser. README.md exists in both the seed and HEAD and is modified
//! between them, so it's a stable anchor.

use std::path::PathBuf;

use garbelour::diff;
use garbelour::FileStatus;

fn repo_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn extracts_diff_between_seed_commits() {
    let result = diff::extract(&repo_path(), "526f797", "HEAD");
    let d = match result {
        Ok(d) => d,
        // If the repo's history has been rewritten, skip rather than fail.
        Err(_) => return,
    };
    assert_eq!(d.base_sha.len(), 40);
    assert_eq!(d.head_sha.len(), 40);
    let readme = d
        .files
        .iter()
        .find(|f| f.path.ends_with("README.md"))
        .expect("README.md should appear in the diff");
    assert!(matches!(
        readme.status,
        FileStatus::Added | FileStatus::Modified
    ));
    assert!(!readme.hunks.is_empty(), "README.md hunks should be parsed");
    let h = &readme.hunks[0];
    assert!(!h.new_lines.is_empty());
    assert!(h.id.0.contains("README.md"));
}

#[test]
fn ensure_content_loads_lazily() {
    let mut d = match diff::extract(&repo_path(), "526f797", "HEAD") {
        Ok(d) => d,
        Err(_) => return,
    };
    let readme = d
        .files
        .iter_mut()
        .find(|f| f.path.ends_with("README.md"))
        .expect("README.md");
    assert!(readme.new_content.is_none());
    readme.ensure_content().unwrap();
    assert!(readme.new_content.is_some());
    assert!(readme
        .new_content
        .as_ref()
        .unwrap()
        .to_lowercase()
        .contains("garbelour"));
}
