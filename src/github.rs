//! GitHub API plumbing: PR-event parsing and the sticky-comment upsert.
//!
//! Just enough to support the `--post-comment` flow. Two endpoints:
//!   - `GET /repos/{owner}/{repo}/issues/{n}/comments` — list comments.
//!   - `POST /repos/{owner}/{repo}/issues/{n}/comments` — create a comment.
//!   - `PATCH /repos/{owner}/{repo}/issues/comments/{id}` — update a comment.
//!
//! The sticky comment is identified by an HTML marker at the top of the
//! body (`<!-- garbelour:sticky -->`); we find or create it idempotently.

use anyhow::{anyhow, bail, Context};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::render::STICKY_MARKER;

pub struct GitHubClient {
    token: String,
    base_url: String,
}

impl GitHubClient {
    /// Build from `GITHUB_TOKEN`. Errors if the env var is missing.
    pub fn from_env() -> anyhow::Result<Self> {
        let token = std::env::var("GITHUB_TOKEN")
            .map_err(|_| anyhow!("GITHUB_TOKEN is not set"))?;
        Ok(Self {
            token,
            base_url: "https://api.github.com".to_string(),
        })
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn list_issue_comments(
        &self,
        owner: &str,
        repo: &str,
        issue: u64,
    ) -> anyhow::Result<Vec<IssueComment>> {
        let url = format!(
            "{}/repos/{}/{}/issues/{}/comments",
            self.base_url, owner, repo, issue
        );
        let resp = ureq::get(&url)
            .header("Accept", "application/vnd.github+json")
            .header("Authorization", &format!("Bearer {}", self.token))
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "garbelour")
            .call()
            .map_err(|e| anyhow!("GET {}: {}", url, e))?;
        let text = resp
            .into_body()
            .read_to_string()
            .context("reading GitHub list-comments response")?;
        let comments: Vec<IssueComment> = serde_json::from_str(&text)
            .with_context(|| format!("parsing GitHub list-comments response: {text:.300}"))?;
        Ok(comments)
    }

    pub fn create_issue_comment(
        &self,
        owner: &str,
        repo: &str,
        issue: u64,
        body: &str,
    ) -> anyhow::Result<IssueComment> {
        let url = format!(
            "{}/repos/{}/{}/issues/{}/comments",
            self.base_url, owner, repo, issue
        );
        let resp = ureq::post(&url)
            .header("Accept", "application/vnd.github+json")
            .header("Authorization", &format!("Bearer {}", self.token))
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "garbelour")
            .header("content-type", "application/json")
            .send_json(json!({ "body": body }))
            .map_err(|e| anyhow!("POST {}: {}", url, e))?;
        let text = resp
            .into_body()
            .read_to_string()
            .context("reading GitHub create-comment response")?;
        let c: IssueComment = serde_json::from_str(&text)
            .with_context(|| format!("parsing GitHub create-comment response: {text:.300}"))?;
        Ok(c)
    }

    pub fn update_issue_comment(
        &self,
        owner: &str,
        repo: &str,
        comment_id: u64,
        body: &str,
    ) -> anyhow::Result<IssueComment> {
        let url = format!(
            "{}/repos/{}/{}/issues/comments/{}",
            self.base_url, owner, repo, comment_id
        );
        let resp = ureq::patch(&url)
            .header("Accept", "application/vnd.github+json")
            .header("Authorization", &format!("Bearer {}", self.token))
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "garbelour")
            .header("content-type", "application/json")
            .send_json(json!({ "body": body }))
            .map_err(|e| anyhow!("PATCH {}: {}", url, e))?;
        let text = resp
            .into_body()
            .read_to_string()
            .context("reading GitHub update-comment response")?;
        let c: IssueComment = serde_json::from_str(&text)
            .with_context(|| format!("parsing GitHub update-comment response: {text:.300}"))?;
        Ok(c)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct IssueComment {
    pub id: u64,
    pub body: String,
}

/// Upsert: find a comment whose body contains the sticky marker; if found,
/// PATCH it; otherwise POST a new one. The body must already start with the
/// marker — the renderer enforces this.
pub fn upsert_sticky_comment(
    client: &GitHubClient,
    owner: &str,
    repo: &str,
    issue: u64,
    body: &str,
) -> anyhow::Result<IssueComment> {
    if !body.contains(STICKY_MARKER) {
        bail!("garbelour: sticky comment body must contain the marker {STICKY_MARKER:?}");
    }
    let comments = client.list_issue_comments(owner, repo, issue)?;
    if let Some(existing) = comments.iter().find(|c| c.body.contains(STICKY_MARKER)) {
        client.update_issue_comment(owner, repo, existing.id, body)
    } else {
        client.create_issue_comment(owner, repo, issue, body)
    }
}

// --- event parsing -------------------------------------------------------

/// Subset of fields garbelour reads from a GitHub Action's event payload.
#[derive(Clone, Debug)]
pub struct PrEvent {
    pub owner: String,
    pub repo: String,
    pub pr_number: u64,
    pub base_sha: String,
    pub head_sha: String,
}

/// Read `GITHUB_EVENT_PATH`, parse the JSON, and extract PR fields.
pub fn parse_event() -> anyhow::Result<PrEvent> {
    let path = std::env::var("GITHUB_EVENT_PATH")
        .map_err(|_| anyhow!("GITHUB_EVENT_PATH is not set"))?;
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("reading GITHUB_EVENT_PATH={path}"))?;
    parse_event_json(&text)
}

pub fn parse_event_json(text: &str) -> anyhow::Result<PrEvent> {
    let v: Value = serde_json::from_str(text).context("parsing GitHub event JSON")?;
    let pr = v
        .get("pull_request")
        .ok_or_else(|| anyhow!("event payload has no `pull_request` field"))?;
    let pr_number = pr
        .get("number")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("pull_request.number missing or not a number"))?;
    let base_sha = pr
        .get("base")
        .and_then(|b| b.get("sha"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("pull_request.base.sha missing"))?
        .to_string();
    let head_sha = pr
        .get("head")
        .and_then(|h| h.get("sha"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("pull_request.head.sha missing"))?
        .to_string();
    let repository = v
        .get("repository")
        .ok_or_else(|| anyhow!("event payload has no `repository` field"))?;
    let owner = repository
        .get("owner")
        .and_then(|o| o.get("login"))
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("repository.owner.login missing"))?
        .to_string();
    let repo = repository
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("repository.name missing"))?
        .to_string();
    Ok(PrEvent {
        owner,
        repo,
        pr_number,
        base_sha,
        head_sha,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_event_json_extracts_all_fields() {
        let json = r#"{
            "pull_request": {
                "number": 42,
                "base": {"sha": "0123456789abcdef0123456789abcdef01234567"},
                "head": {"sha": "abcdef0123456789abcdef0123456789abcdef01"}
            },
            "repository": {
                "name": "widget",
                "owner": {"login": "acme"}
            }
        }"#;
        let event = parse_event_json(json).unwrap();
        assert_eq!(event.owner, "acme");
        assert_eq!(event.repo, "widget");
        assert_eq!(event.pr_number, 42);
        assert_eq!(event.base_sha.len(), 40);
        assert_eq!(event.head_sha.len(), 40);
    }

    #[test]
    fn parse_event_json_errors_on_missing_pr() {
        let json = r#"{"repository": {"name": "x", "owner": {"login": "y"}}}"#;
        assert!(parse_event_json(json).is_err());
    }

    #[test]
    fn parse_event_json_errors_on_missing_base_sha() {
        let json = r#"{
            "pull_request": {"number": 1, "head": {"sha": "x"}},
            "repository": {"name": "r", "owner": {"login": "o"}}
        }"#;
        assert!(parse_event_json(json).is_err());
    }

    #[test]
    fn upsert_requires_sticky_marker() {
        // We don't construct a real client here — just exercise the body check.
        let client = GitHubClient {
            token: "x".into(),
            base_url: "http://localhost:0".into(),
        };
        let result = upsert_sticky_comment(&client, "o", "r", 1, "no marker here");
        assert!(result.is_err());
    }
}
