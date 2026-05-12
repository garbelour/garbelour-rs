//! `garbelour.toml` loading.
//!
//! All fields optional. Missing file → defaults. Section structs are named
//! `Classify` / `Llm` / `Github` (without "Config" suffix) to avoid
//! shadowing the runtime `llm::LlmConfig`.

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Llm {
    pub provider: Option<String>,
    pub model: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Github {
    pub base_url: Option<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Classify {
    #[serde(default)]
    pub generated_globs: Vec<String>,
    #[serde(default)]
    pub lockfile_names: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub classify: Classify,
    #[serde(default)]
    pub llm: Llm,
    #[serde(default)]
    pub github: Github,
}

impl Config {
    /// Load a config from `path`. If the file is missing, return defaults.
    /// Parse errors are propagated.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        if !path.exists() {
            return Ok(Config::default());
        }
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
    }
}

//------------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn missing_file_yields_defaults() {
        let cfg = Config::load(Path::new("/nonexistent/garbelour.toml")).unwrap();
        assert!(cfg.classify.generated_globs.is_empty());
        assert!(cfg.llm.provider.is_none());
    }

    #[test]
    fn parses_full_toml() {
        let text = r#"
            [classify]
            generated_globs = ["generated/**", "*.auto.ts"]
            lockfile_names = ["shrinkwrap.json"]

            [llm]
            provider = "anthropic"
            model = "claude-haiku-4-5"

            [github]
            base_url = "https://github.example.com/api/v3"
        "#;
        let cfg: Config = toml::from_str(text).unwrap();
        assert_eq!(cfg.classify.generated_globs.len(), 2);
        assert_eq!(cfg.llm.provider.as_deref(), Some("anthropic"));
        assert_eq!(
            cfg.github.base_url.as_deref(),
            Some("https://github.example.com/api/v3")
        );
    }

    #[test]
    fn parses_partial_toml() {
        let text = r#"
            [classify]
            generated_globs = ["foo/**"]
        "#;
        let cfg: Config = toml::from_str(text).unwrap();
        assert_eq!(cfg.classify.generated_globs.len(), 1);
        assert!(cfg.llm.provider.is_none());
    }

    #[test]
    fn returns_default_for_empty_file() {
        let dir = std::env::temp_dir().join(format!("garbelour-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path: PathBuf = dir.join("garbelour.toml");
        std::fs::write(&path, "").unwrap();
        let cfg = Config::load(&path).unwrap();
        assert!(cfg.classify.generated_globs.is_empty());
    }
}
