# Security Policy

## Reporting a vulnerability

If you find a security problem in this project, please report it responsibly:

1. **Do not** open a public GitHub issue for it.
2. Open a [private security advisory](https://github.com/dividor/ask-the-map-demo/security/advisories/new) on
   this repository, with steps to reproduce.
3. Allow a reasonable time for a fix before public disclosure.

## What is scanned

Every commit runs, through pre-commit hooks (`.pre-commit-config.yaml`) and again in CI
(`.github/workflows/ci.yml`):

| Tool | Purpose |
|---|---|
| **Bandit** | Python static analysis; MEDIUM and HIGH severity findings fail the check |
| **detect-secrets** | Blocks credentials being committed (`.secrets.baseline` lists the audited false positives) |
| **Gitleaks** | Pattern-based secret scanning, over the whole history in CI |
| **pip-audit** (CI) | Known vulnerabilities in the locked Python dependencies |
| **Dependabot** | Monthly dependency and GitHub Actions updates |

## Credentials this project uses

- **Earth Engine:** Google Application Default Credentials created by `gcloud auth application-default login`.
  They stay in your gcloud configuration folder. The project ID is read from `EE_PROJECT` in `.env` and is never
  printed.
- **Hugging Face:** the token from `hf auth login` (`~/.cache/huggingface/token`) or `HF_TOKEN`. It is used only
  to call the language model and to download the text encoders, and is never written anywhere in this repository.
- **NASA POWER, OpenStreetMap, USGS, USDA, US Census:** public downloads, no credentials.

`.env` is ignored by git. Saved language-model replies in `outputs/llm/` contain the prompts and answers only.

## The demo server

`askmap.demo.serve` binds to `127.0.0.1` only. It is a research demo with no authentication and is not meant to
be exposed to the internet as it stands.
