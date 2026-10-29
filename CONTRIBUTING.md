# Contributing to Ask the Map

Thank you for your interest in contributing! This document explains how the project is set up for development,
what every change must pass, and how to make a pull request.

## Table of Contents

- [Requirements for Acceptable Contributions](#requirements-for-acceptable-contributions)
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Pre-commit Hooks](#pre-commit-hooks)
- [Code Complexity Checks](#code-complexity-checks)
- [Security](#security)
- [Making Changes](#making-changes)
- [Commit Guidelines](#commit-guidelines)
- [Testing](#testing)
- [Notebooks](#notebooks)
- [GitHub Actions (CI)](#github-actions-ci)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)
- [Code Style](#code-style)

## Requirements for Acceptable Contributions

All contributions must meet these requirements before they are accepted:

1. **Pre-commit hooks pass.** Install and run `pre-commit` before opening a pull request (see
   [Pre-commit Hooks](#pre-commit-hooks)). This is mandatory. The hooks enforce formatting, linting, type
   checking, security scanning, secret detection and the complexity limits.
2. **Coding standards.** Python code must conform to the toolchain:
   - **Formatting and import order:** `ruff format` and `ruff check` (line length 110)
   - **Type annotations:** `mypy` (`--ignore-missing-imports`) on `src/`
   - **Security:** `bandit` (medium and high severity findings fail the build)
   - **Complexity:** cyclomatic complexity ≤ 20, cognitive complexity ≤ 20 and maintainability index ≥ 20 per
     file, enforced by `scripts/quality/code_metrics.py`
3. **Tests.** New features and functions must include unit tests; bug fixes should include a regression test.
   `uv run pytest` must pass.
4. **No secrets or credentials.** `detect-secrets` and `gitleaks` run on every commit. Never commit API keys,
   tokens, project IDs or `.env`.
5. **No suppressions.** Do not add `# noqa`, `# type: ignore`, `# nosec` or exclusions to pass a check: fix the
   code instead.
6. **Commit messages** follow [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`,
   `docs:`, `refactor:`, `test:`, `chore:`, `ci:`).

If any of these are not met, CI fails and the pull request is not merged until they are.

## Code of Conduct

By participating in this project you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md). Please read it
before contributing.

## Getting Started

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- Git and a GitHub account
- For the data pipeline, the accounts described in the README ("Accounts and credentials")

### Fork and Clone

1. Fork the repository on GitHub.
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR-USERNAME/ask-the-map-demo.git
   cd ask-the-map-demo
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/dividor/ask-the-map-demo.git
   ```

## Development Setup

```bash
uv sync                     # creates .venv with the project and the development tools
cp .env.example .env        # fill in ASKMAP_DATA_ROOT, EE_PROJECT and HF_HOME
uv run pre-commit install   # the hooks now run on every commit
uv run pytest               # the unit tests need no data or credentials
```

The unit tests run without any data, credentials or network. The notebooks and the demo need the set-up in the
README.

## Pre-commit Hooks

We use pre-commit hooks to keep the code consistent. **This is required for all contributors**, and it mirrors
the CI `code-quality` job. Code metrics run as part of pre-commit and fail if any file is rated **bad** (see
[Code Complexity Checks](#code-complexity-checks)).

Much of the tooling here exists because automated checks, including complexity limits, are the only reliable way
to keep a research codebase readable as it grows. Please keep the hooks on.

### Install Pre-commit

```bash
uv sync                     # pre-commit is a development dependency
uv run pre-commit install   # installs the git hook
```

### What Pre-commit Does

Our configuration (`.pre-commit-config.yaml`) automatically:

- **Formats and lints:** runs `ruff format` and `ruff check --fix` (formatting, import order, lint rules)
- **Type checks:** runs `mypy` on `src/`
- **Scans for security issues:** runs `bandit` on `src/` and `scripts/`
- **Detects secrets:** runs `detect-secrets` (against `.secrets.baseline`) and `gitleaks`
- **Guards against debug code:** fails on `pdb` and similar debug statements
- **File hygiene:** trailing whitespace, end-of-file newlines, mixed line endings
- **Validates configuration:** YAML, JSON and TOML syntax
- **Blocks merge-conflict markers and files over 2 MB**
- **Code metrics gate:** fails if any file is **bad** for cyclomatic complexity, cognitive complexity or
  maintainability index

The mypy, bandit and code-metrics hooks run through `uv run`, so they use the same package versions as the tests.

### Running Pre-commit Manually

```bash
uv run pre-commit run --all-files     # everything, on every file
uv run pre-commit run                 # staged files only
uv run pre-commit autoupdate          # update the hook versions
```

Do not use `git commit --no-verify`. If a hook fails, fix the cause.

## Code Complexity Checks

`scripts/quality/code_metrics.py` measures, for every Python file under `src/` and `tests/`:

- **cyclomatic complexity** (lizard), the number of independent paths through each function;
- **cognitive complexity** (the `cognitive-complexity` package), how hard a function is to follow;
- **maintainability index** (radon).

```bash
uv run python scripts/quality/code_metrics.py                        # the report
uv run python scripts/quality/code_metrics.py --fail-on-bad          # what pre-commit and CI run
uv run python scripts/quality/code_metrics.py --format json --output metrics.json
```

| Metric | Good | Okay | Bad (fails) |
|---|---|---|---|
| Cyclomatic (per function, file maximum) | 1–10 | 11–20 | 21 and over |
| Cognitive (per function, file maximum) | 0–10 | 11–20 | 21 and over |
| Maintainability index (per file) | ≥ 20 | 10–19 | under 10 |

When a function is rated bad, split it into smaller named steps rather than adding an exclusion. The
`--skip-js-cognitive` flag only matters for JavaScript files; the demo page's script is small and is not measured.

## Security

See [SECURITY.md](SECURITY.md) for the scanning tools, the credentials this project uses and how to report a
vulnerability. In short:

- **Bandit**, **detect-secrets** and **gitleaks** run on every commit;
- CI also runs **pip-audit** on the locked dependencies and **gitleaks** over the whole history;
- **Dependabot** opens monthly update pull requests for Python dependencies and GitHub Actions.

If `detect-secrets` flags a value that is not a secret (for example a base64 image in `site/data/`), audit it
into the baseline with `uv run detect-secrets audit .secrets.baseline` rather than excluding the file.

## Making Changes

### Development Workflow

1. **Create a feature branch:**
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Make your changes**, following the [Code Style](#code-style), with tests for new behaviour and
   documentation updates where needed.
3. **Run the checks:** `uv run pre-commit run --all-files` and `uv run pytest`.
4. **Commit** (see [Commit Guidelines](#commit-guidelines)).
5. **Push and open a pull request:**
   ```bash
   git push origin feat/your-feature-name
   ```

### Types of Contributions

- **Bug fixes**
- **Features:** new question types, land classes, study areas, methods
- **Documentation:** the README, the set-up guides, docstrings
- **Tests**
- **Performance**
- **Refactoring** without changing results

A change to how the methods score (the land maps, the plan executor, the climate tests) must come with the
notebook that measures its effect; the numbers in the README come from those notebooks.

## Commit Guidelines

We use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`, `ci`, `build`.

Examples:

```
feat: add a cotton class to the land maps
fix: distance rules ignore pixels outside the tile
docs: Earth Engine registration steps
test: the planner's plan validation
```

Keep commits atomic (one logical change each), write them in the imperative mood ("add", not "added"), and
reference issues where they apply (`fixes #12`).

## Testing

```bash
uv run pytest                      # all unit tests (about 15 seconds, no data needed)
uv run pytest tests/test_plan.py   # one file
uv run pytest -k "distance"        # by name
```

Tests live in `tests/` and exercise the real code in `src/askmap`; they build small synthetic tiles and NASA
POWER grids rather than reading data. The demo engine and server have tests of their own
(`tests/test_demo_*.py`). Do not skip or disable tests to make a change pass.

### Manual verification of the demo

When a change affects what the page shows, start the demo (`uv run python -m askmap.demo.serve`), run the page
examples, and check the Match information window for a tile. For a change to the answers themselves, run the
demo checks (notebooks 16, 22, 24 and 28), which score the answers against the official maps.

## Notebooks

The notebooks are the pipeline and its record. Conventions:

- They are numbered in the order the work was done and run in the order given in `scripts/run_pipeline.sh`.
- Each notebook keeps its outputs when committed; `jupyter nbconvert --execute --inplace` is how they are run.
- Pass marks for a check are written in the notebook's first cell before it is run.
- Heavy work (downloads, training, scoring) belongs in `src/askmap`, with unit tests; a notebook calls it and
  shows the result.
- A notebook never falls back silently: if an input is missing or a check fails, it raises.

## GitHub Actions (CI)

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

- **code-quality:** `uv sync --locked`, then all the pre-commit hooks on every file;
- **tests:** `uv run pytest`;
- **security-scan:** `pip-audit` on the locked dependencies, `bandit`, and `gitleaks` over the history.

Dependabot (`.github/dependabot.yml`) opens monthly pull requests for Python dependencies and GitHub Actions.

## Pull Request Process

### Before Submitting

1. **Update your branch:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```
2. **Run all checks:** `uv run pre-commit run --all-files` and `uv run pytest`.
3. **Update documentation** if the change affects set-up, commands or results.

### PR Guidelines

- **Title:** conventional-commit format
- **Description:** what changed and why
- **Link issues** the change relates to
- **Screenshots** for page changes
- **Results:** for a change to the methods, the notebook numbers before and after

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Other: ___

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed (demo page, notebooks)

## Checklist
- [ ] Pre-commit hooks pass
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

### Review Process

1. CI runs the checks and tests.
2. A maintainer reviews the code.
3. Requested changes are addressed.
4. A maintainer approves and merges.

## Issue Guidelines

### Before Creating an Issue

1. Search existing issues.
2. Check the README and `docs/`.
3. Test with the latest `main`.

### Bug Reports

Include the environment (OS, Python version, `uv --version`), the steps to reproduce, the expected and actual
behaviour, the full error message, and, for a notebook, which cell failed.

### Feature Requests

Include a clear description, the use case, a proposed approach if you have one, and alternatives considered.

## Code Style

- **Formatter and linter:** ruff (line length 110), configured in `pyproject.toml`
- **Type hints** on function signatures; `mypy` must pass
- **Docstrings** on public functions and classes, in plain English
- **Fail loudly:** no silent fallbacks or default substitutions; raise with a message that names the fix
- **Secrets** come from `.env` or the Hugging Face login, never from code, and are never printed
- **Plain language** in names, comments and messages

Thank you for contributing!
