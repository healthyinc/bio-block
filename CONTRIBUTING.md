## Contributing to bio-block

Thanks for your interest in contributing! This document explains how to set up the project locally, branch and commit rules, PR expectations, testing instructions for both backends, and formatting/coding style guidance.

If anything below is unclear, open an issue or ask in a PR and we'll help you get set up.

## Repository layout (assumptions)

- Frontend (React prototype): `prototype/`
- JavaScript backend (Node): `javascript_backend/`
- Python backend: `python_backend/`

If your local layout differs, adapt the commands below. These paths reflect the repository structure in the project root.

## 1) Local setup

Prerequisites

- Node.js (LTS) and npm
- Python 3.8+ and pip
- (Optional) virtualenv / venv for Python

Frontend (prototype)

1. cd into the frontend: `cd prototype`
2. Install dependencies: `npm install`
3. Run the dev server: `npm start`

JavaScript backend

1. cd into the JS backend: `cd javascript_backend`
2. Install dependencies: `npm install`
3. Start server: `node server.js` or `npm start` (if `package.json` defines it)

Python backend

1. cd into the Python backend: `cd python_backend`
2. Create & activate a venv:
   - Windows PowerShell:
     ```powershell
     python -m venv .venv; .\.venv\Scripts\Activate.ps1
     ```
3. Install dependencies: `pip install -r requirements.txt`
4. Start the backend: `python main.py` (or other entrypoint defined in the directory)

Notes / assumptions

- The repo provides `prototype/`, `javascript_backend/`, and `python_backend/`. If an entrypoint differs (e.g., `uvicorn` or a different script), use that command. If dependencies are missing from `requirements.txt` or `package.json`, open an issue so we can update the docs.

## 2) Branch naming and commit conventions

Branch naming

- feature/{short-description} — new features
- fix/{short-description} — bug fixes
- chore/{short-description} — maintenance, tooling, or build changes
- docs/{short-description} — documentation changes
- test/{short-description} — tests only

Examples: `feature/upload-streaming-encryption`, `fix/ipfs-endpoint`.

Commit messages
We follow a Conventional Commits style. Format:

type(scope): short description

Where `type` is one of: `feat`, `fix`, `chore`, `docs`, `test`, `refactor`, `style`, `perf`.

Example:

feat(api): add anonymize endpoint to JS backend

Commit body (optional) can include a longer description and a `BREAKING CHANGE:` section if needed.

## 3) Pull Request guidelines

Before opening a PR

- Make sure your branch name and commit messages follow the rules above.
- Make your changes in a branch, not `main`.
- Run relevant tests and formatters locally.

PR checklist (fill before requesting review)

- [ ] PR targets the `main` branch (or as directed by maintainers)
- [ ] Linked issue (if applicable) — include `Closes #<issue-number>` in the PR description
- [ ] Clear description of what changed and why
- [ ] How to test locally (steps) and expected behavior
- [ ] Unit/integration tests added for new functionality (if applicable)
- [ ] Code formatted and linted

When opening the PR

- Reference any related issues with `Fixes #N` (this will auto-close the issue once merged).
- Add screenshots or logs for UI or runtime changes when helpful.
- Add maintainers or reviewers using GitHub reviewers and mention them in the description when needed.

## 4) Testing

JavaScript backend tests

- Path: `javascript_backend/tests/`
- Run (from repo root):
  ```powershell
  cd javascript_backend; npm test
  ```
- Tests use the project test runner defined in `package.json` (likely Jest). If `npm test` errors due to missing node_modules, run `npm install` first.

Python backend tests

- Path: `python_backend/tests/`
- Run (from repo root):
  ```powershell
  cd python_backend; python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt; pytest -q
  ```

If tests require external services (IPFS, DB, etc.), the repository README or test docs should mention mocks or test fixtures. If tests are flaky or require credentials, open an issue so we can improve CI.

## 5) Coding style and formatting

JavaScript/TypeScript

- Follow consistent style using ESLint and Prettier where possible.
- Recommended (local checks):
  - `npx eslint .` in the package directory
  - `npx prettier --check .`

Python

- Follow Black for formatting, isort for imports, and flake8 for linting where practical.
- Recommended commands:
  ```powershell
  # format
  black .
  # check imports
  isort . --check-only
  # lint
  flake8
  ```

General

- Keep functions small, prefer descriptive names, and add docstrings/comments for public functions and modules.
- Write tests for new behavior and bug fixes.

## 6) CI and checks

Pull requests should pass automated checks (tests/lint/format) before merging. If CI fails for reasons unrelated to your change, notify maintainers.

## 7) Security and sensitive data

- Never commit secrets, API keys, or credentials. Use environment variables, GitHub Secrets, or a secure provider.

## 8) Communication and support

- For onboarding questions, open an issue and tag maintainers. Provide OS and tool versions when asking setup questions.

Thanks for contributing — we appreciate your help in making this project better!
