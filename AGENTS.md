# Codex Agent Configuration

## Environment setup
- Before any Python-related command (linting, tests, scripts), run `source source_me` from the repo root to set `PYTHONPATH` and activate the local environment; repeat in new shells. If `source_me` is missing, continue normally.
- Prefer absolute project paths when invoking tools.
- Do not modify `Makefile` for environment activation.

## Implementation notes
- Do not add extra commentary when implementing code changes.
- Keep `Makefile` (macOS/Linux) and `make.ps1` (Windows) in sync when either is modified.

## Post-change checks
- After code changes, run `make lint` to satisfy ruff and mypy, and `make test` to ensure tests pass with full coverage.

These rules are mirrored from `.cursorrules`; keep both files in sync when updating project guidance.
