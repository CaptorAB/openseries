#!/usr/bin/env bash
set -euo pipefail

readonly ZIZMOR_VERSION=1.25.2

if [[ -z "${GH_TOKEN:-}" && -z "${GITHUB_TOKEN:-}" ]]; then
  if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
    export GH_TOKEN="$(gh auth token)"
  fi
fi

if [[ -z "${GH_TOKEN:-}" && -z "${GITHUB_TOKEN:-}" ]]; then
  echo "zizmor: no GitHub token available; running offline audits only" >&2
fi

exec uvx "zizmor==${ZIZMOR_VERSION}" --no-progress "$@"
