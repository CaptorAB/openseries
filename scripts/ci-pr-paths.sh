#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <glob>..." >&2
  exit 2
fi

write_output() {
  local value="$1"
  if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
    echo "run=${value}" >> "${GITHUB_OUTPUT}"
  fi
  echo "run=${value}"
}

event="${GITHUB_EVENT_NAME:-}"
if [[ "${event}" != "pull_request" ]]; then
  write_output true
  exit 0
fi

if [[ -n "${CI_PR_PATHS_FILES:-}" ]]; then
  files="${CI_PR_PATHS_FILES}"
else
  token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
  repo="${GITHUB_REPOSITORY:-}"
  pr="${PR_NUMBER:-}"
  if [[ -z "${token}" || -z "${repo}" || -z "${pr}" ]]; then
    echo "ci-pr-paths: missing token or PR context; running job" >&2
    write_output true
    exit 0
  fi
  export GH_TOKEN="${token}"
  files="$(gh api --paginate "repos/${repo}/pulls/${pr}/files" --jq '.[].filename')" || {
    echo "ci-pr-paths: failed to list PR files; running job" >&2
    write_output true
    exit 0
  }
fi

while IFS= read -r file; do
  [[ -z "${file}" ]] && continue
  for pattern in "$@"; do
    if [[ "${file}" == ${pattern} ]]; then
      write_output true
      exit 0
    fi
  done
done <<< "${files}"

write_output false
