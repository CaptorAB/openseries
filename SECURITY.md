# Security Policy

## Reporting vulnerabilities

Report security issues privately. Do not open public GitHub issues for undisclosed
vulnerabilities.

- Organization policy:
  [CaptorAB SECURITY.md](https://github.com/CaptorAB/.github/blob/master/SECURITY.md)
- This repository: use
  [GitHub private vulnerability reporting](https://github.com/CaptorAB/openseries/security/advisories/new)
  when enabled, or contact the maintainers listed on PyPI.

## Supply chain controls in this repository

- **PyPI publishing** uses
  [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC from GitHub
  Actions), not long-lived API tokens in CI.
- **CI workflows** use hash-pinned actions, default read-only `contents`, and
  [zizmor](https://github.com/woodruffw/zizmor) audits.
- **Dependencies** are locked in `uv.lock`; CI runs `uv sync --locked` and
  supply-chain scans (`supply-chain.yml`).
- **Releases** build from the signed git tag, record `SHA256SUMS` for artifacts,
  and verify checksums before publish.

## Maintainer release checklist

1. Confirm GitHub environment protection on `release`, `testpypi`, and `pypi`
   (required reviewers).
2. Run `make audit` locally after dependency changes.
3. Run deploy workflow only from `master` with an intentional version bump in
   `pyproject.toml`.
4. After PyPI publish, verify the new version on
   [pypi.org/project/openseries](https://pypi.org/project/openseries/) and update
   conda-forge feedstock from that sdist.

## Incident response (unauthorized release)

1. **Stop**: Disable compromised workflows or revoke PyPI Trusted Publishers; do
   not publish further versions.
2. **PyPI**: Yank malicious versions; rotate/remove API tokens if any exist on the
   account.
3. **GitHub**: Rotate `GPG_PRIVATE_KEY`, `CODECOV_TOKEN`, and review audit log for
   workflow or secret changes.
4. **Notify**: GitHub Security Advisory + user-facing release/issue explaining
   affected versions and remediation.
5. **Conda-forge**: Request feedstock repodata/outdated markers for affected builds.

## Verifying installs

- Prefer installing a specific version: `pip install openseries==<version>`.
- Compare PyPI artifacts to the signed git tag and `SHA256SUMS` from the GitHub
  Actions release workflow when investigating tampering.
