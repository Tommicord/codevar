# Changelog

All notable changes to Codevar are documented here.

The format follows [Keep a Changelog][kac] and the versions follow
[Semantic Versioning][semver]. The version number itself is the single
`[workspace.package] version` in `Cargo.toml`, which every crate inherits, so a
release bumps this file at the same time as that one field.

[kac]: https://keepachangelog.com/en/1.1.0/
[semver]: https://semver.org/spec/v2.0.0.html

Unreleased work lives under `## Unreleased`. Move it into a dated release
heading when a tag is pushed; a release heading covers one version of the
workspace, not one crate, because every crate ships from the same tree and
inherits the one `workspace.package.version`.

## Unreleased

### Added

- `cargo-deny` dependency auditing (`deny.toml`) with a CI gate that also runs
  weekly, so a newly published advisory in an existing dependency fails a check
  instead of waiting for someone to touch `Cargo.lock`.
- `.github/dependabot.yml` for the workspace lockfile and the action refs.
- `.gitattributes`, which pins LF endings so a clone configured with
  `core.autocrlf=true` no longer makes `cargo fmt --check` fail across the
  whole tree.
- A criterion benchmark workflow (`.github/workflows/bench.yml`) that runs the
  existing `harness = false` bench targets on every push to `main` and keeps the
  report as an artifact; it is not a pull-request gate because shared-runner
  timings are not comparable.
- Taplo configuration (`.taplo.toml`) for manifest formatting and Lychee
  configuration (`.lychee.toml`) for link checking in the documentation.
- `.github/CODEOWNERS`, which routes review of each subsystem to the crate that
  owns it instead of pinging the whole org on every pull request.

## 1.0.0

Not yet released. The tag does not exist yet; `workspace.package.version`
already says 1.0.0, so the first release cut is what starts this file's history
below `Unreleased`.
