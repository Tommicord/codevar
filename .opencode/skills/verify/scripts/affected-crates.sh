#!/usr/bin/env bash
# Run scoped cargo checks for the crates touched by the working tree.
#
# Usage:
#   affected-crates.sh              # cargo check + clippy + test for affected crates
#   affected-crates.sh --list       # print affected crates only
#   affected-crates.sh --check      # cargo check only
#   affected-crates.sh --clippy     # cargo clippy only
#   affected-crates.sh --test       # cargo test only
#   affected-crates.sh --workspace  # run the full workspace instead of scoped
#
# Exit codes: 0 all requested checks passed, 1 a check failed, 2 no crates found.
set -uo pipefail

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "affected-crates: not inside a git repository" >&2
  exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root" || exit 2

run_check=1
run_clippy=1
run_test=1
list_only=0
use_workspace=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)      list_only=1; shift ;;
    --check)     run_clippy=0; run_test=0; shift ;;
    --clippy)    run_check=0; run_test=0; shift ;;
    --test)      run_check=0; run_clippy=0; shift ;;
    --workspace) use_workspace=1; shift ;;
    -h|--help)   sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "affected-crates: unknown argument '$1'" >&2; exit 2 ;;
  esac
done

# --- which crates changed? --------------------------------------------------
# Include staged, unstaged and untracked changes so nothing is missed.
mapfile -t changed < <(
  {
    git diff --name-only --diff-filter=ACMR HEAD
    git ls-files --others --exclude-standard
  } 2>/dev/null | sort -u
)

crates=()
workspace_files=0
for path in "${changed[@]:-}"; do
  [[ -z "$path" ]] && continue
  case "$path" in
    crates/*)
      crate=${path#crates/}
      crate=${crate%%/*}
      crates+=("$crate")
      ;;
    *)
      workspace_files=1
      ;;
  esac
done

if [[ ${#crates[@]} -eq 0 ]]; then
  if (( workspace_files )); then
    echo "affected-crates: only workspace-level files changed (Cargo.toml, CI, docs)."
    echo "affected-crates: run the full workspace check before declaring the task done."
  else
    echo "affected-crates: no changes detected." >&2
    exit 2
  fi
  crates=()
fi

mapfile -t crates < <(printf '%s\n' "${crates[@]:-}" | sort -u | grep -v '^$' || true)

if (( ${#crates[@]} == 0 )); then
  if (( use_workspace )); then
    echo "affected-crates: running full workspace checks."
  else
    echo "affected-crates: no crates affected; nothing scoped to run." >&2
    exit 2
  fi
fi

# --- build the package selection --------------------------------------------
if (( use_workspace )); then
  pkg_args=()
  label="--workspace"
else
  pkg_args=()
  for c in "${crates[@]}"; do
    pkg_args+=(-p "$c")
  done
  label="${crates[*]}"
fi

echo "affected-crates: scope = $label"

if (( list_only )); then
  printf '%s\n' "${crates[@]:-}"
  exit 0
fi

status=0

if (( run_check )); then
  echo "--- cargo check ---"
  cargo check "${pkg_args[@]}" || status=1
fi

if (( run_clippy )); then
  echo "--- cargo clippy ---"
  # shellcheck disable=SC2086
  cargo clippy "${pkg_args[@]}" --all-targets -- -D warnings || status=1
fi

if (( run_test )); then
  echo "--- cargo test ---"
  cargo test "${pkg_args[@]}" || status=1
fi

if (( status == 0 )); then
  echo "affected-crates: all checks passed."
else
  echo "affected-crates: FAILURES above." >&2
fi
exit "$status"
