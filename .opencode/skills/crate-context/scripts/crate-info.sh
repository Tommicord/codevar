#!/usr/bin/env bash
# Print a context brief for a Codevar crate: purpose, size, dependencies,
# public API surface, unsafe hotspots, and tests.
#
# Usage:
#   crate-info.sh                 # list every crate with size and dep count
#   crate-info.sh codevar-tls     # brief for one crate
#
# Exit codes: 0 ok, 1 unknown crate, 2 not in the repository.
set -uo pipefail

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "crate-info: not inside a git repository" >&2
  exit 2
fi

repo_root=$(git rev-parse --show-toplevel)
cd "$repo_root" || exit 2

if ! command -v jq >/dev/null 2>&1; then
  echo "crate-info: jq is required" >&2
  exit 2
fi

meta=$(cargo metadata --format-version 1 2>/dev/null)
if [[ -z "$meta" ]]; then
  echo "crate-info: 'cargo metadata' failed" >&2
  exit 2
fi

# Count matches in a file; prints 0 and succeeds even when there are none.
count_in() {
  local pattern=$1 file=$2 n
  n=$(grep -cE "$pattern" "$file" 2>/dev/null)
  printf '%s' "${n:-0}"
}

# ---------------------------------------------------------------- list mode --
if [[ $# -eq 0 ]]; then
  printf '%-42s %8s %6s %6s\n' "CRATE" "CODE" "FILES" "DEPS"
  jq -r '.workspace_members[]' <<<"$meta" | while read -r member; do
    name=$(jq -r --arg id "$member" '.packages[] | select(.id==$id) | .name' <<<"$meta")
    manifest=$(jq -r --arg id "$member" '.packages[] | select(.id==$id) | .manifest_path' <<<"$meta")
    [[ -n "$name" && "$name" != "null" && -n "$manifest" ]] || continue
    dir=${manifest%Cargo.toml}
    dir=${dir%/}
    files=$(find "$dir/src" -name '*.rs' 2>/dev/null | wc -l | tr -d ' ')
    code=$(find "$dir/src" -name '*.rs' -exec cat {} + 2>/dev/null | grep -cvE '^\s*(//.*)?$')
    deps=$(jq -r --arg id "$member" '
      .packages[] | select(.id==$id) | .dependencies
      | map(select(.name | startswith("codevar-"))) | length
    ' <<<"$meta")
    printf '%-42s %8s %6s %6s\n' "$name" "${code:-0}" "${files:-0}" "${deps:-0}"
  done | sort -k2 -rn
  exit 0
fi

# ---------------------------------------------------------------- brief mode --
crate=$1
entry=$(jq -r --arg n "$crate" '.packages[] | select(.name==$n) | .id' <<<"$meta")
if [[ -z "$entry" || "$entry" == "null" ]]; then
  echo "crate-info: no crate named '$crate'" >&2
  echo "crate-info: run without arguments to list every crate." >&2
  exit 1
fi

manifest=$(jq -r --arg id "$entry" '.packages[] | select(.id==$id) | .manifest_path' <<<"$meta")
dir=${manifest%Cargo.toml}
dir=${dir%/}
name=$(jq -r --arg id "$entry" '.packages[] | select(.id==$id) | .name' <<<"$meta")
desc=$(jq -r --arg id "$entry" '.packages[] | select(.id==$id) | .description // empty' <<<"$meta")

echo "=== $name ==="
echo "path      : ${dir#"$repo_root"/}"
echo "purpose   : ${desc:-<none>}"
echo

echo "-- workspace dependencies (this crate uses) --"
jq -r --arg id "$entry" '
  .packages[] | select(.id==$id) | .dependencies
  | map(.name) | map(select(startswith("codevar-"))) | unique | .[]
' <<<"$meta" | sed 's/^/  /'
echo

echo "-- workspace dependents (who uses this crate) --"
jq -r --arg n "$name" '
  . as $m
  | $m.workspace_members as $wm
  | $m.packages[]
  | select(.id as $id | $wm | index($id))
  | select(.dependencies | map(.name) | index($n))
  | .name
' <<<"$meta" | sort -u | sed 's/^/  /'
echo

echo "-- files (code lines, largest first) --"
find "$dir/src" -name '*.rs' 2>/dev/null | while read -r f; do
  c=$(grep -cvE '^\s*(//.*)?$' "$f" 2>/dev/null)
  printf '  %6s  %s\n' "${c:-0}" "${f#"$repo_root"/}"
done | sort -rn | head -20
echo

echo "-- public API surface --"
pubfn=$(grep -rhoE '^[[:space:]]*pub[[:space:]]+(const[[:space:]]+|async[[:space:]]+|unsafe[[:space:]]+|extern[[:space:]]+"[^"]+"[[:space:]]+)*fn[[:space:]]' "$dir/src" 2>/dev/null | wc -l | tr -d ' ')
pubty=$(grep -rhoE '^[[:space:]]*pub[[:space:]]+(struct|enum|trait)[[:space:]]' "$dir/src" 2>/dev/null | wc -l | tr -d ' ')
echo "  pub fn        : $pubfn"
echo "  pub types     : $pubty"
echo

echo "-- unsafe hotspots (blocks / unsafe fn) --"
find "$dir/src" -name '*.rs' 2>/dev/null | while read -r f; do
  b=$(count_in 'unsafe[[:space:]]*\{' "$f")
  u=$(count_in '(^|[^_[:alnum:]])unsafe[[:space:]]+fn[[:space:]]' "$f")
  if (( b + u > 0 )); then
    printf '%s\t%s\t%s\n' "$((b + u))" "$b" "${f#"$repo_root"/}"
  fi
done | sort -rn | head -15 | while IFS=$'\t' read -r tot b f; do
  printf '  total=%-4s blocks=%-4s %s\n' "$tot" "$b" "$f"
done
echo

echo "-- tests --"
tfiles=$(find "$dir/tests" -name '*.rs' 2>/dev/null | wc -l | tr -d ' ')
tfn=$(grep -rhoE '#\[test\]' "$dir/src" "$dir/tests" 2>/dev/null | wc -l | tr -d ' ')
echo "  integration test files : $tfiles"
echo "  #[test] functions      : $tfn"
