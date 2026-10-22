#!/usr/bin/env bash
# Validate a Codevar change against the commit budget.
#
# Budget (see AGENTS.md "Commit discipline" and the commit-discipline skill):
#   <= 800 added lines (Cargo.lock excluded)
#   <= 3 files changed (<= 20 for a mechanical single-crate sweep)
#   1 crate per commit
#
# What is measured:
#   By default the working tree against HEAD, limited to the given pathspec.
#   That is what `git commit --only <paths>` would produce, and it also
#   counts files that are only marked intent-to-add in the index.
#   With --staged, the index against HEAD is measured instead (what a plain
#   `git commit` would produce).
#
# Usage:
#   commit-check.sh                                  # whole tree vs HEAD
#   commit-check.sh --staged                         # index vs HEAD
#   commit-check.sh -- .opencode/skills/verify       # only these paths
#   commit-check.sh --message "scope: summary"       # also validate the message
#   commit-check.sh --allow-sweep -- crates/foo      # permit a mechanical sweep
#
# Exit codes:
#   0  within budget and the message (if given) is well formed
#   1  over budget, or the message is malformed
#   2  nothing to check, or not inside a git repository
set -uo pipefail

MAX_LINES=${MAX_LINES:-800}
MAX_FILES=${MAX_FILES:-3}
MAX_FILES_SWEEP=${MAX_FILES_SWEEP:-20}

message=""
allow_sweep=0
staged=0
paths=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --message)
      [[ $# -ge 2 ]] || { echo "commit-check: --message needs a value" >&2; exit 2; }
      message=$2
      shift 2
      ;;
    --message=*)
      message=${1#*=}
      shift
      ;;
    --allow-sweep)
      allow_sweep=1
      shift
      ;;
    --staged)
      staged=1
      shift
      ;;
    --)
      shift
      paths=("$@")
      break
      ;;
    -h|--help)
      sed -n '2,30p' "$0"
      exit 0
      ;;
    *)
      echo "commit-check: unknown argument '$1'" >&2
      exit 2
      ;;
  esac
done

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "commit-check: not inside a git repository" >&2
  exit 2
fi

# Cargo.lock is always excluded: it is regenerated, not authored.
if (( ${#paths[@]} > 0 )); then
  spec=("${paths[@]}" ':(exclude)Cargo.lock')
else
  spec=(. ':(exclude)Cargo.lock')
fi

if (( staged )); then
  diff_cmd=(git diff --cached)
  mode="index vs HEAD"
  if (( ${#paths[@]} > 0 )); then mode="$mode, paths: ${paths[*]}"; fi
else
  diff_cmd=(git diff HEAD)
  mode="working tree vs HEAD"
  if (( ${#paths[@]} > 0 )); then mode="$mode, paths: ${paths[*]}"; fi
fi

numstat=$("${diff_cmd[@]}" --numstat -- "${spec[@]}")
if [[ -z "$numstat" ]]; then
  echo "commit-check: nothing to check in $mode." >&2
  exit 2
fi

added=0
files=0
binary=0
crates=""
while IFS=$'\t' read -r a _d path; do
  [[ -z "${path:-}" ]] && continue
  files=$((files + 1))
  if [[ "$a" == "-" ]]; then
    binary=$((binary + 1))
  else
    added=$((added + a))
  fi
  case "$path" in
    crates/*)
      crate=${path#crates/}
      crate=${crate%%/*}
      crates="$crates $crate"
      ;;
  esac
done <<<"$numstat"

crate_list=$(printf '%s\n' $crates | sort -u | tr '\n' ' ')
crate_count=$(printf '%s\n' $crates | sort -u | grep -c . || true)

echo "commit-check: $mode"
echo "  files      : $files${binary:+ ($binary binary)}"
echo "  added lines: $added (limit $MAX_LINES)"
echo "  crates     : ${crate_list:-<none - not under crates/>}"

status=0

# --- line budget -------------------------------------------------------------
if (( added > MAX_LINES )); then
  echo "  FAIL: $added added lines exceeds the $MAX_LINES line budget." >&2
  echo "        Split the change (see the commit-discipline skill) before committing." >&2
  status=1
fi

# --- file budget -------------------------------------------------------------
if (( files > MAX_FILES )); then
  if (( files <= MAX_FILES_SWEEP && crate_count <= 1 && allow_sweep )); then
    echo "  NOTE: $files files changed - allowed as a mechanical single-crate sweep." >&2
    echo "        Confirm every change in this commit is mechanical (no behavior change)." >&2
  else
    echo "  FAIL: $files files changed exceeds the $MAX_FILES file budget." >&2
    echo "        Split per file, or pass --allow-sweep if this is a mechanical single-crate sweep." >&2
    status=1
  fi
fi

# --- crate budget ------------------------------------------------------------
if (( crate_count > 1 )); then
  echo "  FAIL: $crate_count crates touched ($crate_list). One crate per commit." >&2
  echo "        Split per crate." >&2
  status=1
fi

# --- message format ----------------------------------------------------------
if [[ -n "$message" ]]; then
  subject=${message%%$'\n'*}
  if (( ${#subject} > 72 )); then
    echo "  FAIL: commit subject is ${#subject} characters (limit 72)." >&2
    status=1
  fi
  if [[ ! "$subject" =~ ^[a-z0-9][a-z0-9./-]*:[[:space:]][[:space:]]*.+ ]]; then
    echo "  FAIL: commit subject must be '<scope>: <imperative summary>'." >&2
    echo "        Got: '$subject'" >&2
    status=1
  fi
  case "$subject" in
    *[Ww][Ii][Pp]*|*"fix stuff"*|*"Fix stuff"*|*"updates"*|*"Updates"*|*"misc"*|*"Misc"*)
      echo "  FAIL: commit subject is a placeholder." >&2
      status=1
      ;;
  esac
fi

# --- leftovers that CI will not catch ---------------------------------------
# Only Rust sources are scanned. Markdown legitimately contains `=======`
# (setext headings) and macro names inside prose.
rs_files=()
while IFS= read -r path; do
  [[ -z "$path" ]] && continue
  case "$path" in
    *.rs) rs_files+=("$path") ;;
  esac
done < <("${diff_cmd[@]}" --name-only -- "${spec[@]}")

if (( ${#rs_files[@]} > 0 )); then
  rs_diff=$("${diff_cmd[@]}" -U0 -- "${rs_files[@]}")
  added_rs=$(grep -E '^\+' <<<"$rs_diff" | grep -vE '^\+\+\+' || true)

  if grep -qF -- 'dbg!(' <<<"$added_rs"; then
    echo "  FAIL: the change contains 'dbg!('." >&2
    echo "        dbg! prints to stderr and is never intended in this codebase." >&2
    status=1
  fi

  for pattern in 'panic!(' 'todo!(' 'unimplemented!(' 'unreachable!(' 'println!(' 'eprintln!(' 'assert!('; do
    if grep -qF -- "$pattern" <<<"$added_rs"; then
      echo "  WARN: the change contains '$pattern'." >&2
      echo "        AGENTS.md forbids it in production code; it is only allowed in tests." >&2
    fi
  done

  if grep -qE '^\+[^+].*(<{7}|>{7})' <<<"$added_rs"; then
    echo "  FAIL: the change contains conflict markers." >&2
    status=1
  fi
fi

# --- index hazards -----------------------------------------------------------
# An intent-to-add entry holds an empty blob, so a plain `git commit` would
# record an empty file. Flag it so the author uses `git add` or `--only`.
ita=$(git ls-files -s -- "${spec[@]}" 2>/dev/null | awk '$2=="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" {print $4}')
if [[ -n "$ita" ]]; then
  echo "  NOTE: these paths are intent-to-add; use 'git add <path>' or" >&2
  echo "        'git commit --only <path>' or they will be committed empty:" >&2
  printf '        %s\n' $ita >&2
fi

if (( status == 0 )); then
  echo "  OK: within budget."
fi
exit "$status"
