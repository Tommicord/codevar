---
name: Commit Discipline
description: Use when committing work in the Codevar repository, when the diff grows past roughly 800 added lines, when a large task needs splitting into reviewable checkpoints, or when writing a commit message. Enforces the <=800-line / <=3-file / 1-crate commit budget and the `<scope>: <imperative summary>` message format.
---

# Commit Discipline

Codevar is maintained mostly by AI agents with supervision of contribuitors (they manage the agents). They cannot read
every line an agent produces, so a commit is a *review unit*, not a save point.

## The budget

| Limit          | Value                                                                                |
|----------------|--------------------------------------------------------------------------------------|
| Added lines    | **<= 800** (excluding `Cargo.lock`)                                                  |
| Files changed  | **<= 3**, or up to 20 for a mechanical single-crate sweep that stays under 800 lines |
| Crates touched | **1** — a second crate needs a second commit                                         |
| Purpose        | **1** — never mix a refactor with a behavior change                                  |

## Workflow

1. **Work in small steps.** Aim for one step that compiles and whose scoped
   tests pass. A step that will not compile is a step that is too big.
2. **Run the scoped checks** for the crates you touched:
   ```bash
   cargo check  -p <crate>
   cargo clippy -p <crate> --all-targets -- -D warnings
   cargo test   -p <crate>
   ```
3. **Stage only what belongs to this step.** `git add` specific paths, never
   `git add -A` — the working tree may hold unrelated in-flight work.
4. **Check the budget** before committing:
   ```bash
   .opencode/skills/commit-discipline/scripts/commit-check.sh
   ```
   The script exits non-zero when the staged diff is over budget and also
   validates the commit message when given one:
   ```bash
   .opencode/skills/commit-discipline/scripts/commit-check.sh \
     --message "codevar-tls: reject renegotiation after handshake failure"
   ```
5. **Commit** with the message format below.
6. **Continue** with the next step.

## When the diff is over budget

Stop implementing and split. Split in this order:

1. **By concern** — behavior change first, refactor as a separate commit.
   A reviewer can then read the refactor and the behavior change separately.
2. **By crate** — never one commit across two crates.
3. **By file** — for mechanical sweeps (adding `SAFETY:` comments, renaming,
   adding doc comments), split per file or per small group of files. These are
   ideal agent tasks: fully mechanical, machine-verifiable, and trivially
   reviewable in batches.

If a single logical change genuinely cannot be split under 800 lines, do not
force a bad split. Instead: land the smallest compilable, tested increment
first, and say so explicitly in the commit body and in your final report.

## Commit message format

```
<scope>: <imperative summary, <= 72 characters>

<why the change was needed and what it does, wrapped at 72 characters, <= 6 lines>
```

- `<scope>` is the crate or module: `codevar-tls`, `ocl-ir`, `logger`,
  `truetype-core`. Use the shortest unambiguous name.
- The summary states the **effect**, not the activity.
  - Good: `codevar-tls: reject renegotiation after handshake failure`
  - Bad: `update tls code`, `fix stuff`, `wip`, `updates`
- The body explains **why**. The diff already shows *what*; the body must show
  the reasoning and tell the reviewer what to check.
- End with `Refs: #123` when the work tracks an issue.
- Never use `WIP`, never bullet-dump filenames, never write a subject line
  longer than 72 characters.

See `references/commit-messages.md` for the full catalog of good and bad
messages drawn from this repository.

## Never

- Never leave more than one logical step uncommitted.
- Never commit code that does not compile or whose scoped tests fail.
- Never amend, rebase, or squash a commit that has already been pushed. Add a
  new commit instead.
- Never push. Publishing is the human's decision.
- Never combine a mechanical sweep with a behavior change in one commit.

## Before reporting the task done

- [ ] Every step was committed separately, not accumulated into one commit.
- [ ] Each commit is within budget (`commit-check.sh` exits 0).
- [ ] Each message is `<scope>: <imperative summary>` plus a body explaining why.
- [ ] No commit mixes a refactor with a behavior change.
- [ ] No commit touches more than one crate.
