---
description: Review the current changes for correctness, unsafe contracts, and AGENTS.md compliance
agent: build
subagent: true
---

Review the current changes in the Codevar repository.

Use the `codevar-reviewer` subagent to perform the review.

Diff under review:

!`git diff HEAD`

Diff stat:

!`git diff HEAD --stat`

Focus on $ARGUMENTS if specific areas were named; otherwise review the whole
diff.

Report findings ordered by severity, with file and line references. List any
public API changes and state plainly whether they are breaking. End with a
verdict: `approve`, `approve with nits`, or `request changes`.

Do not edit any files.
