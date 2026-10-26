---
description: Commit the current work as a reviewable checkpoint within the Codevar commit budget
agent: build
---

Load the `commit-discipline` skill and follow it exactly.

Create a checkpoint commit for the work currently in the working tree.

Context — staged and unstaged changes:

!`git status --short`

Diff stat:

!`git diff HEAD --stat`

## Steps

1. **Determine the scope.** If the working tree contains unrelated in-flight
   work, ask which paths belong to this commit. Do not sweep everything in.

2. **Verify first.** Run the scoped checks for the affected crates. Do not
   commit code that does not compile or whose tests fail:
   ```
   .opencode/skills/verify/scripts/affected-crates.sh
   ```

3. **Stage only what belongs to this commit.** Use explicit paths, never
   `git add -A`.

4. **Check the budget:**
   ```
   .opencode/skills/commit-discipline/scripts/commit-check.sh
   ```
   If it fails, split the change per the skill and commit the pieces
   separately. Do not bypass the check.

5. **Write the message** in the format `<scope>: <imperative summary>` plus a
   body that explains why. If the user supplied a message with `/commit $ARGUMENTS`,
   validate it against that format and correct it if it does not match.

6. **Commit.** Use a heredoc so the message survives shell quoting:
   ```
   git commit -F - <<'EOF'
   <scope>: <summary>

   <why>
   EOF
   ```

7. **Report** the commit hash, the subject line, and the size:
   `git show --stat --oneline HEAD`.

Never push. Never amend a pushed commit. Never mix a refactor with a behavior
change.
