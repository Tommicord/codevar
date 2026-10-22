---
description: Print a context brief for a Codevar crate - purpose, dependencies, public API, unsafe hotspots
agent: build
---

Print the context brief for the crate `$1`.

```
bash .opencode/skills/crate-context/scripts/crate-info.sh $1
```

Then summarize it in a form an agent can act on:

- **What the crate is for** - one sentence, from its purpose and its public API.
- **What it depends on** - the workspace crates it uses, and the workspace
  crates that use it. Note the direction: a change to a leaf crate is cheap,
  a change to a widely-depended-on crate is not.
- **Public API surface** - the count, and the specific items that a change
  would likely touch.
- **Unsafe hotspots** - the files with the most `unsafe` blocks, and whether
  they carry `SAFETY:` comments.
- **Tests** - how much test surface the crate has today.
- **Where to start reading** - the two or three files to read first, in order.

Keep it to one screen. The point is to know what to read next, not to
reproduce the file.
