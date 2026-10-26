# Commit message catalog

The format is fixed:

```
<scope>: <imperative summary, <= 72 characters>

<why the change was needed and what it does, wrapped at 72 characters, <= 6 lines>
```

Optional last line: `Refs: #123`.

## Scope

The scope is the crate or module that owns the change, using the shortest
unambiguous name:

| Crate                                  | Scope to use    |
|----------------------------------------|-----------------|
| `codevar-tls`                          | `codevar-tls`   |
| `codevar-ocl-ir`                       | `ocl-ir`        |
| `codevar-truetype-core`                | `truetype-core` |
| `codevar-truetype-sfnt`                | `truetype-sfnt` |
| `codevar-logger`                       | `logger`        |
| `codevar-wredit`                       | `wredit`        |
| `crates/codevar-base` (shared helpers) | `codevar-base`  |

Prefix `codevar-` only when it disambiguates. `truetype-core` and `truetype-sfnt`
are distinct enough; `tls` and `wsocket` are not, so keep `codevar-tls` there.

## Good subjects

The subject states the **effect** of the change as an imperative verb.

```
codevar-tls: reject renegotiation after handshake failure
wredit: split the rope table into a fixed-size array
ocl-ir: lower barrier intrinsics to inline fences
logger: drop the timestamp allocation on the fast path
truetype-sfnt: reject tables with a negative maxp numGlyphs
codevar-base: add html entity decoding for named references
```

## Bad subjects

```
update tls code                       # activity, not effect
fix stuff                            # no information
WIP                                  # placeholder
updates                              # no effect
optimize performance                  # which performance?
refactor lib.rs                      # no effect, and mixes concerns
[BUG] tls.rs                          # prefix adds nothing, scope does
fix: tls.rs:123                       # the scope field is for the crate
```

A subject that names the file but not the effect is still a bad subject:
`wredit: fix rope.rs` should be `wredit: fix rope split after a deleted range`.

## Bodies

The body answers **why**, and tells the reviewer what to look at. The diff
already shows what changed; repeating it wastes the reviewer's attention.

Good body:

```
RFC 8446 forbids renegotiation once the handshake completes. Accepting it
let a peer reset the cipher suite mid-session, which invalidated the
session ticket binding and broke resumption.

Check that the ticket cache is also cleared on the error path.

Refs: #412
```

Bad body (restates the diff):

```
Added a check in handshake.rs and removed the old code. Updated the tests
to cover the new behavior. Fixed the imports.
```

For mechanical sweeps, say so and say why it is safe:

```
ocl-ir: document safety contracts on the lowering passes

Every unsafe block in this crate dereferences pointers derived from the
validated module buffer. The invariants were implicit in the code and are
now written down, which the next agent needs to reason about them.

No behavior change.
```

For performance work, state the measurement, not the intuition:

```
wredit: keep the rope table in a fixed-size array

The table was a Vec that reallocated on every insert. The scroll
benchmark showed 8% of frame time in the allocator, so the allocation is
now off the hot path.
```

Not:

```
wredit: made rope table faster (much faster now)
```

## Refs

Use `Refs: #123` on the last line when the work tracks an issue. Use `Fixes: #123`
only when the commit genuinely closes it. Do not invent issue numbers.
