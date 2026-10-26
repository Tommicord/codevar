<!-- Thanks for contributing to Codevar! Please fill out the sections below. -->

## Description

<!-- A clear and concise description of what this PR does and why. -->

Fixes # <!-- issue number, e.g. Fixes #123 -->

## Type of Change

<!-- Check the relevant option(s) with an "x". -->

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] CI/build or other tooling change

## How Has This Been Tested?

<!-- Describe the tests you ran to verify your changes. Include details of
     testing steps so reviewers can reproduce. If not applicable, explain why. -->

- [ ] `cargo test --workspace`
- [ ] Unit tests added/updated
- [ ] Manual testing (describe below)

**Test environment**:
- OS:
- Rust version:
- Target (wasm/native):

## Checklist

<!-- Check all that apply with an "x". -->

- [ ] My code follows the project's coding conventions (see [AGENTS.md](../AGENTS.md))
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes (`cargo test --workspace`)
- [ ] `cargo fmt --all` has been run and formatting is clean
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes without errors or warnings
- [ ] I have updated the documentation accordingly (if applicable)
- [ ] My changes generate no new warnings
- [ ] No `.unwrap()`, `.expect()`, `panic!`, `println!`, or `eprintln!` in production code

## Performance Impact

<!-- If relevant, describe any performance implications of this change,
     including benchmarks or profiling results. If not applicable, write "N/A". -->

## Additional Context

<!-- Add any other context, screenshots, or relevant information about the PR here. -->
