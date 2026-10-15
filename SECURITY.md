# Security Policy

## Supported Versions

Codevar is in early development and does not have stable releases yet. Security
updates are applied to the latest code on the `main` branch.

| Version | Supported          |
| ------- | ------------------ |
| `main`  | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them privately via one of the following channels:

- **Email**: Send an email to **tomas.corderov@yahoo.com** with the subject
  line `[SECURITY] Codevar Vulnerability Report`.
- **GitHub Security Advisories**: If you prefer, you can
  [create a private security advisory](https://github.com/Tommicord/codevar/security/advisories/new)
  on the repository.

Please include as much of the following information as possible in your report:

- Type of issue (e.g. buffer overflow, use-after-free, privilege escalation,
  remote code execution, etc.)
- Full paths of the source file(s) related to the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit it

## What to Expect

- You will receive an acknowledgment of your report within **72 hours**.
- We will work with you to understand and validate the issue.
- A fix will be developed and released before any public disclosure, and we
  will credit you in the advisory (unless you prefer to remain anonymous).
- We will coordinate a disclosure timeline with you, typically allowing **90
  days** for a fix before public disclosure, though this may vary depending on
  the severity and complexity of the issue.

## Scope

This policy applies to vulnerabilities in the Codevar codebase itself,
including:

- The Rust core library and its crates
- The WebAssembly (WASM) build targeting browsers
- GPU compute shaders and rendering code
- Any bundled dependencies declared in `Cargo.toml`

Vulnerabilities in third-party dependencies should be reported upstream to the
responsible project, but we would appreciate being notified as well.

## Security Best Practices for Contributors

When contributing to the project, please keep security in mind:

- Validate all external inputs
- Use bounded integer operations to prevent overflow
- Be careful with pointer arithmetic in `unsafe` code and document all safety
  invariants
- Avoid implementing custom cryptography; use well-vetted libraries instead
- Use constant-time operations when handling secret data
