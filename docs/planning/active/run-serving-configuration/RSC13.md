---
id: RSC13
order: 13
plan: run-serving-configuration
state: pending
created-at: '2026-11-17T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P2
work: M
---

# Clang AST inventory of serving injected callables

## Problem

RSC02 step 4 protects current externally supplied callables through private
typed guarded wrappers and backend adapters. The `callback_guard.py` CTest
scanner is only a best-effort regex tripwire. REVIEW4 reproduces two
compiling additions that it fails to see: a function pointer aliased through
`decltype(&callback_impl)` and an inline `public: virtual` declaration.
Both allow an unguarded call. Do not claim the scanner exhaustively checks
new C++ injection sites.

## Implementation

1. Run clang tooling on the actual serving component compilation database.
   Inventory externally injectable callable declarations and dispatches in
   owned .h/.hpp/.cpp/.cc sources, excluding vendored/generated code.
2. Cover std::function-like types, raw function pointers, type aliases,
   decltype and typedef, virtual interfaces and overrides, generic/template
   callable parameters, indirect calls and C-ABI function symbols; capture
   declaration and invocation file:line.
3. Compare against a narrowly scoped, reviewable approved inventory of
   existing guarded adapters, private wrappers and internal recursion.
   Make new injection types or unguarded invocation fail CI, with actionable
   diagnostics. Keep explicit, concrete exemptions, never blanket exclusions.
4. Add *rejected* negative fixtures for both REVIEW4 counterexamples plus
   multiline/qualified virtuals, aliases, templates and indirect dispatch.
   Retain the regex CTest tripwire until the AST mechanism replaces it.

## Acceptance

- Both REVIEW4 bypasses are **caught** by the AST gate, although the
  old scanner accepts them.
- Every current externally injected callable is inventoried and verified
  reachable only via the typed boundary.
- New injection declarations/dispatches require an explicit reviewed change
  to the approved AST inventory; failing fixtures run in CTest/CI without
  hardware, machine paths or a GPU lease.
- Scoped clang-tidy, GCC/Clang strict builds, clang-format, component CTest
  and the full Python suite remain green.

## References

- [RSC02](RSC02.md), step 4, owner and power transaction.
- [Serving callback boundary](../../../../engines/radiance/components/serving/README.md).
- `engines/radiance/components/serving/callback_guard.py`.
- Review evidence: `gpt-briefs/out/REVIEW4-rsc02-step4.md`.
