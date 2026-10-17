---
id: RWQ06
order: 6
plan: patching-radiance-wire-quality
state: pending
created-at: '2026-10-15T00:00:00+00:00'
breadth: ''
skill: advanced
created-by: agent
priority: P1
work: M
---

# Promotion, regression fixtures, runtime defaults and pin migration

## Description

Turn only evidence-backed wire modes into a supported BigCherry Radiance feature. The existing Flash-Next production profile explicitly selects exact and Qwen3.8-27B selects wht6. No plan changes those settings in advance. This item exists so prototypes do not quietly become production settings or bypass default/profile rules.

## Steps

1. Register experimental settings in libr3/component.toml, default off; activation markers and readable modes must match actual GPU path. For a new engine operator enum use a pinned radNNNN patch package with source-file anchor tests and no orphaned compatibility aliases.
2. Model source/patch set/experiment in engines/radiance/recipes.toml with one isolated focal difference per experiment, not a giant patch stack. Production runtime profiles list ONLY switches that differ from the component's declared defaults and only after RWQ05 accepts them.
3. Extend focused permanent tests and fixture coverage: width/shape/type rejection, modes, declared-handle reachability, policy parser, graph replay, cross-rank policy handshake, scratch offset, model-agnostic exact fallback and refusal of unknown modes. Keep synthetic lab fixture separate from device tests and quality receipts.
4. Run build/patch-lint/unit/test suite and actual two-GPU live validation. Review a negative test: mismatched rank policy must error instead of spin; regression tests for both 27B and Flash-Next, with and without draft, and with the current native-attention default.
5. Update runbook, component setting descriptions, activation/evidence capture, deployment-specific selection and source citations. Record CI/hardware provenance. Promote winner only if measured net gain passes quality and speed gate; otherwise close candidates as evaluated/rejected with full evidence, retaining code when useful.

## Detailed Solution & Technical Design

Do not introduce a default engine implicit to APIs, shim for old wire modes, or dual source of truth. If architecture selection can be supplied by an immutable policy ID during graph declaration, do not allow environment changes mid-session. Resolve mismatch on both ranks before any device GPU handshake to avoid deadlock. Keep pin 89cee7ce explicit; a new upstream pin requires checking WHT layout, body hashes and device checker again. Test performance over graph replay after startup, not only first launch.
Release quality evidence must be from reproducible server runs; none of RWQ01's CPU-only mock receipts can satisfy it. Existing TOOLS/AGENTS workflows require planning and ledger events; if their MCP tools are not available, do not forge events or backdate state changes.

## Code Samples & Guidance

New settings must have default, status, summary, activation in component.toml where applicable; recipe profiles must omit all flags equal to the component's default. Fail test when a profile repeats any default.

## Files

engines/radiance/components/libr3/component.toml; engines/radiance/recipes.toml; engines/radiance/patches/...; tools/tests/patch/test_radiance_patch_packages.py; relevant tools/tests/experiment tests; docs/reference/engines/radiance/RUNBOOK.md; engines/radiance/README.md; docs/evidence/RWQ05, RWQ06.

## Validation

Validation gates: offline unit and pinned-source patch checks; GPU kbench and poisoning; exact self-reference, benchmark quality and performance; both rank policies same, robust recovery/errors under unsupported mode; no regression in 27B. After successful promotion, a clean repo-from-main lane builds and serves the selected candidate.

## Effort & Risk

Medium operational risk from mode/version skew across ranks; high failure consequence (silent accuracy loss/hang). Mandatory fail-closed behavior and preservation of exact are deployment safety requirements.

## Standards

- Target Radiance 1.3.0 pinned at 89cee7ce414486c717aa2085c99e1f522e4c0ea6; no unpinned edits.
- Engine edits are explicit Radiance patch packages; gfx1100 kernels and settings belong to the libr3 component. Keep opt-ins OFF unless independently promoted with quality evidence.
- Use engine-specific recipes, source/build identities, runtime profiles containing only settings different from declared defaults, and existing bigcherry ab-benchmark/engine_quality evidence facilities. No second campaign or loader.
- Follow docs/reference/tooling/TOOLING.md, docs/reference/engines/radiance/RUNBOOK.md, patch mechanics, relevant device checker, and main-branch code review requirements.
- Changes must preserve both-rank numerical agreement, graph replay and no-P2P mapped-pinned-host protocol; fail closed on missing kernel support or disagreement.

## Acceptance Criteria

- Winner accepted only with RWQ05 reproducible KLD/ABBA evidence; otherwise exact remains the default.
- Component/recipe/patch runtime identities explicit and validated.
- Documentation, hardware acceptance and regression fixtures committed to main.
- All plan items remain pending until their own criteria are met; no retrospective completion from offline mock evidence.

## Notes

Depends RWQ05. Current baseline production selections remain untouched. No GPU work performed by creation of these plans.

## Change Log

- 2026-10-15T00:00:00+00:00 (created-by): Initial plan; unimplemented GPU work remains pending.
