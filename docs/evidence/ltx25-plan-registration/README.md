# LTX-2.5 IQ8R TP3 — Planning MCP registration handoff

**Status: registration candidates, NOT authoritative Planning MCP state.**
Owning group proposal: `build-radiance-ltx25-tp3` (new group only after Planning MCP verifies no existing equivalent). IDs `LTX01`–`LTX08` are proposed only: verify availability and let the service allocate identifiers. Source scope `docs/design/ltx25-iq8r-tp3.md` LT3-01–LT3-15 is comprehensively mapped without duplicating the full 15 items.

| Proposed item | Gap mapping | Priority | Depends on | Registration status |
| --- | --- | --- | --- | --- |
| [LTX01](LTX01.md): Qualify source-backed IQ8R TP3 declarations and finish the foundation | LT3-01, LT3-03, LT3-04, LT3-07 | P0 | Radiance 2.2 ABI44; candidate PR #1262; none of the downstream LTX plans | Candidate |
| [LTX02](LTX02.md): Execute exact three-rank Q/K RMS and self-attention with checkpoint activations | LT3-05 and self-attention part of LT3-06 | P0 | LTX01; native C08 exact AQL collective proof | Candidate |
| [LTX03](LTX03.md): Complete the 48-block dual video/audio TP3 DiT graph without new dispatch infrastructure | Remaining LT3-06 plus cross-modal and FFN work from LT3-03/04/05 | P0 | LTX02 accepted on both modalities | Candidate |
| [LTX04](LTX04.md): Integrate safe opt-in TP3 native serving, conditioning and guidance | LT3-02, LT3-08, LT3-09 | P0 | LTX03 full 48-block emitted path and approved TP1 parity | Candidate |
| [LTX05](LTX05.md): Qualify three-card LTX video/audio numerical parity and end-to-end performance | LT3-14 P0 (three-card part) | P0 | LTX04 opt-in serving acceptance | Candidate |
| [LTX06](LTX06.md): Qualify optional gfx1030 Gemma text worker and actual memory residency | LT3-10 and LT3-11 | P1 | Independent accepted TP3 three-card serving LTX04; no dependency for LTX01–05 | Candidate |
| [LTX07](LTX07.md): Add optional native text-stack handoff and measured four-card admission | LT3-12 and LT3-13 | P1 | LTX04 and LTX06 accepted; LTX05 baseline as independent comparator | Candidate |
| [LTX08](LTX08.md): Release qualified LTX TP3 configurations with reproducible rollback | LT3-15 and final LT3-14 release decision | P0 | LTX05 accepted; LTX07 only for optional text-worker variant | Candidate |

## One numbering system for execution

**The ONLY proposed Planning MCP work items are LTX01–LTX08 above.** None has
been created in the authoritative service yet. Existing LT3-01–LT3-15
numbers are **design requirements**; do not register them again. The native
C01–C08 engineering milestone names are **historical implementation/evidence
labels**, not plans or sequential user-facing roadmap items. Their names do
not confer completion on LTX01–LTX08. The synthetic component test gates
were formerly also called C00–C08; this document now calls them
[G00–G08](../../design/ltx25-tp3-component-test-ladder.md) to avoid collisions.
Existing evidence filenames keep their historical C labels.

| Engineering evidence | Actual scope and status | Owning proposed plan |
| --- | --- | --- |
| C01–C06 (native LTX source work) | Model-backed declarations; TP1 startup still unchanged; **NOT issued TP3 GPU graph** | LTX01 foundation; LTX02 must execute |
| C06 (per-device kernel checks) | 961/961 checkpoint IQ8R GEMM across three cards; local device kernel vocab | LTX01 evidence only |
| C07 (direct-HIP collective attempt) | Standalone mixed-architecture SIGSEGV; not normal Radiance dispatch | Historical diagnostic, **not an open production fault** |
| C08 (Radiance AQL collective) | Exact native 3-rank gfx1100/gfx1100/gfx1201 all-reduce one-/two-shot, 33/33 BF16 each | LTX01 prerequisite evidence, LTX02 input |
| Next: **LTX01/S1**, not C09 | Delete the **extra direct-HIP lab executable**, HIP-off core dependency and conditional launch path; keep one native AQL lab | LTX01 |
| Then **LTX02/S1–S2** | Issue real checkpoint-backed C05 global Q/K RMS and C06 complete self-attention, TP1 oracle | LTX02 |

**No ninth plan and no second GPU collective implementation.** The
LTX01/S1 reduction should explicitly report authored source lines **deleted
versus moved**, build targets and branches removed, tests retained, AQL
hot-path costs and source/ABI identity; it must not disturb production
`rad0017` cross-library rendezvous, per-rank library selection or the
existing AQL engine backend. Follow the executable acceptance in
[LTX01](LTX01.md) rather than copying a C09 brief into another plan.
A failed C07 direct-HIP launch does **not** justify creating an alternate
transport. The final graph/TP3 admission remains strictly opt-in.

### What belonged to the Radiance 2.2 migration?

**Yes: native reuse and regression qualification.** Completed
[RUM02](../../planning/completed/patching-radiance-upstream/RUM02.md) reduced
BigCherry patch attachments, including the shared rendezvous;
[RUM03](../../planning/completed/patching-radiance-upstream/RUM03.md) owns
ABI44 and mixed-device kernel compatibility; [RUM09](../../planning/completed/patching-radiance-upstream/RUM09.md)
owns migrated *selected production* GPU configurations. This scope should
include checks that retained mixed-card TP3 uses the actual Radiance AQL
dispatch, not direct HIP. C08 gives an additional independent proof of that
existing path. The **BigCherry shared slot is still necessary** for
`libr3` plus `libr4d`; moving or deleting it merely because a lab passes
would be a regression. Do not reopen completed RUM state as a result of C07.

**No: new LTX-2.5 TP3 functionality.** The unselected `rad0060` DiT
architecture patch, real model Q/K normalization, cross-modal blocks,
conditioning, VAE/audio/video, TP1 parity and production admission were
**not** in the RUM09 selected production roster. LTX01–05, plus optional
LTX06–07 and LTX08 release, own those uncompleted gates.

## Why eight, not fifteen

The existing fifteen LT3 entries are **engineering gaps**, not Planning MCP IDs. Combining closely coupled geometry/weight work, guidance/serving and release conditions creates eight independently executable and testable work units, with no second owner for the same native graph or shared collective. TP3 three-card video must be independent of the optional gfx1030 text-worker. LTX01/S1 is the next cleanup and code-reduction slice; LTX02 is the first **model-backed issued-graph** GPU slice. LTX03–05 complete end-to-end qualification. LTX06–07 stay optional.

## Existing proof versus open gates (2026-11-22)

Local `agent/ltx25-tp3-main22` has C01–C06 source-backed host declarations, actual selected checkpoint GEMM tests on all three GPUs (961/961), and C08 exact one-shot/two-shot BF16 collectives on gfx1100/gfx1100/gfx1201 through the correct native AQL dispatch (33/33 per variant). Those are **component proofs**, not an integrated model-weight distributed graph and not a generated video. The original C07 standalone direct-HIP SIGSEGV is a test-harness path and is not evidence of a production collective failure; it must not lead to a duplicated collective implementation. The patch remains unselected for production, TP1 serving is unchanged, and `world_size > 1` still refuses.

PR #1262 is a diagnostic foundation. It must have its current local commits pushed, CI/data catalog passing and review before it can be considered for merge. No LTX01–08 plan is completed just because a related C0x check passed.

## Registration instructions

1. Use Planning MCP: list groups, find or create `build-radiance-ltx25-tp3`, verify ID availability, and create these eight items with their full Description, Steps, Detailed Design, Code Samples, Files, Validation, Effort, Standards, Acceptance and Notes.
2. Link each to the existing `docs/design/ltx25-iq8r-tp3.md` LT3 gap list and PR #1262; include LTX01/S1 as a **step within LTX01**, not another registered plan; mark only actually accepted steps as evidenced. Do not change `docs/planning/` by hand.
3. Use Ledger MCP to record substantive changes as **unreleased** with the selected authoritative plan IDs; do not invent ledger event files.
4. Track external reviews and close after acceptance. Stage implementation in small separate PRs, reuse core and native kernel paths rather than expanding `rad0060` into a second serving system.

## Test and repo state

This index is a versioned evidence/handoff artifact. It is not a replacement for Planning MCP, a production activation or a claim that all tests were re-executed. Revalidation logs from this session and current CI belong to the subsequent code/evidence commit, and may differ from the older C0x receipts.
