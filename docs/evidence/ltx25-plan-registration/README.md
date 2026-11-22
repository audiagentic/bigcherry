# LTX-2.5 IQ8R TP3 — Planning MCP registration handoff

**Registration completed 2026-11-22 through Planning MCP:** authoritative `LTX01–LTX08` are in [the active Radiance LTX plan group](../../planning/active/build-radiance-ltx25-tp3/). These original proposal documents remain **historical evidence, not authoritative Planning state**.
Registered owner group: `build-radiance-ltx25-tp3`. The reserved identifiers `LTX01`–`LTX08` were validated and registered; no parallel LT3/C plan identifiers were created. Source scope `docs/design/ltx25-iq8r-tp3.md` LT3-01–LT3-15 is comprehensively mapped without duplicating the full 15 items.

| Proposed item | Gap mapping | Priority | Depends on | Registration status |
| --- | --- | --- | --- | --- |
| [LTX01](LTX01.md): Qualify only LTX IQ8R declarations and model-source geometry | LT3-01, LT3-03, LT3-04, LT3-07 | P0 | Radiance 2.2 ABI44 and merged RUM13 AQL proof; PR #1262 | Registered; engineering evidenced, Planning close pending |
| [LTX02](LTX02.md): Execute actual LTX Q/K RMS and self-attention with checkpoint activations | LT3-05 and self-attention part of LT3-06 | P0 | LTX01 model source; RUM13 native AQL proof; RUM06 TP-group contract | Registered |
| [LTX03](LTX03.md): Complete the 48-block dual video/audio TP3 DiT graph without new dispatch infrastructure | Remaining LT3-06 plus cross-modal and FFN work from LT3-03/04/05 | P0 | LTX02 accepted on both modalities | Registered |
| [LTX04](LTX04.md): Bind LTX conditioning/guidance/media to existing native TP3 serving | LT3-02, LT3-08, LT3-09 | P0 | LTX03/TP1 parity; specific RUM05 admission and RUM06 lifecycle contracts | Registered |
| [LTX05](LTX05.md): Qualify three-card LTX video/audio numerical parity and end-to-end performance | LT3-14 P0 (three-card part) | P0 | LTX04 opt-in serving acceptance | Registered |
| [LTX06](LTX06.md): Qualify optional gfx1030 Gemma text worker and actual memory residency | LT3-10 and LT3-11 | P1 | Independent accepted TP3 three-card serving LTX04; no dependency for LTX01–05 | Registered |
| [LTX07](LTX07.md): Add optional LTX/Gemma tensor handoff on an admitted worker | LT3-12 and LT3-13 | P1 | LTX04 + LTX06; RUM05 generic worker admission and RUM06 transport lifetime | Registered |
| [LTX08](LTX08.md): Release qualified LTX TP3 configurations with reproducible rollback | LT3-15 and final LT3-14 release decision | P0 | LTX05 accepted; LTX07 only for optional text-worker variant | Registered |


**2026-11-22 source-backed LTX01 engineering closure:** Current main now contains
[the checkpoint-backed LTX01 geometry and declaration acceptance receipt](../2026-11-22-ltx25-ltx01-foundation-closure/README.md).
This is **engineering acceptance**, not a Planning MCP lifecycle transition:
the registered `LTX01` still requires its authoritative state/receipt
reconciliation through Planning MCP. The other seven registered plan items
retain their unaccepted technical gates. In particular, LTX02 has a real
issued C05 Q/K RMS subset but lacks full issued C06 attention and TP1
intermediate parity. RUM13's shared AQL test has merged independently.
No new C0x plan ID was created.

## One numbering system for execution

**The ONLY registered LTX Planning MCP work items are LTX01–LTX08 above.** All eight now exist under `docs/planning/active/build-radiance-ltx25-tp3/`. Existing LT3-01–LT3-15
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
| Shared **RUM13/S1** (native lab merged) | Delete the duplicate direct-HIP **generic** GPU lab executable, HIP-off core and launch branch; retain one Radiance AQL regression gate | **RUM13**, not LTX01 |
| Then **LTX01 source proof → LTX02 GPU graph** | Complete model-derived geometry, then issue checkpoint-backed C05 Q/K RMS and C06 self-attention with TP1 oracle | **LTX01 → LTX02** only |

**Ownership invariant:** every **shared** execution implementation has
exactly one **RUM** owner; LTX implements only model-specific adapters,
data geometry and output/quality checks. No new LTX/C0x plan, shared
device dispatcher, shared allocation, generic kernel, group barrier,
YAML parser, release framework or alternate collective is allowed.
The [registered RUM13 native qualification task](../rum13-native-qualification/README.md)
owns **only** the generic AQL test/launch-path reduction; it does not
take existing RUM05/RUM06 runtime code. RUM13 owns only shared native qualification; RUM06 remains the sole production
collective and group-lifetime owner.

| Shared responsibility (exactly one implementation owner) | Owner; LTX use |
| --- | --- |
| Radiance patch footprint, extracted source hooks | **RUM02** (completed); LTX only attaches to existing native APIs |
| ABI44, mixed-architecture plugin eligibility and loader | **RUM03** (completed); LTX revalidates its binary |
| Physical device/BDF order, measured VRAM/host budget, native-v3 YAML/worker admission | **RUM05** (completed); LTX04/LTX07 supply **LTX model requirements**, not an owner/allocator/parser |
| Cross-DSO shared-slot group, rank rendezvous, collective lifetime, clean/cancel/fault drain | **RUM06** (active); LTX02/LTX03/LTX04/LTX07 only test the consumed contract |
| Selected production migration canaries and release infrastructure | **RUM09** (completed); LTX05/LTX08 own only new-model acceptance/recipes |
| Model-independent native AQL regression executable and direct-HIP lab cleanup | **RUM13** (registered; native lab merged); LTX01/LTX02 consume evidence only |
| Genuine missing shared gfx1030 kernel or role implementation | **RUM-only follow-on** after LTX06/07 supplies a failing concrete requirement; check existing RUM/kernel owners before proposing |

**RUM13 is separate from RUM06:** it removes a **lab-only**
duplicate launch and creates a reusable native regression gate;
RUM06 retains all production group and collective lifetime code.
If a model-specific LTX test references RUM13 evidence, that does
**not** transfer implementation ownership into LTX.


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

The existing fifteen LT3 entries are **engineering gaps**, not Planning MCP IDs. The eight LTX items are independent **model-specific** work units. The generic regression-lab deletion has moved exclusively to registered RUM13; existing RUM05/RUM06 own shared admission/lifetime. LTX01 completes LTX geometry/source proofs and LTX02 owns the first model-backed GPU graph. LTX03–05 complete video/audio integration; LTX06–07 remain optional. No extra LTX/C plan is created.

## Existing proof versus open gates (2026-11-22)

Local `agent/ltx25-tp3-main22` has C01–C06 source-backed host declarations, actual selected checkpoint GEMM tests on all three GPUs (961/961), and C08 exact one-shot/two-shot BF16 collectives on gfx1100/gfx1100/gfx1201 through the correct native AQL dispatch (33/33 per variant). Those are **component proofs**, not an integrated model-weight distributed graph and not a generated video. The C07 direct-HIP SIGSEGV is a historical lab-only negative, not a production regression; RUM13 alone owns deleting its duplicate implementation. The patch remains unselected for production, TP1 serving is unchanged, and `world_size > 1` still refuses.

PR #1262 is a diagnostic foundation. It must have its current local commits pushed, CI/data catalog passing and review before it can be considered for merge. No LTX01–08 plan is completed just because a related C0x check passed.

## Registration instructions

**Do not register this handoff as duplicate code work.** When the
authoritative Planning MCP is available, in order:

- Inspect existing RUM02/RUM03/RUM05/RUM06/RUM09 and any RUM13 ID,
  ownership reviews and linked follow-ons. Retain completed **RUM05** as the only
  generic device/BDF/native-v3 resource-admission owner and **RUM06**
  as the only TP-group/cancel/drain owner; update the *existing items*
  through their normal review/Planning lifecycle only if a genuinely
  missing shared capability has been demonstrated. Do **not** assign
  generic engine changes to an LTX item.
- Check whether the [RUM13 candidate](../rum13-native-qualification/README.md)
  overlaps an existing shared native GPU qualification item. If so,
  **integrate its exact non-LTX deliverable into that RUM owner and
  retire RUM13**; otherwise register it under
  `patching-radiance-upstream` using the service-allocated ID. One
  reusable AQL harness, one code-removal task, no competing owners.
- Register LTX01–08 with the updated **model-specific** scope and
  concrete external RUM requirements. Do not clone RUM05/06/12
  implementation steps into LTX acceptance; record only the native
  interface the LTX model consumes and a failing consumer test.

1. Use Planning MCP: list groups, find or create `build-radiance-ltx25-tp3`, verify ID availability, and create these eight items with their full Description, Steps, Detailed Design, Code Samples, Files, Validation, Effort, Standards, Acceptance and Notes.
2. Link each to the existing `docs/design/ltx25-iq8r-tp3.md` LT3 gap list and PR #1262; link **the specific shared RUM owners above** as external dependencies (not duplicate LTX steps), and register proposed RUM13 separately only if its uniqueness and need are confirmed. Mark only actually accepted steps as evidenced. Do not change `docs/planning/` by hand.
3. Use Ledger MCP to record substantive changes as **unreleased** with the selected authoritative plan IDs; do not invent ledger event files.
4. Track external reviews and close after acceptance. Stage implementation in small separate PRs, reuse core and native kernel paths rather than expanding `rad0060` into a second serving system.

## Test and repo state

This index is a versioned evidence/handoff artifact. It is not a replacement for Planning MCP, a production activation or a claim that all tests were re-executed. Revalidation logs from this session and current CI belong to the subsequent code/evidence commit, and may differ from the older C0x receipts.
