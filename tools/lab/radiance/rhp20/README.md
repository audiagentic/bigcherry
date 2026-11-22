# RHP20 forced-churn acceptance (card-free preparation, 2026-11-11)

The runner does **not** acquire/release a GPU lease, touch power, start anything in
`prepare`, `verify`, or `kld-config`, or assert that a run has passed. The
hardware owner holds the external lease throughout a `run`, including shutdown.
These are *lab* forced experiments, not live-policy runs or policy calibration.
Do not pass `--live-policy` to a forced-churn startup; its separate experiment
digest creates a different calibration key.

## Grounded topology

Latest measured tail witness at preparation time:
`${BC_WORK_ROOT}/flashnext/manifest-1111-131338-1271995.json`,
owner map SHA256 `28f43e45923dd6e0f78e9ad6e631c87af8675acf0a46b11223201a4bb09c1f4d`.
The manifest records 9,546 experts on each gfx1100 helper and 5,986 on
gfx1030 (gross ownership; protected BF16 experts on gfx1100 are excluded
from the w4 eligibility set). Per-expert planned stride 2,534,400 bytes.
BDF order: gfx1100 0000:03:00.0, gfx1100 0000:06:00.0,
gfx1030 0000:17:00.0. All three helpers have w4nl64 support.
The model-side `qwen4exp-w4nl64-i8-hc8m.recipe` documents the BF16
protected experts and w4nl64 routed remainder. Candidate witnesses in
layer 0: head0/head1 experts 1/0, head0/tail 1/6,
head1/tail 0/6. These witnesses are not pinned explicit swaps: the
production AUTO_SWAP re-resolves same-layer eligible pairs from ACTIVE.

**No measured `flashnext-2xtx-rdna2-tail-combine` manifest was present** in
`${BC_WORK_ROOT}/flashnext` when prepared. Its layout uses the same three
helper roles and adds CPU row combine. The no-combine manifest is only
provisional format/placement evidence; the runner refuses PASS unless the
combine startup generates its own matching measured forced manifest.

## Bounded forced schedule

Each JSON has 3 AUTO_SWAP steps (head0/head1, head0/tail, head1/tail)
and `repeat=ceil(1000/3)=334`: **1,002 proposals**, not 1,002
guaranteed commits. The `pace_requests=104` is **measured historical
pacing**, not a guessed time interval: RHP06's 131-commit paced campaign
used `floor(16,849 completed helper lane requests / (161 proposed+1))=104`
(`runs/rhp06-churn-paced/kld.json`). Opening 1,000 proposals consumes
at least `104*1000=104,000` new completed helper lane requests.
The acceptance workload consists of 6-stream, 12-repetition, 1,024-token
decode with MTP depth 3, plus concurrently running ten
`prefill-doc-44k@v1` prompts. A 5,400 s bound is the RHP06 quality
campaign envelope. Neither workload nor quota can *guarantee* commits:
the terminal helper-0 summary must show **>=1,000 committed**, zero aborted,
zero inflight, ordered resolved transitions and coverage of all 3 pairs.
If too few have committed when requests stop, the run is FAIL, not rounded
up from attempts.

Zero-owner destinations are **not available in these recorded owner maps**,
which have all three helpers populated. A zero-owner movement cannot be
claimed as exercised here; it needs a separately measured layout with a
nonzero incoming reserve and no incumbent owner. The card-free suite
includes a zero-owner refusal check.

## Historical forced-move driver (archived)

The one-off RHP20 forced-move harness and its Bash start path were retired under S16 after archival in `archive/lab-rhp20-forced-move`. The old command examples are **not runnable in this checkout** and are not a supported native production mode. For reproducibility, use the archive and original recorded build/lease/manifest inputs; do not fabricate a replacement native profiling mode. New production runs use typed RSC12 configs with `bc-serve serve CONFIG --run-dir DIR`, and the cross-ISA quality acceptance remains pending.

## Layout-specific KLD, separate from 1,000-move stress

The cross-ISA KLD arms are **retired as a job file**: the old arm needed the Python manifest planner and a lab
experiment key (`experiment`) that the native YAML schema does not have, and the generator that wrote the config
(`rhp20-forced-move.py kld-config`) no longer exists. A native version needs the forced-churn experiment expressed in
`x-bigcherry` first (an owned, typed setting), then two native arms like `native-arms/helper-kld/`.

KLD is a **separate numerical arm** that requires its own observed result,
including top-1, KL tails, and RHP12 step-time checks. It is not evidence
for 1,000 swaps in the serving stress arm. In-flight 42K prefills are
verified by the canonical workload's measured prompt token counts; MTP
depth is asserted from the actual serving manifest. Per-step epoch/MTP
identity and zero-owner hardware gates remain separate follow-ups.

## Planning state/ledger text for manual ag-planning application

- RHP20: keep `in_progress`; card-free forced-tail preflight, experiment
  schedule, runner and strict verdicts prepared; no hardware acceptance yet.
- RHP11: keep `in_progress`; forced 1,000-transition stress for both tail
  layouts still awaits user-held lease and evidence.
- RHP06: `pending` is progress-stale; existing migration has real 131-move
  evidence, but 1,000-transition and RHP12/zero-owner coverage remain open.
- Suggested ledger: "2026-11-11 RHP20: prepared layout-specific forced churn
  (1002 proposals, pacing 104 from RHP06 measured request budget), owner-map
  and recipe eligibility checks, strict receipt and KLD preparation. No GPU
  or lease used. Combine measured manifest, 1000 commits, per-expert identity,
  cross-ISA KL, zero-owner and 4gpu stress pending hardware evidence."
