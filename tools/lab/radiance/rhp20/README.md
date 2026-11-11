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

## Exact hardware-owner commands (two independent cells)

From this worktree or after merging to main:

```bash
# From the BigCherry repository root
source tools/env/bigcherry-env.sh
BUILD=${BC_WORK_ROOT}/builds/daf793f13be8d5762deecb73e3372641/a29e62ada0fe691746707b2b87c3e9f3
# Owner acquires and holds the GPU lease externally before running either cell.

PYTHONPATH=tools python3 tools/lab/radiance/rhp20-forced-move.py prepare --layout flashnext-2xtx-rdna2-tail
PYTHONPATH=tools python3 tools/lab/radiance/rhp20-forced-move.py run --layout flashnext-2xtx-rdna2-tail --build "$BUILD" --port 36080 --out ${BC_WORK_ROOT}/runs/rhp20-tail-forced-1000 --lease-held --identity ${BC_WORK_ROOT}/runs/rhp20-tail-forced-1000/identity.json

PYTHONPATH=tools python3 tools/lab/radiance/rhp20-forced-move.py prepare --layout flashnext-2xtx-rdna2-tail-combine
PYTHONPATH=tools python3 tools/lab/radiance/rhp20-forced-move.py run --layout flashnext-2xtx-rdna2-tail-combine --build "$BUILD" --port 36080 --out ${BC_WORK_ROOT}/runs/rhp20-tail-combine-forced-1000 --lease-held --identity ${BC_WORK_ROOT}/runs/rhp20-tail-combine-forced-1000/identity.json
```

The `run` command internally calls:

```bash
bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port 36080 --layout flashnext-2xtx-rdna2-tail --experiment tools/lab/radiance/rhp20/tail.experiment.json --record-period 1
bash tools/lab/radiance/launch-flashnext-best.sh start --build "$BUILD" --port 36080 --layout flashnext-2xtx-rdna2-tail-combine --experiment tools/lab/radiance/rhp20/tail-combine.experiment.json --record-period 1
```

**Only one layout may run at a time.** The runner stops the engine and
helpers in `finally`, then saves correlated helper logs, live manifest,
benchmarks, and a verdict. It never fabricates the per-expert contribution
probe. `--identity` is a required hardware-produced record of
`records: [{expert: "layer:expert", transaction_id: 1, before_epoch: 1, after_epoch: 2, source_isa: "gfx1100",
target_isa: "gfx1100", before_sha256: "...", after_sha256: "..."}]`;
the transaction and epoch must match an actual same-ISA resolved swap, and the two hashes must reflect the actual same expert's output contribution
under identical inputs before/after its **committed** same-ISA move.
An API token-text comparison alone is not valid proof. Until a genuine
per-expert probe exists, the runner **must FAIL** this criterion. Cross-ISA
movements are gated numerically, not by byte identity.

## Layout-specific KLD, separate from 1,000-move stress

The campaign uses the RHP06/RHP22 quality thresholds, a static control
versus forced arm, fixed reference at
`${BC_WORK_ROOT}/runs/rad116-kld/reference`. Build `RAD_BIN`/
`RAD_HOME` must point to the same tested build; use the established
campaign environment and hold the GPU lease externally.

```bash
PYTHONPATH=tools python3 tools/lab/radiance/rhp20-forced-move.py kld-config --layout flashnext-2xtx-rdna2-tail --out ${BC_WORK_ROOT}/runs/rhp20-tail-kld/config.json
PYTHONPATH=tools python3 -m bigcherry ab-benchmark --server-config ${BC_WORK_ROOT}/runs/rhp20-tail-kld/config.json --output ${BC_WORK_ROOT}/runs/rhp20-tail-kld/kld --pairs 2

PYTHONPATH=tools python3 tools/lab/radiance/rhp20-forced-move.py kld-config --layout flashnext-2xtx-rdna2-tail-combine --out ${BC_WORK_ROOT}/runs/rhp20-tail-combine-kld/config.json
PYTHONPATH=tools python3 -m bigcherry ab-benchmark --server-config ${BC_WORK_ROOT}/runs/rhp20-tail-combine-kld/config.json --output ${BC_WORK_ROOT}/runs/rhp20-tail-combine-kld/kld --pairs 2
```

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
