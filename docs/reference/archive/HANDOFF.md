# bigcherry — state of play and what to do next

Written 2026-08-05. Read this first if you are picking the work up, here or on
`10.10.100.10`.

---

## Where things stand

Phases 0–3 of `docs/planning/active/hip-autotune/OVERVIEW.md` are complete and
verified **on hardware** (gfx1100, RX 7900 GRE). The dispatch layer runs end to
end: signature construction, blake2b hashing, dispatch-key derivation, process
cache, replay lookup, native fallback, miss recording, coverage reporting.

| Item | State |
| --- | --- |
| HI01 audit | done — 32/32, passes on pristine *and* patched trees |
| HI01 baseline | done — gfx1100, 1545/1545 `test-backend-ops` MUL_MAT |
| HI02 CMake/ABI | done — all 5 rejection rules verified by real `cmake` runs |
| HI03 catalog | done — 26 AMD architectures, deterministic manifest hash |
| HI04 dispatch hook | done, exercised |
| HI05 signature/blake2b | done — 7/7 vectors match Python `hashlib` |
| HI06 MMQ forced-J | done, verified |
| HI07 MMVF forced block size | done, verified |
| HI08 MMF forced nwarps | done, verified |
| HI09 MMVQ geometry | **done, verified on gfx1100** — routing built and tuned; forced geometries beat native |
| HI11 replay cache | done — loader **and writer**; full chain verified, 100% coverage, 0 misses |
| HI13 collection points | done — tg coverage 66.9% → **100%** on 2B, 9B and 27B |
| HI10 record mode | **done** — JSONL, inventory, SQLite; 92 signatures off the real config |
| HI12 tuning engine | **full sweep completes on gfx1100** — 1151 signatures, 0 failures; cross-arch validation outstanding |
| HI09b resource blacklist | not started |
| HI14 multi-GPU | partly done — 2× XTX tensor split + RCCL + MTP verified; gfx1201/gfx1030 untouched |
| HI15 production hardening | mostly done — cache export and `replay-slim` work; slim runtime untested |
| HI16 test suite | partial — 12 patcher tests, no C++ suite yet |

Full patch set: 16 file patches, applies cleanly from a **pristine** checkout,
and verified against two different upstream revisions (`22dc605` locally,
`6ea215d` on the server) with no adjustment.

---

## SQLite — resolved, no dependency

`libsqlite3-dev` is absent on both machines and `brutus` has no passwordless
sudo, so record mode writes **JSON Lines** from C++ and
`python -m bigcherry.inventory` builds the SQLite database offline with the
stdlib `sqlite3` module. `sql/dispatch-db.sql` is unchanged and remains the
schema of record — only *who* writes it moved.

Three things that buys beyond avoiding the install: a tuning run killed at hour
three keeps everything already flushed; a truncated final line is recoverable
by construction; and standards 9.1 (production links no SQLite) becomes true
because nothing links it, rather than because someone remembered to gate it.

The dead `find_package(SQLite3)` / link logic has been removed from the CMake
patch rather than left as a phantom requirement.

---

## HI09 routing — built, run and verified on gfx1100 (2026-08-05, later)

**Superseded: the section below said this had never been compiled. It has.**
Built `workload-max` for gfx1100 on brutus and ran it. It compiles clean, `nm`
confirms `ggml_hip_mmvq_forced` threaded through the mangled native chain, and
under `GGML_HIP_DISPATCH_MODE=tune` the whole path executes end to end.

## The result, as of 2026-08-05 23:00

**Tuned dispatch is 1.7–4.4% faster than native on prefill, end to end, on the
real workload.** Three interleaved native/replay rounds, same binary, only the
mode differing:

| | native | replay | Δ |
| --- | --- | --- | --- |
| pp256 | 690.8 | **720.9** | **+4.4%** |
| pp1024 | 1029.5 | **1052.0** | **+2.2%** |
| pp4096 | 1253.9 | **1275.4** | **+1.7%** |
| tg128 | 116.8 | 117.0 | flat |
| tg512 / tg2048 | 78.7 / 86.1 | 82.7 / 88.7 | noisy, not established |

Every replay run beat every native run at all three prefill points, no overlap.
Cache fully exercised: 103,168 dispatched, 121 entries, **0 misses**. See RV20.

Underneath it, **10.2% call-weighted saving on matmul time** (RV19), which is
where the prefill gain comes from. **MMQ is 96% of matmul time**; upstream's J
heuristic is right on three of the top five shapes and wrong on two — `J=48`
beats it by 44% on the hottest signature (110,160 calls), `J=16:fb1` by 70% on
another. The wins are in picking a different row of *upstream's own config
table* per shape, not in exotic kernels.

**Do not quote the generation numbers.** tg128 is flat; tg512/tg2048 favour
replay on average but individual rounds contradict.

Three things had to be fixed before any of this could be measured, and each
silently invalidated everything before it: a contended machine (RV11), a build
with `GGML_HIP_RCCL=OFF` (RV15), and tuning on a bench profile that never
produced the hot shapes (RV18).

---

**On the real MTP workload, 79% of signatures prefer a tuned candidate.** Read
`RV10` for that; RV19/RV20 above supersede its aggregate figures. Tuning the actual target configuration
(`llama-server`, `--spec-type draft-mtp --spec-draft-n-max 5`, Q8_0 27B, 2× XTX
tensor split, 3/15 samples) gave **75 forced winners against 20 native out of 95
signatures**. The synthetic `test-backend-ops` sweep gave 8.9% on the same code
— the real workload prefers a tuned candidate roughly **nine times** as often.

The winners cluster exactly where HI09 predicted: `mmvq:q8_0:w6:nw4:rpb4:sk1`
(19.7%), `w4:nw4:rpb4:sk1` (17.1%), `w5:nw4:rpb4:sk1` (19.1%), `w3:nw4:rpb4:sk1`
(14.2%) — **`small_k` geometries at the MTP draft widths**. `small_k` has now
been the standout dimension on two independent workloads, and it is the one that
`_variant_initialiser` silently dropped, so the registry could name it and never
request it. The best geometry in the catalog was unreachable until that was
fixed.

**Do not judge candidate value from synthetic sweeps.** The section below drew a
per-architecture conclusion from `test-backend-ops` and hedged that the real
workload was untested; that hedge was load-bearing, and the real answer is both
far more favourable and concentrated in a different part of the geometry space.

**The tuned winners now reach a running server, and the throughput did not
move.** Both facts matter; see "Getting winners onto the hot path" below.

---

**Per-architecture results on synthetic shapes**, retained because the
architecture dependence is real and worth knowing. Read `RV07`.

Two runs at `SCREEN_SAMPLES=3 FINAL_SAMPLES=15` on idle GPUs, full sweep:

| Winner | RDNA3 gfx1100 (target) | RDNA4 gfx1201 |
| --- | --- | --- |
| `mmvf:f32:*` | 79 | 71 |
| **`mmvq:q8_0:*`** | **24** | **5** |
| `blas:hipblas-auto` | 24 | 4 |
| `mmq:q8_0:*` | 3 | 3 |
| forced total | 103 | 83 |

**MMVQ geometry on RDNA3: median 16.18%, max 25.32%, 20 of 24 wins above 10%.**
The identical code on RDNA4 wins 5 times at ~1–5%. Measure per architecture or
you will draw the opposite conclusion — I did, from the RDNA4 run, before the
RDNA3 numbers arrived.

**`mmvq:q8_0:w1:nw4:rpb4:sk1:v1` alone takes 14 of those 24 wins.** That is a
`small_k` geometry — the dimension `_variant_initialiser` used to drop, so the
registry could name it and had no way to request it. The most valuable compiled
geometry in the catalog was unreachable until that was fixed.

MMVF stays strong and cheap (79 wins, median ~16%, no new compiled code). HI17
gains some support but less than the count suggests: forced BLAS wins 24 on
RDNA3 at median 1.88%, one outlier at 18.6%.

Still the real question, unanswered: record → tune on the actual Q8_0 27B MTP
configuration. Both runs use `test-backend-ops`' synthetic shapes, and none of
the four things that drive the target workload are represented. The dominant
winner being a `w1` geometry is encouraging for the draft-width regime, but it
is not evidence.

Every other number in this document from 2026-08-05 is single-sample and is
correctness evidence only.

**The full sweep now completes: 1151 signatures tuned, 0 failures, exit 0,
2/2 backends passed.** Getting there took five defects, each found only by
running on hardware, each hidden behind the one before it:

| Fix | Sweep reached |
| --- | --- |
| (start) | immediate abort |
| RV01 — candidate src0 type absent from the descriptor | 198 |
| RV02 — MMQ predicate broader than upstream's config tables | 243 |
| RV03 — MMF eligibility ignoring type and arch | 366 |
| RV04 — src1 F32 precondition unchecked in all four families | 810 |
| RV05 — tuner measuring inside HIP graph capture | **1151, complete** |

All five are written up as closed reviews under `docs/planning/`. **Discard any
tuning output produced before RV01** — winners may have been measured on a
different type than they name.

**Read this part if you read nothing else.** Four of the five are one defect
class: *an eligibility predicate that approximates what the build actually
instantiated*, wrong in a different direction each time — type absent from the
descriptor entirely, predicate broader than the tables, type and arch never
checked, a precondition asserted in five upstream files and read in none. Two
did not abort cleanly; RV03 faulted as an HSA hardware exception that took the
queue down and read as a driver fault.

The structural fix is in RV03's conclusion: derive eligibility from the
instantiation set the build already knows — `template-instances/*.cu`, the
config tables the catalog already parses — instead of hand-maintaining
predicates alongside it. Pair it with an HI16 test that launches every
registered candidate against a signature it claims to serve; that finds the
whole class offline in seconds rather than as a device fault 800 signatures in.

The HI09 part-1 abort deserves its keep: it caught RV01, and the decision below
that a miss is fatal rather than a silent native fallback is exactly what made
it visible instead of a quietly wrong measurement.

**RDNA4 (gfx1201) also runs clean: 1157 signatures, 0 failures, exit 0, no
aborts or HSA exceptions.** Two architectures now pass, which is what gives the
arch-dependent fixes (RV02's per-arch table lookup, RV03's `should_use_mmf`
gate) evidence beyond a single GPU.

That comparison **falsified** RV03's own prediction rather than confirming it.
RV03 expected RDNA4 to make MMF ncols 9..16 eligible, since the >8 cap is
RDNA3.0-only. It does not: the eligible-count distributions for float src0 at
ncols 9..16 are identical across both architectures, and every MMF winner on
both was `mmf:native:v1`. The reason is RV06 — all 80 generated MMF candidates
are `mmf:f32`, and MMF's F32 path needs `amd_mfma_available` (CDNA) or
`ampere_mma_available` (NVIDIA ≥ SM80), so under a HIP overlay they are
**CDNA-only**. They would be correct on an MI300; they are unrunnable on all
four GPUs here, and there is no CDNA hardware in this project to validate them
on. RV03's ncols trade-off therefore costs nothing today and only becomes real
once an inventory generates F16/BF16 MMF variants.

Still open: **gfx1030 (RDNA2), the sharpest remaining test** — no WMMA at all,
so MMF should be unavailable for every float type, and it is the first thing
ever to exercise `ggml_cuda_mmq_get_config_rdna2`. The card was in use; rerun
with `HIP_VISIBLE_DEVICES=3` against `~/bc-build-multi`, which is already built
for all three targets.

The original section follows, still accurate on *why* the routing is shaped the
way it is.

### Why the routing is shaped this way

What changed, and why it is shaped this way:

- The forced geometry travels **down the native chain** as one small struct
  (`ggml_hip_mmvq_forced` in `mmvq-autotune.cuh`), through
  `ggml_cuda_mul_mat_vec_q` → `mul_mat_vec_q_switch_type` →
  `mul_mat_vec_q_switch_ncols_dst`, and diverges at the point where every
  launch argument upstream computes is already in hand. The alternative —
  marshalling the arguments in the variant entry point — would be a second copy
  of upstream's stride and fastdiv derivation, drifting silently on every
  release, and any drift would surface as a wrong answer rather than a build
  failure. Principle 6 applied to the one family that needed new compiled code.
- One struct rather than three parameters because it is forwarded through **23**
  call sites in `mul_mat_vec_q_switch_type`, one per quantised type. A future
  geometry dimension then touches the header and two signatures, not 23 cases.
- The `.inc` moved from the end of `mmvq.cu` to just after
  `ggml_hip_mmvq_launch_instance`, because the switch functions below now call
  `ggml_hip_mmvq_find_instance`.

**A miss is fatal, not a fallback to native.** The earlier note here said it
should fall back so a replay cache could outlive its build. That is wrong for
the tuning path: a silent fallback times the *native* geometry under an explicit
candidate name, which is the exact failure the HI09 part 1 abort existed to
prevent. Eligibility (`ggml_hip_mmvq_can_execute`) is what keeps an unbuildable
geometry away from the launch; reaching the abort means the registry and the
compiled instances disagree, which is a bug worth stopping on. Replay-cache
staleness is a *dispatch-layer* concern and belongs there, not in the launcher.

Also fixed on the way: `small_k` was a candidate dimension in the stable name
(`…:sk0:`/`:sk1:`) and in the generated instances, but `_variant_initialiser`
dropped it when packing `ggml_hip_variant_params`. The registry could name a
small-K instance and had no way to request one. It now occupies one of the two
reserved bytes, so the struct size is unchanged.

### Verified so far (offline only)

| Check | Result |
| --- | --- |
| Patches apply from pristine `mmvq.cu`/`mmvq.cuh` | 16 edits, PASS |
| Source audit | 32/32 |
| Patcher regression tests | 12/12 |
| `inventory` profile | lookup emitted with an empty body — compiles with no instances |
| `workload-max` (mtp-inventory, rdna3) | 252 candidates, 98 MMVQ instances |
| `small_k` in descriptors | `{ 2, 2, 2, 0, 0, 1, { 0 } }` for `mmvq:q8_0:w2:nw2:rpb2:sk1:v1` |

### Not verified

Compilation and execution. Build `workload-max` for gfx1101 locally, run
`test-backend-ops` for MUL_MAT parity, then a tuning run on the server.

### Two bugs fixed on the way here, both worth not re-introducing

**Unbounded recursion between HI12 and HI13.** The tuner launches a candidate;
the candidate enters its family entry point; HI13 made that a collection point;
it resolves; which calls the tuner; which launches. The stack died inside the
HSA runtime with no bigcherry frame visible, which is why it looked like a
driver fault. Fixed by holding `ggml_hip_dispatch_scope` for the whole tuning
run, not merely around each launch. **Any future code that launches from inside
the dispatch layer needs that scope.**

**MMQ `fallback` was validated but not forced.** Candidates carry `fallback` in
their identity, but `mul_mat_q_case` derives it from row divisibility, so
eligibility checked `(type, J, fb1)` while the launch used `(type, J, fb0)`.
The config table is sparse in *both* dimensions, so the mismatch reached the
device-side `NO_DEVICE_CODE` guard and aborted. `ggml_hip_mmq_can_execute` now
computes the shape's actual fallback and rejects candidates that disagree.

## Next actions, in order

### 1. Build and run the HI09 routing, then HI12 completes

See the STOP HERE section above. The code is written and every offline check
passes; nothing has compiled it. Build `workload-max` for gfx1101 locally
first — a compile error there is far cheaper to find than on the server.

Two schema fields to add before the first tune, from the prework pack and
absent from my copy: `winner.reason`, `build.compiler`, `build.dispatch_abi`.
(`measurement.gpu_mad_us` and `measurement.host_median_us` are already produced
by the tuner and recorded in its JSONL.)

### 2. HI09b — resource blacklist

Build `full-max` with `GGML_HIP_EXPORT_METRICS=ON` (already an upstream option;
it sets `-Rpass-analysis=kernel-resource-usage`), parse the remarks, map mangled
kernel symbols back to stable names, emit a blacklist the catalog consumes.
Must precede tuning — a spilling geometry that reaches the tuner costs a full
measurement cycle to learn what the compiler already said.

### 3. HI12 — tuning engine

Do not skip the coverage report; see `HI12.md`. Three ways a candidate escapes
measurement (never generated / ineligible / never reached) and two of them look
exactly like success.

Screening retention is specified and was missing from the plan:
`native always + top 3 by median + everything within 10% of best`.

### 4. HI19 / HI17 / HI18 — the taxonomy and the remaining opaque paths

Added 2026-08-05. `docs/planning/active/hip-autotune/FAMILY_MODEL.md` has the
source-level verification; do not re-derive it.

**HI19 first** — the four-record separation (signature / context / candidate /
observation). Establishing it before HI17 and HI18 add fields is cheaper than
retrofitting it onto two new families. It also adds `dispatch_status`, which
matters now: MMVQ is enumerated but gated off, so every "MMQ won" recorded today
may be an artifact of what was reachable rather than a measurement.

**HI17** — BLAS is one candidate standing in for a three-stage plan. Three of
its four hidden choices are llama.cpp's own heuristics, not library internals:
`compute_type` (upstream already ships an env override, so it is provably a free
choice), output conversion (`prefer_f32_output` denies RDNA3 direct F32 output —
that is our gfx1100 rig, on every F16-compute call), and `api_strategy` (four
`cublas*` entry points chosen by an if/else chain sitting under a comment that
admits it is a guess about "some old" GPUs). Start with `compute_type`: the
template `switch` already exists, so it is close to free.

**HI18** — three allreduce implementations, one chosen at backend construction,
never revisited. On the 2× XTX tensor-split config it is on the path of every
split matmul. Needs the server. Implement the no-fallback candidates and the
actual-path telemetry first — a fallback-enabled candidate that silently falls
through makes `winner = nccl` a false statement.

Rejected: WMMA as a family. `rocwmma` is absent from the ggml sources entirely;
`amd_wmma_available` selects paths *inside* MMQ, MMF and FlashAttention. Don't
revisit.

---

## PACK_REVIEW action items — incorporated into plan items

All action items from `docs/reference/PACK_REVIEW.md` have been added to their
owning plan items as notes. Cross-reference:

| Action | Plan item | Status |
| --- | --- | --- |
| A1 | HI02 / HI10 | resolved in code (GGML_HIP_AUTOTUNE_RECORD separate capability) |
| A2 | HI03 / HI06 / standards | resolved in code (J-space from config table) |
| A3 | HI04 | resolved in code (cublas forwarder) |
| A4 | HI15 | noted — export/inspect tools |
| B1 | HI09 | noted — small_k + fusion excluded by design |
| B2 | HI12 | noted — screening retention policy |
| B3 | HI12 / `sql/dispatch-db.sql` | noted — missing schema fields: `winner.reason`, `build.compiler`, `build.dispatch_abi`, `measurement.gpu_mad_us`, `measurement.host_median_us` |
| B4 | HI13 | verified — all 5 collection points covered by patch 0700 |
| B5 | HI11 | noted — bounded-width array cache |
| B6 | HI09b | already covered (resource blacklist) |
| B7 | HI15 / HI16 | noted — export/inspect tools |
| B8 | HI11 / HI15 | noted — restart-only reload constraint |
| B9 | HI05 | noted — divisibility refinement (low priority) |
| B10 | HI02 / `patches/0100_cmake_options/patch.py` | fixed — uppercase variant-set aliases added |
| C3 | general | noted — upstream drift, audit generalization |
| C4 | general | noted — audit survives patches |
| C5 | HI13 | noted — transposed-vector MMVF path |

## Next actions, continued

### 5. HI14 — multi-GPU, on the server

`10.10.100.10` (`brutus`) has **2× gfx1100 (RX 7900 XTX), 1× gfx1201,
1× gfx1030 (RDNA2)**. That covers every row of the validation matrix in one box:
mixed architecture, two identical GPUs, tensor split, and an RDNA2 target the
catalog supports but nothing has ever run on.

As of 2026-08-05 the single-GPU tune sweep passes on gfx1100 and gfx1201;
gfx1030 is still untouched. `~/bc-build-multi` is already configured and built
for all three targets, so the remaining runs need only `HIP_VISIBLE_DEVICES`
(0/1 = gfx1100, 2 = gfx1201, 3 = gfx1030). Nothing multi-GPU has been attempted
yet — every run so far pins a single device.

## THE configuration to optimise

This is the workload that matters. Everything else is a proxy for it.

```text
model  /mnt/vault/llm-models/qwen3.6-27b/gguf/mtp/Qwopus3.6-27B-v2-MTP-Q8_0.gguf

fit                 off          gpu_layers          99
cache_ram           0            ctx_size            245760
kv_cache_type_k     f16          kv_cache_type_v     f16
flash_attn          auto         repetitions         3
ubatch_size         512          batch_size          2048
threads             8
split_mode          tensor       tensor_split        1,1     <- the 2 XTX
spec_type           draft-mtp    spec_draft_n_max    5
spec_draft_type_k   q8_0         spec_draft_type_v   q8_0
```

Four things about it drive the remaining design:

**Tensor split across two identical GPUs.** Standards 5.2 requires signatures to
be built from the *device-local* slice, after splitting. `ggml_hip_make_signature`
already takes the sliced tensors, but this is the first configuration that can
actually prove it: with `tensor_split 1,1` each XTX sees half of each tensor, so
a signature built from the global shape would be wrong and a winner tuned on one
device would be misapplied to the other. Standards 10.2 then says the two, being
the same hardware key with the same local shape, *should* share a winner — so
the correct outcome is one winner used twice, not two identical winners stored
separately. Worth asserting explicitly.

**MTP speculative decoding** (`spec_draft_n_max 5`) produces small, irregular
batch widths — draft widths of 1..5 alongside the verify pass. That lands
squarely in MMVQ/MMVF territory at exactly the widths the explicit geometry
matrix covers (`MMVQ_WIDTHS = 1..8`). It is also the workload most likely to
expose signatures the dense selector never sees, which is what HI13's collection
points were built for.

**245,760 context** means attention shapes dominate late in a sequence and the
hot-signature ranking (standards 7.4) will look very different at 200k than at
1k. Record over a representative long run, not a short one.

**Q8_0 at 27B across 2×24GB** is tight. Workspace filtering
(`max_workspace_bytes`, standards 7.3) is not academic here — a candidate that
wins on time but needs more scratch may not be usable at all.

Sequence I would follow:

1. Record over this exact configuration to get a real inventory.
2. Generate `workload-max` from it and check the coverage report reads 100%.
3. Tune, hot-signatures first.
4. Validate with the same configuration plus HIP graphs enabled.

### Step 1 is done, and here is how to redo it (2026-08-05)

**`llama-bench` cannot record this workload.** It has no speculative decoding,
so it produces `widths: [1]`, no MMVF, no BLAS — 17 signatures. Recording
through a real server with MTP enabled produces 80 signatures across
`widths [1,2,4,5,6,8,10,12,16]`, which is the shape the original
`mtp-inventory.json` (92 signatures) has. The draft widths only exist when
speculation is actually running.

Confirmation that MTP was live: **tg128 was 67.99 t/s through the server against
27.67 t/s from `llama-bench`** on the same model and split — a 2.5× speculative
speedup, not a measurement artefact.

```bash
# 1. server in record mode, target config
GGML_HIP_DISPATCH_MODE=record GGML_HIP_DISPATCH_DB=/tmp/rec.jsonl \
~/bc-build-multi/bin/llama-server -m <Q8_0 27B MTP gguf> \
  --host 127.0.0.1 --port 42099 \
  -dev ROCm0,ROCm1 -sm tensor -ts 1,1 -ngl 99 \
  -c 16384 -ctk f16 -ctv f16 -fa auto -b 2048 -ub 512 -t 8 \
  --spec-type draft-mtp --spec-draft-n-max 5 \
  --spec-draft-type-k q8_0 --spec-draft-type-v q8_0 &

# 2. drive it hard
python3 /mnt/vault/development/llmhosts/llamacpp/bench/run_bench.py \
  --bench-type server-bench --server-url http://127.0.0.1:42099 \
  --model <same gguf> --timeout 300 --upload-dry-run

# 3. inventory
python3 -m bigcherry.inventory /tmp/rec.jsonl --inventory artifacts/<name>.json
```

Two traps in that invocation:

- **`llama-bench` device lists use `/`, not `,`.** `-dev ROCm0,ROCm1` means
  "benchmark ROCm0, then benchmark ROCm1" — two separate single-GPU runs, which
  is why a 28 GB model aborts in `ggml_backend_meta_alloc_ctx_tensors_from_buft`
  and a 16 GB one appears to work. `-dev ROCm0/ROCm1` is one run across both.
  `llama-server` does use commas.
- `run_bench.py` appends to the shared results store under
  `llamacpp/bench/results/` even with `--upload-dry-run`. It added one row to
  `results-qwen35-27b.json` when this was run.

The result is saved as `artifacts/mtp-server-inventory.json`. Generating
`workload-max` from it yields **240 candidates** against 270 from
`mtp-inventory.json` — narrower because a short run never observed width 3.
**The tree is currently generated from `mtp-inventory.json`** (manifest
`d64e79c8`), because that is what every measurement in
`docs/reference/CANDIDATES.md` was taken against. Adopt the server inventory
together with a rebuild and fresh measurements, not separately.

Still to do for a representative record: this ran at `-c 16384`, not 245,760.
Hot-signature ranking at 200k will not look like this.

---

## Working on `10.10.100.10`

**There is no copy and no sync step.** `/mnt/vault` *is* mounted on `brutus`, and
`J:` on the workstation is `\\10.10.100.10\vault`, so

```text
J:\development\llmhosts\bigcherry  ==  /mnt/vault/development/llmhosts/bigcherry
```

are the same directory over SMB. Edit on either side, build on the server, no
`tar` in between. Verified 2026-08-05 by writing a file on one side and reading
it on the other.

Two traps this hides:

- **`~/bigcherry` on brutus is a stale copy** left from before the mount (it sits
  at upstream `6ea215d`, pre-HI09-routing) and an earlier version of this
  document told you to `tar` into it. Doing so silently builds the old code.
  Ignore it, or delete it.
- **Dotfiles written on the server are invisible from `J:`** — the share hides
  them. A `.foo` marker test will read as "different folder" when it is not.
  Test with a normally-named file.
- **Worse: *any* file created by a server-side command may be invisible from
  `J:`**, not just dotfiles, and not just in directory listings — opening it by
  exact path also fails. Observed 2026-08-05: a report written by `cp` on the
  server showed `-rw-rw-r-- audumla audumla` and 14,639 bytes there, and did not
  exist at all from Windows. Files written from the Windows side (owner `mgs`)
  are visible on both.

  So **produce repo files from the Windows side**. If a server-side tool
  generated it, copy it back rather than writing it into the tree in place:

  ```bash
  scp 10.10.100.10:/tmp/thing.md docs/reference/THING.md   # run from Windows
  ```

  This is a silent failure: the server-side command reports success, and the
  file simply is not there for anyone working from `J:`.

ROCm at `/opt/rocm` (7.2), cmake 3.28, ninja 1.11, python 3.12. Note the
workstation also has a gfx1100 (RX 7900 GRE, *not* gfx1101 as earlier notes
said) and ROCm 7.1 for Windows, but every verified build has been on the server.

Standard cycle (`$BC` = `/mnt/vault/development/llmhosts/bigcherry`):

```bash
cd $BC/tools
python3 -m bigcherry audit
python3 -m bigcherry apply
python3 -m bigcherry generate --variant-set workload-max --inventory <inv.json>
cmake -S ../vendor/llama.cpp -B ~/bc-build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON \
  -DGGML_HIP_DISPATCH_REPLAY=ON \
  -DAMDGPU_TARGETS="gfx1100;gfx1201;gfx1030" \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build ~/bc-build -j
```

---

## Things that will bite you

**Editing a patch's *text* is a no-op on an already-patched tree.** The
idempotence guard sees its own output and skips. `git checkout` the target file
first. This cost me three silent no-ops before I noticed.

**Do not use bash heredocs to edit Python containing `\n`.** They mangle the
escape and produce a syntax error, or worse, a silently wrong string. Use an
editor or write the script to a file.

**Patch order matters and is now honoured.** `apply_all` simulates the whole
set in memory before writing anything, so a patch may anchor on an earlier
patch's output (the coverage hook does exactly this). An earlier version
validated against the on-disk file and made that impossible.

**Coverage counting is subtle.** A dispatched launch re-enters its own family
entry point. Count `executed` at whichever site is *outermost* for each route —
`ggml_hip_dispatch_mul_mat` when it handles the op, the family entry otherwise.
Getting it wrong produced 75% for full coverage in one direction and
`dispatched > executed` in the other. The reasoning is in the code.

**`mmvq.cuh` has no include guard upstream.** Including it from a second header
redeclares everything, surfacing as "redefinition of default argument" some
distance from the cause.

**Explicit instantiation macros restate parameter lists.** `DECL_MMQ_CASE` and
`DECL_MMF_CASE_HELPER` each repeat the whole signature, so a new parameter must
be added there too — **without** its default, which is ill-formed on an explicit
instantiation.

---

## Design decisions worth not re-litigating

- **Forced variants are explicit defaulted parameters, not hidden state.** An
  earlier version used thread-locals; production then paid a read on every
  launch for a feature only the tuner uses. See `HI06.md` "Design revision".
- **Fusion is not an MMVQ candidate dimension.** It is chosen at runtime inside
  `mul_mat_vec_q_switch_fusion`; one instance serves both. It belongs to the
  signature (standards 11.1). See `PACK_REVIEW.md` B1.
- **MMQ's J space is the config table, not `range(8,129,8)`.** The tables are
  sparse and uneven — CDNA defines 154 rows to RDNA3's 260. See `PACK_REVIEW.md` A2.
- **The catalog derives everything from upstream**, including the C++
  architecture enum. Anything restated is a copy that can silently disagree.

---

## Getting winners onto the hot path

**Verified end to end on 2026-08-05.** This is the first time tuning output has
ever reached a running inference server; every earlier "replay verified" claim
meant only that the loader rejects bad input correctly.

```bash
# 1. tune (writes <db>.measurements.jsonl -- the only record; llama-server
#    installs its own log callback so `bigcherry: tuned` never reaches its log)
GGML_HIP_DISPATCH_MODE=tune GGML_HIP_DISPATCH_DB=/tmp/t.jsonl \
  GGML_HIP_TUNE_SCREEN_SAMPLES=3 GGML_HIP_TUNE_FINAL_SAMPLES=15 \
  GGML_CUDA_DISABLE_GRAPHS=1 <llama-server …>

# 2. export
python3 -m bigcherry.replay_cache /tmp/t.jsonl.measurements.jsonl \
  --manifest artifacts/<rev>/hip-autotune-manifest.json --output dispatch.cache

# 3. replay -- needs a separate GGML_HIP_DISPATCH_REPLAY=ON build
#    (mutually exclusive with GGML_HIP_AUTOTUNE) from the SAME manifest
GGML_HIP_DISPATCH_MODE=replay GGML_HIP_DISPATCH_CACHE=dispatch.cache \
  GGML_HIP_DISPATCH_COVERAGE=cov.json GGML_HIP_DISPATCH_MISS_LOG=miss.jsonl \
  <llama-server …>
```

Result on the real MTP workload: cache of 6459 bytes holding 95 winners across
49 candidates; **coverage 16082/16082 dispatched, zero misses**.

### But the throughput did not change

| Mode | pp512 t/s | tg128 t/s |
| --- | --- | --- |
| native | 553.16 ± 18.70 | 67.99 ± 6.92 |
| replay (tuned) | 571.03 ± 18.00 | 65.69 ± 7.74 |

Both differences sit inside one standard deviation. **Do not report this as a
speed-up.**

**RV11 now explains why, with numbers.** Aggregate matmul saving across all 95
signatures is **7.6%**, not the 14–20% of the headline winners — those are a
minority:

| Family | Sigs | Saved |
| --- | --- | --- |
| mmvq | 58 | 9.3% |
| mmq | 11 | 5.8% |
| mmvf | 5 | 3.0% |
| blas | 21 | 0.5% |
| **all** | **95** | **7.6%** |

Weighting by real execution counts (mmvq 12246, mmq 3612, blas 172, mmvf 52)
and each family's mean per-call cost gives **~7.5%** — essentially unchanged,
because mmq's high per-call cost and smaller saving offsets mmvq's larger one.

So the null result is arithmetic, not mystery: matmul time falls ~7.5%, matmul
is a fraction of decode wall time, and **the benchmark's own spread is ±10%**.
The expected effect is about a third of one standard deviation. This setup
cannot answer the question in either direction.

To resolve it: measure matmul time directly rather than tokens/sec (the
measurements JSONL already has everything needed); establish what fraction of
decode time is matmul at all, which nothing has measured and which caps
everything; and only then decide whether an end-to-end number is worth the
50+ repetitions it would need.

Note also **BLAS is worth 0.5% across 21 signatures** despite winning often —
counting wins and measuring value are different things, and this weakens HI17's
case relative to how the win count made it look.

### Slimming a production build (`replay-slim`)

Implemented; see RV12. Filters the catalog to the variants a tuning run chose:
**270 candidates → 50** on the real MTP winners (45 winners + the 5 native
wrappers, which always survive because a replay miss falls back to native and
the schema requires one per family).

```bash
python3 -m bigcherry generate --variant-set replay-slim \
  --inventory artifacts/mtp-inventory.json --winners <db>.measurements.jsonl
python3 -m bigcherry.replay_cache <db>.measurements.jsonl \
  --manifest artifacts/<rev>/hip-autotune-manifest.json --output dispatch-slim.cache
cmake -B build-slim -DGGML_HIP_DISPATCH_REPLAY=ON \
  -DGGML_HIP_AUTOTUNE_VARIANT_SET=replay-slim -DGGML_HIP_AUTOTUNE_SIGNATURE_FILE=…
```

**Order matters.** Generate slim *first*, then export the cache against the
slim manifest. CMake does not invoke the generator — it requires the registry
to be in the tree already — so generate must come first.

**Tuning now survives a rebuild (2026-08-05, HI23 interim).** The dispatch key
used to include the manifest hash and source revision, so *any* rebuild that
touched the catalog or bumped upstream moved every key and silently discarded
all tuning — a slim build produced 81 misses whose digests were entirely
disjoint from the 95 tuned ones, while still reporting the cache loaded. The
key is now hardware + signature + objective only, and a manifest mismatch is a
**warning** rather than a rejection: the winners are used and flagged as
possibly stale. Verified on gfx1100 — slim build with a mismatched cache gives
1188/1188 dispatched, 1 miss, and logs:

> replay cache was tuned against a different candidate set … Its winners are
> still valid and are being used, but may no longer be the best available;
> re-tune to refresh them.

Safe because the real guards are per entry, not per key, and both still run:
the loader drops entries naming candidates this binary lacks, and the resolver
re-runs `can_execute` before launching a stored winner.

A cache holds **one generation**; re-tuning replaces it. Per-release history
and newest-wins are still to design — see HI23.

**The size argument does not hold.** `libggml-hip.so` is 67 MiB slim against
68 MiB full, same architecture — about 1.5%. The library is mostly upstream's
own kernel instantiations. Justify `replay-slim` on compile time or on not
shipping unmeasured code, not on binary size. Nobody has timed the build yet.

Runtime replay on the slim build is still unverified — the XTXs were reclaimed
by the machine's production `llama-server` before it could run. Rerun
`/tmp/mtp-slim.sh` when they are free; the cache is byte-identical either way,
so expect the same 16082/16082.

`replay-full` is accepted but behaves exactly like `workload-max`. Either
document that equivalence or drop the name.

---

## How to run the tests

Everything below was run on 2026-08-05 and works. `$BC` is
`/mnt/vault/development/llmhosts/bigcherry`.

### 1. Offline, no GPU (seconds)

```bash
cd $BC
python -m unittest discover -s tools/tests      # 12 patcher tests
cd tools
python3 -m bigcherry audit                      # 32 invariants
python3 -m bigcherry apply --dry-run            # patch placement
python3 -m bigcherry apply                      # idempotent; safe to repeat
python3 -m bigcherry generate --variant-set workload-max \
        --inventory $BC/artifacts/mtp-inventory.json
```

Run all four after touching `src/`, `patches/` or `tools/`. They catch most
mistakes before a 10-minute build does.

**If you edit a patch's *text*, `git checkout` its target file first.** The
idempotence guard sees its own output and skips, so the edit silently does
nothing. Costly and invisible:

```bash
cd $BC/vendor/llama.cpp && git checkout ggml/src/ggml-cuda/mmq.cu
cd $BC/tools && python3 -m bigcherry apply
```

### 2. Builds

**Linux, all three GPUs** (the one to use; already exists as `~/bc-build-multi`):

```bash
cmake -S $BC/vendor/llama.cpp -B ~/bc-build-multi -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_HIP_RCCL=ON \
  -DGGML_HIP_AUTOTUNE=ON -DGGML_HIP_AUTOTUNE_VARIANT_SET=workload-max \
  -DGGML_HIP_AUTOTUNE_SIGNATURE_FILE=$BC/artifacts/mtp-inventory.json \
  -DAMDGPU_TARGETS="gfx1100;gfx1201;gfx1030" -DLLAMA_BUILD_TESTS=ON \
  -DCMAKE_C_COMPILER=/opt/rocm/llvm/bin/clang \
  -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++
cmake --build ~/bc-build-multi --target test-backend-ops -j
```

**`-DGGML_HIP_RCCL=ON` is not optional for anything multi-GPU.** Without it a
`-sm tensor` run falls back to a butterfly allreduce, which sits on the critical
path of every layer and costs **1.5–1.7× end-to-end** — measured against the
production lane: tg128 65.9 here against 112.7 there, pp4096 656 against 1249.
The only symptom is one line among hundreds:

```text
internal AllReduce init failed (n_devices != 2?); falling back to meta-backend butterfly
```

Every dual-GPU throughput number in this document from before 2026-08-05 21:00
was measured without it and is on a crippled baseline. Tuning *results* are
unaffected — the tuner times matmuls in isolation and allreduce is not inside
that measurement — but hot-signature ranking may shift. See RV15.

`workload-max` *requires* `GGML_HIP_AUTOTUNE_SIGNATURE_FILE`; CMake rejects the
combination without it. Useful targets: `ggml-hip` (fastest way to find a
compile error), `test-backend-ops`, `llama-bench`, `llama-server`.

**Windows, on the workstation's 7900 GRE** — works, and is the only idle
gfx1100 when the server's XTXs are busy:

```powershell
$env:PATH = 'C:\Program Files\AMD\ROCm\7.1\bin;' + $env:PATH
$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\7.1'
cmake -S 'J:/development/llmhosts/bigcherry/vendor/llama.cpp' -B 'C:/bcw' -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGGML_HIP_AUTOTUNE=ON `
  -DGGML_HIP_AUTOTUNE_VARIANT_SET=workload-max `
  -DGGML_HIP_AUTOTUNE_SIGNATURE_FILE='J:/development/llmhosts/bigcherry/artifacts/mtp-inventory.json' `
  -DAMDGPU_TARGETS=gfx1100 -DLLAMA_BUILD_TESTS=ON `
  -DCMAKE_C_COMPILER="$env:HIP_PATH\bin\clang.exe" `
  -DCMAKE_CXX_COMPILER="$env:HIP_PATH\bin\clang++.exe"
cmake --build C:/bcw --target test-backend-ops -j
```

- **Use a short build directory (`C:\bcw`).** Anything under the scratchpad path
  exceeds the 250-character Windows object-path limit; CMake warns and the build
  may misbehave.
- **Put ROCm's `bin` on `PATH` when *running*, not only when building.** Without
  it the exe dies instantly with `0xC0000135` (DLL not found), which looks like
  a crash rather than a missing dependency.

### 3. Correctness

```bash
cd ~/bc-build-multi/bin
HIP_VISIBLE_DEVICES=<n> GGML_HIP_DISPATCH_MODE=native ./test-backend-ops test -o MUL_MAT
HIP_VISIBLE_DEVICES=<n> GGML_HIP_DISPATCH_MODE=tune \
  GGML_HIP_TUNE_SCREEN_SAMPLES=1 GGML_HIP_TUNE_FINAL_SAMPLES=1 \
  ./test-backend-ops test -o MUL_MAT
```

Device indices on brutus: **0,1 = gfx1100 XTX, 2 = gfx1201, 3 = gfx1030**.
Expect ~1155 signatures tuned and `2/2 backends passed`. A sweep needs a
measured **568 MiB** of free VRAM; check before running on a shared card.

**Always run `native` as well.** It is the baseline that tells you whether a
failure is yours or the hardware's — that distinction is what identified RV08.

Modes are `native`, `record`, `tune`, `replay` via `GGML_HIP_DISPATCH_MODE`.
Anything unrecognised warns and falls back to native.

### 4. Timing

```bash
GGML_HIP_TUNE_SCREEN_SAMPLES=3 GGML_HIP_TUNE_FINAL_SAMPLES=15   # ~10 min/sweep
```

**Never draw a performance conclusion from a 1/1 run.** It cannot separate a 1%
difference from noise, and it makes RV08's intermittent tolerance failure
reachable. Use an idle GPU — check `rocm-smi --showuse` first; a `llama-server`
on the XTXs will quietly contaminate results.

Other knobs: `GGML_HIP_TUNE_MAX_WORKSPACE`, `GGML_HIP_DISPATCH_DB` (writes
`<path>.measurements.jsonl` with per-candidate `status`, `median_us`, `nmse` —
the fastest way to see *why* a candidate was rejected), and
`GGML_CUDA_DISABLE_GRAPHS=1` (required for a complete tuning run, since tuning
is skipped under graph capture — see RV05).

**`GGML_HIP_TUNE_NOISE_PCT` (default 5) — the noise canary (HI24).** Native and
a forced MMQ candidate at `J == J_best` are *the same kernel*: the patched
`mul_mat_q_switch_J` overwrites `J_best` with `forced_J` and calls one
launcher. So any difference between their medians is measurement error, and the
pair calibrates the harness with no external reference. When divergence exceeds
this threshold the tuner re-measures both interleaved and warns.

Every result records `canary_pct`, `canary_retries` and `canary_pair` in the
measurements JSONL, so a run can be audited offline for whether its timings
were trustworthy. **Check it before believing a narrow margin** — RV21 found
the same kernel reading 14% apart at 3 screening samples and 0.6% apart at 15.

### 5. Coverage

```bash
GGML_HIP_DISPATCH_MODE=replay GGML_HIP_DISPATCH_COVERAGE=cov.json \
llama-bench -m <model> -p 0 -n 64 -r 1 -ngl 99
```

Coverage must read 100% dispatched/executed on token generation. Anything less
means a collection point regressed. Note a sweep with no dispatch mode set
reports 0% dispatched — that is correct, not a regression.

**`dispatched == executed` does not mean the cache was used.** A miss is still
a dispatch, to native, so a fully-covered run can be entirely untuned. That
confusion already produced one wrong conclusion (RV12). Replay builds now emit
provenance beside the totals:

```json
"total_dispatched": 1188,
"replay": { "entries": 1155, "misses": 1, "stale": true }
```

`misses` needs `GGML_HIP_DISPATCH_MISS=native-record` to be meaningful —
without it `ggml_hip_replay_record_miss` returns early and the count is always
zero, which reads as success. `stale` means the winners were measured against a
different candidate set: still valid, possibly no longer best.

### 6. Candidate reference

```bash
python tools/candidate_report.py     # -> docs/reference/CANDIDATES.md
```

Reads the newest manifest plus every log in `artifacts/tuning-logs/`. Drop new
logs there named `<arch>-<family>-s<screen>f<final>.log` (or `-native`) and the
provenance is picked up automatically.

---

## Current handoff update — 2026-08-07

### Replay diagnostics and comparison

Optional per-cache-entry replay diagnostics are now implemented behind
`GGML_HIP_REPLAY_DIAGNOSTICS`. Production replay builds remain free of hit
tracking overhead. Set `GGML_HIP_DISPATCH_HIT_LOG` in a diagnostic replay build
to emit JSONL rows containing dispatch digest, signature digest, candidate name,
and aggregated calls. Shutdown must use the opt-in `POST /shutdown` endpoint so
buffered HIP records are flushed.

Diagnostic replay build:

```text
build/windows-replay-diagnostics
```

Gemma E4B cache and model:

```text
tune4b.dispatch.cache
J:\\llm-models\\gemma-4-E4B\\gguf\\gemma-4-E4B-it-UD-Q5_K_XL.gguf
```

The full nine-configuration server workload produced 21 replay-entry hits and
100 misses. Three `blas:hipblas-auto` entries ran, but the strongest synthetic
tune winners (`mmvq:q8_0:w1:nw1:rpb2:sk0:v1` and
`mmvf:f32:w1:bs32:accf32:v1`) did not execute. Therefore those winners are not
validated for this server signature set; the cache was produced by a different
workload.

A fresh native-vs-replay server comparison was then completed with the same
model and settings. Replay was effectively neutral for generation and faster
for prompt processing:

| Aggregate | Replay vs native |
| --- | ---: |
| Prompt processing geometric mean | **+2.14%** |
| Token generation geometric mean | **−0.05%** |
| Nine-config arithmetic mean | **+1.20%** |

Per-config results were: `pp256 +1.18%`, `pp512 −0.01%`, `pp1024 +0.01%`,
`pp2048 +1.22%`, `pp4096 +8.57%`, `tg128 +0.01%`, `tg256 −0.68%`,
`tg512 +0.32%`, and `tg2048 +0.15%`. The `pp4096` result is the largest current
server benefit, while generation remains within normal run-to-run noise.

Artifacts:

- `replay-hits.jsonl`
- `replay-hit-misses.jsonl`
- `diagnostic-replay-results.txt`
- `compare-native-results.txt`
- `compare-replay-results.txt`

### Next work

- Continue HI27–HI33 routing-transform implementation; fused operations remain
  excluded and disabled paths must have zero routing cost.
- Add transform serialization and replay compatibility to the dispatch cache.
- Add explicit transform-attempt measurements; signature recasting remains
  unproven without them.
- Re-run tuning with signatures matching the production server workload before
  claiming the MMVQ/MMVF synthetic winners on Gemma E4B.
