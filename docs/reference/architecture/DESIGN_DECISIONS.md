# Design decisions and gotchas

Architecture decisions worth not re-litigating, plus operational traps that bite.

## Architecture decisions

### Forced variants are explicit defaulted parameters, not hidden state

An earlier version used thread-locals; production then paid a read on every launch
for a feature only the tuner uses. See the historical [HI06 snapshot](../../archive/OVERVIEW.md) only for provenance.

### Fusion is not an MMVQ candidate dimension

It is chosen at runtime inside `mul_mat_vec_q_switch_fusion`; one instance serves
both. It belongs to the signature (standards 11.1). The compiler settled it —
including fusion in the instance name produced duplicate symbols because generated
names collided on identical geometry. See the historical [PACK_REVIEW](../../archive/PACK_REVIEW.md) B1 note only for provenance.

### MMQ J space is the config table, not `range(8, 129, 8)`

The tables are sparse and uneven — CDNA defines 154 rows to RDNA3's 260.
Enumerating all sixteen J values would manufacture candidates that abort inside
`launch_mul_mat_q`. On CDNA that is two thirds of the `q8_0` space. See the historical [PACK_REVIEW](../../archive/PACK_REVIEW.md) A2 note only for provenance.

### The catalog derives everything from upstream

Including the C++ architecture enum. Anything restated in bigcherry is a copy
that can silently disagree with the tree it patches.

### A miss is fatal, not fallback to native

The earlier design said a miss should fall back so a replay cache could outlive
its build. That is wrong for the tuning path: a silent fallback times the *native*
geometry under an explicit candidate name, which is the exact failure the abort
existed to prevent. Eligibility (`can_execute`) keeps an unbuildable geometry away
from the launch; reaching the abort means the registry and compiled instances
disagree, which is a bug worth stopping on. Replay-cache staleness is a
*dispatch-layer* concern and belongs there, not in the launcher.

### One struct through 23 call sites

The forced MMVQ geometry travels **down the native chain** as one small struct
(`ggml_hip_mmvq_forced` in `mmvq-autotune.cuh`), through
`ggml_cuda_mul_mat_vec_q` → `mul_mat_vec_q_switch_type` →
`mul_mat_vec_q_switch_ncols_dst`, and diverges at the point where every launch
argument upstream computes is already in hand. One struct rather than three
parameters because it is forwarded through **23** call sites, one per quantised
type. A future geometry dimension then touches the header and two signatures, not
23 cases. Principle 6 applied to the one family that needed new compiled code.

### SQLite-free C++ via JSONL

`libsqlite3-dev` is absent on both machines and the build server has no passwordless
sudo, so record mode writes **JSON Lines** from C++ and
`python -m bigcherry.inventory` builds the SQLite database offline with the
stdlib `sqlite3` module. `sql/dispatch-db.sql` remains the schema of record — only
*who* writes it moved. This buys: a killed tuning run keeps everything flushed;
a truncated final line is recoverable by construction; and standards 9.1
(production links no SQLite) is true because nothing links it, rather than because
someone remembered to gate it.

### Tuning survives a rebuild

The dispatch key used to include the manifest hash and source revision, so *any*
rebuild that touched the catalog or bumped upstream moved every key and silently
discarded all tuning — a slim build produced 81 misses whose digests were entirely
disjoint from the tuned ones. The key is now hardware + signature + objective only,
and a manifest mismatch is a **warning** rather than a rejection: the winners are
used and flagged as possibly stale. Safe because the real guards are per entry —
the loader drops entries naming candidates this binary lacks, and the resolver
re-runs `can_execute` before launching a stored winner.

### Unbounded recursion between HI12 and HI13

The tuner launches a candidate; the candidate enters its family entry point;
HI13 made that a collection point; it resolves; which calls the tuner; which
launches. The stack died inside the HSA runtime with no bigcherry frame visible,
which is why it looked like a driver fault. Fixed by holding
`ggml_hip_dispatch_scope` for the whole tuning run, not merely around each
launch.

**Any future code that launches from inside the dispatch layer needs that scope.**

### MMQ `fallback` was validated but not forced

Candidates carry `fallback` in their identity, but `mul_mat_q_case` derives it
from row divisibility, so eligibility checked `(type, J, fb1)` while the launch
used `(type, J, fb0)`. The config table is sparse in *both* dimensions, so the
mismatch reached the device-side `NO_DEVICE_CODE` guard and aborted.
`ggml_hip_mmq_can_execute` now computes the shape's actual fallback and rejects
candidates that disagree.

### Derive eligibility from what the build already knows

Four of five hardware-run defects (RV01–RV04) were the same root cause: *an
eligibility predicate that approximates what the build actually instantiated*,
wrong in a different direction each time — type absent from the descriptor
entirely, predicate broader than the tables, type and arch never checked, a
precondition asserted in five upstream files and read in none. Two of them
faulted as HSA hardware exceptions that took the queue down.

The structural fix: derive eligibility from the instantiation set the build
already knows — `template-instances/*.cu`, the config tables the catalog already
parses — instead of hand-maintaining predicates alongside it. Pair it with an
HI16 test that launches every registered candidate against a signature it claims
to serve; that finds the whole class offline in seconds rather than as a device
fault 800 signatures in.

### The size argument doesn't hold for replay-slim

`libggml-hip.so` is 67 MiB slim against 68 MiB full, same architecture — about 1.5%.
The library is mostly upstream's own kernel instantiations. Justify `replay-slim`
on compile time or on not shipping unmeasured code, not on binary size.

### A tuned candidate can break code that never runs it — check downstream assumptions, not just the candidate's own output

Per-candidate correctness proves the tuned kernel computes the right numbers.
It does not prove nothing *downstream* of that kernel call silently assumed a
property of the *native* kernel that the winning candidate doesn't share. Three
concrete mechanisms found so far, none requiring the downstream code itself to
be tuned:

- **Graph-capture identity (proven, not hypothetical — RD73).** RD73 changed
  the CUDA/HIP graph cache key behavior for a tuned matmul candidate. MTP
  speculative-decode's graph capture/replay path assumed a stable
  kernel-launch identity for a given dispatch; the winning candidate's
  different launch signature broke that assumption. It showed up as a ~2%
  *end-to-end* regression under the exact workload where graph replay
  matters, despite the candidate's own per-call output being numerically
  correct. Demoted (`patches/1233_rd73_stable_graph_cache_key`,
  `docs/planning/completed/hip-autotune/HI162.md`) specifically because
  per-candidate correctness evidence alone didn't catch it.
- **Precision propagation.** `numerical_class` (`exact_baseline` /
  `equivalent_within_backend_tolerance` / `reduced_precision`, HI17) exists
  because a tuned BLAS candidate can pick a different accumulation type than
  native. Whatever precision the winner actually produces is what every
  downstream op (norm, residual add, next layer) receives — not what native
  would have produced. This is gated at candidate-eligibility time
  (reduced-precision candidates rejected when the request's `prec` field
  demands F32), not discovered after the fact.
- **Fusion eligibility.** The BLAS candidate identity carries a `glu_op` /
  fusion field: whether GLU activation runs fused *into* the matmul call, or
  as a separate downstream kernel, is itself a dispatch-time decision. A
  change in which candidate wins can change whether a neighboring,
  never-tuned kernel runs at all, or with different inputs than before.

A fourth, currently unproven but structurally real risk: shared **workspace/
temp-buffer pool** pressure — a candidate that requests a different temp
buffer size than native draws from the same pool allocator neighboring
kernel launches use, even when those neighbors are never themselves tuned.

**Consequence for every future tuning-family item (HI173, HI174–HI179 and
beyond):** a candidate's own correctness/parity gate is necessary but not
sufficient. Where the candidate sits next to graph capture, precision-
sensitive consumers, or fusion-eligible neighbors, the gate needs an explicit
check for *those* properties too — RD73 is the proof that "the numbers match"
was not enough on its own.

## Operational gotchas

### Editing a patch's *text* is a no-op on an already-patched tree

The idempotence guard sees its own output and skips. Do not reset the shared
vendor tree to re-apply a changed patch; use the explicit
`bigcherry.patch.validation_campaign` workflow to validate the changed
representation in isolated source materializations.

```bash
# inspect existing validation evidence
PYTHONPATH=tools python -m bigcherry patch-validate <patch-id>
```

### Do not use bash heredocs to edit Python containing `\n`

They mangle the escape and produce a syntax error, or worse, a silently wrong
string. Use an editor or write the script to a file.

### Patch order matters and is now honoured

`apply_all` simulates the whole set in memory before writing anything, so a patch
may anchor on an earlier patch's output (the coverage hook does exactly this). An
earlier version validated against the on-disk file and made that impossible.

### Coverage counting is subtle

A dispatched launch re-enters its own family entry point. Count `executed` at
whichever site is *outermost* for each route — `ggml_hip_dispatch_mul_mat` when it
handles the op, the family entry otherwise. Getting it wrong produced 75% for full
coverage in one direction and `dispatched > executed` in the other.

### `mmvq.cuh` has no include guard upstream

Including it from a second header redeclares everything, surfacing as
"redefinition of default argument" some distance from the cause.

### Explicit instantiation macros restate parameter lists

`DECL_MMQ_CASE` and `DECL_MMF_CASE_HELPER` each repeat the whole signature, so a
new parameter must be added there too — **without** its default, which is ill-formed
on an explicit instantiation.

### `llama-bench` device lists use `/`, not `,`

`-dev ROCm0,ROCm1` means "benchmark ROCm0, then benchmark ROCm1" — two separate
single-GPU runs. `-dev ROCm0/ROCm1` is one run across both. `llama-server` uses
commas. This makes a 28 GB model abort in meta allocator and a 16 GB one appear to
work, because each GPU runs alone.

### `run_bench.py` appends to the shared results store

Even with `--upload-dry-run`, it writes a row to `llamacpp/bench/results/`.

### Server-side files can be invisible from Windows SMB

Not just dotfiles — any file created by a server-side command may not exist from
the Windows side. Produce repo files from the Windows side, or copy them back:

```bash
scp "$BC_HOST:/tmp/thing.md" docs/reference/THING.md   # run from Windows
```

This is a silent failure: the server-side command reports success, and the file
simply is not there for anyone working from `J:`.

### Use short build directories on Windows

Anything under the scratchpad path exceeds the 250-character Windows object-path
limit; CMake warns and the build may misbehave.
