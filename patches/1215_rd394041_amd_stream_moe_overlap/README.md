# 1215 (RD39/RD40/RD41/RD42): honor active HIP stream, per-(device,stream) cuBLAS handles, dedicated concurrent scratch, and MoE shared-expert overlap

## Scope

Fixes `ggml_cuda_op_mul_mat` to honor the assigned stream instead of
hardcoding stream 0, and gives every (device, stream) pair its own cuBLAS
handle (RD39/RD40); replaces in-place branch-node interleaving with a
dedicated, reused scratch buffer for concurrent graph regions (RD41); and
uses that infrastructure to detect the MoE shared-expert join and run the
shared expert on an auxiliary stream during small-batch decode (RD42).

## Why

The single per-device cuBLAS handle's workspace corrupted concurrent GEMMs
sharing it; the pre-existing branch-node interleaving for concurrency was
fragile against allocator reuse. Ported as one net patch because RD40/RD41
are declared prerequisites of RD39, and RD42 of all three, in the original
plan items -- splitting them would leave the graph optimizer in an
uncompilable intermediate state. The upstream fork measured +7.4% tg128 on
Qwen3.6-35B-A3B Q4_K_M (not yet independently reproduced here -- see
"Known limitations").

## Upstream / provenance

Ported from AMD-Ecosystem/llama.cpp PR #36 (merge commit `367c4d04f`,
https://github.com/AMD-Ecosystem/llama.cpp). Fork-only work, not ancestral
to mainline or this project's pin. Reconciled 2026-08-30 against upstream
PR #26574, which independently widened `cublas_handles` to `[device][stream]`
-- the now-redundant cuBLAS-handle-widening hunk was removed and the
destructor hunk rewritten to layer only the still-needed
`concurrent_scratch` free onto upstream's real current destructor body.

## Real hardware evidence (indirect, via downstream dependents' campaigns)

1215 has never been run through its own dedicated correctness/performance
campaign. However, it IS two required patches' hard prerequisite (RD43/1216
and RD44/1217 both `requires = ["1215_rd394041_amd_stream_moe_overlap"]`),
and both of those patches' real PPL-equality correctness producers resolve
1215 as part of BOTH their control and subject compositions (RD43's own
campaign function docstring: "control = 1215 alone"). This means 1215's
real-hardware evidence to date is:

- **Builds cleanly** on real gfx1100/HIP hardware as part of both RD43's
  and RD44's control and subject builds (multiple real campaign runs this
  session and in prior sessions).
- **Does not corrupt inference output**: RD43's control arm (1215 applied
  alone, no RD43/RD44) produced PPL = 561.6933 on gpt-oss-20B/Q6_K against
  the real wikitext2 corpus -- **identical** to RD13's own control-arm PPL
  on the same model/corpus, which does NOT include 1215 at all. Two
  independently-resolved compositions (one with 1215, one without) landing
  on the exact same PPL is incidental but real evidence that 1215's
  stream/handle/scheduling changes do not alter output correctness for
  this model, at least under ordinary (non-`GGML_CUDA_GRAPH_OPT=1`)
  execution.
- **No graph-capture abort under `GGML_CUDA_GRAPH_OPT=1`**: RD43's control
  arm (1215 alone, that env var set) completed without the
  "capturing stream has unjoined work" abort RD43 exists to prevent for
  the *combined* 1215+1216 case -- a real signal 1215's own concurrent-
  region infrastructure is not, by itself, unstable under this condition.

None of this is a substitute for 1215's own dedicated evidence -- it is
real, but incidental (produced as a side effect of validating its
dependents, not by design).

## Real dedicated performance + activation evidence (2026-09-12, GPT-reviewed, req_893357b8c0bd4a4e / req_58a8c6465fd249f1 / req_40c25c3069ab4852)

First-ever dedicated performance campaign for the combined 1215+1216 unit
(1216/RD43 is required with 1215 -- see "Composition" below). Initial
methodology (dual-XTX `-sm tensor`, `GGML_CUDA_GRAPH_OPT` unset) was
corrected after GPT review: RD42's shared-expert overlap is an
**intra-token, single-GPU, batch-1-decode mechanism**, opt-in behind
`GGML_CUDA_GRAPH_OPT=1` (off by default on gfx1100), and requires the
1215+1216 unit together (1215 alone exposes a join-node fusion hazard
1216 fixes). All results below use the corrected condition.

**gfx1100, single GPU** (Qwen3.6-35B-A3B-UD-Q4_K_M.gguf, fits one 24GB
XTX): 6-round interleaved 2x2 (`GGML_CUDA_GRAPH_OPT=0` control vs `=1`
subject). OFF: deltas [1.25%, -0.40%, 0.15%, 0.17%, -3.07%, 0.50%], mean
-0.23%, SD 1.49% -- flat, as expected for an inactive control. ON: deltas
[0.37%, 1.75%, 5.59%, 2.10%, 2.79%, 2.19%], mean **+2.47%**, SD 1.73%,
**all 6 rounds positive** -- a real, reproducible, non-regressing gain.
Smaller than the fork's own claimed +7.41% (measured on gfx1151/RDNA3.5, a
different architecture) but real on gfx1100.

**Direct profiler proof of concurrency** (`rocprofv3 --kernel-trace`,
103,768 real kernel dispatches across 4 HSA queues): computed the real
wall-clock UNION of each queue's busy intervals, then the time-intersection
between the two populous queues (82,750 and 20,170 dispatches respectively
-- main routed-expert stream and auxiliary shared-expert stream). **82.28%
of the auxiliary stream's busy time genuinely overlaps in real time with
the main stream's execution** -- not just event-marker-fired-but-serialized,
actual concurrent GPU kernel execution. Also confirmed via debug-verbosity
log: real `Adding shared-expert stream at node <name> <ptr>` lines (RD42's
own activation marker) fired multiple times during a real decode run.

**Cross-architecture** (same corrected condition, single GPU each):
- **gfx1201** (32GB, same Q4_K_M model, 4 rounds): OFF mean +1.38% (SD
  1.33%, NOT flat like gfx1100's control -- this device's baseline noise
  floor is higher), ON mean +2.59% (SD 2.75%, one round negative). A
  directionally similar but noisier, weaker-evidenced signal than gfx1100;
  not as clean a result.
- **gfx1030** (16GB, smaller Qwen3.6-35B-A3B-UD-IQ3_S.gguf quant since the
  Q4_K_M doesn't fit, 4 rounds): OFF mean -0.09% (SD 0.35%, flat/clean
  control). ON mean **-1.19%** (SD 3.59%, one severe outlier round at
  -6.25%, two small positives) -- noisy and net negative on this smaller,
  older GPU. No clear benefit here; possibly harmful, possibly measurement
  noise from less VRAM headroom -- not conclusively either way.

**Split-mode impact** (gfx1100 dual-GPU, `GGML_CUDA_GRAPH_OPT=1`, 3 rounds
each):
- `-sm layer`: real, consistent **regression**: deltas [-2.47%, -5.34%,
  -5.46%], mean approximately -4.4%.
- `-sm tensor`: flat/neutral: deltas [+0.11%, -0.19%, -0.30%] -- no
  benefit, no harm.

**Production guidance this implies**: the patch's real benefit is
conditional on single-GPU deployment with `GGML_CUDA_GRAPH_OPT=1`
explicitly set. It should NOT be enabled for multi-GPU `-sm layer`
configurations (real regression); it is safe-but-inert under `-sm tensor`.
Cross-architecture generalization beyond gfx1100 is unproven (gfx1201
weaker/noisier, gfx1030 net negative/inconclusive).

## Real bit_identical correctness evidence (2026-09-12, GPT-designed and GPT-approved, req_3e42043eb71a4a92 / req_d6534fe00b8140ca / req_013ff2ae8b0c4c4c)

The contract's own required check is literal `bit_identical` (raw
pre-softmax model logits, not an HTTP/sampler-output proxy). Used the
real `llama-results` tool (`tools/results/results.cpp`, already in the
vendor tree -- `llama_get_logits_ith()` dumped to a GGUF `logits` tensor,
no bespoke tool needed): single gfx1100, `GGML_CUDA_GRAPH_OPT=1`, `-ub 1`,
control=baseline, subject=baseline+1215+1216, identical real 19-token
prompt.

**A first attempt at `-lv 4` found zero activation-marker hits and was
correctly flagged by GPT as an invalid negative finding** -- RD42's
marker is `GGML_LOG_DEBUG`, which requires `-lv 5`, not `-lv 4`. Rerunning
at the correct verbosity found the "Adding shared-expert stream at node"
marker firing **1000 times** in the subject run -- RD42 genuinely
activates under this test shape.

**Real result**: wrote a minimal pure-Python GGUF binary parser (no
external dependency), extracted the raw `tokens` and `logits` tensors
from both output files, and compared byte-for-byte:
- Tokens: both runs produced the identical 19-token sequence (verified by
  direct integer comparison, and independently confirmed via matching
  SHA256 `c3937aa6...dccac` for both).
- Logits: raw F32 tensor, shape `(19, 248320)` = 4,718,080 values =
  18,872,320 bytes. **Byte-for-byte identical** (`bytes == bytes` in
  Python), independently confirmed via matching SHA256
  `788844674b53...b41fec` for both control and subject.

```
correctness_results = {
    "bit_identical": CorrectnessResult(
        passed=True,
        detail=json.dumps({
            "method": "llama-results-raw-logit-byte-identity",
            "tokens": 19, "vocab_size": 248320,
            "logit_values": 4718080, "total_bytes_compared": 18872320,
            "byte_exact": True, "activation_marker_hits": 1000,
            "control": "baseline", "subject": "baseline+1215+1216",
            "graph_opt": 1, "ubatch_size": 1, "hardware": "gfx1100",
            "model": "Qwen3.6-35B-A3B-UD-Q4_K_M",
            "control_logits_sha256": "788844674b53208fb38bb558507dd7401cb2820dce2d9305fb9b42f81b41fecf",
            "subject_logits_sha256": "788844674b53208fb38bb558507dd7401cb2820dce2d9305fb9b42f81b41fecf",
            "control_tokens_sha256": "c3937aa6cdecba2fe1a20add7c8f9d5ab2bbc36bcd4df880a9d3491efdcdccac",
            "subject_tokens_sha256": "c3937aa6cdecba2fe1a20add7c8f9d5ab2bbc36bcd4df880a9d3491efdcdccac",
            "prompt_sha256": "4d63c5981334809e190f2a2f7a98645f967d5b7be70d47204fc4b357bf7e050f",
        }, sort_keys=True),
    ),
}
```

**GPT-approved disposition**: real, decisive `bit_identical` methodology
PASS, real gfx1100 evidence PASS, RD42 activation coverage PASS. Same
caveat as RD43: the pre-authored `RD39-42-STREAM-MOE-OVERLAP` contract
binds a different model (`tierM-gptoss20b-q6k`) than this run used --
valid, real evidence, but not yet the final contract-qualified result
until that binding is resolved (fix the contract's model binding or
confirm `tierM-gptoss20b-q6k` also triggers RD42, then rerun this exact
producer once).

## GPT-reviewed disposition

Performance/mechanism qualification for the gfx1100 single-GPU case is
substantively complete: real E2E gain, activation marker, and direct
profiler proof of temporal overlap all satisfy RD42's own claimed
mechanism gate. This is recorded as a real positive finding on its own
merits (a small, reproducible, non-regressing gain is valid grounds for
this, independent of any specific percentage threshold). Do NOT promote to
`validated` yet -- this project's policy requires a complete validation
package (`validation.toml`, bound Experiment Contract) and persisted,
current-pin-identity-bound evidence via `patch-verify-evidence` before a
state transition. 1215's own contract requires a `bit_identical`
correctness check (not yet run); 1216's requires `backend_reference` (the
existing PPL-equality evidence is valuable but does not formally
substitute for that named check per this project's policy). GPT: "stop
hardware profiling now -- the remaining gaps are package/final-producer
work plus the two named correctness checks, not more performance
evidence." (The cross-architecture/split-mode sweep above was completed
separately, in response to a direct question about generalization, not as
further promotion-path profiling.)

## Known limitations

- Both named correctness checks now have real, GPT-approved evidence
  (`bit_identical` above, `backend_reference` in patch 1216's README) --
  but neither is yet bound into a formal `validation.toml`/Experiment
  Contract producer function (`run_rd39_42_contract_qualification()` /
  `run_rd43_contract_qualification()`, scoped in PRBE35), and both real
  runs used a different model than the pre-authored contract currently
  binds (`tierM-gptoss20b-q6k`) -- resolving that model-binding mismatch
  is the concrete remaining step before either check is the *formal*
  contract-qualified result, not just real supporting evidence.
- `state` stays `"untested"` -- real, substantial, now-complete-per-check
  evidence exists for gfx1100 single-GPU, but the formal validation
  package/contract-binding work is not complete and cross-architecture/
  split-mode generalization is mixed (see above).
