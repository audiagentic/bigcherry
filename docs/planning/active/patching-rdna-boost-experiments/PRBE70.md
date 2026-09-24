---
id: PRBE70
order: 0
plan: patching-rdna-boost-experiments
state: pending
created-at: '2026-09-09T10:58:27.196577+00:00'
breadth: ''
skill: advanced
created-by: capability-rebaseline-v3
work: S
priority: null
---

# CK-001: Use Composable Kernel profiler as an oracle for hot GEMM signatures

## Description

TODO. No patch/upstream item implements a Composable Kernel offline oracle; no tools/lab/ck-oracle/ directory exists yet. Still relevant for identifying tile/algorithm choices on gfx1100/gfx1201 dense+MoE GEMM. Pure tooling/process, deliberately kept out of the runtime dependency graph.

## Steps

1. Create tools/lab/ck-oracle/manifests/ with a JSON schema capturing hot signatures: {"shape": {"m":..,"n":..,"k":..,"batch":..}, "dtype_a":..,"dtype_b":.., "op": "gemm|moe_gemm", "source": "llama-bench-op-timing", "time_ms_total":.., "calls":..}. Populate it by running llama-bench with GGML_CUDA_OP_TIMING=1 (existing env var, see patch 1203's op-timing instrumentation precedent in patches/1203_rd050607_rdna4_wmma_fa_q6k_mmq/patch.py) against representative dense (qwen35-27B shapes, already used in 1203's test-backend-ops perf cases) and MoE workloads, sorting by time*calls, and taking the top N signatures.
2. Write tools/lab/ck-oracle/run_ck.py: shells out to `${CK_PROFILER_PATH:-ckProfiler}` per manifest entry with the matching CK profiler subcommand (gemm/grouped_gemm for MoE) and the exact M/N/K/batch/dtype, capturing stdout (throughput, best instance/tile config) into tools/lab/ck-oracle/results/<signature-id>.json.
3. Write tools/lab/ck-oracle/run_native.py: runs the SAME signatures through existing native controls -- MMQ/MMVQ (test-backend-ops perf harness, same shapes) and hipBLASLt (if available) -- recording throughput for direct comparison, plus a correctness check (max abs/rel diff) between CK's output and the F32 reference for the same shape (small dedicated correctness run, not the profiler's own perf-only mode).
4. Write tools/lab/ck-oracle/DECISIONS.md documenting, per signature where CK wins: (a) is the winning tile/algorithm choice reproducible in ggml-cuda's own kernel structure without vendoring CK itself (state yes/no and why), (b) if yes, file a new plan item referencing the CK evidence for the narrow specialization (out of scope here); if no, record explicit no-port disposition with the throughput delta and reasoning.
5. Never add CK as a build dependency of the main project; ckProfiler is assumed pre-built and referenced only via env var path, entirely offline/optional tooling.

## Detailed Solution & Technical Design

The oracle's job is comparative offline benchmarking only, producing evidence that feeds separate, later, narrowly-scoped ggml-cuda patches -- it is not itself a patch and never becomes one. Manifest -> CK run -> native run -> decision is a strict one-way data pipeline; committing to CK's tile choice happens only via reproducing the SPECIFIC winning config as ordinary ggml-cuda C++/HIP, never by linking against CK.

## Code Samples & Guidance

run_ck.py skeleton:
```python
import json, os, subprocess, pathlib

CK_PROFILER = os.environ.get("CK_PROFILER_PATH", "ckProfiler")
MANIFEST_DIR = pathlib.Path("tools/lab/ck-oracle/manifests")
RESULTS_DIR = pathlib.Path("tools/lab/ck-oracle/results")

def run_one(sig: dict) -> dict:
    cmd = [CK_PROFILER, "gemm" if sig["op"] == "gemm" else "grouped_gemm",
           str(sig["dtype_a"]), "0",  # layout/verify flags per CK's own CLI -- confirm against installed CK version's --help
           str(sig["shape"]["m"]), str(sig["shape"]["n"]), str(sig["shape"]["k"])]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return {"signature": sig, "stdout": out.stdout, "returncode": out.returncode}

if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for f in MANIFEST_DIR.glob("*.json"):
        sig = json.loads(f.read_text())
        result = run_one(sig)
        (RESULTS_DIR / f"{f.stem}.ck.json").write_text(json.dumps(result, indent=2))
```
(the exact ckProfiler CLI flags must be confirmed against the installed CK build's own --help before this is runnable -- placeholder above.)

## Files

tools/lab/ck-oracle/manifests/*.json; tools/lab/ck-oracle/run_ck.py; tools/lab/ck-oracle/run_native.py; tools/lab/ck-oracle/results/*.json; tools/lab/ck-oracle/DECISIONS.md.

## Validation

Numerical parity (CK output vs F32 reference, max abs/rel diff) recorded per signature; throughput/resource-use comparison table CK vs native winners in DECISIONS.md. No BigCherry patch-lint/rebase-check applies since no patches/ package is produced by this item itself.

## Effort & Risk

S-M; mostly scripting, gated on CK being buildable/available on Brutus (external dependency for the offline tool only, never for the shipped binary).

## Standards

Capability rebaseline v3 REVIEW_PROTOCOL.md; preserve historical provenance.

## Acceptance Criteria

Do not add broad CK dependency; accept only a reproduced narrow specialization or explicitly documented no-port disposition backed by parity/performance evidence.

## Notes

Supersedes: RD88
Migration: capability-rebaseline-v3-2026-09
Successor key: patching-rdna-boost-experiments-rd88

2026-09-24 relevance at b11126: TODO, no existing tooling directory or patch. GPT design request: gateway rejected all submissions this session (VAL-AGW-025 / EXT-GPTAUTO-003); plan authored directly, referencing existing GGML_CUDA_OP_TIMING precedent in patch 1203 -- no GPT request id.

## Change Log

- 2026-09-09T10:58:27.196577+00:00 (created-by): Created by capability-rebaseline-v3
- 2026-09-09T11:15:38.618139+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:standards, section:acceptance_criteria, section:notes

## Ledger-events

- chg_20260909_115759_created-and-populated-the-192_2958
- 2026-09-09T11:58:01.444745+00:00 (updated-by): Updated: section:ledger-events
- chg_20260910_001436_completed-the-planning-rebasel_5794
- 2026-09-10T00:14:43.280599+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-10T03:20:07.380935+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:files, section:validation, section:acceptance_criteria
- chg_20260910_032047_repaired-five-more-active-succ_6361
- 2026-09-10T03:20:47.919311+00:00 (updated-by): Updated: section:ledger-events
- 2026-09-24T02:34:41.598269+00:00 (updated-by): Updated: section:description, section:steps, section:detailed_solution, section:code_samples, section:files, section:validation, section:effort_risk, section:notes
