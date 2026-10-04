"""QFP18 lightweight promotion: write patches/<id>/README.md for the Flash-Next production stack.

Each README records, on the current pin: the profile the patch entered and the base it was measured against (the
cumulative promoted base rule), its own incremental evidence (pointer to SUMMARY.md, which holds the numbers), its
activation evidence, and the shared native llama.cpp comparison of the whole stack (one run per pin bump).
The record is appended after any hand-written README content, below a marker, and regenerated in place.
Usage: python tools/lab/flash-next/promote-readme.py [--check]   (--check: report only, write nothing)
"""

from __future__ import annotations

import sys
from pathlib import Path

PIN = "0504396"
NATIVE = """## Native llama.cpp comparison (profile level, pin {pin})

`tools/lab/flash-next/queue-promote.sh` (2026-10-05): native llama.cpp (source `llama-native`, no patches, default
all-reduce) vs production profile v6 (`BIGCHERRY_FEATURES=flashnext-v6`), Qwen3.8 Flash-Next UD-IQ4_XS, 2x 7900 XTX +
R9700, MTP draft on the 6900 XT, 64K context, f16 KV, ub512, 256 greedy tokens, ABA (v6 / native / v6):

| Depth | v6 decode (t/s) | native decode (t/s) | v6 ms/step | native ms/step | v6 vs native |
|---|---|---|---|---|---|
| ~8K  | 81.9 / 84.1 | 64.3 | 37.7 / 36.7 | 49.2 | +29% t/s |
| ~48K | 69.1 / 69.5 | 47.6 | 44.1 / 43.9 | 66.4 | +45% t/s |

Draft acceptance is comparable (172-175 accepted of 239-249 drafted), so the arms did the same work. v6 arms are
greedy-identical to each other; native differs in text (Q8_1 activation paths and deterministic top-k ties change
low-order bits). Native cannot load the 240K f16 deployment at all (its largest load was 192K with q8_0 KV), which
1302/1303 enable. Runs: `/mnt/data/bigcherry-work/runs/flashnext-native-d8k`, `flashnext-native-d48k`.

## Cross-model no-regression (one release build)

The Flash-Next set ships in every release build (validated-enhancements), with every runtime flag at its default.
Qwen3.8-27B Q8_0 dual-XTX production config (-sm tensor, built-in MTP4, default all-reduce), ABBA per depth,
promoted base (A) vs base + Flash-Next set (B), `tools/lab/flash-next/queue-p27b-3.sh` (2026-10-05):

| Depth | A decode (t/s) | B decode (t/s) | A prefill | B prefill | acceptance A / B |
|---|---|---|---|---|---|
| 10K | 70.9 / 72.9 | 72.8 / 73.5 | 1292.8 / 1292.1 | 1292.2 / 1293.4 | 177/310 / 177/310 |
| 32K | 72.2 / 72.3 | 72.2 / 72.1 | 1251.7 / 1253.7 | 1253.2 / 1253.7 | 181/293 / 181/293 |

Greedy text identical between A and B at both depths. 1303 fails closed when BIGCHERRY_ATTN_TS is set for a
non-qwen4exp model (observed on the first 27B attempt), so the flag cannot silently misplace another model's KV.
"""

# patch id -> (profile entered, base measured against, kind of evidence, activation evidence)
STACK = {
    "1291_ar_cpu_root": ("v1", "validated-enhancements base (RCCL all-reduce)",
                         "incremental ABBA, +5-6% decode, greedy identical", "BIGCHERRY_PATCH_HIT patch=1291_ar_cpu_root"),
    "1292_kpool_tail_truncate": ("v1", "v1 base (1291)", "incremental ABBA, -14% ms/MTP step at 80K, greedy identical",
                                 "no flag; kernel census (tail edits only with MTP)"),
    "1294_topk_deterministic_ties": ("v1", "v1 base", "enabler: cross-start greedy identity at 32K/80K; speed neutral",
                                     "BIGCHERRY_PATCH_HIT patch=1294_topk_deterministic_ties"),
    "1297_draft_vocab_trim": ("v1", "v1 base", "incremental ABBA, -8% ms/step at 10K, acceptance in base range",
                              "BIGCHERRY_DRAFT_VOCAB_N set; draft output rows trimmed (draft timing)"),
    "1302_cuda_graph_oom_evict": ("v2", "v1", "enabler: HIP graphs at 240K f16 without OOM; greedy identical",
                                  "BIGCHERRY_PATCH_HIT patch=1302_cuda_graph_oom_evict"),
    "1303_attn_kv_tensor_split": ("v2", "v1", "profile ABBA +3% ~10K, +7% ~80K; enables 240K f16 KV",
                                  "BIGCHERRY_PATCH_HIT attn_ts=... attn_rotate=... at load"),
    "1235_rd09_q81_activation_cache_foundation": ("v3", "v2", "foundation required by 1307-1313 (Q8_1 cache "
                                                  "reservation + GGML_HIP_Q8_1_CACHE_MODE gate); measured with them in the "
                                                  "v3 profile ABBA", "GGML_HIP_Q8_1_CACHE_MODE=on; BIGCHERRY_Q81_TRACE hits"),
    "1307_q81_activation_cache_mmvq": ("v3", "v2", "profile ABBA +3% ~10K, +3.6% ~80K, complete separation",
                                       "quantize_q8_1 183 -> 45 per token (census); BIGCHERRY_Q81_TRACE hits"),
    "1308_qwen4exp_rollback_copy_no_cont": ("v3", "v2", "profile ABBA (with 1307/1309/1310); kernels/token 1270 -> 1212",
                                            "kernel census"),
    "1309_rms_norm_mul_q81": ("v3", "v2", "profile ABBA (with 1307/1308/1310)", "BIGCHERRY_PATCH_HIT patch=1309_rms_norm_mul_q81"),
    "1310_act_q81": ("v3", "v2", "profile ABBA (with 1307-1309)", "BIGCHERRY_PATCH_HIT patch=1310_act_q81"),
    "1311_hc_pre_q81": ("v3", "v2", "enabler/neutral: quantize/token 74 -> 45, kernels/token 1138 -> 1116",
                        "BIGCHERRY_PATCH_HIT patch=1311_hc_pre_q81"),
    "1312_mul_q81": ("v4", "v3", "profile ABBA v3 -> v4 +2.1% at ~8K and ~64K, complete separation",
                     "kernel census: quantize/token 45 -> 30"),
    "1313_scale_act_fuse": ("v4", "v3", "profile ABBA v3 -> v4 +2.1% at ~8K and ~64K, complete separation",
                            "BIGCHERRY_PATCH_HIT patch=1313_scale_act_fuse; kernels/token 1116 -> 1024"),
    "1326_sched_async_host_inputs": ("v5", "v4", "profile ABBA v4 -> v5 decode +9.4% ~8K, +7.8% ~64K, complete separation",
                                     "submit timing 5.7 -> 3.0 ms/round (1319)"),
    "1327_qsa_host_remap": ("v6", "v5 (post-bump)", "neutral/enabler: recovers ~half of the #29819 kernel cost, 0..-1% ms/step",
                            "kernel census: kernels/token 1053 -> 1027"),
}


MARKER = "<!-- QFP18 promotion record (generated by tools/lab/flash-next/promote-readme.py) -->"


def record(pid: str) -> str:
    profile, base, evidence, activation = STACK[pid]
    return f"""{MARKER}

Promotion record (QFP18 lightweight evidence-reuse tier, pin {PIN}). Mechanism, design and the incremental
measurements are in SUMMARY.md; this file records why the patch is promoted.

## Evidence

- Entered production profile **{profile}**, measured against **{base}** (each patch is measured as promoted base vs
  promoted base + patch; the measurement is valid for that base).
- Incremental result: {evidence}. Greedy output identical between arms in every adoption run.
- Activation: {activation}.
- Mechanics: offline patch tests + patch-lint pass on pin {PIN}; whole stack re-confirmed on {PIN} by the native
  comparison below and the v6 adoption screens.

{NATIVE.format(pin=PIN)}"""


def main() -> int:
    check = "--check" in sys.argv
    root = Path(__file__).resolve().parents[3] / "patches"
    for pid in STACK:
        path = root / pid / "README.md"
        if not (root / pid).is_dir():
            print(f"missing package {pid}")
            return 1
        if check:
            print(f"{pid}: {'exists' if path.exists() else 'would write'}")
            continue
        old = path.read_text(encoding="utf-8") if path.exists() else f"# {pid}\n"
        head = old.split(MARKER, 1)[0].rstrip("\n")
        path.write_text(head + "\n\n" + record(pid), encoding="utf-8", newline="\n")
        print(f"wrote {path.relative_to(root.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
