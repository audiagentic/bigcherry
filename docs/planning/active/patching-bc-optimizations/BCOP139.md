---
id: BCOP139
order: 139
plan: patching-bc-optimizations
state: pending
created-at: '2026-10-11T13:10:43+00:00'
created-by: agent
priority: P3
work: S
---

# PRBE21: retire duplicate multi-token MoE rpb2 candidate

## Discovery / disposition

Pinned llama.cpp b11552 `mmvq.cu::mul_mat_vec_q_switch_ncols_dst` sends `has_ids && ncols_dst > 1` to `mul_mat_vec_q_moe_launch` **before** generic geometry selection. That dedicated MoE kernel already hardcodes `rows_per_block=2`; generic `0600_mmvq_geometry` does not alter it. `tools/bigcherry/tuning/catalog.py` already enumerates rpb2 and small_k generic variants. **Terminate PRBE21's proposed multi-token small-K MoE rpb2 implementation and duplicate candidate addition; no new patch or queue.** A hypothetical width-1 expert variant remains conditional on measured, typed critical-path evidence, not approved implementation.

## Owners / exclusions

PRBE21 records this disposition; HI09/0600 and the existing catalog own generic MMVQ geometry; upstream owns the dedicated MoE kernel. PRBE18's SSM/CONCAT path is separate and protected (independent 2026-10-11 03:05 UTC planning commit); PRBE111/1273 owns IQ tuning. Excluded active b11552/0300/1006/1346/1359 pin and patch work, QFP41/1356, QFP48 MMQ, QFP49/1360/1361 and PA48/PA49. PRBE21's last independent edit was 2026-10-08 06:47 UTC; no independent PRBE21 small-K MoE patch, PR, benchmark or queued experiment was identified in the preceding 12 hours.

## Evidence / terminal gate

Source/catalog/0600 inspection: 16 static assertions, one initially overbroad ordering assertion corrected to function scope; six host route cases, no GPU work. Current upstream master retains rpb2. External P100 rpb4 +1.9% sm_60 and vLLM gfx1100 sorted-token GPTQ MoE are different hardware/format mechanisms, not BigCherry gains. No BigCherry small-K MoE speedup measured. Terminal: close multi-token rpb2 as already implemented; only reopen single-token HI09 catalog comparison after measured exposure >=2.9126% E2E and explicit correct-route evidence, with logits/KLD/MTP/replay controls and four sessions x ten ABBA rounds per architecture, CI95-low >=3%, controls <=1%. See PRBE21 for technical details.
