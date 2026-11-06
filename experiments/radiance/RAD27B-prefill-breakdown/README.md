# RAD27B prefill breakdown (2026-11-04)

**What:** experiment-only layouts to split Qwen3.8-27B cold prefill on the two RX 7900 XTX by cause: qwen27b-2xtx-exact (exact bf16 all-reduce), qwen27b-1xtx (one card, TP=1), qwen27b-2xtx-spec3/5/7 (MTP draft depth), plus an EXP_PROFILE_OPS launcher hook for --profile-ops. Never committed; it lived as uncommitted edits on a worktree.

**Outcome:** recorded on docs/reference/engines/radiance/models/qwen3.8-27b.md: cold prefill of a ~6,880-token prompt 2.57 s with the wht6 wire (2,680 tok/s), 3.18 s with the exact all-reduce, 4.16 s on one card; TP2 is about 1.62x one card, compute-bound on RDNA3.

**Why not merged:** measurement arms only, not serving configurations; the result is already published. The launcher hook became a manifest field (experiment.engine.profile_ops, contract amendment 11).

**Resume:** apply change.patch to engines/radiance/layouts.toml and tools/lab/radiance/launch-flashnext-best.sh at base 509d991b8; with the manifest path, express the arms as manifest settings instead.
