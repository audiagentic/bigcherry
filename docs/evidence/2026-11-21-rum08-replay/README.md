# RUM08: Radiance 2.2 replay and prefill-governor disposition

Review date: 2026-11-21. Exact composition source: Radiance 2.2.0, Git `b6b8751cccb0e834a13ae9e5afca824ee9a487aa` at `pinned upstream source checkout`. This is a **host-code and compiled-callback equivalence review**, not a GPU performance or text-identity campaign. Recipe `engines/radiance/recipes.toml` already omitted both replay patches.

## rad0013: archive intact, superseded by upstream

`rad0013_mtp_pass_replay` restored replay in v1.17 by replacing a blanket refusal to record MTP history, draft and speculative verify with an explicit h256 geometry guard. Its validated v1.17 four-GPU five-prompt evidence reports 1.7–2.7% fewer milliseconds per decode step, identical generated text and acceptance (`releases/evidence/rad0013_mtp_pass_replay-promotion.md`). That is historical **v1.17** evidence, not a measured 2.2 result.

| Earlier rad0013 condition / behavior | Radiance 2.2 upstream replacement |
| --- | --- |
| No replay for mixed/pre-fill speculation; fixed-shape decode history, verify and serial MTP drafts eligible | `arch/qwen4exp_fp8/qwen4exp_fp8.cpp:2013-2043`: `pass_key` returns `1 | sel` for **every** pure decode pass, including `draft_pass=-1,2,3` and `n_spec>0`; refuses prefill/mixed MTP and verify, and refuses unproven prefill geometry |
| Distinguish draft/verify widths, rows, group identity, block table pitch, context bound, model/rank and launch state | `core/runtime/ctx.cpp:1013-1090`: `Ctx::pass_key` hashes phase, `draft_pass`, `n_spec`, token/sequence/width/context fields, group geometry, pointers/table pitch/stream identity, arena/epoch and architecture key; unrecordable dynamic/encoder/generator passes return zero |
| Exact QSA boundary changes whether the indexer scores/selects on a verify | `arch/qwen4exp_fp8/qwen4exp_fp8.cpp:2023-2034`: `sel` is keyed separately for full group `kv_used_max_h <= budget + ratio - 1`; `arch/common/rad_block_attn_gated_fp8.h:218-250` uses the same exact host bound to select the work |
| Full-context, h256 decode split count and its cap cannot change underneath a tape | `libr4d/r4d_attn_decode_h256_gqa6.hip:1162-1195`: deterministic split law by query width, head/group count and depth, with `tiles/2` cap, against batch-keyed context/pitch. `core/runtime/issue.cpp:1024-1035` explicitly replaces positive live decode `kv_used_max_h` with the keyed, rounded `max_ctx_len + max_q_len` before launch |
| When a wide query takes the prefill-kernel splitter, its grid must not change within the same recording | `abi/rad_prefill_split.h:14-57` shares the host-pure live split law and grid fingerprint between launcher and callback; `libr4d/r4d_attn_prefill_split.h:66-98` reads the effective keyed bound. A prefill/mixed recording is keyed by actual split grid, whereas a decode/verify/draft launch uses the rounded bound |

`tests/arch_test.cpp:qsa_reads_the_exact_host_bound_and_keys_its_side_of_the_exact_bound` is an upstream source test of score-selection counts and key separation. `tests/runtime_test.cpp:issue_passes_only_proven_kv_host_summaries`, `an_arch_keyed_program_issues_decode_passes_at_the_keyed_bound`, and `the_prefill_geometry_key_reuses_equal_grids_and_separates_changed_grids` cover the issue and split bounds. Those source tests were **inspected**, not run here. New `tools/tests/patch/test_radiance_upstream_mtp_replay.py` compiles the **unmodified upstream** `pass_key` with synthetic model/batch stubs, proving decode/verify/draft callback behavior including QSA exact-side keying and split-grid keying. It separately asserts the host-side batch/issue/split source contracts; its simplified C++ split function is a stand-in, **not** a GPU decoder benchmark or a proof of every kernel split.

Disposition: archive complete historical `rad0013` package, evidence and v1.17 tests without rewriting their contents, plus an indexed archive entry. The 2.2 recipe remains unchanged.

## rad0047: reject and archive; unneeded for valid production wide verify

`rad0047_mtp_verify_replay_summary` was untested, never in a recipe, and had an ABI43-only fixture. Its proposed guard rejects recording if a **wide** verify (`n_spec>0`, `n_tok>n_seq`) has no positive full-KV host summary. The upstream callback deliberately returns the same above-exact-bound key when directly passed zero/null and when passed a valid above-bound summary: `pass_key` itself does not reject null/zero for decode. The new compiled-callback test explicitly demonstrates that fact. The decision therefore rests on whether such metadata is reachable **through the scheduler**, not on the callback alone.

For the exact production Qwen4exp path it is **not reachable**:

1. `core/sched/scheduler.cpp:2347,2463-2467` constructs verify entries with `want=1+n_draft > 0`, `e.n_tokens=want`, `e.ctx_len=r.n_computed >= 0`.
2. `arch/qwen4exp_fp8/qwen4exp_fp8.cpp:754-760` declares the full attention group as `RAD_KV_FULL` (not a runtime-window group).
3. `core/sched/batch.cpp:918-920` rejects an empty plan, and `batch.cpp:841-858` calculates each group's row `used=ctx_len+n_tokens-first`, where the non-windowed full group has `first=0`, then computes `o.kv_used_max_h[gi]` over the rows. Hence the full group of a wide verify has `used>0`.
4. `core/sched/batch.cpp:1346` binds `B.kv_used_max_h` to that computed host vector; `core/runtime/ctx.cpp:1074-1077` passes it to the architecture key. The indexer's QSA side and effective launched context remain in agreement.

Do **not** interpret this as a blanket proof that arbitrary external `RadBatch` producers are safe with null metadata, nor as a proof of hardware replay behavior. An artificial direct callback can supply zero; any new external batch source that bypasses the scheduler must revisit the invariant. In present accepted production scheduling the escape only adds a redundant branch. Reject the unqualified patch, preserve its implementation and old test in the archive and index the decision; do not compose it. Neither the original v1.17 recipe nor the 2.2 recipe is changed.

## rad0036: retain as independently useful scheduling policy

`rad0036_prefill_governor` is still a distinct scheduler feature: `core/sched/prefill_governor.h` and `core/sched/scheduler.cpp:34-80,1847-1857,2079-2082,2323-2325` implement remaining-prompt ordering, `long_tokens` short-prompt bypass, `kind=cap` or `kind=budget;factor=1.5` admission with KV room checks, and `age_ms` starvation reservation. Radiance 2.2's `--prefill-interleave` and `--mixed-step-target-ms` (`core/rad_core.h:870-880`, `scheduler.cpp:2179-2191,3430-3444`) control decode/prefill timing and mixed-step caps but do **not** replicate admission priority, KV-budget or age reservation. The governor is therefore **not redundant** and must not be removed or silently collapsed into a timing flag.

Previous hardware qualification (`engines/radiance/patches/rad0036_prefill_governor/README.md`, VTP42) measured eight ~100k prompts plus waves: 100k mean first-token 299 to ~227 seconds (-24%), and four mixed decode streams 10.35 to 16.6 chunks/s. Those were earlier experiments, not rerun here. Keep the existing `rad0036` patch and current recipe exactly as-is.

## Verification and limits

- Compiled actual 2.2 Qwen4exp pass-key callback and mock-model fixture, checked pure decode, wide verify, MTP history/serial rounds, QSA boundary, non-decode speculation refusal, split-grid keying and source/recipe invariants: **10 passed**.
- `tools/tests/patch/test_patch_registry.py`: **35 passed**.
- `tools/tests/patch/test_vtp42_prefill_governor.py`: **1 passed, 1 skipped** (optional pinned upstream fixture not configured).
- `bigcherry patch-lint --json`: **passed**, zero problems, pre-existing llama.cpp anchor-span warnings.
- `bigcherry --engine radiance patch-rebase-check --all` against exact `b6b8751`: **38/38 CLEAN**, zero failed/blocked/quarantined; both archived replay patches absent from discovery, and `rad0036` clean.
- Broad `test_radiance_patch_packages.py` attempt initially failed on a compiled test executable on no-exec `/tmp`. Moving `TMPDIR` into worktree removed that environmental fault, but the larger suite did not finish before the remote shell's short command limit; it is **not counted as passed**. The focused checks above completed.
- No GPU workload, throughput, decoded-text identity, full architecture test binary or runtime lane was run; a real replay/quality canary and final production lane acceptance remain separate gates (RUM09).
- Planning and release ledger updates were **not** made: the authoritative `ag-planning` and `ag-ledger` MCPs are unavailable to this session. Per `docs/reference/agents/WORKFLOW.md`, do not hand-edit those managed files or invent a ledger event. RUM08 needs that recorded closure via the respective services.

Use this evidence and the archive index for the eventual RUM08 planning/ledger transition; no upstream source, production recipe, or runtime code was modified in this worktree.

## RUM08 production policy after the 2020.0.0 release

The released fixed-depth `flashnext-4gpu` production composition uses
**upstream scheduler authority only**: `num-speculative-tokens: 3`,
`prefill-interleave: 0`, `mixed-step-target-ms: 0`, and no optional
`prefill-governor` admission override (arrival order). The `rad0036`
patch remains present but its special budget, priority and age controls
are dormant for that profile. `rad0013` and `rad0047` remain archived
and absent from the recipe. For long-Qwen profiles, the explicitly chosen
`rad0036` governor owns admission while upstream interleave=2 owns
step spacing; those are distinct controls and not part of this fixed-depth
Flash-Next qualification. No adaptive/suffix/traversal mode is selected.

On the 2020.0.0 prefix, `--step-phase-trace` is available. Its JSONL
contains scheduler step, pass kind, token count, context and HIP intervals,
**not** the key or per-kernel effective `RadArgs`. Hardware playback
evidence therefore requires recording/play counters or logs paired with
the host-compiled real callback/split and upstream issue-bound tests;
trace shape equality alone cannot establish launch-argument equality.
The follow-up evidence and remaining gaps are recorded in
[2026-11-22 RUM08 completion](../2026-11-22-rum08-completion/README.md).
