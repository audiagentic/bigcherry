# LTX-2.5 TP3: agent resume and merge handoff

**2026-11-22 current integration:** Released Radiance 2.2.0 / ABI44 is now on main at `b80f37d3e`. The active continuation worktree is `main.worktrees/ltx25-tp3-main22` (branch `agent/ltx25-tp3-main22`), rebased by isolated cherry-picks rather than carrying obsolete bump conflicts. C01 and C02 native real-model declaration/host-graph tests pass; see `docs/evidence/2026-11-22-ltx25-main22-c02/README.md`. The older PR #1190 and branch named below are historical foundation context, **not** the checked HEAD or a completed TP3 engine. Planning and Ledger lifecycle records still require their authoritative MCP services. Do not enable production TP3 on these tests.

**Scope:** resume BigCherry PR #1190 (agent/ltx25-tp3-foundation) and complete its **foundation merge** under project conventions. Separately continue LT3-01 to LT3-15 implementation as staged production PRs. Foundation success DOES NOT mean a working TP3 LTX engine.

**Last source review:** 2026-11-20; authoritative repository state is discovered at resume time (do not trust embedded commit hashes without fetching). Source evidence was gathered from the frozen Radiance candidate at upstream commit 3deabdaeca17b58bad90d332f4b204cf7bb8a10e in the project work root, not from an installed released engine.

## 1. Current work, tracked locations and ownership

- PR: https://github.com/audiagentic/bigcherry/pull/1190
- Branch: agent/ltx25-tp3-foundation
- Task worktree: workspaces/main.worktrees/ltx25-tp3-foundation (discover its actual path with git worktree list; do not create another or change the primary checkout).
- Design/backlog: docs/design/ltx25-iq8r-tp3.md. This file carries 15 LT3 engineering gap IDs, proposed files, no-card and device proofs, and ordered slices.
- This handoff: docs/design/ltx25-iq8r-tp3-handoff.md
- Lab README: tools/lab/radiance/ltx-tp3/README.md
- Model preflight (real container verified read-only): docs/design/ltx25-iq8r-model-preflight.md; model test is opt-in and uses BC_LTX_IQ8R_MODEL; no container bytes committed.
- Real-model CPU numerical oracle: tools/tests/lab/test_ltx_iq8r_numeric.py runs in explicit-path mode against the local IQ8R container, reading 123 real video/audio QKV and out rows. It checks two FWHT formulations, three-rank output reductions, real BF16 bias-once and group-boundary refusal. Model-specific result: audio 10/10/12 K slicing is not admissible at IQ8R group 256. Full proof/limitations in docs/design/ltx25-iq8r-model-plane-evidence.md. It is not a production TP3 path.\n- Full-model geometry audit: the real model contains 20 consistently shaped IQ8R projections in each of 48 DiT blocks. For 10/10/12 head ownership, four output projections per block (audio_attn1, audio_attn2, a2v, v2a) cannot use 256-K IQ8R column cuts: 192 invalid across all blocks. No such rejection for 8/8/16 or 12/12/8. This is model-source evidence, not GPU acceptance. See docs/design/ltx25-iq8r-model-plane-evidence.md.\n- Real IQ8R plane evidence: docs/design/ltx25-iq8r-model-plane-evidence.md; tools/tests/lab/test_ltx_iq8r_planes.py provides a CI fixture, a negative extent test, and permitted explicit-path verification of 1,306 actual IQ8R planes without running blocked rad-info.
- Three native C++ CTests under tools/lab/radiance/ltx-tp3:
  1. ltx_tp3_reference_test.cpp — head ownership, fused QKV mapping, IQ8R K-boundary checks, dense global RMS, conservative four-card memory admission, two text-handoff sizes.
  2. ltx_tp3_protocol_test.cpp — deterministic guidance collective schedules, STG and isolated skip patterns, leader pass ownership, exactly-once output bias, epoch/shape/cancelled conditioning mailbox.
  3. ltx_tp3_numerical_test.cpp — independent multirow dense-versus-three-shard self-attention and text cross-attention, six asymmetric cases, all-head Q/K RMS, per-head softmax/2-sigmoid gate, and bias-once output reduction.
- The CMakeLists.txt builds all three with C++20, warnings as errors and assertions enabled. tools/tests/lab/test_ltx_tp3_reference.py calls CMake and CTest in the ignored work root to avoid noexec temporary mounts. The Python runner is **test tooling only**.
- Each lab file is registered TRANSITIONAL in docs/reference/tooling/TOOL_DISPOSITION.md; any new lab file must receive an entry. No source under an installed engine was changed.
- This branch merges upstream updates normally (no rebasing or force pushing shared refs). The previous PR head 77c877401cbfb093e52bcb4567da1f8e4413fc08 had PR-title and PR CI passing, but current commits require a fresh CI check.

## 2. Reproducible card-free validation

Use the task worktree, not the primary checkout, and the existing configured work root. Do not start a GPU job or broad production build without a resource lease.

~~~bash
# From the task worktree, using its actual absolute path:
git status --short --branch
git fetch origin
git merge origin/main             # never rebase or reset unrelated work

cmake -S tools/lab/radiance/ltx-tp3 -B "${BC_WORK_ROOT}/ltx-tp3-reference" -DBUILD_TESTING=ON
cmake --build "${BC_WORK_ROOT}/ltx-tp3-reference" --parallel 2
cmake --build "${BC_WORK_ROOT}/ltx-tp3-reference" --target test
PYTHONPATH=tools python3 -m pytest tools/tests/lab -q
PYTHONPATH=tools python3 -m pytest     tools/tests/core/test_tooling_boundaries.py     tools/tests/core/test_source_lint.py -q
git diff --check
PYTHONPATH=tools python3 -m bigcherry.forbidden_terms --staged
~~~

The native test currently exercises arithmetic at synthetic f64 precision for a tiny 4-channel-per-head attention geometry and a separate 128/64 dimension index oracle. **Do not claim BF16 or IQ8R numerical parity**, transformed weight equality, GPU kernel execution, video/audio quality, synchronization liveness or a speedup.

After any change to the design or lab code, run the three native tests and tooling checks again, record the actual counts, and avoid carrying results from a previous commit into the new PR description. The current test test_ltx_tp3_reference.py deliberately expects an executable CTest target in the test work root and does not require real cards. A previous test failed when building under a noexec /tmp; avoid that path.

## 3. Merging the foundation PR — exact blockers and sequence

**Do not merge this PR by bypassing the required Planning MCP / Ledger MCP project workflow.** Earlier session tool access exposed GitHub and the remote shell, but not the authoritative project Planning/Ledger tools. The project worktree command was denied by that shell's allowlist. Availability must be rechecked in the resumed agent harness, not assumed.

1. Read AGENTS.md, docs/reference/agents/WORKFLOW.md, docs/reference/tooling/TOOLING.md and TOOL_DISPOSITION.md from the **current worktree**. Inspect worktree status, HEAD, remote PR head and current origin/main before editing.
2. Use Planning MCP to list existing groups and find an owning LTX/Radiance group; do not create a duplicate group without checking. If none owns the work, create a proper patching-... group and record the foundation + 15 follow-on items with real Steps, Validation and Acceptance Criteria, using uppercase topic IDs. The LT3 identifiers in docs/design are engineering gap IDs, **not** claimed authoritative plan item IDs. Planning creates/manages its own generated files.
3. Use Ledger MCP for the substantive change event: status unreleased, change-class tests/tooling/design, files (the actual changed list), technical and user summary candidates, plan-item IDs. Commit the generated event file along with any generated planning files. Do not handwrite Planning state or release events to mimic MCP.
4. Reconcile current origin/main normally in this worktree, resolving any conflicts without dropping other actors' rows from TOOL_DISPOSITION.md. Check each maintained tooling entry exactly once, including the numerical test. Run all focused checks and the repo's required pre-push checks.
5. Publish via the project's configured audiagentic worktree ship workflow when available; it performs branch sync/pre-push checks and opens/reuses the PR. If using GitHub app as a transport, ensure **exact tree identity** with the locally checked commit, never force-push, and only update the branch when the expected head matches. Do not equate a successful remote object upload with a conventional worktree ship.
6. Verify the GitHub PR points to the checked HEAD (after any normal merge), its PR title check and PR CI are **completed, successful**, and required review/branch protection is satisfied. Resolve all active discussions. While Planning/Ledger are missing or code checks are pending, keep the PR in **draft**.
7. Only once the above holds, mark ready for review and merge by the supported project workflow, verify PR.merged=true and main contains the merge commit; then run the permitted worktree finish command for this task. Do not remove worktrees manually or claim a queued auto-merge is complete.
8. Update the task conversation with exact PR URL, published SHA, observed CI status, merge SHA if actually merged, and the remaining LT3 items. Do not describe foundation tests as GPU acceptance.

**Potential PR conflict:** the branch lagged upstream main by 28 commits when recovered on 2026-11-20. The task worktree subsequently merged the newer origin/main without conflicts. The remote PR must be updated from the now-tested local result for GitHub mergeability to reflect that fix.

## 4. Smallest genuinely productive next production slice (separate branch)

Continue against the **released Radiance 2.2.0 / ABI44 main** with its exact pinned source and current native builder tests. Never edit the frozen candidate's generated source in place and claim permanence: use the existing engines/radiance/patches recipe/patch-package mechanism, with an opt-in nonproduction LTX package first.

**Files/symbols (verified candidate):**

- Upstream arch/ltx2/ltx2.cpp:229–285 Lin::declare presently declares every big linear as RAD_SHARD_NONE. Its LinearFP8 code path accepts a RAD_SHARD_NONE argument. Introduce a narrow LTX-specific rank policy that selects column/row parallel only for the correct projection types, preserving the full-weight dimensions for load and correct rank-local output widths. Do not project arbitrary name substring matching into global owner rules.
- arch/ltx2/ltx2.cpp:1020–1115 attn_declare and attn_core construct QKV, head_gate, per-all-head RMS, RoPE and attention. The first slice can declare three per-bank Q/K/V spans, rank-local gain/gate and output ownership while continuing to **refuse real world_size > 1** until distributed norm/reductions are implemented.
- abi/rad_builder.h:553–572 **already has** rad_weight_shard_span_parts; core/build/rad_builder.cpp:2829–2834 implements it. arch/common/rad_fp8.h:924–945 LinearFP8::span_parts applies equivalent spans to code and scale planes, subject to row scale-block alignment. arch/gemma4/gemma4.cpp:866–890 demonstrates fused Q/K/V head slices. Prefer these exact existing facilities over designing another ABI or a hand-coded quant scale copier. Their sufficiency for IQ8R **still needs a real encoded plane/layout test**.
- arch/ltx2/ltx2.cpp:1063–1093 uses rms_op over *all* in=H*head_dim on Q and K. Implement f32 sum-of-squares across heads, ordered three-rank partial reduction and one denominator per row **before** RoPE. Text cross-attention uses different Q and K row counts; separate collectives unless shapes are validated equal. Not a local per-head RMS replacement.
- arch/ltx2/ltx2.cpp:1490–1536 declares video/audio self/text/cross-modal attention, and :3590–3710 issues it with guidance STG bypass and isolated cross-modal bypass. All three ranks need identical collective sequence. Output projection / FFN down require one ordered reduction, with bias only once.
- arch/ltx2/ltx2.cpp:3153–3166 rejects world_size > 1. Leave it until a verified multi-rank implementation; do not toggle the guard to make a mock green.
- Current archs.toml gfx1030 entry at lines 31–37 permits only helper; no native Gemma on RDNA2. Do not add gfx1030 as a DiT rank or conflate a MoE helper with a text worker.
- engines/radiance/components/libr2/units.txt currently contains only helper kernels; the optional 6900 path needs its own real Gemma4 operator coverage, build and benchmarks.

**First production proof:** a new native LTX builder CTest fixture generates the rank-local fused QKV part declarations against the *real frozen builder*, checks code/scale-plane source mapping and typed layer shape, and rejects a nonmatching model/encoding, without ever accepting TP3 launch. This is the smallest device-free production-facing slice. The lab numerical test added here can be retained as an independent oracle, but the production reader/weight builder must not import it.

Follow with actual global Q/K norms, exact all-reduce and complete graph before adding a TP3 config.

## 5. 6900 XT text worker is optional, not prerequisite

The RDNA2 6900 has nominal 16 GiB but host-level memory/fragmentation/driver reservations may preclude the ~13 GiB Gemma weight resident set. A full two-prompt Gemma bf16 hidden stack is 770,703,360 bytes (~735 MiB) at C=3840 and 48 layers. The final connector contexts would be just 25,165,824 bytes (24 MiB), **but only if all video/audio connector weights and workspaces also fit on the 6900**, which is unproven.

Recommended first path: TP3 DiT with Gemma+connectors on a qualified fast leader; then optional 6900 Gemma, stack transferred once per VCOND to fast-card connectors, followed by normal TP context distribution. Don't accelerate one prompt by causing every video step to wait for a slow text worker. Do not assume PCIe copy latency from bandwidth specs; measure the physical link.

The protocol mock's ConditioningMailbox uses an in-memory FNV-1a checksum, no actual process IPC, fences, GPU data movement, PID or cross-host security. It is **not** a production wire contract. The final native design must bind exact model/build/config, request generation and epoch, shape and layout, explicit completion fence, cancellation, memory cap and measured GPU access/transfer time.

## 6. Acceptance evidence table for future agents

| Gate | Current foundation | Production TP3 acceptance |
| --- | --- | --- |
| Frozen upstream/source pin | inspected source, pinned in docs | exact RUM recipe composition and patch applicability |
| Native C++ invariants | three CTests | real builder/kernel ctests + device execution |
| IQ8R source mapping | index and alignment mock | actual IQ8R data/scales/rotated activation numerical parity |
| Global all-head norm | f64 dense vs 3-rank oracle | f32/BF16 HIP reductions vs TP1 for every attention kind |
| Guidance and cross-modal | schedule oracle | 48-block real collective order under all four passes |
| Text GPU | static budget sketch and mailbox | gfx1030 Gemma operators, peak VRAM and prompt parity |
| Runtime/Python | no engine files modified | production C++/HIP runtime only, native measured YAML |
| Quality | not tested | identical model/seed, video+audio and intermediate checks |
| Speed | not tested | measured TP1→TP3, then TP3→TP3+text; ABBA and transfer timing |
| Review/merge | PR draft until Planning+Ledger | required checks/reviews, verified merge to main |

## 7. Explicit do-not-do list

- Do not turn on upstream TP3 by deleting or commenting out the one-rank guard.
- Do not call a local Q/K RMS equivalent to the all-head norm.
- Do not claim native IQ8R is sharded correctly from row offsets alone; validate every encoding plane.
- Do not assume memory fit on the 6900 because nominal weights are below nominal VRAM.
- Do not use Python in released engine, worker or launch path.
- Do not introduce an independent YAML parser, placement authority, build recipe manager, or lab-to-production import.
- Do not edit Planning MCP records or Ledger events by hand if their tools are unavailable.
- Do not force-push, rebase, reset, use the primary checkout for edits, delete protected source/builds, or publish a PR as merged before verification.
- Do not move foundation numerical mock results into a production performance claim.

## 8. Recommended next agent starting prompt

"Resume BigCherry PR #1190 in its existing isolated worktree. Read AGENTS.md, the workflow/tooling rules, docs/design/ltx25-iq8r-tp3.md and this handoff. Inspect git status, origin/main and PR CI/reviews before edits. Keep the production LTX TP1 guard intact. Complete Planning MCP + Ledger MCP records for the foundation, rerun three CTests and relevant lab/tooling/forbidden-term checks, publish the exact synced commit using the normal worktree shipping workflow and merge only after required checks/review confirm. Report the SHA/merge status. Then start the first production LTX owner-sharding slice from a fresh dedicated branch after checking the RUM frozen ABI and real LinearFP8 span_parts behavior. Do not test GPUs without a resource lease."

## 9. Publication transport note (supersedes the potential conflict note above)

The remote shell has no GitHub HTTPS credentials. The connected GitHub tool published the verified file tree using a new Git merge commit with the previous PR head and upstream main as parents. Thus local worktree HEAD and remote PR HEAD can have **different commit IDs and ancestry even when HEAD^{tree} is identical**. At resume, fetch both branches, inspect full commit/merge graphs, compare exact tree SHAs, and use normal no-force merge/ship procedures; never reset/rebase to make the two histories look identical. The remote PR was synchronized with upstream main after the earlier stale-branch observation. Its latest CI and draft status must be rechecked, not inferred from this note.

### Tool-policy caveat for resumed tests

The remote shell permits the pytest command but does not support a working-directory argument; attempting cd failed, and its jobs interface was denied. A broad lab run using an absolute test path from the shell default CWD failed only tools/tests/lab/test_flashnext_parity.py::ScriptTests::test_compare_needs_the_launch_records (107 passed, 2 skipped); its shell script sources tools/env/bigcherry-env.sh relative to the working directory and hides source failure output. This is **not** proof of an upstream code regression. The three native CTests, focused LTX pytest tests and 36 tooling/engine guards passed separately. A resumed agent should run the entire lab suite from the actual worktree CWD using an approved runner before merge.

## 2026-11-20 component-first harness expansion

The CPU-first staged test plan is `docs/design/ltx25-tp3-component-test-ladder.md`: nine independent native CTests plus optional real-model IQ8R/DiT/connector numerical sampling (227 real rows in total). It maps the existing LT3 components to separate failure gates and the incremental native/HIP/E2E promotion path. Current passes are **not** production TP3 functionality, and the original TP1-only guard is unchanged. The GPU lease was held by another actor during this CPU harness slice.

## 2026-11-20 C00/C01 follow-on

`ltx_tp3_i8a8h_test` adds source-derived CPU BF16/H128/256-group nearest-even i8 activation tests. `test_ltx_iq8r_shards.py` adds real-plane K-shard and fused-bank tests: 768 source output rows, 1,920 legal rank reconstructions, 384 refused sampled row cuts, 288 fused projection entries with 756 sampled bank boundary rows. Video `ff_down` is biasless; audio FFN-down and the six attention outputs each have a bias added once after reduction. These CPU-only validations do not qualify the native loader or HIP i8a8h. The LTX TP1 guard remains in place.

## 2026-11-20 C01 actual native source staging

The first native Radiance diagnostic package is `engines/radiance/patches/rad0060_ltx2_tp3_linear_declare/`. It adds an optional C01Slice to LTX `Lin::declare` and stages row-bank and column weight declarations with real builder calls; the default path and TP>1 refusal are unchanged. Isolated frozen source applicability/idempotence and C++23 object compilation passed. C01 biased output projections are refused, because collective-and-bias-once is the next slice. Evidence and test commands are in `docs/design/ltx25-c01-native-builder-stage.md`. No model activation or GPU proof is claimed.
