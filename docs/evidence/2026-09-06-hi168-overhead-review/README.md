# HI168 dispatch overhead review

This review establishes avoidable production instrumentation, not its share of
an end-to-end regression. The owner's reported consistent approximately 0.3%
loss remains unattributed. Zero pp/tokens-per-second regression is the target;
small effects must not be dismissed through an invented materiality threshold.

## Production artifact inspected

Brutus, dual gfx1100, Qwen3.8-27B-Q8_0, build plan
`6f1226a6bdbfa5750b567f3747410b4a`, source slice
`0f753581db8cc7d4bbc54553960c3c90`. CMakeCache.txt and build metadata confirm
AUTOTUNE=OFF, DISPATCH_DIAGNOSTICS=OFF, DISPATCH_REPLAY=ON.

Metadata records server SHA256
`ee535d4ebd37fce27b716970801d98282d54302ebd7e7eca57c8f1009cc0e295`
and libggml-hip SHA256
`c92523f6156d7d31e25df6855f5f402cb55b41575fdec2c8a7eb8620240aa60d`.
These are recorded identities, not a new independent hash verification.

Read-only `nm -C` on the actual library found:

```text
000000000a54add0 T ggml_hip_coverage_count_executed(ggml_hip_kernel_family)
000000000a54adf0 T ggml_hip_coverage_count_dispatched(ggml_hip_kernel_family)
000000000a5a3100 b (anonymous namespace)::g_executed
000000000a5a3130 b (anonymous namespace)::g_dispatched
```

`objdump -d -C --start-address=0xa54add0 --stop-address=0xa54ae10`
confirmed `lock incq (%rcx,%rax,8)` at `0xa54adde` and `0xa54adfe`.
The family hooks in patch 0700 call the executed counter without checking
DISPATCH_DIAGNOSTICS. This path is reached in native mode too. The earlier
HI159 change guarded the dense calls but missed the family hooks.

## Corrections and verification

- Patch 0700 now upgrades each family counter block to a diagnostic-only
  block, including the reentrancy probe. Functional family dispatch remains.
- Patch 0100 excludes coverage.cpp from the production source list; diagnostic,
  record and tune configurations still include it. The dispatcher guards the
  corresponding shutdown report call.
- The runtime fingerprint reads object bytes through unsigned char and memcpy
  into uint64_t, preserving the existing word hash without aliasing violations.
- The retained lab analyzer refuses comparative output without positive final
  tuned launch counts for every measured cache cell. It does not constitute a
  complete measurement-admission framework.

Compiled hook tests link and execute the actual emitted blocks with diagnostic
functions deliberately undefined in the OFF configuration. ON provides those
functions and verifies five counters/probes and four functional family hooks;
OFF verifies the four functional hooks with no diagnostic dependencies.
This is a host fixture, not a full HIP library build.

The focused suite passed 44 tests plus five family subtests. The broader suite,
excluding core/test_tree_activity.py because its existing os.kill(pid, 0) probe
terminates the Windows test process, completed with 3,112 passed, 8 skipped,
155 subtests passed and 14 failures. Those failures concern unchanged vendor
fixtures (HI134, HI85, RD73 and HI104) and the existing patch 1244 catalog header
failure. This is not a clean full-suite result; no vendor fixture was refreshed
to hide it. Raw local results: artifacts/hi168-offline-tests-without-tree-activity.xml.

Real CMake evaluates the upgraded source-list fragment for OFF, explicit
diagnostics, record and tune settings. A compiled -O3 -fstrict-aliasing test
compares the production fingerprint against a word-array reference after
changing each storage byte individually.

The upgrade was applied to isolated local copies of the five family files and
HIP CMakeLists.txt from the running source directory
`/home/audumla/.cache/bigcherry/sources/1ad3df0e2664faf5c46f6d4ea748cb76`.
All six files changed successfully; a second application changed none. Neither
the executing Brutus script nor the shared vendor checkout was edited.

## Review and remaining work

dev-gpt-agent request `req_0f532a3d43da4607`, session
`ses_e679cfbc4bce4a22`, reviewed pushed checkpoint
`hi168-overhead-audit@920f2791f527c662e3d77dd94622bcb54971d738`.
Its findings were incorporated through HI168 review RV136.

Native mode already returns before constructing a signature or doing cache
work. Replay L1 hits avoid native selection and persistent digests. L1 misses
still build persistent JSON/BLAKE2 identities before a mutex-protected L2
lookup, and L2-hit insertion recomputes the fingerprint. Their current
frequency is unmeasured: the old 31.9% hit rate came from eight slots, while
the current logical capacity is 128. Do not use the old rate to justify a new
cache. Descriptor copies and repeated family hooks are real CPU work with
unmeasured end-to-end impact. Warm graph replay bypasses CPU node dispatch.

Next, inspect complete new OFF/ON HIP artifacts and obtain current L1/L2 rates
from a provenance-matched diagnostic companion. Extend the maintained
ab-benchmark with ServerRunner lifecycle support for repeatable server-per-arm
measurements. Separate native build/framework effects from same-binary
replay-without-cache versus replay-with-cache effects; retain build identities,
per-cell graceful-exit and activation evidence, fixed workload, balanced order,
and session-aware uncertainty. MTP additionally requires work equivalence.

The already-running `/home/audumla/tuned-nospec.log` eight-round comparison
does not retain per-cell activation or exit evidence. Preserve it as diagnostic
timing history, not an admitted winner-effect result. No new GPU run or build
was started alongside it.

## Reusable build inspection

The owner's follow-up requests all BigCherry build types, standardized for
future models/topologies. The maintained campaign configuration now defines
`e2e-build-matrix`; TEST.md documents each arm's distinct role and remaining
server/activation admission gaps. Historical control remains instrumented;
the new native build is explicit instead of silently redefining control.

The new read-only `ab-benchmark --inspect-build` was exercised locally against
copied CMakeCache.txt and compile_commands.json from the production build above
(artifacts/hi168-production-build-inspection). It observed 149 HIP compiler
commands, all defining DISPATCH/REPLAY and none defining dispatch diagnostics,
while coverage.cpp was still compiled. It returned 1 with the expected
coverage-without-diagnostics finding. This corroborates the prior actual ELF
inspection; it is not a new patched HIP build or a throughput result.

## Post-push integration review

dev-gpt-agent `req_2bae11dae74b4b01` reviewed `f1fba35c` and confirmed the
four implementation changes, but correctly required consumer corrections:
tune-campaign and ab-benchmark still expected coverage from production replay.
RV138 records this. Tune-campaign now builds the diagnostic companion, checks
source/catalog/generated-registry and non-diagnostic-option parity, validates
on that companion, and records both roles in schema-4 receipts. Production
replay remains stripped. The ab-benchmark production activation/server path is
still unfinished and must not be bypassed with diagnostic throughput or by
skipping its coverage check. Remaining dispatcher diagnostic globals/report
bodies also need actual ELF review before literal all-diagnostics-stripped claims.

Follow-up review `req_7b59ce770e694578` found a pre-existing generated-input
verification gap: the worker verified the original run tree, not the copy
CMake compiled. RV139 addresses this with checks around configure/compile on
the actual copy, an independently recomputed input digest, and a versioned
compiled-copy attestation in build metadata and the runtime bundle. Companion
validation now requires that build-bound digest; old unverified builds fail
closed on reuse. Tests inject copy mutations before configure, during configure
and during compile. The option comparison is requested CMake-option parity,
not a claim about observed compiler state. Broader checks passed 687 tests and
21 subtests (three skips) before the additional pre-configure mutation test.

Owner reported a competing XTX run during the old eight-round comparison;
that adds contention to its existing evidence defects. No performance result
is admitted from it. After the owner released Brutus, read-only ROCm inspection
at 2026-09-06 04:33 UTC showed all GPUs idle, zero allocated VRAM and no KFD
processes. This is a point-in-time check, not a reservation or proof for a future run.
