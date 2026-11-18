# Radiance performance and maintainability review

Use the [implementation handoff](IMPLEMENTATION-HANDOFF.md) for edit order, exact checks and stop conditions.

Reviewed source at `a1a48cc0a6bc14a4352e8c9120397410bcab98ae`. Three GPT-6.2 Astra passes covered runtime, deployment and tooling. A separate source pass counted file reads during Radiance recipe resolution. The review examines maintained paths, active patch enrollment, and representative lab/campaign consumers. Generated schema code, vendor trees, historical experiment variants and archived evidence are context where needed, rather than a line-by-line audit. The [prior Radiance migration review](../radiance-upstream-2026-11-18/ASTRA-CONSOLIDATION.md) owns the upstream YAML and patch-port decisions.

| Area | Source findings and implementation notes |
| --- | --- |
| Radiance inference, helpers, kernels and active patches | [Runtime](RUNTIME.md) |
| Serving lifecycle, resource probes, host settings and install | [Deployment](DEPLOYMENT.md) |
| Tools with a direct Radiance build, patch or experiment caller | [Tooling](TOOLING.md) |
| Recipe parsing and dormant build enrollment | [Build configuration](CONTROL-PLANE.md) |

## Suggested execution order

1. **Validate measurement inputs.** Bind benchmark evidence to the installed runtime members and repair active lab inputs that fail preparation. Otherwise a timing result may describe a different plugin build or an experiment that never reaches measurement.
2. **Bound warm-start work.** Instrument bundle-identity calls and bytes read before changing the cache path. Stream hashes and reuse an operation-scoped verified identity only while the same artifact generation remains pinned. Preserve exact identity plus fresh room and topology admission. Separately, split stable device identity from volatile VRAM/link observations in the stored replay scope. Prove that a harmless occupancy change keeps replay eligible and that a genuine room or topology failure still refuses. The same operation-scoped reuse principle applies to install verification under its existing lock.
3. **Prove the consumer-only MoE GPU path.** Keep one checked retirement path for nonconsumers and the current route/inject work on the consumer. Verify remote-row accounting and replay ordering before taking timings. Treat helper publish compaction as a separate change so any benefit can be attributed.
4. **Remove avoidable polling and build work.** Tail each helper log incrementally and preserve partial-line, process and segment readiness semantics. Apply the existing CMake jobs policy to the runtime/component build paths that bypass it, and verify the effective limits before interpreting CI trials. Resolve recipe declarations once per load. Review dormant n-gram artifact consumers before changing release enrollment. These can be landed independently of GPU changes or the upstream bump.
5. **Tune only the measured bottleneck.** Accept a runtime policy setting if deployments benefit from different values and the existing owner can validate it. Prefer launch-time selection or derivation when the right value already follows from device, format, shape or admission inputs. Stage any change that alters numerical order, synchronization or placement after a representative baseline.

## Patterns to reuse

| Repeated problem | Mechanism | Keep explicit |
| --- | --- | --- |
| Each caller reinterprets settings | Resolve a typed input once at the existing owner, pass it to its consumers and log the effective non-secret values | Precedence, stage overrides, identity and the point at which a new invocation sees config edits |
| A hot loop repeatedly picks the same rule | Select a strategy or typed kernel variant once per stable shape, device or generation, then call it directly | Shape/ISA and layout eligibility, narrow fallback, lifetime and invalidation |
| Multiple sources know the same geometry or capacity | Derive strides, rank order, candidate geometry and memory demand from authoritative model/device/format inputs | Units, bounds, alignment, overflow, capability checks and the distinction between estimate and live admission |
| Tiny variants duplicate a mostly shared implementation | Use a short descriptor table, indexed arrays for bounded IDs, or a shared compile-time kernel body with explicit specializations | ABI, ordering, fences, numerical semantics and independent cross-process validation |
| Every poll or operation repeats historical work | Carry an invocation snapshot, generation-scoped state or validated incremental cursor | Crash recovery, corruption checks, authorization, freshness and the owner of persistent state |
| Timeouts and thresholds are embedded at use sites | Give deliberate operator choices a typed setting in the existing profile with units, default, range and observable effect | Protocol limits, enum/format IDs and fixed device properties stay named in code. Safety bounds remain enforced. Experiment thresholds remain tied to evidence. |

For each change capture one structural measure first: launches, bytes copied or hashed, file/probe calls, lock time, wait time, or allocation count. Run a small negative control that should not take the new path, then check the actual selected path for correctness and timing. Report the relevant tail metric as well as throughput. Reuse existing fixtures, receipts and campaigns and expand the test only when it resolves a particular risk. No runtime performance, numerical or hardware result was produced by this source review.

For GPU candidates, use a short HIP API/kernel/copy timeline to distinguish CPU submission, GPU execution and transfer effects. Keep profiler runs separate from unprofiled comparison runs and avoid summing overlapping dispatches as wall time. [AMD rocprofv3 tracing guidance](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-rocprofv3.html)
