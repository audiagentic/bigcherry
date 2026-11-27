# Shared KV capacity and independent session limits

Status: proposed engine contract; not implemented or qualified.
Date: 2026-11-27. Owner scope: Radiance generic serving and memory planning.
Implementation detail: [code-level guide](radiance-shared-kv-capacity-code-guide.md).
Section K records the verified Radiance 2.3 reuse map and consolidated test ladder;
implementation targets the final merged bump, not the historical 2.2 source.
RSC17 owns startup planning and RSC18 owns runtime; RSC16 integrates and qualifies.
The client harness owns client-specific context limits. The engine owns physical
capacity, maximum request length, active sequence slots and safe scheduling.
No model-specific exceptions, static reservations of one max window per client,
or weakening of existing safety checks.

## Evidence and existing owners

On installed 2020.23.1, Qwen27B IQ8R TP3, BF16_I8 and yarn:2, max-model-len
524288, stock power: native declaration resolved YaRN to 524288. Admission
refused 16, 8, 4 slots and 4 slots with batch 512, with RAD_E_NOMEM from
refit_serving_pools. The reported backing was 5%, 11%, 24% of nominal
worst case, not measured successful capacity. Historical 818896 tokens
(ENG06) are not current guarantees. No retrieval completed. Preserve original
logs, YAML, release identity and initial missing FFmpeg path failure.
RSC12 owns named variants and measured-cache reuse; ENG04 owns format and
quality matrix; ENG03 owns decode scaling; MEN17/MEN26 release acceptance.
Do not modify in-progress items or duplicate their evidence. RSC03/RUM09
are historical admission work; retain their invariants. Native YAML remains
the single authored interface, per existing native launch design.
## Contract

Separate L (maximum logical tokens per request including output), S (maximum
active slots), and C (aggregate logical paged-token target). C=auto means
largest safe backing after fixed allocations and configured headroom, not S*L.
Resolve at startup from measured facts, then track actual occupied pages at
runtime. No automatic live physical pool expansion beyond sealed capacity.
An explicit aggregate target means minimum guaranteed shared capacity; if
unattainable refuse with limiting rank and measured deficit. Report achieved
capacity separately. No implicit host tier or reduced precision. L must be
within supported positional geometry and C >= L where one full request is
guaranteed; no promise that S full requests fit. C counts physical occupied
pages with prefix sharing, fragmentation and block granularity, not summed
client logical lengths. Retained prefixes count until reclaimed. Fixed state
and table allocations remain accounted. Expose real capacities by group,
rank and tier; avoid advertising a scalar where groups have different limits.

Proposed native model field: kv-cache-capacity: auto or positive token count.
Name/type subject to upstream interface review; preserve existing omitted-field
semantics until migration is proven. No x-bigcherry duplicate. Review the
existing calibrated kv_backing_floor fraction: in legacy mode preserve it;
in explicit shared mode translate or retire it via versioned calibration
contract, never silently ignore an existing guarantee. Keep percentage of
S*L only as optional worst-case diagnostic, not a mandatory demand. Include
new intent, resolved policy, actual pool sizes, topology, cache format,
rotary and geometry in measured-cache and replay identity.
## Allocation

Audit actual bytes from KVManager::plan_groups, fixed_bytes, paged_split,
pool_mib_for_tokens, runtime-window/drafter and streamed mirror accounting.
Derive per-rank fixed ledger (driver drift, headroom, arena, staging, queues,
weights/expert floor, recurrent/conv state, tables, speculation, pinned tiers)
and admissible elastic weights/KV from the SAME rank facts. Current
vram_budget.cpp common split mixes min capacity and max costs; rad0011
refit occurs before the existing cheaper-rank resplit. Instrument the precise
shortfall before deciding it is avoidable. Solve a common logical page target
by monotone search over each rank's exact rounded byte feasibility, keeping
all fixed costs and tier floors. Allocate independently and intersect actual
group capacity/geometry; ensure identical logical block table semantics.
Retain deterministic and no-lender behavior, expert residency and measured
staging rules. Reuse existing rank receipts and S5 typed room refusals;
report exact bytes, rank, requested vs fitted tokens and actionable remedies.
Do not simply replace max with sum or divide by rank count.

For shared/unified RAM, multi-model fleet grants, host retention and streamed
KV, charge actual common physical resource once, respect host/pin reserves and
per-tier capability. Host prefix retention is not an active host KV mirror.
Unsupported architectures/group combinations fail closed with explanation.
Keep generator branch minima, sliding windows, recurrent-only and zero-page
groups correct. Scope initial production support to qualified paths; all
other paths retain legacy behavior until their gates pass.
## Runtime and observability

Reuse existing paged allocation, prefix cache, preemption and prefill governor.
At request admission check L, S and actual per-group allocatable pages plus
fixed slot state; distinguish queued requests from active slots. Account for
actual prompt, bounded next-step output and speculative/temporary pages (not
each client's maximum output or context); grow incrementally
with atomic multi-group/rank allocation or rollback. Prevent deadlock when
several requests fill cache: evict unreferenced prefixes first, then existing
bounded preempt/resume policy or queue/refuse with actionable capacity reason.
Never truncate a request or silently evict live state. Fairness/aging and
short-request latency must hold when long prefill is active; no permanent
reservation for an unused long client. External harness limits are not trusted
physical reservations. Handle cancellation, disconnect, restart, prefix reuse,
OOM races, multi-model concurrent grants and restored manifest mismatch.
Expose logical max, slots, occupied/free/evictable pages and bytes by group,
fixed slot cost, actual admitted capacity, high-water, queue/preemption events
and limiting rank through existing explain/metrics and managed evidence.
Avoid a separate cache allocator, new telemetry stream or policy scheduler.

## Implementation and qualification sequence

A: reproduce failure and instrument all per-rank ledger terms; compare legacy 256K and 512K with 4/8/16 slots on same release, capture actual minimum and aggregate fit without changing admission. Prove whether 512K is physically possible. No arbitrary reduction of safety headroom.
B: upstream-first native option and typed policy, common page target solver, rounding/overflow/monotonicity tests with heterogeneous, homogeneous, unified, fleet and all KV group types. Extend existing plan schema/receipts/replay validation with versioned compatibility; one source of truth, no second resolver. Keep legacy config behavior and test old stored plan refusal/reuse.
C: scheduler/accounting and metrics integration; CPU tiny-model and fake-rank tests for uneven lengths, concurrent allocation rollback, prefix sharing, full pool, cancellation, speculative growth, fairness, no deadlock.
D: GPU under short queued leases, standard release build and managed records. Stock first, then qualified efficient separately. IQ8R TP3 BF16_I8 yarn:2 first; one 512K (prompt+output), then 512K + 128K and additional mixed 32/64/128K clients up to measured capacity. Verify actual token counts, needle depths 10/50/90%, short-context controls vs unscaled, output quality and DFlash2 acceptance, TTFT/decode/prefill, high-water, preemptions, cancellation, clean teardown. 16 slots is a test target, not a promise.
E: homogeneous TP1/TP2, asymmetric TP3, FlashNext with helpers, other model families and formats (BF16, FP8, BF16_I8, supported lossy formats), streamed and fleet cases; use ENG04/ENG03/MEN17 cohorts rather than repeat them. Legacy default regression gate: matched correctness, memory, throughput and latency; explicit owner-approved thresholds before promotion. Unsupported cases remain legacy. Publish evidence and release notes, only then opt in named configs. If physical capacity is insufficient, retain refusal and report exact limiting resource; evaluate existing offload/streamed options separately with performance and quality evidence.

## Evidence preservation and additional invariants

Original failed launch inputs, logs, host facts, receipts and release install record are retained in host archive RSC16-yarn2-int8-20261127 (40 files; SHA256 inventory verified). This is raw diagnostic evidence, not a successful managed qualification.

Do not treat kv_one_session.bytes / L as marginal KV bytes per token: that measurement includes fixed costs for S slots and MiB rounding. Audit native_measurement and p1_measurement integral-byte checks, calibration identity and manifest reports accordingly; measure the exact page geometry rather than extrapolate this ratio. Mixed-length GPU proof must demonstrate overlapping active/resident sessions, not successive requests with evictable retained prefixes. If 512K plus another 128K cannot fit physically, report the measured shortfall and test the largest feasible mixed workload without presenting it as the requested target passing.
