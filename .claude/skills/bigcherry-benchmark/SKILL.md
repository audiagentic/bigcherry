---
name: bigcherry-benchmark
description: Construct and run comparative end-to-end GPU benchmarks whose results can actually be attributed -- arm isolation, build-composition proof, graceful teardown, activation evidence, and the statistics that apply.
---

# BigCherry Benchmark

Purpose

Own comparative measurement on real hardware:

construct arms that differ in exactly one thing;

prove what is actually in each binary;

run without confounding arm with position;

prove the mechanism under test executed;

interpret the result with statistics that fit the data.

This skill owns HOW a number is produced and whether two numbers may be
compared. It does NOT own contract obligations, evidence records, or
lifecycle state -- those belong to bigcherry-patch-qualification and
bigcherry-patch-lifecycle.

Triggers

Use when asked to:

compare two or more builds, flags, caches, or configurations;

measure the cost or benefit of a patch, build flag, or tuning cache;

check for a regression;

reproduce or challenge an existing performance claim;

decide whether a measured difference is real.

Non-triggers

Do not use for:

single-build liveness or smoke checks with no comparison;

contract evidence interpretation or promotion decisions;

microbenchmark-only kernel search (that is the tuner's own domain);

any run where hardware execution has not been explicitly authorized.

Source of truth

docs/reference/testing/TEST.md -- "Comparative A/B benchmarking" is
authoritative for this skill; read it before constructing any run.

docs/reference/testing/PATCH_VALIDATION.md

docs/reference/build/BUILD.md

tools/bigcherry/tuning/server_runner.py

tools/lab/gp11-replay-bench/ab-balanced.sh (reference implementation)

tools/lab/gp11-replay-bench/analyse.py

If the doc and the implementation disagree, the implementation wins and the
drift must be reported.

Inputs

Require before constructing a run:

the exact question, stated as what single variable differs between arms;

candidate build identities;

model, device set, workload/bench config;

whether the mechanism under test has an observable activation signal;

explicit authorization for hardware execution.

Outputs

per-arm means, the delta against a named baseline, and whether sample
ranges overlap;

per-position means, so drift can be tested;

activation evidence per arm;

work-equivalence evidence (e.g. MTP draft acceptance);

an explicit statement of what the result does NOT attribute;

outliers named and excluded with a reason.

## The five failure modes

Every one of these produced a reported-then-retracted result. Check each
before running, and again before reporting.

### 1. Arm confounded with position

Running A, B, C once each in fixed order makes "C is slowest" and "the third
arm is slowest" the same measurement. Thermal drift over ~10 minutes produces
a clean monotone pattern that looks like a finding.

Required: rotate arm order every round; rounds a multiple of arm count;
record position on every row; report per-position means.

### 2. Arms trusted by label

A digest noted earlier, or a name like "control", is not evidence of
composition. Two arms once differed by a patch, a build flag, AND a cache
while being described as differing by one.

Required, per arm, read from the build itself:

    python3 -c "
    import json; cc=json.load(open('<build>/compile_commands.json'))
    print([e['file'] for e in cc if 'ggml-cuda' in e['file']][0].split('/ggml/')[0])"

    grep -rlq '<patch marker>' <src>/ggml/src/
    grep -a 'GGML_HIP_DISPATCH_REPLAY:BOOL=' <build>/CMakeCache.txt

For a build-flag comparison the two source trees must be byte-identical:

    find <src>/ggml/src -name '*.cu' -o -name '*.cuh' -o -name '*.cpp' \
      | sort | xargs cat | sha256sum

### 3. Teardown that destroys the evidence

`kill -9` skips backend teardown: it discards buffered tune measurements AND
the replay hit/miss and coverage reports, which are emitted at shutdown.

The `/shutdown` route (patch 0800_server_shutdown_endpoint) is registered
ONLY when `LLAMA_SERVER_ENABLE_SHUTDOWN` is set in the server's environment.
Without it the POST 404s and silently falls back to `kill -9`.

Prefer `ServerRunner` (tools/bigcherry/tuning/server_runner.py) over bash: it
already handles launch, health-check, free-port selection, env hygiene, and
graceful teardown. Every ad-hoc reimplementation so far reintroduced a bug it
had already fixed (port 8080 collision; kill -9).

### 4. No proof the mechanism was active

Setting an environment variable is not evidence the feature ran. Without
activation evidence, "the feature made no difference" and "the feature never
engaged" are the same data.

For a replay/cache arm capture BOTH:

  startup:  `replay cache '<path>' loaded, N winner(s)`
  shutdown: `replay v2 N winner(s); exact=.. miss=..`

Coverage `dispatched == executed` does NOT prove tuning applied -- a miss is
still a dispatch, to native, so a fully-covered run can be entirely untuned.

Generalise: for any patch under test, record evidence its path executed, in
the same run that produced the numbers.

### 5. Arms that did different amounts of work

For MTP speculative decode, record draft acceptance per cell. Differing
acceptance means the arms generated different work and the throughput
comparison is void however tight the intervals are.

## Statistics

Complete separation (every sample of one arm beyond every sample of the
other) at n=6/arm is Mann-Whitney p ~= 0.0022. Prefer it to a t-test: samples
are small and drift is not obviously normal.

Do NOT treat "all N metrics moved the same way" as significance. The metrics
are strongly correlated; 6/6 is not p = 1/64.

Between-session drift on Brutus is sd 0.5-0.6%, larger than any single run's
standard error. Cross-run comparisons need that margin; within-run balanced
comparisons do not.

pp256 is startup-sensitive and throws ~10%-low outliers. Exclude it from
diagnosis; tg512/tg2048 are the stable signals.

Small wins count. Owner policy is explicit that a sub-1% improvement is worth
taking if nothing regresses -- do not impose a materiality bar, and do not
invent a promotion "gate". The standard is improvement assessed against
regression.

## Workflow

1. State the single variable under test. If you cannot name it, stop.
2. Select arms; prove composition (failure mode 2) BEFORE running.
3. Confirm hardware authorization and exclusive device access.
4. Use ab-balanced.sh, or ServerRunner; do not write new bash.
5. Run balanced; rounds a multiple of arm count.
6. Check activation and work-equivalence evidence per cell (4, 5).
7. Analyse with analyse.py: means, deltas, separation, position, outliers.
8. Report what is attributed AND what is not.

## Verified commands

    ROUNDS=8 OUT=<log> ARMS="name:digest:cache ..." tools/lab/gp11-replay-bench/ab-balanced.sh
    python3 tools/lab/gp11-replay-bench/analyse.py <log>

Bench harness (endpoint mode only; never llama-bench, which cannot see MTP):

    python3 bench/run_bench.py --bench-type server-bench \
      --server-url http://127.0.0.1:<port> --model <label> --bench-configs <cfg>

## Stop conditions

Stop, and do not report a comparative result, when:

arms differ in more than the single variable under test;

build composition was not read out of the builds;

any arm was torn down with kill -9;

activation evidence is missing for the mechanism under test;

work-equivalence evidence differs between arms;

the run was not order-balanced;

a cross-run comparison is inside the drift margin;

hardware authorization was not explicit.

## Safety rules

Never report an attribution the design cannot support; state the confound
instead.

Never cite a confounded or withdrawn run as evidence; mark it withdrawn and
say why, but keep it recorded.

Retract promptly and completely when a defect is found -- a measured
difference can be real while its explanation is wrong; these are separate
claims and must be retracted separately.

Never perturb a running measurement (no builds, no compiles, no parallel
jobs on the same host).

Never write harness scripts to /tmp; they belong in tools/lab/<topic>/ with
the README the lab template requires.

Do not edit a script on the bench host while it is executing -- bash reads
scripts incrementally and will run corrupted input.

## Handoff rules

To bigcherry-patch-qualification: a measured result intended as contract
evidence, with its activation and work-equivalence evidence attached.

To bigcherry-patch-lifecycle: never directly. A benchmark never mutates
lifecycle state.

From bigcherry-patch-qualification: the specific comparison a contract
requires, including required architectures.

## Self-validation

Before reporting:

Can I name the single variable that differs between arms?

Did I read composition out of every build, rather than trusting a label?

Was the run order-balanced, and did I report per-position means?

Did every arm shut down gracefully?

Do I have activation evidence for the mechanism under test?

Did the arms do the same amount of work?

Am I using a separation/rank test rather than correlated-metric counting?

Have I stated explicitly what this result does NOT attribute?

Any "no" blocks the report.
