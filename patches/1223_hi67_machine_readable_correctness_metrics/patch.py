"""HI67 slice 2b: machine-readable per-tensor correctness metrics from test-backend-ops.

test-backend-ops' own `test` mode already computes exactly what HI67's
contract needs: for each compared tensor, `err = nmse(backend1_output,
backend2_output)`, where `eval()`'s only real caller passes `backend1` = the
backend under test (HIP, forced to a specific candidate via
GGML_HIP_FORCE_CANDIDATE) and `backend2` = the CPU reference backend
(`backend_cpu`, see run_test()'s `test->eval(b, b_cpu, ...)` call). So this
`err` value IS E_N when backend1 is forced to native, and E_C when backend1
is forced to a candidate -- test-backend-ops was already computing the right
number, just never printing it in a form anything could parse; only a
pass/fail line and, on failure, a truncated %.9f appear today.

This patch adds a machine-readable BIGCHERRY_CORRECTNESS_METRIC line,
unconditionally (not just on failure), gated behind the same
BIGCHERRY_TEST_DETERMINISTIC_SEED opt-in as patches/1222_hi67_deterministic_
test_backend_ops_seed.py (this patch REQUIRES it -- it reuses that patch's
bigcherry_deterministic_mode() helper) -- so a normal upstream test-backend-
ops run is unaffected, and BigCherry's own evidence generator (HI67 slice
2c, not yet written) can key each line to the exact test case's op/tensor
name and treat multiple compared tensors per graph (e.g. sentinels)
correctly by filtering to the op it actually asked about.

Also computes max_abs(backend1, backend2) in the same loop already iterating
every element for the NaN/inf checks -- test-backend-ops does not track this
today (only NMSE), and RV49's contract needs max_abs(C,R) <= max_abs(N,R) as
a second, independent bound alongside the NMSE headroom rule.
"""

GROUP = "core"
# See patches/1222_hi67_deterministic_test_backend_ops_seed/patch.py's STATE note:
# offline-verified (apply + idempotence against the real vendored checkout)
# but not yet compiled or run on real hardware.
STATE = "untested"

from bigcherry.patcher import Edit, FilePatch

REQUIRES = ("1222_hi67_deterministic_test_backend_ops_seed",)

_METRIC_LINE = """
            double bigcherry_max_abs = 0.0;
            for (size_t bi = 0; bi < f1.size(); bi++) {
                bigcherry_max_abs = std::max(bigcherry_max_abs, (double) std::fabs(f1[bi] - f2[bi]));
            }
            if (bigcherry_deterministic_mode()) {
                // bigcherry (HI67 slice 2b): err here is NMSE(backend1_output,
                // backend2_output) -- E_N when backend1 was forced to native,
                // E_C when forced to a candidate (see patches/1223_hi67_machine_
                // readable_correctness_metrics.py for the full rationale).
                //
                // bigcherry (HI83): backend1_digest/backend2_digest additionally
                // let a patch-specific producer (e.g. rd08_correctness_evidence.py)
                // compare exact backend output bytes across two SEPARATE
                // deterministic process invocations, not just their NMSE against
                // a shared CPU reference -- two implementations can have equal
                // NMSE without being bit-identical. Reuses bigcherry_fnv1a() from
                // patches/1222_hi67_deterministic_test_backend_ops_seed/patch.py, which
                // this patch already REQUIRES.
                const uint64_t bigcherry_backend1_digest =
                    bigcherry_fnv1a(f1.data(), f1.size() * sizeof(float));
                const uint64_t bigcherry_backend2_digest =
                    bigcherry_fnv1a(f2.data(), f2.size() * sizeof(float));
                fprintf(stderr,
                    "BIGCHERRY_CORRECTNESS_METRIC op=%s tensor=%s backend1=%s backend2=%s "
                    "err=%.17g max_abs=%.17g threshold=%.17g n=%zu "
                    "backend1_digest=%016llx backend2_digest=%016llx\\n",
                    ggml_op_desc(t1), t1->name, bn1, bn2, err,
                    bigcherry_max_abs, ud->tc->max_err(ud->backend1), f1.size(),
                    (unsigned long long) bigcherry_backend1_digest,
                    (unsigned long long) bigcherry_backend2_digest);
            }
"""

PATCH = FilePatch(
    path="tests/test-backend-ops.cpp",
    description="print machine-readable BIGCHERRY_CORRECTNESS_METRIC lines (NMSE + max_abs + "
    "threshold, per compared tensor) so HI67's evidence generator can consume the "
    "same comparison test-backend-ops already performs (HI67 slice 2b)",
    edits=(
        Edit(
            id="hi67-correctness-metric-line",
            anchor=r"double err = ud->tc->err\(f1\.data\(\), f2\.data\(\), f1\.size\(\)\);",
            mode="insert_after",
            rationale="right after err (NMSE) is computed in the graph-compare callback, "
            "before the pass/fail check that follows it",
            text=_METRIC_LINE,
            guard=r"bigcherry \(HI67 slice 2b\): err here is NMSE",
            max_span_lines=5,
        ),
    ),
)

PATCHES = [PATCH]
