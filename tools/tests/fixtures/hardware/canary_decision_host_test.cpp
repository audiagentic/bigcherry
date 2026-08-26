// bigcherry: HI68 host-side unit test for the noise-canary transition logic.
//
// Compiled directly against src/ggml/src/ggml-cuda/hip-autotune-canary.h with
// any C++17 host compiler -- no HIP, no device, no ggml build required.
// Driven by tools/tests/test_hi68_canary_decision.py, which locates a
// compiler, builds this file, and runs it.

#include "hip-autotune-canary.h"

#include <cmath>
#include <cstdio>

static int failures = 0;

#define CHECK(cond) \
    do { \
        if (!(cond)) { \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures; \
        } \
    } while (0)

int main() {
    const double T = 2.0;  // threshold pct for this transition table

    // --- divergence_pct --------------------------------------------------
    CHECK(ggml_hip_canary_divergence_pct(100.0, 100.0) == 0.0);
    CHECK(ggml_hip_canary_divergence_pct(100.0, 110.0) > 9.999 &&
          ggml_hip_canary_divergence_pct(100.0, 110.0) < 10.001);
    CHECK(ggml_hip_canary_divergence_pct(100.0, 90.0) > 9.999 &&
          ggml_hip_canary_divergence_pct(100.0, 90.0) < 10.001);  // abs()
    CHECK(ggml_hip_canary_divergence_pct(0.0, 100.0) == -1.0);     // not evaluable
    CHECK(ggml_hip_canary_divergence_pct(-5.0, 100.0) == -1.0);
    CHECK(ggml_hip_canary_divergence_pct(100.0, -1.0) == -1.0);
    CHECK(ggml_hip_canary_divergence_pct(std::nan(""), 100.0) == -1.0);

    // --- initial stage ---------------------------------------------------
    {
        // Quiet pair: rank the original block, no probe.
        const auto v = ggml_hip_canary_transition(
            GGML_HIP_CANARY_STAGE_INITIAL, 100.0, 101.0, T, 1);
        CHECK(v.passed);
        CHECK(v.pct >= 0.999 && v.pct <= 1.001);
        CHECK(v.next == GGML_HIP_CANARY_RANK);
    }
    {
        // Boundary: exactly at the threshold passes (<= is the contract).
        const auto v = ggml_hip_canary_transition(
            GGML_HIP_CANARY_STAGE_INITIAL, 100.0, 102.0, T, 1);
        CHECK(v.passed);
        CHECK(v.next == GGML_HIP_CANARY_RANK);
    }
    {
        // Loud pair with probe allowed: request the stability probe.
        const auto v = ggml_hip_canary_transition(
            GGML_HIP_CANARY_STAGE_INITIAL, 100.0, 110.0, T, 1);
        CHECK(!v.passed);
        CHECK(v.next == GGML_HIP_CANARY_RUN_PROBE);
    }
    {
        // Loud pair, probe disabled by config: terminal unresolved at once.
        const auto v = ggml_hip_canary_transition(
            GGML_HIP_CANARY_STAGE_INITIAL, 100.0, 110.0, T, 0);
        CHECK(!v.passed);
        CHECK(v.next == GGML_HIP_CANARY_STOP_UNRESOLVED);
    }
    {
        // Unmeasurable pair: fail closed even with probe allowed.
        const auto v = ggml_hip_canary_transition(
            GGML_HIP_CANARY_STAGE_INITIAL, -1.0, 110.0, T, 1);
        CHECK(!v.passed);
        CHECK(v.pct == -1.0);
        CHECK(v.next == GGML_HIP_CANARY_STOP_UNRESOLVED);
    }

    // --- probe stage -----------------------------------------------------
    {
        const auto v = ggml_hip_canary_transition(
            GGML_HIP_CANARY_STAGE_PROBE, 100.0, 101.0, T, 0);
        CHECK(v.passed);
        CHECK(v.next == GGML_HIP_CANARY_RUN_FRESH);
    }
    {
        // Probe still loud: terminal unresolved. No path back to the pair,
        // whatever the retry budget says -- the budget is initial-stage only.
        const auto v = ggml_hip_canary_transition(
            GGML_HIP_CANARY_STAGE_PROBE, 100.0, 110.0, T, 5);
        CHECK(!v.passed);
        CHECK(v.next == GGML_HIP_CANARY_STOP_UNRESOLVED);
    }

    // --- fresh stage: THE F2 GUARANTEE ------------------------------------
    {
        const auto v = ggml_hip_canary_transition(
            GGML_HIP_CANARY_STAGE_FRESH, 100.0, 101.0, T, 9);
        CHECK(v.passed);
        CHECK(v.next == GGML_HIP_CANARY_RANK);
    }
    {
        // A loud fresh block is TERMINAL: the state machine has no "run
        // another probe / another fresh block" branch, whatever the retry
        // budget. Retry-until-quiet would re-introduce the self-selected
        // baseline this header exists to remove (RV49/F2).
        const auto v = ggml_hip_canary_transition(
            GGML_HIP_CANARY_STAGE_FRESH, 100.0, 110.0, T, 9);
        CHECK(!v.passed);
        CHECK(v.next == GGML_HIP_CANARY_STOP_UNRESOLVED);
    }

    if (failures == 0) {
        std::printf("CANARY_DECISION_HOST_TEST_OK\n");
        return 0;
    }
    std::fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
}
