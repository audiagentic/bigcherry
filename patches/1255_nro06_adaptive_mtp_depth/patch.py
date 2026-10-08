"""NRO06 draft: pure adaptive MTP depth controller, intentionally unwired."""

from bigcherry.patcher import Edit, FilePatch

DRAFT_SOURCE_CONTEXT = {
    "repo": "https://github.com/nasone32/llama.cpp-RDNA3-7900xtx-opt",
    "commit": "10579a7365a3bc86c4f8e41aaab20e73e1571e5e",
    "title": "rdna-boosts: block 01: adaptive MTP draft depth",
    "pin": "b10705",
}

_CONTROLLER = r'''

// BIGCHERRY_NRO06_ADAPTIVE_MTP_CONTROLLER_BEGIN
// Pure deterministic controller. Runtime wiring is supplied by 1268.
// Policy input is acceptance counts only: no clocks, allocation addresses, or
// process-global state can influence the chosen depth.
struct bigcherry_nro06_adaptive_mtp {
    int n_cur = 0;
    int n_window_draft = 0;
    int n_window_accept = 0;

    static constexpr int window_tokens = 32;
    static constexpr int climb_pct = 72;
    static constexpr int drop_pct = 60;

    void reset(int n_max, int n_min_adaptive) {
        const int cap = std::max(1, n_max);
        const int floor = std::min(std::max(1, n_min_adaptive), cap);
        // Depth 1/2 is a bad cold start on Flash-Next. Start from the
        // hardware-neutral depth 3, clamped by the caller's floor/cap.
        n_cur = std::min(cap, std::max(floor, 3));
        n_window_draft = 0;
        n_window_accept = 0;
    }

    void update(int n_draft, int n_accepted, int n_max, int n_min_adaptive) {
        if (n_draft <= 0) return;

        const int cap = std::max(1, n_max);
        const int floor = std::min(std::max(1, n_min_adaptive), cap);
        n_window_draft += n_draft;
        n_window_accept += std::min(std::max(0, n_accepted), n_draft);

        if (n_window_draft < window_tokens) {
            return;
        }

        // Integer comparisons keep the policy bit-for-bit deterministic.
        // A dead band prevents depth oscillation on marginal workloads.
        const int64_t accept100 = (int64_t) n_window_accept * 100;
        const int64_t draft100  = (int64_t) n_window_draft;
        if (accept100 <= (int64_t) drop_pct * draft100 && n_cur > floor) {
            --n_cur;
        } else if (accept100 >= (int64_t) climb_pct * draft100 && n_cur < cap) {
            ++n_cur;
        }

        n_window_draft = 0;
        n_window_accept = 0;
    }
};
// BIGCHERRY_NRO06_ADAPTIVE_MTP_CONTROLLER_END
'''

PATCHES = [FilePatch(
    path="common/speculative.cpp",
    description="add pure adaptive MTP controller without changing fixed-MTP runtime behavior",
    edits=(Edit(
        id="adaptive-controller",
        anchor=r"^struct common_speculative_impl_draft_mtp : public common_speculative_impl \{$",
        mode="insert_before",
        text=_CONTROLLER + "\n\n",
        guard=r"BIGCHERRY_NRO06_ADAPTIVE_MTP_CONTROLLER_BEGIN",
        rationale="keep the controller adjacent to its future MTP consumer while initially unwired",
    ),),
)]
