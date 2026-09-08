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
// Pure/testable controller only.  Runtime MTP remains fixed-depth until the
// controller's exhaustive state-machine tests and request-reset plumbing land.
struct bigcherry_nro06_adaptive_mtp {
    int n_cur = 0;
    int n_climb = 0;
    int n_drop = 0;

    static int climb_threshold(int depth) {
        switch (depth) {
            case 1: return 2;
            case 2: return 4;
            case 3: return 10;
            case 4: return 6;
            case 5: return 3;
            case 6: return 2;
            default: return 2;
        }
    }

    static int drop_pressure(int depth) {
        return std::max(depth * 5, 20);
    }

    void reset(int n_max, int n_min_adaptive) {
        const int cap = std::max(1, n_max);
        const int floor = std::max(1, n_min_adaptive);
        n_cur = std::min(floor, cap);
        n_climb = 0;
        n_drop = 0;
    }

    void update(int n_draft, int n_accepted, int n_max, int n_min_adaptive) {
        if (n_draft <= 0) return;
        const int cap = std::max(1, n_max);
        const int floor = std::min(std::max(1, n_min_adaptive), cap);
        if (n_accepted == n_draft) {
            n_drop = 0;
            if (n_cur < cap && ++n_climb >= climb_threshold(n_cur)) {
                ++n_cur;
                n_climb = 0;
            }
            return;
        }
        n_climb = 0;
        if (n_cur > floor) {
            n_drop += std::max(0, n_draft - n_accepted);
            if (n_drop >= drop_pressure(n_cur)) {
                --n_cur;
                n_drop = 0;
            }
        }
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
