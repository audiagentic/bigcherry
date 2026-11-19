"""Radiance 1.8.0: freeze the heat engine's promotions after N steps, for the MTP-collapse bisection (RAD14).

RAD_DIAG_FREEZE_PROMOTION_AFTER=N makes HeatEngine::step() propose no swaps from its (N+1)th call on, so the resident
set stops changing while the engine keeps serving (unlike --deterministic, which pins the loader's plan and does not
fit this card). Unset or 0: no effect. Diagnostic builds only (BC_MOE_REMOTE_DIAG), compiled out otherwise.
"""

from bigcherry.patcher import Edit, FilePatch

PATCH = FilePatch(
    path="core/place/heat.cpp",
    description="Stop proposing swaps after N heat steps (diagnostic freeze)",
    edits=(
        Edit(
            id="heat-freeze-promotion",
            anchor=r"^    const int k = params_\.moves_per_step;$",
            mode="replace",
            text="""#ifdef BC_MOE_REMOTE_DIAG
    {
        static const long freeze_after = [] {
            const char* e = std::getenv("RAD_DIAG_FREEZE_PROMOTION_AFTER");
            return e ? std::atol(e) : 0L;
        }();
        static long calls = 0;
        if (freeze_after > 0 && ++calls > freeze_after) {
            if (calls == freeze_after + 1) RAD_INFO("heat engine: promotions frozen after %ld steps (RAD_DIAG_FREEZE_PROMOTION_AFTER)", freeze_after);
            return RAD_OK;
        }
    }
#endif
    const int k = params_.moves_per_step;""",
            guard=r"RAD_DIAG_FREEZE_PROMOTION_AFTER",
            expect_matches=1,
            rationale="Unique read of the per-step move budget at the top of HeatEngine::step",
        ),
        Edit(
            id="heat-freeze-include",
            anchor=r"^#include <cmath>$",
            mode="replace",
            text="#include <cmath>\n#include <cstdlib>",
            guard=r"#include <cstdlib>",
            expect_matches=1,
            rationale="The freeze reads its limit with getenv/atol",
        ),
    ),
)
