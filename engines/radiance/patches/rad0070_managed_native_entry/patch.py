"""RSC02: the managed native entry, beside the manifest entry for one change.

The owner starts the stock `--config FILE` entry and adds one sealed document (`--bc-admission-fd`, see
components/serving/v2/engine_admission.h). Everything with a policy lives in BigCherry's own component sources
(bc_serving_managed.cpp); this package only attaches it: one call after the YAML is parsed (bind), one check where Fleet
has chosen its cards and no Engine exists yet, and the guards that keep a managed process from reloading or restarting
itself: SIGHUP and the HTTP reload and model-admin routes refuse, and an engine that stops ends the process.

Without the flag every hook returns at once and the stock entry is unchanged. rad0009 (the manifest entry) is retired
in the change that moves the owner onto this entry; until then this package requires it for its include and CMake lines.
"""
from __future__ import annotations

from bigcherry.patcher import Edit, FilePatch

_CORE_CMAKE = r"""
# BigCherry rad0070 (RSC02): the managed native entry's bind reads the engine's Config, so it is built into rad_core here; the
# rest of the entry (bc_serving_managed.cpp) is in the shared serving library every engine build already links.
target_sources(rad_core PRIVATE "${_bc_serving}/bc_serving_managed_bind.cpp")
if(RAD_BUILD_TESTS AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fleet/config.h")
  add_executable(bc_native_managed_test "${_bc_serving}/native_managed_test.cpp")
  target_link_libraries(bc_native_managed_test PRIVATE rad_core bc_jcs)
endif()"""

_TEST_REGISTRATION = r"""
if(TARGET bc_native_managed_test)
  add_test(NAME bc_native_managed_test COMMAND bc_native_managed_test)
endif()
"""

PATCHES = (
    FilePatch(
        path="core/CMakeLists.txt",
        description="Build the managed native entry's component source into rad_core and its card-free test.",
        edits=(Edit(
            id="build-bc-serving-managed",
            anchor=r'^target_include_directories\(rad_core PUBLIC "\$\{_bc_serving\}"\)$',
            text=_CORE_CMAKE,
            guard=r"bc_serving_managed_bind\.cpp",
            max_span_lines=1,
            expect_matches=1,
            rationale="rad0009 defines _bc_serving and the include directory on exactly this line; the managed entry is its sibling source.",
        ),),
    ),
    FilePatch(
        path="tests/CMakeLists.txt",
        description="Register the managed native entry test before the upstream label pass.",
        edits=(Edit(
            id="register-bc-native-managed-test",
            anchor=r"^get_property\(RAD_ALL_TESTS DIRECTORY PROPERTY TESTS\)$",
            text=_TEST_REGISTRATION,
            mode="insert_before",
            guard=r"^if\(TARGET bc_native_managed_test\)$",
            expect_matches=1, max_span_lines=2,
                rationale="Same unique pre-label-pass anchor rad0009 uses for its own reader test.",
        ),),
    ),
    FilePatch(
        path="core/main.cpp",
        description="Bind the owner's admission to the parsed YAML before Fleet::run, once.",
        edits=(
            Edit(
                id="include-bc-serving-managed",
                anchor=r'^#include "bc_serving_manifest\.h"$',
                text='\n#include "bc_serving_managed.h"',
                guard=r'^#include "bc_serving_managed\.h"$',
                expect_matches=1, max_span_lines=2,
                rationale="rad0009's include is the stable neighbour; this package requires it.",
            ),
            Edit(
                id="bind-bc-serving-managed",
                anchor=r"^    if \(check\) \{$",
                text=("    /* BigCherry rad0070: a managed start binds its admission here; the stock entry returns 0. */\n"
                      "    if (const int managed = bc_serving_managed_bind(&fleet)) return managed;\n"),
                mode="insert_before",
                guard=r"bc_serving_managed_bind\(&fleet\)",
                expect_matches=1, max_span_lines=2,
                rationale="After fleet::load and the unsupported-key refusal, before --check and before Fleet::run or the offline Engine, so every later reader sees the bound values.",
            ),
        ),
    ),
    FilePatch(
        path="core/fleet/fleet.cpp",
        description="Managed guards: chosen cards are the admitted ranks, no reload, no model admin, an engine exit ends the process.",
        edits=(
            Edit(
                id="include-bc-serving-managed",
                anchor=r'^#include "util/stallwatch\.h"$',
                text='\n#include "bc_serving_managed.h"',
                guard=r'^#include "bc_serving_managed\.h"$',
                expect_matches=1, max_span_lines=2,
                rationale="Last project include of fleet.cpp, a stable extension point.",
            ),
            Edit(
                id="managed-chosen-cards",
                anchor=r"^    cfg\.devices = chosen;$",
                text=('    if (const std::string w = bc_serving_managed_chosen(chosen); !w.empty()) {   /* BigCherry rad0070 */\n'
                      '        fail(m, w);\n'
                      '        return;\n'
                      '    }\n'),
                mode="insert_before",
                guard=r"bc_serving_managed_chosen\(chosen\)",
                expect_matches=1, max_span_lines=2,
                rationale="The single place Fleet fixes the devices of a load, before the Engine is constructed; the room lock is out of scope here and fail() is the load failure path.",
            ),
            Edit(
                id="managed-no-reload",
                anchor=r"^std::string Fleet::reload\(\) \{$",
                text='\n    if (bc_serving_managed()) return "a managed start is not reloaded; restart it";   /* BigCherry rad0070: SIGHUP and /admin/reload */',
                guard=r"a managed start is not reloaded",
                expect_matches=1, max_span_lines=2,
                rationale="Both the SIGHUP path and the HTTP route reach reload(); one refusal covers the signal.",
            ),
            Edit(
                id="managed-no-model-admin",
                anchor=r"^        bool known = false;$",
                text=('        if (bc_serving_managed()) return bad(q, 403, "model admin is disabled in a managed start");   /* BigCherry rad0070 */\n'),
                mode="insert_before",
                guard=r"if \(bc_serving_managed\(\)\) return bad\(q, 403, [^\n]*managed start",
                expect_matches=1, max_span_lines=2,
                rationale="Inside POST /admin/models/:name/:action (load, park, unload, pin), the only model-mutating route, after its guard and before any action; the route strings are noise to the anchor matcher, so the first plain statement is the anchor.",
            ),
            Edit(
                id="managed-no-reload-route",
                anchor=r"^        const auto r = run_cmd\(\[this\] \{ return reload\(\); \}, 3600000\);$",
                text=('        if (bc_serving_managed()) return bad(q, 403, "reload is disabled in a managed start");   /* BigCherry rad0070 */\n'),
                mode="insert_before",
                guard=r"reload is disabled in a managed start",
                expect_matches=1, max_span_lines=2,
                rationale="POST /admin/reload answers 403 before reload() would refuse with 400; the signal path is covered in reload().",
            ),
            Edit(
                id="managed-engine-exit-ends-process",
                anchor=r"^        front_\.detach\(m\.entry\.name, 0\);$",
                text=('        if (bc_serving_managed()) {   /* BigCherry rad0070: no demand-driven reload of a managed engine */\n'
                      '            bc_serving_managed_note_engine_exit();\n'
                      '            g_stop.store(true);\n'
                      '        }\n'),
                mode="insert_before",
                guard=r"bc_serving_managed_note_engine_exit\(\);",
                expect_matches=1, max_span_lines=2,
                rationale="Fleet::reap is where Fleet learns an engine ended (the other front_.detach, in unload, has different indentation); a failed model would be loaded again by the next demand, which a managed owner never allows.",
            ),
            Edit(
                id="managed-exit-status",
                anchor=r"^    return listen_rc\.load\(\) < 0 \? 1 : 0;$",
                text="    return listen_rc.load() < 0 || bc_serving_managed_engine_exited() ? 1 : 0;   /* BigCherry rad0070 */",
                mode="replace",
                guard=r"bc_serving_managed_engine_exited\(\)",
                expect_matches=1, max_span_lines=2,
                rationale="An engine that stopped while serving is a failed run, not a clean stop.",
            ),
        ),
    ),
)
