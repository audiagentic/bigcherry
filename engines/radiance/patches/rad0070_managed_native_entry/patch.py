"""RSC02: the managed native entry: the engine's only owner-facing entry (the manifest entry rad0009 is archived).

The owner starts the stock `--config FILE` entry and adds one sealed document (`--bc-admission-fd`, see
components/serving/v2/engine_admission.h). Everything with a policy lives in BigCherry's own component sources
(bc_serving_managed.cpp); this package only attaches it: one call after the YAML is parsed (bind), one check where Fleet
has chosen its cards and no Engine exists yet, and the guards that keep a managed process from reloading or restarting
itself: SIGHUP and the HTTP reload and model-admin routes refuse, and an engine that stops ends the process.

Without the flag every hook returns at once and the stock entry is unchanged.
"""
from __future__ import annotations

from bigcherry.patcher import Edit, FilePatch

_CORE_LINK = r"""
# BigCherry rad0070 (RSC02): the managed native entry is BigCherry's own component source, linked into rad_core. The shared
# serving library (BigCherry's components/serving/bc_serving_v2.cmake, one source list for the owner and the engine) holds
# the strict readers, the engine admission document and the flag step; the engine-side files below read the engine's Config.
if(NOT DEFINED BC_SERVING_V2_CMAKE)
  message(FATAL_ERROR "rad0070: -DBC_SERVING_V2_CMAKE=<bigcherry>/engines/radiance/components/serving/bc_serving_v2.cmake "
                      "is required (the shared serving library); the BigCherry lane passes it")
endif()
include("${BC_SERVING_V2_CMAKE}")
target_link_libraries(rad_core PUBLIC bc_serving_v2)
get_filename_component(_bc_serving "${BC_SERVING_V2_CMAKE}" DIRECTORY)
target_sources(rad_core PRIVATE "${_bc_serving}/bc_serving_engine.cpp" "${_bc_serving}/bc_serving_managed_bind.cpp")
target_include_directories(rad_core PUBLIC "${_bc_serving}")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fleet/config.h")
  target_compile_definitions(rad_core PRIVATE BC_RAD_NATIVE_FLEET_CONFIG=1)
  target_sources(rad_core PRIVATE "${_bc_serving}/native_fleet_config.cpp")
  if(RAD_BUILD_TESTS)
    add_executable(bc_native_fleet_config_test "${_bc_serving}/native_fleet_config_test.cpp")
    target_link_libraries(bc_native_fleet_config_test PRIVATE rad_core bc_jcs)
    target_compile_definitions(bc_native_fleet_config_test PRIVATE
        BC_CONTRACT_DIR="${_bc_serving}/../../../../docs/design/serving-manifest-v2")
    add_executable(bc_native_managed_test "${_bc_serving}/native_managed_test.cpp")
    target_link_libraries(bc_native_managed_test PRIVATE rad_core bc_jcs)
  endif()
endif()
"""

_TEST_REGISTRATION = r"""
if(TARGET bc_native_fleet_config_test)
  add_test(NAME bc_native_fleet_config_test COMMAND bc_native_fleet_config_test)
endif()
if(TARGET bc_native_managed_test)
  add_test(NAME bc_native_managed_test COMMAND bc_native_managed_test)
endif()
"""

PATCHES = (
    FilePatch(
        path="core/CMakeLists.txt",
        description="Link the shared serving library and build the managed native entry's engine-side sources and tests into rad_core.",
        edits=(Edit(
            id="link-bc-serving",
            anchor=r"^target_link_libraries\(rad_core PUBLIC rad_abi \$\{CMAKE_DL_LIBS\} Threads::Threads\)$",
            text=_CORE_LINK,
            guard=r"include\(\"\$\{BC_SERVING_V2_CMAKE\}\"\)",
            max_span_lines=1,
            expect_matches=1,
            rationale="rad_core's unique link line; the shared library is included and linked right after it, failing closed when the lane did not pass its .cmake.",
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
            guard=r"^if\(TARGET bc_native_fleet_config_test\)$",
            expect_matches=1, max_span_lines=2,
                rationale="The unique pre-label-pass anchor: the tests are registered after the upstream harness enables CTest.",
        ),),
    ),
    FilePatch(
        path="core/main.cpp",
        description="Bind the owner's admission to the parsed YAML before Fleet::run, once.",
        edits=(
            Edit(
                id="include-bc-serving-engine",
                anchor=r'^#include "util/stallwatch\.h"$',
                text='\n#include "bc_serving_engine.h"\n#include "bc_serving_managed.h"',
                guard=r'^#include "bc_serving_engine\.h"$',
                max_span_lines=2,
                expect_matches=1,
                rationale="The upstream last include is a stable typed extension point.",
            ),
            Edit(
                id="bc-serving-owned-entry",
                anchor=r'^int main\(int argc, char\*\* argv\) \{$',
                text=('\n    return bc_serving_engine_entry(argc, argv, '
                      '[](int argc, char** argv) -> int {'),
                guard=r'bc_serving_engine_entry\(argc, argv,',
                max_span_lines=2,
                expect_matches=1,
                rationale="The owner's flags are taken before the stock entry; the stock YAML startup is the callable, unchanged.",
            ),
            Edit(
                id="bc-serving-stock-entry-end",
                anchor=r'    return 0;\n\}\n?\Z',
                mode="replace",
                text='    return 0;\n    });\n}\n',
                guard=r'(?m)^    \}\);$',
                expect_matches=1, max_span_lines=3,
                rationale="Close the stock YAML callable and the unique main entry; no copy of upstream startup body",
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
                id="managed-usable-cards",
                anchor=r"^    cards_ = usable;$",
                text="    usable = bc_serving_managed_usable(usable);   /* BigCherry rad0070: the admitted ranks, not one architecture */\n",
                mode="insert_before",
                guard=r"bc_serving_managed_usable\(usable\)",
                expect_matches=1, max_span_lines=2,
                rationale="Fleet::run's usable list comes from Engine::usable_cards, which keeps one architecture; a managed start's cards are the admitted ranks (mixed-architecture TP). Engine::start admits rank-devices itself.",
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
