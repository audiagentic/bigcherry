"""Radiance 2.2: native sealed-v3 serving admission via one typed main call.

The manifest parser, replay and managed lifecycle are ordinary sources in
components/serving. The pinned patch attaches the existing CMake owner and
invokes its typed entry with the *unchanged* upstream YAML startup callable.
No vendor-owned implementation body is duplicated or generated here.
"""

import hashlib

# rad0009 ships no generated vendor sources: core builds the ordinary
# BigCherry-owned serving component inputs via BC_SERVING_V2_CMAKE.
# patch.py is the registry implementation identity, so pin the external
# source bytes here and fail closed if either changes without review.
from pathlib import Path

from bigcherry.patcher import Edit, FilePatch

_COMPONENT_SOURCE_SHA256 = {'bc_serving_manifest.h': '3304880bbc401dbbb2549478016d822effabd40079410e05ec462d4072f6f73c', 'bc_serving_manifest.cpp': '4ee5a5222b25ccbed6ef8a8fb28f7847e4bb20f142bde2dfb5ff409ece4b31a6'}


def _verify_owned_component_sources() -> None:
    root = Path(__file__).resolve().parents[2] / "components" / "serving"
    for name, expected in _COMPONENT_SOURCE_SHA256.items():
        got = hashlib.sha256((root / name).read_bytes()).hexdigest()
        if got != expected:
            raise ValueError(f"rad0009 source {name}: {got} != pinned {expected}")


_verify_owned_component_sources()


# The shared library comes from the BigCherry tree, never a copy: the lane passes its .cmake (components/serving).
_CORE_LINK = r"""
# BigCherry rad0009 (RSC08 S2b): the engine reads its manifest and (rad0011) checks its plan through the shared serving manifest v1 library, built here from
# its one source list (BigCherry's components/serving/bc_serving_v2.cmake) against this tree's ABI.
if(NOT DEFINED BC_SERVING_V2_CMAKE)
  message(FATAL_ERROR "rad0009: -DBC_SERVING_V2_CMAKE=<bigcherry>/engines/radiance/components/serving/bc_serving_v2.cmake "
                      "is required (the shared serving manifest v1 library); the BigCherry lane passes it")
endif()
if(BC_SERVING_V3_NATIVE_TEST)
  if(NOT RAD_BUILD_TESTS OR NOT DEFINED BC_SERVING_GENERATED_DIR OR
     NOT DEFINED BC_V3_WRITER_TEST_EXECUTABLE OR NOT EXISTS "${BC_V3_WRITER_TEST_EXECUTABLE}" OR
     NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fleet/config.h")
    message(FATAL_ERROR "rad0009: v3 native consumer test requires RAD_BUILD_TESTS, the v3 codec, writer, config and fleet parser")
  endif()
endif()
include("${BC_SERVING_V2_CMAKE}")
target_link_libraries(rad_core PUBLIC bc_serving_v2)
get_filename_component(_bc_serving "${BC_SERVING_V2_CMAKE}" DIRECTORY)
# rad0009-owned source is part of the serving component, compiled in the
# engine's native rad_core target (one source, no copied vendor payload).
target_sources(rad_core PRIVATE "${_bc_serving}/bc_serving_manifest.cpp")
target_include_directories(rad_core PUBLIC "${_bc_serving}")

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/fleet/config.h")
  target_compile_definitions(rad_core PRIVATE BC_RAD_NATIVE_FLEET_CONFIG=1)
  target_sources(rad_core PRIVATE "${_bc_serving}/native_fleet_config.cpp")
  if(RAD_BUILD_TESTS AND BC_SERVING_V3_NATIVE_TEST)
    add_executable(bc_native_v3_engine_test "${_bc_serving}/native_fleet_v3_engine_test.cpp")
    target_link_libraries(bc_native_v3_engine_test PRIVATE rad_core)
  elseif(RAD_BUILD_TESTS)
    add_executable(bc_native_fleet_config_test
        "${_bc_serving}/native_fleet_config_test.cpp")
    target_link_libraries(bc_native_fleet_config_test PRIVATE rad_core bc_jcs)
    target_compile_definitions(bc_native_fleet_config_test PRIVATE
        BC_CONTRACT_DIR="${_bc_serving}/../../../../docs/design/serving-manifest-v2")
  endif()
endif()
"""

_CORE_TEST_REGISTRATION = r"""
if(TARGET bc_native_fleet_config_test)
  add_test(NAME bc_native_fleet_config_test COMMAND bc_native_fleet_config_test)
endif()
if(TARGET bc_native_v3_engine_test AND DEFINED BC_V3_WRITER_TEST_EXECUTABLE)
  add_test(NAME bc_native_v3_engine_test COMMAND "${BC_V3_WRITER_TEST_EXECUTABLE}"
    "$<TARGET_FILE:bc_native_v3_engine_test>")
endif()
"""



PATCHES = (
    # The manifest header/implementation are owned by components/serving.
    FilePatch(
        path="core/CMakeLists.txt",
        description="Link the shared serving manifest v1 library, built from BigCherry's source list, into rad_core.",
        edits=(Edit(
            id="link-bc-serving-v2",
            anchor=r"^target_link_libraries\(rad_core PUBLIC rad_abi \$\{CMAKE_DL_LIBS\} Threads::Threads\)$",
            text=_CORE_LINK,
            guard=r"include\(\"\$\{BC_SERVING_V2_CMAKE\}\"\)",
            max_span_lines=1,
            expect_matches=1,
            rationale="rad_core's unique link line; the shared library is included and linked right after it, failing "
                      "closed when the lane did not pass its .cmake.",
        ),),
    ),
    FilePatch(
        path="tests/CMakeLists.txt",
        description="Register the native fleet config test after the upstream test harness enables CTest.",
        edits=(Edit(
            id="register-bc-native-fleet-config-test",
            anchor=r"^get_property\(RAD_ALL_TESTS DIRECTORY PROPERTY TESTS\)$",
            text=_CORE_TEST_REGISTRATION,
            mode="insert_before",
            guard=r"^if\(TARGET bc_native_fleet_config_test\)$",
            expect_matches=1,
            rationale="tests/CMakeLists is included after enable_testing; register whichever isolated reader test exists before the final test-label pass.",
        ),),
    ),
    FilePatch(
        path="core/main.cpp",
        description="Call the owned native admission entry once, retaining the original YAML entrypoint as typed callable",
        edits=(
            Edit(
                id="include-bc-serving-manifest",
                anchor=r'^#include "util/stallwatch\.h"$',
                text='\n#include "bc_serving_manifest.h"',
                guard=r'^#include "bc_serving_manifest\.h"$',
                expect_matches=1,
                rationale="The upstream last include is a stable typed extension point.",
            ),
            Edit(
                id="bc-serving-owned-entry",
                anchor=r'^int main\(int argc, char\*\* argv\) \{$',
                text=('\n    return bc_serving_engine_entry(argc, argv, '
                      '[](int argc, char** argv) -> int {'),
                guard=r'bc_serving_engine_entry\(argc, argv,',
                expect_matches=1,
                rationale="Use the component-owned managed startup only when the strict manifest reader activates it; preserve stock YAML in the callable",
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
        ),
    ),
)
