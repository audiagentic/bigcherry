# libr3: radiance's device kernels for RDNA3 (gfx11), built from libr4d's own sources.
#
# libr4d is written for gfx12. libr3 compiles the same units for gfx11 through a compatibility layer (r3_compat.h)
# that supplies the gfx12-only instructions, and leaves out the units that layer does not cover yet (exclude.txt).
# libr4d's row table is compiled in unchanged; rows whose unit is left out are dropped at load (r3_registry.cpp), so
# the engine and rad-kbench see exactly the rows that have a kernel behind them.
#
# Nothing of libr4d is copied into this repository: the sources are read from RADIANCE_SRC at the commit libr3 is
# built against.
#
#   cmake -S engines/radiance/kernels/libr3 -B <build> -G Ninja -DCMAKE_PREFIX_PATH=<installed radiance> \
#         -DRADIANCE_SRC=<radiance source tree> [-DR3_TARGETS=gfx1100]
cmake_minimum_required(VERSION 3.21)
project(libr3 LANGUAGES CXX HIP)
find_package(radiance REQUIRED)
rad_plugin_defaults()

set(R3_TARGETS "gfx1100" CACHE STRING "GPU targets libr3 is compiled for")
set(RADIANCE_SRC "" CACHE PATH "radiance source tree (libr4d is read from it)")
option(R3_NATIVE "apply the native gfx11 forms in native/ (OFF = every kernel through the compatibility layer; OFF while the natives are re-ported to radiance 1.7.0, MEN12)" OFF)
set(R3_ONLY "" CACHE STRING "build only these units (for a compile check of one unit); empty = all not excluded")
if(NOT EXISTS "${RADIANCE_SRC}/libr4d/CMakeLists.txt")
  message(FATAL_ERROR "libr3: -DRADIANCE_SRC must name a radiance source tree (no ${RADIANCE_SRC}/libr4d)")
endif()
# libr4d's sources, copied into the build tree with rewrites.txt applied (text r3_compat.h cannot supply as a
# function, such as an inline-assembly mnemonic). Units and headers are built from the copy.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(r4d ${CMAKE_CURRENT_BINARY_DIR}/r4d-src)
set(native_arg)
if(R3_NATIVE)
  set(native_arg ${CMAKE_CURRENT_SOURCE_DIR}/native)
  file(GLOB native_rules ${CMAKE_CURRENT_SOURCE_DIR}/native/*.rw)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${native_rules})
endif()
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/r3_rewrite.py ${RADIANCE_SRC}/libr4d ${r4d}
          ${CMAKE_CURRENT_SOURCE_DIR}/rewrites.txt ${native_arg}
  RESULT_VARIABLE rewrite_rc OUTPUT_VARIABLE rewrite_out ERROR_VARIABLE rewrite_err)
message(STATUS "libr3: ${rewrite_out}")
if(NOT rewrite_rc EQUAL 0)
  message(FATAL_ERROR "libr3: r3_rewrite.py failed: ${rewrite_err}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rewrites.txt)

# The unit list and each unit's flags are libr4d's own, read from its build file: `"unit|flags"`.
file(STRINGS ${RADIANCE_SRC}/libr4d/CMakeLists.txt unit_lines REGEX "^  \"r4d_[a-z0-9_]+\\|")
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/exclude.txt excluded REGEX "^r4d_")
set(device "")
set(built "")
set(skipped "")
foreach(line ${unit_lines})
  string(REGEX MATCH "\"(r4d_[a-z0-9_]+)\\|([^\"]*)\"" _ "${line}")
  set(unit ${CMAKE_MATCH_1})
  set(flags "${CMAKE_MATCH_2}")
  if(R3_ONLY AND NOT unit IN_LIST R3_ONLY)
    list(APPEND skipped ${unit})
    continue()
  endif()
  if(unit IN_LIST excluded AND NOT R3_ONLY)
    list(APPEND skipped ${unit})
    continue()
  endif()
  set(wrapper ${CMAKE_CURRENT_BINARY_DIR}/units/${unit}.hip)
  file(GENERATE OUTPUT ${wrapper} CONTENT "#include \"r3_compat.h\"\n#include \"${r4d}/${unit}.hip\"\n")
  if(flags)
    set_source_files_properties(${wrapper} PROPERTIES COMPILE_OPTIONS "${flags}")
  endif()
  list(APPEND device ${wrapper})
  list(APPEND built ${unit})
endforeach()
list(LENGTH built n_built)
list(LENGTH skipped n_skipped)
message(STATUS "libr3: ${n_built} unit(s) built for ${R3_TARGETS}, ${n_skipped} left out")
if(NOT device)
  message(FATAL_ERROR "libr3: no unit to build")
endif()

set(declaration ${r4d}/r4d_rows.cpp ${r4d}/r4d_layout.cpp ${r4d}/r4d_fp8_layout.cpp ${r4d}/r4d_shapes.cpp ${r4d}/r4d_fuse.cpp)
# libr4d's registry exports become r4d_up_*; libr3's own (r3_registry.cpp) take their place
set_source_files_properties(${r4d}/r4d_rows.cpp PROPERTIES COMPILE_DEFINITIONS
  "rad_plugin_info=r4d_up_plugin_info;rad_kernel_schema_count=r4d_up_kernel_schema_count;rad_kernel_schema_at=r4d_up_kernel_schema_at;rad_kernel_count=r4d_up_kernel_count;rad_kernel_at=r4d_up_kernel_at;rad_kernel_concurrent=r4d_up_kernel_concurrent;rad_kernel_collective=r4d_up_kernel_collective")

rad_add_plugin(libr3 kernels
  SOURCES     ${declaration} ${CMAKE_CURRENT_SOURCE_DIR}/r3_registry.cpp ${CMAKE_CURRENT_SOURCE_DIR}/r3_ngram_host.cpp
  HIP_SOURCES ${device}
  GPU_TARGETS ${R3_TARGETS}
  INCLUDES    ${r4d} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(libr3 PRIVATE R3_TARGETS="${R3_TARGETS}")
# libr4d builds its device code at C++17 (its own build file says why); the same here
set_target_properties(libr3 PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_link_libraries(libr3 PRIVATE ${CMAKE_DL_LIBS})

# R3_NGRAM_DEV is a HOST kernel in libr3, but its sparse gather runs on a non-rank GPU.
# Keep that helper in a separate multi-arch code object: libr3 itself remains gfx11-only.
add_library(r3_ngram_helper MODULE ${CMAKE_CURRENT_SOURCE_DIR}/r3_ngram_helper.hip)
target_link_libraries(r3_ngram_helper PRIVATE radiance::abi)
set_target_properties(r3_ngram_helper PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
set_property(TARGET r3_ngram_helper PROPERTY HIP_ARCHITECTURES gfx1100 gfx1201 gfx1030)
add_dependencies(libr3 r3_ngram_helper)
# Beside the kernels directory, not in it: the engine loads every library there as a plugin and refuses one that
# exports no rad_plugin_abi_version.
add_custom_command(TARGET libr3 POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:libr3>/../helpers
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:r3_ngram_helper> $<TARGET_FILE_DIR:libr3>/../helpers/libr3_ngram_helper.so
  COMMAND ${CMAKE_COMMAND} -E rm -f $<TARGET_FILE_DIR:libr3>/libr3_ngram_helper.so
  VERBATIM)

# Stubs for the entry points of the units left out, made from what the objects leave undefined.
find_program(R3_NM nm REQUIRED)
set(stub_src ${CMAKE_CURRENT_BINARY_DIR}/r3_stubs.S)
set(stub_obj ${CMAKE_CURRENT_BINARY_DIR}/r3_stubs.o)
add_custom_command(TARGET libr3 PRE_LINK
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/r3_gen_stubs.py ${R3_NM} ${stub_src} $<TARGET_OBJECTS:libr3>
  COMMAND ${CMAKE_CXX_COMPILER} -c ${stub_src} -o ${stub_obj}
  COMMAND_EXPAND_LISTS VERBATIM)
target_link_options(libr3 PRIVATE ${stub_obj})

# radiance's kernel selftest, from the rewritten copy so that it runs on gfx11 cards (rewrites.txt). It dlopens the
# plugin it is given and carries a host reference for each row, including the MXFP4 GEMMs rad-kbench cannot reach.
# Not part of the default build: tools/lab/radiance/libr3-selftest.sh builds it.
if(NOT RAD_HIP_RUNTIME)
  find_library(RAD_HIP_RUNTIME amdhip64 HINTS /opt/rocm/lib ENV ROCM_PATH PATH_SUFFIXES lib REQUIRED)
endif()
add_executable(r3_selftest EXCLUDE_FROM_ALL
  ${r4d}/r4d_selftest.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/r3_wmma_perf.hip)
target_link_libraries(r3_selftest PRIVATE radiance::abi ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME})
target_include_directories(r3_selftest PRIVATE ${r4d} $ENV{ROCM_PATH}/include)
target_compile_definitions(r3_selftest PRIVATE __HIP_PLATFORM_AMD__)
set_property(TARGET r3_selftest PROPERTY HIP_ARCHITECTURES ${R3_TARGETS})

# Direct helper correctness/perf harness. It is separate from r3_selftest because it owns a non-rank HIP device.
add_executable(r3_ngram_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/r3_ngram_test.cpp)
target_link_libraries(r3_ngram_test PRIVATE radiance::abi ${CMAKE_DL_LIBS})

# Remote-MoE helper. The executable consumes only the installed container ABI. Its GPU mailbox
# service owns one HIP stream per producer rank; engine-side publication/join are libr3 kernels.
find_package(Threads REQUIRED)
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/helper/r3_moe_helper.cpp PROPERTIES LANGUAGE HIP)
add_executable(r3_moe_helper ${CMAKE_CURRENT_SOURCE_DIR}/helper/r3_moe_helper.cpp)
set_property(TARGET r3_moe_helper PROPERTY HIP_ARCHITECTURES gfx1100 gfx1201)
target_include_directories(r3_moe_helper PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helper $ENV{ROCM_PATH}/include)
target_include_directories(libr3 PRIVATE ${RADIANCE_SRC}/abi)

# Remote-MoE mailbox: its own plugin, for both card families, so the hand-off runs beside libr3 on the XTX ranks
# and beside libr4d on an R9700 engine (RAD02). Engine patch rad0002 declares the device ops and exports the
# extension point bc_moe_remote_register; the hook resolves it dynamically so schema/selftest loaders keep no hard
# dependency on the engine core.
set(R3MOE_TARGETS "gfx1100;gfx1201" CACHE STRING "GPU targets the remote-MoE mailbox plugin is compiled for")
rad_add_plugin(libr3moe kernels
  SOURCES     ${CMAKE_CURRENT_SOURCE_DIR}/helper/moe_mailbox_plugin.cpp ${CMAKE_CURRENT_SOURCE_DIR}/helper/moe_remote_hook.cpp
  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/helper/moe_mailbox_ops.hip
  GPU_TARGETS ${R3MOE_TARGETS}
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR})
string(REPLACE ";" "," r3moe_targets_csv "${R3MOE_TARGETS}")
target_compile_definitions(libr3moe PRIVATE R3MOE_TARGETS="${r3moe_targets_csv}")
target_include_directories(libr3moe PRIVATE ${RADIANCE_SRC}/abi)
set_target_properties(libr3moe PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_link_libraries(libr3moe PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(libr3moe r3_moe_helper)
target_compile_definitions(r3_moe_helper PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(r3_moe_helper PRIVATE radiance::abi ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME} Threads::Threads)
set_target_properties(r3_moe_helper PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/helpers)

include(CTest)
if(BUILD_TESTING)
  add_executable(r3_moe_shm_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/helper/moe_remote_shm_test.cpp)
  target_include_directories(r3_moe_shm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/helper)
  target_link_libraries(r3_moe_shm_test PRIVATE Threads::Threads)
  set_target_properties(r3_moe_shm_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_moe_shm_test COMMAND r3_moe_shm_test)
endif()
