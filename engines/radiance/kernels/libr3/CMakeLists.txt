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
set(R3_ONLY "" CACHE STRING "build only these units (for a compile check of one unit); empty = all not excluded")
if(NOT EXISTS "${RADIANCE_SRC}/libr4d/CMakeLists.txt")
  message(FATAL_ERROR "libr3: -DRADIANCE_SRC must name a radiance source tree (no ${RADIANCE_SRC}/libr4d)")
endif()
# libr4d's sources, copied into the build tree with rewrites.txt applied (text r3_compat.h cannot supply as a
# function, such as an inline-assembly mnemonic). Units and headers are built from the copy.
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(r4d ${CMAKE_CURRENT_BINARY_DIR}/r4d-src)
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/r3_rewrite.py ${RADIANCE_SRC}/libr4d ${r4d}
          ${CMAKE_CURRENT_SOURCE_DIR}/rewrites.txt
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
  string(REPLACE "\${RAD_W4_GROUP}" "128" flags "${flags}")
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
  "rad_plugin_info=r4d_up_plugin_info;rad_kernel_count=r4d_up_kernel_count;rad_kernel_at=r4d_up_kernel_at;rad_kernel_concurrent=r4d_up_kernel_concurrent")

rad_add_plugin(libr3 kernels
  SOURCES     ${declaration} ${CMAKE_CURRENT_SOURCE_DIR}/r3_registry.cpp
  HIP_SOURCES ${device}
  GPU_TARGETS ${R3_TARGETS}
  INCLUDES    ${r4d} ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(libr3 PRIVATE R3_TARGETS="${R3_TARGETS}")
# libr4d builds its device code at C++17 (its own build file says why); the same here
set_target_properties(libr3 PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)

# Stubs for the entry points of the units left out, made from what the objects leave undefined.
find_program(R3_NM nm REQUIRED)
set(stub_src ${CMAKE_CURRENT_BINARY_DIR}/r3_stubs.S)
set(stub_obj ${CMAKE_CURRENT_BINARY_DIR}/r3_stubs.o)
add_custom_command(TARGET libr3 PRE_LINK
  COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/r3_gen_stubs.py ${R3_NM} ${stub_src} $<TARGET_OBJECTS:libr3>
  COMMAND ${CMAKE_CXX_COMPILER} -c ${stub_src} -o ${stub_obj}
  COMMAND_EXPAND_LISTS VERBATIM)
target_link_options(libr3 PRIVATE ${stub_obj})
