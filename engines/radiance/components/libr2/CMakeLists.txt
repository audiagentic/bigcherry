# libr2: radiance's device kernels for RDNA2 (gfx10.3, the RX 6000 series), built from libr4d's own sources.
#
# The same method as libr3 (../libr3): libr4d's units are compiled through the compatibility layer r3_compat.h, which is shared. What is
# RDNA2's own is the 16x16x16 tile (r3_mma16.h here: gfx10.3 has no WMMA) and the list of units it builds (units.txt: the ones the remote-MoE
# helper needs; the units that use integer WMMA or attention do not build for gfx10.3 yet). libr4d's row table is compiled in unchanged
# and rows whose unit is left out are dropped at load (r2_registry.cpp).
#
# Nothing of libr4d is copied into this repository: the sources are read from RADIANCE_SRC at the commit libr2 is built against.
#
#   cmake -S engines/radiance/components/libr2 -B <build> -G Ninja -DCMAKE_PREFIX_PATH=<installed radiance> \
#         -DRADIANCE_SRC=<radiance source tree> [-DR2_TARGETS=gfx1030]
cmake_minimum_required(VERSION 3.21)
project(libr2 LANGUAGES CXX HIP)
find_package(radiance REQUIRED)
rad_plugin_defaults()

set(R2_TARGETS "gfx1030" CACHE STRING "GPU targets libr2 is compiled for")
set(RADIANCE_SRC "" CACHE PATH "radiance source tree (libr4d is read from it)")
if(NOT EXISTS "${RADIANCE_SRC}/libr4d/CMakeLists.txt")
  message(FATAL_ERROR "libr2: -DRADIANCE_SRC must name a radiance source tree (no ${RADIANCE_SRC}/libr4d)")
endif()
set(LIBR3 ${CMAKE_CURRENT_SOURCE_DIR}/../libr3)

# libr4d's sources, copied into the build tree with rewrites.txt applied (the rewriter and r3_compat.h are libr3's).
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(r4d ${CMAKE_CURRENT_BINARY_DIR}/r4d-src)
execute_process(
  COMMAND ${Python3_EXECUTABLE} ${LIBR3}/r3_rewrite.py --source ${RADIANCE_SRC}/libr4d --out ${r4d}
          --rules ${CMAKE_CURRENT_SOURCE_DIR}/rewrites.txt
  RESULT_VARIABLE rewrite_rc OUTPUT_VARIABLE rewrite_out ERROR_VARIABLE rewrite_err)
message(STATUS "libr2: ${rewrite_out}")
if(NOT rewrite_rc EQUAL 0)
  message(FATAL_ERROR "libr2: r3_rewrite.py failed: ${rewrite_err}")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/rewrites.txt ${CMAKE_CURRENT_SOURCE_DIR}/units.txt)

# The unit list and each unit's flags are libr4d's own, read from its build file: `"unit|flags"`. libr2 builds the units in units.txt.
file(STRINGS ${RADIANCE_SRC}/libr4d/CMakeLists.txt unit_lines REGEX "^  \"r4d_[a-z0-9_]+\\|")
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/units.txt wanted REGEX "^r4d_")
set(device "")
set(built "")
foreach(line ${unit_lines})
  string(REGEX MATCH "\"(r4d_[a-z0-9_]+)\\|([^\"]*)\"" _ "${line}")
  set(unit ${CMAKE_MATCH_1})
  set(flags "${CMAKE_MATCH_2}")
  if(NOT unit IN_LIST wanted)
    continue()
  endif()
  set(wrapper ${CMAKE_CURRENT_BINARY_DIR}/units/${unit}.hip)
  # Relative to the wrapper (units/ beside r4d-src/) so its bytes do not depend on where the build tree lives.
  file(GENERATE OUTPUT ${wrapper} CONTENT "#include \"r3_compat.h\"\n#include \"../r4d-src/${unit}.hip\"\n")
  if(flags)
    set_source_files_properties(${wrapper} PROPERTIES COMPILE_OPTIONS "${flags}")
  endif()
  list(APPEND device ${wrapper})
  list(APPEND built ${unit})
endforeach()
foreach(unit ${wanted})
  if(NOT unit IN_LIST built)
    message(FATAL_ERROR "libr2: units.txt names ${unit}, which libr4d at this revision does not build")
  endif()
endforeach()
list(LENGTH built n_built)
message(STATUS "libr2: ${n_built} unit(s) built for ${R2_TARGETS}")

set(declaration ${r4d}/r4d_rows.cpp ${r4d}/r4d_layout.cpp ${r4d}/r4d_fp8_layout.cpp ${r4d}/r4d_shapes.cpp ${r4d}/r4d_fuse.cpp)
# libr4d's registry exports become r4d_up_*; libr2's own (r2_registry.cpp) take their place
set_source_files_properties(${r4d}/r4d_rows.cpp PROPERTIES COMPILE_DEFINITIONS
  "rad_plugin_info=r4d_up_plugin_info;rad_kernel_schema_count=r4d_up_kernel_schema_count;rad_kernel_schema_at=r4d_up_kernel_schema_at;rad_kernel_count=r4d_up_kernel_count;rad_kernel_at=r4d_up_kernel_at;rad_kernel_concurrent=r4d_up_kernel_concurrent;rad_kernel_collective=r4d_up_kernel_collective")

# r3_mma16.h is looked up with <>: libr2's directory first, so its tile replaces libr3's WMMA one; r3_compat.h itself is libr3's.
rad_add_plugin(libr2 kernels
  SOURCES     ${declaration} ${CMAKE_CURRENT_SOURCE_DIR}/r2_registry.cpp
  HIP_SOURCES ${device}
  GPU_TARGETS ${R2_TARGETS}
  INCLUDES    ${r4d} ${CMAKE_CURRENT_SOURCE_DIR} ${LIBR3})
target_compile_definitions(libr2 PRIVATE R2_TARGETS="${R2_TARGETS}")
# libr4d builds its device code at C++17 (its own build file says why); the same here
set_target_properties(libr2 PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_link_libraries(libr2 PRIVATE ${CMAKE_DL_LIBS})

# Stubs for the entry points of the units left out, made from what the objects leave undefined (libr3's generator).
find_program(R2_NM nm REQUIRED)
set(stub_src ${CMAKE_CURRENT_BINARY_DIR}/r3_stubs.S)
set(stub_obj ${CMAKE_CURRENT_BINARY_DIR}/r3_stubs.o)
add_custom_command(TARGET libr2 PRE_LINK
  COMMAND ${Python3_EXECUTABLE} ${LIBR3}/r3_gen_stubs.py ${R2_NM} ${stub_src} $<TARGET_OBJECTS:libr2>
  COMMAND ${CMAKE_CXX_COMPILER} -c ${stub_src} -o ${stub_obj}
  COMMAND_EXPAND_LISTS VERBATIM)
target_link_options(libr2 PRIVATE ${stub_obj})

target_include_directories(libr2 PRIVATE ${RADIANCE_SRC}/abi)
