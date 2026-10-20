# ngram-helper: the opt-in n-gram embedding gather on a spare GPU (R3_NGRAM_DEV), split out of libr3 (RCS06).
#
#   libr3ngram (kernels/)          a host-domain kernel row for the n-gram lookup; list it in --kernels to use it
#   libr3_ngram_helper (helpers/)  the multi-arch device code the row dlopens for the sparse gather
#   r3_ngram_test                  the helper's correctness and speed harness (owns a non-rank GPU)
#
#   cmake -S engines/radiance/components/ngram-helper -B <build> -G Ninja -DCMAKE_PREFIX_PATH=<installed radiance> \
#         -DRADIANCE_SRC=<radiance source tree>
cmake_minimum_required(VERSION 3.21)
project(ngram_helper LANGUAGES CXX HIP)
find_package(radiance REQUIRED)
rad_plugin_defaults()

set(RADIANCE_SRC "" CACHE PATH "radiance source tree (the plugin ABI headers are read from it)")
if(NOT EXISTS "${RADIANCE_SRC}/abi")
  message(FATAL_ERROR "ngram-helper: -DRADIANCE_SRC must name a radiance source tree (no ${RADIANCE_SRC}/abi)")
endif()
set(R3NGRAM_TARGETS "gfx1100;gfx1201;gfx1030" CACHE STRING "GPU targets of the n-gram gather")

# The row itself runs on the host; the plugin carries no device code, so it builds as a plain host module.
rad_add_plugin(libr3ngram kernels
  SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/ngram_plugin.cpp ${CMAKE_CURRENT_SOURCE_DIR}/r3_ngram_host.cpp
  INCLUDES ${CMAKE_CURRENT_SOURCE_DIR})
string(REPLACE ";" "," r3ngram_targets_csv "${R3NGRAM_TARGETS}")
target_compile_definitions(libr3ngram PRIVATE R3NGRAM_TARGETS="${r3ngram_targets_csv}")
target_include_directories(libr3ngram PRIVATE ${RADIANCE_SRC}/abi)
target_link_libraries(libr3ngram PRIVATE ${CMAKE_DL_LIBS})

# The sparse gather on a non-rank GPU, in a separate multi-arch code object. Beside the kernels directory, not in it:
# the engine loads every library there as a plugin and refuses one that exports no rad_plugin_abi_version.
add_library(r3_ngram_helper MODULE ${CMAKE_CURRENT_SOURCE_DIR}/r3_ngram_helper.hip)
target_link_libraries(r3_ngram_helper PRIVATE radiance::abi)
set_target_properties(r3_ngram_helper PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
set_property(TARGET r3_ngram_helper PROPERTY HIP_ARCHITECTURES ${R3NGRAM_TARGETS})
add_dependencies(libr3ngram r3_ngram_helper)
add_custom_command(TARGET libr3ngram POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:libr3ngram>/../helpers
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          $<TARGET_FILE:r3_ngram_helper> $<TARGET_FILE_DIR:libr3ngram>/../helpers/libr3_ngram_helper.so
  VERBATIM)

# The helper's correctness/perf harness. Not a ctest: it needs a GPU.
add_executable(r3_ngram_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/r3_ngram_test.cpp)
target_link_libraries(r3_ngram_test PRIVATE radiance::abi ${CMAKE_DL_LIBS})
