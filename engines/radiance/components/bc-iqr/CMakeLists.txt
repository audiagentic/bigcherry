# bc-iqr: BigCherry's owned IQR kernel rows for gfx1100 and gfx1201, a plugin listed ahead of libr3 and libr4d in --kernels
# (RCS23). One plugin, both architectures; its rows reuse the base library's layout hooks.
#
#   cmake -S engines/radiance/components/bc-iqr -B <build> -G Ninja -DCMAKE_PREFIX_PATH=<installed radiance> \
#         -DRADIANCE_SRC=<radiance source tree>
cmake_minimum_required(VERSION 3.21)
project(bc_iqr LANGUAGES CXX HIP)
find_package(radiance REQUIRED)
rad_plugin_defaults()

set(RADIANCE_SRC "" CACHE PATH "radiance source tree (the plugin ABI headers and libr4d's headers are read from it)")
if(NOT EXISTS "${RADIANCE_SRC}/abi" OR NOT EXISTS "${RADIANCE_SRC}/libr4d/r4d_iqr.h")
  message(FATAL_ERROR "bc-iqr: -DRADIANCE_SRC must name a radiance source tree (no ${RADIANCE_SRC}/abi or libr4d)")
endif()

rad_add_plugin(libbciqr kernels
  SOURCES     ${CMAKE_CURRENT_SOURCE_DIR}/bc_iqr_plugin.cpp
  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/bc_iqr_embed.hip ${CMAKE_CURRENT_SOURCE_DIR}/bc_iqr_vocab.hip
  GPU_TARGETS gfx1100 gfx1201
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR} ${RADIANCE_SRC}/libr4d ${CMAKE_CURRENT_SOURCE_DIR}/../libr3)   # libr3: r3_compat.h for the gfx1100 pass
target_include_directories(libbciqr PRIVATE ${RADIANCE_SRC}/abi)
# libr4d builds its device code at C++17: the owned copies of its units do the same (contraction is off for every plugin: rad_add_plugin).
set_target_properties(libbciqr PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_link_libraries(libbciqr PRIVATE ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME})
