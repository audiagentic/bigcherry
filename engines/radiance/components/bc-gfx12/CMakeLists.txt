# bc-gfx12: BigCherry's gfx1201 (R9700) kernel rows, a plugin listed ahead of libr4d in --kernels (RAD15).
#
#   cmake -S engines/radiance/components/bc-gfx12 -B <build> -G Ninja -DCMAKE_PREFIX_PATH=<installed radiance> \
#         -DRADIANCE_SRC=<radiance source tree>
cmake_minimum_required(VERSION 3.21)
project(bc_gfx12 LANGUAGES CXX HIP)
find_package(radiance REQUIRED)
rad_plugin_defaults()

set(RADIANCE_SRC "" CACHE PATH "radiance source tree (the plugin ABI headers are read from it)")
if(NOT EXISTS "${RADIANCE_SRC}/abi")
  message(FATAL_ERROR "bc-gfx12: -DRADIANCE_SRC must name a radiance source tree (no ${RADIANCE_SRC}/abi)")
endif()

rad_add_plugin(libbcgfx12 kernels
  SOURCES     ${CMAKE_CURRENT_SOURCE_DIR}/bc_gfx12_plugin.cpp
  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/bc_hc_read.hip
  GPU_TARGETS gfx1201
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(libbcgfx12 PRIVATE ${RADIANCE_SRC}/abi)
set_target_properties(libbcgfx12 PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_link_libraries(libbcgfx12 PRIVATE ${CMAKE_DL_LIBS})
