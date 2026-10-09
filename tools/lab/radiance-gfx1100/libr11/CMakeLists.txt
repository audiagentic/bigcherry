# Out-of-tree gfx1100 kernel plugin for the standalone Radiance engine.
# Source ABI target: Radiance 1.0.5 (mirror 6c11671); rebuild against active ABI.
cmake_minimum_required(VERSION 3.21)
project(libr11 LANGUAGES CXX HIP)
find_package(radiance REQUIRED)
rad_plugin_defaults()
rad_plugin_gpu_targets(R11_GPU_TARGETS "^gfx1100$")
if(NOT R11_GPU_TARGETS)
  message(FATAL_ERROR "libr11 needs -DRAD_GPU_TARGETS=gfx1100 (got '${RAD_GPU_TARGETS}')")
endif()
rad_add_plugin(libr11 kernels
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/rows.cpp
  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/elementwise.hip
  GPU_TARGETS ${R11_GPU_TARGETS})
set_target_properties(libr11 PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)

# Test is GPU-dependent and intentionally not registered with BigCherry's offline CI.
enable_testing()
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/selftest.cpp PROPERTIES LANGUAGE HIP)
add_executable(r11_selftest ${CMAKE_CURRENT_SOURCE_DIR}/selftest.cpp)
target_link_libraries(r11_selftest PRIVATE radiance::abi ${CMAKE_DL_LIBS})
rad_hipify(r11_selftest ${R11_GPU_TARGETS})
add_test(NAME r11_bf16_smoke COMMAND r11_selftest $<TARGET_FILE:libr11>)
