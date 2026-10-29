# moe-remote: the remote-MoE hand-off. The helper process (r3_moe_worker) computes routed experts for an engine on
# another card family, and the libr3moe plugin is the engine-side mailbox (RAD02, RCS04). Arch-neutral in its
# protocol; the lane kernels it runs are gfx1100 today (RCS05 moves them behind moe_backend.h).
#
#   cmake -S engines/radiance/components/moe-remote -B <build> -G Ninja -DCMAKE_PREFIX_PATH=<installed radiance> \
#         -DRADIANCE_SRC=<radiance source tree> [-DR3MOE_TARGETS="gfx1100;gfx1201"]
cmake_minimum_required(VERSION 3.21)
project(moe_remote LANGUAGES CXX HIP)
find_package(radiance REQUIRED)
rad_plugin_defaults()
find_package(Threads REQUIRED)

set(RADIANCE_SRC "" CACHE PATH "radiance source tree (the plugin ABI headers are read from it)")
if(NOT EXISTS "${RADIANCE_SRC}/abi")
  message(FATAL_ERROR "moe-remote: -DRADIANCE_SRC must name a radiance source tree (no ${RADIANCE_SRC}/abi)")
endif()
if(NOT RAD_HIP_RUNTIME)
  find_library(RAD_HIP_RUNTIME amdhip64 HINTS /opt/rocm/lib ENV ROCM_PATH PATH_SUFFIXES lib REQUIRED)
endif()

# The helper process. It consumes only the installed container ABI. Its GPU mailbox service owns one HIP stream per
# producer rank; engine-side publication/join are the plugin's kernels.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/r3_moe_worker.cpp PROPERTIES LANGUAGE HIP)
add_executable(r3_moe_worker ${CMAKE_CURRENT_SOURCE_DIR}/r3_moe_worker.cpp)
set_property(TARGET r3_moe_worker PROPERTY HIP_ARCHITECTURES gfx1100 gfx1201)
target_include_directories(r3_moe_worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} $ENV{ROCM_PATH}/include)
target_compile_definitions(r3_moe_worker PRIVATE __HIP_PLATFORM_AMD__)
# The helper's queue warm-up (worker_queue_warm.h) creates its placeholder queues through the HSA runtime.
if(NOT RAD_HSA_RUNTIME)
  find_library(RAD_HSA_RUNTIME hsa-runtime64 HINTS /opt/rocm/lib ENV ROCM_PATH PATH_SUFFIXES lib REQUIRED)
endif()
target_link_libraries(r3_moe_worker PRIVATE radiance::abi ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME} ${RAD_HSA_RUNTIME} Threads::Threads)
set_target_properties(r3_moe_worker PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/helpers)

# Remote-MoE mailbox: its own plugin, for both card families, so the hand-off runs beside libr3 on the XTX ranks
# and beside libr4d on an R9700 engine. Engine patch rad0002 declares the device ops and exports the extension point
# bc_moe_remote_register; the hook resolves it dynamically so schema/selftest loaders keep no hard dependency on the
# engine core.
set(R3MOE_TARGETS "gfx1100;gfx1201" CACHE STRING "GPU targets the remote-MoE mailbox plugin is compiled for")
rad_add_plugin(libr3moe kernels
  SOURCES     ${CMAKE_CURRENT_SOURCE_DIR}/moe_mailbox_plugin.cpp ${CMAKE_CURRENT_SOURCE_DIR}/moe_remote_hook.cpp
  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/moe_mailbox_ops.hip
  GPU_TARGETS ${R3MOE_TARGETS}
  INCLUDES    ${CMAKE_CURRENT_SOURCE_DIR})
string(REPLACE ";" "," r3moe_targets_csv "${R3MOE_TARGETS}")
target_compile_definitions(libr3moe PRIVATE R3MOE_TARGETS="${r3moe_targets_csv}")
target_include_directories(libr3moe PRIVATE ${RADIANCE_SRC}/abi)
set_target_properties(libr3moe PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
target_link_libraries(libr3moe PRIVATE ${CMAKE_DL_LIBS})
add_dependencies(libr3moe r3_moe_worker)

include(CTest)
if(BUILD_TESTING)
  foreach(t moe_remote_shm_test:r3_moe_shm_test moe_backend_test:r3_moe_backend_test moe_fp8_ref_test:r3_moe_fp8_ref_test combine_f16_test:r3_combine_f16_test)
    string(REPLACE ":" ";" parts ${t})
    list(GET parts 0 src)
    list(GET parts 1 name)
    add_executable(${name} EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/${src}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${name} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
  target_link_libraries(r3_moe_shm_test PRIVATE Threads::Threads)
  # RAD29: route compaction vs the per-route-atomic loop it replaced (equivalence; --bench for timing). Needs a GPU.
  add_executable(r3_route_compact_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/route_compact_test.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/route_compact_test.hip PROPERTIES LANGUAGE HIP)
  set_property(TARGET r3_route_compact_test PROPERTY HIP_ARCHITECTURES gfx1100 gfx1201)
  target_include_directories(r3_route_compact_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_route_compact_test PRIVATE radiance::abi ${RAD_HIP_RUNTIME})
  set_target_properties(r3_route_compact_test PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  add_test(NAME r3_route_compact_test COMMAND r3_route_compact_test)
endif()
