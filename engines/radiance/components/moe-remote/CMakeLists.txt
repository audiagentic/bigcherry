# moe-remote: the remote-MoE hand-off. The helper process (r3_moe_worker) computes routed experts for an engine on
# another card family, and the libr3moe plugin is the engine-side mailbox (RAD02, RCS04). Arch-neutral in its
# protocol; the cards it runs on are the card types in engines/radiance/archs.toml (RAD23): the helper and the mailbox plugin are
# built for every one of them, so a new card type is a stanza there and not an edit here.
#
#   cmake -S engines/radiance/components/moe-remote -B <build> -G Ninja -DCMAKE_PREFIX_PATH=<installed radiance> \
#         -DRADIANCE_SRC=<radiance source tree> [-DR3MOE_TARGETS="gfx1100;gfx1201"]
cmake_minimum_required(VERSION 3.21)
project(moe_remote LANGUAGES CXX HIP)
find_package(radiance REQUIRED)
rad_plugin_defaults()
find_package(Threads REQUIRED)

# GPU targets: the card types of archs.toml, unless -DR3MOE_TARGETS narrows them (a quick build for one card).
file(STRINGS ${CMAKE_CURRENT_SOURCE_DIR}/../../archs.toml arch_lines REGEX "^\\[arch\\.gfx[0-9a-f]+\\]")
set(arch_targets "")
foreach(line ${arch_lines})
  string(REGEX MATCH "arch\\.(gfx[0-9a-f]+)" _ "${line}")
  list(APPEND arch_targets ${CMAKE_MATCH_1})
endforeach()
if(NOT arch_targets)
  message(FATAL_ERROR "moe-remote: no [arch.gfx...] table in engines/radiance/archs.toml")
endif()
set(R3MOE_TARGETS "${arch_targets}" CACHE STRING "GPU targets the remote-MoE helper and mailbox plugin are compiled for (default: every card type in archs.toml)")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../../archs.toml)

set(RADIANCE_SRC "" CACHE PATH "radiance source tree (the plugin ABI headers are read from it)")
if(NOT EXISTS "${RADIANCE_SRC}/abi")
  message(FATAL_ERROR "moe-remote: -DRADIANCE_SRC must name a radiance source tree (no ${RADIANCE_SRC}/abi)")
endif()
# RSC06's helper kernel-configuration ABI is mandatory; never build an environment-dependent helper path.
if(NOT EXISTS "${RADIANCE_SRC}/libr4d/bc_helper_config.h")
  message(FATAL_ERROR "Apply managed patch rad0012_helper_kernel_config to RADIANCE_SRC before building this component")
endif()
if(NOT RAD_HIP_RUNTIME)
  find_library(RAD_HIP_RUNTIME amdhip64 HINTS /opt/rocm/lib ENV ROCM_PATH PATH_SUFFIXES lib REQUIRED)
endif()

# The helper process. It consumes only the installed container ABI. Its GPU mailbox service owns one HIP stream per
# producer rank; engine-side publication/join are the plugin's kernels.
set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/r3_moe_worker.cpp PROPERTIES LANGUAGE HIP)
add_executable(r3_moe_worker ${CMAKE_CURRENT_SOURCE_DIR}/r3_moe_worker.cpp)
set_property(TARGET r3_moe_worker PROPERTY HIP_ARCHITECTURES ${R3MOE_TARGETS})
target_include_directories(r3_moe_worker PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} $ENV{ROCM_PATH}/include ${RADIANCE_SRC}/libr4d)   # libr4d r4d_q8.h, r4d_fwht.h for moe_lane_quant.h
target_compile_definitions(r3_moe_worker PRIVATE __HIP_PLATFORM_AMD__)
# The helper's queue warm-up (worker_queue_warm.h) creates its placeholder queues through the HSA runtime.
if(NOT RAD_HSA_RUNTIME)
  find_library(RAD_HSA_RUNTIME hsa-runtime64 HINTS /opt/rocm/lib ENV ROCM_PATH PATH_SUFFIXES lib REQUIRED)
endif()
# The helper reads its manifest section through the shared strict v1 reader (bc_serving_v2, the one the engine and bc-serve
# link): no second parser, no environment settings.
include("${CMAKE_CURRENT_SOURCE_DIR}/../serving/bc_serving_v2.cmake")
# Expert ownership and the container reader are bc_experts, the one implementation (the serving component builds and tests
# it; bc-serve's planning will use it in a later slice).
include("${CMAKE_CURRENT_SOURCE_DIR}/../serving/bc_experts.cmake")
target_link_libraries(r3_moe_worker PRIVATE radiance::abi bc_experts bc_serving_v2 ${CMAKE_DL_LIBS} ${RAD_HIP_RUNTIME} ${RAD_HSA_RUNTIME} Threads::Threads)
set_target_properties(r3_moe_worker PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/helpers)

# Remote-MoE mailbox: its own plugin, for both card families, so the hand-off runs beside libr3 on the XTX ranks
# and beside libr4d on an R9700 engine. Engine patch rad0002 declares the device ops and opens one generation per engine
# through the factory this plugin exports (bc_moe_remote_plugin_open, found with dlsym on the plugin's own handle), so
# schema/selftest loaders keep no dependency on the engine core and loading the library starts nothing.
rad_add_plugin(libr3moe kernels
  SOURCES     ${CMAKE_CURRENT_SOURCE_DIR}/moe_mailbox_plugin.cpp ${CMAKE_CURRENT_SOURCE_DIR}/moe_remote_hook.cpp
  # BU01: inverse-map device kernel is isolated so edits do not rebuild the
  # substantially larger mailbox publish/merge/route-mark device-code unit.
  HIP_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/moe_mailbox_ops.hip
              ${CMAKE_CURRENT_SOURCE_DIR}/mailbox_inverse_kernel.hip
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
  foreach(t moe_remote_topology_v26_test:r3_topology_v26_test moe_v26_gpu_contract_test:r3_v26_gpu_contract_test worker_topology_v26_test:r3_worker_topology_v26_test gemv_geometry_test:r3_gemv_geometry_test mailbox_trace_test:r3_mailbox_trace_test moe_remote_shm_test:r3_moe_shm_test moe_backend_test:r3_moe_backend_test worker_share_test:r3_worker_share_test worker_heat_test:r3_worker_heat_test worker_heat_source_test:r3_heat_source_test worker_policy_input_test:r3_policy_input_test worker_policy_live_test:r3_policy_live_test worker_policy_config_test:r3_policy_config_test worker_policy_driver_test:r3_policy_driver_test worker_policy_effect_test:r3_policy_effect_test worker_policy_measure_test:r3_policy_measure_test worker_policy_calibration_test:r3_policy_calibration_test worker_policy_derive_test:r3_policy_derive_test worker_policy_probe_test:r3_policy_probe_test worker_policy_replay_test:r3_policy_replay_test worker_slot_pool_test:r3_worker_slot_pool_test replay_memory_test:r3_replay_memory_test replay_bootstrap_test:r3_replay_bootstrap_test moe_remote_epoch_test:r3_moe_epoch_test moe_fp8_ref_test:r3_moe_fp8_ref_test combine_f16_test:r3_combine_f16_test combine_pool_test:r3_combine_pool_test combine_epoch_test:r3_combine_epoch_test forced_move_test:r3_forced_move_test segment_version_test:r3_segment_version_test test_rad46_rows_policy:r3_rad46_rows_policy_test tp_exactly_once_test:r3_tp_exactly_once_test worker_footprint_test:r3_worker_footprint_test worker_credited_heat_source_test:r3_credited_heat_source_test worker_credited_heat_ring_test:r3_credited_heat_ring_test)
    string(REPLACE ":" ";" parts ${t})
    list(GET parts 0 src)
    list(GET parts 1 name)
    add_executable(${name} EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/${src}.cpp)
    target_include_directories(${name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set_target_properties(${name} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_test(NAME ${name} COMMAND ${name})
  endforeach()
  # Registry enumerates kernel rows before Engine group/helper binding.
  # Rows remain fixed while runtime eligibility is bound per generation.
  add_executable(r3_moe_mailbox_lifetime_test EXCLUDE_FROM_ALL
    moe_mailbox_lifetime_test.cpp moe_mailbox_plugin.cpp)
  target_include_directories(r3_moe_mailbox_lifetime_test PRIVATE ${RADIANCE_SRC}/abi)
  target_compile_definitions(r3_moe_mailbox_lifetime_test PRIVATE
    R3MOE_TARGETS="${r3moe_targets_csv}")
  set_target_properties(r3_moe_mailbox_lifetime_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_moe_mailbox_lifetime_test COMMAND r3_moe_mailbox_lifetime_test)
  # The bounded layer-map consumption wait (alive-but-never-consuming helper; sticky fault; teardown not blocked).
  add_executable(r3_layer_map_wait_test EXCLUDE_FROM_ALL layer_map_wait_test.cpp)
  target_include_directories(r3_layer_map_wait_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_layer_map_wait_test PRIVATE Threads::Threads)
  set_target_properties(r3_layer_map_wait_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_layer_map_wait_test COMMAND r3_layer_map_wait_test)
  # Startup stage timing of the helper expert bootstrap: aggregation, the one-line log format, scope measurement. Host-only.
  add_executable(r3_bootstrap_timing_test EXCLUDE_FROM_ALL worker_bootstrap_timing_test.cpp)
  target_include_directories(r3_bootstrap_timing_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_bootstrap_timing_test PRIVATE Threads::Threads)
  set_target_properties(r3_bootstrap_timing_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_bootstrap_timing_test COMMAND r3_bootstrap_timing_test)
  # The helper registration barrier with fake helpers (late valid, never registers, dies) and its deadline source.
  add_executable(r3_helper_registration_test EXCLUDE_FROM_ALL helper_registration_test.cpp)
  target_include_directories(r3_helper_registration_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_helper_registration_test PRIVATE Threads::Threads)
  set_target_properties(r3_helper_registration_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_helper_registration_test COMMAND r3_helper_registration_test)
  # Closing a generation: refuse new borrows, cancel the borrowers' waits, wait for them, then free (real seal wait).
  add_executable(r3_hook_retire_test EXCLUDE_FROM_ALL hook_retire_test.cpp)
  target_include_directories(r3_hook_retire_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_hook_retire_test PRIVATE Threads::Threads)
  set_target_properties(r3_hook_retire_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_hook_retire_test COMMAND r3_hook_retire_test)
  # The production close orchestration and the direct drain wait with two paused borrowers and a clean close.
  add_executable(r3_generation_close_test EXCLUDE_FROM_ALL generation_close_test.cpp)
  target_include_directories(r3_generation_close_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_generation_close_test PRIVATE Threads::Threads)
  set_target_properties(r3_generation_close_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_generation_close_test COMMAND r3_generation_close_test)
  # The identity of the one live hook (serial handles; stale / foreign / address-reuse refused). Pure host code.
  add_executable(r3_hook_identity_test EXCLUDE_FROM_ALL hook_identity_test.cpp)
  target_include_directories(r3_hook_identity_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_hook_identity_test PRIVATE Threads::Threads)
  set_target_properties(r3_hook_identity_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_hook_identity_test COMMAND r3_hook_identity_test)
  # The explicit factory bind against the real libr3moe, through the engine's own generation code (rad0002). Host-only.
  if(EXISTS "${RADIANCE_SRC}/core/place/bc_moe_remote.cpp")
    add_executable(r3_moe_factory_bind_test EXCLUDE_FROM_ALL
      moe_factory_bind_test.cpp ${RADIANCE_SRC}/core/place/bc_moe_remote.cpp)
    target_include_directories(r3_moe_factory_bind_test PRIVATE ${RADIANCE_SRC}/abi)
    target_link_libraries(r3_moe_factory_bind_test PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    set_target_properties(r3_moe_factory_bind_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
    add_dependencies(r3_moe_factory_bind_test libr3moe)
    add_test(NAME r3_moe_factory_bind_test COMMAND r3_moe_factory_bind_test $<TARGET_FILE:libr3moe>)
  endif()
  # Keep the declared, card-free large-K geometry/admission target bounded.
  set_tests_properties(r3_gemv_geometry_test PROPERTIES TIMEOUT 60)
  # The typed policy and forced-schedule settings come from the manifest's generated types (bc_serving_v2's headers).
  target_link_libraries(r3_policy_config_test PRIVATE Threads::Threads bc_serving_v2)
  target_link_libraries(r3_policy_driver_test PRIVATE Threads::Threads bc_serving_v2)
  target_link_libraries(r3_policy_calibration_test PRIVATE bc_serving_v2)
  target_link_libraries(r3_policy_derive_test PRIVATE bc_serving_v2)
  target_link_libraries(r3_policy_probe_test PRIVATE bc_serving_v2)
  target_link_libraries(r3_policy_replay_test PRIVATE bc_serving_v2)
  target_link_libraries(r3_forced_move_test PRIVATE bc_serving_v2)
  target_link_libraries(r3_worker_slot_pool_test PRIVATE bc_experts)
  add_executable(r3_helper_settings_test EXCLUDE_FROM_ALL helper_settings_test.cpp)
  target_link_libraries(r3_helper_settings_test PRIVATE bc_serving_v2)
  target_include_directories(r3_helper_settings_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(r3_helper_settings_test PRIVATE
    BC_CONTRACT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME r3_helper_settings_test COMMAND r3_helper_settings_test)
  # Exercise the real config ABI and both helper/engine access paths without HIP.
  add_library(r3_helper_plugin_fixture SHARED EXCLUDE_FROM_ALL helper_plugin_fixture.cpp)
  target_include_directories(r3_helper_plugin_fixture PRIVATE ${RADIANCE_SRC}/libr4d ${RADIANCE_SRC}/abi)
  target_link_options(r3_helper_plugin_fixture PRIVATE -Wl,-Bsymbolic-functions)
  target_compile_features(r3_helper_plugin_fixture PRIVATE cxx_std_17)
  add_executable(r3_helper_plugin_config_test EXCLUDE_FROM_ALL helper_plugin_config_test.cpp)
  target_include_directories(r3_helper_plugin_config_test PRIVATE ${RADIANCE_SRC}/libr4d)
  target_link_libraries(r3_helper_plugin_config_test PRIVATE bc_serving_v2 ${CMAKE_DL_LIBS})
  add_dependencies(r3_helper_plugin_config_test r3_helper_plugin_fixture)
  add_test(NAME r3_helper_plugin_config_test COMMAND r3_helper_plugin_config_test $<TARGET_FILE:r3_helper_plugin_fixture>)
  target_link_libraries(r3_replay_bootstrap_test PRIVATE Threads::Threads)
  target_compile_definitions(r3_worker_slot_pool_test PRIVATE R3_CONFIGS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../configs")
  # offline replay of the live placement policy over a recorded heat series (a tool, not a test)
  add_executable(r3_policy_replay EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/worker_policy_replay_main.cpp)
  target_include_directories(r3_policy_replay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  set_target_properties(r3_policy_replay PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  target_link_libraries(r3_policy_replay PRIVATE bc_serving_v2)
  add_executable(r3_mailbox_completion_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/mailbox_completion_test.cpp)
  target_include_directories(r3_mailbox_completion_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_mailbox_completion_test PRIVATE Threads::Threads)
  set_target_properties(r3_mailbox_completion_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_mailbox_completion_test COMMAND r3_mailbox_completion_test)
  add_executable(r3_owner_generation_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/owner_generation_test.cpp)
  target_include_directories(r3_owner_generation_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_owner_generation_test PRIVATE Threads::Threads)
  set_target_properties(r3_owner_generation_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_owner_generation_test COMMAND r3_owner_generation_test)
  add_executable(r3_admission_ledger_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/admission_ledger_test.cpp)
  target_include_directories(r3_admission_ledger_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_admission_ledger_test PRIVATE Threads::Threads)
  set_target_properties(r3_admission_ledger_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME r3_admission_ledger_test COMMAND r3_admission_ledger_test)
  target_link_libraries(r3_moe_shm_test PRIVATE Threads::Threads)
  target_link_libraries(r3_combine_pool_test PRIVATE Threads::Threads)
  target_link_libraries(r3_combine_epoch_test PRIVATE Threads::Threads)
  target_link_libraries(r3_segment_version_test PRIVATE Threads::Threads)
  # RAD29: route compaction vs the per-route-atomic loop it replaced (equivalence; --bench for timing). Needs a GPU.
  add_executable(r3_route_compact_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/route_compact_test.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/route_compact_test.hip PROPERTIES LANGUAGE HIP)
  set_property(TARGET r3_route_compact_test PROPERTY HIP_ARCHITECTURES ${R3MOE_TARGETS})
  target_include_directories(r3_route_compact_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_route_compact_test PRIVATE radiance::abi ${RAD_HIP_RUNTIME})
  set_target_properties(r3_route_compact_test PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  add_test(NAME r3_route_compact_test COMMAND r3_route_compact_test)
  # RHP11: GPU-only HIP graph proxy for the immutable AQL kernarg pointer.
  # This target is built card-free but must only run under an explicit GPU lease.
  add_executable(r3_ticket_replay_test EXCLUDE_FROM_ALL ${CMAKE_CURRENT_SOURCE_DIR}/ticket_replay_test.hip)
  set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/ticket_replay_test.hip PROPERTIES LANGUAGE HIP)
  set_property(TARGET r3_ticket_replay_test PROPERTY HIP_ARCHITECTURES ${R3MOE_TARGETS})
  target_include_directories(r3_ticket_replay_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(r3_ticket_replay_test PRIVATE ${RAD_HIP_RUNTIME})
  set_target_properties(r3_ticket_replay_test PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  add_test(NAME r3_ticket_replay_test COMMAND r3_ticket_replay_test)
  # Both need a GPU: labelled so the card-free run (bigcherry component-tests) knows they are deliberately not declared.
  set_tests_properties(r3_route_compact_test r3_ticket_replay_test PROPERTIES LABELS gpu)
  # Radiance's plugin module defaults CMAKE_BUILD_TYPE to RelWithDebInfo, which adds -DNDEBUG and turns every assert()
  # in these tests into a no-op. Test executables keep their assertions; the assert-based tests refuse to build without.
  get_property(_bc_tests DIRECTORY PROPERTY TESTS)
  foreach(_bc_test IN LISTS _bc_tests)
    if(TARGET ${_bc_test})
      target_compile_options(${_bc_test} PRIVATE -UNDEBUG)
    endif()
  endforeach()
endif()
