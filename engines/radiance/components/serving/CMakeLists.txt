cmake_minimum_required(VERSION 3.21)
project(bc_serving_component LANGUAGES CXX)
find_package(radiance REQUIRED)
rad_plugin_defaults()

set(RADIANCE_SRC "" CACHE PATH "Radiance source tree, for the engine's vendored nlohmann/json")
if(NOT EXISTS "${RADIANCE_SRC}/vendor/nlohmann/json.hpp")
  message(FATAL_ERROR "serving: -DRADIANCE_SRC must contain vendor/nlohmann/json.hpp")
endif()

# RFC 8785 JCS, SHA-256 and the shared strict v1 reader (RSC08 slice 2): one source list, bc_serving_v2.cmake, which
# radiance's core also builds (managed patch rad0011).
set(BC_SERVING_V2_NLOHMANN_INCLUDE "${RADIANCE_SRC}/vendor")
include("${CMAKE_CURRENT_SOURCE_DIR}/bc_serving_v2.cmake")

# The serving manifest v2 contract (contract/README.md): the field registry contract/serving-v2.toml and its generator.
# GPU-free and independent of bc_serving, whose typed model it generates (generated/, committed; bc_contract_check
# fails CI when a committed file differs from the generator's output).
add_library(bc_contract STATIC contract/registry.cpp)
target_link_libraries(bc_contract PUBLIC bc_jcs)
target_include_directories(bc_contract PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/contract")
add_executable(bc_contract_gen contract/generate.cpp)
target_link_libraries(bc_contract_gen PRIVATE bc_contract)

# The helpers' expert ownership, shared with moe-remote's r3_moe_worker (bc_experts.cmake). bc-serve's manifest writer plans owner maps with it (manifest_writer.cpp).
include("${CMAKE_CURRENT_SOURCE_DIR}/bc_experts.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/owner_map.cmake")

# calibration.toml is compiled in: the seeds bc-serve derives with, each carrying its evidence.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/calibration.toml" bc_calibration_seed)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/calibration_seed.inc"
     "R\"bccal(${bc_calibration_seed})bccal\"\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/calibration.toml")

# The card types (engines/radiance/archs.toml) are compiled in: the same file the launcher and the Python tools read, so the
# resolver has no copy. The layouts are run-time data (--layouts / BIGCHERRY_LAYOUTS), never compiled in.
foreach(bc_data archs)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../../${bc_data}.toml" bc_data_seed)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/${bc_data}_seed.inc" "R\"bcdata(${bc_data_seed})bcdata\"\n")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../../${bc_data}.toml")
endforeach()

# helper_profile.toml is compiled in: the helper_common values bc-serve writes into a v1 manifest.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/helper_profile.toml" bc_helper_profile_seed)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/helper_profile_seed.inc"
     "R\"bchp(${bc_helper_profile_seed})bchp\"\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/helper_profile.toml")

# One read-only worker telemetry producer also used by the v2 S13 receipt tests.
# The runtime guard consumes the same typed publisher, not a second SHM parser.
add_library(bc_serving_v2_runtime_receipt STATIC v2/runtime_receipt.cpp)
target_link_libraries(bc_serving_v2_runtime_receipt PUBLIC bc_serving_v2)
target_include_directories(bc_serving_v2_runtime_receipt PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../moe-remote" "${RADIANCE_SRC}/vendor")
target_compile_features(bc_serving_v2_runtime_receipt PUBLIC cxx_std_20)

add_library(bc_serving STATIC serving.cpp config_loader.cpp session.cpp stored_manifest.cpp stored_manifest_cache.cpp s8_replay.cpp runtime_guards.cpp calibration_lab.cpp resources.cpp manifest_power.cpp power_transaction.cpp power_linux.cpp power_records.cpp power_diagnostics.cpp calibration.cpp live_policy.cpp live_policy_manifest.cpp live_policy_ingest.cpp ledger.cpp plan_cache.cpp manifest_writer.cpp sdk_check.cpp v2/s10_inputs.cpp)

target_link_libraries(bc_serving PUBLIC bc_jcs bc_serving_v2 bc_owner_map bc_serving_v2_runtime_receipt)
target_compile_features(bc_serving PUBLIC cxx_std_20)
target_include_directories(bc_serving PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SRC}/vendor")
target_link_libraries(bc_serving PUBLIC radiance::abi)
target_include_directories(bc_serving PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
                           "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Inactive P1 native live-room publication. This option is linked into
# bc-serve but does not generate or activate a v2 serving manifest.
add_library(bc_serving_v2_p1_live STATIC v2/p1_live_room.cpp v2/error.cpp)
target_link_libraries(bc_serving_v2_p1_live PUBLIC radiance::abi)
target_compile_features(bc_serving_v2_p1_live PUBLIC cxx_std_20)
target_include_directories(bc_serving_v2_p1_live PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SRC}/vendor")

target_link_libraries(bc_serving PUBLIC bc_serving_v2_p1_live)
add_library(bc_serving_v2_p1 STATIC v2/p1_measurement.cpp)
target_link_libraries(bc_serving_v2_p1 PUBLIC bc_serving_v2)
target_include_directories(bc_serving_v2_p1 PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../moe-remote")
target_compile_features(bc_serving_v2_p1 PUBLIC cxx_std_20)
add_library(bc_serving_v2_p1_producer STATIC v2/p1_producer.cpp)
target_link_libraries(bc_serving_v2_p1_producer PUBLIC
  bc_serving bc_serving_v2_p1 bc_serving_v2_p1_live)
target_include_directories(bc_serving_v2_p1_producer PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/../moe-remote")
target_compile_features(bc_serving_v2_p1_producer PUBLIC cxx_std_20)

add_executable(bc-serve main.cpp release_runtime.cpp)
target_link_libraries(bc-serve PRIVATE bc_serving_v2_p1_producer)
set_target_properties(bc-serve PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
install(TARGETS bc-serve RUNTIME DESTINATION radiance_home/bin)

# RSP01: standalone native release verification/installation. The binary is
# a run-side tool; only the build needs a C++ compiler and zlib headers.
# No Python, shell, Git or GPU service is linked into this executable.
find_package(ZLIB REQUIRED)
add_library(bc_release_core STATIC release_install.cpp release_archive.cpp sha256.cpp)
# bc-serve and bc-release share the same receipt/digest verifier; no second
# runtime implementation or Python/batch wrapper is involved.
target_link_libraries(bc-serve PRIVATE bc_release_core)
target_compile_features(bc_release_core PUBLIC cxx_std_20)
target_link_libraries(bc_release_core PUBLIC ZLIB::ZLIB)
target_include_directories(bc_release_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SRC}/vendor")
add_executable(bc-release release_cli.cpp)
target_link_libraries(bc-release PRIVATE bc_release_core)
set_target_properties(bc-release PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
install(TARGETS bc-release RUNTIME DESTINATION radiance_home/bin)

include(CTest)
if(BUILD_TESTING)
  add_executable(bc_release_install_test EXCLUDE_FROM_ALL release_install_test.cpp)
  target_link_libraries(bc_release_install_test PRIVATE bc_release_core)
  target_compile_definitions(bc_release_install_test PRIVATE
    BC_SERVE_EXE="$<TARGET_FILE:bc-serve>"
    BC_RELEASE_EXE="$<TARGET_FILE:bc-release>"
    BC_FAKE_ENGINE="$<TARGET_FILE:bc_fake_engine>"
    BC_SESSION_STORED_TEST="$<TARGET_FILE:bc_session_stored_test>")
  target_compile_definitions(bc_release_install_test PRIVATE
    BC_RELEASE_TEST_ROOT="${CMAKE_CURRENT_BINARY_DIR}"
    BC_RELEASE_GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/release_testdata")
  add_test(NAME bc_release_install_test COMMAND bc_release_install_test)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  # A build target so the component test gate can declare it; building it runs the tripwire.
  add_custom_target(bc_callback_boundary_guard
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/callback_guard.py"
    VERBATIM)
  add_test(NAME bc_callback_boundary_guard
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/callback_guard.py")
  # The shared header must compile at the HIP remote-MoE worker standard.
  add_executable(bc_callback_boundary_cpp17_test EXCLUDE_FROM_ALL callback_boundary_cpp17_test.cpp)
  set_target_properties(bc_callback_boundary_cpp17_test PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF)
  add_test(NAME bc_callback_boundary_cpp17_test COMMAND bc_callback_boundary_cpp17_test)
  add_executable(bc_callback_boundary_abi_test EXCLUDE_FROM_ALL callback_boundary_abi_test.cpp)
  target_link_libraries(bc_callback_boundary_abi_test PRIVATE ${CMAKE_DL_LIBS})
  target_link_options(bc_callback_boundary_abi_test PRIVATE -rdynamic)
  add_test(NAME bc_callback_boundary_abi_test COMMAND bc_callback_boundary_abi_test)
  add_executable(bc_live_policy_test EXCLUDE_FROM_ALL live_policy_test.cpp)
  target_link_libraries(bc_live_policy_test PRIVATE bc_serving)
  add_test(NAME bc_live_policy_test COMMAND bc_live_policy_test)
  add_executable(bc_live_policy_ingest_test EXCLUDE_FROM_ALL live_policy_ingest_test.cpp)
  target_link_libraries(bc_live_policy_ingest_test PRIVATE bc_serving)
  add_test(NAME bc_live_policy_ingest_test COMMAND bc_live_policy_ingest_test)

  # RSC02: a card-free stand-in for the engine (/health, /v1/completions, --plan-memory), for the session owner's tests.
  add_executable(bc_fake_engine EXCLUDE_FROM_ALL session_fake_engine.cpp)
  target_include_directories(bc_fake_engine PRIVATE "${RADIANCE_SRC}/vendor")
  target_compile_features(bc_fake_engine PRIVATE cxx_std_20)
  add_dependencies(bc_release_install_test bc-serve bc-release bc_fake_engine)
  add_executable(bc_serving_test EXCLUDE_FROM_ALL serving_test.cpp)
  target_link_libraries(bc_serving_test PRIVATE bc_serving bc_serving_v2_p1_producer)
  add_dependencies(bc_serving_test bc_fake_engine bc-serve)
  target_compile_definitions(bc_serving_test PRIVATE BC_FAKE_ENGINE="$<TARGET_FILE:bc_fake_engine>")
  target_compile_definitions(bc_serving_test PRIVATE BC_SERVE_EXE="$<TARGET_FILE:bc-serve>")
  add_test(NAME bc_serving_test COMMAND bc_serving_test)
  # RSC04: real named resolver and strict-v2 writer with only fake OS hardware facts.
  add_executable(bc_typed_settings_test EXCLUDE_FROM_ALL typed_settings_test.cpp)
  target_link_libraries(bc_typed_settings_test PRIVATE bc_serving)
  add_dependencies(bc_typed_settings_test bc-serve)
  target_compile_definitions(bc_typed_settings_test PRIVATE
    BC_SERVE_EXE=\"$<TARGET_FILE:bc-serve>\"
    BC_CONFIGS_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/../../configs\")
  add_test(NAME bc_typed_settings_test COMMAND bc_typed_settings_test)
  set_tests_properties(bc_typed_settings_test PROPERTIES TIMEOUT 60)
  add_executable(bc_power_transaction_test EXCLUDE_FROM_ALL power_transaction_acceptance_test.cpp)
  target_link_libraries(bc_power_transaction_test PRIVATE bc_serving)
  add_test(NAME bc_power_transaction_test COMMAND bc_power_transaction_test)
  # Step 3: independently built card-free guards and real v2 power bridge.
  add_executable(bc_session_guards_test EXCLUDE_FROM_ALL session_guards_test.cpp)
  target_link_libraries(bc_session_guards_test PRIVATE bc_serving)
  target_compile_definitions(bc_session_guards_test PRIVATE
    BC_RSC02_CALIBRATION_GOLDEN="${CMAKE_CURRENT_SOURCE_DIR}/golden/rsc02_calibration_deadlines.json")
  # The declared build target runs all four heat producer contract cases by
  # default; do not register extra CTest aliases without real build targets.
  add_test(NAME bc_session_guards_test COMMAND bc_session_guards_test)
  add_executable(bc_session_stored_test EXCLUDE_FROM_ALL session_stored_test.cpp)
  target_link_libraries(bc_session_stored_test PRIVATE bc_serving)
  add_dependencies(bc_session_stored_test bc_fake_engine)
  target_compile_definitions(bc_session_stored_test PRIVATE
    BC_FAKE_ENGINE="$<TARGET_FILE:bc_fake_engine>"
    BC_CONTRACT_V2_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME bc_session_stored_test COMMAND bc_session_stored_test)
  add_dependencies(bc_release_install_test bc_session_stored_test)
  # RSC02 S15: a real utility target for the no-card, native-and-legacy
  # fake-engine rehearsal. It depends on actual component binaries.
  add_custom_target(bc_s15_dry_run_test
    DEPENDS bc_session_stored_test bc_fake_engine)
  add_test(NAME bc_s15_dry_run_test
    COMMAND ${Python3_EXECUTABLE}
      "${CMAKE_CURRENT_SOURCE_DIR}/../../../../tools/lab/radiance/rsc02_s15_ab.py"
      dry-run --native-test "$<TARGET_FILE:bc_session_stored_test>")

  add_executable(bc_stored_manifest_cache_test EXCLUDE_FROM_ALL stored_manifest_cache_test.cpp)
  target_link_libraries(bc_stored_manifest_cache_test PRIVATE bc_serving)
  add_test(NAME bc_stored_manifest_cache_test COMMAND bc_stored_manifest_cache_test)
  add_executable(bc_stored_manifest_test EXCLUDE_FROM_ALL stored_manifest_test.cpp)
  target_link_libraries(bc_stored_manifest_test PRIVATE bc_serving)
  target_compile_definitions(bc_stored_manifest_test PRIVATE
    BC_CONTRACT_V2_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME bc_stored_manifest_test COMMAND bc_stored_manifest_test)
  add_executable(bc_manifest_power_test EXCLUDE_FROM_ALL manifest_power_test.cpp)
  target_link_libraries(bc_manifest_power_test PRIVATE bc_serving)
  target_compile_definitions(bc_manifest_power_test PRIVATE
    BC_CONTRACT_V2_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME bc_manifest_power_test COMMAND bc_manifest_power_test)
  add_executable(bc_session_test EXCLUDE_FROM_ALL session_test.cpp)
  target_link_libraries(bc_session_test PRIVATE bc_serving)
  add_dependencies(bc_session_test bc_fake_engine)
  target_compile_definitions(bc_session_test PRIVATE BC_FAKE_ENGINE="$<TARGET_FILE:bc_fake_engine>")
  add_test(NAME bc_session_test COMMAND bc_session_test)
  # A build target so the component test gate can declare it; building it runs the tripwire.
  add_custom_target(bc_session_callback_guard
    COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/callback_boundary_guard.cmake"
    VERBATIM)
  add_test(NAME bc_session_callback_guard COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/callback_boundary_guard.cmake")
  # RSC02 step 2: Python-oracle fixture comparison is compiled and exercised by the C++ component suite.
  add_executable(bc_resources_parity_test EXCLUDE_FROM_ALL resources_parity_test.cpp)
  target_link_libraries(bc_resources_parity_test PRIVATE bc_serving)
  target_compile_definitions(bc_resources_parity_test PRIVATE
    BC_RSC02_PARITY_FIXTURE="${CMAKE_CURRENT_SOURCE_DIR}/golden/rsc02_resource_parity.json")
  add_test(NAME bc_resources_parity_test COMMAND bc_resources_parity_test)
  add_executable(bc_jcs_test EXCLUDE_FROM_ALL jcs_test.cpp)
  target_link_libraries(bc_jcs_test PRIVATE bc_serving)
  target_compile_definitions(bc_jcs_test PRIVATE
    BC_CONTRACT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME bc_jcs_test COMMAND bc_jcs_test)
  add_executable(bc_experts_test EXCLUDE_FROM_ALL experts_test.cpp)
  target_link_libraries(bc_experts_test PRIVATE bc_experts bc_jcs)
  add_test(NAME bc_experts_test COMMAND bc_experts_test "${CMAKE_CURRENT_SOURCE_DIR}/golden/owned_golden.txt")
  add_executable(bc_contract_test EXCLUDE_FROM_ALL contract/registry_test.cpp)
  target_link_libraries(bc_contract_test PRIVATE bc_contract)
  target_compile_definitions(bc_contract_test PRIVATE BC_CONTRACT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME bc_contract_test COMMAND bc_contract_test)
  add_executable(bc_contract_check EXCLUDE_FROM_ALL contract/contract_check.cpp)
  target_link_libraries(bc_contract_check PRIVATE bc_contract)
  target_compile_definitions(bc_contract_check PRIVATE BC_CONTRACT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BC_CONTRACT_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2"
    BC_CONTRACT_RADIANCE_SRC="${RADIANCE_SRC}")
  add_test(NAME bc_contract_check COMMAND bc_contract_check)
  # S9: compile and test the inactive generated v2 codec, never production linked.
  # S8: inactive typed resource and handoff rules, linked only by card-free tests.
  add_executable(bc_v2_s8_test EXCLUDE_FROM_ALL v2/s8_rules_test.cpp)
  target_link_libraries(bc_v2_s8_test PRIVATE bc_serving_v2)
  add_test(NAME bc_v2_s8_test COMMAND bc_v2_s8_test)

  # S10: all-v2 inactive resolver and writer; never linked by production.
  add_executable(bc_v2_s10_test EXCLUDE_FROM_ALL v2/s10_resolver_test.cpp)
  target_link_libraries(bc_v2_s10_test PRIVATE bc_serving_v2)
  target_compile_definitions(bc_v2_s10_test PRIVATE
    BC_CONTRACT_V2_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME bc_v2_s10_test COMMAND bc_v2_s10_test)

  # P1: inactive native measurement publication (no v1 activation).
  add_executable(bc_v2_p1_test EXCLUDE_FROM_ALL v2/p1_measurement_test.cpp)
  target_link_libraries(bc_v2_p1_test PRIVATE bc_serving_v2_p1)
  add_test(NAME bc_v2_p1_test COMMAND bc_v2_p1_test)
  add_executable(bc_v2_p1_producer_test EXCLUDE_FROM_ALL v2/p1_producer_test.cpp)
  target_link_libraries(bc_v2_p1_producer_test PRIVATE bc_serving_v2_p1_producer)
  add_test(NAME bc_v2_p1_producer_test COMMAND bc_v2_p1_producer_test)
  # S13a: inactive live host/cgroup/GTT publisher, no production linkage.
  add_executable(bc_v2_p1_live_test EXCLUDE_FROM_ALL v2/p1_live_room_test.cpp)
  target_link_libraries(bc_v2_p1_live_test PRIVATE bc_serving_v2_p1_live)
  add_test(NAME bc_v2_p1_live_test COMMAND bc_v2_p1_live_test)

  # P1 native measurement: inactive card-free native publication.
  add_library(bc_serving_v2_native_measurement STATIC EXCLUDE_FROM_ALL v2/native_measurement.cpp)
  target_link_libraries(bc_serving_v2_native_measurement PUBLIC bc_serving_v2)
  target_include_directories(bc_serving_v2_native_measurement PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../moe-remote"
    "${RADIANCE_SRC}/vendor")
  target_compile_features(bc_serving_v2_native_measurement PUBLIC cxx_std_20)
  add_executable(bc_v2_native_measurement_test EXCLUDE_FROM_ALL v2/native_measurement_test.cpp)
  target_link_libraries(bc_v2_native_measurement_test PRIVATE bc_serving_v2_native_measurement)
  add_test(NAME bc_v2_native_measurement_test COMMAND bc_v2_native_measurement_test)
  # S13b: the same typed read-only publisher is tested independently.
  # Explicit v2-only collector reads the actual worker shared-memory transport.
  # Not installed, not built by default and never linked into v1 production.
  add_executable(bc_v2_runtime_receipt_live EXCLUDE_FROM_ALL v2/runtime_receipt_live.cpp)
  target_link_libraries(bc_v2_runtime_receipt_live PRIVATE
    bc_serving_v2_runtime_receipt bc_jcs)
  add_executable(bc_v2_runtime_receipt_test EXCLUDE_FROM_ALL v2/runtime_receipt_test.cpp)
  target_link_libraries(bc_v2_runtime_receipt_test PRIVATE bc_serving_v2_runtime_receipt)
  add_test(NAME bc_v2_runtime_receipt_test COMMAND bc_v2_runtime_receipt_test)
  add_custom_target(bc_v2_runtime_receipt_check
    COMMAND $<TARGET_FILE:bc_v2_runtime_receipt_test>
    DEPENDS bc_v2_runtime_receipt_test)
  add_executable(bc_v2_runtime_receipt_live_test EXCLUDE_FROM_ALL
    v2/runtime_receipt_live_test.cpp)
  target_link_libraries(bc_v2_runtime_receipt_live_test PRIVATE bc_serving_v2_runtime_receipt bc_jcs)
  add_dependencies(bc_v2_runtime_receipt_live_test bc_v2_runtime_receipt_live)
  target_compile_definitions(bc_v2_runtime_receipt_live_test PRIVATE
    BC_CONTRACT_V2_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME bc_v2_runtime_receipt_live_test
    COMMAND bc_v2_runtime_receipt_live_test $<TARGET_FILE:bc_v2_runtime_receipt_live>)
  add_custom_target(bc_v2_runtime_receipt_live_check
    COMMAND $<TARGET_FILE:bc_v2_runtime_receipt_live_test> $<TARGET_FILE:bc_v2_runtime_receipt_live>
    DEPENDS bc_v2_runtime_receipt_live_test bc_v2_runtime_receipt_live)

  add_executable(bc_v2_codec_test EXCLUDE_FROM_ALL contract/v2_codec_test.cpp)
  target_link_libraries(bc_v2_codec_test PRIVATE bc_serving_v2)
  target_compile_definitions(bc_v2_codec_test PRIVATE
    BC_CONTRACT_V2_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME bc_v2_codec_test COMMAND bc_v2_codec_test)

  add_executable(bc_v2_reader_test EXCLUDE_FROM_ALL v2/reader_test.cpp)
  target_link_libraries(bc_v2_reader_test PRIVATE bc_serving_v2)
  target_compile_definitions(bc_v2_reader_test PRIVATE
    BC_CONTRACT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME bc_v2_reader_test COMMAND bc_v2_reader_test)
  add_executable(bc_s5_refusal_test EXCLUDE_FROM_ALL v2/s5_refusal_test.cpp)
  target_link_libraries(bc_s5_refusal_test PRIVATE bc_serving_v2)
  add_test(NAME bc_s5_refusal_test COMMAND bc_s5_refusal_test)
  add_executable(bc_owner_map_test EXCLUDE_FROM_ALL owner_map_test.cpp)
  target_link_libraries(bc_owner_map_test PRIVATE bc_owner_map bc_jcs)
  target_compile_definitions(bc_owner_map_test PRIVATE
    BC_CONTRACT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v2")
  add_test(NAME bc_owner_map_test COMMAND bc_owner_map_test)
  # Radiance's plugin module defaults CMAKE_BUILD_TYPE to RelWithDebInfo, which adds -DNDEBUG and turns every assert()
  # in these tests into a no-op. Test executables keep their assertions; the assert-based tests refuse to build without.
  get_property(_bc_tests DIRECTORY PROPERTY TESTS)
  foreach(_bc_test IN LISTS _bc_tests)
    if(TARGET ${_bc_test})
      get_target_property(_bc_test_type ${_bc_test} TYPE)
      # Script guards are utility targets: nothing to compile.
      if(NOT _bc_test_type STREQUAL "UTILITY")
        target_compile_options(${_bc_test} PRIVATE -UNDEBUG)
      endif()
    endif()
    # The tests resolve the repository's example layouts; the binaries have no built-in layouts (BIGCHERRY_LAYOUTS).
    set_property(TEST ${_bc_test} APPEND PROPERTY ENVIRONMENT
                 "BIGCHERRY_LAYOUTS=${CMAKE_CURRENT_SOURCE_DIR}/../../layouts")
  endforeach()
  # All declared component tests fail fast on a hung process in CI.
  get_property(bc_component_ctests DIRECTORY PROPERTY TESTS)
  set_tests_properties(${bc_component_ctests} PROPERTIES TIMEOUT 120)
  # S15 composite owner/Bash tests may contend with independent CI CTests.
  set_tests_properties(bc_s15_dry_run_test PROPERTIES TIMEOUT 300 RESOURCE_LOCK rsc02_fake_owner)
  set_tests_properties(bc_session_stored_test PROPERTIES RESOURCE_LOCK rsc02_fake_owner)
endif()
