cmake_minimum_required(VERSION 3.21)
project(bc_serving_component LANGUAGES CXX)
find_package(radiance REQUIRED)
rad_plugin_defaults()

set(RADIANCE_SRC "" CACHE PATH "Radiance source tree, for the engine's vendored nlohmann/json")
if(NOT EXISTS "${RADIANCE_SRC}/vendor/nlohmann/json.hpp")
  message(FATAL_ERROR "serving: -DRADIANCE_SRC must contain vendor/nlohmann/json.hpp")
endif()

# RFC 8785 JCS and SHA-256: shared by bc_serving and the contract generator.
add_library(bc_jcs STATIC jcs.cpp sha256.cpp)
target_compile_features(bc_jcs PUBLIC cxx_std_20)
target_include_directories(bc_jcs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SRC}/vendor")

# The serving manifest v1 contract (contract/README.md): the field registry contract/serving-v1.toml and its generator.
# GPU-free and independent of bc_serving, whose typed model it generates (generated/, committed; bc_contract_check
# fails CI when a committed file differs from the generator's output).
add_library(bc_contract STATIC contract/registry.cpp)
target_link_libraries(bc_contract PUBLIC bc_jcs)
target_include_directories(bc_contract PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/contract")
add_executable(bc_contract_gen contract/generate.cpp)
target_link_libraries(bc_contract_gen PRIVATE bc_contract)

# The shared strict v1 reader (RSC08 slice 2): structured errors, strict JSON, the generated codec, semantics, typed
# views and the admission check. Linked by bc-serve, radiance and r3_moe_worker; GPU-free.
add_library(bc_serving_v1 STATIC v1/error.cpp v1/strict_json.cpp v1/manifest.cpp v1/admission.cpp
            generated/serving_codec.cpp)
target_link_libraries(bc_serving_v1 PUBLIC bc_jcs radiance::abi)
target_include_directories(bc_serving_v1 PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/v1" "${CMAKE_CURRENT_SOURCE_DIR}/generated"
                           "${CMAKE_CURRENT_SOURCE_DIR}")

# The helpers' expert ownership, shared with moe-remote's r3_moe_worker (bc_experts.cmake).
include("${CMAKE_CURRENT_SOURCE_DIR}/bc_experts.cmake")

# calibration.toml is compiled in: the seeds bc-serve derives with, each carrying its evidence.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/calibration.toml" bc_calibration_seed)
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/calibration_seed.inc"
     "R\"bccal(${bc_calibration_seed})bccal\"\n")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/calibration.toml")

add_library(bc_serving STATIC serving.cpp calibration.cpp ledger.cpp plan_cache.cpp)
target_link_libraries(bc_serving PUBLIC bc_jcs bc_serving_v1)
target_compile_features(bc_serving PUBLIC cxx_std_20)
target_include_directories(bc_serving PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${RADIANCE_SRC}/vendor")
target_link_libraries(bc_serving PUBLIC radiance::abi)
target_include_directories(bc_serving PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/third_party"
                           "${CMAKE_CURRENT_BINARY_DIR}/generated")

add_executable(bc-serve main.cpp)
target_link_libraries(bc-serve PRIVATE bc_serving)
set_target_properties(bc-serve PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")
install(TARGETS bc-serve RUNTIME DESTINATION radiance_home/bin)

include(CTest)
if(BUILD_TESTING)
  add_executable(bc_serving_test EXCLUDE_FROM_ALL serving_test.cpp)
  target_link_libraries(bc_serving_test PRIVATE bc_serving)
  add_test(NAME bc_serving_test COMMAND bc_serving_test)
  add_executable(bc_jcs_test EXCLUDE_FROM_ALL jcs_test.cpp)
  target_link_libraries(bc_jcs_test PRIVATE bc_serving)
  target_compile_definitions(bc_jcs_test PRIVATE
    BC_CONTRACT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v1")
  add_test(NAME bc_jcs_test COMMAND bc_jcs_test)
  add_executable(bc_experts_test EXCLUDE_FROM_ALL experts_test.cpp)
  target_link_libraries(bc_experts_test PRIVATE bc_experts bc_jcs)
  add_test(NAME bc_experts_test COMMAND bc_experts_test)
  add_executable(bc_contract_test EXCLUDE_FROM_ALL contract/registry_test.cpp)
  target_link_libraries(bc_contract_test PRIVATE bc_contract)
  target_compile_definitions(bc_contract_test PRIVATE BC_CONTRACT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
  add_test(NAME bc_contract_test COMMAND bc_contract_test)
  add_executable(bc_contract_check EXCLUDE_FROM_ALL contract/contract_check.cpp)
  target_link_libraries(bc_contract_check PRIVATE bc_contract)
  target_compile_definitions(bc_contract_check PRIVATE BC_CONTRACT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    BC_CONTRACT_DOCS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v1")
  add_test(NAME bc_contract_check COMMAND bc_contract_check)
  add_executable(bc_v1_test EXCLUDE_FROM_ALL v1/reader_test.cpp)
  target_link_libraries(bc_v1_test PRIVATE bc_serving_v1)
  target_compile_definitions(bc_v1_test PRIVATE
    BC_CONTRACT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../../../docs/design/serving-manifest-v1")
  add_test(NAME bc_v1_test COMMAND bc_v1_test)
endif()
