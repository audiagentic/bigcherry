# bc_serving_v2 (and bc_jcs): the shared serving manifest v2 reader, GPU-free. ONE source list, included by the serving
# component, by moe-remote's helper and by radiance's core (managed patch rad0011 passes this file's path as
# BC_SERVING_V2_CMAKE), so every build compiles the same files.
#
# The including project provides a radiance::abi or rad_abi target for rad_types.h.
# The serving component owns the pinned JSON header for all three consumers.
# component.content_digest() binds this exact 3.12.0 header and its recorded SHA-256
# (third_party/nlohmann/README.md) into every component build identity.
function(bc_add_manifest_library target abi generated)
  set(src "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
  # Errors, strict JSON, the generated codec, semantics and role views.
  add_library(${target} STATIC
    "${src}/v2/error.cpp" "${src}/v2/strict_json.cpp" "${src}/v2/manifest.cpp"
    "${src}/v2/native_config_snapshot.cpp" "${src}/v2/admission.cpp" "${src}/v2/helper_settings.cpp" "${src}/v2/engine_admission.cpp" "${src}/bc_serving_managed.cpp"
    "${src}/v2/s8_rules.cpp" "${src}/v2/s10_resolver.cpp" "${generated}/serving_codec.cpp")
  target_link_libraries(${target} PUBLIC bc_jcs ${abi})
  target_include_directories(${target} PUBLIC "${src}/v2" "${generated}" "${src}")
endfunction()

function(bc_add_contract_generator)
  if(TARGET bc_contract_gen)
    return()
  endif()
  set(src "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/contract")
  add_library(bc_contract STATIC "${src}/registry.cpp")
  target_link_libraries(bc_contract PUBLIC bc_jcs)
  target_include_directories(bc_contract PUBLIC "${src}")
  add_executable(bc_contract_gen "${src}/generate.cpp")
  target_link_libraries(bc_contract_gen PRIVATE bc_contract)
endfunction()

function(bc_generate_serving_v3 generated)
  bc_add_contract_generator()
  set(src "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")
  set(registry "${src}/contract/serving.toml")
  set(fingerprint "${src}/generated/v3/contract.fingerprint")
  add_custom_command(
    OUTPUT "${generated}/serving_types.h" "${generated}/serving_codec.h"
           "${generated}/serving_codec.cpp"
    BYPRODUCTS "${generated}/serving-v3.schema.json" "${generated}/serving-fields.md"
               "${generated}/contract.fingerprint"
    COMMAND bc_contract_gen "${registry}" "${generated}"
    COMMAND "${CMAKE_COMMAND}" -E compare_files "${generated}/contract.fingerprint" "${fingerprint}"
    DEPENDS bc_contract_gen "${registry}" "${fingerprint}" VERBATIM)
endfunction()

# The serving manifest wire is edition 3: the v2 edition is gone. The cache variable stays as the edition's recorded identity
# (the release tooling reads it) and accepts only "3".
set(BC_SERVING_CONTRACT_VERSION "3" CACHE STRING "Serving manifest wire edition (only 3)")
if(NOT BC_SERVING_CONTRACT_VERSION STREQUAL "3")
  message(FATAL_ERROR "BC_SERVING_CONTRACT_VERSION must be 3: the serving manifest v2 edition was removed")
endif()
if(DEFINED BC_SERVING_GENERATED_DIR AND NOT BC_SERVING_V3_NATIVE_TEST)
  message(FATAL_ERROR "BC_SERVING_GENERATED_DIR is reserved for the isolated v3 native-consumer test")
endif()
if(BC_SERVING_V3_NATIVE_TEST AND NOT DEFINED BC_SERVING_GENERATED_DIR)
  message(FATAL_ERROR "BC_SERVING_V3_NATIVE_TEST requires BC_SERVING_GENERATED_DIR")
endif()

if(NOT TARGET bc_serving_v2)
  set(_bc_v2_dir "${CMAKE_CURRENT_LIST_DIR}")
  if(TARGET radiance::abi)
    set(_bc_v2_abi radiance::abi)
  elseif(TARGET rad_abi)
    set(_bc_v2_abi rad_abi)
  else()
    message(FATAL_ERROR "bc_serving_v2.cmake: needs the radiance::abi (or rad_abi) target for rad_types.h")
  endif()
  if(NOT TARGET bc_json_dependency)
    add_library(bc_json_dependency INTERFACE)
    target_include_directories(bc_json_dependency SYSTEM INTERFACE "${_bc_v2_dir}/third_party")
  endif()
  # RFC 8785 JCS and SHA-256
  add_library(bc_jcs STATIC "${_bc_v2_dir}/jcs.cpp" "${_bc_v2_dir}/sha256.cpp")
  target_compile_features(bc_jcs PUBLIC cxx_std_20)
  target_include_directories(bc_jcs PUBLIC "${_bc_v2_dir}")
  target_link_libraries(bc_jcs PUBLIC bc_json_dependency)
  if(BC_SERVING_V3_NATIVE_TEST)
    set(_bc_manifest_generated "${BC_SERVING_GENERATED_DIR}")
  else()
    set(_bc_manifest_generated "${CMAKE_CURRENT_BINARY_DIR}/bc-serving-generated/v3")
    bc_generate_serving_v3("${_bc_manifest_generated}")
  endif()
  bc_add_manifest_library(bc_serving_v2 ${_bc_v2_abi} "${_bc_manifest_generated}")
endif()
