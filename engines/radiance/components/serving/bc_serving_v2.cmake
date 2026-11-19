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
    "${src}/v2/native_config_snapshot.cpp" "${src}/v2/admission.cpp" "${src}/v2/helper_settings.cpp"
    "${src}/v2/s8_rules.cpp" "${src}/v2/s10_resolver.cpp" "${generated}/serving_codec.cpp")
  target_link_libraries(${target} PUBLIC bc_jcs ${abi})
  target_include_directories(${target} PUBLIC "${src}/v2" "${generated}" "${src}")
endfunction()

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
  bc_add_manifest_library(bc_serving_v2 ${_bc_v2_abi} "${_bc_v2_dir}/generated/v2")
endif()
