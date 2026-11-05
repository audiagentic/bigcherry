# bc_serving_v1 (and bc_jcs): the shared serving manifest v1 reader, GPU-free. ONE source list, included by the serving
# component and by radiance's core (managed patch rad0011 passes this file's path as BC_SERVING_V1_CMAKE), so both
# builds compile the same files.
#
# The including project provides: a radiance::abi or rad_abi target for rad_types.h, and the nlohmann/json include
# directory in BC_SERVING_V1_NLOHMANN_INCLUDE. Nothing here is copied from either tree.
if(NOT TARGET bc_serving_v1)
  set(_bc_v1_dir "${CMAKE_CURRENT_LIST_DIR}")
  if(NOT DEFINED BC_SERVING_V1_NLOHMANN_INCLUDE)
    message(FATAL_ERROR "bc_serving_v1.cmake: set BC_SERVING_V1_NLOHMANN_INCLUDE to the directory holding nlohmann/json.hpp")
  endif()
  if(TARGET radiance::abi)
    set(_bc_v1_abi radiance::abi)
  elseif(TARGET rad_abi)
    set(_bc_v1_abi rad_abi)
  else()
    message(FATAL_ERROR "bc_serving_v1.cmake: needs the radiance::abi (or rad_abi) target for rad_types.h")
  endif()
  # RFC 8785 JCS and SHA-256
  add_library(bc_jcs STATIC "${_bc_v1_dir}/jcs.cpp" "${_bc_v1_dir}/sha256.cpp")
  target_compile_features(bc_jcs PUBLIC cxx_std_20)
  target_include_directories(bc_jcs PUBLIC "${_bc_v1_dir}" "${BC_SERVING_V1_NLOHMANN_INCLUDE}")
  # errors, strict JSON, the generated codec, semantics, typed views and the admission check
  add_library(bc_serving_v1 STATIC
    "${_bc_v1_dir}/v1/error.cpp" "${_bc_v1_dir}/v1/strict_json.cpp" "${_bc_v1_dir}/v1/manifest.cpp"
    "${_bc_v1_dir}/v1/admission.cpp" "${_bc_v1_dir}/v1/helper_settings.cpp" "${_bc_v1_dir}/generated/serving_codec.cpp")
  target_link_libraries(bc_serving_v1 PUBLIC bc_jcs ${_bc_v1_abi})
  target_include_directories(bc_serving_v1 PUBLIC "${_bc_v1_dir}/v1" "${_bc_v1_dir}/generated" "${_bc_v1_dir}")
endif()
