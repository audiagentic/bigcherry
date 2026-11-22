cmake_minimum_required(VERSION 3.21)
project(ltx2_c01_native_syntax LANGUAGES CXX)
include(CTest)

# Compile actual frozen 2.2 LTX translation unit after a diagnostic patch was
# applied to a separate work-root stage. Never mutate shared upstream sources.
if(NOT LTX_SOURCE_ROOT OR NOT LTX_STAGE_ROOT)
  message(FATAL_ERROR "Pass -DLTX_SOURCE_ROOT=... and -DLTX_STAGE_ROOT=...")
endif()
if(NOT EXISTS "${LTX_STAGE_ROOT}/arch/ltx2/bc_ltx_tp3_c01.h")
  message(FATAL_ERROR "Stage lacks the C01 header; source patch has not been applied")
endif()
if(NOT EXISTS "${LTX_STAGE_ROOT}/arch/ltx2/ltx2.cpp")
  message(FATAL_ERROR "Stage lacks the patched LTX2 translation unit")
endif()
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/rad_include")
if(NOT EXISTS "${CMAKE_BINARY_DIR}/rad_include/arch")
  file(CREATE_LINK "${LTX_SOURCE_ROOT}/arch/common"
       "${CMAKE_BINARY_DIR}/rad_include/arch" SYMBOLIC)
endif()

add_library(ltx2_c01_native_compile OBJECT "${LTX_STAGE_ROOT}/arch/ltx2/ltx2.cpp")
target_include_directories(ltx2_c01_native_compile PRIVATE
  "${CMAKE_BINARY_DIR}/rad_include"
  "${LTX_SOURCE_ROOT}/abi")
target_compile_features(ltx2_c01_native_compile PRIVATE cxx_std_23)


# Optional ABI-matched native Builder receipt runner. It reads a real RAD2
# model and records code, scale and BF16 bias spans through RadBuilder itself.
# Link only to an existing ABI44/2.2 core archive, never a mixed-SDK fallback.
if(LTX_NATIVE_CORE_LIB)
  if(NOT EXISTS "${LTX_NATIVE_CORE_LIB}")
    message(FATAL_ERROR "LTX_NATIVE_CORE_LIB is not a built Radiance core archive")
  endif()
  add_executable(ltx_c01_weight_receipt_test ltx_c01_weight_receipt_test.cpp)
  target_compile_features(ltx_c01_weight_receipt_test PRIVATE cxx_std_20)
  target_include_directories(ltx_c01_weight_receipt_test PRIVATE
    "${LTX_SOURCE_ROOT}/core" "${LTX_SOURCE_ROOT}/abi"
    "${LTX_SOURCE_ROOT}/vendor"
    "${LTX_STAGE_ROOT}/arch/ltx2"
    "${CMAKE_BINARY_DIR}/rad_include")
  get_filename_component(_native_core_dir "${LTX_NATIVE_CORE_LIB}" DIRECTORY)
  get_filename_component(_native_build_root "${_native_core_dir}" DIRECTORY)
  if(NOT EXISTS "${_native_build_root}/CMakeCache.txt")
    message(FATAL_ERROR "Native core archive lacks source/version build receipt")
  endif()
  file(STRINGS "${_native_build_root}/CMakeCache.txt" _native_version
    REGEX "^CMAKE_PROJECT_VERSION:STATIC=")
  file(STRINGS "${_native_build_root}/CMakeCache.txt" _native_source
    REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
  if(NOT _native_version STREQUAL "CMAKE_PROJECT_VERSION:STATIC=2.2.0"
     OR NOT _native_source STREQUAL "CMAKE_HOME_DIRECTORY:INTERNAL=${LTX_SOURCE_ROOT}")
    message(FATAL_ERROR "C01 cannot link a different Radiance version or source build")
  endif()
  # Match the 2.2 CMake build_test dependency order; do not use ABI-mismatched
  # global installations or silently fall back to another Radiance archive.
  target_link_libraries(ltx_c01_weight_receipt_test PRIVATE
    "${LTX_NATIVE_CORE_LIB}" dl
    "${_native_core_dir}/libbc_serving_v2.a"
    "${_native_core_dir}/libbc_jcs.a"
    "${_native_build_root}/vendor/librad_vendor_llama.a"
    "${_native_core_dir}/librad_json.a"
    "${_native_core_dir}/librad_simd.a"
    "${_native_build_root}/vendor/librad_vendor_yyjson.a"
    "${_native_build_root}/vendor/librad_vendor_httplib.a"
    "${_native_build_root}/vendor/librad_vendor_ryml.a")
  if(LTX_IQ8R_MODEL)
    add_test(NAME ltx_c01_weight_receipt
      COMMAND ltx_c01_weight_receipt_test "${LTX_IQ8R_MODEL}")
  endif()
  # C02 uses the real patched Lin::declare with one ABI-matched reference
  # kernel. Stage only libref.so: scanning the full SDK directory can load
  # unrelated diagnostic helpers (and must not initialize a GPU).
  if(LTX_REFERENCE_LIB)
    if(NOT LTX_IQ8R_MODEL)
      message(FATAL_ERROR "LTX_REFERENCE_LIB needs LTX_IQ8R_MODEL")
    endif()
    if(NOT EXISTS "${LTX_REFERENCE_LIB}")
      message(FATAL_ERROR "LTX_REFERENCE_LIB is missing")
    endif()
    file(REAL_PATH "${LTX_REFERENCE_LIB}" _ref_source)
    set(_ref_dir "${CMAKE_CURRENT_BINARY_DIR}/ltx_c02_ref")
    file(MAKE_DIRECTORY "${_ref_dir}")
    set(_ref_link "${_ref_dir}/libref.so")
    if(EXISTS "${_ref_link}")
      file(REAL_PATH "${_ref_link}" _ref_existing)
      if(NOT _ref_existing STREQUAL _ref_source)
        message(FATAL_ERROR "C02 reference kernel changed; use a fresh build directory")
      endif()
    else()
      file(CREATE_LINK "${_ref_source}" "${_ref_link}" SYMBOLIC RESULT _ref_result)
      if(NOT _ref_result STREQUAL "0")
        message(FATAL_ERROR "C02 cannot stage isolated native kernel: ${_ref_result}")
      endif()
    endif()
    add_executable(ltx_c02_lin_declare_test ltx_c02_lin_declare_test.cpp)
    target_compile_features(ltx_c02_lin_declare_test PRIVATE cxx_std_23)
    target_include_directories(ltx_c02_lin_declare_test PRIVATE
      "${LTX_SOURCE_ROOT}/core" "${LTX_SOURCE_ROOT}/abi"
      "${LTX_SOURCE_ROOT}/vendor"
      "${LTX_STAGE_ROOT}/arch/ltx2"
      "${CMAKE_BINARY_DIR}/rad_include")
    get_target_property(_c01_core_links ltx_c01_weight_receipt_test LINK_LIBRARIES)
    target_link_libraries(ltx_c02_lin_declare_test PRIVATE ${_c01_core_links})
    add_test(NAME ltx_c02_actual_lin_declare
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}")
    add_test(NAME ltx_c04_deferred_output_reduce
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --c04)
    add_test(NAME ltx_c05_global_qk_rms_graph
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --c05)
    add_test(NAME ltx_c06_real_self_attention_graph
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --c06)

    # LTX02's first actually-issued native RadCtx gate: reuse the real
    # patched arch source, host backend, model and reference registry.
    add_executable(ltx_c05_issued_host_test ltx_c05_issued_host_test.cpp)
    target_compile_features(ltx_c05_issued_host_test PRIVATE cxx_std_23)
    target_include_directories(ltx_c05_issued_host_test PRIVATE
      "${LTX_SOURCE_ROOT}/core" "${LTX_SOURCE_ROOT}/abi"
      "${LTX_SOURCE_ROOT}/vendor" "${LTX_STAGE_ROOT}/arch/ltx2"
      "${CMAKE_BINARY_DIR}/rad_include")
    target_link_libraries(ltx_c05_issued_host_test PRIVATE ${_c01_core_links})
    add_test(NAME ltx_c05_issued_host_global_qk
      COMMAND ltx_c05_issued_host_test "${LTX_IQ8R_MODEL}" "${_ref_dir}")
    set_tests_properties(ltx_c05_issued_host_global_qk PROPERTIES TIMEOUT 25)
  endif()
endif()
