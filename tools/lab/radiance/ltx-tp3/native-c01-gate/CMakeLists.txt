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
    # LTX03 full-depth declaration check: model-derived block count,
    # both self-attention modalities and both legal three-rank head maps.
    # Does not claim executed TP3 or full six-attention coverage.
    add_test(NAME ltx03_all_block_self_attention_declarations
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --ltx03-all-blocks)
    set_tests_properties(ltx03_all_block_self_attention_declarations
      PROPERTIES TIMEOUT 180)
    add_test(NAME ltx03_text_cross_attention_declarations
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --ltx03-text-cross)
    set_tests_properties(ltx03_text_cross_attention_declarations PROPERTIES TIMEOUT 180)
    add_test(NAME ltx03_modal_cross_attention_declarations
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --ltx03-modal-cross)
    set_tests_properties(ltx03_modal_cross_attention_declarations PROPERTIES TIMEOUT 180)
    # Frozen Radiance loader geometry on the real C06 self-attention weights.
    # Identifies relayouts that the next issued-GPU test must not bypass.
    add_test(NAME ltx_c06_native_weight_share_contract
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --c06-weights)
    # Actual native per-rank Builder instance construction. The host
    # collective rendezvous requires concurrent finish on all ranks.
    # No checkpoint weights are loaded or inference issued by this gate.
    add_test(NAME ltx_c06_native_instance_rendezvous
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --c06-instances)
    set_tests_properties(ltx_c06_native_instance_rendezvous PROPERTIES TIMEOUT 90)
    # Opt-in source-backed native RadCtx issue: same graph, real weights
    # and exact collectives. CTest never runs it on the GPU.
    add_test(NAME ltx_c06_issued_host_checkpoint_attention
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --c06-issued-host)
    set_tests_properties(ltx_c06_issued_host_checkpoint_attention PROPERTIES TIMEOUT 120)
    # LTX02: issue the same checkpoint-backed three-rank RadCtx graph
    # with genuinely multirow activations and ordered native collectives.
    foreach(_ltx_rows IN ITEMS 16 64)
      add_test(NAME ltx_c06_issued_host_${_ltx_rows}_rows
        COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}"
                "--c06-issued-host-${_ltx_rows}")
      set_tests_properties(ltx_c06_issued_host_${_ltx_rows}_rows
        PROPERTIES TIMEOUT 120)
    endforeach()
    # Host-linked binary must reject the opt-in native AQL flag, before
    # creating any rank or device instance.
    add_test(NAME ltx_c06_aql_refuses_host_backend
      COMMAND ltx_c02_lin_declare_test "${LTX_IQ8R_MODEL}" "${_ref_dir}" --c06-instances-aql)
    set_tests_properties(ltx_c06_aql_refuses_host_backend
      PROPERTIES WILL_FAIL TRUE TIMEOUT 20)

    # LTX02's first actually-issued native RadCtx gate: reuse the real
    # patched arch source, host backend, model and reference registry.
    add_executable(ltx_c05_issued_host_test ltx_c05_issued_test.cpp)
    target_compile_features(ltx_c05_issued_host_test PRIVATE cxx_std_23)
    target_include_directories(ltx_c05_issued_host_test PRIVATE
      "${LTX_SOURCE_ROOT}/core" "${LTX_SOURCE_ROOT}/abi"
      "${LTX_SOURCE_ROOT}/vendor" "${LTX_STAGE_ROOT}/arch/ltx2"
      "${CMAKE_BINARY_DIR}/rad_include")
    target_link_libraries(ltx_c05_issued_host_test PRIVATE ${_c01_core_links})
    add_test(NAME ltx_c05_issued_host_global_qk
      COMMAND ltx_c05_issued_host_test "${LTX_IQ8R_MODEL}" "${_ref_dir}")
    set_tests_properties(ltx_c05_issued_host_global_qk PROPERTIES TIMEOUT 25)


    # Same LTX02 model C05 graph + numerical oracle, now linked to the
    # real GPU engine core. The GPU run is lease-gated and never an
    # automatic CTest. No direct-HIP launcher and no second graph.
    if(LTX_HIP_NATIVE_CORE_LIB)
      if(NOT EXISTS "${LTX_HIP_NATIVE_CORE_LIB}")
        message(FATAL_ERROR "LTX02 HIP-enabled Radiance core archive not found")
      endif()
      get_filename_component(_hip_core_dir "${LTX_HIP_NATIVE_CORE_LIB}" DIRECTORY)
      get_filename_component(_hip_build_root "${_hip_core_dir}" DIRECTORY)
      if(NOT EXISTS "${_hip_build_root}/CMakeCache.txt")
        message(FATAL_ERROR "LTX02 AQL core has no native build provenance")
      endif()
      file(STRINGS "${_hip_build_root}/CMakeCache.txt" _hip_version
        REGEX "^CMAKE_PROJECT_VERSION:STATIC=")
      file(STRINGS "${_hip_build_root}/CMakeCache.txt" _hip_source
        REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
      file(STRINGS "${_hip_build_root}/CMakeCache.txt" _hip_enabled
        REGEX "^RAD_WITH_HIP:STRING=")
      if(NOT _hip_version STREQUAL "CMAKE_PROJECT_VERSION:STATIC=2.2.0" OR
         NOT _hip_source STREQUAL "CMAKE_HOME_DIRECTORY:INTERNAL=${LTX_SOURCE_ROOT}" OR
         NOT _hip_enabled STREQUAL "RAD_WITH_HIP:STRING=ON")
        message(FATAL_ERROR "LTX02 AQL requires exact HIP-enabled Radiance 2.2 source/archive")
      endif()
      find_package(hip CONFIG REQUIRED)
      get_filename_component(_hip_lib_dir "${hip_DIR}/../.." REALPATH)
      find_library(LTX02_HSA_RUNTIME NAMES hsa-runtime64
        PATHS "${_hip_lib_dir}" NO_DEFAULT_PATH REQUIRED)
      find_package(PkgConfig REQUIRED)
      pkg_check_modules(LTX02_FFMPEG REQUIRED IMPORTED_TARGET
        libavformat libavcodec libavutil libswscale libswresample)
      # Registry scans *all* .so files before applying a kernel hierarchy.
      # Isolate the existing exact SDK rows just as C02 already isolates ref.
      get_filename_component(_ltx02_kernel_dir "${LTX_REFERENCE_LIB}" DIRECTORY)
      set(_ltx02_aql_dir "${CMAKE_CURRENT_BINARY_DIR}/ltx02_aql_kernels")
      file(MAKE_DIRECTORY "${_ltx02_aql_dir}")
      foreach(_ltx02_kernel IN ITEMS libr3 libr4d libref)
        set(_ltx02_source "${_ltx02_kernel_dir}/${_ltx02_kernel}.so")
        set(_ltx02_link "${_ltx02_aql_dir}/${_ltx02_kernel}.so")
        if(NOT EXISTS "${_ltx02_source}")
          message(FATAL_ERROR "LTX02 AQL needs original native SDK kernel ${_ltx02_source}")
        endif()
        if(EXISTS "${_ltx02_link}")
          file(REAL_PATH "${_ltx02_link}" _ltx02_existing)
          file(REAL_PATH "${_ltx02_source}" _ltx02_expected)
          if(NOT _ltx02_existing STREQUAL _ltx02_expected)
            message(FATAL_ERROR "LTX02 AQL staged native kernel changed")
          endif()
        else()
          file(CREATE_LINK "${_ltx02_source}" "${_ltx02_link}" SYMBOLIC
            RESULT _ltx02_link_result)
          if(NOT _ltx02_link_result STREQUAL "0")
            message(FATAL_ERROR "LTX02 AQL native kernel staging refused: ${_ltx02_link_result}")
          endif()
        endif()
      endforeach()
      add_executable(ltx_c05_issued_aql_test ltx_c05_issued_test.cpp)
      target_compile_features(ltx_c05_issued_aql_test PRIVATE cxx_std_23)
      target_compile_definitions(ltx_c05_issued_aql_test PRIVATE __HIP_PLATFORM_AMD__=1)
      target_include_directories(ltx_c05_issued_aql_test PRIVATE
        "${LTX_SOURCE_ROOT}/core" "${LTX_SOURCE_ROOT}/abi"
        "${LTX_SOURCE_ROOT}/vendor" "${LTX_STAGE_ROOT}/arch/ltx2"
        "${CMAKE_BINARY_DIR}/rad_include")
      set_property(TARGET ltx_c05_issued_aql_test APPEND PROPERTY
        BUILD_RPATH "${_hip_lib_dir}")
      target_link_options(ltx_c05_issued_aql_test PRIVATE -Wl,--export-dynamic)
      target_link_libraries(ltx_c05_issued_aql_test PRIVATE
        "$<LINK_LIBRARY:WHOLE_ARCHIVE,${LTX_HIP_NATIVE_CORE_LIB}>"
        hip::host ${LTX02_HSA_RUNTIME}
        "${_hip_core_dir}/libbc_serving_v2.a"
        "${_hip_core_dir}/libbc_jcs.a"
        "${_hip_build_root}/vendor/librad_vendor_llama.a"
        "${_hip_core_dir}/librad_json.a"
        "${_hip_core_dir}/librad_simd.a"
        "${_hip_build_root}/vendor/librad_vendor_yyjson.a"
        "${_hip_build_root}/vendor/librad_vendor_httplib.a"
        "${_hip_build_root}/vendor/librad_vendor_ryml.a"
        PkgConfig::LTX02_FFMPEG
        dl pthread)
      # C06 reuses the exact C05 native HIP/AQL core and kernel registry.
      # There is no second AQL build recipe or launcher. Lease-gated:
      # never auto-run this binary as a CTest.
      add_executable(ltx_c06_instancing_aql_test ltx_c02_lin_declare_test.cpp)
      target_compile_features(ltx_c06_instancing_aql_test PRIVATE cxx_std_23)
      target_compile_definitions(ltx_c06_instancing_aql_test PRIVATE __HIP_PLATFORM_AMD__=1)
      get_target_property(_ltx02_aql_includes ltx_c05_issued_aql_test INCLUDE_DIRECTORIES)
      get_target_property(_ltx02_aql_links ltx_c05_issued_aql_test LINK_LIBRARIES)
      target_include_directories(ltx_c06_instancing_aql_test PRIVATE ${_ltx02_aql_includes})
      target_link_libraries(ltx_c06_instancing_aql_test PRIVATE ${_ltx02_aql_links})
      set_property(TARGET ltx_c06_instancing_aql_test APPEND PROPERTY
        BUILD_RPATH "${_hip_lib_dir}")
      target_link_options(ltx_c06_instancing_aql_test PRIVATE -Wl,--export-dynamic)
    endif()
  endif()
endif()
