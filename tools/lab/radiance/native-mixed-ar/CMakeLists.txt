cmake_minimum_required(VERSION 3.21)
project(radiance_mixed_ar_gate LANGUAGES CXX)
include(CTest)
find_package(hip CONFIG REQUIRED)

# One actual Radiance AQL backend and one architecture-matched HIP-enabled core.
# Never link the earlier HIP-off core, compile another shared slot, or use a
# fallback launcher. This is a lab-only opt-in executable (RUM13).
if(NOT RADIANCE_SOURCE_ROOT OR NOT RUM13_NATIVE_CORE_LIB)
  message(FATAL_ERROR "Pass RADIANCE_SOURCE_ROOT and RUM13_NATIVE_CORE_LIB")
endif()
if(NOT EXISTS "${RADIANCE_SOURCE_ROOT}/core/plugin/bc_shared_slot.cpp")
  message(FATAL_ERROR "Pinned Radiance source lacks native cross-DSO shared-slot implementation")
endif()
if(NOT EXISTS "${RUM13_NATIVE_CORE_LIB}")
    message(FATAL_ERROR "RUM13 HIP-on Radiance core archive missing")
endif()
get_filename_component(_engine_core_dir "${RUM13_NATIVE_CORE_LIB}" DIRECTORY)
get_filename_component(_engine_root "${_engine_core_dir}" DIRECTORY)
if(NOT EXISTS "${_engine_root}/CMakeCache.txt")
    message(FATAL_ERROR "RUM13 Radiance engine build receipt absent")
endif()
file(STRINGS "${_engine_root}/CMakeCache.txt" _engine_version
     REGEX "^CMAKE_PROJECT_VERSION:STATIC=")
file(STRINGS "${_engine_root}/CMakeCache.txt" _engine_source
     REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
file(STRINGS "${_engine_root}/CMakeCache.txt" _engine_hip
     REGEX "^RAD_WITH_HIP:STRING=")
if(NOT _engine_version STREQUAL "CMAKE_PROJECT_VERSION:STATIC=2.2.0" OR
   NOT _engine_source STREQUAL "CMAKE_HOME_DIRECTORY:INTERNAL=${RADIANCE_SOURCE_ROOT}" OR
   NOT _engine_hip STREQUAL "RAD_WITH_HIP:STRING=ON")
    message(FATAL_ERROR "RUM13 requires the exact pinned HIP-enabled Radiance 2.2 core")
endif()
add_executable(radiance_mixed_ar_gate mixed_ar_test.cpp)
target_compile_features(radiance_mixed_ar_gate PRIVATE cxx_std_23)
target_compile_definitions(radiance_mixed_ar_gate PRIVATE
    __HIP_PLATFORM_AMD__=1)
target_include_directories(radiance_mixed_ar_gate PRIVATE
    "${RADIANCE_SOURCE_ROOT}/abi" "${RADIANCE_SOURCE_ROOT}/core"
    "${RADIANCE_SOURCE_ROOT}/core/plugin" "${RADIANCE_SOURCE_ROOT}/vendor")
# Match the explicitly resolved HIP package SDK, not the system ROCm path.
get_filename_component(_radiance_sdk_lib_dir "${hip_DIR}/../.." REALPATH)
find_library(RUM13_HSA_RUNTIME NAMES hsa-runtime64
    PATHS "${_radiance_sdk_lib_dir}" NO_DEFAULT_PATH REQUIRED)
set_property(TARGET radiance_mixed_ar_gate APPEND PROPERTY
    BUILD_RPATH "${_radiance_sdk_lib_dir}")
# Use the same upstream Radiance FFmpeg dependency resolution, rather than
# a host-specific hard-coded shared-library path or alternate codec stack.
find_package(PkgConfig REQUIRED)
pkg_check_modules(RUM13_FFMPEG REQUIRED IMPORTED_TARGET
    libavformat libavcodec libavutil libswscale libswresample)
target_link_options(radiance_mixed_ar_gate PRIVATE -Wl,--export-dynamic)
target_link_libraries(radiance_mixed_ar_gate PRIVATE
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,${RUM13_NATIVE_CORE_LIB}>"
    hip::host ${RUM13_HSA_RUNTIME}
    "${_engine_core_dir}/libbc_serving_v2.a"
    "${_engine_core_dir}/libbc_jcs.a"
    "${_engine_root}/vendor/librad_vendor_llama.a"
    "${_engine_core_dir}/librad_json.a"
    "${_engine_core_dir}/librad_simd.a"
    "${_engine_root}/vendor/librad_vendor_yyjson.a"
    "${_engine_root}/vendor/librad_vendor_httplib.a"
    "${_engine_root}/vendor/librad_vendor_ryml.a"
    PkgConfig::RUM13_FFMPEG
    dl pthread)

# The duplicate-ordinal negative must fail before loading plugins or devices.
add_test(NAME rum13_reject_duplicate_devices
  COMMAND radiance_mixed_ar_gate missing-r3.so missing-r4d.so
          ar_oneshot_3rank_exact_host 0,0,0)
set_tests_properties(rum13_reject_duplicate_devices PROPERTIES
  WILL_FAIL TRUE)
