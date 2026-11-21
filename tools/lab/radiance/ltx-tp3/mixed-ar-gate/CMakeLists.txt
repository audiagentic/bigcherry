cmake_minimum_required(VERSION 3.21)
project(ltx25_mixed_ar_gate LANGUAGES CXX)
find_package(hip CONFIG REQUIRED)
if(NOT LTX_SOURCE_ROOT OR NOT LTX_NATIVE_CORE_LIB)
    message(FATAL_ERROR "Supply frozen Radiance 2.2 source and ABI44 native core archive")
endif()
if(NOT EXISTS "${LTX_SOURCE_ROOT}/core/plugin/bc_shared_slot.cpp" OR
   NOT EXISTS "${LTX_NATIVE_CORE_LIB}")
    message(FATAL_ERROR "Missing native shared-slot implementation or core archive")
endif()
get_filename_component(_native_core_dir "${LTX_NATIVE_CORE_LIB}" DIRECTORY)
get_filename_component(_native_build_root "${_native_core_dir}" DIRECTORY)
if(NOT EXISTS "${_native_build_root}/CMakeCache.txt")
    message(FATAL_ERROR "Native core lacks its build/version receipt")
endif()
file(STRINGS "${_native_build_root}/CMakeCache.txt" _version REGEX "^CMAKE_PROJECT_VERSION:STATIC=")
file(STRINGS "${_native_build_root}/CMakeCache.txt" _source REGEX "^CMAKE_HOME_DIRECTORY:INTERNAL=")
if(NOT _version STREQUAL "CMAKE_PROJECT_VERSION:STATIC=2.2.0" OR
   NOT _source STREQUAL "CMAKE_HOME_DIRECTORY:INTERNAL=${LTX_SOURCE_ROOT}")
    message(FATAL_ERROR "LTX mixed test cannot link a different Radiance version/source")
endif()
# Use actual BigCherry shared-slot implementation, not a mock rendezvous.
add_executable(ltx25_mixed_ar_gate
    mixed_ar_test.cpp
    "${LTX_SOURCE_ROOT}/core/plugin/bc_shared_slot.cpp")
target_compile_features(ltx25_mixed_ar_gate PRIVATE cxx_std_23)
target_compile_definitions(ltx25_mixed_ar_gate PRIVATE __HIP_PLATFORM_AMD__=1)
target_include_directories(ltx25_mixed_ar_gate PRIVATE
    "${LTX_SOURCE_ROOT}/abi"
    "${LTX_SOURCE_ROOT}/core"
    "${LTX_SOURCE_ROOT}/core/plugin"
    "${LTX_SOURCE_ROOT}/vendor")
# The plugin uses dlsym(RTLD_DEFAULT) to join shared slots across gfx11/gfx12.
target_link_options(ltx25_mixed_ar_gate PRIVATE -Wl,--export-dynamic)
target_link_libraries(ltx25_mixed_ar_gate PRIVATE
    hip::host
    "${LTX_NATIVE_CORE_LIB}" dl pthread
    "${_native_core_dir}/libbc_serving_v2.a"
    "${_native_core_dir}/libbc_jcs.a"
    "${_native_build_root}/vendor/librad_vendor_llama.a"
    "${_native_core_dir}/librad_json.a"
    "${_native_core_dir}/librad_simd.a"
    "${_native_build_root}/vendor/librad_vendor_yyjson.a"
    "${_native_build_root}/vendor/librad_vendor_httplib.a"
    "${_native_build_root}/vendor/librad_vendor_ryml.a")
