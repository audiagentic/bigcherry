cmake_minimum_required(VERSION 3.21)
project(ltx2_c01_native_syntax LANGUAGES CXX)

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
