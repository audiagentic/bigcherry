# The host declaration sources are libr4d's own list (R4D_DECLARATION), so one libr4d adds is built here too; a
# hand-kept copy once missed r4d_iqr_layout.cpp and every IQR layout became a stub.
file(READ ${RADIANCE_SRC}/libr4d/CMakeLists.txt r4d_cmake)
string(REGEX MATCH "set\\(R4D_DECLARATION([^)]*)\\)" _ "${r4d_cmake}")
string(REGEX REPLACE "[ \t\r\n]+" ";" r4d_decl "${CMAKE_MATCH_1}")
list(FILTER r4d_decl INCLUDE REGEX "\\.cpp$")
if(NOT "r4d_rows.cpp" IN_LIST r4d_decl)
  message(FATAL_ERROR "${PROJECT_NAME}: could not read R4D_DECLARATION from ${RADIANCE_SRC}/libr4d/CMakeLists.txt")
endif()
set(declaration "")
foreach(f ${r4d_decl})
  list(APPEND declaration ${r4d}/${f})
endforeach()
# libr4d's host dispatch table calls one implementation per supported CPU SIMD level.
# Include those objects in both the link and the symbol scan, so layout calls cannot become stubs.
include(${RADIANCE_SRC}/cmake/RadSimdLevels.cmake)
rad_simd_level_objects(r4d_hsimd ${r4d}/r4d_hsimd_kernels.cpp r4d_simd_objects
  DEFINE R4D_HSIMD_LEVEL INCLUDES ${r4d})
list(APPEND declaration ${r4d_simd_objects})
