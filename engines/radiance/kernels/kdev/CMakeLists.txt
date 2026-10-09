# kdev: build kernel CANDIDATES for radiance as one small plugin each, so rad-kbench can run them side by side.
#
# Every directory under candidates/ is one candidate: its *.cpp are host sources (the row table), its *.hip device
# sources. It becomes the kernel plugin k_<directory>. A candidate may include radiance's own libr4d headers and
# units (RADIANCE_SRC), which is how "the upstream kernel, compiled for this card" is compared with a rewrite.
#
#   cmake -S engines/radiance/kernels/kdev -B <build> -G Ninja -DCMAKE_PREFIX_PATH=<installed radiance> \
#         -DRADIANCE_SRC=<radiance source tree> [-DKDEV_TARGETS=gfx1100] [-DKDEV_CANDIDATES="a;b"]
#
# tools/lab/radiance/kdev.sh drives this and the bench.
cmake_minimum_required(VERSION 3.21)
project(radiance_kdev LANGUAGES CXX HIP)
find_package(radiance REQUIRED)
rad_plugin_defaults()

set(KDEV_TARGETS "gfx1100" CACHE STRING "GPU targets the candidates are compiled for")
set(KDEV_CANDIDATES "" CACHE STRING "candidate directories to build; empty = every directory under candidates/")
set(RADIANCE_SRC "" CACHE PATH "radiance source tree, for candidates that include libr4d's headers or units")

if(KDEV_CANDIDATES)
  set(candidates ${KDEV_CANDIDATES})
else()
  file(GLOB found RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/candidates CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/candidates/*)
  set(candidates "")
  foreach(entry ${found})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/candidates/${entry})
      list(APPEND candidates ${entry})
    endif()
  endforeach()
endif()

set(includes ${CMAKE_CURRENT_SOURCE_DIR}/common)
if(RADIANCE_SRC)
  list(APPEND includes ${RADIANCE_SRC}/libr4d)
endif()

foreach(candidate ${candidates})
  set(dir ${CMAKE_CURRENT_SOURCE_DIR}/candidates/${candidate})
  if(NOT IS_DIRECTORY ${dir})
    message(FATAL_ERROR "kdev: no candidate directory ${dir}")
  endif()
  file(GLOB host CONFIGURE_DEPENDS ${dir}/*.cpp)
  file(GLOB device CONFIGURE_DEPENDS ${dir}/*.hip)
  if(NOT host OR NOT device)
    message(FATAL_ERROR "kdev: candidate ${candidate} needs a row table (*.cpp) and device code (*.hip)")
  endif()
  rad_add_plugin(k_${candidate} kernels
    SOURCES     ${host}
    HIP_SOURCES ${device}
    GPU_TARGETS ${KDEV_TARGETS}
    INCLUDES    ${includes} ${dir})
  target_compile_definitions(k_${candidate} PRIVATE KDEV_PLUGIN_NAME="k_${candidate}" KDEV_TARGETS="${KDEV_TARGETS}")
  # the reference contract: f32 accumulation, no FMA, no reassociation (libr4d and libref build the same way)
  target_compile_options(k_${candidate} PRIVATE -ffp-contract=off)
  set_target_properties(k_${candidate} PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON)
  message(STATUS "kdev: candidate k_${candidate} for ${KDEV_TARGETS}")
endforeach()
