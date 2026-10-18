cmake_minimum_required(VERSION 3.20)
project(ctxshield_native LANGUAGES C CXX)

# Build doc Part IX. Step 3: CPU backend. Step 8: C ABI + Metal (Apple) + CUDA (UNTESTED, L4).
if(APPLE)
  option(SHIELD_METAL "Build the Metal backend (Apple)" ON)
else()
  option(SHIELD_METAL "Build the Metal backend (Apple)" OFF)
endif()
option(SHIELD_CUDA "Build the CUDA backend (UNTESTED until an L4 run)" OFF)
# The pybind11 module (_native) needs Python headers; the standalone scorer (ctxshield-scorer) does not.
# static_build/ builds the scorer alone (SHIELD_PYTHON=OFF, SHIELD_STATIC=ON) inside Alpine.
option(SHIELD_PYTHON "Build the pybind11 module ctxshield._native" ON)
option(SHIELD_STATIC "Link ctxshield-scorer fully static (Linux only: no libc / libstdc++ at run time)" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(SHIELD_PYTHON)
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG REQUIRED)
endif()

add_library(utf8proc STATIC third_party/utf8proc/utf8proc.c)   # includes utf8proc_data.c
target_compile_definitions(utf8proc PUBLIC UTF8PROC_STATIC)
target_include_directories(utf8proc PUBLIC third_party/utf8proc)
set_target_properties(utf8proc PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(SHIELD_SOURCES src/featurize.cpp src/score.cpp src/c_abi.cpp python/bind.cpp)

if(SHIELD_METAL)
  if(NOT APPLE)
    message(FATAL_ERROR "SHIELD_METAL=ON requires Apple")
  endif()
  enable_language(OBJCXX)
  # shield.metal -> shield.air -> shield.metallib -> embedded as bytes (metallib_data.h); the
  # library ships inside the extension, so the kernel binary is bound to the module's sha.
  set(METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/shield.metal)
  set(METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/shield.air)
  set(METAL_LIB ${CMAKE_CURRENT_BINARY_DIR}/shield.metallib)
  set(METAL_HDR ${CMAKE_CURRENT_BINARY_DIR}/metallib_data.h)
  add_custom_command(OUTPUT ${METAL_AIR}
    COMMAND xcrun -sdk macosx metal -O3 -std=metal3.0 -Wall -c ${METAL_SRC} -o ${METAL_AIR}
    DEPENDS ${METAL_SRC} VERBATIM)
  add_custom_command(OUTPUT ${METAL_LIB}
    COMMAND xcrun -sdk macosx metallib ${METAL_AIR} -o ${METAL_LIB}
    DEPENDS ${METAL_AIR} VERBATIM)
  add_custom_command(OUTPUT ${METAL_HDR}
    COMMAND ${CMAKE_COMMAND} -DIN=${METAL_LIB} -DOUT=${METAL_HDR} -P ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/embed.cmake
    DEPENDS ${METAL_LIB} ${CMAKE_CURRENT_SOURCE_DIR}/src/metal/embed.cmake VERBATIM)
  list(APPEND SHIELD_SOURCES src/metal/metal_backend.mm ${METAL_HDR})
endif()

if(SHIELD_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)   # L4 (Ada)
  endif()
  find_package(CUDAToolkit REQUIRED)
  list(APPEND SHIELD_SOURCES src/cuda/shield.cu src/cuda/cuda_backend.cpp)
endif()


# CPU baseline. SHIELD_TARGET_ARCH names the machine the module is FOR (set by tools/build_native_all.sh; one
# module per target). Each baseline runs on every CPU of that family — never -march=native in a shipped build
# (it tunes to the build machine and can die with "illegal instruction" elsewhere). SHIELD_NATIVE_TUNE=ON only for
# local benchmarking.
option(SHIELD_NATIVE_TUNE "Tune for THIS machine (-march=native); never for a shipped module" OFF)
set(SHIELD_TARGET_ARCH "" CACHE STRING "arm64 | x86_64 | aarch64 (empty = the build machine's own)")
if(SHIELD_TARGET_ARCH STREQUAL "")
  set(SHIELD_TARGET_ARCH ${CMAKE_SYSTEM_PROCESSOR})
endif()
if(SHIELD_NATIVE_TUNE)
  set(SHIELD_ARCH_FLAGS -O3 -march=native)
elseif(APPLE AND SHIELD_TARGET_ARCH MATCHES "^(arm64|aarch64)$")
  set(SHIELD_ARCH_FLAGS -O3 -mcpu=apple-m1)          # every Apple Silicon Mac is M1 or newer
elseif(SHIELD_TARGET_ARCH MATCHES "^(arm64|aarch64)$")
  set(SHIELD_ARCH_FLAGS -O3 -march=armv8-a)          # every 64-bit ARM Linux (NEON is part of armv8-a)
elseif(SHIELD_TARGET_ARCH MATCHES "^(x86_64|AMD64|amd64)$")
  set(SHIELD_ARCH_FLAGS -O3 -march=x86-64-v2)        # SSE4.2/POPCNT: every x86-64 CPU since ~2009
else()
  message(FATAL_ERROR "unsupported SHIELD_TARGET_ARCH=${SHIELD_TARGET_ARCH}")
endif()
target_compile_options(utf8proc PRIVATE -O3)

# Standalone scorer: CPU backend only, speaks the CXS1 pipe protocol (scorer/scorer_main.cpp).
add_executable(ctxshield-scorer scorer/scorer_main.cpp src/featurize.cpp src/score.cpp src/c_abi.cpp)
target_include_directories(ctxshield-scorer PRIVATE include src)
target_link_libraries(ctxshield-scorer PRIVATE utf8proc)
target_compile_options(ctxshield-scorer PRIVATE ${SHIELD_ARCH_FLAGS} -Wall -Wextra)
if(SHIELD_STATIC)
  if(APPLE)
    message(FATAL_ERROR "SHIELD_STATIC=ON is Linux only (macOS has no static libc)")
  endif()
  set_target_properties(ctxshield-scorer PROPERTIES POSITION_INDEPENDENT_CODE OFF)
  target_link_options(ctxshield-scorer PRIVATE -static -no-pie -s)
endif()

if(NOT SHIELD_PYTHON)
  if(SHIELD_METAL OR SHIELD_CUDA)
    message(FATAL_ERROR "SHIELD_METAL / SHIELD_CUDA need SHIELD_PYTHON=ON (the scorer is CPU only)")
  endif()
  return()
endif()

pybind11_add_module(_native ${SHIELD_SOURCES})
target_include_directories(_native PRIVATE include src ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(_native PRIVATE utf8proc)
target_compile_options(_native PRIVATE $<$<COMPILE_LANGUAGE:CXX,OBJCXX>:${SHIELD_ARCH_FLAGS} -Wall -Wextra>)

if(SHIELD_METAL)
  target_compile_definitions(_native PRIVATE SHIELD_HAVE_METAL=1)
  set_source_files_properties(src/metal/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
  target_link_libraries(_native PRIVATE "-framework Metal" "-framework Foundation")
endif()
if(SHIELD_CUDA)
  target_compile_definitions(_native PRIVATE SHIELD_HAVE_CUDA=1 SHIELD_CUDA_VERIFIED=0)
  set_target_properties(_native PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
  target_include_directories(_native PRIVATE src/cuda)
  target_compile_options(_native PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo>)
  target_link_libraries(_native PRIVATE CUDA::cudart)
endif()

# Stage-split profiler (no Python): build-*/shield_bench <docs.bin>
add_executable(shield_bench bench/bench_kernel.cpp src/featurize.cpp src/score.cpp)
target_include_directories(shield_bench PRIVATE include src)
target_link_libraries(shield_bench PRIVATE utf8proc)
target_compile_options(shield_bench PRIVATE ${SHIELD_ARCH_FLAGS} -Wall -Wextra)
target_compile_definitions(shield_bench PRIVATE SHIELD_PROFILE=1)
