cmake_minimum_required(VERSION 2.8)
project(cellDetection)
find_package(CUDA REQUIRED)

message("-- CUDA --")
message("Version: " ${CUDA_VERSION})
message("Library: " ${CUDA_CUDA_LIBRARY})
message("Runtime: " ${CUDA_CUDART_LIBRARY})

set(CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_HOST_COMPILER g++)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--std=c++14;-arch=sm_61;-gencode=arch=compute_52,code=sm_52;-gencode=arch=compute_61,code=sm_61;-gencode=arch=compute_86,code=sm_86;-Wno-deprecated-gpu-targets")
set(CUDA_NVCC_FLAGS_DEBUG "-G;-g;-O0")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++14")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

include_directories(${CMAKE_SOURCE_DIR}/deps/cuda-samples/Common)
include_directories(${CMAKE_SOURCE_DIR}/deps/nlohmann)

add_subdirectory(src/3D)
add_subdirectory(test)
add_subdirectory(src/ScaleMerge)
