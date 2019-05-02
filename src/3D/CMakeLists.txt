cmake_minimum_required(VERSION 2.8)

file(GLOB CUDA_FILES "*.cu")
message("CU files: " ${CUDA_FILES})

cuda_add_executable(HDoG3D_main
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_main.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
)

cuda_add_executable(HDoG3D_main2
        ${CMAKE_CURRENT_SOURCE_DIR}/HDoG3D_main2.cpp
        ${CUDA_FILES}
        ${CMAKE_CURRENT_SOURCE_DIR}/utils.h
)
