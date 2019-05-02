cmake_minimum_required(VERSION 2.8)

file(GLOB CUDA_FILES "*.cu")
message("CU files: " ${CUDA_FILES})

cuda_add_executable(ParameterSearch
        ${CMAKE_CURRENT_SOURCE_DIR}/ParameterSearch.cpp
        ${CMAKE_HOME_DIRECTORY}/src/3D/NormalizeFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/DoG3DFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/Hessian3DFilter_element.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/CCL3D.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/RegionalFeatures.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/utils.h)

cuda_add_executable(MaxIntensityForImages
        ${CMAKE_CURRENT_SOURCE_DIR}/MaxIntensityForImages.cpp
        ${CUDA_FILES}
        ${CMAKE_HOME_DIRECTORY}/src/3D/utils.h)
