cmake_minimum_required(VERSION 2.8)

cuda_add_executable(Normalize_test
        ${CMAKE_CURRENT_SOURCE_DIR}/Normalize_test.cpp
        ${CMAKE_HOME_DIRECTORY}/src/3D/NormalizeFilter.cu)

cuda_add_executable(DoG_test
        ${CMAKE_CURRENT_SOURCE_DIR}/DoG_test.cpp
        ${CMAKE_HOME_DIRECTORY}/src/3D/NormalizeFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/DoG3DFilter.cu)

cuda_add_executable(Hessian_test
        ${CMAKE_CURRENT_SOURCE_DIR}/Hessian_test.cpp
        ${CMAKE_HOME_DIRECTORY}/src/3D/NormalizeFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/DoG3DFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/Hessian3DFilter_element.cu)

cuda_add_executable(CCL_test
        ${CMAKE_CURRENT_SOURCE_DIR}/CCL_test.cpp
        ${CMAKE_HOME_DIRECTORY}/src/3D/NormalizeFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/DoG3DFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/Hessian3DFilter_element.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/CCL3D.cu)

cuda_add_executable(Regional_test
        ${CMAKE_CURRENT_SOURCE_DIR}/Regional_test.cpp
        ${CMAKE_HOME_DIRECTORY}/src/3D/NormalizeFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/DoG3DFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/Hessian3DFilter_element.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/CCL3D.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/RegionalFeatures.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/Eigenvalues.cu)

cuda_add_executable(MaxNormalized_test
        ${CMAKE_CURRENT_SOURCE_DIR}/MaxNormalized_test.cpp
        ${CMAKE_HOME_DIRECTORY}/src/3D/NormalizeFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/DoG3DFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/Hessian3DFilter_element.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/CCL3D.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/RegionalFeatures.cu)

cuda_add_executable(AverageNormalized_test
        ${CMAKE_CURRENT_SOURCE_DIR}/AverageNormalized_test.cpp
        ${CMAKE_HOME_DIRECTORY}/src/3D/NormalizeFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/DoG3DFilter.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/Hessian3DFilter_element.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/CCL3D.cu
        ${CMAKE_HOME_DIRECTORY}/src/3D/RegionalFeatures.cu)
