cmake_minimum_required(VERSION 2.8)

add_executable(ScaleMerge
        ${CMAKE_CURRENT_SOURCE_DIR}/ScaleMerge.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tinytiffreader.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/tinytiffwriter.cpp
)

install(TARGETS ScaleMerge RUNTIME DESTINATION ".")
