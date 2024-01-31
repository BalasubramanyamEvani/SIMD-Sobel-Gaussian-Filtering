cmake_minimum_required(VERSION 2.8.12.2)

SET(GCC_FLAGS "-std=c++11 -O3 -mavx -mavx2")

add_definitions(${GCC_FLAGS})

project(sobel_simd_project)

find_package(OpenCV REQUIRED)

message(STATUS "config: ${OpenCV_VERSION}")

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(sobel_simd main.cpp sobellib.h sobellib.cpp)

target_link_libraries(sobel_simd ${OpenCV_LIBS})
