cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
project(GANNS LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(SOURCES
    src/main.cu
)

add_executable(${PROJECT_NAME} ${SOURCES})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  target_compile_options(${PROJECT_NAME} PRIVATE -G -g --device-debug)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
  target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()

target_compile_definitions(${PROJECT_NAME} PRIVATE USE_DIST_L2_)
target_compile_definitions(${PROJECT_NAME} PRIVATE PLACE_HOLDER_DIM=128)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_STANDARD 14
)