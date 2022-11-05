cmake_minimum_required(VERSION 3.10)
set (CMAKE_CXX_STANDARD 11)
project(Ctorch)
set(CMAKE_BUILD_TYPE Debug)

message(STATUS "CMake version: " ${CMAKE_VERSION})
message(STATUS "CMake system name: " ${CMAKE_SYSTEM_NAME})
message(STATUS "CMake system processor: " ${CMAKE_SYSTEM_PROCESSOR})

aux_source_directory(src DIR_SRCS)
set(PROJECT_SOURCE_DIR src)
set(EXECUTABLE_OUTPUT_PATH bin)

file(GLOB_RECURSE ctorch_srcs src/*.cpp)

set(ENV{OPENBLAS_NUM_THREAD} 8)
set(ENV{NUM_THREADS} 8)
set(ENV{USE_OPENMP} 1)

add_executable(blas_test test/blas_test.cpp)
target_link_libraries(blas_test openblas pthread gfortran)

add_executable(tensor_test test/tensor_test.cpp ${ctorch_srcs})
target_link_libraries(tensor_test openblas pthread gfortran)

add_executable(linear_test test/linear_test.cpp ${ctorch_srcs})
target_link_libraries(linear_test openblas pthread gfortran)

add_executable(conv2d_test test/conv2d_test.cpp ${ctorch_srcs})
target_link_libraries(conv2d_test openblas pthread gfortran)

add_executable(transpose_test test/transpose_test.cpp ${ctorch_srcs})
target_link_libraries(transpose_test openblas pthread gfortran)
