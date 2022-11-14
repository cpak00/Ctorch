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
message(STATUS "Source file" ${ctorch_srcs})

find_package(OpenCV REQUIRED)
message(STATUS "Opencv lib" ${OpenCV_LIBS})

set(ENV{OPENBLAS_NUM_THREAD} 8)
set(ENV{NUM_THREADS} 8)
set(ENV{USE_OPENMP} 1)

add_executable(blas_test test/blas_test.cpp)
target_link_libraries(blas_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(tensor_test test/tensor_test.cpp ${ctorch_srcs})
target_link_libraries(tensor_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(add_test test/add_test.cpp ${ctorch_srcs})
target_link_libraries(add_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(linearm_test test/linearm_test.cpp ${ctorch_srcs})
target_link_libraries(linearm_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(conv2dm_test test/conv2dm_test.cpp ${ctorch_srcs})
target_link_libraries(conv2dm_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(transpose_test test/transpose_test.cpp ${ctorch_srcs})
target_link_libraries(transpose_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(pooling_test test/pooling_test.cpp ${ctorch_srcs})
target_link_libraries(pooling_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(loss_test test/loss_test.cpp ${ctorch_srcs})
target_link_libraries(loss_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(optim_test test/optim_test.cpp ${ctorch_srcs})
target_link_libraries(optim_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(opencv_test test/opencv_test.cpp ${ctorch_srcs})
target_link_libraries(opencv_test openblas pthread gfortran ${OpenCV_LIBS})

add_executable(vgg_mnist sample/vgg_mnist.cpp ${ctorch_srcs})
target_link_libraries(vgg_mnist openblas pthread gfortran ${OpenCV_LIBS})
