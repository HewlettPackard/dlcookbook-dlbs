cmake_minimum_required(VERSION 2.6)
project(tensorrt_benchmarks)

add_compile_options(-std=c++11)

add_executable(tensorrt main.cpp)

FIND_PACKAGE(CUDA COMPONENTS cudart REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(tensorrt ${CUDA_LIBRARIES})
endif()

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(tensorrt ${Boost_LIBRARIES})
endif()

target_link_libraries(tensorrt libnvinfer.so.2)
target_link_libraries(tensorrt nvcaffe_parser)

install(TARGETS tensorrt RUNTIME DESTINATION bin)

set(CMAKE_BUILD_TYPE Release)
