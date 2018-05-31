cmake_minimum_required(VERSION 3.2)
project(TensorRT_Benchmarks)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(src)

if (0)
    add_compile_options(-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization)
    add_compile_options(-Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept)
    add_compile_options(-Wold-style-cast -Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo)
    add_compile_options(-Wstrict-null-sentinel -Wswitch-default -Wundef -Werror -Wno-unused)
    add_compile_options(-Wstrict-overflow=5)
endif()

# Base library that does not use CUDA or NvInfer
set(TENSORRT_LIB_SOURCE_FILES "src/core/queues.hpp" "src/core/queues.ipp"
                              "src/core/utils.hpp" "src/core/utils.cpp"
                              "src/core/logger.hpp" "src/core/logger.cpp"
                              "src/core/infer_msg.hpp"
                              "src/core/infer_engine.hpp" "src/core/infer_engine.cpp"
                              "src/core/dataset/dataset.hpp" "src/core/dataset/dataset.cpp"
                              "src/core/dataset/image_dataset.hpp" "src/core/dataset/image_dataset.cpp"
                              "src/core/dataset/tensor_dataset.hpp" "src/core/dataset/tensor_dataset.cpp")
add_library(tensorrt_lib STATIC ${TENSORRT_LIB_SOURCE_FILES})
set_target_properties(tensorrt_lib PROPERTIES LINKER_LANGUAGE CXX)
set(TENSORRT_LIBS tensorrt_lib pthread)


# 
set(TENSORRT_CUDA_LIB_SOURCE_FILES "src/core/cuda_utils.hpp")
add_library(tensorrt_cuda_lib STATIC ${TENSORRT_CUDA_LIB_SOURCE_FILES})
set_target_properties(tensorrt_cuda_lib PROPERTIES LINKER_LANGUAGE CXX)


#
set(TENSORRT_RT_LIB_SOURCE_FILES "src/engines/tensorrt/tensorrt_utils.hpp" "src/engines/tensorrt/tensorrt_utils.cpp"
                                 "src/engines/tensorrt/calibrator.hpp" "src/engines/tensorrt/profiler.hpp"
                                 "src/engines/tensorrt_engine.hpp" "src/engines/tensorrt_engine.cpp"
                                 "src/engines/mgpu_engine.hpp")
add_library(tensorrt_rt_lib STATIC ${TENSORRT_RT_LIB_SOURCE_FILES})
set_target_properties(tensorrt_rt_lib PROPERTIES LINKER_LANGUAGE CXX)


add_executable(tests_queue tests/tests_queue.cpp)
target_link_libraries(tests_queue ${TENSORRT_LIBS})


add_executable(tests_utils tests/tests_utils.cpp)
target_link_libraries(tests_utils ${TENSORRT_LIBS})


add_executable(benchmark_tensor_dataset tools/benchmark_tensor_dataset.cpp)
target_link_libraries(benchmark_tensor_dataset ${TENSORRT_LIBS})

add_executable(benchmark_host2device_copy tools/benchmark_host2device_copy.cpp)
target_link_libraries(benchmark_host2device_copy ${TENSORRT_LIBS} tensorrt_cuda_lib)


add_executable(tests_ipc tests/tests_ipc.cpp)
target_link_libraries(tests_ipc ${TENSORRT_LIBS})


add_executable(tensorrt tools/tensorrt.cpp)
target_link_libraries(tensorrt ${TENSORRT_LIBS} tensorrt_cuda_lib tensorrt_rt_lib libnvinfer.so.2 nvcaffe_parser)


add_executable(images2tensors tools/images2tensors.cpp)
target_link_libraries(images2tensors ${TENSORRT_LIBS})


FIND_PACKAGE(CUDA COMPONENTS cudart REQUIRED)
if(CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(tensorrt ${CUDA_LIBRARIES})
    target_link_libraries(benchmark_host2device_copy ${CUDA_LIBRARIES})
endif()

FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
    target_link_libraries(tensorrt ${Boost_LIBRARIES})
    target_link_libraries(images2tensors ${Boost_LIBRARIES})
    target_link_libraries(benchmark_tensor_dataset ${Boost_LIBRARIES})
    target_link_libraries(benchmark_host2device_copy ${Boost_LIBRARIES})
endif()

find_package(OpenCV 2)
if(OpenCV_FOUND)
    add_definitions(-DHAS_OPENCV)
    include_directories(${OpenCV_INCLUDE_DIRS})
    target_link_libraries(tensorrt ${OpenCV_LIBS})
    target_link_libraries(images2tensors ${OpenCV_LIBS})
endif()


install(TARGETS tensorrt RUNTIME DESTINATION bin)
install(TARGETS images2tensors RUNTIME DESTINATION bin)
install(TARGETS benchmark_tensor_dataset RUNTIME DESTINATION bin)
install(TARGETS benchmark_host2device_copy RUNTIME DESTINATION bin)
install(TARGETS tests_ipc RUNTIME DESTINATION bin)


#if(NOT CMAKE_BUILD_TYPE)
#  set(CMAKE_BUILD_TYPE Release)
#endif()
set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
