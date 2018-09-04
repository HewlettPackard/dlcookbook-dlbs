cmake_minimum_required(VERSION 3.2)
project(TensorRT_Benchmarks)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(src)

if (0)
    # -Wold-style-cast  -Wundef
    add_compile_options(-pedantic -Wall -Wextra -Wcast-align -Wcast-qual -Wctor-dtor-privacy -Wdisabled-optimization)
    add_compile_options(-Wformat=2 -Winit-self -Wlogical-op -Wmissing-declarations -Wmissing-include-dirs -Wnoexcept)
    add_compile_options(-Woverloaded-virtual -Wredundant-decls -Wshadow -Wsign-conversion -Wsign-promo)
    add_compile_options(-Wstrict-null-sentinel -Wswitch-default -Werror -Wno-unused)
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
                                 "src/engines/mgpu_engine.hpp"
                                 "src/engines/tensorrt/gpu_cast.h")
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


add_executable(tests_direct_reader tests/tests_direct_reader.cpp)
target_link_libraries(tests_direct_reader ${TENSORRT_LIBS})


add_executable(tensorrt tools/tensorrt.cpp)
target_link_libraries(tensorrt ${TENSORRT_LIBS} tensorrt_cuda_lib tensorrt_rt_lib libnvinfer.so.2 nvcaffe_parser)


add_executable(images2tensors tools/images2tensors.cpp)
target_link_libraries(images2tensors ${TENSORRT_LIBS})


FIND_PACKAGE(CUDA COMPONENTS cudart REQUIRED)
if(CUDA_FOUND)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O1;-DVERBOSE")
    SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
    CUDA_ADD_LIBRARY(cuda_kernels STATIC "src/engines/tensorrt/gpu_cast.h" "src/engines/tensorrt/gpu_cast.cu")

    include_directories(${CUDA_INCLUDE_DIRS})
    target_link_libraries(tensorrt ${CUDA_LIBRARIES} cuda_kernels)
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
install(TARGETS tests_direct_reader RUNTIME DESTINATION bin)


set(CMAKE_BUILD_TYPE Release)
#set(CMAKE_BUILD_TYPE Debug)
message("Configuring TensorRT benchmark backend in ${CMAKE_BUILD_TYPE} mode.")

set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# -----------------------------------------------------------------------------------
# It must be disable for performance benchmarks.
option(DEBUG_LOG "Enable detailed logging for some of the componenets." OFF)
if (DEBUG_LOG)
    message("Configuring TensorRT benchmark backend with detailed logging.")
    add_definitions(-DDEBUG_LOG)
endif()
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
# The following definitions control the host data type.
set(HOST_DTYPE "INT8")
if("xyz_${HOST_DTYPE}" STREQUAL "xyz_INT8")
    add_definitions(-DHOST_DTYPE_INT8)
elseif("xyz_${HOST_DTYPE}" STREQUAL "xyz_FP32")
    add_definitions(-DHOST_DTYPE_FP32)
else()
    message(FATAL_ERROR "Invalid value for HOST_DTYPE (=${HOST_DTYPE}). Must be INT8 or SP32.")
endif()
message("Configuring TensorRT benchmark backend with ${HOST_DTYPE} data type.")
# -----------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------
find_package(Doxygen)
#if (DOXYGEN_FOUND)
    message("Doxygen tool found. Adding 'build_docs' target. To build HTML documentation, run: make build_docs.")
    # set input and output files
    set(DOXYGEN_IN docs/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    
    # request to configure the file
    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # note the option ALL which allows to build the docs together with the application
    add_custom_target( build_docs
        COMMAND cd ${CMAKE_SOURCE_DIR}/docs && ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/docs
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM )

    install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/html DESTINATION share/doc)
#endif(DOXYGEN_FOUND)
# -----------------------------------------------------------------------------------
