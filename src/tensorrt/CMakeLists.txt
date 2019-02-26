cmake_minimum_required(VERSION 3.2)
project(TensorRT_Benchmarks)


option(DEBUG_LOG "Enable detailed logging for some of the componenets." OFF)
set(HOST_DTYPE "INT8" CACHE STRING "Data type to use to store images in host OS (INT8 or FP32 (=INT8))")

# Following preprocessor variables can be defined: HAVE_CUDA, HAVE_NVINFER, HAVE_OPENCV
find_package(Boost REQUIRED COMPONENTS program_options)
find_package(CUDA COMPONENTS cudart)
find_package(OpenCV 2)
find_package(Doxygen)

# Find TensorRT library (libnvinfer.so). If found, also search for caffe and onnx parsers.
find_path(nvinfer_include_dir NvInfer.h)
find_library(nvinfer_library nvinfer)
if (nvinfer_include_dir AND nvinfer_library)
    set (NvInfer_FOUND TRUE)
    get_filename_component(nvinfer_path ${nvinfer_library} DIRECTORY)

    find_path(caffe_parser_include_dir NvCaffeParser.h)
    find_library(caffe_parser_library nvcaffe_parser HINTS ${nvinfer_path} NO_DEFAULT_PATH)
    if (caffe_parser_include_dir AND caffe_parser_library)
        set (CaffeParser_FOUND TRUE)
    else() 
        set (CaffeParser_FOUND FALSE)
    endif()

    find_path(onnx_parser_include_dir NvOnnxParser.h)
    find_library(onnx_parser_library nvonnxparser HINTS ${nvinfer_path} NO_DEFAULT_PATH)
    if (onnx_parser_include_dir AND onnx_parser_library)
        set (OnnxParser_FOUND TRUE)
    else() 
        set (OnnxParser_FOUND FALSE)
    endif()
else() 
    set (NvInfer_FOUND FALSE)
endif()


include_directories(src)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release CACHE STRING "Release or Debug mode.")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
message("Configuring TensorRT benchmark backend in ${CMAKE_BUILD_TYPE} mode.")

# https://gcc.gnu.org/onlinedocs/gcc/Warning-Options.html#Warning-Options
add_compile_options(-Wpedantic)              # Issue all the warnings demanded by strict ISO C and ISO C++ ...
add_compile_options(-Wall)                   # This enables all the warnings about constructions that some users consider questionable, and that are easy to avoid ...
add_compile_options(-Wextra)                 # This enables some extra warning flags that are not enabled by -Wall ...
add_compile_options(-Wformat)                # Check calls to printf and scanf, etc., to make sure that the arguments supplied have types appropriate ...
add_compile_options(-Wuninitialized)         # Warn if an automatic variable is used without first being initialized or ...
add_compile_options(-Wmaybe-uninitialized)   # For an automatic (i.e. local) variable, if there exists a path from the function entry to a use ...
add_compile_options(-Winit-self)             # Warn about uninitialized variables that are initialized with themselves.
add_compile_options(-Wmissing-include-dirs)  # Warn if a user-supplied include directory does not exist.
add_compile_options(-Wreturn-type)           # Warn whenever a function is defined with a return type that defaults to int ...
add_compile_options(-Wsuggest-override)      # Warn about overriding virtual functions that are not marked with the override keyword.
add_compile_options(-Wshadow)                # Warn whenever a local variable or type declaration shadows another variable ...
add_compile_options(-Wcast-qual)             # Warn whenever a pointer is cast so as to remove a type qualifier from the target type.
add_compile_options(-Wredundant-decls)       # Warn if anything is declared more than once in the same scope, even in cases where ...

if (0)
    add_compile_options(-Wcast-align -Wctor-dtor-privacy -Wdisabled-optimization)
    add_compile_options( -Wlogical-op -Wmissing-declarations  -Wnoexcept)
    add_compile_options(-Woverloaded-virtual -Wshadow -Wsign-conversion -Wsign-promo)
    add_compile_options(-Wstrict-null-sentinel -Wswitch-default -Werror -Wno-unused)
    add_compile_options(-Wstrict-overflow=5 -Wold-style-cast  -Wundef)
endif()
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
if (DEBUG_LOG)
    message("Configuring TensorRT benchmark backend with detailed logging.")
    add_definitions(-DDEBUG_LOG)
endif()

if("xyz_${HOST_DTYPE}" STREQUAL "xyz_INT8")
    add_definitions(-DHOST_DTYPE_INT8)
elseif("xyz_${HOST_DTYPE}" STREQUAL "xyz_FP32")
    add_definitions(-DHOST_DTYPE_FP32)
else()
    message(FATAL_ERROR "Invalid value for HOST_DTYPE (=${HOST_DTYPE}). Must be INT8 or FP32.")
endif()
message("Configuring TensorRT benchmark backend with ${HOST_DTYPE} data type.")
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
include_directories(SYSTEM ${Boost_INCLUDE_DIRS})

if(CUDA_FOUND)
    message("Configuring TensorRT benchmark backend with CUDA library.")
    add_definitions(-DHAVE_CUDA)
    list(APPEND CUDA_NVCC_FLAGS "-std=c++11;-O1;-DVERBOSE")
    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
endif()

if (NvInfer_FOUND)
    message("Configuring TensorRT benchmark backend with NvInfer library.")
    add_definitions(-DHAVE_NVINFER)
    include_directories(SYSTEM ${nvinfer_include_dir})
    link_directories(${nvinfer_path})
    
    if (CaffeParser_FOUND)
        message("Configuring TensorRT benchmark backend with CaffeParser library.")
        add_definitions(-DHAVE_CAFFE_PARSER)
        include_directories(SYSTEM ${caffe_parser_include_dir})
    endif()
    
    if (OnnxParser_FOUND)
        message("Configuring TensorRT benchmark backend with OnnxParser library.")
        add_definitions(-DHAVE_ONNX_PARSER)
        include_directories(SYSTEM ${onnx_parser_include_dir})
    endif()
    
endif()

if(OpenCV_FOUND)
    message("Configuring TensorRT benchmark backend with OpenCV library.")
    add_definitions(-DHAVE_OPENCV)
    include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
endif()

if (DOXYGEN_FOUND)
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
endif(DOXYGEN_FOUND)
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
add_library(trtbenchbase
            STATIC
            "src/core/queues.hpp" "src/core/queues.ipp"
            "src/core/utils.hpp" "src/core/utils.cpp"
            "src/core/logger.hpp" "src/core/logger.cpp"
            "src/core/infer_msg.hpp"
            "src/core/infer_engine.hpp" "src/core/infer_engine.cpp"
            "src/core/dataset/dataset.hpp" "src/core/dataset/dataset.cpp"
            "src/core/dataset/image_dataset.hpp" "src/core/dataset/image_dataset.cpp"
            "src/core/dataset/tensor_dataset.hpp" "src/core/dataset/tensor_dataset.cpp")
set_target_properties(trtbenchbase PROPERTIES LINKER_LANGUAGE CXX)
set(base_libs trtbenchbase pthread ${Boost_LIBRARIES})

if(CUDA_FOUND)
    cuda_add_library(trtbenchcuda
                     STATIC
                     "src/engines/tensorrt/gpu_cast.h" "src/engines/tensorrt/gpu_cast.cu"
                     "src/core/cuda_utils.hpp")
    set(cuda_libs trtbenchcuda ${base_libs})
endif()

if (NvInfer_FOUND)
    add_library(trtbenchinfer 
                STATIC
                "src/engines/tensorrt/tensorrt_utils.hpp" "src/engines/tensorrt/tensorrt_utils.cpp"
                "src/engines/tensorrt/calibrator.hpp" "src/engines/tensorrt/profiler.hpp"
                "src/engines/tensorrt_engine.hpp" "src/engines/tensorrt_engine.cpp"
                "src/engines/mgpu_engine.hpp")
    set_target_properties(trtbenchinfer PROPERTIES LINKER_LANGUAGE CXX)
    set(infer_libs trtbenchinfer ${cuda_libs} libnvinfer.so)
    if (CaffeParser_FOUND)
        set(infer_libs ${infer_libs} libnvcaffe_parser.so)
    endif()
    if (OnnxParser_FOUND)
        set(infer_libs ${infer_libs} libnvonnxparser.so)
    endif()
endif()
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
add_executable(tests_ipc tests/tests_ipc.cpp)
    target_link_libraries(tests_ipc ${base_libs})
    install(TARGETS tests_ipc RUNTIME DESTINATION bin)

add_executable(tests_queue tests/tests_queue.cpp)
    target_link_libraries(tests_queue ${base_libs})

add_executable(tests_direct_reader tests/tests_direct_reader.cpp)
    target_link_libraries(tests_direct_reader ${base_libs})
    install(TARGETS tests_direct_reader RUNTIME DESTINATION bin)

add_executable(benchmark_tensor_dataset tools/benchmark_tensor_dataset.cpp)
    target_link_libraries(benchmark_tensor_dataset ${base_libs})
    install(TARGETS benchmark_tensor_dataset RUNTIME DESTINATION bin)

add_executable(images2tensors tools/images2tensors.cpp)
    target_link_libraries(images2tensors ${base_libs})
    if (OpenCV_FOUND)
        target_link_libraries(images2tensors ${OpenCV_LIBS})
    endif()
    install(TARGETS images2tensors RUNTIME DESTINATION bin)

add_executable(benchmark_host2device_copy tools/benchmark_host2device_copy.cpp)
    if(CUDA_FOUND)
        target_link_libraries(benchmark_host2device_copy ${cuda_libs})
    else()
        target_link_libraries(benchmark_host2device_copy ${base_libs})
    endif()
    install(TARGETS benchmark_host2device_copy RUNTIME DESTINATION bin)

add_executable(tensorrt tools/tensorrt.cpp)
    if(NvInfer_FOUND)
        target_link_libraries(tensorrt ${infer_libs})
    endif()
    if(OpenCV_FOUND)
        target_link_libraries(tensorrt ${OpenCV_LIBS})
    endif()
    install(TARGETS tensorrt RUNTIME DESTINATION bin)
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
add_subdirectory(tests)
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------------------------
