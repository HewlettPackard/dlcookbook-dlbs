/*
 (c) Copyright [2017] Hewlett Packard Enterprise Development LP
 
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#ifndef DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_TENSORRT_UTILS
#define DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_TENSORRT_UTILS
#include <initializer_list>
#include "core/logger.hpp"
#include "core/cuda_utils.hpp"

#include <NvInfer.h>
using namespace nvinfer1;

/**
 * @brief Verifies that this engine has required input and output tensors.
 * 
 * If there are no such tensors, program will terminate.
 * 
 * @param engine is the inference engine.
 * @param input_blob is the name of an input tensor.
 * @param output_blob is the name of an output tensor.
 * @param logger is the TensorRT logger.
 */
void check_bindings(ICudaEngine* engine, const std::string& input_blob, const std::string output_blob, logger_impl& logger);

/**
 * @brief Returns number of elements in a tensor.
 * @param tensor is the tensor instance.
 * @return Number of elements in the tensor.
 */
size_t get_tensor_size(const ITensor* tensor);

/**
 * @brief Returns size of idx-th engine binding.
 * @param engine Inference engine.
 * @param idx is the binding index.
 * @return Size (number of elements) for this binding. It is basically a multiplication if tensor dimensions.
 */
size_t get_binding_size(ICudaEngine* engine, const int idx);

/**
 * @brief Loads TensorRT engine from file
 * @param fname is the name of a file.
 * @param logger is the logger to use.
 * @return Instance of and engine (ICudaEngine pointer)
 */
ICudaEngine* load_engine_from_file(const std::string& fname, logger_impl& logger);

/**
 * @brief Serializes inference engine to a file.
 * @param engine_ Inference engine to serialize.
 * @param fname is the file name.
 */
void serialize_engine_to_file(ICudaEngine *engine_, const std::string& fname);


#endif
