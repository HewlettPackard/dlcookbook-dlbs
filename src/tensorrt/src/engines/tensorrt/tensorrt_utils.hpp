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
#include "core/infer_engine.hpp"

#include <NvInfer.h>
using namespace nvinfer1;


class tensorrt_utils {
public:
    /**
    * @brief Return version of the TensorRT engine as string
    * @return Version as MAJOR.MINOR.PATCH
    */
    static std::string tensorrt_version();
    
    /**
    * @brief Return major version of the TensorRT  engine
    * @return Major version, integer
    */
    static int tensorrt_major_version();

    /**
    * @brief Return version of the Onnx parser as string
    * @return Version as MAJOR.MINOR.PATCH or "not supported".
    */
    static std::string onnx_parser_version();

    /**
     * @brief Computes names of input and output tensors based network definition.
     * 
     * This method identifies the rigth names for input/output tensors. It may override
     * user provided names in several cases. The logic is the following (for both input and
     * output tensors):
     *   - If there are no tensors, print error message and exit.
     *   - If there is a single tensor, use name of that tensor. If that name differs from
     *     a name provided by a user, issue a warning.
     *   - If there are multiple tensors, use the one provided by a user. If there are no such
     *     tensor, print error and exit.
     * 
     * @param me A string that is used to identify inference engine invoking this call.
     * @param logger A logger instance.
     * @param network An instance of a network definition.
     * @param opts Inference options.
     * @param input_name Output parameter that contains name of a model input tensor.
     * @param output_name Output parameter that contains name of a model output tensor.
     */
    static void get_input_output_names(const std::string& me, logger_impl& logger,
                                       INetworkDefinition* network, const inference_engine_opts& opts,
                                       std::string& input_name, std::string& output_name);

        /**
     * @brief Computes names of input and output tensors based network definition.
     * 
     * Same as above method but uses engine instead of a network definition.
     * 
     * @param me A string that is used to identify inference engine invoking this call.
     * @param logger A logger instance.
     * @param engine An instance of an inference engine.
     * @param opts Inference options.
     * @param input_name Output parameter that contains name of a model input tensor.
     * @param output_name Output parameter that contains name of a model output tensor.
     */
    static void get_input_output_names(const std::string& me, logger_impl& logger,
                                       ICudaEngine* engine, const inference_engine_opts& opts,
                                       std::string& input_name, std::string& output_name);

    /**
    * @brief Returns size of idx-th engine binding.
    * @param engine Inference engine.
    * @param idx is the binding index.
    * @return Size (number of elements) for this binding. It is basically a multiplication if tensor dimensions.
    */
    static size_t get_tensor_size(ICudaEngine* engine, const int binding_idx);

    /**
    * @brief Returns number of elements in a tensor.
    * @param tensor is the tensor instance.
    * @return Number of elements in the tensor.
    */    
    static size_t get_tensor_size(INetworkDefinition* network, const std::string& tensor_name);

    /**
    * @brief Loads TensorRT engine from file
    * @param fname is the name of a file.
    * @param logger is the logger to use.
    * @return Instance of and engine (ICudaEngine pointer)
    */
    static ICudaEngine* load_engine_from_file(const std::string& fname, logger_impl& logger);

    /**
    * @brief Serializes inference engine to a file.
    * @param engine_ Inference engine to serialize.
    * @param fname is the file name.
    */
    static void serialize_engine_to_file(ICudaEngine *engine_, const std::string& fname);

};

#endif
