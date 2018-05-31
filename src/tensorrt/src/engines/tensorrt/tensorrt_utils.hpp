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

void check_bindings(ICudaEngine* engine, const std::string& input_blob, const std::string output_blob, logger_impl& logger);
size_t get_tensor_size(const ITensor* tensor);
size_t get_binding_size(ICudaEngine* engine, const int idx);

ICudaEngine* load_engine_from_file(const std::string& fname, logger_impl& logger);
void serialize_engine_to_file(ICudaEngine *engine_, const std::string& fname);


#endif
