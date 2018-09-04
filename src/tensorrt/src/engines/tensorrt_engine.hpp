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

#ifndef DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_ENGINE
#define DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_ENGINE

#include "core/infer_engine.hpp"
#include "engines/tensorrt/calibrator.hpp"
#include "engines/tensorrt/profiler.hpp"

#include <NvCaffeParser.h>

/**
 * @brief Implementation of an inference engine that uses TensorRT library.
 * 
 * This engine works with one GPU.
 */
class tensorrt_inference_engine : public inference_engine {
private:
    calibrator_impl calibrator_;
    profiler_impl *profiler_ = nullptr;
    
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* exec_ctx_ = nullptr;
#ifdef HOST_DTYPE_INT8
    // If data is stored with unsigned char data type in host memory,
    // we need intermidiate input buffer for data that will later
    // be casted to float data type in bindings_ array.
    host_dtype *input_buffer_ = nullptr;
#endif
    std::vector<void*> bindings_;      // Input/output data in GPU memort.
                                       // This is used directly by TensorRT API.
    size_t input_idx_ = 0;             // Index of input data in gpu_mem_.
    size_t output_idx_ = 0;            // Index of output data in gpu_mem_.
private:
    void copy_input_to_gpu_asynch(inference_msg *msg, cudaStream_t stream);
    
    void init_device() override;
    void do_inference(abstract_queue<inference_msg*> &request_queue,
                      abstract_queue<inference_msg*> &response_queue);
    // Past implementations
    /**
     * @brief Original implementation that does everything sequentially:
     *    - Fetch inference request
     *    - Copy to GPU
     *    - Run Inference
     *    - Copy back to host
     *    - Submit results
     */
    void do_inference1(abstract_queue<inference_msg*> &request_queue,
                       abstract_queue<inference_msg*> &response_queue);
public:
    profiler_impl* profiler() { return profiler_; }
    tensorrt_inference_engine(const int engine_id, const int num_engines,
                              logger_impl& logger, const inference_engine_opts& opts);
    ~tensorrt_inference_engine();
};

#endif
