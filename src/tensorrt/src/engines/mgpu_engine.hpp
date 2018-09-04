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

#ifndef DLBS_TENSORRT_BACKEND_ENGINES_MGPU_ENGINE
#define DLBS_TENSORRT_BACKEND_ENGINES_MGPU_ENGINE

#include "core/infer_engine.hpp"
#include "engines/tensorrt_engine.hpp"

/**
 * @brief Multi-GPU inference engine that can work with multiple GPUs in a system.
 * 
 * It instantiates N inference engines each running asynchronously in background
 * threads. Each engine fetches data from input request queue. It then runs inference
 * using this data and sends results back to response queue.
 */
class mgpu_inference_engine {
    std::vector<inference_engine*> engines_;
    thread_safe_queue<inference_msg*> request_queue_;
    thread_safe_queue<inference_msg*> response_queue_;
public:
    mgpu_inference_engine(const inference_engine_opts& opts, logger_impl& logger) {
        const int num_engines = static_cast<int>(opts.gpus_.size());
        for (size_t i=0; i<opts.gpus_.size(); ++i) {
            inference_engine *engine = nullptr;
            const auto gpu_id = opts.gpus_[i];
            if (opts.fake_inference_) {
                engine = new fake_inference_engine(static_cast<int>(gpu_id), num_engines, logger, opts);
            } else {
                engine = new tensorrt_inference_engine(static_cast<int>(gpu_id), num_engines, logger, opts);
            }
            engines_.push_back(engine);
        }
    }
    
    virtual ~mgpu_inference_engine() {
        for (size_t i=0; i<engines_.size(); ++i) {
            delete engines_[i];
        }
    }
    
    size_t num_engines() const { return engines_.size(); }
    inference_engine* engine(const size_t idx) { return engines_[idx]; }
    
    thread_safe_queue<inference_msg*>* request_queue()  { return &request_queue_; }
    thread_safe_queue<inference_msg*>* response_queue() { return &response_queue_; }
    
    size_t batch_size() const { return engines_[0]->batch_size(); }
    size_t input_size() const { return engines_[0]->input_size(); }
    size_t output_size() const { return engines_[0]->output_size(); }
    
    //bool layer_wise_profiling() { return engines_[0]->profiler() != nullptr; }
    void reset() {
        for(auto& engine : engines_) {
            engine->reset();
        }
    }
    
    void start() {
        for(auto& engine : engines_) {
            engine->start(request_queue_, response_queue_);
        }
    }
    void stop() {
        // Eninges can exist on these both events - whatever happens first.
        request_queue_.close();
        response_queue_.close();
        for(auto& engine : engines_) {
            engine->stop(); 
        } 
    }
    void join() { for(auto& engine : engines_) { engine->join(); } }
    
    
    void pause() { for(auto& engine : engines_) { engine->pause(); } }
    void resume() { for(auto& engine : engines_) { engine->resume(); } }
};

#endif
