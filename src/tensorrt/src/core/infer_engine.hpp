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

#ifndef DLBS_TENSORRT_BACKEND_CORE_INFER_ENGINE
#define DLBS_TENSORRT_BACKEND_CORE_INFER_ENGINE

#include "core/infer_msg.hpp"

#include "core/logger.hpp"
#include <atomic>
#include <thread>
//#include <sstream>

/**
 * @brief Options to instantiate an inference engine.
 */
struct inference_engine_opts {
    std::string model_file_;                   //!< Full path to a Caffe's protobuf inference descriptor.
    std::string dtype_ = "float32";            //!< Data type (precision): float32(float), float16, int8.
    size_t batch_size_ = 16;                   //!< Batch size.
    size_t num_warmup_batches_ = 10;           //!< Number of warmup batches.
    size_t num_batches_ = 100;                 //!< Number of benchmark batches.
    bool use_profiler_ = false;                //!< Perform layer-wise model profiling.
    std::string input_name_ = "data";          //!< Name of an input blob.
    std::string output_name_ = "prob";         //!< Name of an output blob.
    
    std::string model_id_;                     //!< Model ID, used to store calibrator cache if data type is int8.
    std::string calibrator_cache_path_ = "";   //!< Path to store calibrator cache if data type is int8.
    
    std::vector<int> gpus_;                    //!< GPUs to use.
    size_t report_frequency_ = 0;              //!< If > 0, report intermidiate progress every this number of batches time GPUs count.
    size_t inference_queue_size_ = 4;          //!< Size of input request queue.
    bool do_not_report_batch_times_ = false;   //!< If true, do not log per-batch performance.
    
    bool fake_inference_ = false;              //!< If true, perform fake inference.
};
std::ostream &operator<<(std::ostream &os, inference_engine_opts const &opts);

/**
 * @brief TensorRT inference engine that works with one GPU.
 * 
 * An engine can operate in two regimes - synchronous and asynchronous. Synchronous
 * regime is useful when users want to submit inference requests on their own. In asynchronous
 * regime, an inference engine runs in separate thread fetching inference request from an input
 * request queue. 
 * 
 * To use multiple inference engines with multiplt GPUs, use mgpu_inference_engine instead.
 * 
 * Synchronous regime (can only be used by one caller thread).
 * @code{.cpp}
 *     inference_engine engine (...);                             // Instantiate inference engine.
 *     inference_msg* task = engine.new_inferene_message(...);    // Create inference task allocating memory for input/output host blobs.
 *     msg->input_ ...;                                           // Fill in input blob
 *     engine.infer(msg);                                         // Run inference. The 'msg' can be reused for subsequent calls.
 *     msg->output_ ...;                                          // Get inference output resutls.
 *     delete msg;                                                // Deallocate memory.
 * @endcode
 * 
 * Asynchronous regime.
 * @code
 *     inference_engine engine (...);                                         // Instantiate inference engine.
 *     abstract_queue<inference_msg*>& request_queue = get_request_queue();   // Create input request queue.
 *     abstract_queue<inference_msg*>& response_queue = get_response_queue(); // Create output response queue.
 *     engine.start(request_queue, response_queue);                           // Run engine in backgroun thread.
 *     while (true) {
 *         inference_msg* msg = response_queue.pop();                         // Fetch inference response
 *         msg->output_ ...;                                                  // Process response.
 *         delete msg;                                                        // If inference message pool is not used, deallocate memory.      
 *     }
 * @endcode
 */
class inference_engine {
protected:
    int engine_id_;           //!< Engine ID, same as GPU ID. Negative ID identifies fake inference engine.
    int num_engines_;         //!< Total number of engines if managed by mGPU inference engine

    logger_impl& logger_;
    time_tracker tm_tracker_;
    
    size_t nbatches_ = 0;
    
    size_t batch_sz_ = 0;
    size_t input_sz_ = 0;
    size_t output_sz_ = 0;
    
    std::atomic_bool stop_;
    std::atomic_bool reset_;
    std::atomic_bool paused_;
 
    std::thread *internal_thread_ = nullptr;
private:
    static void thread_func(abstract_queue<inference_msg*>& request_queue,
                            abstract_queue<inference_msg*>& response_queue,
                            inference_engine* engine);
    
    virtual void init_device() = 0;
    virtual void do_inference(abstract_queue<inference_msg*>& request_queue,
                              abstract_queue<inference_msg*>& response_queue) = 0;
public:
    size_t batch_size() const { return batch_sz_; }
    size_t input_size() const { return input_sz_; }
    size_t output_size() const { return output_sz_; }
    
    int engine_id() const { return engine_id_; }
    time_tracker* get_time_tracker() { return &tm_tracker_; }

    void reset() { reset_ = true; }
    void stop()  { stop_ = true;  }
    void join();
    
    void pause() { paused_ = true; }
    void resume() { paused_ = false; }
    
    /**
     * @brief Create an instance of inference engine. MUST be called from the main thread.
     * 
     * We need to call it from the main due to sevearl reasons:
     *    - To make sure we correctly bind to particular GPU
     *    - To make sure that the very first engine, in case of using int8, will calibrate
     *      model and cache results that will then be resused by other threads.
     */
    inference_engine(const int engine_id, const int num_engines, logger_impl& logger, const inference_engine_opts& opts)
        : engine_id_(engine_id), num_engines_(num_engines), logger_(logger), tm_tracker_(opts.num_batches_),
          batch_sz_(opts.batch_size_), input_sz_(3 * 227 * 227), output_sz_(1000),
          stop_(false), reset_(false), paused_(false) {}
    
    virtual ~inference_engine() {
        if (internal_thread_) {
            delete internal_thread_;
            internal_thread_ = nullptr;
        }
    }
    
    void start(abstract_queue<inference_msg*>& request_queue, abstract_queue<inference_msg*>& response_queue);
};


/**
 * @brief A fake inference engine that does nothing.
 * Can be useful to benchmark various data ingestion components.
 */
class fake_inference_engine : public inference_engine {
private:
    void init_device() override {}
    void do_inference(abstract_queue<inference_msg*>& request_queue,
                      abstract_queue<inference_msg*>& response_queue) override;
public:
    fake_inference_engine(const int engine_id, const int num_engines,
                          logger_impl & logger, const inference_engine_opts& opts)
        : inference_engine(engine_id, num_engines_, logger, opts) {}
};


#endif
