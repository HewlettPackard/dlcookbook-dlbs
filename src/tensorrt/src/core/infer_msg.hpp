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
// https://juanchopanzacpp.wordpress.com/2013/02/26/concurrent-queue-c11/
// https://codereview.stackexchange.com/questions/149676/writing-a-thread-safe-queue-in-c
#ifndef DLBS_TENSORRT_BACKEND_CORE_INFER_MSG
#define DLBS_TENSORRT_BACKEND_CORE_INFER_MSG

#include "utils.hpp"
#include "queues.hpp"

/**
 * @brief A structure that contains input/output data for an inference task. 
 * 
 * It's better to create a pool of these objects and reuse them.
 */
class inference_msg {
private:
    allocator* allocator_ = nullptr;

    host_dtype *input_ = nullptr;   //!< Input data of shape [BatchSize, ...]
    float *output_ = nullptr;       //!< Output data of shape [BatchSize, ...]

    size_t batch_size_ = 0;      //!< Number of instances in this infer message.
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    
    float batch_time_ = 0;       //!< Total batch time including CPU <-> GPU transfer overhead
    float infer_time_ = 0;       //!< Inference time excluding data transfer overhead
    
    int gpu_ = -1;               //!< GPU that processed this task.
public:
    host_dtype* input() { return input_; }
    float* output() { return output_; }
    
    size_t input_size() const { return input_size_; }
    size_t output_size() const { return output_size_; }

    size_t batch_size() const { return batch_size_; }
    float  batch_time() const { return batch_time_; }
    float  infer_time() const { return infer_time_; }
    
    void set_batch_time(const float batch_time) { batch_time_ = batch_time; };
    void set_infer_time(const float infer_time) { infer_time_ = infer_time; };
    void set_gpu(const int gpu) { gpu_ = gpu; }
    /**
     * @brief Construct and initialize inference task.
     * @param batch_size Number of instances in this infer message.
     * @param input_size Number of elements in one input data point
     * @param output_size Number of elements in one output data point
     * @param randomize_input If true, randomly initialize input tensor.
     */
    inference_msg(const size_t batch_size, const size_t input_size, const size_t output_size,
                  allocator& alloc, const bool randomize_input=false) : allocator_(&alloc),
                                                                        batch_size_(batch_size),
                                                                        input_size_(input_size),
                                                                        output_size_(output_size) {
        allocator_->allocate(input_, batch_size_ * input_size_);
        allocator_->allocate(output_, batch_size_ * output_size_);
        if (randomize_input) {
            random_input();
        }
    }
    inference_msg(const inference_msg&) = delete;
    inference_msg(const inference_msg&&) = delete;
    inference_msg& operator=(const inference_msg&) = delete;
    
    virtual ~inference_msg() {
        allocator_->deallocate(input_);
        allocator_->deallocate(output_);
    }
    /**
     * @brief Fill input tensor with random data 
     * uniformly distributed in [0, 1]
     */
    void random_input() { fill_random(input_, batch_size_ * input_size_); }
};


/**
 * @brief Pool of task objects initialized to have correct storage size. 
 * 
 * This is used to not allocate/deallocate memory during benchmarks. To submit new infer
 * request, fetch free task from this pool, initialize with your input data and submit to
 * a data queue. Once results is obtained, release the task by making it avaialble for 
 * subsequent requests.
 */
class inference_msg_pool {
private:
    std::vector<inference_msg*> messages_;             //!< All allocated messages managed by the pool.
    thread_safe_queue<inference_msg*> free_messages_;  //!< Messages that are currently available.
public:
    /**
     * @brief Constructor for a inference message queue.
     * @param count Number of preallocated inference messages.
     * @param batch_size Number of data points in one batch. We need this to pre-allocate memory.
     * @param input_size Number of elements in one data point. We need this to pre-allocate memory.
     * @param output_size Number of elements in one output vector. We need this to pre-allocate memory.
     *                    Usually, it's the same as numebr of neural network outputs (classes).
     */
    inference_msg_pool(const int count, const size_t batch_size, const size_t input_size, const size_t output_size,
                       allocator& alloc, const bool randomize_input=false) {
        for (int i=0; i<count; ++i) {
            inference_msg *msg= new inference_msg(batch_size, input_size, output_size, alloc, randomize_input);
            messages_.push_back(msg);
            free_messages_.push(msg);
        }
    }
    ~inference_msg_pool() {
        destroy();
    }
    void destroy() {
        for (size_t i=0; i<messages_.size(); ++i) {
            delete messages_[i];
        }
        messages_.clear();
    }
    //!< Get new task. The taks may or may not contain data from previous inference.
    //!< You should try to reuse memory allocated for this task.
    inference_msg* get() {
        return free_messages_.pop();
    }
    //!< Make this task object available for subsequent inferences.
    void release(inference_msg *msg) {
        free_messages_.push(msg);
    }
    //!< 'Stop' the pool. After this call, task_pool:get will be returning nullptr.
    void close() {
        free_messages_.close();
    }
};

#endif
