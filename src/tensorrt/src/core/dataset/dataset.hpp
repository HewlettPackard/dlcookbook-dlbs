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
#ifndef DLBS_TENSORRT_BACKEND_CORE_DATASET_DATASET
#define DLBS_TENSORRT_BACKEND_CORE_DATASET_DATASET

#include "core/queues.hpp"
#include "core/infer_msg.hpp"
#include "core/logger.hpp"
#include <thread>
#include <atomic>

/**
 * @brief Resize method.
 * 
 * If an image already has requried shape, no operation is performed. If 'crop'
 * is selected, and an image has smaller size, 'resize' is used instead.
 * Crop is basically a free operation (I think OpenCV just updates matrix header),
 * resize is expensive.
 * Resizing is done on CPUs in one thread. There can be multiple decoders though 
 * decoding different batches in parallel.
 */
enum class resize_method : int {
    crop = 1,    //!< Crop images, if an image has smaller size, resize instead.
    resize = 2   //!< Always resize.
};

/**
 * @brief Options to instantiate a dataset.
 */
struct dataset_opts {
    std::string data_dir_;                 //!< Path to a dataset.
    std::string dtype_ = "float";          //!< Matrix data type in binary files.

    std::string resize_method_ = "crop";   //!< Image resize method - 'crop' or 'resize'.
    
    size_t num_prefetchers_ = 0;           //!< Number of prefetch threads (data readers).
    size_t num_decoders_ = 0;              //!< Number of decoder threads (OpenCV -> std::vector conversion). 
    
    size_t prefetch_batch_size_ = 0;       //!< Same as neural network batch size.
    size_t prefetch_queue_size_ = 0;       //!< Maximal size of prefetched batches.
    
    bool fake_decoder_ = false;            //!< If true, decoder does not decode images but passes inference requests through itself.
    
    size_t height_ = 0;                    //!< This is the target image height. Depends on neural network input.
    size_t width_ = 0;                     //!< This is the target image width. Depends on neural network input.
    
    std::string data_name_ = "images";     //!< Type of input dataset: 'images' for raw images and 'tensors' for preprocesed tensors.
    
    resize_method get_resize_method() const {
        return resize_method_ == "resize"? resize_method::resize : resize_method::crop;
    }
    bool shuffle_files_ = false;           //!< If true, shuffle list of files.
};
std::ostream &operator<<(std::ostream &os, dataset_opts const &opts);


/**
 * @brief Base abstract class for all datasets.
 */
class dataset {
private:
    //!< A data provider can be 'started'. In this case, this is the thread object.
    std::thread *internal_thread_;
    //!< A function that's invoked when internal thread is started. The only
    //!< purpose of this function is to call @see run method that should implement logic.
    static void thread_func(dataset* ds) {
        ds->run();
    }
protected:
    inference_msg_pool* inference_msg_pool_;         //!< [input]  A pool of free tasks that can be reused to submit infer requests.
    abstract_queue<inference_msg*>* request_queue_;  //!< [output] An output data queue containing requests with real data.
    std::atomic_bool stop_;                          //!< The 'stop' flag indicating internal thread must stop.
    // Datasets can run multiple internal threads to prefetch data in parallel. Depending on strategy,
    // they can fail at startup due to absence of input files. The following variables are used to identify
    // such situation and exit.
    int num_threads_;                                //!< Number of worker threads that will do the job.
    std::atomic_int num_live_threads_;               //!< Number of running threads - threads that have data to read.
    std::atomic_int num_dead_threads_;               //!< Number of threads that did not start.
public:
    dataset(inference_msg_pool* pool, abstract_queue<inference_msg*>* request_queue, const int num_threads=1) 
        : inference_msg_pool_(pool), request_queue_(request_queue),
          stop_(false), num_threads_(num_threads), num_live_threads_(0), num_dead_threads_(0) {}
    virtual ~dataset() {
        if (internal_thread_) delete internal_thread_;
    }
    //!< Starts internal thread to load data into the data queue.
    bool start();
    //!< Requests to stop internal thread and returns without waiting.
    virtual void stop(const bool wait=false) {
        stop_ = true;
        if (wait) { join(); }
    }
    //!< Waits for internal thread to shutdown.
    void join() {
        if (internal_thread_ && internal_thread_->joinable())
            internal_thread_->join();
    }
    /**
     * @brief Worker function called from internal thread.
     * 
     * It can create any other number of threads. This function needs to
     * fetch a task structure from task pool, fill it with data and pass it
     * to data queue.
     */
    virtual void run() = 0;
    
    static float benchmark(dataset* ds, const int num_warmup_batches, const int num_batches, logger_impl &logger);
};


/**
 * @brief A very simple implementation of a data provider that just fetches tasks
 * from task pool and passes it immidiately to data queue.
 * Can be used to run inference benchmarks with synthetic data. The data in tasks
 * is randomly initialized and has correct shape for requested neural network.
 */
class synthetic_dataset : public dataset {
public:
    synthetic_dataset(inference_msg_pool* pool, abstract_queue<inference_msg*>* request_queue) 
        : dataset(pool, request_queue, 1) {}
    void run() override;
};

#endif
