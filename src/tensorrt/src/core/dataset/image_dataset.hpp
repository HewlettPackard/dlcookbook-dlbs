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
#ifndef DLBS_TENSORRT_BACKEND_CORE_DATASET_IMAGE_DATASET
#define DLBS_TENSORRT_BACKEND_CORE_DATASET_IMAGE_DATASET

#include "core/dataset/dataset.hpp"
#include "core/logger.hpp"
#include <opencv2/opencv.hpp>

/**
 * @brief A message containing images that have been read from some storage. 
 * 
 * An image reader batches images before sending them. Usually, there
 * can be multiple image readers working in parallel threads. This is internal
 * implementation of this particular dataset.
 */
struct prefetch_msg {
    std::vector<cv::Mat> images_;
    size_t num_images() const { return images_.size(); }
};

/**
 * @brief An implementation of a dataset that reads raw images.
 * 
 * It is not really efficient and thus should not be used. The implementation
 * works like this. It creates N prefetch threads. Each prefetch thread reads
 * raw images and batches them together (prefetch_msg) until batch reaches
 * specified size. Prefetcher then sends this message to decoders.
 * 
 * There are M decoders in the system (all part of this particular dataset
 * implementation). Each decoder decodes OpenCV matrix into C-style 3D tensor
 * and re-batches them into batches of, probably, new size. Then it sends this
 * data (inference request) into an output queue. Inference request data will
 * then be consumed by inference engines.
 * 
 */
class image_dataset : public dataset {
private:
    std::vector<std::string> file_names_;    //!< File names of all images. Each prefetcher will use its own shard.
    std::vector<std::thread*> prefetchers_;  //!< Prefetchers reading raw images into OpenCV objects.
    std::vector<std::thread*> decoders_;     //!< Decoders converting OpenCV objects into C-style 3D tensors.
    
    thread_safe_queue<prefetch_msg*> prefetch_queue_; //!< An internal queue used by prefetchers to communicated with decoders.
    dataset_opts opts_;
    logger_impl& logger_;
private:
    static void prefetcher_func(image_dataset* myself, const size_t prefetcher_id, const size_t num_prefetchers);
    static void decoder_func(image_dataset* myself, const int decoder_id, const int num_decoders);
public:
    thread_safe_queue<prefetch_msg*>& prefetch_queue() { return prefetch_queue_; }
    image_dataset(const dataset_opts& opts, inference_msg_pool* pool,
                   abstract_queue<inference_msg*>* request_queue, logger_impl& logger);
    ~image_dataset();
    void stop(const bool wait=false) override;
    void run() override;
    
    static float benchmark(const std::string dir, const size_t batch_size=512, const size_t img_size=227,
                           const size_t num_prefetches=4, const size_t num_infer_msgs=10,
                           const int num_warmup_batches=10, const int num_batches=100);
};

#endif
