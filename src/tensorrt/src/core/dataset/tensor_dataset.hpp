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

#include "core/dataset/dataset.hpp"
#include "core/logger.hpp"

#ifndef DLBS_TENSORRT_BACKEND_CORE_DATASET_TENSOR_DATASET
#define DLBS_TENSORRT_BACKEND_CORE_DATASET_TENSOR_DATASET

/**
 * @brief An implementation of a dataset that streams images from binary files of custom format.
 * 
 * In order to put as much stress as possible on storage (good for storage/network benchmarks)
 * we need to make sure that other components of an ingestion pipeline are not bottlenecks. In
 * particular, usually CPUs are responsible for preprocessing input data, such as, resizing
 * images. That's why we use this simple binary representation to completely elliminate any other
 * possible bottlenecks.
 * In particular, binary files contain images that are stored as plain C-style arrays (3D tensors)
 * of shape [NumChannels, Width, Height] where Width == Height and NumChannels = 3. Tensor types
 * could be either '`float` or `unsigned char` (specified by a user). All images have the same
 * shape and user needs to provide size on a command line. 
 * In general, in current imlementation we do not really use `images`, though DLBS does provide image
 * converter from raw JPG format to this one that we call `tensors1` (unsigned char data type) or
 * `tensors4` (float data type). Prefetchers just read NumChannels*Width*Height*sizeof(TensorType)
 * bytes from files and consider that to be image data. Thus, users can create large binary files
 * with random data and this tool should work just fine. In future releases though we may reconsider
 * this and start treat data in this binary files as real images.
 * 
 * Implementation is quite straighforward. There are N readers (prefetchers). Each prefetcher reads
 * N images from binary files and batches them together. Then it sends this data straight into inference
 * queue so that inference engines can fetch this data and run inference on it. This basically means
 * that the batch size provided by a used used by these readers. This is different from image dataset
 * that can use different batch sizes to read data and to form inference requests.
 * 
 * If host data type is float and images are stored with unsigned chars, each reader will cast unsigned
 * char array to float array before sending inference requests. This takes some time. If host data type
 * is unsigned char and so is the image type, then no conversion is made and images are sent to GPUs as
 * unsigned char array. GPUs will then cast these arrays to float type.
 * 
 */
class tensor_dataset : public dataset {
private:
    std::vector<std::string> file_names_;
    std::vector<std::thread*> prefetchers_;
    dataset_opts opts_;
    logger_impl& logger_;
private:
    static void prefetcher_func(tensor_dataset* myself, const size_t prefetcher_id, const size_t num_prefetchers);
public:
    tensor_dataset(const dataset_opts& opts, inference_msg_pool* pool,
                   abstract_queue<inference_msg*>* request_queue, logger_impl& logger);
    void run() override;
    static float benchmark(const std::string dir, const size_t batch_size=512, const size_t img_size=227,
                           const size_t num_prefetches=4, const size_t num_infer_msgs=10,
                           const int num_warmup_batches=10, const int num_batches=100,
                           const std::string& dtype="float");
};


#endif
