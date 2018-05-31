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
