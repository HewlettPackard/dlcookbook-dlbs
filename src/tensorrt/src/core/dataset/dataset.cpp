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


bool dataset::start() {
    internal_thread_ = new std::thread(&dataset::thread_func, this);
    if (num_threads_ > 0) {
        while (num_dead_threads_ == 0 && num_live_threads_ != num_threads_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }
    return (num_threads_ == -1 || num_live_threads_ == num_threads_);
}


std::ostream &operator<<(std::ostream &os, dataset_opts const &opts) {
    os << "[dataset_opts          ]: "
       << "data_dir=" << opts.data_dir_ << ", resize_method=" << opts.resize_method_ << ", height=" << opts.height_
       << ", width=" << opts.width_ << ", num_prefetchers=" << opts.num_prefetchers_
       << ", num_decoders=" << opts.num_decoders_ << ", prefetch_batch_size=" << opts.prefetch_batch_size_
       << ", prefetch_queue_size=" << opts.prefetch_queue_size_
       << ", fake_decoder=" << (opts.fake_decoder_ ? "true" : "false")
       << ", data_name=" << opts.data_name_;
    return os;
}

float dataset::benchmark(dataset* ds, const int num_warmup_batches, const int num_batches, logger_impl &logger) {
    if (!ds->start()) {
        ds->stop(true);
        return -1;
    }
    // N warmup iterations
    logger.log_info(fmt("[benchmarks            ]: running %d warmup iterations", num_warmup_batches));
    for (int i=0; i<num_warmup_batches; ++i) {
        ds->inference_msg_pool_->release(ds->request_queue_->pop());
    }
    // N benchmark iterations
    logger.log_info(fmt("[benchmarks            ]: running %d benchmark iterations", num_batches));
    running_average fetch;
    timer t, fetch_clock;
    size_t num_images(0);
    inference_msg *msg;
    for (int i=0; i<num_batches; ++i) {
        fetch_clock.restart();
        msg = ds->request_queue_->pop();
        fetch.update(fetch_clock.ms_elapsed());
        num_images += msg->batch_size();
        ds->inference_msg_pool_->release(msg);
    }
    const float throughput = 1000.0 * num_images / t.ms_elapsed();
    ds->stop();
    logger.log_info(fmt("[benchmarks            ]: {fetch:%f}", fetch.value()));
    return throughput;
}

void synthetic_dataset::run() {
    try {
        num_live_threads_ += 1;
        while(!stop_) {
            inference_msg *msg = inference_msg_pool_->get();
            request_queue_->push(msg);
        }
    } catch(queue_closed) {
    }
}
