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

#include "core/infer_engine.hpp"


std::ostream &operator<<(std::ostream &os, inference_engine_opts const &opts) {
    os << "[inference_engine_opts ]: " 
       << "model_file=" << opts.model_file_ << ", dtype=" << opts.dtype_ << ", batch_size=" << opts.batch_size_
       << ", num_warmup_batches=" << opts.num_warmup_batches_ << ", num_batches=" << opts.num_batches_
       << ", use_profiler=" << opts.use_profiler_ << ", input_blob_name=" << opts.input_name_
       << ", output_blob_name=" << opts.output_name_ << ", model_id=" << opts.model_id_
       << ", calibrator_cache_path=" << opts.calibrator_cache_path_
       << ", fake inference=" << (opts.fake_inference_ ? "true" : "false");
    return os;
}


void inference_engine::thread_func(abstract_queue<inference_msg*>& request_queue,
                                   abstract_queue<inference_msg*>& response_queue,
                                   inference_engine* engine) {
    engine->do_inference(request_queue, response_queue);
}

void inference_engine::join() {
    if (internal_thread_ && internal_thread_->joinable())
        internal_thread_->join();
}

void inference_engine::start(abstract_queue<inference_msg*>& request_queue, abstract_queue<inference_msg*>& response_queue) {
    internal_thread_ = new std::thread(
        &inference_engine::thread_func,
        std::ref(request_queue),
        std::ref(response_queue),
        this
    );
}


void fake_inference_engine::do_inference(abstract_queue<inference_msg*>& request_queue, abstract_queue<inference_msg*>& response_queue) {
    init_device();
    running_average fetch, submit;
    try {
        timer clock;
        while (!stop_) {
            if (paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            clock.restart();  inference_msg *msg = request_queue.pop();  fetch.update(clock.ms_elapsed());
            msg->set_gpu(engine_id_);
            clock.restart();  response_queue.push(msg);                  submit.update(clock.ms_elapsed());
        }
    } catch(queue_closed) {
    }
    logger_.log_info(fmt(
        "[inference engine %02d/%02d]: {fetch:%.5f}-->--{submit:%.5f}",
        abs(engine_id_), num_engines_, fetch.value(), submit.value()
    ));
}
