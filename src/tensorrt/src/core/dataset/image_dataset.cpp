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

#include "core/dataset/image_dataset.hpp"

void image_dataset::prefetcher_func(image_dataset* myself,
                                    const size_t prefetcher_id, const size_t num_prefetchers) {
    // Find out images I am responsible for.
    sharded_vector<std::string> my_files(myself->file_names_, myself->prefetchers_.size(), prefetcher_id);
    prefetch_msg *msg = new prefetch_msg();
    running_average load, submit;
    try {
        timer clock;
        clock.restart();
        while (!myself->stop_) {
            const auto fname = my_files.next();
            cv::Mat img = cv::imread(fname);
            if (img.data == nullptr) {
                myself->logger_.log_warning("Error loading image from file: " + fname);
                continue;
            }
            msg->images_.push_back(img);
            if (msg->num_images() >= myself->opts_.prefetch_batch_size_) {
                load.update(clock.ms_elapsed());
                clock.restart();  myself->prefetch_queue_.push(msg);  submit.update(clock.ms_elapsed());
                msg = new prefetch_msg();
            }
        }
    } catch(queue_closed) {
    }
    myself->logger_.log_info(fmt(
        "[prefetcher       %02d/%02d]: [load:%.5f]-->--{submit:%.5f}",
        prefetcher_id, num_prefetchers, load.value(), submit.value()
    ));
    delete msg;
}

void image_dataset::decoder_func(image_dataset* myself, const int decoder_id, const int num_decoders) {
    const int height(static_cast<int>(myself->opts_.height_)),
              width(static_cast<int>(myself->opts_.width_));
    const resize_method resizem = myself->opts_.get_resize_method();
    const size_t image_size = static_cast<size_t>(3 * height * width);
    running_average fetch_imgs, fetch_reqs, process, submit;
    try {
        timer clock;
        inference_msg *output(nullptr);      // Current inference request.
        prefetch_msg *input(nullptr  );      // Current non-decoded images.
        size_t input_cursor(0),              // Current position in input data.
               output_cursor(0);             // Current position in output data.
        float decode_time(0);
                              
        while(!myself->stop_) {
            // Get free task from the task pool
            if (!output) {
                clock.restart();
                output = myself->inference_msg_pool_->get();
                fetch_reqs.update(clock.ms_elapsed());
                output_cursor = 0;
            }
            // Get prefetched images
            if (!input) {
                clock.restart();
                input = myself->prefetch_queue_.pop();
                fetch_imgs.update(clock.ms_elapsed());
                input_cursor = 0;
            }
            // If output messages is filled with data, send it
            const auto need_to_decode = output->batch_size() - output_cursor;
            if (need_to_decode == 0) {
                process.update(decode_time);
                clock.restart();
                myself->request_queue_->push(output);
                submit.update(clock.ms_elapsed());
                output = nullptr;
                decode_time = 0;
                continue;
            }
            // If there's no data that needs to be decoded, get it.
            const auto can_decode = input->num_images() - input_cursor;
            if (can_decode == 0) {
                delete input;
                input = nullptr;
                continue;
            }
            // This number of instances I will decode
            const auto will_decode = std::min(need_to_decode, can_decode);
            clock.restart();
            if (!myself->opts_.fake_decoder_) {
                for (size_t i=0; i<will_decode; ++i) {
                    cv::Mat img = input->images_[input_cursor];
                    if (img.rows != height || img.cols != width) {
                        if (resizem == resize_method::resize || img.rows < height || img.cols < width) {
                            cv::resize(input->images_[input_cursor], img, cv::Size(height, width), 0, 0, cv::INTER_LINEAR);
                        } else {
                            img = img(cv::Rect(0, 0, height, width));
                        }
                    }
                    std::copy(
                        img.begin<float>(),
                        img.end<float>(),
                        output->input() + static_cast<std::vector<float>::difference_type>(image_size) * static_cast<std::vector<float>::difference_type>(output_cursor)
                    );
                    input_cursor ++;
                    output_cursor ++;
                }
            } else {
                input_cursor +=will_decode;
                output_cursor +=will_decode;
            }
            decode_time += clock.ms_elapsed();
        }
    } catch(queue_closed) {
    }
    myself->logger_.log_info(fmt(
        "[decoder          %02d/%02d]: {fetch_requet:%.5f}-->--{fetch_images:%.5f}-->--[process:%.5f]-->--{submit:%.5f}",
        decoder_id, num_decoders, fetch_reqs.value(), fetch_imgs.value(), process.value(), submit.value()
    ));
}

image_dataset::image_dataset(const dataset_opts& opts, inference_msg_pool* pool,
                               abstract_queue<inference_msg*>* request_queue, logger_impl& logger) 
: dataset(pool, request_queue), prefetch_queue_(opts.prefetch_queue_size_), opts_(opts), logger_(logger) {
    fs_utils::initialize_dataset(opts_.data_dir_, file_names_);
    if (opts.shuffle_files_) {
        std::random_shuffle(file_names_.begin(), file_names_.end());
    }
    prefetchers_.resize(opts_.num_prefetchers_, nullptr);
    decoders_.resize(opts_.num_decoders_, nullptr);
}
    
image_dataset::~image_dataset() {     
    for (size_t i=0; i<prefetchers_.size(); ++i)
        if (prefetchers_[i]) delete prefetchers_[i];
    for (size_t i=0; i<decoders_.size(); ++i)
        if (decoders_[i]) delete decoders_[i];
}
    
void image_dataset::stop(const bool wait) {
    prefetch_queue_.close();
    dataset::stop(wait);
}
    
void image_dataset::run() {
    // Run prefetch workers
    for (size_t i=0; i<prefetchers_.size(); ++i) {
        prefetchers_[i] = new std::thread(&(image_dataset::prefetcher_func), this, i, prefetchers_.size());
    }
    for (size_t i=0; i<decoders_.size(); ++i) {
        decoders_[i] = new std::thread(&(image_dataset::decoder_func), this, i, decoders_.size());
    }
    // Wait
    for (auto& prefetcher : prefetchers_)
        if (prefetcher->joinable()) prefetcher->join();
    for (auto& decoder : decoders_)
        if (decoder->joinable()) decoder->join();
    // Clean prefetch queue
    std::vector<prefetch_msg*> queue_content;
    prefetch_queue_.empty_queue(queue_content);
    for (size_t i=0; i<queue_content.size(); ++i)
        delete queue_content[i];
}

float image_dataset::benchmark(const std::string dir, const size_t batch_size, const size_t img_size,
                               const size_t num_prefetches, const size_t num_infer_msgs,
                               const int num_warmup_batches, const int num_batches) {
    logger_impl logger;
    dataset_opts opts;
    opts.data_dir_ = dir;
    opts.num_prefetchers_ = num_prefetches;
    opts.prefetch_batch_size_=batch_size;
    opts.num_decoders_ = 1;
    opts.fake_decoder_ = true;
    opts.height_ = img_size;
    opts.width_ = img_size;
    opts.shuffle_files_ = true;
    opts.prefetch_queue_size_ = 6;
        
    standard_allocator alloc;
    inference_msg_pool pool(num_infer_msgs, opts.prefetch_batch_size_, 3*opts.height_*opts.width_, 1000, alloc);
    thread_safe_queue<inference_msg*> request_queue;
    image_dataset data(opts, &pool, &request_queue, logger);
        
    const float throughput = dataset::benchmark(&data, num_warmup_batches, num_batches, logger);
    logger.log_info(fmt("[benchmarks            ]: num_readers=%d, throughput=%.2f", opts.num_prefetchers_, throughput));
    data.join();
    return throughput;
}
