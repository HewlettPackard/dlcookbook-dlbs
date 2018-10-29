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


#include "core/utils.hpp"

#define _XOPEN_SOURCE 700
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <thread>
#include <sstream>
#include <algorithm>
#include <regex>

data_type::data_type(const std::string &str_type) {
    if (str_type == "fp32") { 
        type_ = dt_fp32; 
    } else if (str_type == "uint8") {
        type_ = dt_uint8;
    } else {
        throw std::invalid_argument(
            fmt("Invalid data type (%s). Must be one of ('fp32', 'uint8')", str_type.c_str())
        );
    }
}

void fill_random(float *vec, const size_t sz) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine(rnd_device());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  auto gen = std::bind(dist, mersenne_engine);
  std::generate(vec, vec+sz, gen);
}

void fill_random(unsigned char *vec, const size_t sz) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine(rnd_device());
  std::uniform_int_distribution<unsigned int> dist(0, 255);
  auto gen = std::bind(dist, mersenne_engine);
  std::generate(vec, vec+sz, gen);
}

template<>
std::string S<bool>(const bool &t) { return (t ? "true" : "false"); }

template<>
std::string from_string<std::string>(const char* const val) { return std::string(val); }

template<>
int from_string<int>(const char* const val) { return std::stoi(val); }

template<>
bool from_string<bool>(const char* const val) { 
    std::string str = std::string(val);
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return (str == "1" || str == "on" || str == "yes" || str == "true");
}

void rtrim_inplace(std::string& s, const std::string &sep) { s.erase(s.find_last_not_of(sep)+1); }
std::string rtrim(const std::string& s, const std::string &sep) {
   std::string copy = s;
   rtrim_inplace(copy, sep);
   return copy;
}

std::string environment::file_reader() {
    return environment::variable<std::string>("DLBS_TENSORRT_FILE_READER", "directio");
}
bool environment::remove_files_from_os_cache() {
    return environment::variable<bool>("DLBS_TENSORRT_REMOVE_FILES_FROM_OS_CACHE", "yes");
}

bool environment::pinned_memory() {
    return environment::variable<bool>("DLBS_TENSORRT_ALLOCATE_PINNED_MEMORY", "yes");
}

bool environment::allow_image_dataset() {
    return environment::variable<bool>("DLBS_TENSORRT_ALLOW_IMAGE_DATASET", "no");
}

std::string environment::synch_benchmarks() {
    return environment::variable<std::string>("DLBS_TENSORRT_SYNCH_BENCHMARKS", "");
}

int environment::storage_block_size() {
    return environment::variable<int>("DLBS_TENSORRT_STORAGE_BLOCK_SIZE", 512);
}

std::string environment::dataset_split() {
    return environment::variable<std::string>("DLBS_TENSORRT_DATASET_SPLIT", "uniform");
}

bool environment::overlap_copy_compute() {
    return environment::variable<bool>("DLBS_TENSORRT_OVERLAP_COPY_COMPUTE", "yes");
}

std::string environment::inference_impl_ver() {
    return variable<std::string>("DLBS_TENSORRT_INFERENCE_IMPL_VER", "");
}

std::unordered_map<std::string, std::string> environment::hdfs_params() {
    std::string unparsed = environment::variable<std::string>("DLBS_TENSORRT_HDFS_PARAMS", "");
    std::unordered_map<std::string, std::string> params;
    if (unparsed.empty()) return params;

    auto _add_param = [&params](const std::string &param) {
        std::size_t sep_idx = param.find_first_of("=");
        if (sep_idx != std::string::npos) {
            params[param.substr(0, sep_idx)] = param.substr(sep_idx+1);
        }
    };

    std::size_t prev_idx = 0,
                curr_idx = unparsed.find_first_of(",");
    while (curr_idx != std::string::npos) {
        _add_param(unparsed.substr(prev_idx, curr_idx-prev_idx));
        prev_idx = curr_idx + 1;
        curr_idx = unparsed.find_first_of(",", prev_idx);
    }
    _add_param(unparsed.substr(prev_idx, curr_idx-prev_idx));
    return params;
}


/**
 *   We are going to use the following regexp:
 *      ^(?:([a-z]+):\/\/)?(?:([a-z]+)(?::([0-9]+))?)?(\/.*)?$
 *   This is how it works:
 *      ^                                     // Begin of string
 *      (?:([a-z]+):\/\/)?                    // Optional scheme (group 1)
 *      (?:([a-z0-9]+)(?::([0-9]+))?)?        // For HDFS, optional namenode (group 2) and port (group 3)
 *      (\/.*)?                               // Optional path (group 4)
 *      $                                     // End of string
 *   The following groups are returned:
 *      group 1: scheme (file, hdfs)
 *      group 2: namenode for hdfs
 *      group 3: port for hdfs
 *      group 4: URL
 */
url::url(const std::string &str) {
    std::regex url_regex("^(?:([a-z]+):\/\/)?(?:([a-z0-9]+)(?::([0-9]+))?)?(\/.*)?$");
    std::cmatch match;
    good_ = std::regex_match(str.c_str(), match, url_regex);
    if (good_) {
        if (!match.str(1).empty()) scheme_ = match.str(1);
        if (!match.str(2).empty()) hdfs_namenode_ = match.str(2);
        if (!match.str(3).empty()) hdfs_port_ = atoi(match.str(3).c_str());
        if (!match.str(4).empty()) path_ = match.str(4);
    }
}


std::ostream& operator<<(std::ostream &out, const running_average &ra) {
    std::cout << "running_average{size=" << ra.num_steps() << ", value=" << ra.value() << "}";
    return out;
}


stats::stats(const std::vector<float>& nums) {
    if (nums.empty()) return;
    double prev_mean(0), prev_s(0), cur_s(0);
    prev_mean = mean_ = min_ = max_ = static_cast<double>(nums[0]);
    for (size_t i=1; i<nums.size(); ++i) {
        const auto v = static_cast<double>(nums[i]);
        //
        mean_ = prev_mean + (v - prev_mean) / (i+1);
        cur_s = prev_s + (v - prev_mean) * (v - mean_);
        //
        min_ = std::min(min_, v);
        max_ = std::max(max_, v);
        //
        prev_mean = mean_;
        prev_s = cur_s;
    }
    variance_ = nums.size() > 1 ? cur_s / (nums.size() - 1) : 0.0;
}

std::ostream& operator<<(std::ostream &out, const stats &s) {
    std::cout << "stats{min=" << s.min() << ", max=" << s.max() << ", mean=" << s.mean() << ", variance=" << s.variance() << ", stdev=" << s.stdev() << "}";
    return out;
}

template<typename T>
void PictureTool::opencv2tensor(unsigned char* opencv_data, const int nchannels, const int height,
                                const int width, T* tensor) {
    const int channel_size = height * width;
    for(int j = 0; j < height; ++j) {
        const int row_idx = nchannels * width * j;
        for(int i = 0; i < width; ++i) {
            const auto col_rel_idx = i * nchannels;
            const int b = opencv_data[row_idx + col_rel_idx] ;            // b
            const int g = opencv_data[row_idx + col_rel_idx + 1];         // g
            const int r = opencv_data[row_idx + col_rel_idx + 2];         // r
            // [RGB, H, W]
            tensor[width * j + i                              ] = r;
            tensor[width * j + i + channel_size               ] = g;
            tensor[width * j + i + channel_size + channel_size] = b;
        }
    }
}
template void PictureTool::opencv2tensor<float>(unsigned char* opencv_data, const int nchannels, const int height, const int width, float* tensor);
template void PictureTool::opencv2tensor<unsigned char>(unsigned char* opencv_data, const int nchannels, const int height, const int width, unsigned char* tensor);

// ---------------------------------------------------------------------------

process_barrier::process_barrier(std::string specs) : post_mode_(true) {
    std::replace(specs.begin(), specs.end(), ',', ' ');
    std::istringstream is(specs);
    is >> rank_ >> count_ >> name_;
    name_ = "/" + name_;
    init();
}

process_barrier::process_barrier(const std::string& name, const int rank, const int count) 
: name_(name), rank_(rank), count_(count), post_mode_(true) {
    init();
}

void process_barrier::init() {
    if (rank_ == 0) {
        handle_ = sem_open(name_.c_str(), O_CREAT|O_EXCL, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH, 0);
    } else {
        while (true) {
            handle_ = sem_open(name_.c_str(), 0);
            if (handle_ != SEM_FAILED || errno != ENOENT)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

int process_barrier::wait() {
    if (handle_ == SEM_FAILED) {
        return -1;
    }
    int target_value(0);
    if (post_mode_) {
        if (sem_post(handle_) < 0)
            return -1;
        target_value = count_;
    } else {
        if (sem_wait(handle_) < 0)
            return -1;
    }
    int value(-1);
    while (value != target_value) {
        if (sem_getvalue(handle_, &value) < 0)
            return -1;
    }
    post_mode_ = !post_mode_;
    return 0;
}

void process_barrier::close() {
    if (handle_ != SEM_FAILED) {
        if (rank_ == 0) {
            sem_unlink(name_.c_str());
        }
        sem_close(handle_);
        handle_ = SEM_FAILED;
    }
}
    
    
