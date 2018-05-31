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

#include <boost/program_options.hpp>
#include "core/logger.hpp"
#include "core/utils.hpp"
#include "engines/tensorrt/tensorrt_utils.hpp"

namespace po = boost::program_options;
/**
 * https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
 */
void log_device_info(const int gpu=0);
float benchmark(float *host_mem, float *device_mem, const size_t nbytes, const int niters, cuda_helper& helper);
int main(int argc, char **argv) {
    //
    logger_impl logger(std::cout);
    //
    int gpu(0), size(0), num_warmup_batches(0), num_batches(0);
    bool pinned(false);
    // Parse command line options
    po::options_description opt_desc("Images2Tensors");
    po::variables_map var_map;
    opt_desc.add_options()
        ("help", "Print help message")
        ("gpu", po::value<int>(&gpu)->default_value(0), "GPU index to use.")
        ("num_warmup_batches", po::value<int>(&num_warmup_batches)->default_value(10), "Number of warmup iterations.")
        ("num_batches", po::value<int>(&num_batches)->default_value(50), "Number of benchmark iterations.")
        ("size", po::value<int>(&size)->default_value(10), "Size of a data chunk in MegaBytes.")
        ("pinned",  po::bool_switch(&pinned)->default_value(false), "Use pinned memory");
    try {
        po::store(po::parse_command_line(argc, argv, opt_desc), var_map);
        if (var_map.count("help")) {
            std::cout << opt_desc << std::endl;
            return 0;
        }
        po::notify(var_map);
    } catch(po::error& e) {
        logger.log_warning(e.what());
        std::cout << opt_desc << std::endl;
        logger.log_error("Cannot recover from previous errors");
    }
    //
    std::cout << "GPU ID: " << gpu << std::endl;
    std::cout << "Transfer size (MB) CPU --> GPU: " << size << std::endl;
    std::cout << "Host memory: " << (pinned ? "pinned" : "pageable") << std::endl;
    std::cout << "Number of warmup iterations: " << num_warmup_batches<< std::endl;
    std::cout << "Number of benchmark iterations: " << num_batches<< std::endl;
    //
    cudaCheck(cudaSetDevice(gpu));
    log_device_info();
    const size_t nbytes = size * 1024 * 1024;
    float *device_mem(nullptr), *host_mem(nullptr);
    std::unique_ptr<allocator> alloc(pinned ? (allocator*)new pinned_memory_allocator() : (allocator*)new standard_allocator());
    cuda_helper helper({"start", "stop"},{});
    //
    cudaCheck(cudaMalloc((void**)&device_mem, nbytes));
    alloc->allocate(host_mem, nbytes);
    //
    benchmark(host_mem, device_mem, nbytes, num_warmup_batches, helper);
    const float throughput = benchmark(host_mem, device_mem, nbytes, num_batches, helper);
    std::cout << "Throughput (MB/sec): " << throughput << std::endl;
    //
    cudaCheck(cudaFree(device_mem));
    alloc->deallocate(host_mem);
    return 0;
}

void log_device_info(const int gpu) {
    cudaDeviceProp prop;
    cudaCheck(cudaGetDeviceProperties(&prop, gpu));
    std::cout << "Device: " << prop.name << std::endl;
}

float benchmark(float *host_mem, float *device_mem, const size_t nbytes, const int niters, cuda_helper& helper) {
    float time_ms(0), transfer_mb(nbytes/(1024*1024));
    for (int i=0; i<niters; ++i) {
        cudaCheck(cudaEventRecord(helper.event("start"), 0));
        cudaCheck(cudaMemcpy(device_mem, host_mem, nbytes, cudaMemcpyHostToDevice));
        cudaCheck(cudaEventRecord(helper.event("stop"), 0));
        cudaCheck(cudaEventSynchronize(helper.event("stop")));
        float time;
        cudaCheck(cudaEventElapsedTime(&time, helper.event("start"), helper.event("stop")));
        time_ms += time;
    }
    return 1000.0 * (float(nbytes)/(1024*1024)) * niters / time_ms;
}

/**
 * transfer size
 * num warmup iterations
 * num iterations
 * pinned/non pinned memory
 * aligned/non aligned memory
 * 
 */