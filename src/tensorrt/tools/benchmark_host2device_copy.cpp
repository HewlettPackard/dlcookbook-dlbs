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

/**
 * @file benchmark_host2device_copy.cpp
 * @brief Benchmarks host-to-device data transfers. 
 * @details Is used to determine achievable PCIe bandwidth in case when there's
 * no overhead associated with ingestion and compute pipelines.
 * @code{.sh}
 * benchmark_host2device_copy --pinned \
 *                            --gpu 0 \
 *                            --num_warmup_batches 50 \
 *                            --num_batches 100 \
 *                            --size 128
 * # where
 * #     --pinned                 Allocate host pinned memory (else memory is pageable).
 * #     --gpu ID                 GPU identifier to use.
 * #     --num_warmup_batches N   Number of warm-up copy transfers.
 * #     --num_batches M          Number of benchamrk copy transfers.
 * #     --size SIZE              Size of a data in megabytes.
 * @endcode
 * 
 * For instance:
 * @code{.sh}
 * ./benchmark_host2device_copy --gpu 0 --pinned --num_warmup_batches 10 --num_batches 100 --size 50
 * {"gpu": 0, "warmup_iters": 10, "bench_iters": 100, "size_mb": 50, "memory": "pinned", "throughput_mb_s": 11285.326172}
 * @endcode
 * 
 * @see https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
 */

#include <boost/program_options.hpp>
#include "core/logger.hpp"
#include "core/utils.hpp"
#include "engines/tensorrt/tensorrt_utils.hpp"

namespace po = boost::program_options;

/**
 * @brief Run series of data copies.
 * @param host_mem is the data pointer in host memory.
 * @param device_mem is the data pointer in device memory.
 * @param length Length of the data arrays.
 * @param niters Number of data transfers to perform.
 * @param helper is the helper to track transfer times using CUDA events.
 * @return achieved copy speed in MB/s (megabyte/second)
 */
float benchmark(unsigned char *host_mem, unsigned char *device_mem,
                const size_t length, const int niters,
                cuda_helper& helper);

/**
 * @brief Application entry point.
 */
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
    cudaCheck(cudaSetDevice(gpu));
    const size_t nbytes = size * 1024 * 1024;
    unsigned char *device_mem(nullptr), *host_mem(nullptr);
    std::unique_ptr<allocator> alloc(pinned ? (allocator*)new pinned_memory_allocator() : (allocator*)new standard_allocator());
    cuda_helper helper({"start", "stop"},{});
    //
    cudaCheck(cudaMalloc((void**)&device_mem, nbytes));
    alloc->allocate(host_mem, nbytes);
    //
    benchmark(host_mem, device_mem, nbytes, num_warmup_batches, helper);
    const float throughput = benchmark(host_mem, device_mem, nbytes, num_batches, helper);
    std::string memory = (pinned ? "pinned" : "pageable");
    std::cout << fmt("{\"gpu\": %d, \"warmup_iters\": %d, \"bench_iters\": %d, \"size_mb\": %d, \"memory\": \"%s\", \"throughput_mb_s\": %f}",
                     gpu, num_warmup_batches, num_batches, size, memory.c_str(), throughput) << std::endl;
    //
    cudaCheck(cudaFree(device_mem));
    alloc->deallocate(host_mem);
    return 0;
}

float benchmark(unsigned char *host_mem, unsigned char *device_mem, const size_t length,
                const int niters, cuda_helper& helper) {
    float time_ms(0);
    for (int i=0; i<niters; ++i) {
        cudaCheck(cudaEventRecord(helper.event("start"), 0));
        cudaCheck(cudaMemcpy(device_mem, host_mem, length, cudaMemcpyHostToDevice));
        cudaCheck(cudaEventRecord(helper.event("stop"), 0));
        cudaCheck(cudaEventSynchronize(helper.event("stop")));
        float time;
        cudaCheck(cudaEventElapsedTime(&time, helper.event("start"), helper.event("stop")));
        time_ms += time;
    }
    // Return achieved MB/s
    return 1000.0 * (float(length)/(1024*1024)) * niters / time_ms;
}
