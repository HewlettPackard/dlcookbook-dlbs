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
 * @file benchmark_tensor_dataset.cpp
 * @brief Benchmarks tensor dataset. 
 * @details Is used to determine achievable throughput assuming there's no overhead associated 
 * with compute. This is not a true IO benchmark tool, rather, a tool to identify whether a 
 * storage/network or this software benchmark backend is a bottleneck in each particular case.
 * 
 * For example:
 * @code{.sh}
 * benchmark_tensor_dataset --data_dir=/mnt/imagenet/uchar227 --batch_size=512 \
 *                          --dtype=uchar --img_size=227 --num_prefetchers=3 \
 *                          --prefetch_pool_size=9 --num_warmup_batches=1000 \
 *                          --num_batches=5000
 * @endcode
 * 
 * The tool accepts the following parameters:
 * 1. `--data_dir` Path to a dataset to use.
 * 2. `--batch_size` Create batches of this size.
 * 3. `--img_size` Size of images in a dataset (width = height).
 * 4. `--num_prefetchers` Number of prefetchers (data readers).
 * 5. `--prefetch_pool_size` Number of pre-allocated batches. Memory for batches is
 *    preallocated in advance and then reused by prefetchers.
 * 6. `--num_warmup_batches` Number of warmup iterations.
 * 7. `--num_batches` Number of benchmark iterations.
 * 8. `--dtype` Tensor data type in the dataset- 'float' or 'uchar'.
 */
#include "core/dataset/tensor_dataset.hpp"
#include <boost/program_options.hpp>
namespace po = boost::program_options;

/**
 * @brief Application entry point.
 */
int main(int argc, char **argv) {
    //
    logger_impl logger;
    std::string data_dir, dtype;
    int batch_size, img_size,
        num_prefetchers, prefetch_pool_size,
        num_warmup_batches, num_batches;
    // Parse command line options
    po::options_description opt_desc("Benchmark Tensor Datasets");
    po::variables_map var_map;
    opt_desc.add_options()
        ("help", "Print help message")
        ("data_dir", po::value<std::string>(&data_dir)->required(), "Path to a dataset to use.")
        ("batch_size", po::value<int>(&batch_size)->default_value(512), "Batch size.")
        ("img_size", po::value<int>(&img_size)->default_value(227), "Size of images in a dataset (width = height).")
        ("num_prefetchers", po::value<int>(&num_prefetchers)->default_value(1), "Number of prefetchers (readers).")
        ("prefetch_pool_size", po::value<int>(&prefetch_pool_size)->default_value(2), "Number of pre-allocated batches.")
        ("num_warmup_batches", po::value<int>(&num_warmup_batches)->default_value(10), "Number of warmup iterations.")
        ("num_batches", po::value<int>(&num_batches)->default_value(50), "Number of benchmark iterations.")
        ("dtype", po::value<std::string>(&dtype)->default_value("float"), "Tensor data type - 'float' or 'uchar'.");
    //
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
    logger.log_info(fmt(
        "[benchmarks            ]: data_dir=%s, batch_size=%d, img_size=%d, num_prefetchers=%d, prefetch_pool_size=%d, num_warmup_batches=%d, num_batches=%d, dtype=%s",
        data_dir.c_str(), batch_size, img_size, num_prefetchers, prefetch_pool_size, num_warmup_batches, num_batches, dtype.c_str()
    ));
    const auto images_sec = tensor_dataset::benchmark(
        data_dir, batch_size, img_size, num_prefetchers, prefetch_pool_size,
        num_warmup_batches, num_batches, dtype
    );
    if (images_sec >= 0) {
        // 3 channels times number of elements per channel times element size in bytes devided by bytes in megabyte
        const float img_mb = float(3*img_size*img_size)*(dtype == "float" ? 4 : 1) / (1024*1024);
        const float mb_sec = images_sec * img_mb;
        logger.log_info(fmt("[benchmarks            ]: images/sec=%f, MB/sec=%f", images_sec, mb_sec));
    } else {
        logger.log_warning("[benchmarks            ]: Benchmark returned error code indicating there was an error.");
    }
}
