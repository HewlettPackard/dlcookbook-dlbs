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
 * @file images2tensors.cpp
 * @brief Convert ImageNet type dataset to custom binary representation for performance benchmarks.
 * @details This tool converts images (JPEGs) to a binary representation that can directly be used
 * by the inference engine. This is a 'benchmark' dataset and it cannot be used in any real use case.
 * The goal is to have a collection of real images and completely eliminate preprocessing by storing
 * images in a format that does not require such a preprocessing.
 * 
 * Images are stored as tensors of shape `[3, Width, Height]` where `Width` and `Height` are always
 * the same (`Size`). Each tensor is an array of length `3*Size*Size`. The tool can create tensors of
 * type `float` or `unsigned char` (`uchar`). For instance, in case of unsigned char data type and 
 * images of shape [3,227,227], each image will be represented as an array of 157587 elements (bytes)
 * or 151KB per image.
 * 
 * Each output file can contain one or more tensors. Such files do not contain any information on data
 * types and exact shapes.
 * 
 * For example:
 * @code{.sh}
 * images2tensors --input_dir=/mnt/imagenet100k/jpegs --output_dir=/mnt/imagenet100k/tensors1 \
 *                --size=227 --dtype=uchar --nthreads=5 --images_per_file=20000
 * @endcode
 * 
 * The tool accepts the following parameters:
 * 1. `--input_dir` Input directory. This directory must exist and must contain images (jpg, jpeg) in 
 *    that directory or one of its sub-directories. ImageNet directory with raw images is one example
 *    of a valid directory structure.
 * 2. `--output_dir` Output directory. The tool will write output files in this directory.
 * 3. `--size` Resize images to this size. Output images will have the following shape [3, size, size].
 * 4. `--dtype` A data type to use. Two types are supported: 'float' and 'uchar'. The 'float' is a
 *    single precision 4 byte numbers. Images take more space but are read directly into an inference
 *    buffer. The 'uchar' (unsigned char) is a one byte numbers that takes less disk space but need to
 *    be converted from unsigned char to float array.
 * 5. `--shuffle` Shuffle list of images. Is used with combination `--nimages` to convert only a small
 *    random subset.
 * 6. `--nimages` If nimages > 0, only convert this number of images. Use `--shuffle` to randomly shuffle
 *    list of images with this option.
 * 7. `--nthreads` Use this number of threads to convert images. This will significantly increase overall
 *    throughput.
 * 8. `--images_per_file` Number of images per output file.
 */

#include "core/logger.hpp"
#include "core/utils.hpp"

#include <boost/program_options.hpp>
#include <thread>

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace po = boost::program_options;

/**
 * @brief Convert a subset of images into a tensor representation. This function runs in its own thread.
 * 
 * @param input_files List of all input files to convert. It's relative to \p input_dir.
 * @param input_dir Input directory containing input files.
 * @param output_dir Output directory. Directory structure will be the same as in \p input_dir.
 * @param num_shards Total number of parallel workers converting a dataset (word size).
 * @param my_shard My index in the list of workers (my rank).
 * @param img_size Spatial dimensions of images (3 * img_size * img_size), i.e. a three channel image
 *                 of square shape.
 * @param logger A thread safe logger.
 * @param images_per_file is the number of images to write into one file.
 * 
 * Output images are always three channel images with the following shape:
 * [Channel, Height, Width] where channel is always 3. Height and Width are always the same (img_size).
 * Channels are RGB. Individual elements are floats (single precision). Pixel values are in range [0, 255].
 */
template<typename T>
void convert(std::vector<std::string>& input_files, const std::string input_dir, const std::string output_dir,
             const size_t num_shards, const size_t my_shard, const size_t img_size, logger_impl& logger,
             const int images_per_file) {
#ifdef HAS_OPENCV
    const size_t channel_size = img_size * img_size;
    std::vector<T> tensor(3 * channel_size);
    sharded_vector<std::string> my_files(input_files, num_shards, my_shard, true);
    const int nchannels = 3;
    
    logger.log_info(fmt("Thread %d: shard_begin=%d, shard_length=%d", my_shard, my_files.shard_begin(), my_files.shard_length()));
    long nprocessed(0);
    timer tm;
    const char pixel_encoding = PictureTool::pixel<T>::encoding;
    
    std::ofstream out;
    int num_images_written(0);
    int num_files_written(0);
    
    while(my_files.has_next()) {
        const std::string file_name = my_files.next();
        cv::Mat img = cv::imread(input_dir + file_name, CV_LOAD_IMAGE_COLOR);
        if (!img.data) {
            logger.log_warning(fmt("Thread %d: bad input file %s", int(my_shard), file_name.c_str()));
            continue;
        }
        if (img.channels() != 3) {
            logger.log_warning(fmt("Thread %d: wrong number of channels (%d). Expecting 3.", int(my_shard), img.channels()));
            continue;
        }
        if (!img.isContinuous()) {
            logger.log_warning(fmt("Thread %d: matrix is not continious in memory, no support for it now.", int(my_shard)));
            continue;
        }
        cv::resize(img, img, cv::Size(img_size, img_size), 0, 0, cv::INTER_LINEAR);
        PictureTool::opencv2tensor<T>((unsigned char*)(img.data), img.channels(), img.rows, img.cols, tensor.data());

        if (!out.is_open()) {
            std::string output_file_name;
            if (images_per_file == 1) {
                output_file_name = output_dir + file_name;
            } else {
                output_file_name = output_dir + fmt("images-%d-%d.tensors", my_shard, num_files_written);
            }
            fs_utils::mk_dir(fs_utils::parent_dir(output_file_name));
            out.open(output_file_name.c_str(), std::ios_base::binary);
            if (!out.is_open()) {
                logger.log_warning(fmt("Thread %d: cannot open file %s", int(my_shard), output_file_name.c_str()));
                continue; // exit?
            }
        }
        // I want to try one read call per image OR one read call per multiple images
        //out.write((const char*)&nchannels, sizeof(int))
        //   .write((const char*)&img_size, sizeof(int))
        //   .write((const char*)&img_size, sizeof(int))
        //   .write((const char*)&pixel_encoding, sizeof(char))
        out.write((const char*)tensor.data(), tensor.size()*sizeof(T));
        
        nprocessed ++;
        num_images_written ++;
        
        if (num_images_written >= images_per_file) {
            out.close();
            num_files_written ++;
            num_images_written = 0;
        }
    }
    if (out.is_open()) {
        out.close();
    }
    const float throughput = 1000.0 * nprocessed / tm.ms_elapsed();
    logger.log_info(fmt("Thread %d: throughput %f images/sec", my_shard, throughput));
#endif
}

/**
 * @brief Application entry point.
 */
int main(int argc, char **argv) {
    logger_impl logger(std::cout);
    std::string input_dir,
                output_dir,
                dtype;
    int size,
        nthreads,
        nimages,
        images_per_file;
    bool shuffle;
#ifndef HAS_OPENCV
    std::cerr << "The images2tensors tool was compiled without OpenCV support and hence cannot load and resize images." << std::endl
              << "It does not support generating artificial datasets for benchmarking purposes. Open a new issue on" << std::endl
              << "GitHub and we will add this functionaity" << std::endl;
    logger.log_error("Can not do anything without OpenCV support.");
#endif
    
    // Parse command line options
    po::options_description opt_desc("Images2Tensors");
    po::variables_map var_map;
    opt_desc.add_options()
        ("help", "Print help message")
        ("input_dir", po::value<std::string>(&input_dir)->required(), 
            "Input directory. This directory must exist and must contain images (jpg, jpeg) in that directory "
            "or one of its sub-directories. ImageNet directory with raw images is one example of a valid directory."
        )
        ("output_dir", po::value<std::string>(&output_dir)->required(),
            "Output directory. Directory that will have exactly the same structure as input directory. Each input "
            "file will get same relative path, will have same name and extension. Even though file extension will "
            "remain the same, the content will be different. It will not be a valid image files."
        )
        ("size", po::value<int>(&size)->default_value(227), 
            "Resize images to this size. Output images will have square shape [3, size, size]."
        )
        ("dtype", po::value<std::string>(&dtype)->required()->default_value("float"),
            "A data type for a matrix storage. Two types are supported: 'float' and 'uchar'. "
            "The 'float' is a single precision 4 byte storage. Images take more space but are read "
            "directly into an inference buffer. The 'uchar' (unsigned char) is a one byte storage "
            "that takes less disk space but needs to be converted from unsigned char to float array."
        )
        ("shuffle",  po::bool_switch(&shuffle)->default_value(false),
            "Shuffle list if images. Usefull with combination --nimages to convert only a small random subset."
        )
        ("nimages", po::value<int>(&nimages)->default_value(0),
            "If nimages > 0, only convert this number of images. Use --shuffle to randomly shuffle list of images "
            "with this option."
        )
        ("nthreads", po::value<int>(&nthreads)->default_value(1),
            "Use this number of threads to convert images. This will significantly increase overall throughput. "
            "On my dev box, single-threaded performance is ~300-400 images/sec."
        )
        ("images_per_file", po::value<int>(&images_per_file)->default_value(1),
            "Number of images per file. If this value is 1, images2tensors will create the same directory structure "
            "with the same file names in --input_dir. If this value is greater than 1, images2tensors will "
            "create a flat directory with *.tensors files."
        );
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

    // Get list of input files
    input_dir = fs_utils::normalize_path(input_dir);
    std::vector<std::string> file_names;
    fs_utils::get_image_files(input_dir, file_names);
    if (file_names.empty())
        logger.log_error(fmt("images2tensors: No input files found in '%s'", input_dir.c_str()));
    logger.log_info(fmt("images2tensors: have found %d images in '%s'", file_names.size(), input_dir.c_str()));
    //
    if (shuffle) {
        logger.log_info("images2tensors: shuffling list of file names");
        std::random_shuffle(file_names.begin(), file_names.end());
    }
    if (nimages > 0 && nimages < file_names.size()) {
        logger.log_info(fmt("images2tensors: Reducing number of images to convert to %d", nimages));
        file_names.resize(nimages);
    }
    logger.log_info(fmt("images2tensors: Number of images per file is %d", images_per_file));
    
    // Convert and write
    output_dir = fs_utils::normalize_path(output_dir);
    std::vector<std::thread*> workers(nthreads, nullptr);
    timer tm;
    for (int i=0; i<nthreads; ++i) {
        if (dtype == "float")
            workers[i] = new std::thread(convert<float>, std::ref(file_names), input_dir, output_dir, nthreads, i, size, std::ref(logger), images_per_file);
        else
            workers[i] = new std::thread(convert<unsigned char>, std::ref(file_names), input_dir, output_dir, nthreads, i, size, std::ref(logger), images_per_file);
    }
    for (int i=0; i<nthreads; ++i) {
        workers[i]->join();
        delete workers[i];
    }
    const float throughput = 1000.0 * file_names.size() / tm.ms_elapsed();
    logger.log_info(fmt("images2tensors: total throughput %f images/sec", throughput));
    return 0;
}
