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
   Completely based on giexec.cpp example in /usr/src/gie_samples/samples/giexec/
 */

#include <signal.h>
#include <execinfo.h>
#include "core/infer_engine.hpp"
#include "core/dataset/dataset.hpp"
#include "core/dataset/image_dataset.hpp"
#include "core/dataset/tensor_dataset.hpp"
#include "engines/mgpu_engine.hpp"
#include <boost/program_options.hpp>

/**
 *  @brief An inference benchmark based on NVIDIA's TensorRT library.
 */

namespace po = boost::program_options;

void parse_command_line(int argc, char **argv,
                        po::options_description opt_desc, po::variables_map& var_map,
                        inference_engine_opts& engine_opts, dataset_opts& data_opts,
                        logger_impl& logger);
void print_file_reader_warnings(logger_impl& logger, const std::string& me);

void segfault_sigaction(int signal, siginfo_t *si, void *arg) {
    void *array[10];
  size_t size;

  // get void*'s for all entries on the stack
  size = backtrace(array, 10);

  // print out all the frames to stderr
  fprintf(stderr, "Error: signal %d:\n", signal);
  backtrace_symbols_fd(array, size, STDERR_FILENO);
  exit(1);
}

int main(int argc, char **argv) {
    // Set SIGSEGV handler
    struct sigaction sa;
    memset(&sa, 0, sizeof(struct sigaction));
    sigemptyset(&sa.sa_mask);
    sa.sa_sigaction = segfault_sigaction;
    sa.sa_flags   = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, NULL);
    // Create one global logger.
    logger_impl logger(std::cout);
    // Parse command line arguments
    inference_engine_opts engine_opts;
    dataset_opts data_opts;
    try {
        po::options_description opt_desc("Options");
        po::variables_map var_map;
        parse_command_line(argc, argv, opt_desc, var_map, engine_opts, data_opts, logger);
        if (var_map.count("version")) {
            std::cout << NV_GIE_MAJOR << "." << NV_GIE_MINOR << "." << NV_GIE_PATCH << std::endl;
            return 0;
        }
        if (var_map.count("help")) { 
            std::cout << "HPE Deep Learning Benchmarking Suite - TensorRT backend" << std::endl
                      << opt_desc << std::endl
                      << "TensorRT version " << NV_GIE_MAJOR << "." << NV_GIE_MINOR << "." << NV_GIE_PATCH << std::endl;
            return 0;
        }
    } catch(po::error& e) {
        logger.log_error(e.what());
        return 1;
    }
    const std::string me = "[main                  ]: ";
    //
#ifdef DEBUG_LOG
    logger.log_warning(me + "DEBUG logging is enabled. For real performance tests this should be disabled (recompile without -DDEBUG_LOG)");
#endif
#if defined HOST_DTYPE_SP32
    logger.log_info(me + "Input data will be stored in host memory as tensor<float> tensors (reason: compiled with -DHOST_DTYPE_SP32).");
#elif defined HOST_DTYPE_INT8
    logger.log_info(me + "Input data will be stored in host memory as tensor<unsigned char> tensors (reason: compiled with -DHOST_DTYPE_INT8).");
#else
    logger.log_error(me + "Input data will be stored in host memory as tensor<?> tensors. Recompile with -DHOST_DTYPE_SP32 or -DHOST_DTYPE_INT8.");
#endif
    //
    logger.log_info(engine_opts);
    logger.log_info(data_opts);
    //
    allocator *alloc(nullptr);
    if (environment::pinned_memory()) {
        logger.log_info(me + "Creating pinned memory allocator.");
        alloc = new pinned_memory_allocator();
    } else {
        logger.log_info(me + "Creating standard memory allocator.");
        alloc = new standard_allocator();
    }
    // Create pool of inference engines. All inference engines will be listening to data queue
    // for new inference requests. All engines will be exactly the same - model, batch size etc.
    // There will be a 1:1 mapping between GPU and inference engines.
    logger.log_info(me + "Creating multi-GPU inference engine");
    mgpu_inference_engine engine(engine_opts, logger);
    const size_t num_engines = engine.num_engines();
    // Create pool of available task request objects. These objects (infer_task) will be initialized
    // to store input/output tensors so there will be no need to do memory allocations during benchmark.
    const float est_mp_mem = static_cast<float>(engine_opts.inference_queue_size_*(8+4+4+4+engine.batch_size()*(engine.input_size()+engine.output_size()))) / (1024*1024);
    logger.log_info(me + "Creating inference message pool with " + S(engine_opts.inference_queue_size_) + " messages, estimated memory is " + std::to_string(est_mp_mem) + " mb.");
    inference_msg_pool infer_msg_pool(engine_opts.inference_queue_size_, engine.batch_size(), engine.input_size(), engine.output_size(), *alloc, true);
    // Create data provider. The data provider will spawn at least one thread. It will fetch free task objects
    // from pool of task objects, will populate them with data and will submit tasks to data queue. All
    // preprocessing logic needs to be implemented in data provider.
    dataset* data(nullptr);
    if (data_opts.data_name_ == "synthetic" || data_opts.data_dir_ == "") {
        data = new synthetic_dataset(&infer_msg_pool, engine.request_queue());
        logger.log_info(me + "Will use 'synthetic' data set");
    } else {
        logger.log_info(me + "Will use real data set (" + data_opts.data_dir_ + ")");
        logger.log_warning(me + "Computing resize dimensions assuming input data has shape [BatchSize, 3, H, W] where H == W.");
        data_opts.height_ = data_opts.width_ = std::sqrt(engine.input_size() / 3);
        if (data_opts.data_name_ == "images") {
            if (!environment::allow_image_dataset()) {
                logger.log_warning(me + "Image dataset is disabled by default due to serious performance issues.");
                logger.log_error(me + "If you really want to use it, provide the following env variable: DLBS_TENSORRT_ALLOW_IMAGE_DATASET=yes");
            }
            logger.log_warning(me + "Will use 'images' data set (found DLBS_TENSORRT_ALLOW_IMAGE_DATASET env variable). Expect bad performance due to unoptimized ingestion pipeline.");
            data = new image_dataset(data_opts, &infer_msg_pool, engine.request_queue(), logger);
        } else {
            if (data_opts.data_name_ == "tensors1") {
                logger.log_info(me + "Will use 'tensors1' data set");
                data_opts.dtype_ = "uchar";
            } else if (data_opts.data_name_ == "tensors4") {
                logger.log_info(me + "Will use 'tensors4' data set");
                data_opts.dtype_ = "float";
            } else {
                logger.log_error(me + "Invalid input dataset (" + data_opts.data_name_ + ")");
            }
            print_file_reader_warnings(logger, me);
            data = new tensor_dataset(data_opts, &infer_msg_pool, engine.request_queue(), logger);
        }
    }
    logger.log_info(me + "Starting dataset threads");
    if (!data->start()) {
        data->stop(true);
        infer_msg_pool.destroy();
        delete data;
        delete alloc;
        std::cout << "__exp.status__=\"failure\"" << std::endl;
        std::cout << "__exp.status_msg__=\"Some of the dataset threads failed to start (no data or not enough data)."
                  <<" Check path (" << data_opts.data_dir_ << ") and/or make sure number of files >= number of prefetchers.\"" << std::endl;
        logger.log_error("[main                  ]: Some of the dataset threads failed to start. Aborting.");
    }
    // Start pool of inference engines. This will start one thread per engine. Individual inference engines
    // will be fetching data from data queue, will be doing inference and will be submitting same task request
    // objects with inference results and statistics to decision queue.
    logger.log_info(me + "Starting engine threads");
    engine.start();

    logger.log_info(me + "Running warmup iterations");
    for (size_t i=0; i<engine_opts.num_warmup_batches_; ++i) {
        // Do warmup iterations. Just fetch inference results from decision queue
        // and put them back to pool of free task objects. All data preprocessing,
        // submission and classification are done in backgroud threads.
        for (size_t j=0; j<num_engines; ++j) {
            inference_msg *msg = engine.response_queue()->pop();
            infer_msg_pool.release(msg);
        }
    }
    // This reset will not happen immidiately, but next time an engine processes a batch.
    // So, they may reset their states at slightly different moments.
    engine.reset();
    // Sync with othet processes if need to do so
    process_barrier* barrier(nullptr);
    timer synch_timer;
    if (!environment::synch_benchmarks().empty()) {
        engine.pause();
        barrier = new process_barrier(environment::synch_benchmarks());
        logger.log_info(fmt("%sSynching with other processes (%d/%d)", me.c_str(), barrier->rank(), barrier->count()));
        barrier->wait();
        engine.resume();
    }
    // Run benchmarks
    logger.log_info(me + "Running benchmarks");
    time_tracker tm_tracker(engine_opts.num_batches_);
    long num_processed_instances(0);    
    for (size_t i=0; i<engine_opts.num_batches_; ++i) {
        tm_tracker.batch_started();
        for (size_t j=0; j<num_engines; ++j) {
            inference_msg *msg = engine.response_queue()->pop();
            num_processed_instances += msg->batch_size();
            infer_msg_pool.release(msg);
        }
        tm_tracker.batch_done();
        if (engine_opts.report_frequency_ > 0 && i>0 && i%engine_opts.report_frequency_ == 0) {
            logger.log_progress(tm_tracker.get_batch_times(), tm_tracker.get_iter_idx(), engine_opts.batch_size_, "total_");
            tm_tracker.new_iteration();
        }
    }
    if (barrier) {
        logger.log_info(fmt("%sSynching with other processes (%d/%d)", me.c_str(), barrier->rank(), barrier->count()));
        barrier->wait();
        logger.log_key_value(
            "results.mgpu_effective_throughput",
            1000.0 * num_processed_instances  / synch_timer.ms_elapsed()
        );
        barrier->close();
        delete barrier;
        barrier = nullptr;
    }
    // Shutdown everything and wait for all threads to exit.
    logger.log_info(me + "Stopping and joining threads");
    data->stop();  engine.stop();  infer_msg_pool.close();
    logger.log_info(me + "Waiting for data provider ...");
    data->join();
    logger.log_info(me + "Waiting for inference engine ...");
    engine.join();
    infer_msg_pool.destroy();        // This is not very good. It must be destroyed before we delete allocator.
    delete  data;  data = nullptr;
    delete alloc;  alloc = nullptr;
    // Log final results.
    logger.log_info(me + "Reporting results for each inference engine ...");
    logger.log_info(me + "  |-> inference time (*_infer_*) is a compute time without host<->device transfers");
    logger.log_info(me + "  |-> batch time (*_batch_*) is a real inference time including host<->device transfers");
    logger.log_info(me + "        |-> if engines overlap copy/compute, observable (*_total*_) throughput can be better");
    for (size_t i=0; i<num_engines; ++i) {
        time_tracker *tracker = engine.engine(i)->get_time_tracker();
        const std::string gpu_id = std::to_string(engine.engine(i)->engine_id());
        logger.log_final_results(tracker->get_infer_times(), engine_opts.batch_size_, "gpu_" + gpu_id + "_infer_", !engine_opts.do_not_report_batch_times_);
        logger.log_final_results(tracker->get_batch_times(), engine_opts.batch_size_, "gpu_" + gpu_id + "_batch_", !engine_opts.do_not_report_batch_times_);
    }
    logger.log_info(me + "Reporting results for all inference engines ...");
    logger.log_info(me + "  |-> times and throughput reported below are 'weak-scaling' times i.e.");
    logger.log_info(me + "      batch time is the time to process BATCH_SIZE*NUM_ENGINES images");
    logger.log_final_results(tm_tracker.get_batch_times(), engine_opts.batch_size_*num_engines, "total_", false);
    logger.log_final_results(tm_tracker.get_batch_times(), engine_opts.batch_size_*num_engines, "", !engine_opts.do_not_report_batch_times_);
    return 0;
}

void parse_command_line(int argc, char **argv,
                        boost::program_options::options_description opt_desc, po::variables_map& var_map,
                        inference_engine_opts& engine_opts, dataset_opts& data_opts,
                        logger_impl& logger) {
    namespace po = boost::program_options;
    std::string gpus;
    int batch_size(static_cast<int>(engine_opts.batch_size_)),
        num_warmup_batches(static_cast<int>(engine_opts.num_warmup_batches_)),
        num_batches(static_cast<int>(engine_opts.num_batches_)),
        report_frequency(static_cast<int>(engine_opts.report_frequency_)),
        inference_queue_size(static_cast<int>(engine_opts.inference_queue_size_));
    int num_prefetchers(static_cast<int>(data_opts.num_prefetchers_)),
        num_decoders(static_cast<int>(data_opts.num_decoders_)),
        prefetch_queue_size(static_cast<int>(data_opts.prefetch_queue_size_)),
        prefetch_batch_size(static_cast<int>(data_opts.prefetch_batch_size_));

    opt_desc.add_options()
        ("help", "Print help message")
        ("version", "Print version")
        ("gpus", po::value<std::string>(&gpus), "A comma seperated list of GPU identifiers to use.")
        ("model", po::value<std::string>(&engine_opts.model_id_), "Model identifier like alexnet, resent18 etc. Used to store calibration caches.")
        ("model_file", po::value<std::string>(&engine_opts.model_file_), "Caffe's prototxt deploy (inference) model.")
        ("batch_size", po::value<int>(&batch_size), "Per device batch size.")
        ("dtype", po::value<std::string>(&engine_opts.dtype_), "Type of data variables: float(same as float32), float32, float16 or int8.")
        ("num_warmup_batches", po::value<int>(&num_warmup_batches), "Number of warmup iterations.")
        ("num_batches", po::value<int>(&num_batches), "Number of benchmark iterations.")
        ("profile",  po::bool_switch(&engine_opts.use_profiler_)->default_value(false), "Profile model and report results.")
        ("input", po::value<std::string>(&engine_opts.input_name_), "Name of an input data tensor (data).")
        ("output", po::value<std::string>(&engine_opts.output_name_), "Name of an output data tensor (prob).")
        ("cache", po::value<std::string>(&engine_opts.calibrator_cache_path_), "Path to folder that will be used to store models calibration data.")
        ("report_frequency", po::value<int>(&report_frequency),
            "Report performance every 'report_frequency' processed batches. "\
            "Default (-1) means report in the end. For benchmarks that last not very long time "\
            "this may be a good option. For very long lasting benchmarks, set this to some positive "\
            "value.")
        ("no_batch_times", po::bool_switch(&engine_opts.do_not_report_batch_times_)->default_value(false),
            "Do not collect and report individual batch times. You may want not "\
            "to report individual batch times when running very long lasting benchmarks. "\
            "Usually, it's used in combination with --report_frequency=N. If you do "\
            "not set the report_frequency and use no_batch_times, the app will still be "\
            "collecting batch times but will not log them.")
        ("data_dir", po::value<std::string>(&data_opts.data_dir_), "Path to a dataset.")
        ("data_name", po::value<std::string>(&data_opts.data_name_), "Name of a dataset - 'images', 'tensors1' or 'tensors4'.")
        ("resize_method", po::value<std::string>(&data_opts.resize_method_), "How to resize images: 'crop' or 'resize'.")
        ("num_prefetchers", po::value<int>(&num_prefetchers), "Number of prefetch threads (data readers).")
        ("prefetch_queue_size", po::value<int>(&prefetch_queue_size), "Number of batches to prefetch.")
        ("prefetch_batch_size", po::value<int>(&prefetch_batch_size), "Size of a prefetch batch.")
        ("num_decoders", po::value<int>(&num_decoders), "Number of decoder threads (that convert JPEG to input blobs).")
        ("fake_decoder",  po::bool_switch(&data_opts.fake_decoder_)->default_value(false),
            "If set, fake decoder will be used. Fake decoder is a decoder that does not decode JPEG images into "\
            "different representation, but just passes through itself inference requests. This option is useful "\
            "to benchmark prefetchers and/or storage.")
        ("inference_queue_size", po::value<int>(&inference_queue_size), "Number of pre-allocated inference requests.")
        ("fake_inference",  po::bool_switch(&engine_opts.fake_inference_)->default_value(false));
   
    po::store(po::parse_command_line(argc, argv, opt_desc), var_map);
    if (var_map.count("version") > 0 || var_map.count("help") > 0)
        return;
    po::notify(var_map);
    
    if (batch_size <= 0)
        throw po::error("Batch size must be strictly positive (size=" + std::to_string(batch_size) + ")");
    if (inference_queue_size <= 0)
        throw po::error("Inference queue size must be strictly positive (size=" + std::to_string(inference_queue_size) + ")");
    engine_opts.batch_size_ = static_cast<size_t>(batch_size);
    engine_opts.inference_queue_size_ = static_cast<size_t>(inference_queue_size);
    engine_opts.num_warmup_batches_ = static_cast<size_t>(std::max(num_warmup_batches, 0));
    engine_opts.num_batches_ = static_cast<size_t>(std::max(num_batches, 0));
    engine_opts.report_frequency_ = static_cast<size_t>(std::max(report_frequency, 0));

    engine_opts.gpus_.clear();
    std::replace(gpus.begin(), gpus.end(), ',', ' ');
    std::istringstream stream(gpus);
    int gpu_id = 0;
    while (stream>>gpu_id) { 
        engine_opts.gpus_.push_back(gpu_id);
        logger.log_info("Will use GPU: " + std::to_string(gpu_id));
    }
    
    if (engine_opts.fake_inference_ && engine_opts.gpus_.size() > 1)
        logger.log_warning("Fake inference will be used but number of engines is > 1. You may want to set it to 1.");
 
    if (data_opts.data_dir_ != "") {
        if (num_prefetchers <= 0)      num_prefetchers = 3 * engine_opts.gpus_.size();
        if (num_decoders <= 0)         num_decoders = 3 * engine_opts.gpus_.size();
        if (prefetch_queue_size <= 0)  prefetch_queue_size = 3 * engine_opts.gpus_.size();
        if (prefetch_batch_size <= 0)  prefetch_batch_size = engine_opts.batch_size_;
    }
    data_opts.num_prefetchers_ = static_cast<size_t>(std::max(num_prefetchers, 0));
    data_opts.num_decoders_ = static_cast<size_t>(std::max(num_decoders, 0));
    data_opts.prefetch_queue_size_ = static_cast<size_t>(std::max(prefetch_queue_size, 0));
    data_opts.prefetch_batch_size_ = static_cast<size_t>(std::max(prefetch_batch_size, 0));;
    if (data_opts.fake_decoder_ && data_opts.num_decoders_ > 1)
        logger.log_warning("Fake decoder will be used but number of decoders > 1. You may want to set it to 1.");
}

void print_file_reader_warnings(logger_impl& logger, const std::string& me) {
    const auto& file_reader = environment::file_reader();
    if (file_reader == "") {
        logger.log_warning(me + "DLBS_TENSORRT_FILE_READER is not set. In this version default file reader changed from 'default' to 'directio'.");
    }
    if (file_reader == "directio") {
        logger.log_info(me + "You are using file reader with DIRECT IO. This is not very well tested feature. If you will be experiencing issues with it, set DLBS_TENSORRT_FILE_READER=default");
    } else if (file_reader == "default" || file_reader == "") {
        if (!environment::remove_files_from_os_cache()) {
            logger.log_warning(me + "Your dataset will be cached by OS (if not disabled). If you do not want this, set DLBS_TENSORRT_FILE_READER=directio");
        } else {
            logger.log_warning(me + "You will be using standard file reader with option to remove files from OS cache. Better use directio reader: DLBS_TENSORRT_FILE_READER=directio");
        }
    }
}