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
 * What could be a new inference loop with compute/data transfers overlaps?
 * 
 * We have the following possible events:
 *   1. New inference request.
 *   2. Host2Device copy (H2DC) done.
 *   3. Compute done.
 *   4. Device2Host (D2HC) copy done.
 * 
 * The input data may be as large as 300-500 MB while output should be within
 * several MBs. In the first version we can assume steps 3 and 4 are performed
 * sequentially - we still have multiple events though.
 * 
 * Assumption - fetching requests is not overlapepd.
 * 
 * Possible states:
 *      H2DC_None, H2DC_Progress, H2DC_Done
 *      Compute_None, Compute_Progress, Compute_Done
 * 
 * Compute_Progress
 * 
 * while (not stop) {
 *     if (H2DC_Progress and Compute_Progress) {
 *         synch(Compute);
 *         continue;
 *     }
 *     if ()
 *     
 * }
 */

#include <signal.h>
#include "engines/tensorrt_engine.hpp"
#ifdef HOST_DTYPE_INT8
    #include "engines/tensorrt/gpu_cast.h"
#endif
#ifdef HAVE_CAFFE_PARSER
    #include <NvCaffeParser.h>
#endif
#ifdef HAVE_ONNX_PARSER
    #include <NvOnnxParser.h>
    #include <NvOnnxConfig.h>
#endif

class parser {
public:
    const std::string& me_;
    logger_impl& logger_;
    const inference_engine_opts& opts_;
    IBuilder* builder_ = nullptr;
public:
    static parser* get_parser(const std::string& me, logger_impl& logger, const inference_engine_opts& opts, IBuilder* builder);
    static DataType str2dtype(const std::string& dtype) {
        if (dtype == "float32" || dtype == "float")
            return DataType::kFLOAT;
        if (dtype == "float16")
            return DataType::kHALF;
        return DataType::kINT8;
    }

    parser(const std::string& me, logger_impl& logger,
           const inference_engine_opts& opts, IBuilder* builder): me_(me), logger_(logger), opts_(opts), builder_(builder) {
    }
    virtual ~parser() {}
    virtual INetworkDefinition* parse() = 0;
};

template<typename ParserType>
class trt_parser: public parser {
public:
    ParserType* parser_impl_ = nullptr;
public:
    trt_parser(const std::string& me, logger_impl& logger,
               const inference_engine_opts& opts, IBuilder* builder): parser(me, logger, opts, builder) {}
    ~trt_parser() { if (parser_impl_ != nullptr) { parser_impl_->destroy(); } }
    INetworkDefinition* parse() override;
};

#ifdef HAVE_CAFFE_PARSER
template <>
INetworkDefinition* trt_parser<nvcaffeparser1::ICaffeParser>::parse() {
    logger_.log_info(me_ + " Creating network with Caffe parser (model: " + opts_.model_file_ + ")");
    parser_impl_ = nvcaffeparser1::createCaffeParser();
    const DataType data_type = str2dtype(opts_.dtype_);
    INetworkDefinition* network = builder_->createNetwork();
    const nvcaffeparser1::IBlobNameToTensor* blob_name_to_tensor = parser_impl_->parse(
        opts_.model_file_.c_str(), // *.prototxt caffe model definition
        nullptr,       // if null, random weights?
        *network, 
        (data_type == DataType::kINT8 ? DataType::kFLOAT : data_type)
    );
    // Specify what tensors are output tensors. With Caffe format, we need to mark what blob is an output blob.
    // The `IBlobNameToTensor` interface provides only `find` method and does not allow users to iterate over all blobs.
    nvinfer1::ITensor* output_tensor = blob_name_to_tensor->find(opts_.output_name_.c_str());
    // In case it was not found, print warning and continue. In case there's no output already specified, the engine will
    // exit at later stage.
    if (output_tensor == nullptr) {
        logger_.log_warning(fmt("%s No output tensor ('%s') found in a model definition loaded from prototxt file.", me_.c_str(), opts_.output_name_.c_str()));
    } else {
        network->markOutput(*output_tensor);
    }
    return network;
}
#endif

#ifdef HAVE_ONNX_PARSER
template <>
INetworkDefinition* trt_parser<nvonnxparser::IParser>::parse() {
    logger_.log_warning(me_ + " You are using a model in ONNX format. It's under active development and anything can happen, in particular, segmentation fault. Good luck!");
    logger_.log_info(me_ + " Creating network with ONNX parser. Model: " + opts_.model_file_ + ". Parser version: " + tensorrt_utils::onnx_parser_version() + ".");
    INetworkDefinition* network = builder_->createNetwork();
    parser_impl_ = nvonnxparser::createParser(*network, logger_);
    if (!parser_impl_->parseFromFile(opts_.model_file_.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        logger_.log_error("Failed to parse ONNX file.");
    }
    return network;
}
#endif

parser* parser::get_parser(const std::string& me, logger_impl& logger, const inference_engine_opts& opts, IBuilder* builder) {
    if (ends_with(opts.model_file_, ".prototxt")) {
#ifdef HAVE_CAFFE_PARSER
        return new trt_parser<nvcaffeparser1::ICaffeParser>(me, logger, opts, builder);
#else
        logger.log_error("No Caffe parser module was found with TensorRT.");
#endif
    } else if (ends_with(opts.model_file_, ".onnx")) {
#ifdef HAVE_ONNX_PARSER
        return new trt_parser<nvonnxparser::IParser>(me, logger, opts, builder);
#else
        auto msg = fmt("TensorRT version 5 and above is required to load models in ONNX format. "\
                       "Your TensorRT version is %d.%d.%d.", int(NV_TENSORRT_MAJOR), int(NV_TENSORRT_MINOR), int(NV_TENSORRT_PATCH));
        throw std::runtime_error(msg);
#endif
    }
    logger.log_error("Invalid model file. Only Caffe (.prototxt) and ONNX (.oxxn) formats are supported.");
    return nullptr; // To avoid compiler warning.
}


tensorrt_inference_engine::tensorrt_inference_engine(const int engine_id, const int num_engines,
                                                     logger_impl& logger, const inference_engine_opts& opts)
: inference_engine(engine_id, num_engines, logger, opts), calibrator_(logger) {

    using namespace nvcaffeparser1;
    const auto me = fmt("[inference engine %02d/%02d]:", engine_id, num_engines);
    int prev_cuda_device(0);
    cudaCheck(cudaGetDevice(&prev_cuda_device));
    cudaCheck(cudaSetDevice(engine_id));
    // Build and cache a new engine OR load previously cached engine
    engine_ = nullptr;
    std::string engine_fname = "";
    if (!opts.calibrator_cache_path_.empty()) {
        // resnet50_float16_128_v3.bin
        const int major_version = tensorrt_utils::tensorrt_major_version();
        engine_fname = fmt(
            "%s/%s_%s_%d_v%d.bin", opts.calibrator_cache_path_.c_str(), opts.model_id_.c_str(),
                                   opts.dtype_.c_str(), opts.batch_size_, major_version
        );
        timer clock;
        engine_ = tensorrt_utils::load_engine_from_file(engine_fname, logger_);
        if (engine_) {
            logger_.log_info(fmt("%s Inference engine was loaded from file (%s) in %f ms.", me.c_str(), engine_fname.c_str(), clock.ms_elapsed()));
        } else {
            logger_.log_warning(fmt("%s Failed to loaded inference engine from file (%s). That's OK.", me.c_str(), engine_fname.c_str()));
        }
    }
    // In DLPG models, input/output names have predefined standard values - 'data' and 'prob'. Morover,
    // current implementation expects one input and one output tensor. In many cases with user models
    // (prototxt or onnx formats), these inputs and outputs may have different names. This means users
    // will need to provide this names on a command line.
    // In certain cases however, it is possible to correct (override) user selection:
    //    - If model defines exactly two bindings
    //    - One binding is an input and one is an output.
    // Whenever in this function input/output names need to be used, this variables will point to the
    // right names that can differ from what user has provided.
    std::string input_name, output_name;
    if (!engine_) {
        timer clock;
        const DataType data_type = parser::str2dtype(opts.dtype_);
        logger.log_info(me + " Creating inference builder");
        IBuilder* builder = createInferBuilder(logger_);
        // Build model definition (from caffe's prototxt or onnx format)
        parser* model_parser = parser::get_parser(me, logger, opts, builder);
        INetworkDefinition* network = model_parser->parse();
        // Log model's inputs/outputs and make sure we have the right names for tensors.
        logger.log_info(me + " Getting network bindings (from INetworkDefinition)");
        logger.log_bindings(network, me);
        tensorrt_utils::get_input_output_names(me, logger, network, opts, input_name, output_name);
        // Build the engine.
        builder->setMaxBatchSize(opts.batch_size_);
        builder->setMaxWorkspaceSize(1 << 30); 
        // Half and INT8 precision specific options
        if (data_type == DataType::kHALF) {
            logger.log_info(me + " Enabling FP16 mode");
            builder->setHalf2Mode(true);
        } else if (data_type == DataType::kINT8) {
            logger.log_info(me + " Enabling INT8 mode");
            calibrator_.setBatchSize(opts.batch_size_);
            // Allocate memory but before figure out the size of an input tensor.
            calibrator_.initialize(tensorrt_utils::get_tensor_size(network, input_name),
                                   10, opts.model_id_, opts.calibrator_cache_path_);

            builder->setInt8Mode(true);
            builder->setInt8Calibrator(&calibrator_);
        } else {
            logger.log_info(me + " Enabling FP32 mode");
        }
        // This is where we need to use calibrator
        engine_ = builder->buildCudaEngine(*network);
        // Destroy objects that we do not need anymore
        network->destroy();
        builder->destroy();
        delete model_parser;
        logger.log_info(me + " Cleaning buffers");
        if (data_type == DataType::kINT8) {
            calibrator_.freeCalibrationMemory();
        }
        logger_.log_info(fmt("%s Inference engine was created in %f seconds.", me.c_str(), (clock.ms_elapsed()/1000.0)));
        if (!engine_fname.empty()) {
            clock.restart();
            tensorrt_utils::serialize_engine_to_file(engine_, engine_fname);
            logger_.log_info(fmt("%s Inference engine (model: '%s') was serialized to file (%s) in %f ms.",
                                 me.c_str(), opts.model_file_.c_str(), engine_fname.c_str(), clock.ms_elapsed()));
        }
    }
    logger.log_info(me + " Creating execution context");
    exec_ctx_ = engine_->createExecutionContext();
    if (opts.use_profiler_) {
        profiler_ = new profiler_impl();
        exec_ctx_->setProfiler(profiler_);
    }
    if (input_name == "" || output_name == "") {
        // If these are empty, it means we have loaded the serialized engine. We do not need to
        // check bindings in this case, because we have already done so in previous runs, but just
        // in case this model came from somebody else.
        logger.log_info(me + " Getting network bindings (from ICudaEngine)");
        logger.log_bindings(engine_, me);
        tensorrt_utils::get_input_output_names(me, logger, engine_, opts, input_name, output_name);
    }
    bindings_.resize(static_cast<size_t>(engine_->getNbBindings()), nullptr);
    input_idx_ = static_cast<size_t>(engine_->getBindingIndex(input_name.c_str()));
    output_idx_ = static_cast<size_t>(engine_->getBindingIndex(output_name.c_str()));
    input_sz_ = tensorrt_utils::get_tensor_size(engine_, input_idx_),    // Number of elements in 'data' tensor.
    output_sz_ = tensorrt_utils::get_tensor_size(engine_, output_idx_);  // Number of elements in 'prob' tensor.
    cudaCheck(cudaSetDevice(prev_cuda_device));
}
    
tensorrt_inference_engine::~tensorrt_inference_engine() {
    if (profiler_) {
        profiler_->printLayerTimes(nbatches_);
        delete profiler_;
    }
    exec_ctx_->destroy();
    engine_->destroy();
    if (bindings_[input_idx_]) cudaFree(bindings_[input_idx_]);
    if (bindings_[output_idx_]) cudaFree(bindings_[output_idx_]);
#ifdef HOST_DTYPE_INT8
    if (input_buffer_) cudaFree(input_buffer_);
#endif
}
    
void tensorrt_inference_engine::init_device() {
    const auto me = fmt("[inference engine %02d/%02d]:", engine_id_, num_engines_);
    cudaCheck(cudaSetDevice(engine_id_));
    cudaCheck(cudaMalloc(&(bindings_[input_idx_]),  sizeof(float) * batch_sz_ * input_sz_));
    cudaCheck(cudaMalloc(&(bindings_[output_idx_]), sizeof(float) * batch_sz_ * output_sz_));
#ifdef HOST_DTYPE_INT8
    cudaCheck(cudaMalloc(&input_buffer_,  sizeof(host_dtype) * batch_sz_ * input_sz_));
#endif
}



void tensorrt_inference_engine::copy_input_to_gpu_asynch(inference_msg *msg ,cudaStream_t stream) {
#ifdef HOST_DTYPE_FP32
    // Just copy data into TensorRT buffer.
    cudaCheck(cudaMemcpyAsync(
        bindings_[input_idx_],
        msg->input(),
        sizeof(host_dtype)*msg->input_size()*msg->batch_size(),
        cudaMemcpyHostToDevice,
        stream
    ));
#elif defined HOST_DTYPE_INT8
    // Copy data into intermidiate buffer and then cast to single precision.
    cudaCheck(cudaMemcpyAsync(
        input_buffer_,
        msg->input(),
        sizeof(host_dtype)*msg->input_size()*msg->batch_size(),
        cudaMemcpyHostToDevice,
        stream
    ));
    gpu_cast(msg->batch_size(), msg->input_size(), input_buffer_, static_cast<float*>(bindings_[input_idx_]), stream);
    //
#endif
}

void tensorrt_inference_engine::do_inference(abstract_queue<inference_msg*> &request_queue,
                                             abstract_queue<inference_msg*> &response_queue) {
    init_device();
    if (environment::inference_impl_ver() == "1") {
        do_inference1(request_queue, response_queue);
        return;
    }
    const std::string me = fmt("[inference engine %02d/%02d]", abs(engine_id_), num_engines_);
    logger_.log_info(me + ": Implementation version is latest");
#ifdef HOST_DTYPE_FP32
    logger_.log_info(me + ": Will consume tensor<float> tensors.");
#elif defined HOST_DTYPE_INT8
    logger_.log_info(me + ": Will consume tensor<unsigned char> tensors. Casting to 'float' will be done on GPU.");
#else
    logger_.log_warning(me + ": Will consume tensor<?> tensors. This is BUG.");
#endif
    // If it's true, the code below will be overlaping
    // copy/compute ops. This makes sense when time to fetch data fro, request queue
    // is very small. If it's large (greater than inference time), you may want to do
    // everything sequentially.
    const bool overlap_copy_compute = environment::overlap_copy_compute();
    if (overlap_copy_compute) {
        logger_.log_info(me + ": Will overlap compute (inference) and data transfers (host to device).");
    } else {
        logger_.log_info(me + ": Will NOT overlap compute (inference) and data transfers (host to device).");
    }
    cuda_helper helper({"input_consumed","infer_start","infer_stop"}, {"copy","compute"});
    running_average copy2device_synch, fetch, process, process_fetch, copy2host, submit;
    try {
        inference_msg *current_msg(nullptr), *next_msg(nullptr);
        timer clock, curr_batch_clock, next_batch_clock, infer_clock;
        while(!stop_) {
            if (reset_) {
                reset_ = false;
                if (profiler_) profiler_->reset();
                tm_tracker_.reset();
            }
            if (paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            //
            if (!current_msg) {
                DLOG(me + ": getting first inference request");
                clock.restart();  current_msg = request_queue.pop();  fetch.update(clock.ms_elapsed());
                DLOG(me + ": request has been fetched, starting async transfer to device.");
                curr_batch_clock.restart();
                copy_input_to_gpu_asynch(current_msg, helper.stream("copy"));
                DLOG(me + ": async device transfer started.");
            }
            DLOG(me + ": waiting for previous device copy to complete.");
            clock.restart();  helper.synch_stream("copy");  copy2device_synch.update(clock.ms_elapsed());
            DLOG(me + ": previous device copy completed.");
            // Run asynchronously GPU kernel and wait for input data consumed
            infer_clock.restart();
            helper.record_event("infer_start", "compute");
            if (!exec_ctx_->enqueue(batch_sz_, bindings_.data(), helper.stream("compute"), &helper.event("input_consumed") )) {
                logger_.log_error(fmt("%s: Inference request was not enqueued.", me.c_str()));
            }
            helper.record_event("infer_stop", "compute");
            // Fetch new data and start copying it to GPU memory
            if (overlap_copy_compute) {
                clock.restart();  next_msg = request_queue.pop();  fetch.update(clock.ms_elapsed());
                next_batch_clock.restart();
                helper.synch_event("input_consumed");
                copy_input_to_gpu_asynch(next_msg, helper.stream("copy"));
            }
            // Wait for comptue to complete and copy result back to CPU memory
            helper.synch_event("infer_stop");
            const auto process_tm = helper.duration("infer_start", "infer_stop");   // Inference GPU time.
            process.update(process_tm);
            tm_tracker_.infer_done(process_tm);   
            current_msg->set_infer_time(process_tm);
            process_fetch.update(infer_clock.ms_elapsed());
            clock.restart();
            cudaCheck(cudaMemcpyAsync(
                current_msg->output(),
                bindings_[output_idx_],
                sizeof(float)*current_msg->output_size()*current_msg->batch_size(),
                cudaMemcpyDeviceToHost,
                helper.stream("compute")
            ));
            helper.synch_stream("compute");
            copy2host.update(clock.ms_elapsed());
            tm_tracker_.batch_done(curr_batch_clock.ms_elapsed());                  // Actual batch time (remember, batches overlap)
            current_msg->set_batch_time(tm_tracker_.last_batch_time());
            // Send response and update current message (if not overlap_copy_compute, next_msg is nullptr here).
            current_msg->set_gpu(engine_id_);
            clock.restart();  response_queue.push(current_msg);  submit.update(clock.ms_elapsed());
            current_msg = next_msg;
            next_msg = nullptr;
            curr_batch_clock = next_batch_clock;
        }
    }catch (queue_closed) {
    }
    helper.destroy();
    if (!overlap_copy_compute) {
        logger_.log_info(fmt("%s: {fetch:%.5f}-->--[process:%.5f]-->--{submit:%.5f}", me.c_str(), fetch.value(), process.value(), submit.value()));
    } else {
        logger_.log_info(fmt("%s: {({process:%.5f}|{fetch:%.5f}):%.5f}-->--{copy2host:%.5f}-->--{submit:%.5f}-->--{copy2device_synch:%.5f}",
                             me.c_str(), process.value(), fetch.value(), process_fetch.value(),
                             copy2host.value(), submit.value(), copy2device_synch.value()));
        // Issue warning if fetch time takes more than 50% of compute time that may be an indication of
        // an ingestion pipeline being a bottleneck.
        if (fetch.value() > process.value() * 0.5) {
            logger_.log_warning(fmt("%s: Fetch time (%.5f) seems to be large relative to compute time (%.5f). This may indicate "\
                                    "that ingestion pipeline is a bottleneck. ", me.c_str(), fetch.value(), process.value()));
        }
    }
}

void tensorrt_inference_engine::do_inference1(abstract_queue<inference_msg*> &request_queue,
                                              abstract_queue<inference_msg*> &response_queue) {
    const std::string me = fmt("[inference engine %02d/%02d]", abs(engine_id_), num_engines_);
    logger_.log_info(me + ": Version of inference engine is 1");
    running_average fetch, process, submit;
    try {
        inference_msg *msg(nullptr);
        timer clock;
        while(!stop_) {
            if (reset_) {
                reset_ = false;
                if (profiler_) profiler_->reset();
                tm_tracker_.reset();
            }
            if (paused_) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            //
            clock.restart();  msg = request_queue.pop();  fetch.update(clock.ms_elapsed());
            // Copy Input Data to the GPU.
            tm_tracker_.batch_started();
            cudaCheck(cudaMemcpy(bindings_[input_idx_], msg->input(), sizeof(float)*msg->input_size()*msg->batch_size(), cudaMemcpyHostToDevice));
            // Launch an instance of the GIE compute kernel.
            tm_tracker_.infer_started();
            if(!exec_ctx_->execute(batch_sz_, bindings_.data())) {logger_.log_error("Kernel was not run");}
            tm_tracker_.infer_done();
            // Copy Output Data to the Host.
            cudaCheck(cudaMemcpy(msg->output(), bindings_[output_idx_], sizeof(float)*msg->output_size()*msg->batch_size(), cudaMemcpyDeviceToHost));
            tm_tracker_.batch_done();
            //
            process.update(tm_tracker_.last_batch_time());
            msg->set_infer_time(tm_tracker_.last_infer_time());
            msg->set_batch_time(tm_tracker_.last_batch_time());
            nbatches_ ++;
            //
            clock.restart();  response_queue.push(msg);  submit.update(clock.ms_elapsed());
        }
    }catch (queue_closed) {
    }
    logger_.log_info(fmt("%s: {fetch:%.5f}-->--[process:%.5f]-->--{submit:%.5f}", me.c_str(), fetch.value(), process.value(), submit.value()));
}

