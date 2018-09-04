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

#include "engines/tensorrt_engine.hpp"
#ifdef HOST_DTYPE_INT8
    #include "engines/tensorrt/gpu_cast.h"
#endif

DataType str2dtype(const std::string& dtype) {
    if (dtype == "float32" || dtype == "float")
        return DataType::kFLOAT;
    if (dtype == "float16")
        return DataType::kHALF;
    return DataType::kINT8;
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
        engine_fname = fmt(
            "%s/%s_engine_%s_%d.bin", opts.calibrator_cache_path_.c_str(), opts.model_id_.c_str(),
                                      opts.dtype_.c_str(), opts.batch_size_
        );
        timer clock;
        engine_ = load_engine_from_file(engine_fname, logger_);
        if (engine_) {
            logger_.log_info(fmt("%s Inference engine was loaded from file (%s) in %f ms.", me.c_str(), engine_fname.c_str(), clock.ms_elapsed()));
        } else {
            logger_.log_warning(fmt("%s Failed to loaded inference engine from file (%s).", me.c_str(), engine_fname.c_str()));
        }
    }
    if (!engine_) {
        timer clock;
        const DataType data_type = str2dtype(opts.dtype_);
        logger.log_info(me + " Creating inference builder");
        IBuilder* builder = createInferBuilder(logger_);
        // Parse the caffe model to populate the network, then set the outputs.
        // For INT8 inference, the input model must be specified with 32-bit weights.
        logger.log_info(me + " Creating network and Caffe parser (model: " + opts.model_file_ + ")");
        INetworkDefinition* network = builder->createNetwork();
        ICaffeParser* caffe_parser = createCaffeParser();
        const IBlobNameToTensor* blob_name_to_tensor = caffe_parser->parse(
            opts.model_file_.c_str(), // *.prototxt caffe model definition
            nullptr,       // if null, random weights?
            *network, 
            (data_type == DataType::kINT8 ? DataType::kFLOAT : data_type)
        );
        // Specify what tensors are output tensors.
        network->markOutput(*blob_name_to_tensor->find(opts.output_name_.c_str()));
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

            // Allocate memory but before figure out size of input tensor.
            const nvinfer1::ITensor* input_tensor = blob_name_to_tensor->find(opts.input_name_.c_str());
            calibrator_.initialize(get_tensor_size(input_tensor), 10, opts.model_id_, opts.calibrator_cache_path_);

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
        caffe_parser->destroy();
        logger.log_info(me + " Cleaning buffers");
        if (data_type == DataType::kINT8) {
            calibrator_.freeCalibrationMemory();
        }
        logger_.log_info(fmt("%s Inference engine was created in %f seconds.", me.c_str(), (clock.ms_elapsed()/1000.0)));
        if (!engine_fname.empty()) {
            clock.restart();
            serialize_engine_to_file(engine_, engine_fname);
            logger_.log_info(fmt("%s Inference engine was serialized to file (%s) in %f ms.", me.c_str(), engine_fname.c_str(), clock.ms_elapsed()));
        }
    }
    logger.log_info(me + " Creating execution context");
    exec_ctx_ = engine_->createExecutionContext();
    if (opts.use_profiler_) {
        profiler_ = new profiler_impl();
        exec_ctx_->setProfiler(profiler_);
    }
    logger.log_info(me + " Getting network bindings");
    logger.log_bindings(engine_, me);
    // We need to figure out number of elements in input/output tensors.
    // Also, we need to figure out their indices.
    check_bindings(engine_, opts.input_name_, opts.output_name_, logger_);
    bindings_.resize(static_cast<size_t>(engine_->getNbBindings()), nullptr);
    input_idx_ = static_cast<size_t>(engine_->getBindingIndex(opts.input_name_.c_str()));
    output_idx_ = static_cast<size_t>(engine_->getBindingIndex(opts.output_name_.c_str()));
    input_sz_ = get_binding_size(engine_, input_idx_),    // Number of elements in 'data' tensor.
    output_sz_ = get_binding_size(engine_, output_idx_);  // Number of elements in 'prob' tensor.
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
#ifdef HOST_DTYPE_SP32
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
#ifdef HOST_DTYPE_SP32
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

