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

#include "engines/tensorrt/tensorrt_utils.hpp"
#include <sstream>
#ifdef HAVE_ONNX_PARSER
    #include <NvOnnxParser.h>
#endif

std::string get_tensor_name(const std::string& me, logger_impl& logger, const std::vector<std::string>& names,
                            const std::string& in_or_out, const std::string& requested_name) {
    if (names.empty()) {
        logger.log_error(fmt("%s Model does not have %s, cannot proceed.", me.c_str(), in_or_out.c_str()));
    } 
    if (names.size() == 1) {
        if (requested_name != names[0]) {
            logger.log_warning(fmt("%s overriding user's %s tensor name ('%s') with models single %s ('%s').",
                                   me.c_str(), in_or_out.c_str(), requested_name.c_str(), in_or_out.c_str(), names[0].c_str()));
        }
        return names[0];
    }
    const auto iter = std::find(names.begin(), names.end(), requested_name);
    if (iter == names.end()) {
        int size = names.size();
        logger.log_error(fmt("%s model has multiple %ss (count=%d) but does not have requested %s (name='%s').",
                                 me.c_str(), in_or_out.c_str(), size, in_or_out.c_str(), requested_name.c_str()));
    }
    return *iter;
}

nvinfer1::ITensor* get_tensor(nvinfer1::INetworkDefinition* network, const std::string& tensor_name) {
    for (int i=0; i<network->getNbInputs(); ++i) {
        ITensor* input = network->getInput(i);
        if (input->getName() == tensor_name) { return input; }
    }
    for (int i=0; i<network->getNbOutputs(); ++i) {
        ITensor* output = network->getOutput(i);
        if (output->getName() == tensor_name) { return output; }
    }
    return nullptr;
}


std::string tensorrt_utils::tensorrt_version() {
    std::ostringstream os;
    os << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "." << NV_TENSORRT_PATCH;
    return os.str();
}

std::string tensorrt_utils::onnx_parser_version() {
#ifdef HAVE_ONNX_PARSER
    std::ostringstream os;
    os << NV_ONNX_PARSER_MAJOR << "." << NV_ONNX_PARSER_MINOR << "." << NV_ONNX_PARSER_PATCH;
    return os.str();
#endif
    return "not supported";
}

int tensorrt_utils::tensorrt_major_version() {
    return static_cast<int>(NV_TENSORRT_MAJOR);
}

void tensorrt_utils::get_input_output_names(const std::string& me, logger_impl& logger,
                                            INetworkDefinition* network, const inference_engine_opts& opts,
                                            std::string& input_name, std::string& output_name) {
    std::vector<std::string> names;
    for (int i=0; i<network->getNbInputs(); ++i) {
        names.push_back(network->getInput(i)->getName());
    }
    input_name = get_tensor_name(me, logger, names, "input", opts.input_name_);

    names.clear();
    for (int i=0; i<network->getNbOutputs(); ++i) {
        names.push_back(network->getOutput(i)->getName());
    }
    output_name = get_tensor_name(me, logger, names, "output", opts.output_name_);
}

void tensorrt_utils::get_input_output_names(const std::string& me, logger_impl& logger,
                                            ICudaEngine* engine, const inference_engine_opts& opts,
                                            std::string& input_name, std::string& output_name) {
    std::vector<std::string> input_names, output_names;
    for (int i=0; i<engine->getNbBindings(); ++i) {
        if (engine->bindingIsInput(i)) {
            input_names.push_back(engine->getBindingName(i));
        } else {
            output_names.push_back(engine->getBindingName(i));
        }
    }
    input_name = get_tensor_name(me, logger, input_names, "input", opts.input_name_);
    output_name = get_tensor_name(me, logger, output_names, "output", opts.output_name_);
}

/**
 * @brief Return number of elements in \p tensor.
 * @param tensor A pointer to a tensor object.
 * @return Number of elements in \p tensor.
 */
size_t tensorrt_utils::get_tensor_size(INetworkDefinition* network, const std::string& tensor_name) {
    ITensor* tensor = get_tensor(network, tensor_name);
#if NV_TENSORRT_MAJOR >= 3
    Dims shape = tensor->getDimensions();
    long sz = 1;
    for (int i=0; i<shape.nbDims; ++i) {
        sz *= shape.d[i];
    }
    return static_cast<size_t>(sz);
#else
    // Legacy TensorRT returns Dims3 object
    Dims3 shape = tensor->getDimensions();
    return long(shape.c) * shape.w * shape.h;
#endif
}



size_t tensorrt_utils::get_tensor_size(ICudaEngine* engine, const int idx) {
#if NV_TENSORRT_MAJOR >= 3
  const Dims shape = engine->getBindingDimensions(idx);
  long sz = 1;
  for (int i=0; i<shape.nbDims; ++i) {
    sz *= shape.d[i];
  }
  return static_cast<size_t>(sz);
#else
  // Legacy TensorRT returns Dims3 object
  const Dims3 dims = engine->getBindingDimensions(idx);
  return dims.c * dims.h * dims.w;
#endif
}

ICudaEngine* tensorrt_utils::load_engine_from_file(const std::string& fname, logger_impl& logger) {
    size_t nbytes_read(0);
    char* data = fs_utils::read_data(fname, nbytes_read);
    ICudaEngine* engine(nullptr);
    if (nbytes_read > 0 && data) {
        IRuntime *runtime = createInferRuntime(logger);
        engine = runtime->deserializeCudaEngine(data, nbytes_read,nullptr);
        runtime->destroy();
        delete [] data;
    }
    return engine;
}

void tensorrt_utils::serialize_engine_to_file(ICudaEngine *engine_, const std::string& fname) {
    if (fname.empty()) {
        return;
    }
    IHostMemory *se = engine_->serialize();
    fs_utils::write_data(fname, se->data(), se->size());
    se->destroy();
}
