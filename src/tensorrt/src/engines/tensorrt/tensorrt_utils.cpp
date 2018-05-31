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

void check_bindings(ICudaEngine* engine, const std::string& input_blob, const std::string output_blob, logger_impl& logger) {
    const auto nb = engine->getNbBindings();
    if (nb <= 0)
        logger.log_error("Invalid number of a model's IO bindings (" + std::to_string(nb) + ").");
    const auto ib = engine->getBindingIndex(input_blob.c_str());
    const auto ob = engine->getBindingIndex(output_blob.c_str());
    if (ib < 0 || ob < 0 || ib > nb || ob > nb)
        logger.log_error(
            "Invalid indice(s) of IO blob(s). Number of bindings = " + std::to_string(nb) +
            ", input blob (" + input_blob + ") index = " + std::to_string(ib) +
            ", output blob (" + output_blob + ") index = " + std::to_string(ob)
        );
}

/**
 * @brief Return number of elements in \p tensor.
 * @param tensor A pointer to a tensor object.
 * @return Number of elements in \p tensor.
 */
size_t get_tensor_size(const ITensor* tensor) {
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

/**
 * @brief Return number of elements in tensor from binding
 * list accosiated with index \p idx.
 * @param engine Pointer to an engine.
 * @param idx Index of the tensor.
 * @return Number of elements in tensor.
 */
// Get number of elements in this tensor (blob).
size_t get_binding_size(ICudaEngine* engine, const int idx) {
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


ICudaEngine* load_engine_from_file(const std::string& fname, logger_impl& logger) {
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

void serialize_engine_to_file(ICudaEngine *engine_, const std::string& fname) {
    if (fname.empty()) {
        return;
    }
    IHostMemory *se = engine_->serialize();
    fs_utils::write_data(fname, se->data(), se->size());
    se->destroy();
}