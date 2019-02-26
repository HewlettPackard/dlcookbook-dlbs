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

#include "core/logger.hpp"
#include "core/utils.hpp"


void logger_impl::log_key_value(const std::string& key, const float value) {
    std::lock_guard<std::mutex> lock(m_);
    ostream_ << "__" << key << "__=" << value  << std::endl;
}

void logger_impl::log_progress(const std::vector<float>& times, const int /*iter_index*/,
                               const int data_size, const std::string& key_prefix) {
    if (times.empty()) return;
    stats statistics(times);
    const float throughput = 1000.0 * data_size / statistics.mean();
    std::lock_guard<std::mutex> lock(m_);
    ostream_ << "__results." << key_prefix << "progress__=[" << statistics.mean() << ", " << statistics.stdev()<< ", " << throughput << "]" << std::endl;
}

void logger_impl::log_final_results(const std::vector<float>& times, const size_t data_size,
                                    const std::string& key_prefix, const bool report_times) {
    if (times.empty()) return;
    stats statistics(times);
    const float throughput = 1000.0 * data_size / statistics.mean();

    std::lock_guard<std::mutex> lock(m_);
    ostream_ << "__results." << key_prefix << "time__= " << statistics.mean()  << "\n";
    ostream_ << "__results." << key_prefix << "throughput__= " << throughput  << "\n";
    if (report_times) {
        ostream_ << "__results." << key_prefix << "time_data__=[";
        for (std::vector<float>::size_type i=0; i<times.size(); ++i) {
            if (i != 0) { ostream_ << ","; }
            ostream_ << times[i];
        }
        ostream_ << "]" << "\n";
    }
    ostream_ << "__results." << key_prefix << "time_stdev__= " << statistics.stdev() << "\n";
    ostream_ << "__results." << key_prefix << "time_min__= " << statistics.min() << "\n";
    ostream_ << "__results." << key_prefix << "time_max__= " << statistics.max()  << "\n";
    ostream_ << std::flush;
}

#if defined HAVE_NVINFER  

#if NV_TENSORRT_MAJOR < 3
template<>
void logger_impl::log_tensor_shape<Dims3>(const Dims3& shape) {
    ostream_ << ", shape=[" << shape.c << ", " << shape.h << ", " << shape.w << "]" << "\n";
}
#endif

#if NV_TENSORRT_MAJOR >= 3
template<>
void logger_impl::log_tensor_shape<Dims>(const Dims& shape) {
    ostream_ << ", shape=[";
    for (int j=0; j<shape.nbDims; ++j) {
        if (j != 0) { ostream_ << ", "; }
        ostream_ << shape.d[j];
    }
    ostream_ << "]";
}
#endif

void logger_impl::log_bindings(ICudaEngine* engine, const std::string& log_prefix) {
    std::lock_guard<std::mutex> lock(m_);
    const auto num_bindings = engine->getNbBindings();
    ostream_ << time_stamp() << " "  << log_levels_[severity::info] 
             << " "  << log_prefix << " Number of engine bindings (based on ICudaEngine instance) is " << num_bindings << "\n";
    for (auto i=0; i<num_bindings; ++i) {
        ostream_ << time_stamp() << " "  << log_levels_[severity::info]
                 << " " << log_prefix << " Engine binding index = " << i << ", name = " << engine->getBindingName(i) << ", is input = " << engine->bindingIsInput(i);
        log_tensor_shape(engine->getBindingDimensions(i));
        ostream_ << std::endl;
    }
    ostream_ << std::flush;
}

void logger_impl::log_bindings(nvinfer1::INetworkDefinition* network, const std::string& log_prefix) {
    std::lock_guard<std::mutex> lock(m_);
    ostream_ << time_stamp() << " "  << log_levels_[severity::info] << " "  << log_prefix
             << " Model's inputs/outputs based on INetworkDefinition instance, num_inputs=" << network->getNbInputs() << ", num_outputs=" << network->getNbOutputs() << std::endl;
    for (int i=0; i<network->getNbInputs(); ++i) {
        nvinfer1::ITensor* input = network->getInput(i);
        ostream_ << time_stamp() << " "  << log_levels_[severity::info] << " "  << log_prefix
                 << " Model input index = " << i << ", name = " << input->getName();
        log_tensor_shape(input->getDimensions());
        ostream_ << std::endl << std::flush;
    }
    for (int i=0; i<network->getNbOutputs(); ++i) {
        nvinfer1::ITensor* output = network->getOutput(i);
        ostream_ << time_stamp() << " "  << log_levels_[severity::info] << " "  << log_prefix
                 << " Model output index = " << i << ", name = " << output->getName();
        log_tensor_shape(output->getDimensions());
        ostream_ << std::endl << std::flush;
    }
}
#endif
std::string logger_impl::time_stamp() {
    time_t rawtime;
    time (&rawtime);
    struct tm * timeinfo = localtime(&rawtime);
    // YYYY-mm-dd HH:MM:SS    19 characters
    char buffer[20];
    const auto len = strftime(buffer,sizeof(buffer),"%F %T",timeinfo);
    return (len > 0 ? std::string(buffer) : std::string(19, ' '));
}