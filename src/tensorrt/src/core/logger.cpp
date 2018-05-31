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


void logger_impl::log_key_value(const std::string& key, const float value) {
    std::lock_guard<std::mutex> lock(m_);
    ostream_ << "__" << key << "__=" << value  << std::endl;
}

void logger_impl::log_progress(const std::vector<float>& times, const int iter_index,
                               const int data_size, const std::string& key_prefix) {
    if (times.empty()) return;
    const float sum = std::accumulate(times.begin(), times.end(), 0.0f);
    const float mean = sum / times.size();
    const float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
    const float stdev = std::sqrt(sq_sum / times.size() - mean * mean);
    const float throughput = 1000.0 * data_size / mean;
    std::lock_guard<std::mutex> lock(m_);
    ostream_ << "__results." << key_prefix << "progress__=[" << mean << ", " << stdev << ", " << throughput << "]" << std::endl;
}

void logger_impl::log_final_results(const std::vector<float>& times, const size_t data_size,
                                    const std::string& key_prefix, const bool report_times) {
    if (times.empty()) return;
    const float sum = std::accumulate(times.begin(), times.end(), 0.0f);
    const float mean = sum / times.size();
    const float sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0f);
    const float stdev = std::sqrt(sq_sum / times.size() - mean * mean);
    const float throughput = 1000.0 * data_size / mean;

    std::lock_guard<std::mutex> lock(m_);
    ostream_ << "__results." << key_prefix << "time__= " << mean  << "\n";
    ostream_ << "__results." << key_prefix << "throughput__= " << throughput  << "\n";
    if (report_times) {
        ostream_ << "__results." << key_prefix << "time_data__=[";
        for (std::vector<float>::size_type i=0; i<times.size(); ++i) {
            if (i != 0) { ostream_ << ","; }
            ostream_ << times[i];
        }
        ostream_ << "]" << "\n";
    }
    ostream_ << "__results." << key_prefix << "time_stdev__= " << stdev  << "\n";
    //ostream_ << "__results." << key_prefix << "time_min__= " << times.front()  << "\n";
    //ostream_ << "__results." << key_prefix << "time_max__= " << times.back()  << "\n";
    ostream_ << std::flush;
}

void logger_impl::log_bindings(ICudaEngine* engine, const std::string& log_prefix) {
    std::lock_guard<std::mutex> lock(m_);
    const auto num_bindings = engine->getNbBindings();
    ostream_ << time_stamp() << " "  << log_levels_[ILogger::Severity::kINFO] 
             << " "  << log_prefix << " Number of engine bindings is " << num_bindings << "\n";
    for (auto i=0; i<num_bindings; ++i) {
        ostream_ << time_stamp() << " "  << log_levels_[ILogger::Severity::kINFO]
                 << " " << log_prefix << " Engine binding index = " << i << ", name = " << engine->getBindingName(i) << ", is input = " << engine->bindingIsInput(i);
#if NV_TENSORRT_MAJOR >= 3
        const Dims shape = engine->getBindingDimensions(i);
        ostream_ << ", shape=[";
        for (int j=0; j<shape.nbDims; ++j) {
            if (j != 0) {
                ostream_ << ", ";
            }
            ostream_ << shape.d[j];
        }
        ostream_ << "]" << "\n";
#else
        const Dims3 dims = engine->getBindingDimensions(i);
        ostream_ << ", shape=[" << dims.c << ", " << dims.h << ", " << dims.w << "]" << "\n";
#endif
    }
    ostream_ << std::flush;
}

std::string logger_impl::time_stamp() {
    time_t rawtime;
    time (&rawtime);
    struct tm * timeinfo = localtime(&rawtime);
    // YYYY-mm-dd HH:MM:SS    19 characters
    char buffer[20];
    const auto len = strftime(buffer,sizeof(buffer),"%F %T",timeinfo);
    return (len > 0 ? std::string(buffer) : std::string(19, ' '));
}