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

#ifndef DLBS_TENSORRT_BACKEND_CORE_LOGGER
#define DLBS_TENSORRT_BACKEND_CORE_LOGGER

#include <mutex>
#include <iostream>
#include <vector>
#include <algorithm>
#include <map>

#if defined HAVE_NVINFER
    #include <NvInfer.h>
    using namespace nvinfer1;
#else
    class ILogger {};
#endif

/**
 * A simple logger implementation for TensorRT library.
 */
class logger_impl : public ILogger {
private:
    std::mutex m_;           //!< Mutex that guards output stream.
    std::ostream& ostream_;  //!< Output logging stream.

    enum class severity {
        internal_error = 0,
        error = 1,
        warning = 2,
        info = 3
    };
#if defined HAVE_NVINFER
    std::map<ILogger::Severity, severity> severity_transform_ = {
        {ILogger::Severity::kINTERNAL_ERROR, severity::internal_error},
        {ILogger::Severity::kERROR, severity::error},
        {ILogger::Severity::kWARNING, severity::warning},
        {ILogger::Severity::kINFO, severity::info}
    };
#endif

public:
    explicit logger_impl(std::ostream& ostream=std::cout) : ostream_(ostream) {}
    
    void log_key_value(const std::string& key, const float value);
    /**
     * @brief Log intermidiate performance results This is usefull to estimate jitter online or when
     * running long lasting benchmarks.
     * 
     * @param times A vector of individual batch times in milliseconds collected so far. We are interested
     * in times starting from \p iter_index value.
     * @param iter_index An index of the last iteration start
     * @param data_size Data size as number of instances for which individial \p times are reported. In
     * most cases this is the same as effective batch size.
     */
    void log_progress(const std::vector<float>& times, const int iter_index,
                      const int data_size, const std::string& key_prefix);

    /** 
     * @brief Log final benchmark results to a standard output.
     * 
     * @param times A vector of individual batch times in milliseconds. Each element is a time 
     * in seconds it took to process \p data_size input instances.
     * @param data_size Data size as number of instances for which individial \p times are reported. In
     * most cases this is the same as effective batch size.
     * @param key_prefix A key prefix for a key. Identifies what is to be logged. It can be empty
     * to log inference times not taking into account CPU <--> GPU data transfers or 'total_' to log
     * total inference times including data transfers to and from GPU.
     * @param report_times If true, write the content of \p times as well.
     * 
     * This method logs the following keys:
     *   results.${key_prefix}time           A mean value of \p times vector.
     *   results.${key_prefix}throughput     Throughput - number of input instances per second.
     *   results.${key_prefix}time_data      Content of \p times.
     *   results.${key_prefix}time_stdev     Standard deviation of \p times vector.
     *   results.${key_prefix}time_min       Minimal time in \p times vector.
     *   results.${key_prefix}time_max       Maximal time in \p times vector.
     */
    void log_final_results(const std::vector<float>& times, const size_t data_size,
                           const std::string& key_prefix="", const bool report_times=true);
#if defined HAVE_NVINFER  
    // Print engine bindings (input/output blobs)
    void log_bindings(nvinfer1::ICudaEngine* engine, const std::string& log_prefix);
    void log_bindings(nvinfer1::INetworkDefinition* network, const std::string& log_prefix);

    void log(nvinfer1::ILogger::Severity the_severity, const char* msg) override { log_internal(severity_transform_[the_severity], msg); }
#endif
    template <typename T> void log_internal_error(const T& msg) { log_internal(severity::internal_error, msg); }
    template <typename T> void log_error(const T& msg) { log_internal(severity::error, msg); }
    template <typename T> void log_warning(const T& msg) { log_internal(severity::warning, msg); }
    template <typename T> void log_info(const T& msg) { log_internal(severity::info, msg); }
private:
    template <typename T>
    void log_internal(severity the_severity, const T& msg) {
        std::lock_guard<std::mutex> lock(m_);
        ostream_ << time_stamp() << " "  << log_levels_[the_severity] << " "  << msg << std::endl;
        if (the_severity == severity::internal_error || the_severity == severity::error) {
            exit(1);
        }
    }
    template<typename ShapeType>
    void log_tensor_shape(const ShapeType& shape);
    
    std::string time_stamp();
private:
    std::map<severity, std::string> log_levels_ = {
        {severity::internal_error, "INTERNAL_ERROR"},
        {severity::error,          "         ERROR"},
        {severity::warning,        "       WARNING"},
        {severity::info,           "          INFO"}
    };
  
};

#endif