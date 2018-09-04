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

#ifndef DLBS_TENSORRT_BACKEND_CORE_CUDA_UTILS
#define DLBS_TENSORRT_BACKEND_CORE_CUDA_UTILS
#include <initializer_list>
#include "core/logger.hpp"
#include "core/utils.hpp"
#include <cuda_runtime.h>

/**
 * @brief Check CUDA result.
 */
#define cudaCheck(ans) { cudaCheckf((ans), __FILE__, __LINE__); }
inline void cudaCheckf(const cudaError_t code, const char *file, const int line, const bool abort=true) {
  if (code != cudaSuccess) {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

/**
 * @brief An implementation of an allocator that allocates pinned memory.
 * 
 * This speeds up host to device data transfers.
 */
class pinned_memory_allocator : public  allocator {
public:
    void allocate(float *&buf, const size_t sz) override {
        cudaCheck(cudaHostAlloc(&buf, sz*sizeof(float), cudaHostAllocPortable|cudaHostAllocWriteCombined));
    }
    void allocate(unsigned char *&buf, const size_t sz) override {
        cudaCheck(cudaHostAlloc(&buf, sz*sizeof(unsigned char), cudaHostAllocPortable|cudaHostAllocWriteCombined));
    }
    void deallocate(float *&buf) override {
        if (buf) {
            cudaCheck(cudaFreeHost(buf));
            buf = nullptr;
        }
    }
    void deallocate(unsigned char *&buf) override {
        if (buf) {
            cudaCheck(cudaFreeHost(buf));
            buf = nullptr;
        }
    }
};

/**
 * @brief Helper class to work with CUDA streams and events.
 * 
 * User provides lists of names for events and streams and then can use these names
 * to work with events/streams like syncing/recording.
 */
class cuda_helper {
private:
    std::map<std::string, cudaEvent_t> events_;    //!< Named events.
    std::map<std::string, cudaStream_t> streams_;  //!< Named streams.
public:
    cuda_helper(const std::initializer_list<std::string>& events, const std::initializer_list<std::string>& streams) {
        for (const auto event: events) {
            cudaEvent_t cuda_event;
            cudaCheck(cudaEventCreate(&cuda_event)); 
            events_[event] = cuda_event;
        }
        for (const auto stream: streams) {
            cudaStream_t cuda_strream;
            cudaCheck(cudaStreamCreate(&cuda_strream)); 
            streams_[stream] = cuda_strream;
        }
    }
    virtual ~cuda_helper() { destroy(); }
    
    void destroy() {
        for (auto& event: events_) {
            cudaCheck(cudaEventDestroy(event.second));
        }
        events_.clear();

        for (auto& stream: streams_) {
            cudaCheck(cudaStreamSynchronize(stream.second));
            cudaCheck(cudaStreamDestroy(stream.second));
        }
        streams_.clear();
    }
    
    cudaEvent_t& event(const std::string& name) { return events_[name]; }
    cudaStream_t& stream(const std::string& name) { return streams_[name]; }
    
    void synch_stream(const std::string& name) { 
        cudaCheck(cudaStreamSynchronize(streams_[name]));
    }
    void synch_event(const std::string& name)  {
        cudaCheck(cudaEventSynchronize(events_[name]));
    }
    void record_event(const std::string& event, const std::string& stream) {
        cudaCheck(cudaEventRecord(events_[event], streams_[stream]));
    }
    float duration(const std::string& start, const std::string& stop) {
        float ms(0);
        cudaEventElapsedTime(&ms, events_[start], events_[stop]);
        return ms;
    }
    
};

#endif
