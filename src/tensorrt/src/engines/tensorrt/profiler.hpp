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

#ifndef DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_PROFILER
#define DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_PROFILER

#include "engines/tensorrt/tensorrt_utils.hpp"
#include <algorithm>

/**
 * @brief The profiler, if enabled by a user, profiles execution times of individual layers.
 */
class profiler_impl : public IProfiler {
private:
    typedef std::pair<std::string, float> Record;
    std::vector<Record> mProfile;

public:
    void reset() { 
        mProfile.clear();
    }
  
    virtual void reportLayerTime(const char* layerName, float ms) {
        auto record = std::find_if(mProfile.begin(), mProfile.end(), [&](const Record& r){ return r.first == layerName; });
        if (record == mProfile.end()) {
            mProfile.push_back(std::make_pair(layerName, ms));
        } else {
            record->second += ms;
        }
    }

    void printLayerTimes(const int num_iterations) {
        float totalTime = 0;
        for (size_t i = 0; i < mProfile.size(); i++) {
            printf("%-40.40s %4.3fms\n", mProfile[i].first.c_str(), mProfile[i].second / num_iterations);
            totalTime += mProfile[i].second;
        }
        printf("Time over all layers: %4.3f\n", totalTime / num_iterations);
    }
};

#endif
