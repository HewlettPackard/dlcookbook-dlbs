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

#ifndef DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_CALIBRATOR
#define DLBS_TENSORRT_BACKEND_ENGINES_TENSORRT_CALIBRATOR

#include "core/logger.hpp"
#include "core/utils.hpp"
#include "engines/tensorrt/tensorrt_utils.hpp"
#include <sstream>

/**
 * @brief TensorRT calibrator to calibrate engine with INT8 data type.
 */
#if NV_TENSORRT_MAJOR >= 3
class calibrator_impl : public IInt8LegacyCalibrator {
#else
class calibrator_impl : public IInt8Calibrator {
#endif
private:
    logger_impl& logger_;

    size_t batch_size_ = 0;               // Batch size (number of instances)
    size_t input_size_ = 0;               // Size of one instance (multiplication of all dimensions)
    size_t num_batches_ = 0;              // Number of batches to use for calibration
    size_t next_batch_ = 0;               // During calibration, index of the next batch
    std::vector<float> host_batch_;       // Batch data in host memory
    void *gpu_batch_ = nullptr;           // Batch data in GPU memory
  
    double quantile_ = 0.5;
    double cutoff_ = 0.5;
  
    bool do_log_ = true;
  
    std::string cache_path_;              // Path to calibration cache.
    std::string model_;                   // Neural network model (to save/load calibration caches).
    char* calibration_cache_ = nullptr;   // Calibration cache loaded from file.
    size_t calibration_cache_length_ = 0;
    char* histogram_cache_ = nullptr;     // Histogram cache loaded from file.
    size_t histogram_cache_length_ = 0;
public:
    explicit calibrator_impl(logger_impl& logger, const bool do_log=false) : logger_(logger), do_log_(do_log) {}
    
    // The batch size is for a calibration stage.
    int getBatchSize() const override {
        if (do_log_) {
            logger_.log_info("[calibrator] Calibrator::getBatchSize() { return " + std::to_string(batch_size_) + "; }");
        }
        return batch_size_; 
    }
    // For ImageNet networks and MNIST, 500 images is a reasonable size for the calibration set.
    // In a simpliest case, nbBindings is 1 and names[0] = 'data'
    // For each input tensor, a pointer to input data in GPU memory must be written into the bindings 
    // array. The names array contains the names of the input tensors, and the position for each tensor 
    // in the bindings array matches the position of its name in the names array. Both arrays have size 
    // nbBindings.
    bool getBatch(void* bindings[], const char* names[], int nbBindings) override { 
        if (do_log_) {
            std::ostringstream stream;
            stream << "[calibrator] Calibrator::getBatch(names=[";
            for (int i=0; i<nbBindings; ++i) {
                if (i != 0) {
                    stream << ", ";
                }
                stream << "'" << names[i]  << "'";
            }
            stream << "], nbBindings=" << nbBindings << ")";
            logger_.log_info(stream.str());
        }
        if (nbBindings != 1) {
            logger_.log_error("***ERROR*** Exactly one input must present but found " + std::to_string(nbBindings) + " input(s).");
        }
        // Make sure that this calibrator was initialized.
        if (num_batches_ <= 0) {
            logger_.log_error("***ERROR***: Suspicious number of batches (0) in calibrator::getBatch().");
        }
        // Lazy memory allocation - allocate only if we are here.
        if (gpu_batch_ == nullptr) {
            host_batch_.resize(batch_size_ * input_size_);
            cudaCheck(cudaMalloc(&(gpu_batch_), sizeof(float) * batch_size_ * input_size_));
        }
        //
        if (next_batch_ >= num_batches_) { return false; }
        //
        fill_random(host_batch_.data(), host_batch_.size());
        cudaCheck(cudaMemcpy(gpu_batch_, host_batch_.data(), sizeof(float) * host_batch_.size(), cudaMemcpyHostToDevice));
        bindings[0] = gpu_batch_;
        next_batch_ ++;
        return true; 
    }
    // The cutoff and quantile parameters take values in the range [0,1]; their meaning is discussed 
    // in detail in the accompanying white paper. To find the best calibration parameters, it can be 
    // useful to search over the parameter combinations and score the network for each combination 
    // using some additional images. searchCalibrations() illustrates how to do this. For ImageNet 
    // networks, 5000 images were used to find the optimal calibration. Since the calibration process 
    // will run many times varying only the regression and cutoff parameters, histogram caching is 
    // strongly recommended.
    double getQuantile() const override { 
        if (do_log_) {
            logger_.log_info("[calibrator] Calibrator::getQuantile() { return " + std::to_string(quantile_) + "; }");
        }
        return quantile_; 
    }
    double getRegressionCutoff() const override { 
        if (do_log_) {
            logger_.log_info("[calibrator] Calibrator::getRegressionCutoff() { return " + std::to_string(cutoff_) + "; }");
        }
        return cutoff_; 
    }
    const void* readCalibrationCache(std::size_t& length/*output param*/) override {
        if (do_log_) {
            logger_.log_info("[calibrator] Calibrator::readCalibrationCache()");
        }
        update_cache(calibration_cache_, calibration_cache_length_, get_cache_file("calibration"));
        length = calibration_cache_length_;
        if (calibration_cache_ != nullptr) {
            logger_.log_info("Calibration cache has succesfully been read (length=" + std::to_string(length) + ").");
        }
        return static_cast<const void*>(calibration_cache_);
    }
    void writeCalibrationCache(const void* ptr, std::size_t length) override {
        if (do_log_) {
            logger_.log_info("[calibrator] Calibrator::writeCalibrationCache(length=" + std::to_string(length) + ")");
        }
        fs_utils::write_data(get_cache_file("calibration"), ptr, length);
    }
    const void* readHistogramCache(std::size_t& length/*output param*/) override { 
        if (do_log_) {
            logger_.log_info("[calibrator] Calibrator::readHistogramCache()");
        }
        update_cache(histogram_cache_, histogram_cache_length_, get_cache_file("histogram"));
        length = histogram_cache_length_;
        if (histogram_cache_ != nullptr) {
            logger_.log_info("Histogram cache has succesfully been read(length=" + std::to_string(length) + ").");
        }
        return static_cast<const void*>(histogram_cache_);
    }
    void writeHistogramCache(const void* ptr, std::size_t length) override { 
        if (do_log_) {
            logger_.log_info("[calibrator] Calibrator::writeHistogramCache(length=" + std::to_string(length) + ")");
        }
        fs_utils::write_data(get_cache_file("histogram"), ptr, length);
    }
    void setLog(const bool do_log=true) {
        if (do_log_ || do_log) {
            logger_.log_info(std::string("[calibrator] Calibrator::setLog(do_log=") + (do_log ? "true" : "false") + ")");
        }
        do_log_ = do_log; 
    }
    void setBatchSize(const size_t batch_size) {
        if (do_log_) {
            logger_.log_info("[calibrator] Calibrator::setBatchSize(batch_size=" + std::to_string(batch_size) + ")");
        }
        batch_size_ = batch_size;
    }
    /**
     * Initialize Calibrator. No memory allocations are done here.
     * @param input_size Size (number of elements) of a single input instance. Does not include batch dimension.
     * @param num_batches Numebr of calibration iterations.
     * @param model A neural network model identifier such as alexnet, resnet101, vgg13 etc.
     * @param cache_path A path to folder that contains calibration cache data. With every model two 
     * files are associated - calibration and hostogram cache files.
     */
    void initialize(const size_t input_size, const size_t num_batches, const std::string& model, const std::string& cache_path) {
        if (do_log_) {
            logger_.log_info(
                "[calibrator] Calibrator::initialize(input_size=" + std::to_string(input_size) + 
                              ", num_batches=" + std::to_string(num_batches) + ", model = " + model + ")"
            );
        }
        input_size_ = input_size;
        num_batches_ = num_batches;
        next_batch_ = 0;

        cache_path_ = cache_path;
        model_ = model;

        if (cache_path_ == "") {
            logger_.log_warning("***WARNING***: Calibration cache path is not set.");
        } else {
            logger_.log_info(
                "Calibration cache file: " + get_cache_file("calibration") +
                ", histogram cache file: " + get_cache_file("histogram")
            );
        }
    }
    void freeCalibrationMemory() {
        if (do_log_) { logger_.log_info("[calibrator] Calibrator::freeCalibrationMemory"); }
        cudaFree(gpu_batch_);
        host_batch_.clear();
        if (calibration_cache_ != nullptr) {
            delete [] calibration_cache_;
            calibration_cache_length_ = 0;
        }
        if (histogram_cache_ != nullptr) {
            delete [] histogram_cache_;
            histogram_cache_length_ = 0;
        }
    }
private:
    std::string get_cache_file(const std::string& suffix) const {
        if (cache_path_ != "" && model_ != "") {
            return cache_path_ + "/" + model_ + "_" + suffix + ".bin";
        }
        return "";
    }
    void update_cache(char*& cache_data, size_t& cache_length, const std::string& fname) {
        if (cache_data == nullptr) {
            cache_data = fs_utils::read_data(fname, cache_length);
        }
    }
};

#endif