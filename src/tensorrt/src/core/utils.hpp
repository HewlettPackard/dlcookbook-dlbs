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

#ifndef DLBS_TENSORRT_BACKEND_CORE_UTILS
#define DLBS_TENSORRT_BACKEND_CORE_UTILS

#include <memory>
#include <exception>
#include <vector>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <dirent.h>
#include <cstring>
#include <sys/types.h>
#include <sys/stat.h>
#include <chrono>
#include <sys/stat.h>
#include <semaphore.h>

/**
 * @brief Fill vector with random numbers uniformly dsitributed in [0, 1).
 * @param vec Vector to initialize.
 */
void fill_random(float *vec, const size_t sz);

/**
 * @brief A short wrapper for std::to_string conversion.
 */
template<typename T>
std::string S(const T &t) { return std::to_string(t); }

template<>
std::string S<bool>(const bool &t);

std::string get_env_var(std::string const &var, const std::string& default_val="");


/**
 * @brief The 'printf' like string formatting.
 * 
 * https://stackoverflow.com/a/26221725/575749
 * http://www.cplusplus.com/reference/cstdio/snprintf/
 */
template<typename ... Args>
std::string fmt(const std::string& format, Args ... args) {
    size_t size = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    std::unique_ptr<char[]> buf(new char[ size ]); 
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1);                   // We don't want the '\0' inside
}

/**
 * @brief Various utility methods to work with file system, in particular, 
 * working with raw image datasets.
 */
class fs_utils {
private:
    /**
     * @brief Return human readeable 'errno' description.
     */
    static std::string get_errno() {
        return std::string(strerror(errno));
    }
public:
    /**
     * @brief Return parent directory.
     */
    static std::string parent_dir(std::string dir);
    /**
     * @brief Create directory.
     */
    static int mk_dir(std::string dir, const mode_t mode=0700);
    /**
     * @brief Makes sure that a path 'dir', which is supposed to by a directory,
     * ends with one forward slash '/'
     * @param dir A directory name.
     */
    static std::string normalize_path(std::string dir);
    /**
     * @brief Write binary data to a file.
     * @param fname Name of a file. If exists, it will be overwritten.
     * @param ptr Pointer to a data.
     * @param length Number of bytes to write.
     */
    static void  write_data(const std::string& fname, const void* ptr, std::size_t length);
    /**
     * @brief Read binary data from a file.
     * @param fname Name of a file.
     * @param data_length Number of bytes read.
     * @return Pointer to a data. User is responsible for deallocating this memory (delete [] p;)
     */
    static char* read_data(const std::string& fname, std::size_t& data_length);
    /**
    * @brief Read text file \p name line by line putting lines into \p lines.
    * @param name[in] A file name.
    * @param lines[out] Vector with lines from this file.
    * @return True of file exists, false otherwise
    */
    static bool read_cache(const std::string& dir, std::vector<std::string>& fnames);
    /**
     * @brief Writes a cache with file names if that cache does not exist.
     * @param dir A dataset root directory.
     * @param fnames A list of image file names.
     * @return True if file exists or has been written, false otherwise.
     * 
     */
    static bool write_cache(const std::string& dir, const std::vector<std::string>& fnames);
    /**
     * @brief Prepend \p dir to all file names in \p files
     * @param dir Full path to a dataset
     * @param files Image files with relative file paths.
     */
    static void to_absolute_paths(const std::string& dir, std::vector<std::string>& fnames) {
        for (auto& fname : fnames) {
            fname = dir + fname;
        }
    }
    /**
     * @brief Scan recursively directory \p dir and return image files. List of image files will 
     * contain paths relative to \p dir.
     * @param dir A root dataset directory.
     * @param files A list of image files.
     * @param subdir A subdirectory relative to \p dir. Used for recusrive scanning.
     * @return A list of image files found in \p dir or its subdirectories. Images files are identified
     * by relative paths from \p dir.
     */
    static void get_image_files(std::string dir, std::vector<std::string>& files, std::string subdir="");
    
    static void initialize_dataset(std::string& data_dir, std::vector<std::string>& files);
};

/**
 * @brief Sharded vector iterates forever over provided chunk. For instance, we can have
 * a vector of file names of images. We then can have multiple image readers that will read
 * images from their own chunk.
 */
template <typename T>
class sharded_vector {
private:
    std::vector<T>* vec_;

    size_t num_shards_;
    size_t my_shard_;

    size_t begin_ = 0;
    size_t length_ = 0;

    size_t pos_ = 0;
    
    bool iterate_once_;
public:
    size_t size() const { return vec_->size(); }
    size_t num_shards() const { return num_shards_; }
    size_t my_shard() const { return my_shard_; }
    
    size_t shard_begin() const { return begin_; }
    size_t shard_length() const { return length_; }

    sharded_vector(std::vector<T>& vec, const size_t num_shards, const size_t my_shard,
                   const bool iterate_once=false) 
    : vec_(&vec), num_shards_(num_shards), my_shard_(my_shard), iterate_once_(iterate_once) {
        size_t shard_idx(0),
               shard_len(vec.size() / num_shards),
               ctrl(vec.size() % num_shards);
        
        for (size_t shard=0; shard < num_shards; ++shard) {
            size_t this_shard_len(0);
            if (ctrl > 0) {
                this_shard_len = shard_len + 1;
                ctrl -= 1;
            } else {
                this_shard_len = shard_len;
            }
            if (my_shard == shard) {
                length_ =  this_shard_len;
                begin_ = pos_ = shard_idx;
                break;
            }
            shard_idx += this_shard_len;
        }
        if (length_ == 0) {
            vec_ = nullptr;
        }
        
    }
    
    bool has_next() const {
        return (vec_ && (!iterate_once_ || pos_ - begin_ < length_));
    }
    
    T& next() {
        T& item = (*vec_)[pos_++];
        if (pos_ >= length_ && !iterate_once_)
            pos_ = begin_;
        return item;
    }
};
template <typename T>
std::ostream& operator<<(std::ostream &out, const sharded_vector<T> &v) {
    out << "sharded_vector{size=" << v.size() << ", num_shards=" << v.num_shards() 
        << ", my_shard=" << v.my_shard() << ", shard=[" << v.shard_begin() << ", " << (v.shard_begin() + v.shard_length()) << ")}";
    return out;
}


/**
 * @brief Simple timer to measure execution times.
 */
class timer {
public:
  timer() {
    restart();
  }
  /**
   * @brief Restart timer.
   */
  void restart() {
    start_ = std::chrono::high_resolution_clock::now();
  }
  /**
   * @brief Return time in milliseconds elapsed since last 
   * call to @see restart.
   */
  float ms_elapsed() const {
    const auto now = std::chrono::high_resolution_clock::now();
    return float(std::chrono::duration_cast<std::chrono::microseconds>(now - start_).count()) / 1000.0;
  }
private:
  std::chrono::high_resolution_clock::time_point start_;
};

/**
 * @brief Class that can track batch processing times including pure inference
 * time and total batch time including data transfer overhead (CPU <--> GPU).
 * Useful when CPU-GPU data transfers are not pipelined.
 */
class time_tracker {
private:
  // The 'inference' time is a time without CPU <--> GPU data transfers, the 'total' time
  // is a complete time including those data transfers. The 'inference' and 'total' are used
  // if report_frequency is > 0.
  // The vectors with 'all_' prefix store batch times for entire benchmark.
  std::vector<float> batch_times_;  //!< Total batch times including CPU <--> GPU data transfers.
  std::vector<float> infer_times_;  //!< Pure inference batch times excluding data transfers overhead.
  
  timer batch_timer_;  //!< Timer to measure total batch times.
  timer infer_timer_;  //!< Timer to measure pure inference times.
  
  int iter_idx_;       //!< Index when current iteration has started. Used when user requested intermidiate output.
  
  size_t num_batches_; //!< In case we need to reset the state,
public:
    /**
     * @brief Initializes time tracker.
     * @param num_batches Number of input data instances associated with
     * each element in time_tracker#batch_times_ and time_tracker#infer_times_.
     */
    explicit time_tracker(const size_t num_batches) : num_batches_(num_batches) {
        reset();
    }
    
    void reset() {
        infer_times_.reserve(num_batches_);
        batch_times_.reserve(num_batches_);
        iter_idx_ = 0;
    }

    void batch_started() {batch_timer_.restart();};
    void infer_started() {infer_timer_.restart();};
    void infer_done(const float ms = -1)  {
        if (ms >= 0) {
            infer_times_.push_back(ms);
        } else {
            infer_times_.push_back(infer_timer_.ms_elapsed());
        }
    };
    void batch_done(const float ms = -1)  {
        if (ms >= 0) {
            batch_times_.push_back(ms);
        } else {
            batch_times_.push_back(batch_timer_.ms_elapsed());
        }
    };
    /** @brief Start new iteration*/
    void new_iteration() { iter_idx_ = infer_times_.size(); }
    
    std::vector<float>& get_batch_times() { return batch_times_; }
    std::vector<float>& get_infer_times() { return infer_times_; }
    
    float last_batch_time() const { return batch_times_.back(); }
    float last_infer_time() const { return infer_times_.back(); }
    
    int get_iter_idx() const { return iter_idx_; }
};

/**
 * @brief Estimating average based on input stream without storign complete history.
 */
class running_average {
private:
    size_t i_ = 1;
    double average_ = 0;
public:
    void update(const float val) {
        average_ = average_ + (static_cast<double>(val) - average_) / i_;
        i_ ++;
    }
    double value() const { return average_; }
    size_t num_steps() const { return i_-1; }
};
std::ostream& operator<<(std::ostream &out, const running_average &ra);

class PictureTool {
public:
    template<typename T> struct pixel{};

    template<typename T>
    static void opencv2tensor(unsigned char* opencv_data, const int nchannels, const int height,
                              const int width, T* tensor);
};

template<> struct PictureTool::pixel<float> { static const char encoding = 1; };
template<> struct PictureTool::pixel<unsigned char> { static const char encoding = 10; };

/**
 * @brief A file wrapper that can use either C approach or C++ streams
 * to read binary data.
 */
class binary_file {
private:
    int fd_ = -1;                          //!< File descriptor.
    bool advise_no_cache_ = false;         //!< If true, advise OS not to cache file.
    const std::string dtype_;              //!< Matrix data type in a binary file ('float', 'uchar').
    std::vector<unsigned char> buffer_;    //!< If images are stored as unsigned chars, use this buffer.
public:
    binary_file(const std::string& dtype="float",
                const bool advise_no_cache=false);
    virtual ~binary_file() { close(); }
    bool is_opened();
    void open(const std::string& fname);
    void close();
    ssize_t read(float* dest, const size_t count);
    void allocate_if_needed(const size_t count);
};

class allocator {
public:
    virtual void allocate(float *&buf, const size_t sz) = 0;
    virtual void deallocate(float *&buf) = 0;
};

class standard_allocator : public  allocator {
public:
    void allocate(float *&buf, const size_t sz) override {
        buf = new float[sz];
    }
    void deallocate(float *&buf) override {
        if (buf) {
            delete [] buf;
            buf = nullptr;
        }
    }
};

/**
 * @brief A simple multiprocess synchronization. If running with docker, use --ipc=host.
 */
class process_barrier {
private:
    std::string name_;
    sem_t *handle_ = SEM_FAILED;
    int rank_;
    int count_;
    bool post_mode_;
private:
    void init();
public:
    process_barrier(std::string specs);
    process_barrier(const std::string& name, const int rank, const int count);
    virtual ~process_barrier() {
        close();
    }
    bool good() const { return handle_ != SEM_FAILED; }
    int rank() const { return rank_; }
    int count() const { return count_; }
    /**
     * @brief Waits until all processes reach a synchronization point.
     * @return 0 on success; on error, -1 is returned, errno is set and process
     *         barrier transitions to undefined state.
     */
    int wait();
    void close();
};

#endif
