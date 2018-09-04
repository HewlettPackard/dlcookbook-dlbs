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

// The tensor type that we use to store images in host memory. It is a compile time
// definition and is defined in CMakeLists.txt.
//   - single precision (HOST_DTYPE_FP32: float) will take more host memory and also
//     will put more stress on host to device data transfers.
//   - unsigned char (HOST_DTYPE_INT8: unsigned char) data type will take less host
//     memory and will not stress PCIe lanes. However, due to TensorRT API, we will
//     need to cast unsigned char tensor to float tensor in GPU.
// Based on my experiments, unsigned char type provides much better results when
// multiple GPUs are used or in case of a non-compute intensive model. 
#if defined HOST_DTYPE_FP32
    typedef float host_dtype;
#elif defined HOST_DTYPE_INT8
    typedef unsigned char host_dtype;
#else
    #error "Unknown host data type. Define either HOST_DTYPE_FP32 or HOST_DTYPE_INT8"
#endif

// Provide DEBUG_LOG to enable detailed logging of some of the components. It is set
// in CMakeLists.txt file.
#ifdef DEBUG_LOG
    #define DLOG(msg) do { std::cerr << msg << std::endl; } while(0)
#else
    #define DLOG(msg)
#endif

/**
 * @brief Fill vector with random numbers uniformly dsitributed in [0, 1).
 * @param vec Vector to initialize.
 * @param sz Length of te vector.
 */
void fill_random(float *vec, const size_t sz);

/**
 * @brief Fill vector with random numbers uniformly dsitributed in [0, 255].
 * @param vec Vector to initialize.
 * @param sz Length of te vector.
 */
void fill_random(unsigned char *vec, const size_t sz);

/**
 * @brief A short wrapper for std::to_string conversion.
 * @param t Something that will be converted to std::string.
 * @return std::string
 */
template<typename T> std::string S(const T &t) { return std::to_string(t); }

/**
 * @brief A template specialziation of S for bool type.
 * @param t Boolean value to convert to std::string.
 * @return std::string (true" or "false")
 */
template<> std::string S<bool>(const bool &t);


template<typename T> T from_string(const char* const val);
template<> std::string from_string<std::string>(const char* const val);
template<> int from_string<int>(const char* const val);
template<> bool from_string<bool>(const char* const val);


/**
 * @brief Class to access environment variables that affect benchmark behaviour.
 * 
 * Some of the configuration parameters that either are experimental or are not
 * used very often are environmen variables. 
 * 
 * Boolen variables can be configured with the following string values:
 *   - **true**:  1, yes, true, on
 *   - **false**: 0, no, false, off
 * 
 * This class is a helper class to access these variables:
 * 
 *   - @link environment#file_reader DLBS_TENSORRT_FILE_READER @endlink
 *     File reader to use with custom file format.
 * 
 *   - @link environment#remove_files_from_os_cache DLBS_TENSORRT_REMOVE_FILES_FROM_OS_CACHE @endlink
 *     Whether opened files with images should be removed from cache (applied when file reader is
 *     'default'). Maybe time consuming process.
 * 
 *   - @link environment#pinned_memory DLBS_TENSORRT_ALLOCATE_PINNED_MEMORY @endlink
 *     Allocate host memory for batches as pinned memory.
 * 
 *   - @link environment#allow_image_dataset DLBS_TENSORRT_ALLOW_IMAGE_DATASET @endlink
 *     Whether raw images can be used as dataset when user specifies
 *     that on a command line.
 * 
 *   - @link environment#synch_benchmarks DLBS_TENSORRT_SYNCH_BENCHMARKS @endlink
 *     Used to synch processes in case multi-process benchmarks.
 * 
 *   - @link environment#storage_block_size DLBS_TENSORRT_STORAGE_BLOCK_SIZE @endlink
 *     Block size when reader is 'directio' reader.
 * 
 *   - @link environment#dataset_split DLBS_TENSORRT_DATASET_SPLIT @endlink
 *     How input dataset files are split across multiple prefetchers.
 * 
 *   - @link environment#overlap_copy_compute DLBS_TENSORRT_OVERLAP_COPY_COMPUTE @endlink
 *     Overlap eecution of inference kernel with host-to-device data transfers.
 * 
 *   - @link environment#inference_impl_ver DLBS_TENSORRT_INFERENCE_IMPL_VER @endlink
 *     Implementation version of inference function. Version 1 is the original
 *     sequential implementation and is kept for historical reasons.
 */
class environment {
public:
    /**
     * @brief A file reader to use with custom file format. Default value is **directio**.
     * 
     * Is only applicable when dataset type is either *tensors1* or *tensors2*.\n
     * DLBS_TENSORRT_FILE_READER = ['directio', 'default']
     * 
     *   - **directio** Default option that enables reader that uses direct io (files
     *     are opened with O_DIRECT flag to bypass OS caches.
     *   - **default** Traditional file reader. Files read with this reader may be cached
     *     by OS. See @link environment#remove_files_from_os_cache @endlink for how to
     *     intruct OS to remove files from cache with this reader. This reader should be 
     *     used in cases when file systems do not support direct IO calls.
     *     
     * Originally, we wanted to benchmark HPC storage with inferencing workloads. To ensure
     * that we always stream data from disks, we implemented traditional data reader and before
     * closing files we used \c "posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED)" to
     * instruct OS to remove this file from cache. This however does not elliminate caching
     * during reading and also \c posix_fadvise call itself introduces significant
     * overhead. So, later on we implemented reader that uses direct IO.
     */
    static std::string file_reader();
    
    /**
     * @brief Whether opened files with images should be removed from cache. Default value is **true**.
     * 
     * DLBS_TENSORRT_REMOVE_FILES_FROM_OS_CACHE=['true', 'false']
     * This variable has effect only when file reader is 'default' (@link file_reader @endlink).
     * 
     * If it is false, do not advise OS to not to cache dataset files. By default, data readers advise
     * OS that files they open will not be needed in future meaning that OS should not cache these files
     * (\c "posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED)"). This can be used to emulate presence of a 
     * large dataset and to benchmark storage.
     */
    static bool remove_files_from_os_cache();
    
    /**
     * @brief Whether host memory for batches should be allocated as pinned memory. Default value is **true**.
     * 
     * DLBS_TENSORRT_ALLOCATE_PINNED_MEMORY = ['true', 'false']\n
     * If true, host memory for input data is allocated as pinned memory. This improves performance of host to
     * device data transfers.
     */
    static bool pinned_memory();
    /**
     * @brief Allow using the dataset that reads raw JPEG images. Default is **false**.
     * 
     * DLBS_TENSORRT_ALLOW_IMAGE_DATASET = ['true', 'false']\n
     * The inference benchmark tool can stream data from raw JPEG files (images) or custom binary files (tensors1,
     * tensors4). We never optimzied input pipelines for raw JPEG images and this is very slow. For this reason,
     * even if user provides dataset that consists of raw JPEG images, we disable this and print warning messages.
     * In order to enable this type of dataset, user needs to define the above mentioned variable.
     */
    static bool allow_image_dataset();
    
    /**
     * @brief Used to synch processes in case multi-process benchmarks. Default is **""**.
     * 
     * DLBS_TENSORRT_SYNCH_BENCHMARKS = 'PROCESS_RANK,NUM_PROCESSES,SEM_FILE'
     * 
     * The inference benchmark tool can use multiple GPUs to run inference benchmarks. For instance, if you have an 
     * 8-GPU box, one process can use all 8 GPUs. Sometimes, depending on a HW configuration, it can be more beneficial
     * in terms of performance to run multiple processes with smaller number of GPUs. For instance, if you have a two
     * socket NUMA server in a 4 by 1 configuration (4 GPUs connected to one CPU and another 4 GPUs connected to another
     * CPU), you can run two benchmark processes on these two CPUs pinning these processes with numactl, something like:
     * @code
     *     numactl --localalloc --physcpubind=0-17 ...
     * @endcode
     * In this particular case, the question is how to measure the overall performance across all benchmarks. There are
     * several ways, one of which is to synch processes at the start and at the end and then use the total time to compute
     * aggregated throughput. The DLBS_TENSORRT_SYNCH_BENCHMARKS variable is used exactly for this purpose. The format of
     * this variable is the following: my_rank,num_processes,name where:
     *    - num_processes is the total number of processes to synchronize
     *    - my_rank is the process identifier and 
     *    - name is a semaphore name (semaphores are used for cross-process synchronization, think about it as a file name
     *      in /dev/shm).
     * If you run benchmarks in docker containers, do not forget to run containers with --ipc=host. For instance:
     * @code
     *   # Process 1
     *   export DLBS_TENSORRT_SYNCH_BENCHMARKS=0,2,dlbs_ipc
     * 
     *   # Process 2
     *   export DLBS_TENSORRT_SYNCH_BENCHMARKS=1,2,dlbs_ipc
     *@endcode
     */
    static std::string synch_benchmarks();
    /**
     * @brief When file reader is a **directio** file reader, use this block size. Default is **512**.
     * 
     * DLBS_TENSORRT_STORAGE_BLOCK_SIZE = integer number
     */
    static int storage_block_size();
    
    /**
     * @brief How input dataset files are split across multiple prefetchers. Default is **uniform**.
     * 
     * DLBS_TENSORRT_DATASET_SPLIT = ['uniform', 'nosplit']
     * 
     * This environment variable defines how multiple readers split dataset files. By default, this split is uniform (uniform).
     * Each reader gets its own unique collection of files. If there are more readers than files, some readers will not get
     * their files and benchmark application will exit.
     * Another option is when each reader reads entire dataset (nosplit) randomly shuffling files during the startup.
     */
    static std::string dataset_split();
    /**
     * @brief Overlap execution of inference kernel with host to device data transfers. Default is **true**.
     * 
     * DLBS_TENSORRT_OVERLAP_COPY_COMPUTE = ['true', 'false']\n
     * Overlapping these two operations using two CUDA streams is a key to good performance.
     */
    static bool overlap_copy_compute();
    
    /**
     * @brief Implementation version of inference function. Default is **""**.
     * 
     * DLBS_TENSORRT_INFERENCE_IMPL_VER =['1', ''] \n
     * Version 1 is the original sequential implementation and is kept for historical reasons.
     * 
     * The inference benchmark tool provides two implementations of an inference function. Version 1, or legacy, is
     * the original implementation that uses default CUDA stream to copy data and do inference. The high level logic
     * is the following: (1) copy data to GPU, (2) do inference and (3) copy results back to host memory. To enable
     * this implementation, export the following variale:
     * @code
     *   export DLBS_TENSORRT_INFERENCE_IMPL_VER=1
     * @endcode
     * This implementation is sort of OK when input data arrives at slow pace and there is no much benefit from a
     * better implementation. Default choice is the implementation that uses two CUDA streams and overlaps copy/compute
     * phases. This implementation provides a better throughput when input requests arrive at high frequency.
     */
    static std::string inference_impl_ver();
private:
    /**
    * @brief Returns environmental variable.
    * @param name Environmental variable name
    * @param value Default value if variable is not defined
    * @return Value of an environmental variable or its default value
    */
    template<typename T>
    static T variable(std::string const &name, const T& value) {
        char *val = getenv( name.c_str() );
        return val == nullptr ? value : from_string<T>(val);
    }
};

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
     * @brief Makes sure that a path 'dir', which is supposed to be a directory,
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
    * @brief Read file names from a cache located in \p dir directory.
    * @param dir is the directory to search cache in.
    * @param fnames Vector of file names read from cache file.
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
     * @brief Prepend \p dir to all file names in \p fnames
     * @param dir Full path to a dataset
     * @param fnames Image files with relative file paths.
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
    
    static int get_direct_io_block_size();
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

/**
 * @brief Compute variance and other statistics based on 
 *        Welford's method.
 */
class stats {
private:
    double mean_ = 0;
    double variance_ = 0;
    double min_ = 0;
    double max_ = 0;
public:
    stats(const std::vector<float>& nums);

    double mean() const { return mean_; }
    double variance() const { return variance_; }
    double stdev() const { return variance_ >= 0 ? std::sqrt(variance_) : 0.0; }
    double min() const { return min_; }
    double max() const { return max_; }
};
std::ostream& operator<<(std::ostream &out, const stats &s);

/**
 * @brief Converts OpenCV matrix data to C-style 3D array.
 */
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
 * @brief Abstract class for file readers. 
 * 
 * File readers are used to read images from dataset files. In particular, provided
 * implementations read data from binary files of custom, quite simplified, format.
 */
class abstract_reader {
public:
    virtual bool is_opened() = 0;
    virtual bool open(const std::string& fname) = 0;
    virtual void close() = 0;
    virtual ssize_t read(host_dtype* dest, const size_t count) = 0;
    virtual void allocate_if_needed(const size_t count) = 0;
};

/**
 * @brief A basic implementation that uses standard IO calls to read binary data.
 * 
 * This reader can be configured with environment variable @link environment#remove_files_from_os_cache @endlink
 * to instruct OS to remove opened files on close. Is suitable for file systems that do not support direct IO.
 */
class reader : public abstract_reader {
private:
    int fd_ = -1;                          //!< File descriptor.
    bool advise_no_cache_ = false;         //!< If true, advise OS not to cache file.
    const std::string dtype_;              //!< Matrix data type in a binary file ('float', 'uchar').
    std::vector<unsigned char> buffer_;    //!< If images are stored as unsigned chars, use this buffer.
public:
    reader(const std::string& dtype="float",
           const bool advise_no_cache=false);
    virtual ~reader() { close(); }
    bool is_opened();
    bool open(const std::string& fname);
    void close();
    ssize_t read(host_dtype* dest, const size_t count);
    void allocate_if_needed(const size_t count);
};

/**
 * @brief An implementation of a file reader that uses DIRECT IO.
 * 
 * This is useful to bypass system caches and make sure that files
 * that we read are not cached. We use this to benchmark storage to
 * make sure we always stream data from it.
 */
class direct_reader : public abstract_reader {
private:
    enum class data_type {
        dt_float,
        dt_unsigned_char
    };
    int block_sz_ = 512;                   //!< Block size for O_DIRECT. Use DLBS_TENSORRT_STORAGE_BLOCK_SIZE to overwrite this value.
    int fd_ = -1;                          //!< File descriptor.
    const data_type dtype_;                //!< Data type for batch tensor.
    
    unsigned char* buffer_ = nullptr;      // If images are stored as unsigned chars, use this buffer. This is an aligned 
                                           // buffer on the block_sz_ boundary and its size is a multiple of block_sz_.
    size_t buffer_size_ = 0;               // Size of the buffer in bytes.
    size_t buffer_offset_ = 0;             // Offset in buffer if we have some bytes from previous read. In this case next read must 
                                           // write to buffer_ starting from block_sz_ position. The value of buffer_offset_ is
                                           // always < block_sz_.
public:
    /**
     * @brief Class constructor
     * @param dtype Data type used to store images. One of 'float' or 'uchar'.
     */
    direct_reader(const std::string& dtype="float");
    virtual ~direct_reader() { close(); }
    bool is_opened();
    bool open(const std::string& fname);
    void close();
    /**
     * @brief Read 'count' elements from file.
     */
    ssize_t read(host_dtype* dest, const size_t count);
    // This is deprecated and does nothing. All allocations
    // are done on the fly in read function.
    void allocate_if_needed(const size_t count) {}
private:
    void allocate(const size_t new_sz);
    void deallocate();
};

/**
 * @brief A base class for memory allocators.
 * 
 * The benchmark tool can use either standard or pinned allocators. Pinned memory
 * significantly improved host to device transfer performance.
 */
class allocator {
public:
    virtual void allocate(float *&buf, const size_t sz) = 0;
    virtual void allocate(unsigned char *&buf, const size_t sz) = 0;

    virtual void deallocate(float *&buf) = 0;
    virtual void deallocate(unsigned char *&buf) = 0;
};

/**
 * @brief An implementation of an allocator that uses standard C++ new/delete operators.
 */
class standard_allocator : public  allocator {
public:
    void allocate(float *&buf, const size_t sz) override { buf = new float[sz]; }
    void allocate(unsigned char *&buf, const size_t sz) override { buf = new unsigned char[sz]; }

    void deallocate(float *&buf) override {
        if (buf) {
            delete [] buf;
            buf = nullptr;
        }
    }
    void deallocate(unsigned char *&buf) override {
        if (buf) {
            delete [] buf;
            buf = nullptr;
        }
    }
};

/**
 * @brief A simple multiprocess synchronization. 
 * 
 * If running with docker, use --ipc=host. Sometimes, when large number of GPUs
 * is used, it's beneficial to run multiple instances of a benchmark tool each
 * runnning, for instance, on its own CPU. To better estimate aggregated throughput,
 * these processes can synch with each other. We use this class to do that.
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
