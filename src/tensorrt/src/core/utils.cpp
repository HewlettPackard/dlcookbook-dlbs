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


#include "core/utils.hpp"

#define _XOPEN_SOURCE 700
#include <unistd.h>
#include <fcntl.h>
#include <chrono>
#include <thread>
#include <sstream>
#include <algorithm>

void fill_random(float *vec, const size_t sz) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine(rnd_device());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  auto gen = std::bind(dist, mersenne_engine);
  std::generate(vec, vec+sz, gen);
}

void fill_random(unsigned char *vec, const size_t sz) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine(rnd_device());
  std::uniform_int_distribution<unsigned int> dist(0, 255);
  auto gen = std::bind(dist, mersenne_engine);
  std::generate(vec, vec+sz, gen);
}

template<>
std::string S<bool>(const bool &t) { return (t ? "true" : "false"); }

template<>
std::string from_string<std::string>(const char* const val) { return std::string(val); }

template<>
int from_string<int>(const char* const val) { return std::stoi(val); }

template<>
bool from_string<bool>(const char* const val) { 
    std::string str = std::string(val);
    std::transform(str.begin(), str.end(), str.begin(), ::tolower);
    return (str == "1" || str == "on" || str == "yes" || str == "true");
}

std::string environment::file_reader() {
    return environment::variable<std::string>("DLBS_TENSORRT_FILE_READER", "");
}
bool environment::remove_files_from_os_cache() {
    return environment::variable<bool>("DLBS_TENSORRT_REMOVE_FILES_FROM_OS_CACHE", "yes");
}

bool environment::pinned_memory() {
    return environment::variable<bool>("DLBS_TENSORRT_ALLOCATE_PINNED_MEMORY", "yes");
}

bool environment::allow_image_dataset() {
    return environment::variable<bool>("DLBS_TENSORRT_ALLOW_IMAGE_DATASET", "no");
}

std::string environment::synch_benchmarks() {
    return environment::variable<std::string>("DLBS_TENSORRT_SYNCH_BENCHMARKS", "");
}

int environment::storage_block_size() {
    return environment::variable<int>("DLBS_TENSORRT_STORAGE_BLOCK_SIZE", 512);
}

std::string environment::dataset_split() {
    return environment::variable<std::string>("DLBS_TENSORRT_DATASET_SPLIT", "uniform");
}

bool environment::overlap_copy_compute() {
    return environment::variable<bool>("DLBS_TENSORRT_OVERLAP_COPY_COMPUTE", "yes");
}

std::string environment::inference_impl_ver() {
    return variable<std::string>("DLBS_TENSORRT_INFERENCE_IMPL_VER", "");
}


std::string fs_utils::parent_dir(std::string dir) {
    if (dir.empty())
        return "";
    dir = normalize_path(dir);
    dir = dir.erase(dir.size() - 1);
    const auto pos = dir.find_last_of("/");
    if(pos == std::string::npos)
        return "";
    return dir.substr(0, pos);
}

int fs_utils::mk_dir(std::string dir, const mode_t mode) {
    struct stat sb;
    if (stat(dir.c_str(), &sb) != 0 ) {
        // Path does not exist
        const int ret = mk_dir(parent_dir(dir), mode);
        if(ret < 0)
            return ret;
        return mkdir(dir.c_str(), mode);
    }
    if (!S_ISDIR(sb.st_mode)) {
        // Path exists and is not a directory
        return -1;
    }
    // Path exists and is a directory.
    return 0;
}

std::string fs_utils::normalize_path(std::string dir) {
    const auto pos = dir.find_last_not_of("/");
    if (pos != std::string::npos && pos + 1 < dir.size())
        dir = dir.substr(0, pos + 1);
    dir += "/";
    return dir;
}

void fs_utils::write_data(const std::string& fname, const void* ptr, std::size_t length) {
    if (fname != "") {
        std::ofstream file(fname.c_str(), std::ios::binary);
        if (file.is_open()) {
            file.write(static_cast<const char*>(ptr), static_cast<std::streamsize>(length));
        }
    }
}

char* fs_utils::read_data(const std::string& fname, std::size_t& data_length) {
    if (fname != "") {
        std::ifstream file(fname.c_str(), std::ios::binary|std::ios::ate);
        if (file.is_open()) {
            auto fsize = file.tellg();
            char* data = new char[fsize];
            file.seekg(0, std::ios::beg);
            file.read(data, fsize);
            data_length = static_cast<std::size_t>(fsize);
            return data;
        }
    }
    data_length = 0;
    return nullptr;
}

bool fs_utils::read_cache(const std::string& dir, std::vector<std::string>& fnames) {
    std::ifstream fstream(dir + "/" + "dlbs_image_cache");
    if (!fstream) return false;
    std::string fname;
    while (std::getline(fstream, fname))
        fnames.push_back(fname);
    return true;
}
    
bool fs_utils::write_cache(const std::string& dir, const std::vector<std::string>& fnames) {
    struct stat sb;
    const std::string cache_fname = dir + "/" + "dlbs_image_cache";
    if (stat(cache_fname.c_str(), &sb) == 0)
        return true;
    std::ofstream fstream(cache_fname.c_str());
    if (!fstream)
        return false;
    for (const auto& fname : fnames) {
        fstream << fname << std::endl;
    }
    return true;
}

void fs_utils::get_image_files(std::string dir, std::vector<std::string>& files, std::string subdir) {
    // Check that directory exists
    struct stat sb;
    if (stat(dir.c_str(), &sb) != 0) {
        std::cerr << "[get_image_files       ]: skipping path ('" << dir << "') - cannot stat directory (reason: " << get_errno() << ")" << std::endl;
        return;
    }
    if (!S_ISDIR(sb.st_mode)) {
        std::cerr << "[get_image_files       ]: skipping path ('" << dir << "') - not a directory" << std::endl;
        return;
    }
    // Scan this directory for files and other directories
    const std::string abs_path = dir + subdir;
    DIR *dir_handle = opendir(abs_path.c_str());
    if (dir_handle == nullptr)
        return;
    struct dirent *de(nullptr);
    while ((de = readdir(dir_handle)) != nullptr) {
        const std::string dir_item(de->d_name);
        if (dir_item == "." || dir_item == "..") {
            continue;
        }
        bool is_file(false), is_dir(false);
        if (de->d_type != DT_UNKNOWN) {
            is_file = de->d_type == DT_REG;
            is_dir = de->d_type == DT_DIR;
        } else {
            const std::string dir_item_path = dir + subdir + dir_item;
            if (stat(dir_item_path.c_str(), &sb) != 0) {
                continue;
            }
            is_file = S_ISREG(sb.st_mode);
            is_dir = S_ISDIR(sb.st_mode);
        }
        if (is_dir) {
            get_image_files(dir, files, subdir + dir_item + "/");
        } else if (is_file) {
            const auto pos = dir_item.find_last_of('.');
            if (pos != std::string::npos) {
                std::string fext = dir_item.substr(pos + 1);
                std::transform(fext.begin(), fext.end(), fext.begin(), ::tolower);
                if (fext == "jpg" || fext == "jpeg" || fext == "tensors") {
                    files.push_back(subdir + dir_item);
                }
            }
        } {
        }
    }
    closedir(dir_handle);
}

void fs_utils::initialize_dataset(std::string& data_dir, std::vector<std::string>& files) {
    data_dir = fs_utils::normalize_path(data_dir);
    if (!fs_utils::read_cache(data_dir, files)) {
        fs_utils::get_image_files(data_dir, files);
        std::cout << "[image_provider        ]: found " + S(files.size()) +  " image files in file system." << std::endl;
        if (!files.empty()) {
            //if (!fs_utils::write_cache(data_dir, files)) {
            //    std::cout << "[image_provider        ]: failed to write file cache." << std::endl;
            //}
        }
    } else {
        std::cout << "[image_provider        ]: read " + S(files.size()) +  " from cache." << std::endl;
        if (files.empty()) { 
            std::cout << "[image_provider        ]: found empty cache file. Please, delete it and restart DLBS. " << std::endl;
        }
    }
    if (files.empty()) {
        std::cout << "[image_provider        ]: no input data found, exiting." << std::endl;
    }
    fs_utils::to_absolute_paths(data_dir, files);
}

int fs_utils::get_direct_io_block_size() {
    return environment::storage_block_size();
}


std::ostream& operator<<(std::ostream &out, const running_average &ra) {
    std::cout << "running_average{size=" << ra.num_steps() << ", value=" << ra.value() << "}";
    return out;
}


stats::stats(const std::vector<float>& nums) {
    if (nums.empty()) return;
    double prev_mean(0), prev_s(0), cur_s(0);
    prev_mean = mean_ = min_ = max_ = static_cast<double>(nums[0]);
    for (size_t i=1; i<nums.size(); ++i) {
        const auto v = static_cast<double>(nums[i]);
        //
        mean_ = prev_mean + (v - prev_mean) / (i+1);
        cur_s = prev_s + (v - prev_mean) * (v - mean_);
        //
        min_ = std::min(min_, v);
        max_ = std::max(max_, v);
        //
        prev_mean = mean_;
        prev_s = cur_s;
    }
    variance_ = nums.size() > 1 ? cur_s / (nums.size() - 1) : 0.0;
}

std::ostream& operator<<(std::ostream &out, const stats &s) {
    std::cout << "stats{min=" << s.min() << ", max=" << s.max() << ", mean=" << s.mean() << ", variance=" << s.variance() << ", stdev=" << s.stdev() << "}";
    return out;
}

template<typename T>
void PictureTool::opencv2tensor(unsigned char* opencv_data, const int nchannels, const int height,
                                const int width, T* tensor) {
    const int channel_size = height * width;
    for(int j = 0; j < height; ++j) {
        const int row_idx = nchannels * width * j;
        for(int i = 0; i < width; ++i) {
            const auto col_rel_idx = i * nchannels;
            const int b = opencv_data[row_idx + col_rel_idx] ;            // b
            const int g = opencv_data[row_idx + col_rel_idx + 1];         // g
            const int r = opencv_data[row_idx + col_rel_idx + 2];         // r
            // [RGB, H, W]
            tensor[width * j + i                              ] = r;
            tensor[width * j + i + channel_size               ] = g;
            tensor[width * j + i + channel_size + channel_size] = b;
        }
    }
}
template void PictureTool::opencv2tensor<float>(unsigned char* opencv_data, const int nchannels, const int height, const int width, float* tensor);
template void PictureTool::opencv2tensor<unsigned char>(unsigned char* opencv_data, const int nchannels, const int height, const int width, unsigned char* tensor);


reader::reader(const std::string& dtype,
               const bool advise_no_cache) : advise_no_cache_(advise_no_cache), dtype_(dtype) {
}

bool reader::is_opened() {
    return (fd_ > 0);
}

bool reader::open(const std::string& fname) {
    fd_ = ::open(fname.c_str(), O_RDONLY);
    if (advise_no_cache_) {
        fdatasync(fd_);
    }
    return is_opened();
}

void reader::close() {
    if (fd_ > 0) {
        if (advise_no_cache_) {
            posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
        }
        ::close(fd_);
        fd_ = -1;
    }
}

ssize_t reader::read(host_dtype* dest, const size_t count) {
    ssize_t read_count;
    if (buffer_.empty()) {
        const ssize_t num_bytes_read = ::read(fd_, (void*)dest,  sizeof(host_dtype)*count);
        read_count = num_bytes_read / sizeof(host_dtype);
    } else {
        const ssize_t num_bytes_read = ::read(fd_, (void*)buffer_.data(),  sizeof(unsigned char)*count);
        if (num_bytes_read > 0) {
            std::copy(buffer_.data(), buffer_.data() + num_bytes_read, dest);
        }
        read_count = num_bytes_read;
    }
    return read_count;
}

void reader::allocate_if_needed(const size_t count) {
#if defined HOST_DTYPE_SP32
    // To convert from unsigned char in files to SP32 in host memory
    if (dtype_ == "uchar" && buffer_.size() != count) {
        buffer_.resize(count);
    }
#else
    if (dtype_ == "float") {
        throw "With unsigned char host data type files with SP32 elements are not supported.";
    }
#endif
}


// ---------------------------------------------------------------------------
direct_reader::direct_reader(const std::string& dtype) : 
    dtype_(dtype == "float" ? data_type::dt_float : data_type::dt_unsigned_char) {

    if (dtype != "float" && dtype != "uchar") {
        throw std::invalid_argument(
            "Invalid image data type (expecting 'float' or 'uchar') but given '" + dtype + "'."
        );
    }
    if (dtype_ == data_type::dt_float) {
        throw std::invalid_argument(
            "Single precision format is not supported yet."
        );
    }
    
    block_sz_ = fs_utils::get_direct_io_block_size();
    DLOG(fmt("[direct reader] direct_reader::direct_reader(dtype=%s, block size=%u).", dtype.c_str(), block_sz_));
}

bool direct_reader::is_opened() {
    return (fd_ > 0);
}

bool direct_reader::open(const std::string& fname) {
    DLOG(fmt("[direct reader] opening file (fname=%s).", fname.c_str()));
    // Reser buffer offset each time new file is opened.
    buffer_offset_ = 0;
    // http://man7.org/linux/man-pages/man2/open.2.html
    fd_ = ::open(fname.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ < 0) {
        std::cerr << "Input file (" << fname << ") has not been opened. Errno is " << errno << "." << std::endl;
        if (errno == EINVAL) {
            std::cerr << "This is the EINVAL error (the filesystem does not support the O_DIRECT flag)." << std::endl;
        }
    }
    return is_opened();
}

void direct_reader::close() {
    DLOG("[direct reader] closing file.");
    if (is_opened()) {
        ::close(fd_);
        fd_ = -1;
    }
}

/** Implementation comments.
 *    buffer_offset_: Number of bytes associated with previous read. If it's not 0,
 *                    the `block_sz_ - buffer_offset_` value defines how many bytes we
 *                    have for current batch.
 *    In the comments below I use number of bytes/number of elements interchangeably 
 *    because we always read elements of size 1 byte.
 *    This method will fail if batch is less than 1 block.
 */
ssize_t direct_reader::read(host_dtype* dest, const size_t count) {
    // Number of elements preloaded from last read for this batch.
    const size_t nelements_preloaded = (buffer_offset_ == 0 ? 0 : block_sz_ - buffer_offset_);
    // Number of elements to read. This number is 'aligned' on block_sz_ boundary. But it's length
    // is not necesserily a multiple of block_sz_. Alignment here means that we either write into
    // buffer starting from 1st block (offset is 0), or starting from 2nd block (offset is block_sz_).
    const size_t nelements_to_read = count - nelements_preloaded;
    // Number of elements in a last block. If it's zero, it means we need to read a whole number of
    // blocks. If it's not 0, this number will later become buffer_offset_ for a next batch.
    const size_t last_block_nelements = nelements_to_read % block_sz_;
    // Number of blocks/bytes to read.
    const int num_blocks_to_read = nelements_to_read / block_sz_ + (last_block_nelements == 0 ? 0 : 1);
    const size_t nbytes_to_read = num_blocks_to_read * block_sz_;
    
    // Allocate aligned memory if it has not been allocated.
    allocate(block_sz_ * (1 + count / block_sz_ + (count % block_sz_ == 0 ? 0 : 1)));
    // Read from file.
    const ssize_t num_bytes_read = ::read(
        fd_,                                                            // File descriptor.
        (void*)(buffer_ + (nelements_preloaded == 0 ? 0 : block_sz_)),  // Write offset is at most one block.
        nbytes_to_read                                                  // Number of bytes to read, always a whole number of blocks.
    );
    if (num_bytes_read < 0) {
        DLOG(fmt("Error reading file (errno=%d). Debug me.", int(errno)));
        // Can it be the case that when I try to read something after I reached EOF, read when working
        // on files opened with O_DIRECT fails with EINVAL error instead of returning 0 bytes?
        // I am getting this error on a very last batch.
        return 0;
    }
    if (num_bytes_read == 0) {
        // This is fine. The higher level code will close this file and will open another one.
        DLOG("Read 0 bytes, EOF?");
        return 0;
    }
    
    // Copy data from internal buffer to user provided dest buffer
    const size_t ntotal_bytes = nelements_preloaded + std::min(nelements_to_read, size_t(num_bytes_read));
    int read_idx(buffer_offset_);
    for (size_t i=0; i<ntotal_bytes; ++i) {
        dest[i] = static_cast<host_dtype>(buffer_[read_idx++]);
    }
#ifdef DEBUG_LOG
    std::cerr << "[direct reader] reading data (buffer_offset_=" << buffer_offset_<< ", nelements_preloaded=" << nelements_preloaded
              << ", nelements_to_read=" << nelements_to_read << ", last_block_nelements=" << last_block_nelements
              << ", num_blocks_to_read=" << num_blocks_to_read << ", nbytes_to_read=" << nbytes_to_read
              << ", num_bytes_read=" << num_bytes_read << ", ntotal_bytes=" << ntotal_bytes
              << ", count=" << count
              <<  ")." << std::endl;
#endif
    // Deal with offset for a next batch only if we have read entire batch
    if (num_bytes_read == nbytes_to_read) {
        if (last_block_nelements == 0) {
            buffer_offset_ = 0;
        } else {
            // We have read a whole number of blocks and last 'nelements_next_batch' elements
            // are from next batch.
            const size_t nelements_next_batch = block_sz_ - last_block_nelements;
            for (size_t i=0; i<nelements_next_batch; ++i) {
                buffer_[last_block_nelements+i] = buffer_[ntotal_bytes-nelements_next_batch+i];
            }
            buffer_offset_ = last_block_nelements;
        }
    } else {
        // Not all data have been read. It's either end of file or something else. This something else
        // can be (I think) some iterrupt signal. I should probably wrap the above call to read in a 
        // loop.
        // If it's end of file, next call will return 0.
        buffer_offset_ = 0;  // Better solution?
    }

    return ntotal_bytes;
}

void direct_reader::allocate(const size_t new_sz) {
    if (new_sz % block_sz_ != 0) {
        throw fmt("Invalid buffer size (%u). Must be a multiple of block size (%u).", new_sz, block_sz_);
    }
    if (buffer_size_ < new_sz) {
        DLOG(fmt("[direct reader] allocating memory (buffer size=%u, new size=%u).", buffer_size_, new_sz));
        deallocate();
        const size_t alignment = block_sz_;
        buffer_ = static_cast<unsigned char*>(aligned_alloc(alignment, new_sz));
        if (buffer_ == nullptr) {
            throw std::bad_alloc();
        }
        buffer_size_ = new_sz;
    } else {
        DLOG(fmt("[direct reader] skipping memory allocation (buffer size=%u, new size=%u).", buffer_size_, new_sz));
    }
}

void direct_reader::deallocate() {
    DLOG(fmt("[direct reader] deallocating memory (buffer size=%u).", buffer_size_));
    if (buffer_ != nullptr) {
        free(buffer_);
        buffer_ = nullptr;
        buffer_size_ = 0;
    }
}

// ---------------------------------------------------------------------------

process_barrier::process_barrier(std::string specs) : post_mode_(true) {
    std::replace(specs.begin(), specs.end(), ',', ' ');
    std::istringstream is(specs);
    is >> rank_ >> count_ >> name_;
    name_ = "/" + name_;
    init();
}

process_barrier::process_barrier(const std::string& name, const int rank, const int count) 
: name_(name), rank_(rank), count_(count), post_mode_(true) {
    init();
}

void process_barrier::init() {
    if (rank_ == 0) {
        handle_ = sem_open(name_.c_str(), O_CREAT|O_EXCL, S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH, 0);
    } else {
        while (true) {
            handle_ = sem_open(name_.c_str(), 0);
            if (handle_ != SEM_FAILED || errno != ENOENT)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

int process_barrier::wait() {
    if (handle_ == SEM_FAILED) {
        return -1;
    }
    int target_value(0);
    if (post_mode_) {
        if (sem_post(handle_) < 0)
            return -1;
        target_value = count_;
    } else {
        if (sem_wait(handle_) < 0)
            return -1;
    }
    int value(-1);
    while (value != target_value) {
        if (sem_getvalue(handle_, &value) < 0)
            return -1;
    }
    post_mode_ = !post_mode_;
    return 0;
}

void process_barrier::close() {
    if (handle_ != SEM_FAILED) {
        if (rank_ == 0) {
            sem_unlink(name_.c_str());
        }
        sem_close(handle_);
        handle_ = SEM_FAILED;
    }
}
    
    
