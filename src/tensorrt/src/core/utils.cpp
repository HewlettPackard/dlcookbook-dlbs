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


template<>
std::string S<bool>(const bool &t) { return (t ? "true" : "false"); }

void fill_random(float *vec, const size_t sz) {
  std::random_device rnd_device;
  std::mt19937 mersenne_engine(rnd_device());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  auto gen = std::bind(dist, mersenne_engine);
  std::generate(vec, vec+sz, gen);
}

std::string get_env_var(std::string const &var, const std::string& default_val) {
    char *val = getenv( var.c_str() );
    return val == nullptr ? default_val : std::string(val);
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
        std::cout << "[image_provider        ]: found " + S(files.size()) +  " image files in file system." << std::endl;
        fs_utils::get_image_files(data_dir, files);
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


std::ostream& operator<<(std::ostream &out, const running_average &ra) {
    std::cout << "running_average{size=" << ra.num_steps() << ", value=" << ra.value() << "}";
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


binary_file::binary_file(const std::string& dtype,
                         const bool advise_no_cache) : advise_no_cache_(advise_no_cache), dtype_(dtype) {
}

bool binary_file::is_opened() {
    return (fd_ > 0);
}

void binary_file::open(const std::string& fname) {
    fd_ = ::open(fname.c_str(), O_RDONLY);
    if (advise_no_cache_) {
        fdatasync(fd_);
    }
}

void binary_file::close() {
    if (fd_ > 0) {
        if (advise_no_cache_) {
            posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
        }
        ::close(fd_);
        fd_ = -1;
    }
}

ssize_t binary_file::read(float* dest, const size_t count) {
    ssize_t read_count;
    if (buffer_.empty()) {
        const ssize_t num_bytes_read = ::read(fd_, (void*)dest,  sizeof(float)*count);
        read_count = num_bytes_read / sizeof(float);
    } else {
        const ssize_t num_bytes_read = ::read(fd_, (void*)buffer_.data(),  sizeof(unsigned char)*count);
        if (num_bytes_read > 0) {
            std::copy(buffer_.data(), buffer_.data() + num_bytes_read, dest);
        }
        read_count = num_bytes_read;
    }
    return read_count;
}

void binary_file::allocate_if_needed(const size_t count) {
    if (dtype_ == "uchar" && buffer_.size() != count) {
        buffer_.resize(count);
    }
}


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
    
    
