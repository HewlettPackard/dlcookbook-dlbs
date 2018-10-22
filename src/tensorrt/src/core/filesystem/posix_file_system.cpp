
#include "core/filesystem/posix_file_system.hpp"
#include <fstream>
#include <iostream>
#define _XOPEN_SOURCE 700
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <dirent.h>
#include <ios>
#include <boost/any.hpp>
#include <unordered_map>

template<typename stream_type>
class posix_file {
protected:
    stream_type stream_;
protected:
    bool impl_open(const std::string &path) {
        stream_.open(path.c_str(), std::ios_base::binary);
        return stream_.is_open();
    }
    bool impl_is_open() { return stream_.is_open(); }
    void impl_close() { stream_.close(); }
};

class posix_writable_file : public writable_file, public posix_file<std::ofstream> {
public:
    bool open(const std::string &path) override { 
        path_ = path;
        return impl_open(path); 
    }
    bool is_open() override { return impl_is_open(); }
    void close() override { path_ = ""; impl_close(); }

    ssize_t write(const uint8_t *ptr, const size_t length) override {
        const auto pos = stream_.tellp();
        stream_.write((const char*)(ptr), static_cast<std::streamsize>(length));
        return static_cast<ssize_t>(stream_.tellp() - pos);
    }
    
    std::string description() { return "posix_writable_file"; }
};
    
class posix_readable_file : public readable_file, public posix_file<std::ifstream> {
public:
    bool open(const std::string &path) override { 
        path_ = path;
        return impl_open(path); 
    }
    bool is_open() override { return impl_is_open(); }
    void close() override { path_ = ""; impl_close(); }

    ssize_t read(uint8_t *ptr, const size_t length) override {
        stream_.read((char*)(ptr), static_cast<std::streamsize>(length));
        return static_cast<ssize_t>(stream_.gcount());
    }
    std::string description() { return "posix_readable_buffered_file"; }
};


class posix_readable_buffered_file : public readable_file {
private:
    int fd_ = -1;                          //!< File descriptor.
    bool advise_no_cache_ = false;         //!< If true, advise OS not to cache file.
    const std::string dtype_;              //!< Matrix data type in a binary file ('float', 'uchar').
    std::vector<unsigned char> buffer_;    //!< If images are stored as unsigned chars, use this buffer.
public:
    posix_readable_buffered_file(const std::string& dtype="float", const bool advise_no_cache=false);
    bool open(const std::string &path) override;
    bool is_open() override { return (fd_ > 0); }
    ssize_t read(host_dtype* dest, const size_t count) override;
    void close() override;
    std::string description() { return "posix_readable_buffered_file"; }
};

class posix_readable_unbuffered_file : public readable_file {
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
    bool eof_reached_ = false;             // !< We have reached EOF with previous call. 
public:
    posix_readable_unbuffered_file(const std::string& dtype="float", const int block_sz=512);
    bool open(const std::string &path) override;
    bool is_open() override { return (fd_ > 0); }
    ssize_t read(host_dtype* dest, const size_t count) override;
    void close() override;
    std::string description() { return "posix_readable_unbuffered_file"; }
private:
    void allocate(const size_t new_sz);
    void deallocate();
};

file_system::file_status posix_file_system::status(const std::string &path) {
    // http://man7.org/linux/man-pages/man2/stat.2.html
    // Add in the future various checks for ENOENT, ENOTDIR etc.
    struct stat sb;
    if (stat(path.c_str(), &sb) != 0) { return file_system::file_status::na; }
    if (S_ISREG(sb.st_mode)) { return file_system::file_status::file; }
    if (S_ISDIR(sb.st_mode)) { return file_system::file_status::directory; }
    return file_system::file_status::na;
}

bool posix_file_system::make_dir(std::string dir) {
    file_status fstatus = status(dir);
    if (fstatus == file_status::na) {
        return (mkdir(dir.c_str(), 0700) == 0);
    }
    return (fstatus == file_status::directory);
}

writable_file* posix_file_system::new_writable_file(parameters params) {
    return new posix_writable_file();
}

readable_file* posix_file_system::new_readable_file(parameters params) {
    std::string reader_type = boost::any_cast<std::string>(params["reader_type"]);
    std::string data_type = boost::any_cast<std::string>(params["data_type"]);
    readable_file *file(nullptr);
    if (reader_type == "default" || reader_type == "") {
        file = new posix_readable_buffered_file(
            data_type,
            environment::remove_files_from_os_cache()
        );
    } else if (reader_type == "directio") {
        file = new posix_readable_unbuffered_file(
            data_type,
            environment::storage_block_size()
        );
    }
    return file;
}

void posix_file_system::read_lines(const std::string &path, std::vector<std::string>& content) {
    content.clear();
    std::ifstream fstream(path.c_str());
    if (fstream) {
        std::string line;
        while (std::getline(fstream, line))
            content.push_back(line);
    }
}

void posix_file_system::get_children(const std::string &dir, std::vector<std::string> *files, std::vector<std::string> *dirs) {
    if (files) files->clear();
    if (dirs) dirs->clear();
    
    DIR *dir_handle = opendir(dir.c_str());
    if (dir_handle == nullptr)
        return;
    struct dirent *de(nullptr);
    while ((de = readdir(dir_handle)) != nullptr) {
        const std::string dir_item(de->d_name);
        if (dir_item == "." || dir_item == "..") {
            continue;
        }
        file_status fstatus = file_status::na;
        if (de->d_type == DT_UNKNOWN)    fstatus = status(dir + dir_item);
        else if (de->d_type == DT_REG)   fstatus = file_status::file;
        else if (de->d_type == DT_DIR)   fstatus = file_status::directory;
        if (fstatus == file_status::directory) {
            if (dirs) dirs->push_back(dir_item);
        } else if (fstatus == file_status::file) {
            if (files) files->push_back(dir_item);
        }
    }
    closedir(dir_handle);
}


posix_readable_buffered_file::posix_readable_buffered_file(const std::string& dtype,
                                                           const bool advise_no_cache) : advise_no_cache_(advise_no_cache), dtype_(dtype) {
}

bool posix_readable_buffered_file::open(const std::string &path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (advise_no_cache_) {
        fdatasync(fd_);
    }
    return is_open();
}

ssize_t posix_readable_buffered_file::read(host_dtype* dest, const size_t count)  {
#if defined HOST_DTYPE_FP32
    // To convert from unsigned char in files to SP32 in host memory
    if (dtype_ == "uchar" && buffer_.size() != count) {
        buffer_.resize(count);
    }
#else
    if (dtype_ == "float") {
        throw "With unsigned char host data type files with SP32 elements are not supported.";
    }
#endif
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

void posix_readable_buffered_file::close()  {
    if (fd_ > 0) {
        if (advise_no_cache_) {
            posix_fadvise(fd_, 0, 0, POSIX_FADV_DONTNEED);
        }
        ::close(fd_);
        fd_ = -1;
    }
}


posix_readable_unbuffered_file::posix_readable_unbuffered_file(const std::string& dtype, const int block_sz) : 
dtype_(dtype == "float" ? data_type::dt_float : data_type::dt_unsigned_char), block_sz_(block_sz) {

    if (dtype != "float" && dtype != "uchar") {
        throw std::invalid_argument(
            "Invalid image data type (expecting 'float' or 'uchar') but given '" + dtype + "'."
        );
    }
    if (dtype_ == data_type::dt_float) {
        throw std::invalid_argument(
            "DirectIO file reader does not support input files storing images with 4 bytes per element (float data type). "\
            "Probably, you forgort to provide data_name parameter: --data_name=tensors1. If running with DLBS, this will "\
            "be tensorrt.data_name i.e. -Ptensorrt.data_name='\"tensors1\"'."
        );
    }
    
    block_sz_ = environment::storage_block_size();
    DLOG(fmt("[direct reader] direct_reader::direct_reader(dtype=%s, block size=%u).", dtype.c_str(), block_sz_));
}

bool posix_readable_unbuffered_file::open(const std::string &path) {
    DLOG(fmt("[direct reader] opening file (fname=%s).", fname.c_str()));
    // Reser buffer offset each time new file is opened.
    buffer_offset_ = 0;
    eof_reached_ = false;
    // http://man7.org/linux/man-pages/man2/open.2.html
    fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
    if (fd_ < 0) {
        std::cerr << "Input file (" << path << ") has not been opened. Errno is " << errno << "." << std::endl;
        if (errno == EINVAL) {
            std::cerr << "This is the EINVAL error (the filesystem does not support the O_DIRECT flag)." << std::endl;
        }
    }
    return is_open();
}

/** Implementation comments.
 *    buffer_offset_: Number of bytes associated with previous read. If it's not 0,
 *                    the `block_sz_ - buffer_offset_` value defines how many bytes we
 *                    have for current batch.
 *    In the comments below I use number of bytes/number of elements interchangeably 
 *    because we always read elements of size 1 byte.
 *    This method will fail if batch is less than 1 block.
 */
ssize_t posix_readable_unbuffered_file::read(host_dtype* dest, const size_t count)  {
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
        // This can happen in several cases:
        //   - Invalid block size which is by default equals to 512. For instance, with some Lustre file systems,
        //     the block size I use is 4096.
        //   - We have reached EOF with previous call. Most likely, file offset now is not properly aligned, so we
        //     have gotten EINVAL. For more details, see these URLs:
        //         https://github.com/coreutils/coreutils/blob/master/src/dd.c#L1123
        //         https://linux.die.net/man/2/read
        if (num_bytes_read == -1 && errno == EINVAL && eof_reached_) {
            // End of file reached at previous call.
            return 0;
        }
        auto msg = fmt("Error reading file using Direct IO (errno=%d). Is the block size correct (block "\
                       "size=%d)? You can change this value with DLBS_TENSORRT_STORAGE_BLOCK_SIZE "\
                       "environment variable. If you use Lustre, try 4096.", int(errno), block_sz_);
        throw std::runtime_error(msg);
    }
    if (num_bytes_read == 0) {
        // This is fine. The higher level code will close this file and will open another one.
        // Can we get this error at all with O_DIRECT - if lenght of a file is a perfect multiple
        // of a block size?
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
        // If it's end of file, next call will return 0 or, most likely, EINVAL.
        buffer_offset_ = 0;  // Better solution?
        eof_reached_ = true;
    }

    return ntotal_bytes;
}

void posix_readable_unbuffered_file::close()  {
    DLOG("[direct reader] closing file.");
    if (is_open()) {
        ::close(fd_);
        fd_ = -1;
    }
}

void posix_readable_unbuffered_file::allocate(const size_t new_sz) {
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

void posix_readable_unbuffered_file::deallocate() {
    DLOG(fmt("[direct reader] deallocating memory (buffer size=%u).", buffer_size_));
    if (buffer_ != nullptr) {
        free(buffer_);
        buffer_ = nullptr;
        buffer_size_ = 0;
    }
}