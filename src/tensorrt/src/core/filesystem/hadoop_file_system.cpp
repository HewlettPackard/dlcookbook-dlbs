#include "core/filesystem/hadoop_file_system.hpp"
#if defined HAVE_HDFS
#include <hdfs/hdfs.h>
//
// Everything here is completely untested. I only checked it compiles.
//
class hadoop_readable_file : public readable_file {
private:
    hdfsFS hdfs_ = nullptr;
    hdfsFile file_ = nullptr;
public:
    hadoop_readable_file(hdfsFS hdfs_handle) : hdfs_(hdfs_handle) {};
    bool open(const std::string &path) override;
    bool is_open() override { return (file_ != nullptr); };
    ssize_t read(host_dtype* dest, const size_t count) override;
    void close() override { 
        if (file_) { 
            hdfsCloseFile(hdfs_, file_);
            file_ = nullptr;
        }
    }
    std::string description() { return "hadoop_readable_file"; }
};
//
//
hadoop_file_system::hadoop_file_system(const url &the_url) : file_system() {
    const std::string &namenode = the_url.hdfs_namenode();
    const int port = the_url.hdfs_port();
    
    hdfsBuilder *builder = hdfsNewBuilder();
    if (builder == nullptr) { throw "Cannot create hdfsNewBuilder"; }
    
    hdfsBuilderSetNameNode(builder, namenode.c_str());
    hdfsBuilderSetNameNodePort(builder, port);
    
    hdfs_ = hdfsBuilderConnect(builder);
    if (hdfs_ == nullptr) { throw "Cannot connect to HDFS"; }
    
    hdfsFreeBuilder(builder);
}

hadoop_file_system::~hadoop_file_system() {
    if (hdfs_) {
        hdfsDisconnect(hdfs_);
        hdfs_ = nullptr;
    }
}

file_system::file_status hadoop_file_system::status(const std::string &path) {
    // Implement me! Return status of a `path` - is it file, path, does not exist
    // or something else
    if (hdfsExists(hdfs_, path.c_str()) < 0) {
        return file_system::file_status::na;
    }
    hdfsFileInfo *info = hdfsGetPathInfo(hdfs_, path.c_str());
    if (!info) { return file_system::file_status::na; }

    const file_system::file_status fstatus = (
        info->mKind == kObjectKindFile ? file_system::file_status::file : file_system::file_status::directory
    );

    hdfsFreeFileInfo(info, 1);
    return fstatus;
}

bool hadoop_file_system::make_dir(std::string dir) {
    // Implement me! Create directory `dir`. This is not recursive call. Assumption
    // is that parent dir exists.
    return (hdfsCreateDirectory(hdfs_, dir.c_str()) >= 0);
}

writable_file* hadoop_file_system::new_writable_file(parameters params) {
    // If I am not mistaken, this method is only required if DLBS is used to
    // generate dataset stored in hadoop file system. If this dataset is copied
    // there with hadoop command line tools, this method is not required.
    return nullptr;
}

readable_file* hadoop_file_system::new_readable_file(parameters params) {
    // Implement me! It should be similar to posix_readable_unbuffered_file or
    // posix_readable_buffered_file implementations. This readable file will be used
    // to read tensors files.
    return new hadoop_readable_file(hdfs_);
}

void hadoop_file_system::read_lines(const std::string &path, std::vector<std::string>& content) {
    // This method is not requried to support streaming data from HDFS.
    content.clear();
}

void hadoop_file_system::get_children(const std::string &dir, std::vector<std::string> *files, std::vector<std::string> *dirs) {
    // Implement me. Return (not recursively) files and folders in the 'dir'.
    if (files) files->empty();
    if (dirs) dirs->empty();
}
//
//
bool hadoop_readable_file::open(const std::string &path) {
    // Implement me! Open file pointed by a `path`.
    return false;
}

ssize_t hadoop_readable_file::read(host_dtype* dest, const size_t count) {
    // Implement me! Assumption is that `host_dtype` is unsigned char here.
    // Read count bytes from a file from current position. This function can read
    // less bytes if there are no more data left.
    // Return 0 to indicate we are at the end of a file.
    return 0;
}



#endif