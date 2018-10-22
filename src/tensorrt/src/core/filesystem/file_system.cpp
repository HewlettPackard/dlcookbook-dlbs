
#include "core/utils.hpp"
#include "core/filesystem/file_system.hpp"
#include "core/filesystem/posix_file_system.hpp"
#include "core/filesystem/hadoop_file_system.hpp"
#include <iostream>
#include <queue>

ssize_t file_system::write_bytes(const std::string &fname, const uint8_t *ptr, const size_t length, const bool header) {
    std::unique_ptr<writable_file> file(new_writable_file());
    ssize_t num_bytes_written(0);
    if (file->open(fname)) {
        if (header) {
            const long unsigned int len = static_cast<long unsigned int>(length);
            file->write((const uint8_t*)(&len), sizeof(long unsigned int));
        }
        num_bytes_written = file->write(ptr, length);
        file->close();
    }
    return num_bytes_written;
}
ssize_t file_system::read_bytes(const std::string &fname, uint8_t *&ptr, size_t &length, const bool header) {
    std::unique_ptr<readable_file> file(new_readable_file());
    ssize_t num_bytes_read(0);
    if (file->open(fname)) {
        if (header) {
            long unsigned int nbytes_to_read(0);
            file->read((uint8_t*)&nbytes_to_read, sizeof(long unsigned int));
            length = nbytes_to_read;
        }
        if (length> 0) {
            ptr = new uint8_t[length];
            num_bytes_read = file->read(ptr, length);
        }
    } else {
        length = 0;
    }
    return num_bytes_read;
}

std::string file_system::parent_dir(std::string dir) {
    if (dir == "/") { return dir; }
    const auto pos = rtrim(dir, "/").find_last_of("/");
    if (pos == 0) return "/";
    if (pos == std::string::npos)
        return "";
    return dir.substr(0, pos);
}

void file_system::make_absolute(std::string base, std::vector<std::string> &rpaths) {
    if (base != "/") {
        rtrim_inplace(base, "/");
        if (base.empty())
            return;
        base += "/";
    }
    for (size_t i=0; i<rpaths.size(); ++i) { rpaths[i] = base + rpaths[i]; }
}

void file_system::find_files(std::string base_dir, std::vector<std::string>& files) {
    rtrim_inplace(base_dir, "/");
    files.clear();
    std::queue<std::string> rel_dirs;
    rel_dirs.push("");
    std::vector<std::string> children_files, children_dirs;
    while (rel_dirs.empty() == false) {
        const std::string rel_dir = rel_dirs.front();
        const std::string sep = (rel_dir.empty() ? "" : "/");
        rel_dirs.pop();

        get_children(base_dir + sep + rel_dir, &children_files, &children_dirs);

        for (size_t j=0; j<children_files.size(); ++j) { 
            files.push_back(rel_dir + sep + children_files[j]);
        }
        for (size_t j=0; j<children_dirs.size(); ++j)  {
            rel_dirs.push(rel_dir+ sep + children_dirs[j]);
        }
    }
}

bool file_system::make_dirs(std::string dir) {
    file_status fstatus = status(dir);
    if (fstatus == file_status::na) {
        const auto parent = file_system::parent_dir(dir);
        if (parent.empty() || parent == dir) return false;
        if(!make_dirs(parent)) return false;
        return make_dir(dir.c_str());
    }
    return (fstatus == file_status::directory);
}


void file_system::check_io(io_op op, const std::string &fname, const size_t nrequested, const ssize_t ngotten,
                           logger_impl &logger, logger_impl::severity severity) {
    if (ngotten != nrequested) {
        std::string message = "";
        if (op == io_op::write) { message = "File write error (%s). Num bytes to write (%ld), num bytes written (%ld)"; }
        else { message = "File read error (%s). Num bytes to read (%ld), num bytes read (%ld)"; }
        logger.log(severity, fmt(message, fname.c_str(), long(nrequested), long(ngotten)));
    }
}


file_system* file_system_registry::get(const url &fs_url) {
    const std::string& scheme = fs_url.scheme();
    if (scheme.empty() || scheme == "file") {
        return new posix_file_system(fs_url);
    }
#if defined HAVE_HDFS
    if (scheme == "hdfs") {
        return new hadoop_file_system(fs_url);
    }
#endif
    return nullptr;
}
