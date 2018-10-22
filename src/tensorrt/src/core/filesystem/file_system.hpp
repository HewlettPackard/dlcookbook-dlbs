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

#ifndef DLBS_TENSORRT_BACKEND_CORE_FILESYSEM_FILESYSTEM
#define DLBS_TENSORRT_BACKEND_CORE_FILESYSEM_FILESYSTEM

#include "core/utils.hpp"
#include "core/logger.hpp"
#include <string>
#include <boost/any.hpp>
#include <unordered_map>

using parameters = std::unordered_map<std::string, boost::any>;

/**
 * @brief A base interface for all file objects (files).
 * 
 * Paths that are used to open files must be local with respect to a particular
 * file system layer i.e., for instance, for hdfs file systems, to open a file
 * do not use hdfs:// or hdfs://namenode:port prefix.
 */
class base_file {
protected:
    std::string path_;
public:
    virtual ~base_file() {}
    virtual bool open(const std::string &path) = 0;
    virtual bool is_open() = 0;
    virtual void close() = 0;
    virtual std::string description() = 0;
    const std::string path() const { return path_; }
};

/**
 * @brief An interface for writable files.
 */
class writable_file : public base_file {
public:
    virtual ssize_t write(const uint8_t *ptr, const size_t length) = 0;
};

/**
 * @brief An interface for readable files.
 */
class readable_file : public base_file {
public:
    virtual ssize_t read(uint8_t *ptr, const size_t length) = 0;
};

/**
 * @brief An interface for file systems.
 * 
 * The following methods are pure virtual and must be impemented by a
 * particular implementation:
 *    status, make_dir, get_children, new_writable_file, new_readable_file
 */
class file_system {
public:
    /**
     * @brief Type of IO operation - read or write, used for checking results.
     */
    enum class io_op {
        read,
        write
    };
    /**
     * @brief Status of a file in a file system.
     */
    enum class file_status {
        na,          /**< Error or status not available (does not exist) */
        file,        /**< It's a file */
        directory    /**< It's a directory */
    };
public:
    virtual ~file_system() {}
    /**
     * @brief Return status of the `path`.
     * 
     * @param path A path within this file system without schema prefix.
     * @return File status - file, directory or other (na).
     */
    virtual file_status status(const std::string &path) = 0;
    
    /**
     * @brief Make a directory. This is not a recursive call.
     * 
     * @param dir Directory to create. This method must only create
     *            this directory if possible. If, for instance, parent of
     *            thid directory does not exist, this method returns false
     * @return true if directory exists or has been created.
     */
    virtual bool make_dir(std::string dir) = 0;
    
     /**
     * @brief Find children files and directories.
     * 
     * Returned paths are relative to `dir`. Special directories like `.` and `..`
     * must not be returned. This is not a recursive call. This method returns
     * direct children of the provided directory `dir`
     * 
     * @param dir A directory path
     * @param files On exit, must contain files. Can be null that indicates user
     *              is not interested in files.
     * @param dirs On exit, must contain directories excluding `.` and `..`. Can
     *             be null what indicates user is not interested in directories.
     */
    virtual void get_children(const std::string &dir, std::vector<std::string> *files, std::vector<std::string> *dirs) = 0;

    /**
     * @brief Return implementation that supports writing files in this file system.
     * 
     * A file implementation must know what file system it writes to. It must work
     * with paths without prfixes like schema identifier or access credentials.
     * 
     * @return An implementation that can write files. Users are responsible for
     *         destroying a memory allocated for the file object.
     */
    virtual writable_file* new_writable_file(parameters params = {}) = 0;
    
    /**
     * @brief Return imlementation that can read files.
     * 
     * @param params Additional parameters. This methid is free to ignore parameters
     *        that are not relevant/unknown for this file system implementation.
     */
    virtual readable_file* new_readable_file(parameters params = {}) = 0;
    
    
    
    /**
     * @brief Read textual file `path` line by line.
     */
    virtual void read_lines(const std::string &path, std::vector<std::string>& content) = 0;
    
    


    /**
     * @brief A wrapper that writes/overwrites file with provided data.
     * 
     * @param fname A file name. If exists, will be overwritten.
     * @param ptr A pointer to a data.
     * @param length Lenght of data in bytes.
     * @param header If true, write the length of the data (in bytes)
     * @return Number of byets written excluding possible header. On success,
     *         it should equal to `length`.
     */
    ssize_t write_bytes(const std::string &fname, const uint8_t *ptr, const size_t length, const bool header=true);
    /**
     * @brief Read some bytes from a file.
     * 
     * The method reads some bytes from the `fname` file. If `header` is false, `length`
     * bytes are read. Else, number of bytes to read is read from the file. In this case,
     * on exit `length` will be this value read from file.
     * The method will allocate `length` bytes for `ptr` memory.
     * 
     * @param fname A file name.
     * @param ptr A pointer to use. Must be nullptr. The method will allocate memory internally and
     *        it is user responsibility to deallocate memory with delete [] ptr.
     * @param length Number of bytes to read if `header` is false. If `header` is true, number of bytes
     *        to read is determined from the file. In this case, on exit the `length` will be assigned to
     *        that value. If length > 0, length of `ptr` is `length`.
     * @param header If true, read length of array to be read from the file.
     * @return Number of bytes read excluding header. On success, it should equal to `length`.
     */
    ssize_t read_bytes(const std::string &fname, uint8_t *&ptr, size_t &length, const bool header=true);
    
    /**
     * @brief Find all files recursively and return relative paths.
     * 
     * @param base_dir An absolute directory to search files in recursively.
     * @param files On output, will contain found files. Paths will be relative
     *        to `base_dir`.
     */
    void find_files(std::string base_dir, std::vector<std::string>& files);
    
    /**
     * @brief Create directory `dir`. The call is recursive.
     * 
     * @param dir Directory to create. It is assumed this path does not contain
     *            `.`, `..` or symbolic links.
     */
    bool make_dirs(std::string dir);

    /**
     * @brief Returns parent directory.
     * 
     * This function assumes that the `dir` parameter does not contain `.` or `..`
     * entries. Also, no symbolic links. Works simply by splitting the string using
     * `/` delimeter. This function must work for directories that does not exist.
     * This function is used to create directories when generating dataset. So, it is
     * assumed that at some point in time there will be a folder that exists - at least
     * the one that is a mount mount inside a docker container.
     * 
     * @param dir A directory path that may not exist. See exampes above.
     * @returns Parent of this directory or empty if parent cannot be computed. Empty string is
     *          only returned for relative input directories. For absolute paths starting
     *          with `/`, parent of `/` is a root folder itself '/'.
     */
    static std::string parent_dir(std::string dir);
    
    /**
     * @brief Prepend `base` directory to all items in `rpaths`.
     * 
     * It is assumed that base is a valid absolute directory. The paths in rpaths are
     * considered to be relative to base. The method just concatenates together `base`
     * and `rel_paths`.
     * 
     * @param base An absolute directory.
     * @param rpaths A list of paths relative to `base`.
     */
    static void make_absolute(std::string base, std::vector<std::string> &rpaths);

    static void check_io(io_op op, const std::string &fname, const size_t nrequested, const ssize_t ngotten,
                         logger_impl &logger, logger_impl::severity severity);
};


class file_system_registry {
public:
    /**
     * @brief Return file system object associated with the given url.
     * 
     */
    static file_system* get(const url &fs_url);
};

#endif
