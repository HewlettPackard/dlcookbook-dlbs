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

#ifndef DLBS_TENSORRT_BACKEND_CORE_FILESYSEM_HADOOPFILESYSTEM
#define DLBS_TENSORRT_BACKEND_CORE_FILESYSEM_HADOOPFILESYSTEM

#include "core/filesystem/file_system.hpp"

#if defined HAVE_HDFS
// Forward declarations to not include hsdf.h header here.
struct HdfsFileSystemInternalWrapper;
//
/**
 * It should have a prefix, like:
 *    hdfs://namenode  OR  hdfs://namenode:port
 * All external URLs need to be checked against this prefix
 */
class hadoop_file_system : public file_system {
private:
    HdfsFileSystemInternalWrapper *hdfs_;
public:
    hadoop_file_system(const url &the_url);
    virtual ~hadoop_file_system();

    file_status status(const std::string &path) override;
    bool make_dir(std::string dir) override;
    
    writable_file* new_writable_file(parameters params = {}) override;
    readable_file* new_readable_file(parameters params = {}) override;
    
    void read_lines(const std::string &path, std::vector<std::string>& content) override;
    void get_children(const std::string &dir, std::vector<std::string> *files, std::vector<std::string> *dirs) override;
};
#endif

#endif
