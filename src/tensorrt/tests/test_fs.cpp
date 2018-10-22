#include <catch2/catch.hpp>
#include "core/filesystem/file_system.hpp"
//
//
TEST_CASE("Test url", "[fs][filesystem][url]") {
    SECTION("posix") {
        std::unordered_map<std::string, std::string> posix_test_cases = {
            {"/my/some/path",         "/my/some/path"},
            {"file:///my/some/path",  "/my/some/path"}
        };
        for(const auto& test_case: posix_test_cases) {
            url posix_url(test_case.first);
            REQUIRE(posix_url.scheme() == "file");
            REQUIRE(posix_url.path() == test_case.second);
            REQUIRE(posix_url.good() == true);
        }
    }
    SECTION("hdfs") {
        std::vector<std::string> inputs = {
            "hdfs:///my/super/path1", "hdfs:///my/super/path2",
            "hdfs://mynamenode1/my/super/path3", "hdfs://mynamenode2/my/super/path4",
            "hdfs://mynamenode3:124/my/super/path5", "hdfs://mynamenode4:6666/my/super/path6"
        };
        std::vector<std::string> output_namenodes = {
            "localhost", "localhost", "mynamenode1", "mynamenode2", "mynamenode3", "mynamenode4"
            
        };
        std::vector<int> output_ports = { 9820, 9820, 9820, 9820, 124, 6666};
        std::vector<std::string> output_paths = {
            "/my/super/path1", "/my/super/path2",
            "/my/super/path3", "/my/super/path4",
            "/my/super/path5", "/my/super/path6"
            
        };
        for(size_t i=0; i<inputs.size(); ++i) {
            url hdfs_url(inputs[i]);
            REQUIRE(hdfs_url.scheme() == "hdfs");
            REQUIRE(hdfs_url.hdfs_namenode() == output_namenodes[i]);
            REQUIRE(hdfs_url.hdfs_port() == output_ports[i]);
            REQUIRE(hdfs_url.path() == output_paths[i]);
            REQUIRE(hdfs_url.good() == true);
        }
    }
}
//
//
TEST_CASE("Test parent_dir function", "[fs][filesystem][parentdir]") {
    std::unordered_map<std::string, std::string> test_cases = {
        {"/hello/world/",  "/hello"},
        {"/hello/world",   "/hello"},
        {"/hello",         "/"},
        {"hello/world",    "hello"},
        {"hello",          ""},
    };
    for(const auto& test_case: test_cases) {
        REQUIRE(file_system::parent_dir(test_case.first) == test_case.second);
    };
}
//
//
TEST_CASE("Test make_absolute function", "[fs][filesystem][make_absolute]") {
    std::string base = "/my/some/base/path/";
    std::vector<std::string> rpaths = {"images", "docs/work", "docs/work/list.txt"};
    file_system::make_absolute(base, rpaths);
    
    std::vector<std::string> outputs = {"/my/some/base/path/images", "/my/some/base/path/docs/work", "/my/some/base/path/docs/work/list.txt"};
    for (size_t i=0; i<rpaths.size(); ++i) {
        REQUIRE(rpaths[i] == outputs[i]);
    }
}
//
//
TEST_CASE("Test POSIX file_system_registry", "[fs][filesystem][filesystemregistry][posix]") {
    url posix_url("/my/path");
    std::unique_ptr<file_system> fs(file_system_registry::get(posix_url));
    REQUIRE(fs != nullptr);
    
    std::vector<std::unique_ptr<base_file>> files;
    files.emplace_back(fs->new_writable_file());
    files.emplace_back(fs->new_readable_file({{"reader_type",_S("default")},{"data_type",_S("uchar")}}));
    files.emplace_back(fs->new_readable_file({{"reader_type",_S("directio")},{"data_type",_S("uchar")}}));
    
    std::vector<std::string> names = {"posix_writable_file", "posix_readable_buffered_file", "posix_readable_unbuffered_file"};
    for (size_t i=0; i<files.size(); ++i) {
        auto &file = files[i];
        const auto &name = names[i];
        REQUIRE(file != nullptr);
        REQUIRE(file->description().size() >= name.size());
        REQUIRE(file->description().substr(0,name.size()+1) == name);
    }    
}
