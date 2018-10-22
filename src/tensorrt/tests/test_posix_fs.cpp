#include <catch2/catch.hpp>
#include "core/filesystem/file_system.hpp"

TEST_CASE("Test Posix Filesystem", "[fs][filesystem][posixfilesystem]") {
    url base_url("/");
    std::unique_ptr<file_system> fs(file_system_registry::get(base_url));
    REQUIRE(fs != nullptr);

    SECTION("status (standard directories)") {
        REQUIRE(fs->status("/etc") == file_system::file_status::directory);
        REQUIRE(fs->status("/bin") == file_system::file_status::directory);
        REQUIRE(fs->status("/proc") == file_system::file_status::directory);
    }
    
    SECTION("posix_file_system tests") {
        //
        //
        REQUIRE(fs->status("/tmp/bcac407") == file_system::file_status::na);
        //
        //
        REQUIRE(fs->make_dirs("/tmp/bcac407") == true);
        REQUIRE(fs->make_dirs("/tmp/bcac407/etc/s1/s2") == true);
        REQUIRE(fs->make_dirs("/tmp/bcac407/etc") == true);
        REQUIRE(fs->make_dirs("/tmp/bcac407/bin") == true);
        REQUIRE(fs->make_dirs("/tmp/bcac407/proc") == true);
        
        REQUIRE(fs->status("/tmp/bcac407") == file_system::file_status::directory);
        REQUIRE(fs->status("/tmp/bcac407/etc/s1/s2") == file_system::file_status::directory);
        REQUIRE(fs->status("/tmp/bcac407/etc") == file_system::file_status::directory);
        REQUIRE(fs->status("/tmp/bcac407/bin") == file_system::file_status::directory);
        REQUIRE(fs->status("/tmp/bcac407/proc") == file_system::file_status::directory);
        //
        //
        std::string message = "Hello world !!!";
        std::unique_ptr<writable_file> wfile(fs->new_writable_file());
        REQUIRE(wfile->open("/tmp/bcac407/etc/readme.txt") == true);
        wfile->write((uint8_t*)message.data(), message.size());
        wfile->close();
        //
        //
        REQUIRE(fs->status("/tmp/bcac407/etc/readme.txt") == file_system::file_status::file);
        //
        //
        std::unique_ptr<readable_file> rfile(fs->new_readable_file({{"reader_type", _S("default")}, {"data_type", _S("char")}}));
        unsigned char* buffer = new unsigned char[message.size()];
        REQUIRE(rfile->open("/tmp/bcac407/etc/readme.txt") == true);
        ssize_t nread = rfile->read(buffer, message.size());
        rfile->close();
        REQUIRE(nread == message.size());
        REQUIRE(std::string((char*)buffer) == message);
        //
        // The file system layer does not really support removing files because we do not need this
        system("/bin/bash -c \"rm -rf /tmp/bcac407 \"");
    }
}
