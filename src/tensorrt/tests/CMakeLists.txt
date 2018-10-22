# Install Catch2 into /usr/local
#     git clone https://github.com/catchorg/Catch2.git ./catch2
#     cd ./catch2 && mkdir ./build && cd ./build
#     cmake .. && make -j$(nproc) && make install

# How to run tests:
#     1. Go to `build/tests` and run `ctest`.

option(TESTS "Build tests. Catch2 is required." OFF)
if(NOT TESTS)
    return()
endif()

find_package(Catch2 REQUIRED)
enable_testing()


include(CTest)
include(Catch)

add_executable(test_utils tests_main.cpp test_utils.cpp test_fs.cpp test_posix_fs.cpp)
    target_link_libraries(test_utils Catch2::Catch2 ${base_libs})
    catch_discover_tests(test_utils)