#include "core/utils.hpp"


int main(int argc, char **argv) {
    // Test Split Vector Algorithm
    {
        const size_t len = 13;
        std::vector<int> vec(len, 0);
        for (int num_shards=1; num_shards<=len+2; ++num_shards) {
            std::cout << len << " into " << num_shards << " workers:";
            for (int my_shard=0; my_shard<num_shards; ++my_shard) {
                sharded_vector<int> svec(vec, num_shards, my_shard);
                std::cout << " " << svec.shard_length();
            }
            std::cout << std::endl;
        }
    }
    // Test Type cast
    /*
    {
        const size_t num_batch_elements = 512 * 3 * 227 * 227;
        std::vector<unsigned char> char_data(num_batch_elements);
        std::vector<float> float_data(num_batch_elements);
        for (int i=0; i<num_batch_elements; ++i) {
            char_data[i] = 2;
            float_data[i] = 2;
        }
        timer tm;
        const int num_iters = 100;
        for (int i=0; i<num_iters; ++i) {
            for (int j=0; j<num_batch_elements; ++j) {
                float_data[j] = static_cast<float>(char_data[j]);
            }
        }
        const float elapsed = tm.ms_elapsed();
        const float elements_sec = (1000.0 * (num_iters * num_batch_elements) / elapsed);
        std::cout << "elements / sec = " << elements_sec << ", batches / sec = " << (elements_sec / num_batch_elements) << std::endl;
    }
    return 0;
    */
    // Test Random
    {
        std::vector<float> vec (100, 0);
        fill_random(vec.data(), vec.size());
        std::sort(vec.begin(), vec.end());
        const auto mean = std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
        std::cout << "Test Random vector mean is: " << mean << ", min=" << vec.front() << ", max=" << vec.back() << std::endl;
    }
    // Test To String
    {
        std::cout << "Test to String: int=" << S((int)1) << ", float=" << S((float)2.343) << 
                                      ", bool=" << S(true) << ", bool=" << S(false) << std::endl;
    }
    // Test Format
    {
        std::cout << fmt("TestFormat: Hello %s", "world!") << std::endl;
        std::cout << fmt("TestFormat: Hello robot number %d", 1) << std::endl;
        std::cout << fmt("TestFormat: Floating point number is %.3f", 3.462464) << std::endl;
        std::cout << fmt("TestFormat: boolean value is %s", "false") << std::endl;
        const std::string s = "str instance";
        std::cout << fmt("TestFormat: Hello %s", s.c_str()) << std::endl;
    }
    // Test Sharded Vector
    {
        std::vector<int> vec(1000, 0);
        std::cout << "Test Sharded Vector: " << sharded_vector<int>(vec,  1,  0) << std::endl;
        std::cout << "Test Sharded Vector: " << sharded_vector<int>(vec,  2,  0) << std::endl;
        std::cout << "Test Sharded Vector: " << sharded_vector<int>(vec,  2,  1) << std::endl;
        std::cout << "Test Sharded Vector: " << sharded_vector<int>(vec, 33, 32) << std::endl;
    }
    // Test Running Average
    {
        running_average ra;
        for (int i=1; i<=10; ++i)
            ra.update(i);
        std::cout << "Test Running Average: " << ra << std::endl;
    }
}