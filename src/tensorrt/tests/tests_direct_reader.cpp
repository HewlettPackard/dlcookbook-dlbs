#include "core/utils.hpp"

int main(int argc, char **argv) {
#if not defined TRACE_ALL
    std::cout << "This test is supposed to be ran with detailed tracing enabled. Uncomment "
              << "'add_definitions(-DTRACE_ALL)' in CMakeLists.txt and rebuild the project."
              << std::endl;
#endif
    
    std::string fpath = "/home/serebrya/data/tensors1/images-0-0.tensors";
    int batch_size = 64;
    int image_size = 224;
    
    const size_t num_batch_elements = batch_size * ( 3 * image_size * image_size );
    std::vector<host_dtype> batch(num_batch_elements);
    
    direct_reader reader("uchar");
    if (!reader.open(fpath)) {
        std::cerr << "Cannot open file (" << fpath << ")." << std::endl;
    }
    int num_batches_read(0);
    size_t num_elements_read = reader.read(batch.data(), num_batch_elements);
    while ( num_elements_read != 0) {
        num_batches_read ++;
        std::cout << "Read batch (" << num_batches_read << "), num_batch_elements=" << num_batch_elements
                  << ", num_elements_read=" << num_elements_read << std::endl;
        num_elements_read = reader.read(batch.data(), num_batch_elements);
    }
    
    return 0;
}
