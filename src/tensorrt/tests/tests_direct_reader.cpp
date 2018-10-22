#include "core/utils.hpp"
#include "core/filesystem/file_system.hpp"

int main(int argc, char **argv) {
#if not defined TRACE_ALL
    std::cout << "This test is supposed to be ran with detailed tracing enabled. Uncomment "
              << "'add_definitions(-DTRACE_ALL)' in CMakeLists.txt and rebuild the project."
              << std::endl;
#endif
    
    url fpath("/home/serebrya/data/tensors1/images-0-0.tensors");
    int batch_size = 64;
    int image_size = 224;
    
    const size_t num_batch_elements = batch_size * ( 3 * image_size * image_size );
    std::vector<host_dtype> batch(num_batch_elements);
    
    std::unique_ptr<file_system> fs(file_system_registry::get(fpath));
    std::unique_ptr<readable_file> reader(fs->new_readable_file(
        {{"reader_type", "directio"}, {"data_type", data_type::uint8()}}
    ));
    
    if (!reader->open(fpath.path())) {
        std::cerr << "Cannot open file (" << fpath.path() << ")." << std::endl;
    }
    int num_batches_read(0);
    size_t num_elements_read = reader->read(batch.data(), num_batch_elements);
    while ( num_elements_read != 0) {
        num_batches_read ++;
        std::cout << "Read batch (" << num_batches_read << "), num_batch_elements=" << num_batch_elements
                  << ", num_elements_read=" << num_elements_read << std::endl;
        num_elements_read = reader->read(batch.data(), num_batch_elements);
    }
    
    return 0;
}
