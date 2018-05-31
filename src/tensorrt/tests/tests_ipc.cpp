#include "core/logger.hpp"
#include "core/utils.hpp"

#include <chrono>
#include <thread>

/**
 * If running in multiple docker containers, run containers with --ipc=host.
 */
int main(int argc, char **argv) {
    logger_impl logger;
    if (argc != 3) {
        logger.log_error("Usage: tests_ipc my_rank comm_world");
    }
    const int my_rank = std::atoi(argv[1]);
    const int world_size = std::atoi(argv[2]);
    process_barrier barrier("/dlbs_ipc", my_rank, world_size);
    if (!barrier.good()) {
        fprintf(stderr, "sem_open() failed.  errno:%d, description=%s\n", errno, strerror(errno));
        logger.log_error("Failed opening semaphore.");
    }
    //
    logger.log_info(fmt("Rank %d: simualting initialization and warmup procedure", my_rank));
    std::this_thread::sleep_for(std::chrono::milliseconds(1000*my_rank));
    
    // Barrier one
    logger.log_info(fmt("Rank %d: Waiting fo barrier #1", my_rank));
    barrier.wait();
    //
    logger.log_info(fmt("Rank %d: simualting benchmark procedure", my_rank));
    std::this_thread::sleep_for(std::chrono::milliseconds(1000*my_rank));
    // Barrier two
    logger.log_info(fmt("Rank %d: Waiting fo barrier #2", my_rank));
    barrier.wait();
    //
    logger.log_info(fmt("Rank %d: all done", my_rank));
    barrier.close();
    return 0;
}