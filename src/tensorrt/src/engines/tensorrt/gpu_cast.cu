#include "engines/tensorrt/gpu_cast.h"

/**
 * The 'input'/'output' parameters are 4D tensors of shape (Batch, Channels, Width, Height)
 * The input_size is Channels * Width * Height.
 * 
 *     blockIdx.x,y,z    is the block index
 *     blockDim.x,y,z    is the number of threads in a block
 *     threadIdx.x,y,z   is the thread index within the block
 */
__global__
void gpu_cast_impl(int batch_size, int input_size, unsigned char* __restrict__ input, float* __restrict__ output) {
    const int start_pos = blockDim.x * blockIdx.x;
    const int last_pos = blockDim.x * (blockIdx.x + 1);
    for (int i=start_pos; i<last_pos; ++i) {
        output[i] = static_cast<float>(input[i]);
    }
}


void gpu_cast(int batch_size, int input_size, unsigned char* input, float* output, cudaStream_t stream) {
    const int nblocks = batch_size;
    const int threads_per_block = 1;
    gpu_cast_impl<<<nblocks, threads_per_block, 0, stream>>>(batch_size, input_size, input, output);
}
