#ifndef DLBS_TENSORRT_BACKEND_ENGINES_GPU_CAST
#define DLBS_TENSORRT_BACKEND_ENGINES_GPU_CAST

/**
 * @brief CUDA kernel to cast an array of type unsigned char to an array of type float.
 * @param batch_size is the number of instances in one batch.
 * @param input_size is the number of elements in one instance (for instance, one image).
 * @param input is the input array of type unsigned char
 * @param output is the output array of type float
 * @param stream is the cuda stream to use for computations.
 */
void gpu_cast(int batch_size, int input_size, unsigned char* __restrict__ input, float* __restrict__ output, cudaStream_t stream);

#endif