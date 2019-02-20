#ifndef _CUDA_ERROR_H_
#define _CUDA_ERROR_H_

void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
        cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}

#endif