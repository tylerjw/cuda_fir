#include <cuda.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cuda.h>
#include "cmix.h"

// block width must be wider than number of taps
#define BLOCK_WIDTH     1024
#define MAX_N_TAPS      100
#define SHARED_WIDTH    (MAX_N_TAPS - 1 + BLOCK_WIDTH)

 

cudaCmix::cudaCmix(float* coeffs, size_t length)
    : cTapsLen(length)
    , stateLen(length - 1)
{
    if (length > MAX_N_TAPS)
    {
        std::stringstream ss;
        ss << "cudaCmix: Filter Length " << length 
            << " out of range (max=" << MAX_N_TAPS << ")";
        throw std::out_of_range(ss.str());
    }
    taps = new float[cTapsLen];
    memcpy(taps, coeffs, sizeof(float) * length);
    state = new sampleType[stateLen];
    memset(state, 0, sizeof(sampleType) * stateLen);
}

cudaCmix::~cudaCmix()
{
    delete[] taps;
    delete[] state;
}


void cudaCmix::filter(sampleType* input, sampleType* output, size_t length)
{
    if (length == 0) 
    {
        // nothing to do here
        return;
    }

    float2 *din, *dout;
    float * dtaps;
    size_t outputSize = length*sizeof(float2);
    size_t inputSize = (stateLen+length)*sizeof(float2);
    size_t stateSize = stateLen*sizeof(float2);
    size_t tapsSize = cTapsLen*sizeof(float);

    cudaMalloc(&din, inputSize);
    cudaMalloc(&dout, outputSize);
    cudaMalloc(&dtaps, tapsSize);

    cudaMemcpy(din, state, stateSize, cudaMemcpyHostToDevice);
    cudaMemcpy(&din[stateLen], input, outputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dtaps, taps, tapsSize, cudaMemcpyHostToDevice);

    const int threadsPerBlock = BLOCK_WIDTH;
    const int numBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;

    //cudaCmix<<< numBlocks, threadsPerBlock >>>(dtaps, cTapsLen, din, dout, length);

    cudaDeviceSynchronize();

    // check for errors running kernel
    //checkCUDAError("kernel invocation");
 
    // device to host copy
    cudaMemcpy(output, dout, outputSize, cudaMemcpyDeviceToHost );
    cudaMemcpy(state, &din[length], stateSize, cudaMemcpyDeviceToHost);
 
    // Check for any CUDA errors
    //checkCUDAError("memcpy");
    
    cudaFree(dtaps);
    cudaFree(dout);
    cudaFree(din);
}

