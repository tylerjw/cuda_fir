#include <cuda.h>
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <cuda.h>
#include "cuda_fir.h"

// block width must be wider than number of taps
#define BLOCK_WIDTH		1024
#define MAX_N_TAPS		100
#define SHARED_WIDTH	(MAX_N_TAPS - 1 + BLOCK_WIDTH)

void checkCUDAError(const char* msg);

__global__ void cudaFir(
	float *taps, const size_t n_taps, 
	float2* input, 
	float2* output,
	size_t length)
{
	int remaining_outputs = length-blockDim.x*blockIdx.x;
	int n_outputs = (blockDim.x > remaining_outputs ? remaining_outputs : blockDim.x);
	int sharedWidth = n_taps-1 + n_outputs;
	__shared__ float2 inputShared[SHARED_WIDTH];
	__shared__ float sharedTaps[MAX_N_TAPS];

	int start = blockDim.x*blockIdx.x;
	int end = start+sharedWidth;
	int n_copies = ((end-start)/blockDim.x);
	int B = start + (n_copies*blockDim.x);
	int bOffset = (n_copies*blockDim.x);

	// copy 128 byte alligned sections in the middle
	for (int i = 0; i < n_copies; i++)
	{
		inputShared[i*blockDim.x + threadIdx.x] =
			input[start + i*blockDim.x + threadIdx.x];
	}

	// copy end
	if ((B+threadIdx.x) < end)
	{
		inputShared[bOffset+threadIdx.x] = input[B+threadIdx.x];
	}

	// copy taps into shared memory
	if (threadIdx.x < n_taps)
	{
		sharedTaps[threadIdx.x] = taps[threadIdx.x];
	}

	__syncthreads();

	if ((start + threadIdx.x) < length)
	{
		float2 acc = make_float2(0.f,0.f);

		for (size_t j = 0; j < n_taps; j++)
		{
			acc.x += inputShared[threadIdx.x+j].x*sharedTaps[j]; 
			acc.y += inputShared[threadIdx.x+j].y*sharedTaps[j];
		}
		output[start + threadIdx.x] = acc;
	}
}

CudaFir::CudaFir(float* coeffs, size_t length)
	: cTapsLen(length)
	, stateLen(length - 1)
{
	if (length > MAX_N_TAPS)
	{
		std::stringstream ss;
		ss << "CudaFir: Filter Length " << length 
			<< " out of range (max=" << MAX_N_TAPS << ")";
		throw std::out_of_range(ss.str());
	}
	taps = new float[cTapsLen];
	memcpy(taps, coeffs, sizeof(float) * length);
	state = new sampleType[stateLen];
	memset(state, 0, sizeof(sampleType) * stateLen);
}

CudaFir::~CudaFir()
{
	delete[] taps;
	delete[] state;
}


void CudaFir::filter(sampleType* input, sampleType* output, size_t length)
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

	cudaFir<<< numBlocks, threadsPerBlock >>>(dtaps, cTapsLen, din, dout, length);

	cudaDeviceSynchronize();

	// check for errors running kernel
	checkCUDAError("kernel invocation");
 
    // device to host copy
    cudaMemcpy(output, dout, outputSize, cudaMemcpyDeviceToHost );
    cudaMemcpy(state, &din[length], stateSize, cudaMemcpyDeviceToHost);
 
    // Check for any CUDA errors
    checkCUDAError("memcpy");
	
    cudaFree(dtaps);
    cudaFree(dout);
    cudaFree(din);
}

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