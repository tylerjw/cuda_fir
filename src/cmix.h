#ifndef CUDA_CMIX_H
#define CUDA_CMIX_H

#include <complex>

typedef std::complex<float> sampleType;

class cudaCmix 
{
public:
	cudaCmix(float* coeffs, size_t length);
	~cudaCmix();

	void filter(sampleType * input, 
		sampleType * output, 
		size_t length);

private:
	void resizeState(size_t length);

	float * taps;
	const size_t cTapsLen;
	sampleType * state;
	size_t stateLen;
};

#endif // CUDA_CMIX_H