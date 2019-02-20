#ifndef CUDA_FIR_H
#define CUDA_FIR_H

#include <complex>

typedef std::complex<float> sampleType;

class CudaFir 
{
public:
	CudaFir(float* coeffs, size_t length);
	~CudaFir();

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

#endif // CUDA_FIR_H