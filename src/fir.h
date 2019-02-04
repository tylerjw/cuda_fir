#ifndef FIR_H
#define FIR_H

#include <complex>

typedef std::complex<float> sampleType;

class Fir 
{
public:
	Fir(float* coeffs, size_t length);
	~Fir();

	void filter(sampleType * input, 
		sampleType * output, 
		size_t length);

private:
	void resizeState(size_t length);

	float * taps;
	const size_t cTapsLen;
	const size_t cOffset;
	sampleType * state;
	size_t blockLen;
	size_t stateLen;
};

#endif // FIR_H