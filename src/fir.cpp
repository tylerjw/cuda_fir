#include <cstring>
#include "fir.h"

Fir::Fir(float* coeffs, size_t length)
	: cTapsLen(length)
	, cOffset(length - 1)
	, state(NULL)
	, blockLen(0)
	, stateLen(0)
{
	taps = new float[cTapsLen];
	memcpy(taps, coeffs, sizeof(float) * length);
}

Fir::~Fir()
{
	delete[] taps;
	if (state) {
		delete[] state;
	}
}

void Fir::resizeState(size_t length)
{
	if (length > blockLen)
	{
		size_t tempLen = cTapsLen + length - 1;
		sampleType* temp = new sampleType[tempLen];
		memset(temp, 0, sizeof(sampleType) * tempLen);

		if (state) {
			memcpy(temp, state, sizeof(sampleType) * stateLen);
		}
		delete[] state;
		state = temp;
		stateLen = tempLen;
		blockLen = length;
	}
}


void Fir::filter(sampleType* input, sampleType* output, size_t length)
{
	sampleType acc;
	
	if (length == 0) 
	{
		// nothing to do here
		return;
	}

	// resize the state buffer if we need to
	resizeState(length);

	for (size_t i; i < length; i++)
	{
		state[i + cOffset] = input[i];
		acc = sampleType(0,0);

		for (size_t j = 0; j < cTapsLen; j++)
		{
			acc += sampleType(state[i+j].real() * taps[j], 
				state[i+j].imag() * taps[j]);
		}
		output[i] = acc;
	}

	for (size_t i = 0; i < cOffset; i++)
	{
		state[i] = state[i+length];
	}
}

