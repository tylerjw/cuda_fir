#include <iostream>
#include <fstream>
#include <string>
#include "cuda_fir.h"
#include "fir.h"

//Low pass filter (fc = 0.20)
#define FILTER_LEN  63
float taps[ FILTER_LEN ] =
{
   4.8345302E-04,  -6.4293204E-19,  -5.7762280E-04,  -1.0944333E-03,  -1.3166539E-03,
  -9.9289565E-04,   1.2681327E-18,   1.4906743E-03,   2.9399308E-03,   3.5607048E-03,
   2.6466098E-03,  -2.6340092E-18,  -3.7491232E-03,  -7.1519293E-03,  -8.3864776E-03,
  -6.0498642E-03,   4.3973074E-18,   8.1579935E-03,   1.5279549E-02,   1.7675909E-02,
   1.2646703E-02,  -6.1148988E-18,  -1.7101232E-02,  -3.2468808E-02,  -3.8497399E-02,
  -2.8648707E-02,   7.3551408E-18,   4.5088548E-02,   9.8908245E-02,   1.5013254E-01,
   1.8689591E-01,   2.0025675E-01,   1.8689591E-01,   1.5013254E-01,   9.8908245E-02,
   4.5088548E-02,   7.3551408E-18,  -2.8648707E-02,  -3.8497399E-02,  -3.2468808E-02,
  -1.7101232E-02,  -6.1148988E-18,   1.2646703E-02,   1.7675909E-02,   1.5279549E-02,
   8.1579935E-03,   4.3973074E-18,  -6.0498642E-03,  -8.3864776E-03,  -7.1519293E-03,
  -3.7491232E-03,  -2.6340092E-18,   2.6466098E-03,   3.5607048E-03,   2.9399308E-03,
   1.4906743E-03,   1.2681327E-18,  -9.9289565E-04,  -1.3166539E-03,  -1.0944333E-03,
  -5.7762280E-04,  -6.4293204E-19,   4.8345302E-04
};

int main(int argc, char const *argv[])
{
	const size_t cBlockLen = (size_t)20e6;

	// parse input arguments
	if (argc < 2)
	{
		std::cout << "Usage: " << argv[0] << " input cpu_out cuda_out " << std::endl;
		return 1;
	}

	// Filters
	Fir cpu_fir(taps, FILTER_LEN);
	CudaFir cuda_fir(taps, FILTER_LEN);

	std::string inFilename(argv[1]);
	std::string cpuFilename(argv[2]);
	std::string cudaFilename(argv[3]);
	std::ifstream in (inFilename, std::ios::in | std::ios::binary);
	std::ofstream cpu_out (cpuFilename, std::ios::out | std::ios::binary);
	std::ofstream cuda_out (cudaFilename, std::ios::out | std::ios::binary);


	// get the length of the input file
	sampleType * inBuffer = new sampleType [cBlockLen];
	sampleType * outBuffer = new sampleType [cBlockLen];

	while (in.good() and cpu_out.good() and cuda_out.good() and !in.eof())
	{
		try {
			in.read((char*)inBuffer, sizeof(sampleType) * cBlockLen);
		} catch (std::ifstream::failure e) {
			std::cerr << "Exception reading file" << std::endl;
			std::cerr << e.what() << std::endl;
			break;
		}

		size_t length = in.gcount() / sizeof(sampleType);
		if (length == 0)
		{
			// we are done
			break;
		}

		cpu_fir.filter(inBuffer, outBuffer, length);

		try {
			cpu_out.write((char*)outBuffer, sizeof(sampleType) * length);
		} catch (std::ostream::failure e) {
			std::cerr << "Exception writing file" << std::endl;
			std::cerr << e.what() << std::endl;
			break;
		}

		cuda_fir.filter(inBuffer, outBuffer, length);
		
		try {
			cuda_out.write((char*)outBuffer, sizeof(sampleType) * length);
		} catch (std::ostream::failure e) {
			std::cerr << "Exception writing file" << std::endl;
			std::cerr << e.what() << std::endl;
			break;
		}
	}

	in.close();
	cpu_out.close();
	cuda_out.close();

	delete[] inBuffer;
	delete[] outBuffer;

	return 0;
}