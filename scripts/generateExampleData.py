from samples import *
import numpy as np 
from scipy import signal
import scipy

def print_filter(taps, fc):
	numtaps = len(taps)
	print ("//Low pass filter (fc = {:.2f})".format(fc))
	print ("#define FILTER_LEN  {}".format(numtaps))
	print ("float taps[ FILTER_LEN ] = ")
	print ("{")
	for start in range(0,numtaps,5):

		for idx in (x for x in range(start,start+5) if x < numtaps):
			comma = ","
			if idx == numtaps-1:
				comma = ""
			print ("{:16.7E}{}".format(taps[idx],comma), end="")
		print ("")
	print ("};")


if __name__ == '__main__':
	# generate filter
	numtaps = 63
	fc = 0.2
	taps = signal.firwin(numtaps, fc)
	print_filter(taps, fc)

	# generate noise sample file
	n = int(20e6)
	mu, sigma = 0, 0.1
	noise = np.random.normal(mu,sigma,n) + 1j*np.random.normal(mu,sigma,n)
	
	fs = 200e3 # for plotting
	fftLen = 1024
	freqScale = 1e3

	write_sample_file(".", "noise.cap", noise, scipy.float32)
	fft_plot(noise, fs, "noise", fftLen, freqScale)

	y = signal.lfilter(taps, [1.0], noise)

	write_sample_file(".", "filtered.cap", y, scipy.float32)
	fft_plot(y, fs, "filtered", fftLen, freqScale)