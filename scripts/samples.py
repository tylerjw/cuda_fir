import numpy as np 
import scipy, os
import matplotlib as mpl 
import matplotlib.pyplot as plt 

def read_sample_file(path, filename, basetype=scipy.float32):
	cwd = os.path.dirname(os.path.realpath(__file__))
	os.chdir(path)
	packed_samples = scipy.fromfile(open(filename), dtype=basetype)
	y = packed_samples[::2] + 1j*packed_samples[1::2];
	os.chdir(cwd)
	return y

def write_sample_file(path, filename, samples, basetype=scipy.float32):
	cwd = os.path.dirname(os.path.realpath(__file__))
	os.chdir(path)
	y = [None]*len(samples)*2
	y[::2] = [x.real for x in samples]
	y[1::2] = [x.imag for x in samples]
	with open(filename, 'wb') as f:
		np.array(y, dtype=basetype).tofile(f)
	os.chdir(cwd)

def time_slice(y, Fs, start, end):
	return y[int(start*Fs):int(end*Fs)]

def fft_plot(y, Fs, filename, fftLen=0, freqScale = 1e3):
	if fftLen:
		y = y[:fftLen]

	Ts = 1/Fs
	t = np.arange(0,len(y)/Fs,Ts)
	n = len(y)
	T = n/Fs

	Y = np.abs(np.fft.fftshift(np.fft.fft(y)))
	frq = np.fft.fftshift(np.fft.fftfreq(n, d=Ts)) / freqScale

	fig = plt.figure()
	ax1 = fig.add_subplot(311)
	ax1.plot(t,y.real)
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Amplitude (Real)')

	ax2 = fig.add_subplot(312)
	ax2.plot(t,y.imag)
	ax1.set_xlabel('Time')
	ax1.set_ylabel('Amplitude (Imag)')

	ax3 = fig.add_subplot(313)
	ax3.plot(frq, Y, 'r')
	ax3.set_xlabel('Freq ({:.0E} Hz)'.format(freqScale))
	ax3.set_ylabel('|Y(freq)|')

	plt.savefig(filename+'.png')

if __name__ == '__main__':
	fs = 200e3 # for plotting
	fftLen = 1024
	freqScale = 1e3
	y = read_sample_file('.','cpuout.cap')
	fft_plot(y, fs, "cpuout", fftLen, freqScale)

	y = read_sample_file('.','cudaout.cap')
	fft_plot(y, fs, "cudaout", fftLen, freqScale)