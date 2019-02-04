from samples import *

if __name__ == '__main__':
	vector = read_sample_file('.', 'cpuout.cap')
	test = read_sample_file('.','cudaout.cap')

	error = 0;

	assert(len(vector) == len(test))

	for i in range(len(vector)):
		error += abs(vector[i] - test[i])

	avg_error = error / len(vector)

	print("Total Error: {}".format(error))
	print("Avg Error: {}".format(avg_error))