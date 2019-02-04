# CUDA FIR filter

Optimized cuda FIR filter implementation for complex sample data.  Generic CPU implementation for reference.  Python code for generating and testing vectors.

## Building

You will need a working install of the [Cuda Development Enviroment](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) with the binary directory in your path.

1. Add the compiler to your path.  Put this in your `.bashrc`:
```bash
# Cuda library includes
export PATH=$PATH:/usr/local/cuda/bin
```

2. Checkout the code
```bash
git clone https://github.com/tylerjw/cuda_fir.git
```

3. Create build directory
```bash
cd cuda_fir
mkdir build
cd build
```

4. Run cmake
```bash
cmake ../
```

5. Build
```bash
make
```

## Python Scripts

In the scripts directory there are a few python scripts.  You will need python3 with scipy and numpy to run them.

* `generateExampleData.py` - creates `noise.cap` and `filtered.cap` and plots fft of them into png files
* `samples.py` - plots fft of `cpuout.cap` and `cudaout.cap` in png files
* `testout.py` - compares `cpuout.cap` and `cudaout.cap`

## Running

1. Use `scripts/generateExampleData.py` to generate sample data.
2. Run the code to generate outputs.
3. Use `scripts/samples.py` and `scripts/testout.py` to compare.

```bash
python3 scripts/generateExampleData.py
build/bin/main noise.cap cpuout.cap cudaout.cap
python3 scripts/samples.py
python3 scripts/testout.py
```