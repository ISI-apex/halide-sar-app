# Halide SAR Application Demo

An (in progress) SAR application written in Halide.
Based on the [RITSAR](https://github.com/dm6718/RITSAR) backprojection implementation.


## Prerequisites:

* C/C++ compiler with C++17 support, including standard library features, e.g. GCC >= 7.3
* [Distributed Halide](https://github.com/BachiLi/Halide/tree/distributed) and transitive dependencies (e.g., LLVM 11 and MPI); see also [Halide](https://halide-lang.org/)
* [cnpy](https://github.com/rogersce/cnpy)
* [FFTW3](http://www.fftw.org/)
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
* MPI, e.g., [OpenMPI](https://www.open-mpi.org/)


## Compiling

Build with CMake:

```sh
mkdir build
cd build
export PKG_CONFIG_PATH=/path/to/fftw-install-prefix/lib/pkgconfig/
cmake .. -DCMAKE_PREFIX_PATH="/path/to/halide-install-prefix/lib64/cmake/Halide/;/path/to/cnpy-install-prefix/"
make -j
```


## Testing

Two datasets are available for testing.

* The AFRL dataset contains 469 radar pulses, and the output image is 512x512
* The Sandia dataset contains 1999 radar pulses, and the output image is 2048x2048

### AFRL

To run the backprojection test with the AFRL dataset:

```sh
./sarbp -p ../data/AFRL/pass1/HH_npy -o AFRL.png -d -30.0 -D 0.0 -t 17 -u 2
```

or with a distributed CPU schedule:

```sh
mpirun -np 4 ./sarbp -p ../data/AFRL/pass1/HH_npy -o AFRL-cpu_distributed.png -d -30.0 -D 0.0 -t 17 -u 2 -s cpu_distributed
```

or with a GPU CUDA schedule:

```sh
./sarbp -p ../data/AFRL/pass1/HH_npy -o AFRL-cuda.png -d -30.0 -D 0.0 -t 17 -u 2 -s cuda
```

or with a distributed GPU CUDA schedule:

```sh
mpirun -np 4 ./sarbp -p ../data/AFRL/pass1/HH_npy -o AFRL-cuda_distributed.png -d -30.0 -D 0.0 -t 17 -u 2 -s cuda_distributed
```

### Sandia

To run the backprojection test with the Sandia dataset:
```sh
./sarbp -p ../data/Sandia/npy -o Sandia.png -d -45.0 -D 0.0 -t 30 -u 2
```

or with a GPU CUDA schedule:

```sh
./sarbp -p ../data/Sandia/npy -o Sandia-cuda.png -d -45.0 -D 0.0 -t 30 -u 2 -s cuda
```
