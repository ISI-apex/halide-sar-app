# Halide SAR Application Demo

A SAR application written in Halide.
Based on the [RITSAR](https://github.com/dm6718/RITSAR) backprojection implementation.


## Prerequisites:

Off-the-shelf dependencies (probably available from your preferred package manager):

* [LLVM/Clang](https://llvm.org/) - exact version constraints depend on Halide's requirements; recommend LLVM >= 11
  * LLVM is required for Halide-generated libraries, so we recommend using the same LLVM to build all sources. However, other compilers with C++17 support (including standard library features, e.g., GCC >= 7.3) might work for non-Halide-generated sources.
* [CMake](https://cmake.org/) >= 3.16
* [CUDA](https://developer.nvidia.com/cuda-toolkit)
* [FFTW3](http://www.fftw.org/) - e.g., `fftw-devel` for RHEL-based systems, `libfftw3-dev` for Debian-based systems
* [libpng](http://www.libpng.org/pub/png/libpng.html) - e.g., `libpng-devel` for RHEL-based systems, `libpng-dev` for Debian-based systems
* [zlib](https://zlib.net/) - e.g., `zlib-devel` for RHEL-based systems, `zlib1g-dev` for Debian-based systems

Custom dependencies (may need to be built from source):

* [Halide](https://halide-lang.org/) >= 10.0.0
* [cnpy](https://github.com/rogersce/cnpy)

To enable distributed scheduling support:

* MPI - e.g., [OpenMPI](https://www.open-mpi.org/)
* [Distributed Halide](https://github.com/BachiLi/Halide/tree/distributed) (instead of upstream Halide listed above)


## Compiling

Build with CMake:

```sh
mkdir build
cd build
cmake .. -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_PREFIX_PATH="/path/to/halide-install-prefix/lib64/cmake/Halide/;/path/to/cnpy-install-prefix/"
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
