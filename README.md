# Halide SAR Application Demo

An (in progress) SAR application written in Halide.
Based on the [RITSAR](https://github.com/dm6718/RITSAR) backprojection implementation.


## Prerequisites:

* C/C++ compiler with C++17 support, including standard library features, e.g. GCC >= 7.3
* [Halide](https://halide-lang.org/) (>= 10.0.0) and transitive dependencies (e.g., LLVM >= 9.x)
* [cnpy](https://github.com/rogersce/cnpy)
* [FFTW3](http://www.fftw.org/)


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
./sarbp_test ../data/AFRL/pass1/HH_npy 17 2 output.png -30.0 0.0
```

or with a GPU CUDA schedule:

```sh
./sarbp_test ../data/AFRL/pass1/HH_npy 17 2 output.png -30.0 0.0 cuda
```

### Sandia

To run the backprojection test with the Sandia dataset:
```sh
./sarbp_test ../data/Sandia/npy 30 2 output-Sandia.png -45.0 0.0
```

or with a GPU CUDA schedule:

```sh
./sarbp_test ../data/Sandia/npy 30 2 output-Sandia.png -45.0 0.0 cuda
```
