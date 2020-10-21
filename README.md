# Halide SAR Application Demo

An (in progress) SAR application written in Halide.
Based on the [RITSAR](https://github.com/dm6718/RITSAR) backprojection implementation.


## Prerequisites:

* [Halide](https://halide-lang.org/) (>= 10.0.0) and transitive dependencies (e.g., LLVM >= 9.x)
* [cnpy](https://github.com/rogersce/cnpy)
* [FFTW3](http://www.fftw.org/) - single precision library (`libfftw3f`).


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

To run the backprojection test:

```sh
./sarbp_test ../data/AFRL/pass1/HH_npy
```
