# Halide SAR Application Demo

An (in progress) SAR application written in Halide.
Based on the [RITSAR](https://github.com/dm6718/RITSAR) backprojection implementation.


## Prerequisites:

* [Halide](https://halide-lang.org/) (>= 10.0.0) and transitive dependencies (e.g., LLVM >= 9.x)
* [cnpy](https://github.com/rogersce/cnpy)


## Compiling

Build with CMake:

```sh
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/halide-install-prefix/lib64/cmake/Halide/;/path/to/cnpy-install-prefix/"
make -j
```


## Testing

To run the backprojection test:

```sh
./sarbp_test ../data/AFRL/pass1/HH_npy ../data/AFRL/pass1/HH_ip_npy
```
