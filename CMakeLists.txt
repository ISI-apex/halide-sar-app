cmake_minimum_required(VERSION 3.16)
project(halide-sar-app)


# Set up language settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)


# Find Halide
find_package(Halide REQUIRED)

# Find cnpy
find_library(CNPY cnpy REQUIRED)
find_path(CNPY_INCLUDE_DIR cnpy.h
          HINTS ${CMAKE_PREFIX_PATH}
          PATH_SUFFIXES include
          REQUIRED)

# Find zlib - public dependency of cnpy and we need its header
find_package(ZLIB REQUIRED)

# Find pkg-config
find_package(PkgConfig REQUIRED)

# Find FFTW3 single precision
pkg_check_modules(FFTWF fftw3f REQUIRED)
link_directories(${FFTWF_LIBRARY_DIRS})


# Generators and Filters (libraries)

add_executable(linspace.generator linspace.cpp)
target_link_libraries(linspace.generator PRIVATE Halide::Generator)
add_halide_library(linspace FROM linspace.generator)

add_executable(arange.generator arange.cpp)
target_link_libraries(arange.generator PRIVATE Halide::Generator)
add_halide_library(arange FROM arange.generator)

add_executable(cross.generator cross.cpp)
target_link_libraries(cross.generator PRIVATE Halide::Generator)
add_halide_library(cross FROM cross.generator)

add_executable(hstack1.generator stack.cpp)
target_link_libraries(hstack1.generator PRIVATE Halide::Generator)
add_halide_library(hstack1 FROM hstack1.generator)

add_executable(hstack2.generator stack.cpp)
target_link_libraries(hstack2.generator PRIVATE Halide::Generator)
add_halide_library(hstack2 FROM hstack2.generator)

add_executable(vstack1.generator stack.cpp)
target_link_libraries(vstack1.generator PRIVATE Halide::Generator)
add_halide_library(vstack1 FROM vstack1.generator)

add_executable(vstack2.generator stack.cpp)
target_link_libraries(vstack2.generator PRIVATE Halide::Generator)
add_halide_library(vstack2 FROM vstack2.generator)

add_executable(taylor.generator taylor.cpp)
target_link_libraries(taylor.generator PRIVATE Halide::Generator)
add_halide_library(taylor FROM taylor.generator)

add_executable(norm1d.generator norm.cpp)
target_link_libraries(norm1d.generator PRIVATE Halide::Generator)
add_halide_library(norm1d FROM norm1d.generator)

add_executable(norm2d.generator norm.cpp)
target_link_libraries(norm2d.generator PRIVATE Halide::Generator)
add_halide_library(norm2d FROM norm2d.generator)

add_executable(interp.generator interp.cpp)
target_link_libraries(interp.generator PRIVATE Halide::Generator)
add_halide_library(interp FROM interp.generator)

add_executable(backprojection_pre_fft.generator backprojection.cpp)
target_link_libraries(backprojection_pre_fft.generator PRIVATE Halide::Generator)
add_halide_library(backprojection_pre_fft FROM backprojection_pre_fft.generator
                                          FEATURES large_buffers)

add_executable(backprojection_post_fft.generator backprojection.cpp)
target_link_libraries(backprojection_post_fft.generator PRIVATE Halide::Generator)
add_halide_library(backprojection_post_fft FROM backprojection_post_fft.generator
                                           FEATURES large_buffers)


# Executables

add_executable(linspace_test linspace_test.cpp)
target_link_libraries(linspace_test PRIVATE Halide::Halide
                                            linspace)

add_executable(arange_test arange_test.cpp)
target_link_libraries(arange_test PRIVATE Halide::Halide
                                          arange)

add_executable(taylor_test taylor_test.cpp)
target_link_libraries(taylor_test PRIVATE Halide::Halide
                                          taylor)

add_executable(norm_test norm_test.cpp)
target_link_libraries(norm_test PRIVATE Halide::Halide
                                        norm1d
                                        norm2d)

add_executable(sarbp_test sarbp.cpp)
target_compile_options(sarbp_test PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>)
target_include_directories(sarbp_test PRIVATE ${CNPY_INCLUDE_DIR}
                                              ${ZLIB_INCLUDE_DIRS}
                                              ${FFTWF_INCLUDE_DIRS})
target_link_libraries(sarbp_test PRIVATE Halide::Halide
                                         ${CNPY}
                                         ${FFTWF_STATIC_LIBRARIES}
                                         backprojection_pre_fft
                                         backprojection_post_fft)


# Tests

enable_testing()

add_executable(cross_test cross_test.cpp)
target_link_libraries(cross_test PRIVATE Halide::Halide
                                         cross)
add_test(cross cross_test)

add_executable(interp_test interp_test.cpp)
target_link_libraries(interp_test PRIVATE Halide::Halide
                                          interp)
add_test(interp interp_test)

add_executable(stack_test stack_test.cpp)
target_link_libraries(stack_test PRIVATE Halide::Halide
                                         hstack1
                                         hstack2
                                         vstack1
                                         vstack2)
add_test(stack stack_test)
