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

# Find FFTW3 double precision
pkg_check_modules(FFTW fftw3 REQUIRED)
link_directories(${FFTW_LIBRARY_DIRS})


enable_testing()
add_subdirectory(test)


# Generators and Filters (libraries)

add_executable(ip_uv.generator img_plane.cpp)
target_link_libraries(ip_uv.generator PRIVATE Halide::Generator)
add_halide_library(ip_uv FROM ip_uv.generator)

add_executable(ip_k.generator img_plane.cpp)
target_link_libraries(ip_k.generator PRIVATE Halide::Generator)
add_halide_library(ip_k FROM ip_k.generator)

add_executable(ip_v_hat.generator img_plane.cpp)
target_link_libraries(ip_v_hat.generator PRIVATE Halide::Generator)
add_halide_library(ip_v_hat FROM ip_v_hat.generator)

add_executable(ip_u_hat.generator img_plane.cpp)
target_link_libraries(ip_u_hat.generator PRIVATE Halide::Generator)
add_halide_library(ip_u_hat FROM ip_u_hat.generator)

add_executable(ip_pixel_locs.generator img_plane.cpp)
target_link_libraries(ip_pixel_locs.generator PRIVATE Halide::Generator)
add_halide_library(ip_pixel_locs FROM ip_pixel_locs.generator)

add_executable(backprojection_pre_fft.generator backprojection.cpp)
target_link_libraries(backprojection_pre_fft.generator PRIVATE Halide::Generator)
add_halide_library(backprojection_pre_fft FROM backprojection_pre_fft.generator
                                          FEATURES large_buffers)

add_executable(backprojection_post_fft.generator backprojection.cpp)
target_link_libraries(backprojection_post_fft.generator PRIVATE Halide::Generator)
add_halide_library(backprojection_post_fft FROM backprojection_post_fft.generator
                                           FEATURES large_buffers)

add_executable(img_output_to_dB.generator img_output.cpp)
target_link_libraries(img_output_to_dB.generator PRIVATE Halide::Generator)
add_halide_library(img_output_to_dB FROM img_output_to_dB.generator)

add_executable(img_output_u8.generator img_output.cpp)
target_link_libraries(img_output_u8.generator PRIVATE Halide::Generator)
add_halide_library(img_output_u8 FROM img_output_u8.generator)


# Executables

add_executable(img_output_u8_test img_output_u8_test.cpp)
target_include_directories(img_output_u8_test PRIVATE ${CNPY_INCLUDE_DIR}
                                                      ${ZLIB_INCLUDE_DIRS})
target_link_libraries(img_output_u8_test PRIVATE Halide::Halide
                                                 Halide::ImageIO
                                                 ${CNPY}
                                                 img_output_u8)

add_executable(sarbp_test sarbp.cpp)
target_compile_options(sarbp_test PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>)
target_include_directories(sarbp_test PRIVATE ${CNPY_INCLUDE_DIR}
                                              ${ZLIB_INCLUDE_DIRS}
                                              ${FFTW_INCLUDE_DIRS})
target_link_libraries(sarbp_test PRIVATE Halide::Halide
                                         Halide::ImageIO
                                         ${CNPY}
                                         ${FFTW_STATIC_LIBRARIES}
                                         ip_uv
                                         ip_k
                                         ip_v_hat
                                         ip_u_hat
                                         ip_pixel_locs
                                         backprojection_pre_fft
                                         backprojection_post_fft
                                         img_output_u8
                                         img_output_to_dB)
