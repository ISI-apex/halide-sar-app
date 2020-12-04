cmake_minimum_required(VERSION 3.16)
project(halide-sar-app)


# Set up language settings
set(CMAKE_CXX_STANDARD 17)
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


enable_testing()
add_subdirectory(test)


# Generators and Filters (libraries)

add_executable(img_plane.generator img_plane.cpp)
target_link_libraries(img_plane.generator PRIVATE Halide::Generator)
add_halide_library(ip_uv FROM img_plane.generator)
add_halide_library(ip_k FROM img_plane.generator)
add_halide_library(ip_v_hat FROM img_plane.generator)
add_halide_library(ip_u_hat FROM img_plane.generator)
add_halide_library(ip_pixel_locs FROM img_plane.generator)

add_executable(backprojection_ritsar.generator backprojection_ritsar.cpp)
target_link_libraries(backprojection_ritsar.generator PRIVATE Halide::Generator)
add_halide_library(backprojection_ritsar FROM backprojection_ritsar.generator
                                         FEATURES c_plus_plus_name_mangling
                                                  large_buffers
                                         PARAMS schedule=v)
add_halide_library(backprojection_ritsar_s FROM backprojection_ritsar.generator
                                           GENERATOR backprojection_ritsar
                                           FEATURES c_plus_plus_name_mangling
                                                    large_buffers
                                           PARAMS schedule=s)
add_halide_library(backprojection_ritsar_p FROM backprojection_ritsar.generator
                                           GENERATOR backprojection_ritsar
                                           FEATURES c_plus_plus_name_mangling
                                                    large_buffers
                                           PARAMS schedule=p)
add_halide_library(backprojection_ritsar_vp FROM backprojection_ritsar.generator
                                            GENERATOR backprojection_ritsar
                                            FEATURES c_plus_plus_name_mangling
                                                     large_buffers
                                            PARAMS schedule=vp)

add_executable(backprojection.generator backprojection.cpp)
target_link_libraries(backprojection.generator PRIVATE Halide::Generator)
add_halide_library(backprojection FROM backprojection.generator
                                  FEATURES c_plus_plus_name_mangling
                                           large_buffers)
add_halide_library(backprojection_cuda FROM backprojection.generator
                                       FEATURES c_plus_plus_name_mangling
                                                large_buffers
                                                cuda
                                       PARAMS blocksize=1024)
add_halide_library(backprojection_auto_m16 FROM backprojection.generator
                                           FEATURES c_plus_plus_name_mangling
                                                    large_buffers
                                           AUTOSCHEDULER Halide::Mullapudi2016)

add_executable(img_output.generator img_output.cpp)
target_link_libraries(img_output.generator PRIVATE Halide::Generator)
add_halide_library(img_output_to_dB FROM img_output.generator)
add_halide_library(img_output_u8 FROM img_output.generator)


# Executables

add_executable(img_output_u8_test img_output_u8_test.cpp)
target_include_directories(img_output_u8_test PRIVATE ${CNPY_INCLUDE_DIR}
                                                      ${ZLIB_INCLUDE_DIRS})
target_link_libraries(img_output_u8_test PRIVATE Halide::Halide
                                                 Halide::ImageIO
                                                 ${CNPY}
                                                 img_output_u8)

add_executable(sarbp_test sarbp.cpp PlatformData.cpp ImgPlane.cpp)
target_compile_options(sarbp_test PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>)
target_include_directories(sarbp_test PRIVATE ${CNPY_INCLUDE_DIR}
                                              ${ZLIB_INCLUDE_DIRS}
                                              ${FFTW_INCLUDE_DIRS})
target_link_directories(sarbp_test PRIVATE ${FFTW_LIBRARY_DIRS})
target_link_libraries(sarbp_test PRIVATE Halide::Halide
                                         Halide::ImageIO
                                         ${CNPY}
                                         ${FFTW_STATIC_LIBRARIES}
                                         ip_uv
                                         ip_k
                                         ip_v_hat
                                         ip_u_hat
                                         ip_pixel_locs
                                         backprojection
                                         backprojection_cuda
                                         backprojection_ritsar
                                         backprojection_ritsar_s
                                         backprojection_ritsar_p
                                         backprojection_ritsar_vp
                                         backprojection_auto_m16
                                         img_output_u8
                                         img_output_to_dB)
