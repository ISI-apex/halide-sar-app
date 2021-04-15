cmake_minimum_required(VERSION 3.16)
project(halide-sar-app)

# Set up language settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)


list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Find Halide
find_package(Halide REQUIRED COMPONENTS Halide PNG)
include(CheckHalideDistributed)
check_halide_set_distributed(Halide_HAS_DISTRIBUTE Halide::shared::Halide)

# Find MPI
find_package(MPI)

# Find cnpy
find_package(Cnpy REQUIRED)

# Find pkg-config
find_package(PkgConfig REQUIRED)

# Find FFTW3 double precision
pkg_check_modules(FFTW fftw3 REQUIRED IMPORTED_TARGET)


option(HALIDE_SAR_TESTS "Enable tests" ON)
if(HALIDE_SAR_TESTS)
    enable_testing()
    add_subdirectory(test)
endif()


# Generators and Filters (libraries)

add_executable(img_plane.generator img_plane.cpp)
target_link_libraries(img_plane.generator PRIVATE Halide::Generator)
add_halide_library(ip_uv FROM img_plane.generator)
add_halide_library(ip_k FROM img_plane.generator)
add_halide_library(ip_v_hat FROM img_plane.generator)
add_halide_library(ip_u_hat FROM img_plane.generator)
add_halide_library(ip_pixel_locs FROM img_plane.generator
                                 FEATURES large_buffers)

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
target_compile_definitions(backprojection.generator PRIVATE $<$<BOOL:${Halide_HAS_DISTRIBUTE}>:WITH_DISTRIBUTE>)
target_link_libraries(backprojection.generator PRIVATE Halide::Generator)
add_halide_library(backprojection FROM backprojection.generator
                                  FEATURES c_plus_plus_name_mangling
                                           large_buffers
                                  PARAMS schedule=cpu)
add_halide_library(backprojection_cuda FROM backprojection.generator
                                       FEATURES c_plus_plus_name_mangling
                                                large_buffers
                                                cuda
                                       PARAMS schedule=gpu
                                              blocksize=16
                                              blocksize_gpu_tile=16
                                              blocksize_gpu_split_x=16)
add_halide_library(backprojection_cuda_split FROM backprojection.generator
                                       FEATURES c_plus_plus_name_mangling
                                                large_buffers
                                                cuda
                                       PARAMS schedule=gpu-split
                                              blocksize=16
                                              blocksize_gpu_tile=16
                                              blocksize_gpu_split_x=16)
if(Halide_HAS_DISTRIBUTE)
    add_halide_library(backprojection_distributed FROM backprojection.generator
                                                  FEATURES c_plus_plus_name_mangling
                                                           large_buffers
                                                  PARAMS schedule=cpu-distributed)
    add_halide_library(backprojection_cuda_distributed FROM backprojection.generator
                                                       FEATURES c_plus_plus_name_mangling
                                                                large_buffers
                                                                cuda
                                                       PARAMS schedule=gpu-distributed
                                                              blocksize=16
                                                              blocksize_gpu_tile=16
                                                              blocksize_gpu_split_x=16)
    add_halide_library(backprojection_cuda_split_distributed FROM backprojection.generator
                                                             FEATURES c_plus_plus_name_mangling
                                                                      large_buffers
                                                                      cuda
                                                             PARAMS schedule=gpu-split-distributed
                                                                    blocksize=16
                                                                    blocksize_gpu_tile=16
                                                                    blocksize_gpu_split_x=16)
endif()
add_halide_library(backprojection_auto_m16 FROM backprojection.generator
                                           FEATURES c_plus_plus_name_mangling
                                                    large_buffers
                                           AUTOSCHEDULER Halide::Mullapudi2016)

add_executable(img_output.generator img_output.cpp)
target_link_libraries(img_output.generator PRIVATE Halide::Generator)
add_halide_library(img_output_to_dB FROM img_output.generator
                                    FEATURES large_buffers)
add_halide_library(img_output_u8 FROM img_output.generator
                                 FEATURES large_buffers)


# Executables

add_executable(img_output_u8_test img_output_u8_test.cpp)
target_link_libraries(img_output_u8_test PRIVATE Halide::Halide
                                                 Halide::ImageIO
                                                 Cnpy::Cnpy
                                                 img_output_u8)

add_executable(sarbp sarbp.cpp dft.cpp PlatformData.cpp ImgPlane.cpp)
target_compile_options(sarbp PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-O2>)
target_link_libraries(sarbp PRIVATE Halide::Halide
                                    Halide::ImageIO
                                    Cnpy::Cnpy
                                    PkgConfig::FFTW
                                    ip_uv
                                    ip_k
                                    ip_v_hat
                                    ip_u_hat
                                    ip_pixel_locs
                                    backprojection
                                    backprojection_cuda
                                    backprojection_cuda_split
                                    backprojection_ritsar
                                    backprojection_ritsar_s
                                    backprojection_ritsar_p
                                    backprojection_ritsar_vp
                                    backprojection_auto_m16
                                    img_output_u8
                                    img_output_to_dB)
if(Halide_HAS_DISTRIBUTE)
    target_compile_definitions(sarbp PRIVATE WITH_DISTRIBUTE)
    target_link_libraries(sarbp PRIVATE backprojection_distributed
                                        backprojection_cuda_distributed
                                        backprojection_cuda_split_distributed)
endif()
if(MPI_FOUND)
    target_compile_definitions(sarbp PRIVATE WITH_MPI)
    target_include_directories(sarbp PRIVATE ${MPI_CXX_INCLUDE_DIRS})
    target_link_libraries(sarbp PRIVATE MPI::MPI_CXX)
endif()
