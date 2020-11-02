#ifndef PLATFORM_DATA_H
#define PLATFORM_DATA_H

#include <string>

#include <Halide.h>

struct PlatformData {
    PlatformData(float B_IF,
                 double delta_r,
                 double chirprate,
                 double f_0,
                 int nsamples,
                 int npulses,
                 Halide::Runtime::Buffer<float, 1> freq,
                 Halide::Runtime::Buffer<float, 1> k_r,
                 Halide::Runtime::Buffer<double, 1>k_y,
                 Halide::Runtime::Buffer<float, 1> R_c,
                 Halide::Runtime::Buffer<double, 1> t,
                 Halide::Runtime::Buffer<float, 2> pos,
                 Halide::Runtime::Buffer<float, 3> phs):
                 B_IF(B_IF),
                 delta_r(delta_r),
                 chirprate(chirprate),
                 f_0(f_0),
                 nsamples(nsamples),
                 npulses(npulses),
                 freq(freq),
                 k_r(k_r),
                 k_y(k_y),
                 R_c(R_c),
                 t(t),
                 pos(pos),
                 phs(phs) {}

    float B_IF;
    double delta_r;
    double chirprate;
    double f_0;
    int nsamples;
    int npulses;
    Halide::Runtime::Buffer<float, 1> freq;
    Halide::Runtime::Buffer<float, 1> k_r;
    Halide::Runtime::Buffer<double, 1> k_y;
    Halide::Runtime::Buffer<float, 1> R_c;
    Halide::Runtime::Buffer<double, 1> t;
    Halide::Runtime::Buffer<float, 2> pos;
    Halide::Runtime::Buffer<float, 3> phs;
};

PlatformData platform_load(std::string platform_dir);

#endif
