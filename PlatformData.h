#ifndef PLATFORM_DATA_H
#define PLATFORM_DATA_H

#include <string>

#include <Halide.h>

struct PlatformData {
    PlatformData(std::optional<double> B,
                 std::optional<float> B_IF,
                 double delta_r,
                 std::optional<double> delta_t,
                 std::optional<double> chirprate,
                 std::optional<double> f_0,
                 int nsamples,
                 int npulses,
                 std::optional<double> vp,
                 std::optional<Halide::Runtime::Buffer<float, 1>> freq,
                 Halide::Runtime::Buffer<float, 1> k_r,
                 std::optional<Halide::Runtime::Buffer<double, 1>> k_y,
                 std::optional<Halide::Runtime::Buffer<float, 1>> n_hat,
                 Halide::Runtime::Buffer<float, 1> R_c,
                 std::optional<Halide::Runtime::Buffer<double, 1>> t,
                 Halide::Runtime::Buffer<float, 2> pos,
                 Halide::Runtime::Buffer<float, 3> phs):
                 B(B),
                 B_IF(B_IF),
                 delta_r(delta_r),
                 delta_t(delta_t),
                 chirprate(chirprate),
                 f_0(f_0),
                 nsamples(nsamples),
                 npulses(npulses),
                 vp(vp),
                 freq(freq),
                 k_r(k_r),
                 k_y(k_y),
                 n_hat(n_hat),
                 R_c(R_c),
                 t(t),
                 pos(pos),
                 phs(phs) {}

    std::optional<double> B;
    std::optional<float> B_IF;
    double delta_r;
    std::optional<double> delta_t;
    std::optional<double> chirprate;
    std::optional<double> f_0;
    int nsamples;
    int npulses;
    std::optional<double> vp;
    std::optional<Halide::Runtime::Buffer<float, 1>> freq;
    Halide::Runtime::Buffer<float, 1> k_r;
    std::optional<Halide::Runtime::Buffer<double, 1>> k_y;
    std::optional<Halide::Runtime::Buffer<float, 1>> n_hat;
    Halide::Runtime::Buffer<float, 1> R_c;
    std::optional<Halide::Runtime::Buffer<double, 1>> t;
    Halide::Runtime::Buffer<float, 2> pos;
    Halide::Runtime::Buffer<float, 3> phs;
};

PlatformData platform_load(std::string platform_dir);

#endif
