#ifndef IMG_PLANE_H
#define IMG_PLANE_H

#include <Halide.h>

#include "PlatformData.h"

struct ImgPlane {
    ImgPlane(int nu, int nv, int d_u, int d_v,
             Halide::Runtime::Buffer<double, 1> u,
             Halide::Runtime::Buffer<double, 1> v,
             Halide::Runtime::Buffer<double, 1> k_u,
             Halide::Runtime::Buffer<double, 1> k_v,
             Halide::Runtime::Buffer<double, 1> u_hat,
             Halide::Runtime::Buffer<double, 1> v_hat,
             Halide::Runtime::Buffer<double, 2> pixel_locs) :
             nu(nu), nv(nv), d_u(d_u), d_v(d_v),
             u(u), v(v), k_u(k_u), k_v(k_v), u_hat(u_hat), v_hat(v_hat),
             pixel_locs(pixel_locs) {}

    int nu;
    int nv;
    int d_u;
    int d_v;
    Halide::Runtime::Buffer<double, 1> u;
    Halide::Runtime::Buffer<double, 1> v;
    Halide::Runtime::Buffer<double, 1> k_u;
    Halide::Runtime::Buffer<double, 1> k_v;
    Halide::Runtime::Buffer<double, 1> u_hat;
    Halide::Runtime::Buffer<double, 1> v_hat;
    Halide::Runtime::Buffer<double, 2> pixel_locs;
};

const double RES_FACTOR = 1.0;
const float N_HAT[] = { 0, 0, 1 };
const double ASPECT = 1.0;

// Note: n_hat must be array of length 3
ImgPlane img_plane_create(PlatformData &pd,
                          double res_factor = RES_FACTOR,
                          const float *n_hat = &N_HAT[0],
                          double aspect = ASPECT,
                          bool upsample = true);

#endif
