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

const float N_HAT[] = { 0, 0, 1 };

ImgPlane img_plane_create(PlatformData &pd,
                          double res_factor = 1.0,
                          const float n_hat[3] = N_HAT,
                          double aspect = 1.0,
                          bool upsample = true);

#endif
