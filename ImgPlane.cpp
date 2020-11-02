#include <math.h>

#include <Halide.h>

#include "ImgPlane.h"

// Halide generators
#include "ip_uv.h"
#include "ip_k.h"
#include "ip_v_hat.h"
#include "ip_u_hat.h"
#include "ip_pixel_locs.h"

using namespace std;
using Halide::Runtime::Buffer;

#define RES_FACTOR 1.0
#define ASPECT 1.0

const int N_HAT[] = { 0, 0, 1 };

inline double ip_upsample(int n) {
    double l = log2((double) n);
    int r = (l - (int)l) > 0;
    return pow(2, (int)(l + r));
}

inline double ip_du(double delta_r, double res_factor, int nsamples, int nu) {
    return delta_r * res_factor * nsamples / nu;
}

inline double ip_dv(double aspect, double du) {
    return aspect * du;
}

ImgPlane img_plane_create(PlatformData &pd) {
    int nu = ip_upsample(pd.nsamples);
    int nv = ip_upsample(pd.npulses);

    double d_u = ip_du(pd.delta_r, RES_FACTOR, pd.nsamples, nu);
    double d_v = ip_dv(ASPECT, d_u);

    Buffer<double, 1> u(nu);
    ip_uv(nu, d_u, u);

    Buffer<double, 1> v(nv);
    ip_uv(nv, d_v, v);

    Buffer<double, 1> k_u(nu);
    ip_k(nu, d_u, k_u);

    Buffer<double, 1> k_v(nv);
    ip_k(nv, d_v, k_v);

    Buffer<const int, 1> n_hat(&N_HAT[0], 3);

    Buffer<double, 1> v_hat(3);
    ip_v_hat(n_hat, pd.R_c, v_hat);

    Buffer<double, 1> u_hat(3);
    ip_u_hat(v_hat, n_hat, u_hat);

    Buffer<double, 2> pixel_locs(nu*nv, 3);
    ip_pixel_locs(u, v, u_hat, v_hat, pixel_locs);

    return ImgPlane(nu, nv, d_u, d_v, u, v, k_u, k_v, u_hat, v_hat, pixel_locs);
}
