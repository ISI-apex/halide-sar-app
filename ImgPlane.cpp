#include <Halide.h>

#include "ImgPlane.h"
#include "ip.h"

// Halide generators
#include "ip_uv.h"
#include "ip_k.h"
#include "ip_v_hat.h"
#include "ip_u_hat.h"
#include "ip_pixel_locs.h"

using namespace std;
using Halide::Runtime::Buffer;

ImgPlane img_plane_create(PlatformData &pd, double res_factor,
                          const float *_n_hat, double aspect, bool upsample,
                          bool is_distributed) {
    int nu;
    int nv;
    if (upsample) {
        nu = ip_upsample(pd.nsamples);
        nv = ip_upsample(pd.npulses);
    } else {
        nu = pd.nsamples;
        nv = pd.npulses;
    }

    double d_u = ip_du(pd.delta_r, res_factor, pd.nsamples, nu);
    double d_v = ip_dv(aspect, d_u);

    Buffer<double, 1> u(nu);
    ip_uv(nu, d_u, u);
    if (is_distributed) {
        u.set_distributed({nu});
    }
    u.set_host_dirty();

    Buffer<double, 1> v(nv);
    ip_uv(nv, d_v, v);
    if (is_distributed) {
        v.set_distributed({nv});
    }
    v.set_host_dirty();

    Buffer<double, 1> k_u(nu);
    ip_k(nu, d_u, k_u);
    if (is_distributed) {
        k_u.set_distributed({nu});
    }
    k_u.set_host_dirty();

    Buffer<double, 1> k_v(nv);
    ip_k(nv, d_v, k_v);
    if (is_distributed) {
        k_v.set_distributed({nv});
    }
    k_v.set_host_dirty();

    Buffer<const float, 1> n_hat(_n_hat, 3);
    n_hat.set_host_dirty();

    Buffer<double, 1> v_hat(3);
    ip_v_hat(n_hat, pd.R_c, v_hat);
    if (is_distributed) {
        v_hat.set_distributed({3});
    }
    v_hat.set_host_dirty();

    Buffer<double, 1> u_hat(3);
    ip_u_hat(v_hat, n_hat, u_hat);
    if (is_distributed) {
        u_hat.set_distributed({3});
    }
    u_hat.set_host_dirty();

    Buffer<double, 2> pixel_locs(nu*nv, 3);
    ip_pixel_locs(u, v, u_hat, v_hat, pixel_locs);
    if (is_distributed) {
        pixel_locs.set_distributed({nu*nv, 3});
    }
    pixel_locs.set_host_dirty();

    return ImgPlane(nu, nv, d_u, d_v, u, v, k_u, k_v, u_hat, v_hat, pixel_locs);
}
