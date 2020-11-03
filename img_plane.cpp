#include <Halide.h>

#include "util.h"
#include "util_func.h"

using namespace Halide;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// u and v
Func ip_uv(Expr n, Expr d, const std::string &name = "uv") {
    Var x{"x"};
    Func arange = arange_func(-n / 2, n / 2, 1);
    Func uv(name);
    uv(x) = arange(x) * d;
    return uv;
} 

// k_u and k_v
Expr ip_k(Expr n, Expr d, Var x) {
    RDom r(0, n, "r");
    Func ls = linspace_func(Expr(-1.0) / (2 * d), Expr(1.0) / (2 * d), r);
    return Expr(2 * M_PI) * ls(x);
}

// v_hat and u_hat
Expr ip_hat(Func a, Func b, Expr extent, Var x) {
    RDom r(0, extent);
    Func cross("cross");
    cross(x) = cross3(a, b, x);
    return cross(x) / norm(cross(r));
}

class ImgPlaneUVGenerator : public Halide::Generator<ImgPlaneUVGenerator> {
public:
    Input<int> n {"n"}; // depends on upsample
    Input<double> d {"d"};
    Output<Buffer<double>> out {"out", 1}; // of n length 

    void generate() {
        out = ip_uv(n, d);
    }
};

class ImgPlaneKGenerator : public Halide::Generator<ImgPlaneKGenerator> {
public:
    Input<int> n {"n"}; // depends on upsample
    Input<double> d {"d"};
    Output<Buffer<double>> k {"k", 1}; // of n length 

    void generate() {
        Var x{"x"};
        k(x) = ip_k(n, d, x);
    }
};

class ImgPlaneVHatGenerator : public Halide::Generator<ImgPlaneVHatGenerator> {
public:
    Input<Buffer<float>> n_hat {"n_hat", 1};
    Input<Buffer<float>> R_c {"R_c", 1};
    Output<Buffer<double>> v_hat {"v_hat", 1};

    void generate() {
        Var x{"x"};
        v_hat(x) = ConciseCasts::f64(ip_hat(n_hat, R_c, n_hat.dim(0).extent(), x));
    }
};

class ImgPlaneUHatGenerator : public Halide::Generator<ImgPlaneUHatGenerator> {
public:
    Input<Buffer<double>> v_hat {"v_hat", 1};
    Input<Buffer<float>> n_hat {"n_hat", 1};
    Output<Buffer<double>> u_hat {"u_hat", 1};

    void generate() {
        Var x{"x"};
        u_hat(x) = ip_hat(v_hat, n_hat, v_hat.dim(0).extent(), x);
    }
};

class ImgPlanePixelLocsGenerator : public Halide::Generator<ImgPlanePixelLocsGenerator> {
public:
    Input<Buffer<double>> u {"u", 1};
    Input<Buffer<double>> v {"v", 1};
    Input<Buffer<double>> u_hat {"u_hat", 1};
    Input<Buffer<double>> v_hat {"v_hat", 1};
    Output<Buffer<double>> pixel_locs {"pixel_locs", 2};

    void generate() {
        Var x{"x"}, y{"y"};
        Expr u_extent = u.dim(0).extent();
        Expr v_extent = v.dim(0).extent();
        Expr uv_extent = u_extent * v_extent;

        // produces shape {2, 3}
        Func A("A");
        A(x, y) = select(x % 2 == 0, u_hat(y), v_hat(y));

        // produces shape {uv_extent, 2}
        Func b("b");
        b(x, y) = select(y == 0, u(x % u_extent), v(x / v_extent));

        // produces shapes {uv_extent, 3}
        RDom r(0, 2, "r");
        pixel_locs(x, y) = sum(A(r, y) * b(x, r));
    }
};

HALIDE_REGISTER_GENERATOR(ImgPlaneUVGenerator, ip_uv)
HALIDE_REGISTER_GENERATOR(ImgPlaneKGenerator, ip_k)
HALIDE_REGISTER_GENERATOR(ImgPlaneVHatGenerator, ip_v_hat)
HALIDE_REGISTER_GENERATOR(ImgPlaneUHatGenerator, ip_u_hat)
HALIDE_REGISTER_GENERATOR(ImgPlanePixelLocsGenerator, ip_pixel_locs)
