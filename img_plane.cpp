#include <Halide.h>

#include "util.h"

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
    Func cross = cross3_func(a, b);
    return cross(x) / norm_expr(cross(r));
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
    Input<Buffer<int>> n_hat {"n_hat", 1};
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
    Input<Buffer<int>> n_hat {"n_hat", 1};
    Output<Buffer<double>> u_hat {"u_hat", 1};

    void generate() {
        Var x{"x"};
        u_hat(x) = ip_hat(v_hat, n_hat, v_hat.dim(0).extent(), x);
    }
};

HALIDE_REGISTER_GENERATOR(ImgPlaneUVGenerator, ip_uv)
HALIDE_REGISTER_GENERATOR(ImgPlaneKGenerator, ip_k)
HALIDE_REGISTER_GENERATOR(ImgPlaneVHatGenerator, ip_v_hat)
HALIDE_REGISTER_GENERATOR(ImgPlaneUHatGenerator, ip_u_hat)
