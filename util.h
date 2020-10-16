#ifndef UTIL_H
#define UTIL_H

#include <Halide.h>

using namespace Halide;

inline Func linspace_func(Expr start, Expr stop, RDom r) {
    Var x{"x"};
    Expr step("step");
    step = (stop - start) / (r.x.extent() - Expr(1));
    Func linspace("linspace");
    linspace(x) = start;
    linspace(r) = start + r * step;
    // force "stop" value to avoid floating point deviations
    linspace(r.x.extent() - 1) = stop;
    return linspace;
}

inline Func arange_func(Expr start, Expr stop, Expr step) {
    Var x{"x"};
    Expr extent("extent");
    extent = ConciseCasts::i32(ceil((stop - start) / step));
    RDom r(0, extent, "r");
    Func arange("arange");
    arange(x) = start;
    arange(r) = start + r * step;
    return arange;
}

// a and b are assumed to be vectors of length 3
inline Func cross3_func(Func a, Func b) {
    Var x{"x"};
    Func cross("cross");
    // // c_x = a_y * b_z − a_z * b_y
    // cross(0) = a(1) * b(2) - a(2) * b(1);
    // // c_y = a_z * b_x − a_x * b_z
    // cross(1) = a(2) * b(0) - a(0) * b(2);
    // // c_z = a_x * b_y − a_y * b_x
    // cross(2) = a(0) * b(1) - a(1) * b(0);
    cross(x) = a((x + 1) % 3) * b((x + 2) % 3) - a((x + 2) % 3) * b((x + 1) % 3);
    return cross;
}

inline Expr log2f_expr(Expr x) {
    Expr log2 = log(x) / log(Expr(2.0f));
    return log2;
}

inline Expr norm_expr(Expr in) {
    return sqrt(sum(in * in));
}

#endif
