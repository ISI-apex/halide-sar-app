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

inline Expr log2f_expr(Expr x) {
    Expr log2 = log(x) / log(Expr(2.0f));
    return log2;
}

inline Expr norm_expr(Expr in) {
    return sqrt(sum(in * in));
}

#endif
