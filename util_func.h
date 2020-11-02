/**
 * Func utilities.
 */
#ifndef UTIL_FUNC_H
#define UTIL_FUNC_H

#include <Halide.h>

using namespace Halide;

inline Func linspace_func(Expr start, Expr stop, RDom r, const std::string &name = "linspace") {
    Var x{"x"};
    Expr step("step");
    step = (stop - start) / (r.x.extent() - Expr(1));
    Func linspace(name);
    linspace(x) = start;
    linspace(r) = start + r * step;
    // force "stop" value to avoid floating point deviations
    linspace(r.x.extent() - 1) = stop;
    return linspace;
}

inline Func arange_func(Expr start, Expr stop, Expr step, const std::string &name = "arange") {
    Var x{"x"};
    Expr extent("extent");
    extent = ConciseCasts::i32(ceil((stop - start) / step));
    RDom r(0, extent, "r");
    Func arange(name);
    arange(x) = start;
    arange(r) = start + r * step;
    return arange;
}

#endif
