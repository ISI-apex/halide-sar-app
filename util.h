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

#endif
