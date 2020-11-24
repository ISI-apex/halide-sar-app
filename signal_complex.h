#ifndef SIGNAL_COMPLEX_H
#define SIGNAL_COMPLEX_H

#include <Halide.h>

#include "halide_complexfunc.h"

using namespace Halide;

inline Halide::Tools::ComplexExpr pad(Halide::Tools::ComplexFunc in, Expr in_x_len, Expr in_y_len,
                       Halide::Tools::ComplexExpr pad_val,
                       Expr out_x_len, Expr out_y_len, Var c, Var x, Var y) {
    // leading pad entries (trailing entries are of equal size or +1 larger if length difference is odd)
    Expr x_pad = (out_x_len - in_x_len) / 2;
    Expr y_pad = (out_y_len - in_y_len) / 2;
    // The clamp works around a bounds-related compile issue when lengths are non-trivially computed
    return select(c,
                  x < x_pad || x >= in_x_len + x_pad || y < y_pad || y >= in_y_len + y_pad,
                  pad_val,
                  in(clamp(x - x_pad, 0, in_x_len - 1), clamp(y - y_pad, 0, in_y_len - 1)));
}

// swap left and right halves, top and bottom halves (swapping quadrants)
inline Halide::Tools::ComplexExpr fftshift(Halide::Tools::ComplexFunc in, Expr x_extent, Expr y_extent, Var x, Var y) {
    // use the ceiling for midpoint computation
    Expr x_mid = (x_extent / 2) + (x_extent % 2);
    Expr y_mid = (y_extent / 2) + (y_extent % 2);
    return in((x + x_mid) % x_extent, (y + y_mid) % y_extent);
}

#endif
