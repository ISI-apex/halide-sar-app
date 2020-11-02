#ifndef SIGNAL_COMPLEX_H
#define SIGNAL_COMPLEX_H

#include <Halide.h>

#include "complexfunc.h"

using namespace Halide;

inline ComplexExpr pad(ComplexFunc in, Var c, Expr in_x_len, Expr in_y_len,
                       ComplexExpr pad_val, RDom r) {
    // leading pad entries (trailing entries are of equal size or +1 larger if length difference is odd)
    Expr x_pad = (r.x.extent() - in_x_len) / 2;
    Expr y_pad = (r.y.extent() - in_y_len) / 2;
    // The clamp works around a bounds-related compile issue when lengths are non-trivially computed
    return select(c,
                  r.x < x_pad || r.x >= in_x_len + x_pad || r.y < y_pad || r.y >= in_y_len + y_pad,
                  pad_val,
                  in(clamp(r.x - x_pad, 0, in_x_len - 1), clamp(r.y - y_pad, 0, in_y_len - 1)));
}

// swap left and right halves, top and bottom halves (swapping quadrants)
inline ComplexExpr fftshift(ComplexFunc in, Expr x_extent, Expr y_extent, Var x, Var y) {
    // use the ceiling for midpoint computation
    Expr x_mid = (x_extent / 2) + (x_extent % 2);
    Expr y_mid = (y_extent / 2) + (y_extent % 2);
    return in((x + x_mid) % x_extent, (y + y_mid) % y_extent);
}

#endif
