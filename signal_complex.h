#ifndef SIGNAL_COMPLEX_H
#define SIGNAL_COMPLEX_H

#include <Halide.h>

#include "complexfunc.h"

using namespace Halide;

inline ComplexFunc pad_func(ComplexFunc in, Expr in_x_len, Expr in_y_len, Expr out_x_len, Expr out_y_len) {
    RDom r(0, out_x_len, 0, out_y_len, "r");
    Var x{"x"}, y{"y"}, c{"c"};
    // leading pad entries (trailing entries are of equal size or +1 larger if length difference is odd)
    Expr x_pad("x_pad");
    x_pad = (out_x_len - in_x_len) / 2;
    Expr y_pad("y_pad");
    y_pad = (out_y_len - in_y_len) / 2;
    ComplexFunc pad(c, "pad");
    pad(x, y) = ComplexExpr(c, 0.0f, 0.0f);
    // The clamp works around a bounds-related compile issue when lengths are non-trivially computed
    pad(r.x, r.y) =
        select(c,
               r.x < x_pad || r.x >= in_x_len + x_pad || r.y < y_pad || r.y >= in_y_len + y_pad,
               ComplexExpr(c, 0.0f, 0.0f),
               in(clamp(r.x - x_pad, 0, in_x_len - 1), clamp(r.y - y_pad, 0, in_y_len - 1)));
    return pad;
}

#endif
