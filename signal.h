#ifndef SIGNAL_H
#define SIGNAL_H

#include <Halide.h>

#include "util.h"

using namespace Halide;

inline Func taylor_func(Expr num, Expr S_L = Expr(43), const std::string &name = "taylor") {
    // need to define our domain
    RDom n(0, num, "n");
    Func xi("xi");
    xi = linspace_func(Expr(-0.5f), Expr(0.5f), n);

    Expr A("A");
    A = Expr(1.0f / (float)M_PI) * acosh(pow(10, S_L * Expr(1.0f/20)));
    Expr n_bar("n_bar");
    n_bar = ConciseCasts::i32(2 * pow(A, 2) + Expr(0.5f)) + Expr(1);
    Expr sigma_p("sigma_p");
    sigma_p = n_bar / sqrt(pow(A, 2) + (pow(n_bar - Expr(0.5f), 2)));

    Var x{"x"};
    Func F_m_num;
    Func F_m_den;
    Func F_m;
    RDom r = RDom(0, n_bar - 1, "r");
    F_m_num(x) = Expr(1.0f);
    F_m_den(x) = Expr(1.0f);
    F_m_num(x) *= pow(-1, x + 2) * (Expr(1.0f) - pow(x + 1, 2) / pow(sigma_p, 2) / (pow(A, 2) + pow(r.x + Expr(0.5f), 2)));
    F_m_den(x) = select(x == r.x, F_m_den(x), F_m_den(x) * (Expr(1.0f) - pow(x + 1, 2) / pow(r.x + 1, 2)));
    F_m(x) = F_m_num(x) / F_m_den(x);

    Func w("w");
    w(x) = Expr(1.0f);
    w(x) += F_m(r) * cos(Expr(2 * (float)M_PI) * (r + 1) * xi(x));

    // separate iteration domain (RDom) to get maximum
    Func out(name);
    out(x) = w(x) / maximum(w(n));
    return out;
}

// Normalizes dB between 0 and 1, then scales by Type t's max (e.g., UInt(8) or UInt(16))
// Return type depends on dB, but can then be safely cast to Type t
inline Expr dB_scale(Expr dB, Expr min_dB, Expr max_dB, Type t) {
    return clamp(normalize(dB, min_dB, max_dB) * t.max(), 0, t.max());
}

#endif
