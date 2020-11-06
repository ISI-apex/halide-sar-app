#ifndef SIGNAL_H
#define SIGNAL_H

#include <Halide.h>

#include "util.h"

using namespace Halide;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline Func taylor_func(Expr num, Expr S_L = Expr(43), const std::string &name = "taylor") {
    Var x{"x"};
    // need to define our domain
    RDom n(0, num, "n");

    Expr xi = linspace(Expr(-0.5), Expr(0.5), num, x);

    Expr A("A");
    A = Expr(1.0 / (double)M_PI) * acosh(pow(10, S_L * Expr(1.0/20)));
    Expr n_bar("n_bar");
    n_bar = ConciseCasts::i32(2 * pow(A, 2) + Expr(0.5)) + Expr(1);
    Expr sigma_p("sigma_p");
    sigma_p = n_bar / sqrt(pow(A, 2) + (pow(n_bar - Expr(0.5), 2)));

    Func F_m_num;
    Func F_m_den;
    Func F_m;
    RDom r = RDom(0, n_bar - 1, "r");
    F_m_num(x) = Expr(1.0);
    F_m_den(x) = Expr(1.0);
    F_m_num(x) *= pow(-1, x + 2) * (Expr(1.0) - pow(x + 1, 2) / pow(sigma_p, 2) / (pow(A, 2) + pow(r.x + Expr(0.5), 2)));
    F_m_den(x) = select(x == r.x, F_m_den(x), F_m_den(x) * (Expr(1.0) - pow(x + 1, 2) / pow(r.x + 1, 2)));
    F_m(x) = F_m_num(x) / F_m_den(x);

    Func w("w");
    w(x) = Expr(1.0);
    w(x) += F_m(r) * cos(Expr(2 * (double)M_PI) * (r + 1) * xi);

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
