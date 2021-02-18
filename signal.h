#ifndef SIGNAL_H
#define SIGNAL_H

#include <Halide.h>

#include "util.h"

using namespace Halide;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class Taylor {
public:
    RDom n;
    RDom r;
    Func taylor;
    Func w;
    Func F_m;
    Func F_m_num;
    Func F_m_den;
    Taylor(Expr num, Expr S_L, Var x, const std::string &inner_name_prefix = "taylor") {
    // need to define our domain
        n = RDom(0, num, "n");

        Expr xi = linspace(Expr(-0.5), Expr(0.5), num, x);
        Expr A = Expr(1.0 / (double)M_PI) * acosh(pow(10, S_L * Expr(1.0/20)));
        Expr n_bar = ConciseCasts::i32(2 * pow(A, 2) + Expr(0.5)) + Expr(1);
        Expr sigma_p = n_bar / sqrt(pow(A, 2) + (pow(n_bar - Expr(0.5), 2)));

        taylor = Func(inner_name_prefix);
        F_m_num = Func(inner_name_prefix + "_F_m_num");
        F_m_den = Func(inner_name_prefix + "_F_m_den");
        F_m = Func(inner_name_prefix + "_F_m");
        r = RDom(0, n_bar - 1, inner_name_prefix + "_r");
        F_m_num(x) = Expr(1.0);
        F_m_den(x) = Expr(1.0);
        F_m_num(x) *= pow(-1, x + 2) * (Expr(1.0) - pow(x + 1, 2) / pow(sigma_p, 2) / (pow(A, 2) + pow(r.x + Expr(0.5), 2)));
        F_m_den(x) = select(x == r.x, F_m_den(x), F_m_den(x) * (Expr(1.0) - pow(x + 1, 2) / pow(r.x + 1, 2)));
        F_m(x) = F_m_num(x) / F_m_den(x);

        w = Func(inner_name_prefix + "_w");
        w(x) = Expr(1.0);
        w(x) += F_m(r) * cos(Expr(2 * (double)M_PI) * (r + 1) * xi);

        // separate iteration domain (RDom) to get maximum
        taylor(x) = w(x) / maximum(w(n), inner_name_prefix + "_maximum");
    }
    Taylor() {}
};

// Normalizes dB between 0 and 1, then scales by Type t's max (e.g., UInt(8) or UInt(16))
// Return type depends on dB, but can then be safely cast to Type t
inline Expr dB_scale(Expr dB, Expr min_dB, Expr max_dB, Type t) {
    return clamp(normalize(dB, min_dB, max_dB) * t.max(), 0, t.max());
}

#endif
