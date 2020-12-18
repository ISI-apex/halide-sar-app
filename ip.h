#pragma once

#include <cmath>

inline double ip_upsample(int n) {
    double l = log2((double) n);
    int r = (l - (int)l) > 0;
    return pow(2, (int)(l + r));
}

inline double ip_du(double delta_r, double res_factor, int nsamples, int nu) {
    return delta_r * res_factor * nsamples / nu;
}

inline double ip_dv(double aspect, double du) {
    return aspect * du;
}

