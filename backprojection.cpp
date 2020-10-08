#include <stdio.h>

#include <Halide.h>

#include "complexfunc.h"
#include "signal.h"
#include "signal_complex.h"

using namespace Halide;

#define TAYLOR_S_L 17

class BackprojectionPreFFTGenerator : public Halide::Generator<BackprojectionPreFFTGenerator> {
public:
    Input<Buffer<float>> phs {"phs", 3}; // complex 2d input  (Halide thinks this is 3d: [2, x, y])
    Input<Buffer<float>> k_r {"k_r", 1};
    Input<int> N_fft {"N_fft"};

    Output<Buffer<float>> output_buffer{"output_packed", 3}; // complex 2d output (Halide thinks this is 3d: [2, x, y])

    Var c{"c"}, x{"x"}, y{"y"};

    void generate() {
        Expr nsamples("nsamples");
        nsamples = phs.dim(1).extent();
        Expr npulses("npulses");
        npulses = phs.dim(2).extent();
        Func phs_func = phs;
        ComplexFunc input(c, phs_func, "input");

        // Create window
        Func win_x("win_x");
        win_x = taylor_func(nsamples, TAYLOR_S_L);
        Func win_y("win_y");
        win_y = taylor_func(npulses, TAYLOR_S_L);
        Func win("win");
        win(x, y) = win_x(x) * win_y(y);

        // Filter phase history
        Func filt("filt");
        filt(x) = abs(k_r(x));
        ComplexFunc phs_filt(c, "phs_filt");
        phs_filt(x, y) = input(x, y) * filt(x) * win(x, y);

        // Zero pad phase history
        ComplexFunc phs_pad(c, "phs_pad");
        phs_pad = pad_func(phs_filt, nsamples, npulses, N_fft, npulses);

        // shift
        ComplexFunc fftshift(c, "fftshift");
        fftshift = fftshift_func(phs_pad, N_fft, npulses);

        output_buffer(c, x, y) = fftshift.inner(c, x, y);

        phs_func.compute_root();
        win_x.compute_root();
        win_y.compute_root();
        win.compute_root();
        filt.compute_root();
        phs_filt.inner.compute_root();
        phs_pad.inner.compute_root();
        fftshift.inner.compute_root();
    }
};

class BackprojectionPostFFTGenerator : public Halide::Generator<BackprojectionPostFFTGenerator> {
public:
    Input<Buffer<float>> in {"in", 3};
    Input<int> nsamples {"nsamples"};
    Input<double> delta_r {"delta_r"};
    Input<Buffer<double>> u {"u", 1};
    Input<Buffer<double>> v {"v", 1};
    Input<Buffer<float>> pos {"pos", 2};
    Input<Buffer<double>> r {"r", 2};

    Output<Buffer<float>> output_buffer{"output_packed", 3};

    Var c{"c"}, x{"x"}, y{"y"};

    void generate() {
        Expr N_fft("N_fft");
        N_fft = in.dim(1).extent();
        Expr npulses("npulses");
        npulses = in.dim(2).extent();
        Func in_func = in;
        ComplexFunc input(c, in_func, "input");

        // shift
        ComplexFunc fftshift(c, "fftshift");
        fftshift = fftshift_func(input, N_fft, npulses);

        // dr
        RDom rn(0, N_fft, "rn");
        Func dr("dr");
        dr = linspace_func(-nsamples * delta_r / 2, nsamples * delta_r / 2, rn);

        Expr nu = u.dim(0).extent();
        Expr nv = v.dim(0).extent();

        // ri
        ComplexFunc img(c, "img");
        img(x) = ComplexExpr(c, 0.0f, 0.0f);
        // RDom ri(0, nu * nv, "ri");
        // img(ri) = ComplexExpr(c, 0.0f, 0.0f);

        // TODO: actually compute img values

        // reshape img
        // y dim is determined by output_buffer, but still need 'nu' for offset
        ComplexFunc img_rect(c, "img_rect");
        img_rect(x, y) = img(nu * y + x);
        output_buffer(c, x, y) = img_rect.inner(c, x, y);

        in_func.compute_root();
        dr.compute_root();
        fftshift.inner.compute_root();
        img.inner.compute_root();
        img_rect.inner.compute_root();
    }
};

HALIDE_REGISTER_GENERATOR(BackprojectionPreFFTGenerator, backprojection_pre_fft)
HALIDE_REGISTER_GENERATOR(BackprojectionPostFFTGenerator, backprojection_post_fft)
