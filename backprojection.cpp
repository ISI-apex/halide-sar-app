#include <stdio.h>

#include <Halide.h>

#include "complexfunc.h"
#include "signal.h"

using namespace Halide;

#define TAYLOR_S_L 17
#define UPSAMPLE 2

class BackprojectionGenerator : public Halide::Generator<BackprojectionGenerator> {
public:
    Input<Buffer<float>> phs {"phs", 3}; // complex 2d input  (Halide thinks this is 3d: [2, x, y])
    Input<Buffer<float>> k_r {"k_r", 1};

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
        Expr N_fft("N_fft");
        N_fft = ConciseCasts::i32(pow(2, ConciseCasts::i32(log2f_expr(nsamples * UPSAMPLE)) + 1));
        // if odd values, an extra padded column/row exists at the right/bottom
        Expr x_pad("x_pad");
        x_pad = (N_fft - nsamples) / 2;
        Expr y_pad("y_pad");
        y_pad = 0;
        RDom r(0, N_fft, 0, npulses, "r");
        ComplexFunc phs_pad(c, "phs_pad");
        phs_pad(x, y) = ComplexExpr(c, 0.0f, 0.0f);
        // The clamp in phs_filt works around a weird compile bug - maybe it can't reason through N_fft's computation
        phs_pad(r.x, r.y) = select(c,
                                   r.x < x_pad || r.x >= nsamples + x_pad || r.y < y_pad || r.y >= npulses + y_pad,
                                   ComplexExpr(c, 0.0f, 0.0f),
                                   phs_filt(clamp(r.x - x_pad, 0, nsamples - 1), r.y - y_pad));

        output_buffer(c, x, y) = phs_pad.inner(c, x, y);

        phs_func.compute_root();
        win_x.compute_root();
        win_y.compute_root();
        win.compute_root();
        filt.compute_root();
        phs_pad.inner.compute_root();
    }
};

HALIDE_REGISTER_GENERATOR(BackprojectionGenerator, backprojection)
