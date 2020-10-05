#include <stdio.h>

#include <Halide.h>

#include "complexfunc.h"
#include "signal.h"
#include "signal_complex.h"

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
        N_fft = ConciseCasts::i32(pow(2, ConciseCasts::i32(log2f_expr(nsamples  * UPSAMPLE)) + 1));
        ComplexFunc phs_pad(c, "phs_pad");
        phs_pad = pad_func(phs_filt, nsamples, npulses, N_fft, npulses);

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
