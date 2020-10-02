#include <stdio.h>

#include <Halide.h>

#include "complexfunc.h"
#include "signal.h"

using namespace Halide;

#define TAYLOR_S_L 17

class BackprojectionGenerator : public Halide::Generator<BackprojectionGenerator> {
public:
    Input<Buffer<float>> phs {"phs", 3}; // complex 2d input  (Halide thinks this is 3d: [2, x, y])
    Input<Buffer<float>> k_r {"k_r", 1};

    Output<Buffer<float>> output_buffer{"output_packed", 3}; // complex 2d output (Halide thinks this is 3d: [2, x, y])
    // Output<Buffer<float>> output_buffer{"output_packed", 2};

    Var c{"c"}, x{"x"}, y{"y"};

    void generate() {
        Expr nsamples("nsamples");
        nsamples = phs.dim(1).extent();
        Expr npulses("npulses");
        npulses = phs.dim(2).extent();
        // Func input_bounded = BoundaryConditions::constant_exterior(phs, Expr(0.0f));
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

        output_buffer(c, x, y) = phs_filt.inner(c, x, y);

        // ComplexFunc output(c, "output");
        // output(x, y) = input(x, y) * ComplexExpr(c, Expr(0.0f), Expr(1.0f));
        // output_buffer(c, x, y) = output.inner(c, x, y);
    }
};

HALIDE_REGISTER_GENERATOR(BackprojectionGenerator, backprojection)
