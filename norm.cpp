#include <Halide.h>

#include "util.h"

using namespace Halide;

class Norm1DGenerator : public Halide::Generator<Norm1DGenerator> {
public:
    Input<Buffer<float>> input{"input", 1};
    Output<float> output{"output"};

    void generate() {
        RDom r(0, input.dim(0).extent(), "r");
        Func in_func = input;
        output() = norm_expr(in_func(r));
    }
};

class Norm2DGenerator : public Halide::Generator<Norm2DGenerator> {
public:
    Input<Buffer<float>> input{"input", 2};
    Output<Buffer<float>> output{"output", 1};

    void generate() {
        Var x{"x"};
        RDom r(0, input.dim(0).extent(), "r");
        Func in_func = input;
        output(x) = norm_expr(in_func(r, x));
    }
};

HALIDE_REGISTER_GENERATOR(Norm1DGenerator, norm1d)
HALIDE_REGISTER_GENERATOR(Norm2DGenerator, norm2d)
