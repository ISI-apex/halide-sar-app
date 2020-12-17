#include <Halide.h>

#include "signal.h"

using namespace Halide;

class TaylorGenerator : public Halide::Generator<TaylorGenerator> {
public:
    Input<int> nsamples {"nsamples"};
    Input<float> S_L {"S_L"};
    Taylor taylor;
    Output<Buffer<double>> output{"output", 1};

    void generate() {
        Var x{"x"};
        taylor = Taylor(nsamples, S_L, x);
        output(x) = taylor.series(x);
    }
};

HALIDE_REGISTER_GENERATOR(TaylorGenerator, taylor)
