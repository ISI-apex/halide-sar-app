#include <Halide.h>

#include "signal.h"

using namespace Halide;

class TaylorGenerator : public Halide::Generator<TaylorGenerator> {
public:
    Input<int> nsamples {"nsamples"};
    Input<float> S_L {"S_L"};
    Output<Buffer<double>> output{"output", 1};

    void generate() {
        Var x{"x"};
        output(x) = taylor(nsamples, S_L, x);
    }
};

HALIDE_REGISTER_GENERATOR(TaylorGenerator, taylor)
