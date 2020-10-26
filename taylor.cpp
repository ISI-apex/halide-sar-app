#include <Halide.h>

#include "signal.h"

using namespace Halide;

class TaylorGenerator : public Halide::Generator<TaylorGenerator> {
public:
    Input<int> nsamples {"nsamples"};
    Input<float> S_L {"S_L"};
    Output<Buffer<double>> output{"output", 1};

    void generate() {
        output = taylor_func(nsamples, S_L);
    }
};

HALIDE_REGISTER_GENERATOR(TaylorGenerator, taylor)
