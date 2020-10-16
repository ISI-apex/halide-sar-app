#include <Halide.h>

#include "util.h"

using namespace Halide;

class CrossGenerator : public Halide::Generator<CrossGenerator> {
public:
    Input<Buffer<float>> input_a {"input_a", 1};
    Input<Buffer<float>> input_b {"input_b", 1};
    Output<Buffer<float>> out{"out", 1};

    void generate() {
        out = cross3_func(input_a, input_b);
    }
};

HALIDE_REGISTER_GENERATOR(CrossGenerator, cross)
