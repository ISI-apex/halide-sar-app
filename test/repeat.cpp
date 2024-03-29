#include <Halide.h>

#include "util.h"

using namespace Halide;

class Repeat1Generator : public Halide::Generator<Repeat1Generator> {
public:
    Input<Buffer<float>> input_a {"input_a", 1};
    Output<Buffer<float>> output_buffer{"output_buffer", 1};

    void generate() {
        Var x{"x"};
        output_buffer(x) = repeat1(input_a, input_a.dim(0).extent(), x);
    }
};

HALIDE_REGISTER_GENERATOR(Repeat1Generator, repeat1)
