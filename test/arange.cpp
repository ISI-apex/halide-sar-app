#include <Halide.h>

#include "util.h"

using namespace Halide;

class ArangeGenerator : public Halide::Generator<ArangeGenerator> {
public:
    Input<float> input_start {"input_start"};
    Input<float> input_step {"input_step"};
    Output<Buffer<float>> output_buffer{"output_buffer", 1};

    void generate() {
    	Var x{"x"};
        output_buffer(x) = arange(input_start, input_step, x);
    }
};

HALIDE_REGISTER_GENERATOR(ArangeGenerator, arange)
