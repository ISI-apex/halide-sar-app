#include <Halide.h>

#include "util_func.h"

using namespace Halide;

class ArangeGenerator : public Halide::Generator<ArangeGenerator> {
public:
    Input<float> input_start {"input_start"};
    Input<float> input_stop {"input_stop"};
    Input<float> input_step {"input_step"};
    Output<Buffer<float>> output_buffer{"output_buffer", 1};

    void generate() {
        output_buffer = arange_func(input_start, input_stop, input_step);
    }
};

HALIDE_REGISTER_GENERATOR(ArangeGenerator, arange)
