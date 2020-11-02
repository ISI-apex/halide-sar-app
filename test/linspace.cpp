#include <Halide.h>

#include "util_func.h"

using namespace Halide;

class LinspaceGenerator : public Halide::Generator<LinspaceGenerator> {
public:
    Input<float> input_start {"input_start"};
    Input<float> input_stop {"input_stop"};
    Input<int> input_num {"input_num"};
    Output<Buffer<float>> output_buffer{"output_buffer", 1};

    void generate() {
        RDom r(0, input_num, "r");
        output_buffer = linspace_func(input_start, input_stop, r);
    }
};

HALIDE_REGISTER_GENERATOR(LinspaceGenerator, linspace)
