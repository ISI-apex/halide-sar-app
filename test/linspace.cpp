#include <Halide.h>

#include "util.h"

using namespace Halide;

class LinspaceGenerator : public Halide::Generator<LinspaceGenerator> {
public:
    Input<float> input_start {"input_start"};
    Input<float> input_stop {"input_stop"};
    Input<int> input_num {"input_num"};
    Output<Buffer<float>> output_buffer{"output_buffer", 1};

    void generate() {
    	Var x{"x"};
        output_buffer(x) = linspace(input_start, input_stop, input_num, x);
    }
};

HALIDE_REGISTER_GENERATOR(LinspaceGenerator, linspace)
