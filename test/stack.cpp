#include <Halide.h>

#include "util.h"

using namespace Halide;

class HStack1Generator : public Halide::Generator<HStack1Generator> {
public:
    Input<Buffer<float>> input_a {"input_a", 1};
    Input<Buffer<float>> input_b {"input_b", 1};
    Output<Buffer<float>> out{"out", 1};

    void generate() {
        out = hstack1_func(input_a, input_b, input_a.dim(0).extent());
    }
};

class HStack2Generator : public Halide::Generator<HStack2Generator> {
public:
    Input<Buffer<float>> input_a {"input_a", 2};
    Input<Buffer<float>> input_b {"input_b", 2};
    Output<Buffer<float>> out{"out", 2};

    void generate() {
        out = hstack2_func(input_a, input_b, input_a.dim(0).extent());
    }
};


class VStack1Generator : public Halide::Generator<VStack1Generator> {
public:
    Input<Buffer<float>> input_a {"input_a", 1};
    Input<Buffer<float>> input_b {"input_b", 1};
    Output<Buffer<float>> out{"out", 2};

    void generate() {
        out = vstack1_func(input_a, input_b, input_a.dim(0).extent());
    }
};

class VStack2Generator : public Halide::Generator<VStack2Generator> {
public:
    Input<Buffer<float>> input_a {"input_a", 2};
    Input<Buffer<float>> input_b {"input_b", 2};
    Output<Buffer<float>> out{"out", 2};

    void generate() {
        out = vstack2_func(input_a, input_b, input_a.dim(1).extent());
    }
};

HALIDE_REGISTER_GENERATOR(HStack1Generator, hstack1)
HALIDE_REGISTER_GENERATOR(HStack2Generator, hstack2)
HALIDE_REGISTER_GENERATOR(VStack1Generator, vstack1)
HALIDE_REGISTER_GENERATOR(VStack2Generator, vstack2)
