#include <Halide.h>

#include "util.h"

using namespace Halide;

class HStack1Generator : public Halide::Generator<HStack1Generator> {
public:
    Input<Buffer<float>> input_a {"input_a", 1};
    Input<Buffer<float>> input_b {"input_b", 1};
    Output<Buffer<float>> out{"out", 1};

    void generate() {
        Var x{"x"};
        out(x) = hstack1(input_a, input_b, input_a.dim(0).extent(), x);
    }
};

class HStack2Generator : public Halide::Generator<HStack2Generator> {
public:
    Input<Buffer<float>> input_a {"input_a", 2};
    Input<Buffer<float>> input_b {"input_b", 2};
    Output<Buffer<float>> out{"out", 2};

    void generate() {
        Var x{"x"}, y{"y"};
        out(x, y) = hstack2(input_a, input_b, input_a.dim(0).extent(), x, y);
    }
};


class VStack1Generator : public Halide::Generator<VStack1Generator> {
public:
    Input<Buffer<float>> input_a {"input_a", 1};
    Input<Buffer<float>> input_b {"input_b", 1};
    Output<Buffer<float>> out{"out", 2};

    void generate() {
        Var x{"x"}, y{"y"};
        out(x, y) = vstack1(input_a, input_b, input_a.dim(0).extent(), x, y);
    }
};

class VStack2Generator : public Halide::Generator<VStack2Generator> {
public:
    Input<Buffer<float>> input_a {"input_a", 2};
    Input<Buffer<float>> input_b {"input_b", 2};
    Output<Buffer<float>> out{"out", 2};

    void generate() {
        Var x{"x"}, y{"y"};
        out(x, y) = vstack2(input_a, input_b, input_a.dim(1).extent(), x, y);
    }
};

HALIDE_REGISTER_GENERATOR(HStack1Generator, hstack1)
HALIDE_REGISTER_GENERATOR(HStack2Generator, hstack2)
HALIDE_REGISTER_GENERATOR(VStack1Generator, vstack1)
HALIDE_REGISTER_GENERATOR(VStack2Generator, vstack2)
