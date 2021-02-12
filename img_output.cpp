#include <Halide.h>

#include "halide_complexfunc.h"
#include "signal.h"
#include "util.h"

using namespace Halide;
using namespace Halide::Tools;

class ImgOutputToDBGenerator : public Halide::Generator<ImgOutputToDBGenerator> {
public:
    GeneratorParam<int32_t> vectorsize {"vectorsize", 4};
    GeneratorParam<int32_t> blocksize {"blocksize", 64};

    Input<Buffer<double>> img {"img", 3}; // complex input
    Output<Buffer<double>> out{"out", 2};

    void generate() {
        Func img_func("cimg");
        img_func = img;
        ComplexFunc cimg(c, img_func);

        RDom r(0, img.dim(1).extent(), 0, img.dim(2).extent(), "r");
        m() = maximum(abs(cimg(r.x, r.y)));

        out(x, y) = Expr(10) * log10(abs(cimg(x, y)) / m());
    }

    void schedule() {
        m.compute_root();
        out.compute_root()
           .vectorize(x, vectorsize)
           .parallel(y, blocksize);
    }

private:
    Var x{"x"}, y{"y"}, c{"c"};
    Func m{"m"};
};

class ImgOutputU8Generator : public Halide::Generator<ImgOutputU8Generator> {
public:
    Input<Buffer<double>> dB {"dB", 2};
    Input<double> dB_min {"dB_min"};
    Input<double> dB_max {"dB_max"};
    Output<Buffer<uint8_t>> out{"out", 2};

    void generate() {
        Var x{"x"}, y{"y"};
        out(x, y) = ConciseCasts::u8(dB_scale(dB(x, y), dB_min, dB_max, UInt(8)));
    }
};

HALIDE_REGISTER_GENERATOR(ImgOutputToDBGenerator, img_output_to_dB)
HALIDE_REGISTER_GENERATOR(ImgOutputU8Generator, img_output_u8)
