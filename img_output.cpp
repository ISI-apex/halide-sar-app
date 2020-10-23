#include <Halide.h>

#include "signal.h"
#include "util.h"

using namespace Halide;

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

HALIDE_REGISTER_GENERATOR(ImgOutputU8Generator, img_output_u8)
