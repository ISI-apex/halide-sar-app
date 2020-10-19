#include <stdio.h>

#include <Halide.h>

#include "linspace.h"
#include "test.h"

using namespace std;
using Halide::Runtime::Buffer;

#define INPUT_NUM 10
#define START -0.5f
#define STOP 0.5f

static void reference(float *ref) {
    float step = (STOP - START) / (INPUT_NUM - 1);
    for (size_t i = 0; i < INPUT_NUM; i++) {
        ref[i] = START + i * step;
    }
    ref[INPUT_NUM - 1] = STOP;
}

int main(int argc, char **argv) {
    float ref[INPUT_NUM] = { 0 };
    Buffer<float, 1> out(INPUT_NUM);
    int rv = linspace(START, STOP, INPUT_NUM, out);

    if (!rv) {
        print_1d(out);
        reference(ref);
        float *obuf = out.begin();
        for (size_t i = 0; i < INPUT_NUM; i++) {
            if (abs(ref[i] - obuf[i]) >= 0.1f) {
                cerr << "Verification failed at index " << i << endl;
                return -1;
            }
        }
    }

    return rv;
}
