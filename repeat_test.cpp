#include <stdio.h>

#include <Halide.h>

#include "repeat1.h"
#include "test.h"

using namespace std;
using Halide::Runtime::Buffer;

#define INPUT_NUM 10
#define REPEAT 2.5
#define OUTPUT_NUM (size_t)(INPUT_NUM * REPEAT)

static void reference(float *ref) {
    for (size_t i = 0; i < OUTPUT_NUM; i++) {
        ref[i] = i % INPUT_NUM;
    }
}

int main(int argc, char **argv) {
    float in[INPUT_NUM];
    for (size_t i = 0; i < INPUT_NUM; i++) {
        in[i] = i;
    }
    Buffer<float, 1> in_buf(in, INPUT_NUM);
    float ref[OUTPUT_NUM] = { 0 };
    Buffer<float, 1> out(OUTPUT_NUM);
    int rv = repeat1(in_buf, out);

    if (!rv) {
        print_1d(out);
        reference(ref);
        float *obuf = out.begin();
        for (size_t i = 0; i < OUTPUT_NUM; i++) {
            if (abs(ref[i] - obuf[i]) >= 0.1f) {
                cerr << "Verification failed at index " << i << endl;
                return -1;
            }
        }
    }

    return rv;
}
