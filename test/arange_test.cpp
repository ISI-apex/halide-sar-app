#include <stdio.h>
#include <math.h>

#include <Halide.h>

#include "arange.h"
#include "test.h"

using namespace std;
using Halide::Runtime::Buffer;

#define START 0.0f
#define STOP 15.0f
#define STEP 1.5f

#define N ceil((((STOP - START) / STEP)))

static void reference(float *ref) {
    for (size_t i = 0; i < N; i++) {
        ref[i] = START + i * STEP;
    }
}

int main(int argc, char **argv) {
    float ref[(size_t)N] = { 0 };
    Buffer<float, 1> out((int)N);
    int rv = arange(START, STOP, STEP, out);

    // output should be: [ 0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5 ]
    if (!rv) {
        print_1d(out);
        reference(ref);
        float *obuf = out.begin();
        for (size_t i = 0; i < N; i++) {
            if (abs(ref[i] - obuf[i]) >= 0.1f) {
                cerr << "Verification failed at index " << i << endl;
                return -1;
            }
        }
    }

    return rv;
}
