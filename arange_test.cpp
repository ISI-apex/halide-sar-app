#include <stdio.h>
#include <math.h>

#include <Halide.h>

#include "arange.h"

using namespace std;
using Halide::Runtime::Buffer;

#define START 0.0f
#define STOP 15.0f
#define STEP 1.5f

#define N ceil((((STOP - START) / STEP)))

int main(int argc, char **argv) {
    Buffer<float> out((int)N);
    int rv = arange(START, STOP, STEP, out);

    // output should be: [ 0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5, 12, 13.5 ]
    if (!rv) {
        float *obuf = out.begin();
        cout << "[ " << obuf[0];
        for (size_t i = 1; i < (size_t)N; i++) {
            cout << ", " << obuf[i];
        }
        cout << " ]" << endl;
    }

    return rv;
}
