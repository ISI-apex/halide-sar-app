#include <stdio.h>

#include <Halide.h>

#include "cross.h"
#include "test.h"

using namespace std;
using Halide::Runtime::Buffer;

#define N 3

int main(int argc, char **argv) {
    float a[N] = {2.0f, 3.0f, 4.0f};
    float b[N] = {5.0f, 6.0f, 7.0f};

    Buffer<float, 1> in_a(a, N);
    Buffer<float, 1> in_b(b, N);
    Buffer<float, 1> out(N);
    int rv = cross(in_a, in_b, out);

    // output should be: [ -3, 6, -3 ]
    if (!rv) {
        print_1d(out);
        float *obuf = out.begin();
        if (abs(obuf[0] + 3) >= 0.01f ||
            abs(obuf[1] - 6) >= 0.01f ||
            abs(obuf[2] + 3) >= 0.01f) {
            cerr << "Verification failed" << endl;
            return -1;
        }
    }

    return rv;
}
