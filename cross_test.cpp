#include <stdio.h>

#include <Halide.h>

#include "cross.h"

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

    // output should be: [ -3, 6, 3 ]
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
