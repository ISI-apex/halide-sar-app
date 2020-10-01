#include <stdio.h>

#include <Halide.h>

#include "linspace.h"

using namespace std;
using Halide::Runtime::Buffer;

#define INPUT_NUM 424

int main(int argc, char **argv) {
    Buffer<float> out(INPUT_NUM);
    int rv = linspace(-0.5f, 0.5f, INPUT_NUM, out);

    if (!rv) {
        float *obuf = out.begin();
        cout << "[ " << obuf[0];
        for (size_t i = 1; i < INPUT_NUM; i++) {
            cout << ", " << obuf[i];
        }
        cout << " ]" << endl;
    }

    return rv;
}
