#include <stdio.h>

#include <Halide.h>

#include "taylor.h"

using namespace std;
using Halide::Runtime::Buffer;

#define INPUT_NUM 424

int main(int argc, char **argv) {
    Buffer<double> out(INPUT_NUM);
    int rv = taylor(INPUT_NUM, 17, out);

    if (!rv) {
        double *obuf = out.begin();
        cout << "[ " << obuf[0];
        for (size_t i = 1; i < INPUT_NUM; i++) {
            cout << ", " << obuf[i];
        }
        cout << " ]" << endl;
    }

    return rv;
}
