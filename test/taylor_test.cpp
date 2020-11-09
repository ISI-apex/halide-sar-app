#include <stdio.h>

#include <Halide.h>

#include "test.h"
#include "taylor.h"

using namespace std;
using Halide::Runtime::Buffer;

#define INPUT_NUM 12

// TODO: need an actual reference function implementation
static const double reference[INPUT_NUM] = {
    0.632594, 0.662359, 0.742204, 0.846779, 0.942881, 1,
    1, 0.942881, 0.846779, 0.742204, 0.662359, 0.632594
};

int main(int argc, char **argv) {
    Buffer<double, 1> out(INPUT_NUM);
    int rv = taylor(INPUT_NUM, 17, out);

    if (!rv) {
        print_1d(out);
        double *obuf = out.begin();
        for (size_t i = 0; i < INPUT_NUM; i++) {
            if (abs(reference[i] - obuf[i]) >= 0.0001f) {
                cerr << "Verification failed at index " << i << endl;
                return -1;
            }
        }
    }

    return rv;
}
