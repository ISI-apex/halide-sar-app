#include <stdio.h>

#include <Halide.h>

#include "interp.h"
#include "test.h"

using namespace std;
using Halide::Runtime::Buffer;

#define XS_LEN 11
#define P_LEN 3

int main(int argc, char **argv) {
    float xs[XS_LEN] = {-0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25};
    float xp[P_LEN] = {0.0, 1.0, 2.0};
    float fp[P_LEN] = {0.0, 10.0, 30.0};

    Buffer<float, 1> in_xs(xs, XS_LEN);
    Buffer<float, 1> in_xp(xp, P_LEN);
    Buffer<float, 1> in_fp(fp, P_LEN);
    Buffer<float, 1> out(XS_LEN);
    int rv = interp(in_xs, in_xp, in_fp, out);

    // output should be: [ 0, 0, 2.5, 5, 7.5, 10, 15, 20, 25, 30, 30 ]
    if (!rv) {
        print_1d(out);
        float *obuf = out.begin();
        for (size_t i = 1; i < out.number_of_elements() - 2; i++) {
            float exp;
            if (i == 0) {
                exp = 0.0;
            } else if (i < XS_LEN / 2) {
                exp = (i - 1) * 2.5;
            } else if (i < XS_LEN - 1) {
                exp = (i - 3) * 5.0;
            } else {
                exp = 30.0;
            }
            if (abs(exp - obuf[i]) >= 0.01f) {
                cerr << "Verification failed at index " << i << endl;
                return -1;
            }
        }
    }

    return rv;
}
