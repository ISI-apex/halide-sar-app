#include <stdio.h>

#include <Halide.h>

#include "test.h"
#include "norm1d.h"
#include "norm2d.h"

using namespace std;
using Halide::Runtime::Buffer;

#define X_LEN 2
#define Y_LEN 2

int test_1d(void) {
    float in[X_LEN] = {3, 4};
    Buffer<float, 1> in_buf(in, X_LEN);
    Buffer<float> out_buf = Buffer<float>::make_scalar();
    int rv = norm1d(in_buf, out_buf);
    if (!rv) {
        // should be 5
        cout << "norm1d:" << endl << out_buf() << endl;
        if (out_buf.number_of_elements() != 1 || out_buf() != 5) {
            cerr << "Verification failed" << endl;
            return -1;
        }
    }
    return rv;
}

int test_2d(void) {
    float in[X_LEN * Y_LEN] = {3, 4, 5, 12};
    Buffer<float, 2> in_buf(in, X_LEN, Y_LEN);
    Buffer<float, 1> out_buf(Y_LEN);
    int rv = norm2d(in_buf, out_buf);
    if (!rv) {
        // should be [ 5, 13 ]
        cout << "norm2d:" << endl;
        print_1d(out_buf);
        if (out_buf.number_of_elements() != 2 || out_buf(0) != 5 || out_buf(1) != 13) {
            cerr << "Verification failed" << endl;
            return -1;
        }
    }
    return rv;
}

int main(int argc, char **argv) {
    int rv = test_1d();
    if (!rv) {
        rv = test_2d();
    }
    return rv;
}
