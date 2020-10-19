#include <stdio.h>

#include <Halide.h>

#include "hstack1.h"
#include "hstack2.h"
#include "vstack1.h"
#include "vstack2.h"

using namespace std;
using Halide::Runtime::Buffer;

#define LEN_X_1D 4
#define LEN_X_2D 2
#define LEN_Y_2D 2

static void print_1d(Buffer<float, 1> out_buf) {
    float *obuf = out_buf.begin();
    cout << "[ " << obuf[0];
    for (size_t i = 1; i < out_buf.dim(0).extent(); i++) {
        cout << ", " << obuf[i];
    }
    cout << " ]" << endl;
}

static void print_2d(Buffer<float, 2> out_buf) {
    float *obuf = out_buf.begin();
    cout << "[";
    for (size_t i = 0; i < out_buf.dim(1).extent(); i++) {
        if (i > 0) {
            cout << endl << " ";
        }
        cout << "[ ";
        for (size_t j = 0; j < out_buf.dim(0).extent(); j++) {
            if (j > 0) {
                cout << ", ";
            }
            cout << obuf[i * out_buf.dim(0).extent() + j];
        }
        cout << " ]";
    }
    cout << "]" << endl;
}

int test_hstack1(void) {
    float in1[LEN_X_1D] = {1, 2, 3, 4};
    float in2[LEN_X_1D] = {5, 6, 7, 8};
    Buffer<float, 1> in1_buf(in1, LEN_X_1D);
    Buffer<float, 1> in2_buf(in2, LEN_X_1D);
    Buffer<float, 1> out_buf(LEN_X_1D * 2);
    int rv = hstack1(in1_buf, in2_buf, out_buf);
    // should be [ 1, 2, 3, 4, 5, 6, 7, 8 ]
    if (!rv) {
        print_1d(out_buf);
    }
    return rv;
}

int test_hstack2(void) {
    float in1[LEN_X_2D * LEN_Y_2D] = {1, 2, 3, 4};
    float in2[LEN_X_2D * LEN_Y_2D] = {5, 6, 7, 8};
    Buffer<float, 2> in1_buf(in1, LEN_X_2D, LEN_Y_2D);
    Buffer<float, 2> in2_buf(in2, LEN_X_2D, LEN_Y_2D);
    Buffer<float, 2> out_buf(LEN_X_2D * 2, LEN_Y_2D);
    int rv = hstack2(in1_buf, in2_buf, out_buf);
    // should be [[ 1, 2, 5, 6 ]
    //            [ 3, 4, 7, 8 ]]
    if (!rv) {
        print_2d(out_buf);
    }
    return rv;
}

int test_vstack1(void) {
    float in1[LEN_X_1D] = {1, 2, 3, 4};
    float in2[LEN_X_1D] = {5, 6, 7, 8};
    Buffer<float, 1> in1_buf(in1, LEN_X_1D);
    Buffer<float, 1> in2_buf(in2, LEN_X_1D);
    Buffer<float, 2> out_buf(LEN_X_1D, 2);
    int rv = vstack1(in1_buf, in2_buf, out_buf);
    // should be [[ 1, 2, 3, 4 ]
    //            [ 5, 6, 7, 8 ]]
    if (!rv) {
        print_2d(out_buf);
    }
    return rv;
}

int test_vstack2(void) {
    float in1[LEN_X_2D * LEN_Y_2D] = {1, 2, 3, 4};
    float in2[LEN_X_2D * LEN_Y_2D] = {5, 6, 7, 8};
    Buffer<float, 2> in1_buf(in1, LEN_X_2D, LEN_Y_2D);
    Buffer<float, 2> in2_buf(in2, LEN_X_2D, LEN_Y_2D);
    Buffer<float, 2> out_buf(LEN_X_2D, LEN_Y_2D * 2);
    int rv = vstack2(in1_buf, in2_buf, out_buf);
    // should be [[ 1, 2 [
    //            [ 3, 4 ]
    //            [ 5, 6 ]
    //            [ 7, 8 ]]
    if (!rv) {
        print_2d(out_buf);
    }
    return rv;
}

int main(int argc, char **argv) {
    int rv = test_hstack1();
    if (!rv) {
        rv = test_hstack2();
    }
    if (!rv) {
        rv = test_vstack1();
    }
    if (!rv) {
        rv = test_vstack2();
    }
    return rv;
}
