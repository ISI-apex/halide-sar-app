#ifndef TEST_H
#define TEST_H

#include <stdio.h>

#include <Halide.h>

using namespace std;
using Halide::Runtime::Buffer;

template <class T>
inline void print_1d(Buffer<T, 1> out_buf) {
    T *obuf = out_buf.begin();
    cout << "[ " << obuf[0];
    for (size_t i = 1; i < out_buf.dim(0).extent(); i++) {
        cout << ", " << obuf[i];
    }
    cout << " ]" << endl;
}

template <class T>
inline void print_2d(Buffer<T, 2> out_buf) {
    T *obuf = out_buf.begin();
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

#endif
