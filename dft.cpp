#include "dft.h"

#include <Halide.h>
#include <fftw3.h>

#include <cassert>
#include <iostream>

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

using namespace std;

// FFTW plan initialization can have high overhead, so share and reuse it
static fftw_plan fft_plan = nullptr;

void dft_init_fftw(size_t N_fft) {
    assert(N_fft > 0);
    fftw_complex *fft_in = fftw_alloc_complex(N_fft);
    assert(fft_in);
    fftw_complex *fft_out = fftw_alloc_complex(N_fft);
    assert(fft_out);
    fft_plan = fftw_plan_dft_1d(N_fft, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    assert(fft_plan);
    fftw_free(fft_in);
    fftw_free(fft_out);
}

void dft_destroy_fftw(void) {
    fftw_destroy_plan(fft_plan);
    fft_plan = nullptr;
}

#ifdef NO_MANGLE
extern "C" DLLEXPORT
#endif
int call_dft(halide_buffer_t *in, int N_fft, halide_buffer_t *out) {
    // input and output are complex data
    assert(in->dimensions == 3);
    assert(out->dimensions == 3);
    assert(out->dim[0].min == 0);
    assert(out->dim[0].extent == 2);
    // FFTW needs entire rows of complex data
    assert(out->dim[1].min == 0);
    assert(out->dim[1].extent == N_fft);
    if(in->is_bounds_query()) {
#if 0
        cout << "call_dft: bounds query" << endl;
#endif
        // input and output shapes are the same
        for (int i = 0; i < in->dimensions; i++) {
            in->dim[i].min = out->dim[i].min;
            in->dim[i].extent = out->dim[i].extent;
        }
    } else {
        assert(in->host);
        assert(in->type == halide_type_of<double>());
        assert(out->host);
        assert(fft_plan);
#if 0
        cout << "call_dft: FFTW: processing vectors ["
             << out->dim[2].min << ", " << out->dim[2].min + out->dim[2].extent << ")" << endl;
#endif
        int upper_bound = min(in->dim[2].min + in->dim[2].extent, out->dim[2].min + out->dim[2].extent);
        for (int i = out->dim[2].min; i < upper_bound; i++) {
            int coords[3] = {0, 0, i};
            fftw_complex *fft_in = (fftw_complex *)in->address_of(coords);
            fftw_complex *fft_out = (fftw_complex *)out->address_of(coords);
            fftw_execute_dft(fft_plan, fft_in, fft_out);
        }
#if 0
        cout << "call_dft: FFTW: finished" << endl;
#endif
    }
    return 0;
}
