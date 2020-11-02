#include <complex>
#include <cstring>

#include <cnpy.h>
#include <Halide.h>

#include "PlatformData.h"

using namespace std;
using namespace cnpy;
using Halide::Runtime::Buffer;

PlatformData platform_load(string platform_dir) {
    // Load primitives

    NpyArray npy_B_IF = npy_load(platform_dir + "/B_IF.npy");
    float B_IF = *npy_B_IF.data<float>();

    NpyArray npy_delta_r = npy_load(platform_dir + "/delta_r.npy");
    double delta_r = *npy_delta_r.data<double>();

    NpyArray npy_chirprate = npy_load(platform_dir + "/chirprate.npy");
    double chirprate = *npy_chirprate.data<double>();

    NpyArray npy_f_0 = npy_load(platform_dir + "/f_0.npy");
    double f_0 = *npy_f_0.data<double>();

    NpyArray npy_nsamples = npy_load(platform_dir + "/nsamples.npy");
    int nsamples = *npy_nsamples.data<int>();

    NpyArray npy_npulses = npy_load(platform_dir + "/npulses.npy");
    int npulses = *npy_npulses.data<int>();

    // Load arrays

    NpyArray npy_freq = npy_load(platform_dir + "/freq.npy");
    if (npy_freq.shape.size() != 1 || npy_freq.shape[0] != nsamples) {
        throw runtime_error("Bad shape: freq");
    }
    Buffer<float, 1> freq(nsamples);
    memcpy(freq.begin(), npy_freq.data<float>(), npy_freq.num_bytes());

    NpyArray npy_k_r = npy_load(platform_dir + "/k_r.npy");
    if (npy_k_r.shape.size() != 1 || npy_k_r.shape[0] != nsamples) {
        throw runtime_error("Bad shape: k_r");
    }
    Buffer<float, 1> k_r(nsamples);
    memcpy(k_r.begin(), npy_k_r.data<float>(), npy_k_r.num_bytes());

    NpyArray npy_k_y = npy_load(platform_dir + "/k_y.npy");
    if (npy_k_y.shape.size() != 1 || npy_k_y.shape[0] != npulses) {
        throw runtime_error("Bad shape: k_y");
    }
    Buffer<double, 1> k_y(npulses);
    memcpy(k_y.begin(), npy_k_y.data<double>(), npy_k_y.num_bytes());

    NpyArray npy_R_c = npy_load(platform_dir + "/R_c.npy");
    if (npy_R_c.shape.size() != 1 || npy_R_c.shape[0] != 3) {
        throw runtime_error("Bad shape: R_c");
    }
    Buffer<float, 1> R_c(3);
    memcpy(R_c.begin(), npy_R_c.data<float>(), npy_R_c.num_bytes());

    NpyArray npy_t = npy_load(platform_dir + "/t.npy");
    if (npy_t.shape.size() != 1 || npy_t.shape[0] != nsamples) {
        throw runtime_error("Bad shape: t");
    }
    Buffer<double, 1> t(nsamples);
    memcpy(t.begin(), npy_t.data<double>(), npy_t.num_bytes());

    // Load matrices
    
    NpyArray npy_pos = npy_load(platform_dir + "/pos.npy");
    if (npy_pos.shape.size() != 2 || npy_pos.shape[0] != npulses || npy_pos.shape[1] != 3) {
        throw runtime_error("Bad shape: pos");
    }
    Buffer<float, 2> pos(3, npulses);
    memcpy(pos.begin(), npy_pos.data<float>(), npy_pos.num_bytes());

    NpyArray npy_phs = npy_load(platform_dir + "/phs.npy");
    if (npy_phs.shape.size() != 2 || npy_phs.shape[0] != npulses || npy_phs.shape[1] != nsamples) {
        throw runtime_error("Bad shape: phs");
    }
    Buffer<float, 3> phs(2, nsamples, npulses);
    memcpy(phs.begin(), reinterpret_cast<float *>(npy_phs.data<complex<float>>()), npy_phs.num_bytes());

    return PlatformData(B_IF, delta_r, chirprate, f_0, nsamples, npulses,
                        freq, k_r, k_y, R_c, t, pos, phs);
}
