#include <complex>
#include <cstring>

#include <cnpy.h>
#include <Halide.h>

#include "PlatformData.h"

using namespace std;
using namespace cnpy;
using Halide::Runtime::Buffer;

static bool file_exists(const string& path) {
    if (FILE *f = fopen(path.c_str(), "rb")) {
        fclose(f);
        return true;
    }
    return false;
}

PlatformData platform_load(string platform_dir) {
    // Load primitives

    optional<double> B = nullopt;
    if (file_exists(platform_dir + "/B.npy")) {
        NpyArray npy_B = npy_load(platform_dir + "/B.npy");
        if (npy_B.word_size != sizeof(double)) {
            throw runtime_error("Bad word size: B");
        }
        B.emplace(*npy_B.data<double>());
    }

    NpyArray npy_B_IF = npy_load(platform_dir + "/B_IF.npy");
    float B_IF;
    if (npy_B_IF.word_size == sizeof(float)) {
        B_IF = *npy_B_IF.data<float>();
    } else if (npy_B_IF.word_size == sizeof(double)) {
        cout << "PlatformData: downcasting B_IF from double to float" << endl;
        B_IF = (float)*npy_B_IF.data<double>();
    } else {
        throw runtime_error("Bad word size: B_IF");
    }

    NpyArray npy_delta_r = npy_load(platform_dir + "/delta_r.npy");
    if (npy_delta_r.word_size != sizeof(double)) {
        throw runtime_error("Bad word size: delta_r");
    }
    double delta_r = *npy_delta_r.data<double>();

    optional<double> delta_t = nullopt;
    if (file_exists(platform_dir + "/delta_t.npy")) {
        NpyArray npy_delta_t = npy_load(platform_dir + "/delta_t.npy");
        if (npy_delta_t.word_size != sizeof(double)) {
            throw runtime_error("Bad word size: delta_t");
        }
        delta_t.emplace(*npy_delta_t.data<double>());
    }

    NpyArray npy_chirprate = npy_load(platform_dir + "/chirprate.npy");
    if (npy_chirprate.word_size != sizeof(double)) {
        throw runtime_error("Bad word size: chirprate");
    }
    double chirprate = *npy_chirprate.data<double>();

    NpyArray npy_f_0 = npy_load(platform_dir + "/f_0.npy");
    if (npy_f_0.word_size != sizeof(double)) {
        throw runtime_error("Bad word size: f_0");
    }
    double f_0 = *npy_f_0.data<double>();

    NpyArray npy_nsamples = npy_load(platform_dir + "/nsamples.npy");
    int nsamples;
    if (npy_nsamples.word_size == sizeof(int)) {
        nsamples = *npy_nsamples.data<int>();
    } else if (npy_nsamples.word_size == sizeof(int64_t)) {
        // no need to warn of downcasting
        nsamples = (int)*npy_nsamples.data<int64_t>();
    } else {
        throw runtime_error("Bad word size: nsamples");
    }

    NpyArray npy_npulses = npy_load(platform_dir + "/npulses.npy");
    int npulses;
    if (npy_npulses.word_size == sizeof(int)) {
        npulses = *npy_npulses.data<int>();
    } else if (npy_npulses.word_size == sizeof(int64_t)) {
        // no need to warn of downcasting
        npulses = (int)*npy_npulses.data<int64_t>();
    } else {
        throw runtime_error("Bad word size: npulses");
    }

    optional<double> vp = nullopt;
    if (file_exists(platform_dir + "/vp.npy")) {
        NpyArray npy_vp = npy_load(platform_dir + "/vp.npy");
        if (npy_vp.word_size != sizeof(double)) {
            throw runtime_error("Bad word size: vp");
        }
        vp.emplace(*npy_vp.data<double>());
    }

    // Load arrays

    optional<Buffer<float, 1>> freq = nullopt;
    if (file_exists(platform_dir + "/freq.npy")) {
        NpyArray npy_freq = npy_load(platform_dir + "/freq.npy");
        if (npy_freq.shape.size() != 1 || npy_freq.shape[0] != nsamples) {
            throw runtime_error("Bad shape: freq");
        }
        freq.emplace(Buffer<float, 1>(nsamples));
        if (npy_freq.word_size == sizeof(float)) {
            memcpy(freq.value().begin(), npy_freq.data<float>(), npy_freq.num_bytes());
        } else if (npy_freq.word_size == sizeof(double)) {
            cout << "PlatformData: downcasting freq from double to float" << endl;
            const double *src = npy_freq.data<double>();
            float *dest = freq.value().begin();
            for (size_t i = 0; i < npy_freq.num_vals; i++) {
                dest[i] = (float)src[i];
            }
        } else {
            throw runtime_error("Bad word size: freq");
        }
    }

    NpyArray npy_k_r = npy_load(platform_dir + "/k_r.npy");
    if (npy_k_r.shape.size() != 1 || npy_k_r.shape[0] != nsamples) {
        throw runtime_error("Bad shape: k_r");
    }
    Buffer<float, 1> k_r(nsamples);
    if (npy_k_r.word_size == sizeof(float)) {
        memcpy(k_r.begin(), npy_k_r.data<float>(), npy_k_r.num_bytes());
    } else if (npy_k_r.word_size == sizeof(double)) {
        cout << "PlatformData: downcasting k_r from double to float" << endl;
        const double *src = npy_k_r.data<double>();
        float *dest = k_r.begin();
        for (size_t i = 0; i < npy_k_r.num_vals; i++) {
            dest[i] = (float)src[i];
        }
    } else {
        throw runtime_error("Bad word size: k_r");
    }

    optional<Buffer<double, 1>> k_y = nullopt;
    if (file_exists(platform_dir + "/k_y.npy")) {
        NpyArray npy_k_y = npy_load(platform_dir + "/k_y.npy");
        if (npy_k_y.shape.size() != 1 || npy_k_y.shape[0] != npulses) {
            throw runtime_error("Bad shape: k_y");
        }
        if (npy_k_y.word_size != sizeof(double)) {
            throw runtime_error("Bad word size: k_y");
        }
        k_y.emplace(Buffer<double, 1>(npulses));
        memcpy(k_y.value().begin(), npy_k_y.data<double>(), npy_k_y.num_bytes());
    }

    optional<Buffer<float, 1>> n_hat = nullopt;
    if (file_exists(platform_dir + "/n_hat.npy")) {
        NpyArray npy_n_hat = npy_load(platform_dir + "/n_hat.npy");
        if (npy_n_hat.shape.size() != 1 || npy_n_hat.shape[0] != 3) {
            throw runtime_error("Bad shape: n_hat");
        }
        if (npy_n_hat.word_size != sizeof(float)) {
            throw runtime_error("Bad word size: n_hat");
        }
        n_hat.emplace(Buffer<float, 1>(3));
        memcpy(n_hat.value().begin(), npy_n_hat.data<float>(), npy_n_hat.num_bytes());
    }

    NpyArray npy_R_c = npy_load(platform_dir + "/R_c.npy");
    if (npy_R_c.shape.size() != 1 || npy_R_c.shape[0] != 3) {
        throw runtime_error("Bad shape: R_c");
    }
    Buffer<float, 1> R_c(3);
    if (npy_R_c.word_size == sizeof(float)) {
        memcpy(R_c.begin(), npy_R_c.data<float>(), npy_R_c.num_bytes());
    } else if (npy_R_c.word_size == sizeof(double)) {
        cout << "PlatformData: downcasting R_c from double to float" << endl;
        const double *src = npy_R_c.data<double>();
        float *dest = R_c.begin();
        for (size_t i = 0; i < npy_R_c.num_vals; i++) {
            dest[i] = (float)src[i];
        }
    } else {
        throw runtime_error("Bad word size: R_c");
    }

    NpyArray npy_t = npy_load(platform_dir + "/t.npy");
    if (npy_t.shape.size() != 1 || npy_t.shape[0] != nsamples) {
        throw runtime_error("Bad shape: t");
    }
    if (npy_t.word_size != sizeof(double)) {
        throw runtime_error("Bad word size: t");
    }
    Buffer<double, 1> t(nsamples);
    memcpy(t.begin(), npy_t.data<double>(), npy_t.num_bytes());

    // Load matrices

    NpyArray npy_pos = npy_load(platform_dir + "/pos.npy");
    if (npy_pos.shape.size() != 2 || npy_pos.shape[0] != npulses || npy_pos.shape[1] != 3) {
        throw runtime_error("Bad shape: pos");
    }
    Buffer<float, 2> pos(3, npulses);
    if (npy_pos.word_size == sizeof(float)) {
        memcpy(pos.begin(), npy_pos.data<float>(), npy_pos.num_bytes());
    } else if (npy_pos.word_size == sizeof(double)) { 
        cout << "PlatformData: downcasting pos from double to float" << endl;
        const double *src = npy_pos.data<double>();
        float *dest = pos.begin();
        for (size_t i = 0; i < npy_pos.num_vals; i++) {
            dest[i] = (float)src[i];
        }
    } else {
        throw runtime_error("Bad word size: pos");
    }

    NpyArray npy_phs = npy_load(platform_dir + "/phs.npy");
    if (npy_phs.shape.size() != 2 || npy_phs.shape[0] != npulses || npy_phs.shape[1] != nsamples) {
        throw runtime_error("Bad shape: phs");
    }
    Buffer<float, 3> phs(2, nsamples, npulses);
    if (npy_phs.word_size == sizeof(complex<float>)) {
        memcpy(phs.begin(), reinterpret_cast<float *>(npy_phs.data<complex<float>>()), npy_phs.num_bytes());
    } else if (npy_phs.word_size == sizeof(complex<double>)) {
        cout << "PlatformData: downcasting phs from complex<double> to complex<float>" << endl;
        const complex<double> *src = npy_phs.data<complex<double>>();
        float *dest = phs.begin();
        for (size_t i = 0; i < npy_phs.num_vals; i++) {
            dest[i * 2] = (float)src[i].real();
            dest[(i * 2) + 1] = (float)src[i].imag();
        }
    } else {
        throw runtime_error("Bad word size: phs");
    }

    return PlatformData(B, B_IF, delta_r, delta_t, chirprate, f_0, nsamples, npulses, vp,
                        freq, k_r, k_y, n_hat, R_c, t, pos, phs);
}
