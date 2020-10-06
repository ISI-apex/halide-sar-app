#include <complex>
#include <stdio.h>

#include <Halide.h>
#include <cnpy.h>
#include <fftw3.h>

#include "backprojection.h"

using namespace std;
using Halide::Runtime::Buffer;

// TODO: Are all these memcpy necessary?

#define UPSAMPLE 2

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " platform_dir img_plane_dir" << endl;
        return 1;
    }
    string platform_dir = string(argv[1]);
    string img_plane_dir = string(argv[2]);

    // Load primitives
    // B_IF: <class 'numpy.float32'>
    // chirprate: <class 'numpy.float64'>
    // delta_r: <class 'numpy.float64'>
    // f_0: <class 'numpy.float64'>
    // nsamples: <class 'int'>
    // npulses: <class 'int'>

    cnpy::NpyArray npydata;
    float *ptr_flt;
    double *ptr_dbl;
    int *ptr_int;

    npydata = cnpy::npy_load(platform_dir + "/B_IF.npy");
    ptr_flt = npydata.data<float>();
    // int B_IF = 622360576;
    float B_IF = *ptr_flt;
    printf("B_IF: %f\n", B_IF);

    npydata = cnpy::npy_load(platform_dir + "/delta_r.npy");
    ptr_dbl = npydata.data<double>();
    // double delta_r = 0.24085;
    double delta_r = *ptr_dbl;
    printf("delta_r: %lf\n", delta_r);

    npydata = cnpy::npy_load(platform_dir + "/chirprate.npy");
    ptr_dbl = npydata.data<double>();
    // double chirprate = 913520151990310.75;
    double chirprate = *ptr_dbl;
    printf("chirprate: %lf\n", chirprate);

    npydata = cnpy::npy_load(platform_dir + "/f_0.npy");
    ptr_dbl = npydata.data<double>();
    // uint64_t f_0 = 9599260672;
    double f_0 = *ptr_dbl;
    printf("f_0: %lf\n", f_0);

    npydata = cnpy::npy_load(platform_dir + "/nsamples.npy");
    ptr_int = npydata.data<int>();
    // int nsamples = 424;
    int nsamples = *ptr_int;
    printf("nsamples: %d\n", nsamples);

    npydata = cnpy::npy_load(platform_dir + "/npulses.npy");
    ptr_int = npydata.data<int>();
    // int npulses = 469;
    int npulses = *ptr_int;
    printf("npulses: %d\n", npulses);

    // Load arrays
    // freq: <class 'numpy.ndarray'>
    // freq[0]: <class 'numpy.float32'>
    // k_r: <class 'numpy.ndarray'>
    // k_r[0]: <class 'numpy.float32'>
    // k_y: <class 'numpy.ndarray'>
    // k_y[0]: <class 'numpy.float64'>
    // R_c: <class 'numpy.ndarray'>
    // R_c[0]: <class 'numpy.float32'>
    // t: <class 'numpy.ndarray'>
    // t[0]: <class 'numpy.float64'>

    npydata = cnpy::npy_load(platform_dir + "/freq.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != nsamples) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    float* freq = static_cast<float *>(malloc(nsamples * sizeof(float)));
    memcpy(freq, npydata.data<float>(), nsamples * sizeof(float));

    npydata = cnpy::npy_load(platform_dir + "/k_r.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != nsamples) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    float *k_r = static_cast<float *>(malloc(nsamples * sizeof(float)));
    memcpy(k_r, npydata.data<float>(), nsamples * sizeof(float));

    npydata = cnpy::npy_load(platform_dir + "/k_y.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != npulses) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double *k_y = static_cast<double *>(malloc(npulses * sizeof(double)));
    memcpy(k_y, npydata.data<double>(), npulses * sizeof(double));

    npydata = cnpy::npy_load(platform_dir + "/R_c.npy");
    // TODO: 3 is magic number?
    if (npydata.shape.size() != 1 || npydata.shape[0] != 3) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    float *R_c = static_cast<float *>(malloc(3 * sizeof(float)));
    memcpy(R_c, npydata.data<float>(), 3 * sizeof(float));

    npydata = cnpy::npy_load(platform_dir + "/t.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != nsamples) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double *t = static_cast<double *>(malloc(nsamples * sizeof(double)));
    memcpy(t, npydata.data<double>(), nsamples * sizeof(double));

    // Load matrices
    // pos: <class 'numpy.ndarray'>
    // pos[0]: <class 'numpy.ndarray'>
    // pos[0][0]: <class 'numpy.float32'>
    // phs: <class 'numpy.ndarray'>
    // phs[0]: <class 'numpy.ndarray'>
    // phs[0][0]: <class 'numpy.complex64'>
    
    npydata = cnpy::npy_load(platform_dir + "/pos.npy");
    // TODO: 3 is magic number?
    if (npydata.shape.size() != 2 || npydata.shape[0] != npulses || npydata.shape[1] != 3) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    float *pos = static_cast<float *>(malloc(npulses * 3 * sizeof(float)));
    memcpy(pos, npydata.data<float>(), npulses * 3 * sizeof(float));

    npydata = cnpy::npy_load(platform_dir + "/phs.npy");
    if (npydata.shape.size() != 2 || npydata.shape[0] != npulses || npydata.shape[1] != nsamples) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    complex<float> *phs = static_cast<complex<float> *>(malloc(npulses * nsamples * sizeof(complex<float>)));
    memcpy(phs, npydata.data<complex<float>>(), npulses * nsamples * sizeof(complex<float>));

    // Load primitives
    // du: <class 'numpy.float64'>
    // dv: <class 'numpy.float64'>

    npydata = cnpy::npy_load(img_plane_dir + "/du.npy");
    ptr_dbl = npydata.data<double>();
    // double delta_r = 0.24085;
    double d_u = *ptr_dbl;
    printf("d_u: %lf\n", d_u);

    npydata = cnpy::npy_load(img_plane_dir + "/dv.npy");
    ptr_dbl = npydata.data<double>();
    // double delta_r = 0.24085;
    double d_v = *ptr_dbl;
    printf("d_v: %lf\n", d_v);

    // Load arrays
    // k_u: <class 'numpy.ndarray'>
    // k_u[0]: <class 'numpy.float64'>
    // k_v: <class 'numpy.ndarray'>
    // k_v[0]: <class 'numpy.float64'>
    // n_hat: <class 'numpy.ndarray'>
    // n_hat[0]: <class 'numpy.int64'>
    // u: <class 'numpy.ndarray'>
    // u[0]: <class 'numpy.float64'>
    // u_hat: <class 'numpy.ndarray'>
    // u_hat[0]: <class 'numpy.float64'>
    // v: <class 'numpy.ndarray'>
    // v[0]: <class 'numpy.float64'>
    // v_hat: <class 'numpy.ndarray'>
    // v_hat[0]: <class 'numpy.float64'>

    // TODO: 512 is magic number?
    npydata = cnpy::npy_load(img_plane_dir + "/k_u.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != 512) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double* k_u = static_cast<double *>(malloc(512 * sizeof(double)));
    memcpy(k_u, npydata.data<double>(), 512 * sizeof(double));

    npydata = cnpy::npy_load(img_plane_dir + "/k_v.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != 512) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double* k_v = static_cast<double *>(malloc(512 * sizeof(double)));
    memcpy(k_v, npydata.data<double>(), 512 * sizeof(double));

    npydata = cnpy::npy_load(img_plane_dir + "/n_hat.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != 3) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    int64_t* n_hat = static_cast<int64_t *>(malloc(3 * sizeof(int64_t)));
    memcpy(n_hat, npydata.data<int64_t>(), 3 * sizeof(int64_t));

    npydata = cnpy::npy_load(img_plane_dir + "/u.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != 512) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double* u = static_cast<double *>(malloc(512 * sizeof(double)));
    memcpy(u, npydata.data<double>(), 512 * sizeof(double));

    npydata = cnpy::npy_load(img_plane_dir + "/u_hat.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != 3) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double* u_hat = static_cast<double *>(malloc(3 * sizeof(double)));
    memcpy(u_hat, npydata.data<double>(), 3 * sizeof(double));

    npydata = cnpy::npy_load(img_plane_dir + "/v.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != 512) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double* v = static_cast<double *>(malloc(512 * sizeof(double)));
    memcpy(v, npydata.data<double>(), 512 * sizeof(double));

    npydata = cnpy::npy_load(img_plane_dir + "/v_hat.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != 3) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double* v_hat = static_cast<double *>(malloc(3 * sizeof(double)));
    memcpy(v_hat, npydata.data<double>(), 3 * sizeof(double));

    // Now do something with the data
    // pixel_locs: <class 'numpy.ndarray'>
    // pixel_locs[0]: <class 'numpy.ndarray'>
    // pixel_locs[0][0]: <class 'numpy.float64'>

    npydata = cnpy::npy_load(img_plane_dir + "/pixel_locs.npy");
    if (npydata.shape.size() != 2 || npydata.shape[0] != 3 || npydata.shape[1] != 512*512) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double* pixel_locs = static_cast<double *>(malloc(3 * 512*512 * sizeof(double)));
    memcpy(pixel_locs, npydata.data<double>(), 3 * 512*512 * sizeof(double));

    int N_fft = static_cast<int>(pow(2, static_cast<int>(log2(nsamples * UPSAMPLE)) + 1));

    // Copy input data
    Buffer<float, 3> fftshift_buf(2, N_fft, npulses);
    Buffer<float, 3> inbuf(2, nsamples, npulses);
    cout << "Width: " << inbuf.width() << endl;
    cout << "Height: " << inbuf.height() << endl;
    cout << "Channels: " << inbuf.channels() << endl;
    complex<float> *indata = (complex<float> *)inbuf.begin();
    for (int x = 0; x < npulses; x++) {
        for (int y = 0; y < nsamples; y++) {
            indata[x * nsamples + y] = phs[x * nsamples + y];
        }
    }
    Buffer<float, 1> in_k_r(k_r, nsamples);

    // backprojection
    Buffer<float, 3> outbuf(2, N_fft, npulses); // TODO: This is changing as we develop
    int rv = backprojection(inbuf, in_k_r, N_fft, fftshift_buf);
    printf("Halide kernel returned %d\n", rv);
    // FFT
    fftwf_complex *fft_in = reinterpret_cast<fftwf_complex *>(fftshift_buf.begin());
    fftwf_complex *fft_out = reinterpret_cast<fftwf_complex *>(outbuf.begin());
    fftwf_plan plan = fftwf_plan_dft_1d(N_fft, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int i = 0; i < npulses; i++) {
        fftwf_execute_dft(plan, &fft_in[i * N_fft], &fft_out[i * N_fft]);
    }
    fftwf_destroy_plan(plan);

    // write output
    vector<size_t> shape_out { static_cast<size_t>(outbuf.dim(2).extent()),
                               static_cast<size_t>(outbuf.dim(1).extent()) };
    cnpy::npy_save("sarbp_test.npy", (complex<float> *)outbuf.begin(), shape_out);

    return rv;
}
