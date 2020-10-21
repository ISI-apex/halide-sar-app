#include <complex>
#include <stdio.h>

#include <Halide.h>
#include <cnpy.h>
#include <fftw3.h>

#include "img_plane.h"
#include "ip_uv.h"
#include "ip_k.h"
#include "ip_v_hat.h"
#include "ip_u_hat.h"
#include "ip_pixel_locs.h"
#include "backprojection_pre_fft.h"
#include "backprojection_post_fft.h"

using namespace std;
using Halide::Runtime::Buffer;

#define DEBUG_Q 0
#define DEBUG_NORM_R0 0
#define DEBUG_RR0 0
#define DEBUG_NORM_RR0 0
#define DEBUG_DR_I 0
#define DEBUG_Q_REAL 0
#define DEBUG_Q_IMAG 0
#define DEBUG_Q_HAT 0
#define DEBUG_IMG 0
#define DEBUG_FIMG 0

// TODO: Are all these memcpy necessary?

#define UPSAMPLE 2
#define RES_FACTOR 1.0
#define ASPECT 1.0

int main(int argc, char **argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " platform_dir" << endl;
        return 1;
    }
    string platform_dir = string(argv[1]);

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
    Buffer<float, 1> in_k_r(k_r, nsamples);

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
    Buffer<float, 1> buf_R_c(R_c, 3);

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
    Buffer<float, 2> in_pos(pos, 3, npulses);

    npydata = cnpy::npy_load(platform_dir + "/phs.npy");
    if (npydata.shape.size() != 2 || npydata.shape[0] != npulses || npydata.shape[1] != nsamples) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    complex<float> *phs = static_cast<complex<float> *>(malloc(npulses * nsamples * sizeof(complex<float>)));
    memcpy(phs, npydata.data<complex<float>>(), npulses * nsamples * sizeof(complex<float>));

    // Img plane variables

    int nu = ip_upsample(nsamples);
    cout << "nu: " << nu << endl;
    int nv = ip_upsample(npulses);
    cout << "nv: " << nv << endl;

    double d_u = ip_du(delta_r, RES_FACTOR, nsamples, nu);
    cout << "d_u: " << d_u << endl;
    double d_v = ip_dv(ASPECT, d_u);
    cout << "d_v: " << d_v << endl;

    Buffer<double, 1> in_u(nu);
    ip_uv(nu, d_u, in_u);

    Buffer<double, 1> in_v(nv);
    ip_uv(nv, d_v, in_v);

    Buffer<double, 1> buf_k_u(nu);
    ip_k(nu, d_u, buf_k_u);

    Buffer<double, 1> buf_k_v(nv);
    ip_k(nv, d_v, buf_k_v);

    Buffer<const int, 1> buf_n_hat(&N_HAT[0], 3);

    Buffer<double, 1> buf_v_hat(3);
    ip_v_hat(buf_n_hat, buf_R_c, buf_v_hat);

    Buffer<double, 1> buf_u_hat(3);
    ip_u_hat(buf_v_hat, buf_n_hat, buf_u_hat);

    Buffer<double, 2> in_pixel_locs(nu*nv, 3);
    ip_pixel_locs(in_u, in_v, buf_u_hat, buf_v_hat, in_pixel_locs);

    // Compute FFT width (power of 2)
    int N_fft = static_cast<int>(pow(2, static_cast<int>(log2(nsamples * UPSAMPLE)) + 1));

    // Copy input data
    Buffer<float, 3> fftshift_buf(2, N_fft, npulses);
    Buffer<float, 3> inbuf(2, nsamples, npulses);
    // cout << "Width: " << inbuf.width() << endl;
    // cout << "Height: " << inbuf.height() << endl;
    // cout << "Channels: " << inbuf.channels() << endl;
    complex<float> *indata = (complex<float> *)inbuf.begin();
    for (int x = 0; x < npulses; x++) {
        for (int y = 0; y < nsamples; y++) {
            indata[x * nsamples + y] = phs[x * nsamples + y];
        }
    }

    // backprojection - pre-FFT
    cout <<"Halide pre-fft start" << endl;
    int rv = backprojection_pre_fft(inbuf, in_k_r, N_fft, fftshift_buf);
    cout << "Halide pre-fft returned " << rv << endl;
    if (rv != 0) {
        return rv;
    }
    // FFT
    Buffer<float, 3> fft_outbuf(2, N_fft, npulses);
    fftwf_complex *fft_in = reinterpret_cast<fftwf_complex *>(fftshift_buf.begin());
    fftwf_complex *fft_out = reinterpret_cast<fftwf_complex *>(fft_outbuf.begin());
    fftwf_plan plan = fftwf_plan_dft_1d(N_fft, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    cout << "FFTWF: processing " << fft_outbuf.dim(2).extent() << " DFTs" << endl;
    for (int i = 0; i < fft_outbuf.dim(2).extent(); i++) {
        fftwf_execute_dft(plan, &fft_in[i * N_fft], &fft_out[i * N_fft]);
    }
    cout << "FFTWF: finished" << endl;
    fftwf_destroy_plan(plan);
    // backprojection - post-FFT
#if DEBUG_Q
    Buffer<float, 3> out_q(2, N_fft, fft_outbuf.dim(2).extent());
#endif
#if DEBUG_NORM_R0
    Buffer<float, 1> out_norm_r0(fft_outbuf.dim(2).extent());
#endif
#if DEBUG_RR0
    Buffer<double, 3> out_rr0(in_u.dim(0).extent() * in_v.dim(0).extent(),
                              in_pos.dim(0).extent(),
                              fft_outbuf.dim(2).extent());
#endif
#if DEBUG_NORM_RR0
    Buffer<double, 2> out_norm_rr0(in_u.dim(0).extent() * in_v.dim(0).extent(),
                                   fft_outbuf.dim(2).extent());
#endif
#if DEBUG_DR_I
    Buffer<double, 2> out_dr_i(in_u.dim(0).extent() * in_v.dim(0).extent(),
                               fft_outbuf.dim(2).extent());
#endif
#if DEBUG_Q_REAL
    Buffer<double, 2> out_q_real(in_u.dim(0).extent() * in_v.dim(0).extent(),
                                 fft_outbuf.dim(2).extent());
#endif
#if DEBUG_Q_IMAG
    Buffer<double, 2> out_q_imag(in_u.dim(0).extent() * in_v.dim(0).extent(),
                                 fft_outbuf.dim(2).extent());
#endif
#if DEBUG_Q_HAT
    Buffer<double, 3> out_q_hat(2,
                                in_u.dim(0).extent() * in_v.dim(0).extent(),
                                fft_outbuf.dim(2).extent());
#endif
#if DEBUG_IMG
    Buffer<double, 2> out_img(2, in_u.dim(0).extent() * in_v.dim(0).extent());
#endif
#if DEBUG_FIMG
    Buffer<double, 2> out_fimg(2, in_u.dim(0).extent() * in_v.dim(0).extent());
#endif
    Buffer<float, 3> outbuf(2, in_u.dim(0).extent(), in_v.dim(0).extent());
    cout <<"Halide post-fft start" << endl;
    rv = backprojection_post_fft(fft_outbuf, nsamples, delta_r, in_k_r, in_u, in_v, in_pos, in_pixel_locs,
#if DEBUG_Q
        out_q,
#endif
#if DEBUG_NORM_R0
        out_norm_r0,
#endif
#if DEBUG_RR0
        out_rr0,
#endif
#if DEBUG_NORM_RR0
        out_norm_rr0,
#endif
#if DEBUG_DR_I
        out_dr_i,
#endif
#if DEBUG_Q_REAL
        out_q_real,
#endif
#if DEBUG_Q_IMAG
        out_q_imag,
#endif
#if DEBUG_Q_HAT
        out_q_hat,
#endif
#if DEBUG_IMG
        out_img,
#endif
#if DEBUG_FIMG
        out_fimg,
#endif
        outbuf);
    cout << "Halide post-fft returned " << rv << endl;

    // write output
#if DEBUG_Q
    vector<size_t> shape_q { static_cast<size_t>(out_q.dim(2).extent()),
                             static_cast<size_t>(out_q.dim(1).extent()),
                             static_cast<size_t>(out_q.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-q.npy", (float *)out_q.begin(), shape_q);
#endif
#if DEBUG_NORM_R0
    vector<size_t> shape_norm_r0 { static_cast<size_t>(out_norm_r0.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-norm_r0.npy", (float *)out_norm_r0.begin(), shape_norm_r0);
#endif
#if DEBUG_RR0
    vector<size_t> shape_rr0 { static_cast<size_t>(out_rr0.dim(2).extent()),
                               static_cast<size_t>(out_rr0.dim(1).extent()),
                               static_cast<size_t>(out_rr0.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-rr0.npy", (double *)out_rr0.begin(), shape_rr0);
#endif
#if DEBUG_NORM_RR0
    vector<size_t> shape_norm_rr0 { static_cast<size_t>(out_norm_rr0.dim(1).extent()),
                                    static_cast<size_t>(out_norm_rr0.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-norm_rr0.npy", (double *)out_norm_rr0.begin(), shape_norm_rr0);
#endif
#if DEBUG_DR_I
    vector<size_t> shape_dr_i { static_cast<size_t>(out_dr_i.dim(1).extent()),
                                static_cast<size_t>(out_dr_i.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-dr_i.npy", (double *)out_dr_i.begin(), shape_dr_i);
#endif
#if DEBUG_Q_REAL
    vector<size_t> shape_q_real { static_cast<size_t>(out_q_real.dim(1).extent()),
                                  static_cast<size_t>(out_q_real.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-q_real.npy", (double *)out_q_real.begin(), shape_q_real);
#endif
#if DEBUG_Q_IMAG
    vector<size_t> shape_q_imag { static_cast<size_t>(out_q_imag.dim(1).extent()),
                                  static_cast<size_t>(out_q_imag.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-q_imag.npy", (double *)out_q_imag.begin(), shape_q_imag);
#endif
#if DEBUG_Q_HAT
    vector<size_t> shape_q_hat { static_cast<size_t>(out_q_hat.dim(2).extent()),
                                 static_cast<size_t>(out_q_hat.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-q_hat.npy", (complex<double> *)out_q_hat.begin(), shape_q_hat);
#endif
#if DEBUG_IMG
    vector<size_t> shape_img { static_cast<size_t>(out_img.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-img.npy", (complex<double> *)out_img.begin(), shape_img);
#endif
#if DEBUG_FIMG
    vector<size_t> shape_fimg { static_cast<size_t>(out_fimg.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-fimg.npy", (complex<double> *)out_fimg.begin(), shape_fimg);
#endif
    vector<size_t> shape_out { static_cast<size_t>(outbuf.dim(2).extent()),
                               static_cast<size_t>(outbuf.dim(1).extent()) };
    cnpy::npy_save("sarbp_test.npy", (complex<float> *)outbuf.begin(), shape_out);

    return rv;
}
