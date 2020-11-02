#include <complex>
#include <iostream>

#include <Halide.h>
#include <halide_image_io.h>

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
#include "img_output_u8.h"
#include "img_output_to_dB.h"

using namespace std;
using Halide::Runtime::Buffer;

// pre-fft
#define DEBUG_WIN 0
#define DEBUG_FILT 0
#define DEBUG_PHS_FILT 0
#define DEBUG_PHS_PAD 0

// post-fft
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

#define DEBUG_PRE_FFT 0
#define DEBUG_POST_FFT 0
#define DEBUG_BP 0
#define DEBUG_BP_DB 0

#define UPSAMPLE 2
#define RES_FACTOR 1.0
#define ASPECT 1.0

// TODO: These values should depend on the dataset and/or platform?
#define DB_MIN -30
#define DB_MAX 0

int main(int argc, char **argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <platform_dir> <output_png>" << endl;
        return 1;
    }
    string platform_dir = string(argv[1]);
    string output_png = string(argv[2]);

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
    float B_IF = *ptr_flt;

    npydata = cnpy::npy_load(platform_dir + "/delta_r.npy");
    ptr_dbl = npydata.data<double>();
    double delta_r = *ptr_dbl;

    npydata = cnpy::npy_load(platform_dir + "/chirprate.npy");
    ptr_dbl = npydata.data<double>();
    double chirprate = *ptr_dbl;

    npydata = cnpy::npy_load(platform_dir + "/f_0.npy");
    ptr_dbl = npydata.data<double>();
    double f_0 = *ptr_dbl;

    npydata = cnpy::npy_load(platform_dir + "/nsamples.npy");
    ptr_int = npydata.data<int>();
    int nsamples = *ptr_int;

    npydata = cnpy::npy_load(platform_dir + "/npulses.npy");
    ptr_int = npydata.data<int>();
    int npulses = *ptr_int;

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
    Buffer<float, 1> buf_k_r(k_r, nsamples);

    npydata = cnpy::npy_load(platform_dir + "/k_y.npy");
    if (npydata.shape.size() != 1 || npydata.shape[0] != npulses) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    double *k_y = static_cast<double *>(malloc(npulses * sizeof(double)));
    memcpy(k_y, npydata.data<double>(), npulses * sizeof(double));

    npydata = cnpy::npy_load(platform_dir + "/R_c.npy");
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
    if (npydata.shape.size() != 2 || npydata.shape[0] != npulses || npydata.shape[1] != 3) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    float *pos = static_cast<float *>(malloc(npulses * 3 * sizeof(float)));
    memcpy(pos, npydata.data<float>(), npulses * 3 * sizeof(float));
    Buffer<float, 2> buf_pos(pos, 3, npulses);

    npydata = cnpy::npy_load(platform_dir + "/phs.npy");
    if (npydata.shape.size() != 2 || npydata.shape[0] != npulses || npydata.shape[1] != nsamples) {
        cerr << "Bad shape!" << endl;
        return 1;
    }
    complex<float> *phs = static_cast<complex<float> *>(malloc(npulses * nsamples * sizeof(complex<float>)));
    memcpy(phs, npydata.data<complex<float>>(), npulses * nsamples * sizeof(complex<float>));
    Buffer<float, 3> buf_phs(2, nsamples, npulses);
    complex<float> *buf_phs_data = (complex<float> *)buf_phs.begin();
    for (int y = 0; y < npulses; y++) {
        for (int x = 0; x < nsamples; x++) {
            buf_phs_data[y * nsamples + x] = phs[y * nsamples + x];
        }
    }

    cout << "Loaded platform data" << endl;
    cout << "Number of pulses: " << npulses << endl;
    cout << "Pulse sample size: " << nsamples << endl;

    // Img plane variables

    int nu = ip_upsample(nsamples);
    int nv = ip_upsample(npulses);

    double d_u = ip_du(delta_r, RES_FACTOR, nsamples, nu);
    double d_v = ip_dv(ASPECT, d_u);

    Buffer<double, 1> buf_u(nu);
    ip_uv(nu, d_u, buf_u);

    Buffer<double, 1> buf_v(nv);
    ip_uv(nv, d_v, buf_v);

    Buffer<double, 1> buf_k_u(nu);
    ip_k(nu, d_u, buf_k_u);

    Buffer<double, 1> buf_k_v(nv);
    ip_k(nv, d_v, buf_k_v);

    Buffer<const int, 1> buf_n_hat(&N_HAT[0], 3);

    Buffer<double, 1> buf_v_hat(3);
    ip_v_hat(buf_n_hat, buf_R_c, buf_v_hat);

    Buffer<double, 1> buf_u_hat(3);
    ip_u_hat(buf_v_hat, buf_n_hat, buf_u_hat);

    Buffer<double, 2> buf_pixel_locs(nu*nv, 3);
    ip_pixel_locs(buf_u, buf_v, buf_u_hat, buf_v_hat, buf_pixel_locs);

    cout << "Computed image plane parameters" << endl;
    cout << "X length: " << nu << endl;
    cout << "Y length: " << nv << endl;

    // Compute FFT width (power of 2)
    int N_fft = static_cast<int>(pow(2, static_cast<int>(log2(nsamples * UPSAMPLE)) + 1));

    // backprojection - pre-FFT
#if DEBUG_WIN
    Buffer<double, 2> buf_win(buf_phs.dim(1).extent(), buf_phs.dim(2).extent());
#endif
#if DEBUG_FILT
    Buffer<float, 1> buf_filt(buf_phs.dim(1).extent());
#endif
#if DEBUG_PHS_FILT
    Buffer<double, 3> buf_phs_filt(2, buf_phs.dim(1).extent(), buf_phs.dim(2).extent());
#endif
#if DEBUG_PHS_PAD
    Buffer<double, 3> buf_phs_pad(2, N_fft, buf_phs.dim(2).extent());
#endif
    Buffer<double, 3> buf_pre_fft(2, N_fft, npulses);
    cout <<"Halide pre-fft start" << endl;
    int rv = backprojection_pre_fft(buf_phs, buf_k_r, N_fft,
#if DEBUG_WIN
        buf_win,
#endif
#if DEBUG_FILT
        buf_filt,
#endif
#if DEBUG_PHS_FILT
        buf_phs_filt,
#endif
#if DEBUG_PHS_PAD
        buf_phs_pad,
#endif
        buf_pre_fft);
    cout << "Halide pre-fft returned " << rv << endl;
    if (rv != 0) {
        return rv;
    }
    // write debug output
#if DEBUG_WIN
    vector<size_t> shape_win { static_cast<size_t>(buf_win.dim(1).extent()),
                               static_cast<size_t>(buf_win.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-win.npy", (double *)buf_win.begin(), shape_win);
#endif
#if DEBUG_FILT
    vector<size_t> shape_filt { static_cast<size_t>(buf_filt.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-filt.npy", (float *)buf_filt.begin(), shape_filt);
#endif
#if DEBUG_PHS_FILT
    vector<size_t> shape_phs_filt { static_cast<size_t>(buf_phs_filt.dim(2).extent()),
                                    static_cast<size_t>(buf_phs_filt.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-phs_filt.npy", (complex<double> *)buf_phs_filt.begin(), shape_phs_filt);
#endif
#if DEBUG_PHS_PAD
    vector<size_t> shape_phs_pad { static_cast<size_t>(buf_phs_pad.dim(2).extent()),
                                   static_cast<size_t>(buf_phs_pad.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-phs_pad.npy", (complex<double> *)buf_phs_pad.begin(), shape_phs_pad);
#endif
#if DEBUG_PRE_FFT
    vector<size_t> shape_pre_fft { static_cast<size_t>(buf_pre_fft.dim(2).extent()),
                                   static_cast<size_t>(buf_pre_fft.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-pre_fft.npy", (complex<double> *)buf_pre_fft.begin(), shape_pre_fft);
#endif

    // FFT
    Buffer<double, 3> buf_post_fft(2, N_fft, npulses);
    fftw_complex *fft_in = reinterpret_cast<fftw_complex *>(buf_pre_fft.begin());
    fftw_complex *fft_out = reinterpret_cast<fftw_complex *>(buf_post_fft.begin());
    fftw_plan plan = fftw_plan_dft_1d(N_fft, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);
    cout << "FFTW: processing " << buf_post_fft.dim(2).extent() << " DFTs" << endl;
    for (int i = 0; i < buf_post_fft.dim(2).extent(); i++) {
        fftw_execute_dft(plan, &fft_in[i * N_fft], &fft_out[i * N_fft]);
    }
    cout << "FFTW: finished" << endl;
    fftw_destroy_plan(plan);
    // write debug output
#if DEBUG_POST_FFT
    vector<size_t> shape_post_fft { static_cast<size_t>(buf_post_fft.dim(2).extent()),
                                    static_cast<size_t>(buf_post_fft.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-post_fft.npy", (complex<double> *)buf_post_fft.begin(), shape_post_fft);
#endif

    // backprojection - post-FFT
#if DEBUG_Q
    Buffer<double, 3> buf_q(2, N_fft, buf_post_fft.dim(2).extent());
#endif
#if DEBUG_NORM_R0
    Buffer<float, 1> buf_norm_r0(buf_post_fft.dim(2).extent());
#endif
#if DEBUG_RR0
    Buffer<double, 3> buf_rr0(buf_u.dim(0).extent() * buf_v.dim(0).extent(),
                              buf_pos.dim(0).extent(),
                              buf_post_fft.dim(2).extent());
#endif
#if DEBUG_NORM_RR0
    Buffer<double, 2> buf_norm_rr0(buf_u.dim(0).extent() * buf_v.dim(0).extent(),
                                   buf_post_fft.dim(2).extent());
#endif
#if DEBUG_DR_I
    Buffer<double, 2> buf_dr_i(buf_u.dim(0).extent() * buf_v.dim(0).extent(),
                               buf_post_fft.dim(2).extent());
#endif
#if DEBUG_Q_REAL
    Buffer<double, 2> buf_q_real(buf_u.dim(0).extent() * buf_v.dim(0).extent(),
                                 buf_post_fft.dim(2).extent());
#endif
#if DEBUG_Q_IMAG
    Buffer<double, 2> buf_q_imag(buf_u.dim(0).extent() * buf_v.dim(0).extent(),
                                 buf_post_fft.dim(2).extent());
#endif
#if DEBUG_Q_HAT
    Buffer<double, 3> buf_q_hat(2,
                                buf_u.dim(0).extent() * buf_v.dim(0).extent(),
                                buf_post_fft.dim(2).extent());
#endif
#if DEBUG_IMG
    Buffer<double, 2> buf_img(2, buf_u.dim(0).extent() * buf_v.dim(0).extent());
#endif
#if DEBUG_FIMG
    Buffer<double, 2> buf_fimg(2, buf_u.dim(0).extent() * buf_v.dim(0).extent());
#endif
    Buffer<double, 3> buf_bp(2, buf_u.dim(0).extent(), buf_v.dim(0).extent());
    cout << "Halide post-fft start" << endl;
    rv = backprojection_post_fft(buf_post_fft, nsamples, delta_r, buf_k_r, buf_u, buf_v, buf_pos, buf_pixel_locs,
#if DEBUG_Q
        buf_q,
#endif
#if DEBUG_NORM_R0
        buf_norm_r0,
#endif
#if DEBUG_RR0
        buf_rr0,
#endif
#if DEBUG_NORM_RR0
        buf_norm_rr0,
#endif
#if DEBUG_DR_I
        buf_dr_i,
#endif
#if DEBUG_Q_REAL
        buf_q_real,
#endif
#if DEBUG_Q_IMAG
        buf_q_imag,
#endif
#if DEBUG_Q_HAT
        buf_q_hat,
#endif
#if DEBUG_IMG
        buf_img,
#endif
#if DEBUG_FIMG
        buf_fimg,
#endif
        buf_bp);
    cout << "Halide post-fft returned " << rv << endl;
    if (rv != 0) {
        return rv;
    }

    // write output
#if DEBUG_Q
    vector<size_t> shape_q { static_cast<size_t>(buf_q.dim(2).extent()),
                             static_cast<size_t>(buf_q.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-q.npy", (complex<double> *)buf_q.begin(), shape_q);
#endif
#if DEBUG_NORM_R0
    vector<size_t> shape_norm_r0 { static_cast<size_t>(buf_norm_r0.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-norm_r0.npy", (float *)buf_norm_r0.begin(), shape_norm_r0);
#endif
#if DEBUG_RR0
    vector<size_t> shape_rr0 { static_cast<size_t>(buf_rr0.dim(2).extent()),
                               static_cast<size_t>(buf_rr0.dim(1).extent()),
                               static_cast<size_t>(buf_rr0.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-rr0.npy", (double *)buf_rr0.begin(), shape_rr0);
#endif
#if DEBUG_NORM_RR0
    vector<size_t> shape_norm_rr0 { static_cast<size_t>(buf_norm_rr0.dim(1).extent()),
                                    static_cast<size_t>(buf_norm_rr0.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-norm_rr0.npy", (double *)buf_norm_rr0.begin(), shape_norm_rr0);
#endif
#if DEBUG_DR_I
    vector<size_t> shape_dr_i { static_cast<size_t>(buf_dr_i.dim(1).extent()),
                                static_cast<size_t>(buf_dr_i.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-dr_i.npy", (double *)buf_dr_i.begin(), shape_dr_i);
#endif
#if DEBUG_Q_REAL
    vector<size_t> shape_q_real { static_cast<size_t>(buf_q_real.dim(1).extent()),
                                  static_cast<size_t>(buf_q_real.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-q_real.npy", (double *)buf_q_real.begin(), shape_q_real);
#endif
#if DEBUG_Q_IMAG
    vector<size_t> shape_q_imag { static_cast<size_t>(buf_q_imag.dim(1).extent()),
                                  static_cast<size_t>(buf_q_imag.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-q_imag.npy", (double *)buf_q_imag.begin(), shape_q_imag);
#endif
#if DEBUG_Q_HAT
    vector<size_t> shape_q_hat { static_cast<size_t>(buf_q_hat.dim(2).extent()),
                                 static_cast<size_t>(buf_q_hat.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-q_hat.npy", (complex<double> *)buf_q_hat.begin(), shape_q_hat);
#endif
#if DEBUG_IMG
    vector<size_t> shape_img { static_cast<size_t>(buf_img.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-img.npy", (complex<double> *)buf_img.begin(), shape_img);
#endif
#if DEBUG_FIMG
    vector<size_t> shape_fimg { static_cast<size_t>(buf_fimg.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-fimg.npy", (complex<double> *)buf_fimg.begin(), shape_fimg);
#endif
#if DEBUG_BP
    vector<size_t> shape_bp { static_cast<size_t>(buf_bp.dim(2).extent()),
                              static_cast<size_t>(buf_bp.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-bp.npy", (complex<double> *)buf_bp.begin(), shape_bp);
#endif

    // Convert to dB
    Buffer<double, 2> buf_bp_dB(buf_u.dim(0).extent(), buf_v.dim(0).extent());
    cout << "Halide dB conversion start" << endl;
    rv = img_output_to_dB(buf_bp, buf_bp_dB);
    cout << "Halide dB conversion returned " << rv << endl;
    if (rv != 0) {
        return rv;
    }
#if DEBUG_BP_DB
    vector<size_t> shape_bp_dB { static_cast<size_t>(buf_bp_dB.dim(1).extent()),
                                 static_cast<size_t>(buf_bp_dB.dim(0).extent()) };
    cnpy::npy_save("sarbp_debug-bp_dB.npy", (double *)buf_bp_dB.begin(), shape_bp_dB);
#endif

    // Produce output image
    Buffer<uint8_t, 2> buf_bp_u8(buf_bp_dB.dim(0).extent(), buf_bp_dB.dim(1).extent());
    cout << "Halide PNG production start" << endl;
    rv = img_output_u8(buf_bp_dB, DB_MIN, DB_MAX, buf_bp_u8);
    cout << "Halide PNG production returned " << rv << endl;
    if (rv != 0) {
        return rv;
    }
    Halide::Tools::convert_and_save_image(buf_bp_u8, output_png);

    return rv;
}
