#undef NDEBUG // force assertions
#include <assert.h>
#include <chrono>
#include <complex>
#include <getopt.h>
#include <iostream>

#include <Halide.h>
#include <halide_image_io.h>

#include <cnpy.h>

#include <mpi.h>

#include "dft.h"
#include "PlatformData.h"
#include "ImgPlane.h"

// Halide generators
#include "backprojection_debug.h"
#include "backprojection.h"
#include "backprojection_distributed.h"
#include "backprojection_cuda.h"
#include "backprojection_cuda_distributed.h"
#include "backprojection_ritsar.h"
#include "backprojection_ritsar_s.h"
#include "backprojection_ritsar_p.h"
#include "backprojection_ritsar_vp.h"
#include "backprojection_auto_m16.h"
#include "img_output_u8.h"
#include "img_output_to_dB.h"

using namespace std;
using namespace std::chrono;
using Halide::Runtime::Buffer;

// local to this file
#define DEBUG_BP 0
#define DEBUG_BP_DB 0

#define SCHED_DEFAULT "cpu"
// the following defaults are the same as RITSAR
#define DB_MIN_DEFAULT 0
#define DB_MAX_DEFAULT 0
#define UPSAMPLE_DEFAULT 6
#define TAYLOR_DEFAULT 20
#define RES_FACTOR_DEFAULT RES_FACTOR
#define ASPECT_DEFAULT ASPECT

static const char short_options[] = "p:o:d:D:s:t:u:r:a:h";
static const struct option long_options[] = {
  {"platform-dir",    required_argument,  NULL, 'p'},
  {"output",          required_argument,  NULL, 'o'},
  {"db-min",          required_argument,  NULL, 'd'},
  {"db-max",          required_argument,  NULL, 'D'},
  {"schedule",        required_argument,  NULL, 's'},
  {"taylor",          required_argument,  NULL, 't'},
  {"upsample",        required_argument,  NULL, 'u'},
  {"res-factor",      required_argument,  NULL, 'r'},
  {"aspect",          required_argument,  NULL, 'a'},
  {"help",            no_argument,        NULL, 'h'},
  {0, 0, 0, 0}
};

static void print_usage(string prog, ostream& os) {
    os << "Usage: " << prog << " -p DIR -o FILE [OPTION]..." << endl;
    os << "Options:" << endl;
    os << "  -p, --platform-dir=DIR  Platform input directory" << endl;
    os << "  -o, --output=FILE       Output image file (PNG)" << endl;
    os << "  -d, --db-min=REAL       Output image min dB" << endl;
    os << "                          Default: " << DB_MIN_DEFAULT << endl;
    os << "  -D, --db-max=REAL       Output image max dB" << endl;
    os << "                          Default: " << DB_MAX_DEFAULT << endl;
    os << "  -s, --schedule=NAME     One of: cpu[_distributed]" << endl;
    os << "                                  cuda[_distributed]" << endl;
    os << "                                  ritsar[-s|-p|-vp]" << endl;
    os << "                          Default: " << SCHED_DEFAULT << endl;
    os << "  -t, --taylor=INT        Taylor count" << endl;
    os << "                          Default: " << TAYLOR_DEFAULT << endl;
    os << "  -u, --upsample=INT      Upsample factor" << endl;
    os << "                          Default: " << UPSAMPLE_DEFAULT << endl;
    os << "  -r, --res-factor=REAL   Image res in units of theoretical resolution size" << endl;
    os << "                          Default: " << RES_FACTOR_DEFAULT << endl;
    os << "  -a, --aspect=REAL       Aspect ratio of range to cross range" << endl;
    os << "                          Default: " << ASPECT_DEFAULT << endl;
    os << "  -h, --help              Print this message and exit" << endl;
}

int main(int argc, char **argv) {
    string platform_dir = "";
    string output_png = "";
    string bp_sched = SCHED_DEFAULT;
    int taylor = TAYLOR_DEFAULT;
    int upsample = UPSAMPLE_DEFAULT;
    double dB_min = DB_MIN_DEFAULT;
    double dB_max = DB_MAX_DEFAULT;
    double res_factor = RES_FACTOR_DEFAULT;
    double aspect = ASPECT_DEFAULT;
    while (1) {
        switch (getopt_long(argc, argv, short_options, long_options, NULL)) {
            case -1:
                break;
            case 'p':
                platform_dir = string(optarg);
                continue;
            case 'o':
                output_png = string(optarg);
                continue;
            case 'd':
                dB_min = atof(optarg);
                continue;
            case 'D':
                dB_max = atof(optarg);
                continue;
            case 's':
                bp_sched = string(optarg);
                continue;
            case 't':
                taylor = atoi(optarg);
                continue;
            case 'u':
                upsample = atoi(optarg);
                continue;
            case 'r':
                res_factor = atof(optarg);
                continue;
            case 'a':
                aspect = atof(optarg);
                continue;
            case 'h':
                print_usage(argv[0], cout);
                return 0;
            default:
                print_usage(argv[0], cerr);
                return EXIT_FAILURE;
        }
        break;
    }
    if (platform_dir.empty() || output_png.empty()) {
        cerr << "Missing required parameters" << endl;
        print_usage(argv[0], cerr);
        return EXIT_FAILURE;
    }
    if (dB_min >= dB_max) {
        // Can add support in img_output_u8(...) to avoid this constraint
        cerr << "Constraint failed: dB_min < dB_max" << endl;
        print_usage(argv[0], cerr);
        return EXIT_FAILURE;
    }
    if (taylor < 1) {
        cerr << "Constraint failed: taylor >= 1" << endl;
        print_usage(argv[0], cerr);
        return EXIT_FAILURE;
    }
    // TODO: How can upsample be disabled?
    if (upsample < 1) {
        cerr << "Constraint failed: upsample >= 1" << endl;
        print_usage(argv[0], cerr);
        return EXIT_FAILURE;
    }
    if (res_factor <= 0) {
        cerr << "Constraint failed: res-factor > 0" << endl;
        print_usage(argv[0], cerr);
        return EXIT_FAILURE;
    }
    if (aspect <= 0) {
        cerr << "Constraint failed: aspect > 0" << endl;
        print_usage(argv[0], cerr);
        return EXIT_FAILURE;
    }

    // determine which backprojection implementation to use
    auto backprojection_impl = backprojection;
    bool is_distributed = false;
    if (bp_sched == "cpu") {
        backprojection_impl = backprojection;
        cout << "Using schedule for CPU only" << endl;
    } else if (bp_sched == "cpu_distributed") {
        backprojection_impl = backprojection_distributed;
        is_distributed = true;
        cout << "Using schedule for distributed CPU" << endl;
    } else if (bp_sched == "cuda_distributed") {
        backprojection_impl = backprojection_cuda_distributed;
        is_distributed = true;
        cout << "Using schedule for distributed CUDA" << endl;
    } else if (bp_sched == "cuda") {
        backprojection_impl = backprojection_cuda;
        cout << "Using schedule with CUDA" << endl;
    } else if (bp_sched == "ritsar") {
        backprojection_impl = backprojection_ritsar;
        cout << "Using RITSAR baseline (vectorize)" << endl;
    } else if (bp_sched == "ritsar-s") {
        backprojection_impl = backprojection_ritsar_s;
        cout << "Using RITSAR baseline (serial)" << endl;
    } else if (bp_sched == "ritsar-p") {
        backprojection_impl = backprojection_ritsar_p;
        cout << "Using RITSAR baseline (parallel)" << endl;
    } else if (bp_sched == "ritsar-vp") {
        backprojection_impl = backprojection_ritsar_vp;
        cout << "Using RITSAR baseline (vectorize+parallel)" << endl;
    } else if (bp_sched == "auto-m16") {
        backprojection_impl = backprojection_auto_m16;
        cout << "Using CPU autoschedule (Mullapudi2016)" << endl;
    } else {
        cerr << "Unknown schedule: " << bp_sched << endl;
        print_usage(argv[0], cerr);
        return EXIT_FAILURE;
    }

    int rank = 0, numprocs = 0;
    if (is_distributed) {
        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    }

    auto start = high_resolution_clock::now();
    PlatformData pd = platform_load(platform_dir, is_distributed);
    auto stop = high_resolution_clock::now();
    cout << "Loaded platform data in "
         << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
    cout << "Number of pulses: " << pd.npulses << endl;
    cout << "Pulse sample size: " << pd.nsamples << endl;

    const float *n_hat = pd.n_hat.has_value() ? pd.n_hat.value().begin() : &N_HAT[0];
    start = high_resolution_clock::now();
    ImgPlane ip = img_plane_create(pd, res_factor, n_hat, aspect, upsample, is_distributed);
    stop = high_resolution_clock::now();
    cout << "Computed image plane parameters in "
         << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
    cout << "X length: " << ip.nu << endl;
    cout << "Y length: " << ip.nv << endl;

    // Compute FFT width (power of 2)
    int N_fft = static_cast<int>(pow(2, static_cast<int>(log2(pd.nsamples * upsample)) + 1));

    // FFTW: init shared context
    dft_init_fftw(static_cast<size_t>(N_fft));

    // backprojection - pre-FFT
#if DEBUG_WIN
    Buffer<double, 2> buf_win(pd.phs.dim(1).extent(), pd.phs.dim(2).extent());
#endif
#if DEBUG_FILT
    Buffer<float, 1> buf_filt(pd.phs.dim(1).extent());
#endif
#if DEBUG_PHS_FILT
    Buffer<double, 3> buf_phs_filt(2, pd.phs.dim(1).extent(), pd.phs.dim(2).extent());
#endif
#if DEBUG_PHS_PAD
    Buffer<double, 3> buf_phs_pad(2, N_fft, pd.phs.dim(2).extent());
#endif
#if DEBUG_PRE_FFT
    Buffer<double, 3> buf_pre_fft(2, N_fft, pd.phs.dim(2).extent());
#endif
#if DEBUG_POST_FFT
    Buffer<double, 3> buf_post_fft(2, N_fft, pd.phs.dim(2).extent());
#endif
#if DEBUG_Q
    Buffer<double, 3> buf_q(2, N_fft, buf_post_fft.dim(2).extent());
#endif
#if DEBUG_NORM_R0
    Buffer<float, 1> buf_norm_r0(buf_post_fft.dim(2).extent());
#endif
#if DEBUG_RR0
    Buffer<double, 3> buf_rr0(ip.u.dim(0).extent() * ip.v.dim(0).extent(),
                              pd.pos.dim(0).extent(),
                              buf_post_fft.dim(2).extent());
#endif
#if DEBUG_NORM_RR0
    Buffer<double, 2> buf_norm_rr0(ip.u.dim(0).extent() * ip.v.dim(0).extent(),
                                   buf_post_fft.dim(2).extent());
#endif
#if DEBUG_DR_I
    Buffer<double, 2> buf_dr_i(ip.u.dim(0).extent() * ip.v.dim(0).extent(),
                               buf_post_fft.dim(2).extent());
#endif
#if DEBUG_Q_REAL
    Buffer<double, 2> buf_q_real(ip.u.dim(0).extent() * ip.v.dim(0).extent(),
                                 buf_post_fft.dim(2).extent());
#endif
#if DEBUG_Q_IMAG
    Buffer<double, 2> buf_q_imag(ip.u.dim(0).extent() * ip.v.dim(0).extent(),
                                 buf_post_fft.dim(2).extent());
#endif
#if DEBUG_Q_HAT
    Buffer<double, 3> buf_q_hat(2,
                                ip.u.dim(0).extent() * ip.v.dim(0).extent(),
                                buf_post_fft.dim(2).extent());
#endif
#if DEBUG_IMG
    Buffer<double, 2> buf_img(2, ip.u.dim(0).extent() * ip.v.dim(0).extent());
#endif
#if DEBUG_FIMG
    Buffer<double, 2> buf_fimg(2, ip.u.dim(0).extent() * ip.v.dim(0).extent());
#endif

    Buffer<double, 2> buf_bp(nullptr, {2, ip.u.dim(0).extent(), ip.v.dim(0).extent()});
    if (is_distributed) {
        buf_bp.set_distributed({2, ip.u.dim(0).extent(), ip.v.dim(0).extent()});
        // Query local buffer size
        backprojection_impl(pd.phs, pd.k_r, taylor, N_fft, pd.delta_r, ip.u, ip.v, pd.pos, ip.pixel_locs, buf_bp);
        buf_bp.allocate();
    } else {
        buf_bp = Buffer<double, 3>(2, ip.u.dim(0).extent(), ip.v.dim(0).extent());
    }
    cout << "Halide backprojection start " << endl;
    start = high_resolution_clock::now();
    int rv = backprojection_impl(pd.phs, pd.k_r, taylor, N_fft, pd.delta_r, ip.u, ip.v, pd.pos, ip.pixel_locs,
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
#if DEBUG_PRE_FFT
        buf_pre_fft,
#endif
#if DEBUG_POST_FFT
        buf_post_fft,
#endif
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
    stop = high_resolution_clock::now();
    cout << "Halide backprojection returned " << rv << " in "
         << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
    if (rv != 0) {
        return rv;
    }

    Buffer<double, 2> buf_bp_full(nullptr, 0);
    if (is_distributed) {
        // Send data to rank 0
        buf_bp.copy_to_host();
        if (rank == 0) {
            cout << "MPI full backprojection receive start" << endl;
            start = high_resolution_clock::now();
            buf_bp_full = Buffer<double, 3>(2, ip.u.dim(0).extent(), ip.v.dim(0).extent());
            // Copy buf_bp to buf_bp_full
            memcpy(buf_bp_full.data(), buf_bp.data(), sizeof(double) * buf_bp.dim(1).extent() * buf_bp.dim(2).extent() * 2);
            for (int r = 1; r < numprocs; r++) {
                // Obtain the min & extent from the node
                int r_min = 0, r_extent = 0;
                MPI_Recv(&r_min, 1, MPI_INT, r, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Recv(&r_extent, 1, MPI_INT, r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                cout << "Received r_min = " << r_min << ", r_extent = " << r_extent << " from rank " << r << std::endl;
                // Receving the actual content
                MPI_Recv(buf_bp_full.data() + r_min * 2,
                         r_extent * 2,
                         MPI_DOUBLE,
                         r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            stop = high_resolution_clock::now();
            cout << "MPI full backprojection receive completed in "
                 << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
        } else {
            cout << "MPI local backprojection send start" << endl;
            start = high_resolution_clock::now();
            int r_min = buf_bp.dim(2).min() * buf_bp.dim(1).extent(), r_extent = buf_bp.dim(2).extent() * buf_bp.dim(1).extent();
            MPI_Send(&r_min, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&r_extent, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            // Sending the actual content
            MPI_Send(buf_bp.data(), r_extent * 2, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
            stop = high_resolution_clock::now();
            cout << "MPI local backprojection send completed in "
                 << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
        }
    } else {
        buf_bp_full = buf_bp;
    }

    // FFTW: clean up shared context
    dft_destroy_fftw();

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
#if DEBUG_POST_FFT
    vector<size_t> shape_post_fft { static_cast<size_t>(buf_post_fft.dim(2).extent()),
                                    static_cast<size_t>(buf_post_fft.dim(1).extent()) };
    cnpy::npy_save("sarbp_debug-post_fft.npy", (complex<double> *)buf_post_fft.begin(), shape_post_fft);
#endif
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

    if (!is_distributed || rank == 0) {
        // Convert to dB
        Buffer<double, 2> buf_bp_dB(ip.u.dim(0).extent(), ip.v.dim(0).extent());
        cout << "Halide dB conversion start" << endl;
        start = high_resolution_clock::now();
        rv = img_output_to_dB(buf_bp_full, buf_bp_dB);
        stop = high_resolution_clock::now();
        cout << "Halide dB conversion returned " << rv << " in "
             << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
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
        start = high_resolution_clock::now();
        rv = img_output_u8(buf_bp_dB, dB_min, dB_max, buf_bp_u8);
        stop = high_resolution_clock::now();
        cout << "Halide PNG production returned " << rv << " in "
             << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
        if (rv != 0) {
            return rv;
        }
        start = high_resolution_clock::now();
        Halide::Tools::convert_and_save_image(buf_bp_u8, output_png);
        stop = high_resolution_clock::now();
        cout << "Wrote " << output_png << " in "
             << duration_cast<milliseconds>(stop - start).count() << " ms" << endl;
    }

    if (is_distributed) {
        MPI_Finalize();
    }

    return rv;
}
