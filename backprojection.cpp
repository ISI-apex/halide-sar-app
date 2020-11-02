#include <stdio.h>

#include <Halide.h>

#include "complexfunc.h"
#include "signal.h"
#include "signal_complex.h"
#include "util.h"

using namespace Halide;

// pre-fft
#define DEBUG_WIN 0
#define DEBUG_FILT 0
#define DEBUG_PHS_FILT 0
#define DEBUG_PHS_PAD 0

// post-fft
#define DEBUG_Q 0
#define DEBUG_DR 0
#define DEBUG_NORM_R0 0
#define DEBUG_RR0 0
#define DEBUG_NORM_RR0 0
#define DEBUG_DR_I 0
#define DEBUG_Q_REAL 0
#define DEBUG_Q_IMAG 0
#define DEBUG_Q_HAT 0
#define DEBUG_IMG 0
#define DEBUG_FIMG 0

#define TAYLOR_S_L 17

class BackprojectionPreFFTGenerator : public Halide::Generator<BackprojectionPreFFTGenerator> {
public:
    Input<Buffer<float>> phs {"phs", 3}; // complex 2d input  (Halide thinks this is 3d: [2, x, y])
    Input<Buffer<float>> k_r {"k_r", 1};
    Input<int> N_fft {"N_fft"};

#if DEBUG_WIN
    Output<Buffer<double>> out_win{"out_win", 2};
#endif
#if DEBUG_FILT
    Output<Buffer<float>> out_filt{"out_filt", 1};
#endif
#if DEBUG_PHS_FILT
    Output<Buffer<double>> out_phs_filt{"out_phs_filt", 3}; // complex
#endif
#if DEBUG_PHS_PAD
    Output<Buffer<double>> out_phs_pad{"out_phs_pad", 3}; // complex
#endif

    Output<Buffer<double>> output_buffer{"output_packed", 3}; // complex 2d output (Halide thinks this is 3d: [2, x, y])

    Var c{"c"}, x{"x"}, y{"y"};

    void generate() {
        Expr nsamples("nsamples");
        nsamples = phs.dim(1).extent();
        Expr npulses("npulses");
        npulses = phs.dim(2).extent();
        Func phs_func = phs;
        ComplexFunc input(c, phs_func);

        // Create window
        Func win_x("win_x");
        win_x = taylor_func(nsamples, TAYLOR_S_L);
        Func win_y("win_y");
        win_y = taylor_func(npulses, TAYLOR_S_L);
        Func win("win");
        win(x, y) = win_x(x) * win_y(y);
#if DEBUG_WIN
        out_win(x, y) = win(x, y);
#endif

        // Filter phase history
        Func filt("filt");
        filt(x) = abs(k_r(x));
#if DEBUG_FILT
        out_filt(x) = filt(x);
#endif

        ComplexFunc phs_filt(c, "phs_filt");
        phs_filt(x, y) = input(x, y) * filt(x) * win(x, y);
#if DEBUG_PHS_FILT
        out_phs_filt(c, x, y) = phs_filt.inner(c, x, y);
#endif

        // Zero pad phase history
        ComplexFunc phs_pad(c, "phs_pad");
        phs_pad(x, y) = pad(phs_filt, nsamples, npulses,
                            ComplexExpr(c, Expr(0.0), Expr(0.0)),
                            N_fft, npulses, c, x, y);
#if DEBUG_PHS_PAD
        out_phs_pad(c, x, y) = phs_pad.inner(c, x, y);
#endif

        // shift
        ComplexFunc fftsh(c, "fftshift");
        fftsh(x, y) = fftshift(phs_pad, N_fft, npulses, x, y);

        output_buffer(c, x, y) = fftsh.inner(c, x, y);

        phs_func.compute_root();
        win_x.compute_root();
        win_y.compute_root();
        win.compute_root();
        filt.compute_root();
        phs_filt.inner.compute_root();
        phs_pad.inner.compute_root();
        fftsh.inner.compute_root();
    }
};

class BackprojectionPostFFTGenerator : public Halide::Generator<BackprojectionPostFFTGenerator> {
public:
    Input<Buffer<double>> in {"in", 3};
    Input<int> nsamples {"nsamples"};
    Input<double> delta_r {"delta_r"};
    Input<Buffer<float>> k_r {"k_r", 1};
    Input<Buffer<double>> u {"u", 1};
    Input<Buffer<double>> v {"v", 1};
    Input<Buffer<float>> pos {"pos", 2};
    Input<Buffer<double>> r {"r", 2};

#if DEBUG_Q
    Output<Buffer<double>> out_Q{"out_Q", 3};
#endif
#if DEBUG_NORM_R0
    Output<Buffer<float>> out_norm_r0{"out_norm_r0", 1};
#endif
#if DEBUG_RR0
    Output<Buffer<double>> out_rr0{"out_rr0", 3};
#endif
#if DEBUG_NORM_RR0
    Output<Buffer<double>> out_norm_rr0{"out_norm_rr0", 2};
#endif
#if DEBUG_DR_I
    Output<Buffer<double>> out_dr_i{"out_dr_i", 2};
#endif
#if DEBUG_Q_REAL
    Output<Buffer<double>> out_q_real{"out_q_real", 2};
#endif
#if DEBUG_Q_IMAG
    Output<Buffer<double>> out_q_imag{"out_q_imag", 2};
#endif
#if DEBUG_Q_HAT
    Output<Buffer<double>> out_q_hat{"out_q_hat", 3};
#endif
#if DEBUG_IMG
    Output<Buffer<double>> out_img{"out_img", 2};
#endif
#if DEBUG_FIMG
    Output<Buffer<double>> out_fimg{"out_fimg", 2};
#endif

    Output<Buffer<double>> output_buffer{"output_packed", 3};

    Var c{"c"}, x{"x"}, y{"y"}, z{"z"};

    // xs: {nu*nv, npulses}
    // lsa, lsb, lsn: implicit linspace parameters (min, max, count)
    // fp: {N_fft, npulses}
    // output: {nu*nv, npulses}
    inline Expr interp(Func xs, Expr lsa, Expr lsb, Expr lsn, ComplexFunc fp, Expr c) {
        Expr lsr = (lsb-lsa) / (lsn-1);             // linspace rate of increase
        Expr luts = (xs(x, y) - lsa) / lsr;         // input value scaled to linspace
        Expr lutl = ConciseCasts::i32(floor(luts)); // lower index
        Expr lutu = lutl + 1;                       // upper index
        Expr luto = luts - lutl;                    // offset within lower-upper span

        // clamps to ensure fp accesses occur within the expected range, even if the input is crazy
        Expr cll = clamp(lutl, 0, lsn - 1);
        Expr clu = clamp(lutu, 0, lsn - 1);
        Expr pos = clamp(luto, Expr(0.0), Expr(1.0));

        return lerp(fp.inner(c, cll, y), fp.inner(c, clu, y), pos);
    }

    void generate() {
        // inputs as functions
        Func in_func("in_func");
        in_func = in;
        ComplexFunc input(c, in_func);

        // some extents and related RDoms
        Expr N_fft = in.dim(1).extent();
        Expr npulses = in.dim(2).extent();
        Expr nu = u.dim(0).extent();
        Expr nv = v.dim(0).extent();
        Expr nd = pos.dim(0).extent(); // nd = 3 (spatial dimensions)
        RDom rnpulses(0, npulses, "rnpulses");
        RDom rnd(0, nd, "rnd");

        // k_c: produces scalar
        Expr k_c = k_r(nsamples / 2);

        // Q: produces shape {N_fft, npulses}
        ComplexFunc Q(c, "Q");
        Q(x, y) = fftshift(input, N_fft, npulses, x, y);
#if DEBUG_Q
        out_Q(c, x, y) = Q.inner(c, x, y);
#endif

        // norm(r0): produces shape {npulses}
        Func norm_r0("norm_r0");
        norm_r0(x) = norm(pos(rnd, x));
#if DEBUG_NORM_R0
        out_norm_r0(x) = norm_r0(x);
#endif

        // r - r0: produces shape {nu*nv, nd, npulses}
        Func rr0("rr0");
        rr0(x, y, z) = r(x, y) - pos(y, z);
#if DEBUG_RR0
        out_rr0(x, y, z) = rr0(x, y, z);
#endif

        // norm(r - r0): produces shape {nu*nv, npulses}
        Func norm_rr0("norm_rr0");
        norm_rr0(x, y) = norm(rr0(x, rnd, y));
#if DEBUG_NORM_RR0
        out_norm_rr0(x, y) = norm_rr0(x, y);
#endif

        // dr_i: produces shape {nu*nv, npulses}
        Func dr_i("dr_i");
        dr_i(x, y) = norm_r0(y) - norm_rr0(x, y);
#if DEBUG_DR_I
        out_dr_i(x, y) = dr_i(x, y);
#endif

        // Q_{real,imag,hat}: produce shape {nu*nv, npulses}
        Func Q_real("Q_real");
        Func Q_imag("Q_imag");
        ComplexFunc Q_hat(c, "Q_hat");
        Q_real(x, y) = interp(dr_i, floor(-nsamples * delta_r / 2), floor(nsamples * delta_r / 2), N_fft, Q, 0);
#if DEBUG_Q_REAL
        out_q_real(x, y) = Q_real(x, y);
#endif
        Q_imag(x, y) = interp(dr_i, floor(-nsamples * delta_r / 2), floor(nsamples * delta_r / 2), N_fft, Q, 1);
#if DEBUG_Q_IMAG
        out_q_imag(x, y) = Q_imag(x, y);
#endif
        // NOTE: it is possible to do this, directly
        //Q_hat(x, y) = interp(dr_i, floor(-nsamples * delta_r / 2), floor(nsamples * delta_r / 2), N_fft, Q, c);
        Q_hat(x, y) = ComplexExpr(c, Q_real(x, y), Q_imag(x, y));
#if DEBUG_Q_HAT
        out_q_hat(c, x, y) = Q_hat.inner(c, x, y);
#endif

        // img: produces shape {nu*nv}
        ComplexFunc img(c, "img");
        img(x) = ComplexExpr(c, Expr(0.0), Expr(0.0));
        img(x) += Q_hat(x, rnpulses) * exp(ComplexExpr(c, Expr(0.0), Expr(-1.0)) * k_c * dr_i(x, rnpulses));
#if DEBUG_IMG
        out_img(c, x) = img.inner(c, x);
#endif

        // finally...
        Expr fdr_i = norm_r0(npulses / 2) - norm_rr0(x, npulses / 2);
        ComplexFunc fimg(c, "fimg");
        fimg(x) = img(x) * exp(ComplexExpr(c, Expr(0.0), Expr(1.0)) * k_c * fdr_i);
#if DEBUG_FIMG
        out_fimg(c, x) = fimg.inner(c, x);
#endif

        // img_rect: produce shape {nu, nv}, but reverse row order
        ComplexFunc img_rect(c, "img_rect");
        img_rect(x, y) = fimg((nu * (nv - y - 1)) + x);
        output_buffer(c, x, y) = img_rect.inner(c, x, y);

        int vectorsize = 16;
        int blocksize = 64;
        in_func.compute_root();
        Q.inner.compute_root();
        norm_r0.compute_root();
        rr0.compute_root().parallel(z).vectorize(x, vectorsize);
        norm_rr0.compute_root().parallel(y).vectorize(x, vectorsize);
        dr_i.in(Q_real).compute_inline();
        dr_i.in(Q_imag).compute_inline();
        Q_real.in(Q_hat.inner).compute_inline();
        Q_imag.in(Q_hat.inner).compute_inline();
        Q_hat.inner.compute_root().unroll(c).reorder(x,y).vectorize(x, vectorsize).parallel(y);
        img.inner.compute_root();
        img.inner.update(0).parallel(x, blocksize);
        fimg.inner.in(img_rect.inner).compute_inline();
        img_rect.inner.compute_root().parallel(y).vectorize(x, vectorsize);
        //output_buffer.print_loop_nest();
    }
};

HALIDE_REGISTER_GENERATOR(BackprojectionPreFFTGenerator, backprojection_pre_fft)
HALIDE_REGISTER_GENERATOR(BackprojectionPostFFTGenerator, backprojection_post_fft)
