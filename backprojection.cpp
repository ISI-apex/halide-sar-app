#include <Halide.h>

#include "backprojection_debug.h"
#include "halide_complexfunc.h"
#include "signal.h"
#include "signal_complex.h"
#include "util.h"

using namespace Halide;
using namespace Halide::Tools;

class BackprojectionGenerator : public Halide::Generator<BackprojectionGenerator> {
public:
    GeneratorParam<int32_t> vectorsize {"vectorsize", 16};
    GeneratorParam<int32_t> blocksize {"blocksize", 64};
    GeneratorParam<bool> print_loop_nest {"print_loop_nest", false};

    // 2-D complex data (3-D when handled as primitive data: {2, x, y})
    Input<Buffer<float>> phs {"phs", 3};
    Input<Buffer<float>> k_r {"k_r", 1};
    Input<int> taylor_s_l {"taylor_s_l"};
    Input<int> N_fft {"N_fft"};
    Input<double> delta_r {"delta_r"};
    Input<Buffer<double>> u {"u", 1};
    Input<Buffer<double>> v {"v", 1};
    Input<Buffer<float>> pos {"pos", 2};
    Input<Buffer<double>> r {"r", 2};

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
#if DEBUG_PRE_FFT
    Output<Buffer<double>> out_pre_fft{"out_pre_fft", 3};
#endif
#if DEBUG_POST_FFT
    Output<Buffer<double>> out_post_fft{"out_post_fft", 3};
#endif
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

    // 2-D complex data (3-D when handled as primitive data: {2, x, y})
    Output<Buffer<double>> output_img{"output_img", 3};

    // xs: {nu*nv, npulses}
    // lsa, lsb, lsn: implicit linspace parameters (min, max, count)
    // fp: {N_fft, npulses}
    // output: {nu*nv, npulses}
    inline Expr interp(Func xs, Expr lsa, Expr lsb, Expr lsn, ComplexFunc fp, Var c, Var x, Var y) {
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
        // some extents and related RDoms
        Expr nsamples = phs.dim(1).extent();
        Expr npulses = phs.dim(2).extent();
        Expr nu = u.dim(0).extent();
        Expr nv = v.dim(0).extent();
        Expr nd = pos.dim(0).extent(); // nd = 3 (spatial dimensions)
        rnpulses = RDom(0, npulses, "rnpulses");
        rnd = RDom(0, nd, "rnd");

        // Create window: produces shape {nsamples, npulses}
        win_x(x) = taylor(nsamples, taylor_s_l, x, "win_x");
        win_y(y) = taylor(npulses, taylor_s_l, y, "win_y");
        win(x, y) = win_x(x) * win_y(y);
#if DEBUG_WIN
        out_win(x, y) = win(x, y);
#endif

        // Filter phase history: produces shape {nsamples}
        filt(x) = abs(k_r(x));
#if DEBUG_FILT
        out_filt(x) = filt(x);
#endif

        // phs_filt: produces shape {nsamples, npulses}
        Func phs_func = phs;
        ComplexFunc phs_cmplx(c, phs_func);
        phs_filt(x, y) = phs_cmplx(x, y) * filt(x) * win(x, y);
#if DEBUG_PHS_FILT
        out_phs_filt(c, x, y) = phs_filt.inner(c, x, y);
#endif

        // Zero pad phase history: produces shape {N_fft, npulses}
        phs_pad(x, y) = pad(phs_filt, nsamples, npulses,
                            ComplexExpr(c, Expr(0.0), Expr(0.0)),
                            N_fft, npulses, c, x, y);
#if DEBUG_PHS_PAD
        out_phs_pad(c, x, y) = phs_pad.inner(c, x, y);
#endif

        // shift: produces shape {N_fft, npulses}
        fftsh(x, y) = fftshift(phs_pad, N_fft, npulses, x, y);
#if DEBUG_PRE_FFT
        out_pre_fft(c, x, y) = fftsh.inner(c, x, y);
#endif

        // dft: produces shape {N_fft, npulses}
        dft.inner.define_extern("call_dft", {fftsh.inner, N_fft}, Float(64), 3, NameMangling::C);
#if DEBUG_POST_FFT
        out_post_fft(c, x, y) = dft.inner(c, x, y);
#endif

        dft_out(x, y) = dft(x, y);

        // Q: produces shape {N_fft, npulses}
        Q(x, y) = fftshift(dft_out, N_fft, npulses, x, y);
#if DEBUG_Q
        out_Q(c, x, y) = Q.inner(c, x, y);
#endif

        // norm(r0): produces shape {npulses}
        norm_r0(x) = norm(pos(rnd, x));
#if DEBUG_NORM_R0
        out_norm_r0(x) = norm_r0(x);
#endif

        // r - r0: produces shape {nu*nv, nd, npulses}
        rr0(x, y, z) = r(x, y) - pos(y, z);
#if DEBUG_RR0
        out_rr0(x, y, z) = rr0(x, y, z);
#endif

        // norm(r - r0): produces shape {nu*nv, npulses}
        norm_rr0(x, y) = norm(rr0(x, rnd, y));
#if DEBUG_NORM_RR0
        out_norm_rr0(x, y) = norm_rr0(x, y);
#endif

        // dr_i: produces shape {nu*nv, npulses}
        dr_i(x, y) = norm_r0(y) - norm_rr0(x, y);
#if DEBUG_DR_I
        out_dr_i(x, y) = dr_i(x, y);
#endif

        // Q_hat: produces shape {nu*nv, npulses}
        Q_hat.inner(c, x, y) = interp(dr_i, floor(-nsamples * delta_r / 2), floor(nsamples * delta_r / 2), N_fft, Q, c, x, y);
#if DEBUG_Q_REAL
        out_q_real(x, y) = Q_hat.inner(0, x, y);
#endif
#if DEBUG_Q_IMAG
        out_q_imag(x, y) = Q_hat.inner(1, x, y);
#endif
#if DEBUG_Q_HAT
        out_q_hat(c, x, y) = Q_hat.inner(c, x, y);
#endif

        // k_c: produces scalar
        Expr k_c = k_r(nsamples / 2);

        // img: produces shape {nu*nv}
        img(x) = ComplexExpr(c, Expr(0.0), Expr(0.0));
        img(x) += Q_hat(x, rnpulses) * expj(c, -k_c * dr_i(x, rnpulses));
#if DEBUG_IMG
        out_img(c, x) = img.inner(c, x);
#endif

        // finally...
        Expr fdr_i = norm_r0(npulses / 2) - norm_rr0(x, npulses / 2);
        fimg(x) = img(x) * expj(c, k_c * fdr_i);
#if DEBUG_FIMG
        out_fimg(c, x) = fimg.inner(c, x);
#endif

        // output_img: produce shape {nu, nv}, but reverse row order
        output_img(c, x, y) = fimg.inner(c, (nu * (nv - y - 1)) + x);
    }

    void schedule() {
        Target tgt(target);
        if(auto_schedule) {
            std::cout << "setting size/scalar estimates for autoscheduler" << std::endl;
            phs.set_estimates({{0, 2}, {0, 424}, {0, 469}});
            k_r.set_estimates({{0, 424}});
            u.set_estimates({{0, 512}});
            v.set_estimates({{0, 512}});
            pos.set_estimates({{0, 3}, {0, 469}});
            r.set_estimates({{0, 262144}, {0, 3}});
            output_img.set_estimates({{0, 2}, {0, 512}, {0, 512}});
            delta_r.set_estimate(0.240851);
            N_fft.set_estimate(1024);
            taylor_s_l.set_estimate(17);
            fftsh.inner.compute_root(); // helps the Mullapudi2016 autoscheduler pass full vectors of input to DFT
        } else if(tgt.has_gpu_feature()) {
            // GPU target
            std::cout << "Scheduling for GPU: " << tgt << std::endl
                      << "Block size: " << blocksize.value() << std::endl
                      << "Vector size: " << vectorsize.value() << std::endl;
            Var block{"block"}, thread{"thread"};
            win_x.compute_root();
            win_y.compute_root();
            win.compute_root();
            filt.compute_root();
            phs_filt.inner.compute_root();
            phs_pad.inner.compute_root();
            fftsh.inner.compute_root();
            dft.inner.compute_root();
            dft_out.inner.compute_root();
            Q.inner.compute_root();
            norm_r0.compute_root();
            rr0.compute_inline();
            norm_rr0.compute_root().gpu_tile(x, block, thread, blocksize);
            dr_i.compute_inline();
            Q_hat.inner.compute_inline();
            img.inner.compute_root().gpu_tile(x, block, thread, blocksize);
            img.inner.update(0).gpu_tile(x, block, thread, blocksize);
            fimg.inner.compute_root().gpu_tile(x, block, thread, blocksize);
            output_img.compute_root().vectorize(x, vectorsize).parallel(y);
            if (print_loop_nest) {
                output_img.print_loop_nest();
            }
        } else {
            // CPU target
            std::cout << "Scheduling for CPU: " << tgt << std::endl
                      << "Block size: " << blocksize.value() << std::endl
                      << "Vector size: " << vectorsize.value() << std::endl;
            Var xi{"xi"}, xo{"xo"};
            win_x.compute_root();
            win_y.compute_root();
            win.compute_root();
            filt.compute_root();
            phs_filt.inner.compute_root();
            phs_pad.inner.compute_root();
            fftsh.inner.compute_root();
            dft.inner.compute_root();
            dft_out.inner.compute_inline();
            Q.inner.compute_root();
            norm_r0.compute_root();
            norm_rr0.compute_root().parallel(y).vectorize(x, vectorsize);
            dr_i.compute_inline();
            Q_hat.inner.compute_inline();
            img.inner.compute_root().unroll(c).split(x, xo, xi, blocksize).vectorize(xi, vectorsize).parallel(xo, blocksize);
            img.inner.update(0).reorder(c, rnpulses, x).unroll(c).parallel(x, blocksize);
            fimg.inner.in(output_img).compute_inline();
            output_img.compute_root().parallel(y).vectorize(x, vectorsize);
            if (print_loop_nest) {
                output_img.print_loop_nest();
            }
        }
    }

private:
    Var c{"c"}, x{"x"}, y{"y"}, z{"z"};

    Func win_x{"win_x"};
    Func win_y{"win_y"};
    Func win{"win"};
    Func filt{"filt"};
    ComplexFunc phs_filt{c, "phs_filt"};
    ComplexFunc phs_pad{c, "phs_pad"};
    ComplexFunc fftsh{c, "fftshift"};
    ComplexFunc dft{c, "dft"};
    ComplexFunc dft_out{c, "dft_out"};
    ComplexFunc Q{c, "Q"};
    Func norm_r0{"norm_r0"};
    Func rr0{"rr0"};
    Func norm_rr0{"norm_rr0"};
    Func dr_i{"dr_i"};
    ComplexFunc Q_hat{c, "Q_hat"};
    ComplexFunc img{c, "img"};
    ComplexFunc fimg{c, "fimg"};
    RDom rnpulses;
    RDom rnd;
};

HALIDE_REGISTER_GENERATOR(BackprojectionGenerator, backprojection)
HALIDE_REGISTER_GENERATOR(BackprojectionGenerator, backprojection_cuda)
