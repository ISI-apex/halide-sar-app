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
        win_sample(sample) = taylor(nsamples, taylor_s_l, sample, "win_sample");
        win_pulse(pulse) = taylor(npulses, taylor_s_l, pulse, "win_pulse");
        win(sample, pulse) = win_sample(sample) * win_pulse(pulse);
#if DEBUG_WIN
        out_win(sample, pulse) = win(sample, pulse);
#endif

        // Filter phase history: produces shape {nsamples}
        filt(x) = abs(k_r(x));
#if DEBUG_FILT
        out_filt(x) = filt(x);
#endif

        // phs_filt: produces shape {nsamples, npulses}
        Func phs_func = phs;
        ComplexFunc phs_cmplx(c, phs_func);
        phs_filt(x, pulse) = phs_cmplx(x, pulse) * filt(x) * win(x, pulse);
#if DEBUG_PHS_FILT
        out_phs_filt(c, x, pulse) = phs_filt.inner(c, x, pulse);
#endif

        // Zero pad phase history: produces shape {N_fft, npulses}
        phs_pad(sample, pulse) = pad(phs_filt, nsamples, npulses,
                            ComplexExpr(c, Expr(0.0), Expr(0.0)),
                            N_fft, npulses, c, sample, pulse);
#if DEBUG_PHS_PAD
        out_phs_pad(c, sample, pulse) = phs_pad.inner(c, sample, pulse);
#endif

        // shift: produces shape {N_fft, npulses}
        fftsh(sample, pulse) = fftshift(phs_pad, N_fft, npulses, sample, pulse);
#if DEBUG_PRE_FFT
        out_pre_fft(c, sample, pulse) = fftsh.inner(c, sample, pulse);
#endif

        // dft: produces shape {N_fft, npulses}
        dft.inner.define_extern("call_dft", {fftsh.inner, N_fft}, Float(64), {c, sample, pulse});
#if DEBUG_POST_FFT
        out_post_fft(c, sample, pulse) = dft.inner(c, sample, pulse);
#endif

        // Q: produces shape {N_fft, npulses}
        Q(sample, pulse) = fftshift(dft, N_fft, npulses, sample, pulse);
#if DEBUG_Q
        out_Q(c, sample, pulse) = Q.inner(c, sample, pulse);
#endif

        // norm(r0): produces shape {npulses}
        norm_r0(pulse) = norm(pos(rnd, pulse));
#if DEBUG_NORM_R0
        out_norm_r0(pulse) = norm_r0(pulse);
#endif

        // r - r0: produces shape {nu*nv, nd, npulses}
        rr0(pixel, dim, pulse) = r(pixel, dim) - pos(dim, pulse);
#if DEBUG_RR0
        out_rr0(pixel, dim, pulse) = rr0(pixel, dim, pulse);
#endif

        // norm(r - r0): produces shape {nu*nv, npulses}
        norm_rr0(pixel, pulse) = norm(rr0(pixel, rnd, pulse));
#if DEBUG_NORM_RR0
        out_norm_rr0(pixel, pulse) = norm_rr0(pixel, pulse);
#endif

        // dr_i: produces shape {nu*nv, npulses}
        dr_i(pixel, pulse) = norm_r0(pulse) - norm_rr0(pixel, pulse);
#if DEBUG_DR_I
        out_dr_i(pixel, pulse) = dr_i(pixel, pulse);
#endif

        // Q_hat: produces shape {nu*nv, npulses}
        Q_hat.inner(c, pixel, pulse) = interp(dr_i, floor(-nsamples * delta_r / 2), floor(nsamples * delta_r / 2), N_fft, Q, c, pixel, pulse);
#if DEBUG_Q_REAL
        out_q_real(pixel, pulse) = Q_hat.inner(0, pixel, pulse);
#endif
#if DEBUG_Q_IMAG
        out_q_imag(pixel, pulse) = Q_hat.inner(1, pixel, pulse);
#endif
#if DEBUG_Q_HAT
        out_q_hat(c, pixel, pulse) = Q_hat.inner(c, pixel, pulse);
#endif

        // k_c: produces scalar
        Expr k_c = k_r(nsamples / 2);

        // img: produces shape {nu*nv}
        img(pixel) = ComplexExpr(c, Expr(0.0), Expr(0.0));
        img(pixel) += Q_hat(pixel, rnpulses) * expj(c, -k_c * dr_i(pixel, rnpulses));
#if DEBUG_IMG
        out_img(c, pixel) = img.inner(c, pixel);
#endif

        // finally...
        fimg(pixel) = img(pixel) * expj(c, k_c * dr_i(pixel, npulses / 2));
#if DEBUG_FIMG
        out_fimg(c, pixel) = fimg.inner(c, pixel);
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
                      << "Block size: " << blocksize.value() << std::endl;
            Var pixeli{"pixeli"}, block{"block"};
            win_sample.compute_root();
            win_pulse.compute_root();
            win.compute_root();
            filt.compute_root();
            phs_filt.inner.compute_root();
            phs_pad.inner.compute_root();
            fftsh.inner.compute_root();
            dft.inner.compute_root();
            Q.inner.compute_root();
            norm_r0.compute_root().vectorize(pulse, vectorsize);
            rr0.compute_inline();
            norm_rr0.compute_inline();
            dr_i.compute_inline();
            Q_hat.inner.compute_inline();
            img.inner.compute_root().bound(c, 0, 2).unroll(c).gpu_tile(pixel, block, pixeli, blocksize);
            img.inner.update(0).reorder(c, rnpulses, pixel).gpu_tile(pixel, block, pixeli, blocksize);
            fimg.inner.compute_root().bound(c, 0, 2).unroll(c).gpu_tile(pixel, block, pixeli, blocksize);
            output_img.compute_root().bound(c, 0, 2).unroll(c).parallel(y).vectorize(x, vectorsize);
            if (print_loop_nest) {
                output_img.print_loop_nest();
            }
        } else {
            // CPU target
            std::cout << "Scheduling for CPU: " << tgt << std::endl
                      << "Block size: " << blocksize.value() << std::endl
                      << "Vector size: " << vectorsize.value() << std::endl;
            Var pixeli{"pixeli"}, block{"block"};
            win_sample.compute_root();
            win_pulse.compute_root();
            win.compute_root();
            filt.compute_root();
            phs_filt.inner.compute_root();
            phs_pad.inner.compute_root();
            fftsh.inner.compute_root();
            dft.inner.compute_root().parallel(pulse);
            Q.inner.compute_root();
            norm_r0.compute_root().vectorize(pulse, vectorsize);
            rr0.compute_inline();
            Func dr_i_in_fimg = dr_i.clone_in(fimg.inner).compute_inline();
            norm_rr0.clone_in(dr_i_in_fimg).compute_inline();
            norm_rr0.reorder(pulse, pixel).compute_at(img.inner, pixel).store_at(img.inner, pixel).vectorize(pulse, vectorsize);
            dr_i.reorder(pulse, pixel).reorder(pixel, pulse).compute_at(img.inner, pixel).store_at(img.inner, pixel).vectorize(pulse, vectorsize);
            Q_hat.inner.compute_at(img.inner, pixel).store_at(img.inner, pixel).vectorize(pixel, vectorsize);
            img.inner.compute_root().unroll(c).split(pixel, block, pixeli, blocksize).vectorize(pixeli, vectorsize).parallel(block, blocksize);
            img.inner.update(0).reorder(c, rnpulses, pixel).unroll(c).parallel(pixel, blocksize);
            fimg.inner.compute_root().bound(c, 0, 2).unroll(c).split(pixel, block, pixeli, blocksize).vectorize(pixeli, vectorsize).parallel(block, blocksize);
            output_img.compute_root().bound(c, 0, 2).unroll(c).parallel(y).vectorize(x, vectorsize);
            if (print_loop_nest) {
                output_img.print_loop_nest();
            }
        }
    }

private:
    Var x{"x"}, y{"y"};
    Var c{"c"}, sample{"sample"}, pixel{"pixel"}, pulse{"pulse"}, dim{"dim"};

    Func win_sample{"win_sample"};
    Func win_pulse{"win_pulse"};
    Func win{"win"};
    Func filt{"filt"};
    ComplexFunc phs_filt{c, "phs_filt"};
    ComplexFunc phs_pad{c, "phs_pad"};
    ComplexFunc fftsh{c, "fftshift"};
    ComplexFunc dft{c, "dft"};
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
