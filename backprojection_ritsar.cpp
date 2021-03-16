#include <Halide.h>

#include "halide_complexfunc.h"
#include "signal.h"
#include "signal_complex.h"
#include "util.h"

using namespace Halide;
using namespace Halide::Tools;

class BackprojectionRitsarGenerator : public Halide::Generator<BackprojectionRitsarGenerator> {
public:
    enum class Schedule { Serial,
                          Vectorize,
                          Parallel,
                          VectorizeParallel };
    GeneratorParam<Schedule> sched {"schedule",
                                    Schedule::Vectorize, // closest to RITSAR
                                    {{"s", Schedule::Serial},
                                     {"v", Schedule::Vectorize},
                                     {"p", Schedule::Parallel},
                                     {"vp", Schedule::VectorizeParallel}}};
    GeneratorParam<int32_t> vectorsize {"vectorsize", 4};

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

    // 2-D complex data (3-D when handled as primitive data: {2, x, y})
    Output<Buffer<double>> output_img{"output_img", 3};

    // xs: {nu*nv, npulses}
    // lsa, lsb, lsn: implicit linspace parameters (min, max, count)
    // fp: {N_fft, npulses}
    // output: {nu*nv, npulses}
    inline Expr interp(Func xs, Expr lsa, Expr lsb, Expr lsn, ComplexFunc fp, Expr c, Var x, Var y) {
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
        RDom rnpulses(0, npulses, "rnpulses");
        RDom rnd(0, nd, "rnd");

        // Create window: produces shape {nsamples, npulses}
        win_x(x) = taylor(nsamples, taylor_s_l, x, "win_x");
        win_y(y) = taylor(npulses, taylor_s_l, y, "win_y");
        win(x, y) = win_x(x) * win_y(y);

        // Filter phase history: produces shape {nsamples}
        filt(x) = abs(k_r(x));

        // phs_filt: produces shape {nsamples, npulses}
        Func phs_func = phs;
        ComplexFunc phs_cmplx(c, phs_func);
        phs_filt(x, y) = phs_cmplx(x, y) * filt(x) * win(x, y);

        // Zero pad phase history: produces shape {N_fft, npulses}
        phs_pad(x, y) = pad(phs_filt, nsamples, npulses,
                            ComplexExpr(c, Expr(0.0), Expr(0.0)),
                            N_fft, npulses, c, x, y);

        // shift: produces shape {N_fft, npulses}
        fftsh(x, y) = fftshift(phs_pad, N_fft, npulses, x, y);

        // dft: produces shape {N_fft, npulses}
        dft.inner.define_extern("call_dft", {fftsh.inner, N_fft}, Float(64), {c, x, y});

        // Q: produces shape {N_fft, npulses}
        Q(x, y) = fftshift(dft, N_fft, npulses, x, y);

        // norm(r0): produces shape {npulses}
        norm_r0(x) = norm(pos(rnd, x), "norm_r0_norm");

        // r - r0: produces shape {nu*nv, nd, npulses}
        rr0(x, y, z) = r(x, y) - pos(y, z);

        // norm(r - r0): produces shape {nu*nv, npulses}
        norm_rr0(x, y) = norm(rr0(x, rnd, y), "norm_rr0_norm");

        // dr_i: produces shape {nu*nv, npulses}
        dr_i(x, y) = norm_r0(y) - norm_rr0(x, y);

        // Q_{real,imag,hat}: produce shape {nu*nv, npulses}
        Q_real(x, y) = interp(dr_i, floor(-nsamples * delta_r / 2), floor(nsamples * delta_r / 2), N_fft, Q, 0, x, y);
        Q_imag(x, y) = interp(dr_i, floor(-nsamples * delta_r / 2), floor(nsamples * delta_r / 2), N_fft, Q, 1, x, y);
        Q_hat(x, y) = ComplexExpr(c, Q_real(x, y), Q_imag(x, y));

        // k_c: produces scalar
        Expr k_c = k_r(nsamples / 2);

        // img: produces shape {nu*nv}
        img(x) = ComplexExpr(c, Expr(0.0), Expr(0.0));
        img(x) += Q_hat(x, rnpulses) * exp(ComplexExpr(c, Expr(0.0), Expr(-1.0)) * k_c * dr_i(x, rnpulses));

        // finally...
        Expr fdr_i = norm_r0(npulses / 2) - norm_rr0(x, npulses / 2);
        fimg(x) = img(x) * exp(ComplexExpr(c, Expr(0.0), Expr(1.0)) * k_c * fdr_i);

        // output_img: produce shape {nu, nv}, but reverse row order
        output_img(c, x, y) = fimg.inner(c, (nu * (nv - y - 1)) + x);
    }

    void schedule() {
        switch (sched) {
        case Schedule::Serial:
            win_x.compute_root();
            win_y.compute_root();
            win.compute_root();
            filt.compute_root();
            phs_filt.inner.compute_root();
            phs_pad.inner.compute_root();
            fftsh.inner.compute_root();
            dft.inner.compute_root();
            Q.inner.compute_root();
            norm_r0.compute_root();
            rr0.compute_root();
            norm_rr0.compute_root();
            dr_i.compute_root();
            Q_real.compute_root();
            Q_imag.compute_root();
            Q_hat.inner.compute_root();
            img.inner.compute_root();
            fimg.inner.compute_root();
            output_img.compute_root();
            break;
        case Schedule::Vectorize:
            win_x.compute_root().vectorize(x, vectorsize);
            win_y.compute_root().vectorize(y, vectorsize);
            win.compute_root().vectorize(x, vectorsize);
            filt.compute_root().vectorize(x, vectorsize);
            phs_filt.inner.compute_root().vectorize(x, vectorsize);
            phs_pad.inner.compute_root().vectorize(x, vectorsize);
            fftsh.inner.compute_root().vectorize(x, vectorsize);
            // cannot vectorize extern func from here, but func's impl can
            dft.inner.compute_root();
            Q.inner.compute_root().vectorize(x, vectorsize);
            norm_r0.compute_root().vectorize(x, vectorsize);
            rr0.compute_root().vectorize(x, vectorsize);
            norm_rr0.compute_root().vectorize(x, vectorsize);
            dr_i.compute_root().vectorize(x, vectorsize);
            Q_real.compute_root().vectorize(x, vectorsize);
            Q_imag.compute_root().vectorize(x, vectorsize);
            Q_hat.inner.compute_root().vectorize(x, vectorsize);
            img.inner.compute_root().vectorize(x, vectorsize);
            fimg.inner.compute_root().vectorize(x, vectorsize);
            output_img.compute_root().vectorize(x, vectorsize);
            break;
        case Schedule::Parallel:
            // TODO: can win_x and win_y be parallelized?
            win_x.compute_root();
            win_y.compute_root();
            win.compute_root().parallel(y);
            filt.compute_root().parallel(x);
            phs_filt.inner.compute_root().parallel(y);
            phs_pad.inner.compute_root().parallel(y);
            fftsh.inner.compute_root().parallel(y);
            dft.inner.compute_root().parallel(y);
            Q.inner.compute_root().parallel(y);
            norm_r0.compute_root().parallel(x);
            rr0.compute_root().parallel(z);
            norm_rr0.compute_root().parallel(y);
            dr_i.compute_root().parallel(y);
            Q_real.compute_root().parallel(y);
            Q_imag.compute_root().parallel(y);
            Q_hat.inner.compute_root().parallel(y);
            img.inner.compute_root().parallel(x);
            img.inner.update(0).parallel(x);
            fimg.inner.compute_root().parallel(x);
            output_img.compute_root().parallel(y);
            break;
        case Schedule::VectorizeParallel:
            // TODO: can win_x and win_y be parallelized?
            win_x.compute_root().vectorize(x, vectorsize);
            win_y.compute_root().vectorize(y, vectorsize);
            win.compute_root().vectorize(x, vectorsize).parallel(y);
            filt.compute_root().vectorize(x, vectorsize).parallel(x);
            phs_filt.inner.compute_root().vectorize(x, vectorsize).parallel(y);
            phs_pad.inner.compute_root().vectorize(x, vectorsize).parallel(y);
            fftsh.inner.compute_root().vectorize(x, vectorsize).parallel(y);
            // cannot vectorize extern func from here, but func's impl can
            dft.inner.compute_root().parallel(y);
            Q.inner.compute_root().vectorize(x, vectorsize).parallel(y);
            norm_r0.compute_root().vectorize(x, vectorsize).parallel(x);
            rr0.compute_root().vectorize(x, vectorsize).parallel(z);
            norm_rr0.compute_root().vectorize(x, vectorsize).parallel(y);
            dr_i.compute_root().vectorize(x, vectorsize).parallel(y);
            Q_real.compute_root().vectorize(x, vectorsize).parallel(y);
            Q_imag.compute_root().vectorize(x, vectorsize).parallel(y);
            Q_hat.inner.compute_root().vectorize(x, vectorsize).parallel(y);
            img.inner.compute_root().vectorize(x, vectorsize).parallel(x);
            img.inner.update(0).vectorize(x, vectorsize).parallel(x);
            fimg.inner.compute_root().vectorize(x, vectorsize).parallel(x);
            output_img.compute_root().vectorize(x, vectorsize).parallel(y);
            break;
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
    ComplexFunc Q{c, "Q"};
    Func norm_r0{"norm_r0"};
    Func rr0{"rr0"};
    Func norm_rr0{"norm_rr0"};
    Func dr_i{"dr_i"};
    Func Q_real{"Q_real"};
    Func Q_imag{"Q_imag"};
    ComplexFunc Q_hat{c, "Q_hat"};
    ComplexFunc img{c, "img"};
    ComplexFunc fimg{c, "fimg"};
};

HALIDE_REGISTER_GENERATOR(BackprojectionRitsarGenerator, backprojection_ritsar)
