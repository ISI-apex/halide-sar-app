#include <stdio.h>

#include <Halide.h>

#include "complexfunc.h"
#include "signal.h"
#include "signal_complex.h"
#include "util.h"

using namespace Halide;

#define TAYLOR_S_L 17

class BackprojectionPreFFTGenerator : public Halide::Generator<BackprojectionPreFFTGenerator> {
public:
    Input<Buffer<float>> phs {"phs", 3}; // complex 2d input  (Halide thinks this is 3d: [2, x, y])
    Input<Buffer<float>> k_r {"k_r", 1};
    Input<int> N_fft {"N_fft"};

    Output<Buffer<float>> output_buffer{"output_packed", 3}; // complex 2d output (Halide thinks this is 3d: [2, x, y])

    Var c{"c"}, x{"x"}, y{"y"};

    void generate() {
        Expr nsamples("nsamples");
        nsamples = phs.dim(1).extent();
        Expr npulses("npulses");
        npulses = phs.dim(2).extent();
        Func phs_func = phs;
        ComplexFunc input(c, phs_func, "input");

        // Create window
        Func win_x("win_x");
        win_x = taylor_func(nsamples, TAYLOR_S_L);
        Func win_y("win_y");
        win_y = taylor_func(npulses, TAYLOR_S_L);
        Func win("win");
        win(x, y) = win_x(x) * win_y(y);

        // Filter phase history
        Func filt("filt");
        filt(x) = abs(k_r(x));
        ComplexFunc phs_filt(c, "phs_filt");
        phs_filt(x, y) = input(x, y) * filt(x) * win(x, y);

        // Zero pad phase history
        ComplexFunc phs_pad(c, "phs_pad");
        phs_pad = pad_func(phs_filt, nsamples, npulses, N_fft, npulses);

        // shift
        ComplexFunc fftshift(c, "fftshift");
        fftshift = fftshift_func(phs_pad, N_fft, npulses);

        output_buffer(c, x, y) = fftshift.inner(c, x, y);

        phs_func.compute_root();
        win_x.compute_root();
        win_y.compute_root();
        win.compute_root();
        filt.compute_root();
        phs_filt.inner.compute_root();
        phs_pad.inner.compute_root();
        fftshift.inner.compute_root();
    }
};

class BackprojectionPostFFTGenerator : public Halide::Generator<BackprojectionPostFFTGenerator> {
public:
    Input<Buffer<float>> in {"in", 3};
    Input<int> nsamples {"nsamples"};
    Input<double> delta_r {"delta_r"};
    Input<Buffer<float>> k_r {"k_r", 1};
    Input<Buffer<double>> u {"u", 1};
    Input<Buffer<double>> v {"v", 1};
    Input<Buffer<float>> pos {"pos", 2};
    Input<Buffer<double>> r {"r", 2};

    Output<Buffer<float>> output_buffer{"output_packed", 3};

    Var c{"c"}, x{"x"}, y{"y"}, z{"z"};

    // xs: {nu*nv, npulses}
    // xp: {N_fft}
    // fp: {N_fft, npulses}
    // output: {nu*nv, npulses}
    inline Expr interp(Func xs, Func xp, ComplexFunc fp, Expr c, Expr extent) {
        RDom r(0, extent, "r");
        // index lookups into xp
        Expr lutl("lutl"); // lower index
        Expr lutu("lutu"); // upper index
        // last index in xp where xp(r) < xs(x, y): shape = {extent}
        lutl = max(argmax(r, xp(r) >= xs(x, y))[0] - 1, 0);
        // first index in xp where xp(r) >= xs(x, y): shape = {extent}
        lutu = argmax(r, xp(r) >= xs(x, y))[0];
        // Halide complains if we don't clamp
        Expr cll = clamp(lutl, 0, extent - 1);
        Expr clu = clamp(lutu, 0, extent - 1);
        Expr interp("interp");
        // Can avoid these first two selects if we enforce that xs values are in xp's value range
        interp =
            select(xs(x, y) < xp(0), ConciseCasts::f64(fp.inner(c, 0, y)),
                   select(xs(x, y) > xp(extent - 1), ConciseCasts::f64(fp.inner(c, extent - 1, y)),
                          select(cll == clu,
                                 ConciseCasts::f64(fp.inner(c, cll, y)),
                                 lerp(ConciseCasts::f64(fp.inner(c, cll, y)),
                                      ConciseCasts::f64(fp.inner(c, clu, y)),
                                      ConciseCasts::f64((xs(x, y) - xp(cll)) / (xp(clu) - xp(cll)))))));
        return interp;
    }

    void generate() {
        // inputs as functions
        Func in_func("in_func");
        in_func = in;
        ComplexFunc input(c, in_func, "input");
        Func pos_func("pos_func");
        pos_func = pos;
        Func r_func("r_func");
        r_func = r;

        // some extents and related RDoms
        Expr N_fft = in.dim(1).extent();
        Expr npulses = in.dim(2).extent();
        Expr nu = u.dim(0).extent();
        Expr nv = v.dim(0).extent();
        Expr nd = pos.dim(0).extent(); // nd = 3 (spatial dimensions)
        RDom rnfft(0, N_fft, "rnfft");
        RDom rnpulses(0, npulses, "rnpulses");
        RDom rnd(0, nd, "rnd");

        // k_c_im: produces scalar
        ComplexExpr k_c_im(c, Expr(0.0f), k_r(nsamples / 2));

        // Q: produces shape {N_fft, npulses}
        ComplexFunc Q(c, "Q");
        Q = fftshift_func(input, N_fft, npulses);

        // dr: produces shape {N_fft}
        Func dr("dr");
        dr = linspace_func(floor(-nsamples * delta_r / 2), floor(nsamples * delta_r / 2), rnfft);

        // norm(r0): produces shape {npulses}
        Func norm_r0("norm_r0");
        norm_r0(x) = norm_expr(pos_func(rnd, x));

        // r - r0: produces shape {nu*nv, nd, npulses}
        Func rr0("rr0");
        rr0(x, y, z) = r_func(x, y) - pos_func(y, z);

        // norm(r - r0): produces shape {nu*nv, npulses}
        Func norm_rr0("norm_rr0");
        norm_rr0(x, y) = norm_expr(rr0(x, rnd, y));

        // dr_i: produces shape {nu*nv, npulses}
        Func dr_i("dr_i");
        dr_i(x, y) = norm_r0(y) - norm_rr0(x, y);

        // Q_{real,imag,hat}: produce shape {nu*nv, npulses}
        Func Q_real("Q_real");
        Func Q_imag("Q_imag");
        ComplexFunc Q_hat(c, "Q_hat");
        Q_real(x, y) = interp(dr_i, dr, Q, 0, N_fft);
        Q_imag(x, y) = interp(dr_i, dr, Q, 1, N_fft);
        Q_hat(x, y) = ComplexExpr(c, Q_real(x, y), Q_imag(x, y));

        // img: produces shape {nu*nv}
        ComplexFunc img(c, "img");
        img(x) = sum(Q_hat(x, rnpulses) * exp(k_c_im * dr_i(x, rnpulses)));

        // finally...
        Expr fdr_i = norm_r0(npulses / 2) - norm_rr0(x, npulses / 2);
        ComplexFunc fimg(c, "fimg");
        fimg(x) = img(x) * exp(k_c_im * fdr_i);

        // img_rect: produce shape {nu, nv}, but reverse row order
        ComplexFunc img_rect(c, "img_rect");
        img_rect(x, y) = fimg((nu * (nv - y - 1)) + x);
        output_buffer(c, x, y) = ConciseCasts::f32(img_rect.inner(c, x, y));

        in_func.compute_root();
        dr.compute_root();
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
        img_rect.inner.compute_root();
    }
};

HALIDE_REGISTER_GENERATOR(BackprojectionPreFFTGenerator, backprojection_pre_fft)
HALIDE_REGISTER_GENERATOR(BackprojectionPostFFTGenerator, backprojection_post_fft)
