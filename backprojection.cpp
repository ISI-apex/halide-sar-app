#include <Halide.h>

#include "halide_complexfunc.h"
#include "signal.h"
#include "signal_complex.h"
#include "util.h"

using namespace Halide;
using namespace Halide::Tools;

class BackprojectionGenerator : public Halide::Generator<BackprojectionGenerator> {
public:
    enum class Schedule { CPU,
                          GPU,
                          CPUDistributed,
                          GPUDistributed,
                        };
    GeneratorParam<Schedule> sched {"schedule",
                                    Schedule::CPU,
                                    {{"cpu", Schedule::CPU},
                                     {"gpu", Schedule::GPU},
#if defined(WITH_DISTRIBUTE)
                                     {"cpu-distributed", Schedule::CPUDistributed},
                                     {"gpu-distributed", Schedule::GPUDistributed},
#endif // WITH_DISTRIBUTE
                                    }
                                   };
    GeneratorParam<int32_t> vectorsize {"vectorsize", 4};
    GeneratorParam<int32_t> blocksize {"blocksize", 64};
    GeneratorParam<int32_t> blocksize_gpu_tile {"blocksize_gpu_tile", 64};
    GeneratorParam<int32_t> blocksize_gpu_split_x {"blocksize_gpu_split_x", 64};
    GeneratorParam<bool> print_loop_nest {"print_loop_nest", false};

    // 2-D complex data (3-D when handled as primitive data: {2, x, y})
    Input<Buffer<float>> phs {"phs", 3};
    Input<Buffer<float>> k_r {"k_r", 1};
    Input<int> taylor_s_l {"taylor_s_l"};
    Input<int> N_fft {"N_fft"};
    Input<double> delta_r {"delta_r"};
    Input<Buffer<double>> u {"u", 1};
    Input<Buffer<double>> v {"v", 1};
    Input<Buffer<float>> pos_in {"pos", 2};
    Input<Buffer<double>> r_in {"r", 2};

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
        Expr nd = pos_in.dim(0).extent(); // nd = 3 (spatial dimensions)
        rnpulses = RDom(0, npulses, "rnpulses");
        rnd = RDom(0, nd, "rnd");

        // Boundary conditions
        pos = BoundaryConditions::constant_exterior(pos_in, Expr(0.0f));
        r = BoundaryConditions::constant_exterior(r_in, Expr(0.0));

        // Create window: produces f64, shape {nsamples, npulses}
        win_sample = Taylor(nsamples, taylor_s_l, sample, "win_sample");
        win_pulse = Taylor(npulses, taylor_s_l, pulse, "win_pulse");
        win(sample, pulse) = win_sample.taylor(sample) * win_pulse.taylor(pulse);

        // Filter phase history: produces f32, shape {nsamples}
        filt(sample) = abs(k_r(sample));

        // phs_filt: produces complex f64, shape {nsamples, npulses}
        Func phs_func = phs;
        ComplexFunc phs_cmplx(c, phs_func);
        phs_filt(sample, pulse) = phs_cmplx(sample, pulse) * filt(sample) * win(sample, pulse);

        // Zero pad phase history: produces complex f64, shape {N_fft, npulses}
        phs_pad(sample, pulse) = pad(phs_filt, nsamples, npulses,
                                     ComplexExpr(c, Expr(0.0), Expr(0.0)),
                                     N_fft, npulses, c, sample, pulse);

        // shift: produces complex f64, shape {N_fft, npulses}
        fftsh(sample, pulse) = fftshift(phs_pad, N_fft, npulses, sample, pulse);

        // dft: produces f64, shape {N_fft, npulses}
        dft.inner.define_extern("call_dft", {fftsh.inner, N_fft}, Float(64), {c, sample, pulse});

        // Q: produces complex f64, shape {N_fft, npulses}
        Q(sample, pulse) = fftshift(dft, N_fft, npulses, sample, pulse);

        // norm(r0): produces f64, shape {npulses}
        // f32 in RITSAR, but f64 produces better image with negligible performance impact
        norm_r0(pulse) = Expr(0.0);
        norm_r0(pulse) += pos(rnd, pulse) * pos(rnd, pulse);
        norm_r0(pulse) = sqrt(norm_r0(pulse));

        // r - r0: produces f64, shape {nu*nv, nd, npulses}
        rr0(pixel, dim, pulse) = r(pixel, dim) - pos(dim, pulse);

        // norm(r - r0): produces f64, shape {nu*nv, npulses}
        norm_rr0(pixel, pulse) = Expr(0.0);
        norm_rr0(pixel, pulse) += rr0(pixel, rnd, pulse) * rr0(pixel, rnd, pulse);
        norm_rr0(pixel, pulse) = sqrt(norm_rr0(pixel, pulse));

        // dr_i: produces f64, shape {nu*nv, npulses}
        dr_i(pixel, pulse) = norm_r0(pulse) - norm_rr0(pixel, pulse);

        // Q_hat: produces complex f64, shape {nu*nv, npulses}
        Q_hat.inner(c, pixel, pulse) = interp(dr_i, floor(-nsamples * delta_r / 2), floor(nsamples * delta_r / 2), N_fft, Q, c, pixel, pulse);

        // k_c: produces f32 scalar
        Expr k_c = k_r(nsamples / 2);

        // img: produces complex f64, shape {nu*nv}
        img(pixel) = ComplexExpr(c, Expr(0.0), Expr(0.0));
        img(pixel) += Q_hat(pixel, rnpulses) * expj(c, -k_c * dr_i(pixel, rnpulses));

        // mapping from image (x,y) coordinates to elements in the per-pixel vectors
        Expr xy_to_pixel = (nu * (nv - y - 1)) + x;

        // finally...
        // fimg: produces complex f64, shape {nu, nv}
        // Updates img in RITSAR, but we need a separate Func which is also easier to schedule
        fimg(x, y) = img(xy_to_pixel) * expj(c, k_c * dr_i(xy_to_pixel, npulses / 2));

        // output_img: produce complex f64, shape {nu, nv}, but reverse row order
        output_img(c, x, y) = fimg.inner(c, x, y);
    }

    void schedule() {
        Target tgt(target);
        if (auto_schedule) {
            std::cout << "setting size/scalar estimates for autoscheduler" << std::endl;
            // Dimension sizes based on Sandia dataset
            phs.set_estimates({{0, 2}, {0, 1800}, {0, 1999}});
            k_r.set_estimates({{0, 1800}});
            u.set_estimates({{0, 2048}});
            v.set_estimates({{0, 2048}});
            pos_in.set_estimates({{0, 3}, {0, 1999}});
            r_in.set_estimates({{0, 4194304}, {0, 3}});
            output_img.set_estimates({{0, 2}, {0, 2048}, {0, 2048}});
            delta_r.set_estimate(0.539505);
            N_fft.set_estimate(4096);
            taylor_s_l.set_estimate(30);
            return;
        }
        Var sample_vo{"sample_vo"}, sample_vi{"sample_vi"};
        Var pulse_vo{"pulse_vo"}, pulse_vi{"pulse_vi"};
        Var x_vo{"x_vo"}, x_vi{"x_vi"};
        switch (sched) {
        case Schedule::GPU:
#if defined(WITH_DISTRIBUTE)
        case Schedule::GPUDistributed:
#endif // WITH_DISTRIBUTE
            if (!tgt.has_gpu_feature()) {
                throw std::runtime_error("GPU schedules require GPU feature");
            }
            // GPU target
            std::cout << "Scheduling for GPU: " << tgt << std::endl
                      << "Vector size: " << vectorsize.value() << std::endl
                      << "Block size: " << blocksize.value() << std::endl
                      << "Block size GPU tile: " << blocksize_gpu_tile.value() << std::endl
                      << "Block size GPU split x: " << blocksize_gpu_split_x.value() << std::endl;
            win_sample.taylor.compute_root()
                             .vectorize(sample, vectorsize)
                             .parallel(sample, blocksize);
            win_sample.w.compute_root()
                        .split(sample, sample_vo, sample_vi, vectorsize)
                        .vectorize(sample_vi)
                        .parallel(sample_vo);
            win_sample.w.update(0)
                        .split(sample, sample_vo, sample_vi, vectorsize, TailStrategy::GuardWithIf)
                        .vectorize(sample_vi)
                        .parallel(sample_vo);
            win_pulse.taylor.compute_root()
                            .vectorize(pulse, vectorsize)
                            .parallel(pulse, blocksize);
            win_pulse.w.compute_root()
                       .split(pulse, pulse_vo, pulse_vi, vectorsize)
                       .vectorize(pulse_vi)
                       .parallel(pulse_vo);
            win_pulse.w.update(0)
                       .split(pulse, pulse_vo, pulse_vi, vectorsize, TailStrategy::GuardWithIf)
                       .vectorize(pulse_vi)
                       .parallel(pulse_vo);
            win.compute_root();
            filt.compute_root();
            phs_filt.inner.compute_root()
                          .bound(c, 0, 2).unroll(c)
                          .vectorize(sample, vectorsize)
                          .parallel(pulse, blocksize);
            phs_pad.inner.compute_root()
                         .bound(c, 0, 2).unroll(c)
                         .vectorize(sample, vectorsize)
                         .parallel(pulse, blocksize);
            fftsh.inner.compute_root()
                       .bound(c, 0, 2).unroll(c)
                       .vectorize(sample, vectorsize)
                       .parallel(pulse, blocksize);
            dft.inner.compute_root().parallel(pulse);
            Q.inner.compute_root()
                   .split(sample, sample_vo, sample_vi, vectorsize)
                   .vectorize(sample_vi)
                   .parallel(pulse);
                   //.gpu_tile(pulse, pulse_vo, pulse_vi, blocksize_gpu_tile); // causes dft to segfault
            norm_r0.compute_root()
                   .gpu_tile(pulse, pulse_vo, pulse_vi, blocksize_gpu_tile);
            norm_r0.update(0)
                   .gpu_tile(pulse, pulse_vo, pulse_vi, blocksize_gpu_tile);
            norm_r0.update(1)
                   .gpu_tile(pulse, pulse_vo, pulse_vi, blocksize_gpu_tile);
            rr0.compute_inline();
            norm_rr0.compute_at(fimg.inner, x_vi)
                    .reorder(pulse, pixel)
                    .reorder_storage(pulse, pixel);
            dr_i.compute_inline();
            Q_hat.inner.compute_inline();
            img.inner.compute_at(fimg.inner, x_vi);
            img.inner.update(0)
                     .reorder(c, pixel, rnpulses.x);
            fimg.inner.compute_root()
                      .bound(c, 0, 2)
                      .unroll(c)
                      .split(x, x_vo, x_vi, blocksize_gpu_split_x)
                      .gpu_blocks(y)
                      .gpu_threads(x_vi);
            output_img.compute_root()
                      .bound(c, 0, 2)
                      .unroll(c)
                      .vectorize(x, vectorsize)
                      .parallel(y);
#if defined(WITH_DISTRIBUTE)
            if (sched == Schedule::GPUDistributed) {
                output_img.distribute(y);
                fimg.inner.distribute(y);
            }
#endif // WITH_DISTRIBUTE
            if (print_loop_nest) {
                output_img.print_loop_nest();
            }
            break;
        case Schedule::CPU:
#if defined(WITH_DISTRIBUTE)
        case Schedule::CPUDistributed:
#endif // WITH_DISTRIBUTE
            // CPU target
            std::cout << "Scheduling for CPU: " << tgt << std::endl
                      << "Vector size: " << vectorsize.value() << std::endl
                      << "Block size: " << blocksize.value() << std::endl;
            win_sample.taylor.compute_root()
                             .vectorize(sample, vectorsize)
                             .parallel(sample, blocksize);
            win_sample.w.compute_root()
                        .split(sample, sample_vo, sample_vi, vectorsize)
                        .vectorize(sample_vi)
                        .parallel(sample_vo);
            win_sample.w.update(0)
                        .split(sample, sample_vo, sample_vi, vectorsize, TailStrategy::GuardWithIf)
                        .vectorize(sample_vi)
                        .parallel(sample_vo);
            win_pulse.taylor.compute_root()
                            .vectorize(pulse, vectorsize)
                            .parallel(pulse, blocksize);
            win_pulse.w.compute_root()
                       .split(pulse, pulse_vo, pulse_vi, vectorsize)
                       .vectorize(pulse_vi)
                       .parallel(pulse_vo);
            win_pulse.w.update(0)
                       .split(pulse, pulse_vo, pulse_vi, vectorsize, TailStrategy::GuardWithIf)
                       .vectorize(pulse_vi)
                       .parallel(pulse_vo);
            win.compute_root();
            filt.compute_root();
            phs_filt.inner.compute_root()
                          .bound(c, 0, 2).unroll(c)
                          .vectorize(sample, vectorsize)
                          .parallel(pulse, blocksize);;
            phs_pad.inner.compute_root()
                         .bound(c, 0, 2).unroll(c)
                         .vectorize(sample, vectorsize)
                         .parallel(pulse, blocksize);
            fftsh.inner.compute_root()
                       .bound(c, 0, 2).unroll(c)
                       .vectorize(sample, vectorsize)
                       .parallel(pulse, blocksize);
            dft.inner.compute_root().parallel(pulse);
            Q.inner.compute_root()
                   .split(sample, sample_vo, sample_vi, vectorsize)
                   .vectorize(sample_vi)
                   .parallel(pulse);
            norm_r0.compute_root()
                   .split(pulse, pulse_vo, pulse_vi, vectorsize)
                   .vectorize(pulse_vi)
                   .parallel(pulse_vo);
            norm_r0.update(0)
                   .split(pulse, pulse_vo, pulse_vi, vectorsize, TailStrategy::GuardWithIf)
                   .vectorize(pulse_vi)
                   .parallel(pulse_vo);
            norm_r0.update(1)
                   .split(pulse, pulse_vo, pulse_vi, vectorsize, TailStrategy::GuardWithIf)
                   .vectorize(pulse_vi)
                   .parallel(pulse_vo);
            rr0.compute_inline();
            norm_rr0.compute_at(output_img, x_vo)
                    .split(pulse, pulse_vo, pulse_vi, vectorsize)
                    .vectorize(pulse_vi);
            norm_rr0.update(0)
                    .split(pulse, pulse_vo, pulse_vi, vectorsize, TailStrategy::GuardWithIf)
                    .vectorize(pulse_vi);
            norm_rr0.update(1)
                    .split(pulse, pulse_vo, pulse_vi, vectorsize, TailStrategy::GuardWithIf)
                    .vectorize(pulse_vi);
            dr_i.compute_inline();
            Q_hat.inner.compute_inline();
            img.inner.compute_at(output_img, x_vo);
            img.inner.update(0).reorder(c, pixel, rnpulses.x);
            fimg.inner.compute_inline();
            output_img.compute_root()
                      .split(x, x_vo, x_vi, vectorsize)
                      .vectorize(x_vi)
                      .parallel(x_vo);
#if defined(WITH_DISTRIBUTE)
            if (sched == Schedule::CPUDistributed) {
                output_img.distribute(y);
            }
#endif // WITH_DISTRIBUTE
            if (print_loop_nest) {
                output_img.print_loop_nest();
            }
            break;
        default:
            throw std::runtime_error("Unknown schedule: " + sched.name());
        }
    }

private:
    Var x{"x"}, y{"y"};
    Var c{"c"}, sample{"sample"}, pixel{"pixel"}, pulse{"pulse"}, dim{"dim"};

    Taylor win_sample;
    Taylor win_pulse;
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
    Func r{"r"};
    Func pos{"pos"};
    ComplexFunc Q_hat{c, "Q_hat"};
    ComplexFunc img{c, "img"};
    ComplexFunc fimg{c, "fimg"};
    RDom rnpulses;
    RDom rnd;
};

HALIDE_REGISTER_GENERATOR(BackprojectionGenerator, backprojection)
HALIDE_REGISTER_GENERATOR(BackprojectionGenerator, backprojection_distributed)
HALIDE_REGISTER_GENERATOR(BackprojectionGenerator, backprojection_cuda)
HALIDE_REGISTER_GENERATOR(BackprojectionGenerator, backprojection_cuda_distributed)
HALIDE_REGISTER_GENERATOR(BackprojectionGenerator, backprojection_auto_m16)
