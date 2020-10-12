#include <Halide.h>

using namespace Halide;

class InterpGenerator : public Halide::Generator<InterpGenerator> {
public:
    Input<Buffer<float>> input_xs {"input_xs", 1};
    Input<Buffer<float>> input_xp {"input_xp", 1};
    Input<Buffer<float>> input_fp {"input_fp", 1};
    Output<Buffer<float>> output_buffer{"output_buffer", 1};

    void generate() {
        output_buffer = _interp(input_xs, input_xp, input_fp, input_xp.dim(0).extent());
    }

private:
    // Essentially what numpy's interp function does
    // xp must be _increasing_ (no duplicate values)
    inline Func _interp(Func xs, Func xp, Func fp, Expr extent) {
        Var x{"x"};
        RDom r(0, extent, "r");
        // index lookups into xp
        Expr lutl("lutl"); // lower index
        Expr lutu("lutu"); // upper index
        // last index in xp where xp(r) < xs(x)
        lutl = max(argmax(r, xp(r) >= xs(x))[0] - 1, 0);
        // first index in xp where xp(r) >= xs(x)
        lutu = argmax(r, xp(r) >= xs(x))[0];
        // Halide complains if we don't clamp
        Expr cll = clamp(lutl, 0, extent - 1);
        Expr clu = clamp(lutu, 0, extent - 1);
        Func interp("interp");
        // Can avoid these first two selects if we enforce that xs values are in xp's value range
        interp(x) =
            select(xs(x) < xp(0), fp(0),
                   select(xs(x) > xp(extent - 1), fp(extent - 1),
                          select(cll == clu, fp(cll),
                                 lerp(fp(cll), fp(clu), (xs(x) - xp(cll)) / (xp(clu) - xp(cll))))));
        return interp;
    }
};

HALIDE_REGISTER_GENERATOR(InterpGenerator, interp)
