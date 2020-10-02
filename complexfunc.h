#ifndef _COMPLEXFUNC_H
#define _COMPLEXFUNC_H

#include <vector>
#include <Halide.h>

using std::string;

class ComplexExpr;
/*
 * ComplexFunc wraps a Func in a way that intercepts index expressions and generates ComplexExprs for them.
 * A 2d ComplexFunc of size [i,j] will have an underlying (inner) 3d Func of size [2,i,j].
 * The complex axis is passed in explicitly, it should be consistent across all ComplexFuncs that participate in
 * complex mathematical expressions.
 */
class ComplexFunc {
public:
    string name;
    Halide::Func inner;
    Halide::Var element;

    ComplexFunc(Halide::Var &element, string name);
    ComplexFunc(Halide::Var &element, Halide::Func &inner, string name);
    ComplexExpr operator()(std::vector<Halide::Expr>);
    ComplexExpr operator()(Halide::Expr idx1);
    ComplexExpr operator()(Halide::Expr idx1, Halide::Expr idx2);
    ComplexExpr operator()(Halide::Expr idx1, Halide::Expr idx2, Halide::Expr idx3);
};


/*
 * ComplexExpr represents a Complex value.  It uses operator overloading to
 * implement mathematical operations.  These act on either the complex number
 * as a whole or the real/imaginary elements separately, depending on the
 * operation.
 *
 * Some ComplexExprs represent a position in a ComplexFunc.  This allows it to
 * act as an lvalue and be assigned to.  All ComplexExprs can act as rvalues,
 * except for the ones representing ComplexFuncs which have never been assigned
 * to yet.
 */
class ComplexExpr {
public:

    Halide::Var element;
    Halide::Expr real;
    Halide::Expr imag;
    Halide::Expr pair; // this is a mux expression
    const ComplexFunc *func; // Func that writes are passed through to
    std::vector<Halide::Expr> pair_idx; // saved index for writes

    bool can_read;
    bool can_write;

    inline ComplexExpr(const ComplexFunc *func, const std::vector<Halide::Expr> &idx);              // lvalue constructor
    inline ComplexExpr(const Halide::Var &element, const Halide::Expr &v1, const Halide::Expr &v2); // rvalue constructor

    // read ops
    ComplexExpr operator-();                      // negation
    ComplexExpr operator+(Halide::Expr other);    // addition (of real element)
    ComplexExpr operator-(Halide::Expr other);    // subtraction (of real element)
    ComplexExpr operator*(Halide::Expr other);    // scalar multiplication
    ComplexExpr operator/(Halide::Expr other);    // scalar division
    ComplexExpr operator+(ComplexExpr other); // addition
    ComplexExpr operator-(ComplexExpr other); // subtraction
    ComplexExpr operator*(ComplexExpr other); // multiplication
    ComplexExpr operator/(ComplexExpr other); // division

    // write ops
    ComplexExpr &operator=(ComplexExpr rvalue);
};


/*
 * Create a ComplexExpr that represents an element of a ComplexFunc.  This
 * ComplexExpr can be assigned to as an lvalue.  If the underlying Func
 * is defined (by having been assigned to previously), this ComplexExpr
 * can also be used as an rvalue, or as an element in a larger mathematical
 * expression.
 */
inline ComplexExpr::ComplexExpr(const ComplexFunc *func, const std::vector<Halide::Expr> &idx) :func(func) {
    element = func->element;
    std::vector<Halide::Expr> real_idx({Halide::Expr(0)});
    std::vector<Halide::Expr> imag_idx({Halide::Expr(1)});
    pair_idx.reserve(idx.size() + 1);
    pair_idx.push_back(element);
    real_idx.reserve(idx.size() + 1);
    imag_idx.reserve(idx.size() + 1);
    copy(idx.begin(), idx.end(), back_inserter(real_idx));
    copy(idx.begin(), idx.end(), back_inserter(imag_idx));
    copy(idx.begin(), idx.end(), back_inserter(pair_idx));
    can_write = true;
    can_read = func->inner.defined();
    if(can_read) {
        real = func->inner(real_idx);
        imag = func->inner(imag_idx);
        pair = func->inner(pair_idx);
    }
}

/*
 * Create a ComplexExpr representing a read-only value.  This ComplexExpr has
 * no ComplexFunc, hence it cannot be assigned to.  It can be used as an
 * rvalue, or as an element in a larger mathematical expression.
 */
inline ComplexExpr::ComplexExpr(const Halide::Var &element, const Halide::Expr &v1, const Halide::Expr &v2) {
    real = v1;
    imag = v2;
    can_read = true;
    can_write = false;
    this->element = element;
    pair = Halide::mux(element, {v1, v2});
}


// negation
ComplexExpr ComplexExpr::operator-() {
    if(can_read == false)
        throw;
    return ComplexExpr(element, -real, -imag);
}

// addition of complex and real
ComplexExpr ComplexExpr::operator+(Halide::Expr other) {
    if(can_read == false)
        throw;
    return ComplexExpr(element, real + other, imag);
}

// addition of 2 complex
ComplexExpr ComplexExpr::operator+(ComplexExpr other) {
    if(can_read == false || other.can_read == false)
        throw;
    return ComplexExpr(element, real + other.real, imag + other.imag);
}


// subtraction of complex and real
ComplexExpr ComplexExpr::operator-(Halide::Expr other) {
    if(can_read == false)
        throw;
    return ComplexExpr(element, real - other, imag);
}

// subtraction of 2 complex
ComplexExpr ComplexExpr::operator-(ComplexExpr other) {
    if(can_read == false || other.can_read == false)
        throw;
    return ComplexExpr(element, real - other.real, imag - other.imag);
}


// multiplication of complex and real
ComplexExpr ComplexExpr::operator*(Halide::Expr other) {
    if(can_read == false)
        throw;
    return ComplexExpr(element, real * other, imag * other);
}

// multiplication of 2 complex
ComplexExpr ComplexExpr::operator*(ComplexExpr other) {
    if(can_read == false || other.can_read == false)
        throw;
    return ComplexExpr(element, real * other.real - imag * other.imag, real * other.imag + imag * other.real);
}


// division of complex and real
ComplexExpr ComplexExpr::operator/(Halide::Expr other) {
    if(can_read == false)
        throw;
    return ComplexExpr(element, real / other, imag / other);
}

// division of 2 complex
ComplexExpr ComplexExpr::operator/(ComplexExpr other) {
    if(can_read == false || other.can_read == false)
        throw;
    ComplexExpr conjugate = other;
    conjugate.imag = -conjugate.imag;
    ComplexExpr numerator   = *this * conjugate;
    ComplexExpr denominator = other * conjugate;
    return ComplexExpr(element, numerator.real / denominator.real, numerator.imag / denominator.real);
}

// assignment
ComplexExpr &ComplexExpr::operator=(ComplexExpr rvalue) {
    if(rvalue.can_read == false)
        throw;
    Halide::FuncRef funcref = func->inner(pair_idx);
    funcref = rvalue.pair;
    pair = rvalue.pair;
    real = rvalue.real;
    imag = rvalue.imag;
    can_read = true;
    return *this;
}

inline ComplexExpr select(const Halide::Var &element, Halide::Expr c, ComplexExpr t, ComplexExpr f) {
    return ComplexExpr(element,
                       Halide::select(c, t.real, f.real),
                       Halide::select(c, t.imag, f.imag));
}

inline ComplexExpr select(const Halide::Var &element,
                          Halide::Expr c1, ComplexExpr t1,
                          Halide::Expr c2, ComplexExpr t2,
                          ComplexExpr f) {
    return ComplexExpr(element,
                       Halide::select(c1, t1.real, c2, t2.real, f.real),
                       Halide::select(c1, t1.imag, c2, t2.imag, f.imag));
}

// ComplexFunc methods

ComplexFunc::ComplexFunc(Halide::Var &element, string name) :name(name), element(element) {
    inner(name+"_inner");
}

ComplexFunc::ComplexFunc(Halide::Var &element, Halide::Func &inner, string name) :name(name), inner(inner), element(element) {
}

ComplexExpr ComplexFunc::operator()(std::vector<Halide::Expr> idx) {
    return ComplexExpr(this, idx);
}

ComplexExpr ComplexFunc::operator()(Halide::Expr idx1) {
    std::vector<Halide::Expr> idx({idx1});
    return (*this)(idx);
}
ComplexExpr ComplexFunc::operator()(Halide::Expr idx1, Halide::Expr idx2) {
    std::vector<Halide::Expr> idx({idx1, idx2});
    return (*this)(idx);
}
ComplexExpr ComplexFunc::operator()(Halide::Expr idx1, Halide::Expr idx2, Halide::Expr idx3) {
    std::vector<Halide::Expr> idx({idx1, idx2, idx3});
    return (*this)(idx);
}

#endif /* _COMPLEXFUNC_H */
