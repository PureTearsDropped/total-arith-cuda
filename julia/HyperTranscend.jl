# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
HyperTot — analytic functions (exp, log, sqrt, ^) for a hypercomplex number of ANY
dimension M = 2^k (real / complex / quaternion / octonion / sedenion / …), computed
uniformly through the structure tensor and the regular representation.

  Core identity (this session's result):
      a hypercomplex element x  ≙  its left-multiplication matrix  L_x  (M×M),
      and for any analytic f,   f(x) = f(L_x) · e0.
  So *, ^, exp, log, sqrt are ONE recipe with M as the only knob; the scalar (M=1)
  is the 1×1 case.

  Forward ops (*, exp) are total for every input, zero divisors included — the tensor
  rule never has an exception.  Only INVERSE-type ops (/, log, sqrt, x^negative/fractional)
  can break, and they break in exactly one place: when L_x is singular (a zero divisor,
  which first appears at M ≥ 16).  There we do NOT emit NaN — we name it with a flag.

  Flags (per number): SING = L_x singular ⇒ no unique inverse (zero divisor);
                      CPLX = result left the reals (imag residue) ⇒ go to a bigger field;
                      OVER = a component saturated to ±MAX (range overflow).
"""
module HyperTranscend

using LinearAlgebra
export Hyper, hexp, hlog, hsqrt, e0, isreal_ok, flags, SING, CPLX, OVER

const SING = 0x01
const CPLX = 0x02
const OVER = 0x04
const MAXF = floatmax(Float64)

struct Hyper
    c::Vector{Float64}
    flag::UInt8
end
Hyper(c::AbstractVector) = Hyper(Float64.(collect(c)), 0x00)
flags(x::Hyper) = x.flag
dim(x::Hyper) = length(x.c)

# ---- Cayley–Dickson sign table (cached) : e_i · e_j = OM[i,j] · e_{i⊻j} ----
const _OMCACHE = Dict{Int,Matrix{Int}}()
_cdconj(x) = length(x) == 1 ? copy(x) : vcat(_cdconj(@view x[1:end÷2]), -x[end÷2+1:end])
function _cdprod(x, y)
    n = length(x); n == 1 && return x .* y
    h = n ÷ 2
    a, b, c, d = x[1:h], x[h+1:end], y[1:h], y[h+1:end]
    vcat(_cdprod(a, c) .- _cdprod(_cdconj(d), b), _cdprod(d, a) .+ _cdprod(b, _cdconj(c)))
end
function cd_omega(M::Int)
    haskey(_OMCACHE, M) && return _OMCACHE[M]
    OM = zeros(Int, M, M)
    E = [Float64.(1:M .== i) for i in 1:M]
    for i in 1:M, j in 1:M
        v = _cdprod(E[i], E[j]); k = argmax(abs.(v))
        @assert k == ((i-1) ⊻ (j-1)) + 1 "XOR routing broken M=$M ($i,$j)"
        OM[i, j] = Int(sign(v[k]))
    end
    _OMCACHE[M] = OM
end

# ---- structure-tensor multiply and the regular representation L_x ----
function Base.:*(a::Hyper, b::Hyper)
    M = dim(a); OM = cd_omega(M); r = zeros(M)
    for i in 0:M-1, j in 0:M-1
        r[(i ⊻ j) + 1] += OM[i+1, j+1] * a.c[i+1] * b.c[j+1]
    end
    _tot(Hyper(r, a.flag | b.flag))
end
Base.:+(a::Hyper, b::Hyper) = _tot(Hyper(a.c .+ b.c, a.flag | b.flag))
Base.:-(a::Hyper, b::Hyper) = _tot(Hyper(a.c .- b.c, a.flag | b.flag))
Base.:*(s::Real, a::Hyper) = Hyper(s .* a.c, a.flag)

function Lmatrix(x::Hyper)
    M = dim(x); OM = cd_omega(M); L = zeros(M, M)
    for i in 0:M-1, j in 0:M-1
        L[(i ⊻ j) + 1, j + 1] += OM[i+1, j+1] * x.c[i+1]
    end
    L
end
e0(M::Int) = (v = zeros(M); v[1] = 1.0; v)

# ---- totalize output components: never NaN/Inf; saturate → OVER ----
function _tot(x::Hyper)
    c = copy(x.c); f = x.flag
    for i in eachindex(c)
        v = c[i]
        if isnan(v); c[i] = 0.0; f |= SING
        elseif !isfinite(v) || abs(v) > MAXF; c[i] = sign(v) * MAXF; f |= OVER
        end
    end
    Hyper(c, f)
end

# ---- the analytic functions, uniformly = f(L_x) · e0 ----
# forward: exp is entire ⇒ total for every input (zero divisors included)
function hexp(x::Hyper)
    M = dim(x)
    _tot(Hyper(real.(exp(Lmatrix(x)) * e0(M)), x.flag))
end

# inverse-type: guard the single failure point — L_x singular ⇒ flag, don't NaN
function _singular(L)
    s = svdvals(L)
    s[end] <= 1e-9 * max(s[1], 1.0)          # smallest singular value ~ 0
end
function _matfun(f, x::Hyper; inverse::Bool)
    M = dim(x); L = Lmatrix(x); flag = x.flag
    if inverse && _singular(L)
        return Hyper(zeros(M), flag | SING)   # zero divisor: no unique answer, named
    end
    Y = try f(L) catch; return Hyper(zeros(M), flag | SING) end
    v = Y * e0(M)
    imres = maximum(abs.(imag.(v)))
    rmag  = max(maximum(abs.(real.(v))), 1.0)
    (imres > 1e-8 * rmag) && (flag |= CPLX)   # left the reals ⇒ bigger field
    _tot(Hyper(real.(v), flag))
end
hlog(x::Hyper)  = _matfun(log, x;  inverse=true)
hsqrt(x::Hyper) = _matfun(sqrt, x; inverse=true)
Base.:^(x::Hyper, p::Real) = isinteger(p) && p >= 0 ?
    _matfun(A -> A^Int(p), x; inverse=false) :   # nonneg integer power: forward, total
    _matfun(A -> A^float(p), x; inverse=true)    # fractional/neg: needs inverse

isreal_ok(x::Hyper) = (x.flag & (SING | CPLX)) == 0
function Base.show(io::IO, x::Hyper)
    tag = x.flag == 0 ? "" : "⟦" * (x.flag&SING>0 ? "零因子" : "") *
          (x.flag&CPLX>0 ? "ℂ" : "") * (x.flag&OVER>0 ? "≥" : "") * "⟧"
    print(io, "Hyper", dim(x), "(", join(round.(x.c, digits=4), ","), ")", tag)
end

end # module
