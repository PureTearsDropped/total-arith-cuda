# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
HyperTranscend — an **experimental** unified computation of transcendental functions
(exp, log, sqrt, ^) for a hypercomplex number of any dimension M = 2^k, via the LEFT
regular representation.  These are *function values defined through L_x*, not a claim
that every hypercomplex analytic function is captured; which algebraic identities hold
is CHECKED per dimension in `self_test()`, not assumed (for non-associative M ≥ 8 the
branch of a matrix function, and left- vs right-functions, need case-by-case care).

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
import Random
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
# `needs_inverse`: only ops that literally invert (log, x^negative) require L_x nonsingular.
# √ and x^(positive) do NOT — √0 = 0 is fine even though L_0 is singular. So we flag SING
# *only when the matrix function actually fails*, not preemptively (external AI review found
# hsqrt(0) was a false positive: a well-defined op reported as a zero-divisor failure).
function _matfun(f, x::Hyper; needs_inverse::Bool)
    M = dim(x); L = Lmatrix(x); flag = x.flag
    all(iszero, x.c) && return Hyper(zeros(M), flag)   # f(0): resolved directly (√0=0, 0^p=0)
    if needs_inverse && _singular(L)
        return Hyper(zeros(M), flag | SING)            # inversion of a zero divisor: no answer
    end
    v = try
        Y = f(L); Y * e0(M)
    catch
        return Hyper(zeros(M), flag | SING)            # matrix function genuinely failed
    end
    imres = maximum(abs.(imag.(v)))
    rmag  = max(maximum(abs.(real.(v))), 1.0)
    (imres > 1e-8 * rmag) && (flag |= CPLX)            # left the reals ⇒ bigger field
    _tot(Hyper(real.(v), flag))
end
hlog(x::Hyper)  = _matfun(log,  x; needs_inverse=true)    # log(0)=−∞, log(zero-div) singular
hsqrt(x::Hyper) = _matfun(sqrt, x; needs_inverse=false)   # √ never needs inversion
function Base.:^(x::Hyper, p::Real)
    if isinteger(p) && p >= 0
        _matfun(A -> A^Int(p),   x; needs_inverse=false)  # forward, total
    elseif p > 0
        _matfun(A -> A^float(p), x; needs_inverse=false)  # positive fractional: no inversion
    else
        _matfun(A -> A^float(p), x; needs_inverse=true)   # negative power: needs inverse
    end
end

isreal_ok(x::Hyper) = (x.flag & (SING | CPLX)) == 0
function Base.show(io::IO, x::Hyper)
    tag = x.flag == 0 ? "" : "⟦" * (x.flag&SING>0 ? "零因子" : "") *
          (x.flag&CPLX>0 ? "ℂ" : "") * (x.flag&OVER>0 ? "≥" : "") * "⟧"
    print(io, "Hyper", dim(x), "(", join(round.(x.c, digits=4), ","), ")", tag)
end

# ---- self-test: algebraic identities per dimension + the audit regressions ----
function self_test()
    println("HyperTranscend self-test — identities are CHECKED, not assumed")
    rng = Random.MersenneTwister(7)
    approx(a::Hyper, b::Hyper) = maximum(abs.(a.c .- b.c)) < 1e-6
    for M in (1, 2, 4, 8, 16)
        x = Hyper([1.4; 0.25 .* randn(rng, M-1)])       # near identity: log/√ well-conditioned
        oks = [
            ("√x·√x==x",        approx(hsqrt(x)*hsqrt(x), x)),
            ("exp(log x)==x",   approx(hexp(hlog(x)), x)),
            ("x^2==x·x",        approx(x^2, x*x)),
            ("x^0.5·x^0.5==x",  approx((x^0.5)*(x^0.5), x)),
        ]
        s = join(["$n $(v ? "✓" : "✗")" for (n,v) in oks], "  ")
        println("  M=$M: $s")
        @assert all(v for (_,v) in oks)
    end
    # audit regressions (external AI review 2026-07-20): √0 / 0^p must NOT false-flag
    for f in (hsqrt, x->x^0.5, x->x^2.5)
        r = f(Hyper([0.0]))
        @assert flags(r) == 0x00 && r.c[1] == 0.0 "√0 / 0^p wrongly flagged"
    end
    println("  regression √0=0, 0^0.5=0, 0^2.5=0 — no false zero-divisor flag ✓")
    # a genuine zero divisor: forward exp ok, inverse (log) flagged
    z = Hyper([i∈(4,11) ? 1.0 : 0.0 for i in 1:16])
    @assert flags(hexp(z)) & SING == 0 "exp(zero-divisor) should stay total"
    @assert flags(hlog(z)) & SING != 0 "log(zero-divisor) should be flagged"
    println("  zero divisor z=e3+e10: exp forward-total, log flagged ⟦零因子⟧ ✓")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    HyperTranscend.self_test()
end
