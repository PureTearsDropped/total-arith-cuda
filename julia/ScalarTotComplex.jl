# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
isdefined(@__MODULE__, :ScalarTot) || include(joinpath(@__DIR__, "ScalarTot.jl"))

"""
ScalarTotComplex — total arithmetic on ℂ (2026-09-03), the complex twin of `ScalarTot.TotNum`.

  A value is **polar**: |z| ≥ 0 and arg z / π ∈ (−1, 1] (units of π: i = 1∠½π, so i·i = 1∠1π = −1
  *exactly*, and e^{iπ} + 1 = 0 comes out as an exact 0; `sinpi` / `cospi` are exact at the
  quarter turns).  Two flag systems, in the same bits as `TotNum`'s:
      GE / LE  — bounds on the magnitude |z| (exactly TotNum's bounds on |x|);
      AUNK     — the direction (arg) is unknown.  This is TotNum's SUNK bit read in ℂ: "sign
                 unknown" on ℝ is "arg ∈ {0, π} unknown", the two-point case of a direction.
  There is no ℂ flag in ℂ: nothing leaves an algebraically closed field.  A `TotNum` value in the
  ℂ seat (0⟦≥≤±ℂ⟧, "the answer exists, somewhere in ℂ") converts to the *unknown seat* of this
  type, 0⟦≥≤ ∠?⟧ — a dangerous zero with unknown direction — and √(−1), log(−1), (−1)^½ deliver the
  value the real type could only name: i, iπ, i.

  The reserved word.  0 is exactly 0 and has **no direction: arg 0 = 0 by definition**, the same
  shelf as a/0 = 0 and log 0 = 0.  The polar identity z = |z|·e^{i arg z} holds there (0 = 0·1),
  the unit direction u = z/|z| = z/0 = 0 does not lie on the circle, and log 0 = log|0| + i·arg 0
  = 0 + 0i agrees with TotNum on the real axis.  The limits are ε's job, and ε **carries its
  direction**: an underflow is MIN⟦≤⟧∠θ with θ kept, so log ε = log MIN⟦≥⟧ + iθ — "0 is a value
  without direction, ε is a limit with one".  IEEE's ±0 and atan2(±0, −1) = ±π are the limit
  reading; here a single 0 has a single arg.  The price, stated: exp(log 0) = 1 ≠ 0, so powers are
  **not** derived from exp(w·log z) at 0 (0^w = 0 for w ≠ 0, 0^0 = 1 stay definitions), and
  u = e^{i·arg z} fails at 0 only — where x·(1/x) = 1 already fails.

  Every claim is checked by `audit_cplx.jl` (the semantic oracle of `audit_flags.jl` lifted to
  complex truths) and the real axis is twin-checked against `TotNum`.
"""
module ScalarTotComplex

using ..ScalarTot
using ..ScalarTot: GE, LE, SUNK, CPLX, NOB, MAXF, MINF, _sat, _sat_nz, _mulflag, _powflag
export TotComplex, AUNK, arg, polar, IM, tflag

const AUNK = SUNK                                  # the same bit: ℝ's "sign unknown" is ℂ's "direction unknown"

"""arg/π wrapped into (−1, 1] (−0.0 → 0.0)."""
@inline function wrap(t::Float64)
    t = rem(t, 2.0)
    t > 1.0 && (t -= 2.0)
    t <= -1.0 && (t += 2.0)
    t == 0 ? 0.0 : t
end

struct TotComplex <: Number
    r::Float64        # |z| ∈ {0} ∪ [MIN, MAX]
    t::Float64        # arg z / π ∈ (−1, 1]; 0 when r == 0 (0 has no direction)
    flag::UInt8       # GE / LE on |z|, AUNK on the direction
    function TotComplex(r::Float64, t::Float64, flag::UInt8)
        r = abs(r)
        if r == 0
            r = 0.0; t = 0.0                                          # one zero, one arg
            (flag & GE) != 0 && (flag = NOB | AUNK)                   # a dangerous zero is the unknown seat: any magnitude, any direction
        else
            t = wrap(t)
        end
        new(r, t, flag & (GE | LE | AUNK))
    end
end
tflag(z::TotComplex) = z.flag

# ---- construction ----------------------------------------------------------------------------
"""r∠tπ with a flag (the magnitude is totalized: Inf → MAX⟦≥⟧, subnormal → MIN⟦≤⟧, NaN → the unknown seat)."""
function polar(r::Real, t::Real, flag::UInt8 = 0x00)
    s = _sat(abs(Float64(r)))
    isnan(Float64(r)) && return TotComplex(0.0, 0.0, NOB | AUNK)
    TotComplex(s.val, Float64(t), flag | s.flag)
end
"""Cartesian re + i·im (both exact Float64s) → polar; the arg from atan2, a single 0 → arg 0."""
function fromcart(re::Float64, im::Float64, flag::UInt8 = 0x00)
    (isnan(re) || isnan(im)) && return TotComplex(0.0, 0.0, NOB | AUNK)
    h = hypot(re, im)
    h == 0 && return TotComplex(0.0, 0.0, flag)
    s = _sat(h)
    TotComplex(s.val, atan(im, re) / pi, flag | s.flag)
end
TotComplex(re::Real, im::Real) = fromcart(Float64(re), Float64(im))
TotComplex(x::Real) = fromcart(Float64(x), 0.0)
TotComplex(z::Complex) = fromcart(Float64(real(z)), Float64(imag(z)))
"""a `TotNum` on the real axis: sign → arg ∈ {0, π}, SUNK → AUNK, the ℂ seat → the unknown seat."""
function TotComplex(x::TotNum)
    (x.flag & CPLX) != 0 && return TotComplex(0.0, 0.0, NOB | AUNK)
    TotComplex(abs(x.val), x.val < 0 ? 1.0 : 0.0, x.flag & (GE | LE | SUNK))
end
TotComplex(z::TotComplex) = z
const IM = TotComplex(1.0, 0.5, 0x00)
Base.zero(::Type{TotComplex}) = TotComplex(0.0, 0.0, 0x00)
Base.one(::Type{TotComplex}) = TotComplex(1.0, 0.0, 0x00)
Base.zero(::TotComplex) = zero(TotComplex)
Base.one(::TotComplex) = one(TotComplex)
Base.promote_rule(::Type{TotComplex}, ::Type{<:Real}) = TotComplex
Base.promote_rule(::Type{TotComplex}, ::Type{TotNum}) = TotComplex
Base.promote_rule(::Type{TotComplex}, ::Type{<:Complex}) = TotComplex
Base.convert(::Type{TotComplex}, x::Number) = TotComplex(x)
Base.:(==)(a::TotComplex, b::TotComplex) = a.r == b.r && a.t == b.t

"""the unknown seat: a value that exists somewhere in ℂ (the ℂ seat of TotNum, cashed in)."""
unknown() = TotComplex(0.0, 0.0, NOB | AUNK)
"""a true zero (exactly 0); a zero with GE is a *dangerous* zero (unknown magnitude and direction)."""
@inline truezero(z::TotComplex) = z.r == 0 && (z.flag & GE) == 0
@inline dangerous(z::TotComplex) = z.r == 0 && (z.flag & GE) != 0
@inline aunk(z::TotComplex) = (z.flag & AUNK) != 0
@inline swapdir(f::UInt8) = ((f & GE) << 1) | ((f & LE) >> 1) | (f & AUNK)   # GE ↔ LE (a reciprocal)

# ---- parts -----------------------------------------------------------------------------------
cart(z::TotComplex) = (z.r * cospi(z.t), z.r * sinpi(z.t))
Base.abs(z::TotComplex) = TotNum(z.r, z.flag & NOB)                      # |z| ≥ 0: the sign is certain
"""arg z in radians; arg 0 = 0 by definition (a true zero); an unknown direction → 0⟦≥≤±⟧."""
arg(z::TotComplex) = aunk(z) ? TotNum(0.0, NOB | SUNK) : TotNum(z.t * pi, 0x00)
Base.angle(z::TotComplex) = arg(z)
function Base.real(z::TotComplex)
    aunk(z) && return TotNum(0.0, NOB | SUNK)
    c = cospi(z.t)
    c == 0 && return TotNum(0.0, 0x00)                                   # t·0 = 0 whatever t: exact
    TotNum(z.r * c, z.flag & NOB)                                         # |re| = |z|·|cos|: monotone; sign = sign of cos
end
function Base.imag(z::TotComplex)
    aunk(z) && return TotNum(0.0, NOB | SUNK)
    s = sinpi(z.t)
    s == 0 && return TotNum(0.0, 0x00)
    TotNum(z.r * s, z.flag & NOB)
end
Base.conj(z::TotComplex) = TotComplex(z.r, -z.t, z.flag)
Base.:-(z::TotComplex) = TotComplex(z.r, z.t + 1.0, z.flag)
Base.isreal(z::TotComplex) = !aunk(z) && (z.t == 0 || z.t == 1)

# ---- × ÷ (polar: exact in the direction) -------------------------------------------------
function Base.:*(a::TotComplex, b::TotComplex)
    (truezero(a) || truezero(b)) && return zero(TotComplex)             # a true zero absorbs (0·MAX⟦≥⟧ = 0)
    (dangerous(a) || dangerous(b)) && return unknown()                  # nothing known about one factor
    s = _sat_nz(a.r * b.r)                                              # nonzero·nonzero ≠ 0: a raw 0 is ε
    TotComplex(s.val, a.t + b.t, s.flag | _mulflag(a.flag, b.flag))
end
function Base.:/(a::TotComplex, b::TotComplex)
    truezero(b) && return zero(TotComplex)                              # a/0 = 0 for every a: a definition
    truezero(a) && return zero(TotComplex)                              # 0/b = 0 (b ≠ 0, or unknown: still 0)
    (dangerous(a) || dangerous(b)) && return unknown()
    s = _sat_nz(a.r / b.r)
    TotComplex(s.val, a.t - b.t, s.flag | _mulflag(a.flag, swapdir(b.flag)))   # a bound on the divisor flips
end
Base.inv(z::TotComplex) = one(TotComplex) / z

# ---- + − (Cartesian, then the direction rule) ------------------------------------------
# |t₁e^{iθ₁} + t₂e^{iθ₂}|² = t₁² + t₂² + 2t₁t₂cos Δ is monotone in both magnitudes iff cos Δ ≥ 0, so a
# magnitude bound survives an addition only within 90°; the direction of the sum survives only when the
# two are collinear (Δ = 0); beyond 90° the sum can cancel: no bound, no direction.  On ℝ: Δ ∈ {0, π}.
function addsub(a::TotComplex, b::TotComplex, flip::Bool)
    truezero(a) && return flip ? -b : b
    truezero(b) && return a
    ra, ia = cart(a); rb, ib = cart(b)
    flip && (rb = -rb; ib = -ib)                                        # negate the parts, not the angle: a − a = 0 exactly
    z = fromcart(ra + rb, ia + ib)
    fin = a.flag | b.flag
    fin == 0x00 && return z
    (aunk(a) || aunk(b) || dangerous(a) || dangerous(b)) && return TotComplex(z.r, z.t, z.flag | NOB | AUNK)
    Δ = wrap(a.t - b.t + (flip ? 1.0 : 0.0))
    if Δ == 0
        out = fin & NOB                                                 # collinear: bounds add up, the ray is kept
    elseif cospi(Δ) ≥ 0
        out = (fin & NOB) | AUNK                                        # within 90°: |sum| monotone, direction moves
    else
        out = NOB | AUNK                                                # cancellation possible
    end
    TotComplex(z.r, z.t, z.flag | out)
end
Base.:+(a::TotComplex, b::TotComplex) = addsub(a, b, false)
Base.:-(a::TotComplex, b::TotComplex) = addsub(a, b, true)

# ---- √ exp log ^ ---------------------------------------------------------------------------
function Base.sqrt(z::TotComplex)
    truezero(z) && return zero(TotComplex)
    dangerous(z) && return unknown()
    s = _sat(sqrt(z.r))
    TotComplex(s.val, z.t / 2, s.flag | z.flag)                         # |·|^½ monotone; the half-angle is exact
end
function Base.exp(z::TotComplex)
    truezero(z) && return one(TotComplex)
    (dangerous(z) || aunk(z)) && return TotComplex(1.0, 0.0, NOB | AUNK)   # x = |z|cos θ unknown in [−|z|, |z|]
    x, y = cart(z)
    s = _sat_nz(exp(x))                                                 # e^x > 0: exp(−MAX) = MIN⟦≤⟧
    t = rem2pi(y, RoundNearest) / pi                                    # y mod 2π exactly (Payne–Hanek): y/π alone loses the arg for |y| > 2^23
    dir = z.flag & NOB
    dir == 0x00 && return TotComplex(s.val, t, s.flag)
    c = cospi(z.t); sn = sinpi(z.t)
    mag = dir == NOB ? NOB : (c == 0 ? 0x00 : (c > 0 ? dir : swapdir(dir)))   # true x = t·cos θ: grows with t iff cos θ > 0
    TotComplex(s.val, t, s.flag | mag | (sn == 0 ? 0x00 : AUNK))        # y = t·sin θ moves the direction unless sin θ = 0
end
function Base.log(z::TotComplex)
    truezero(z) && return zero(TotComplex)                              # log 0 = 0: the reserved word
    dangerous(z) && return unknown()
    lr = log(TotNum(z.r, z.flag & NOB))                                 # TotNum's rule: GE only in the two provable cells
    if aunk(z)                                                          # |log z|² = (log|z|)² + θ² ≥ (log|z|)²
        return TotComplex(abs(lr.val), 0.0, (lr.flag & SUNK) == 0 ? GE | AUNK : NOB | AUNK)   # |log|z|| exact or ≥: a floor
    end
    w = fromcart(lr.val, z.t * pi)
    lr.flag == 0x00 && return w
    (lr.flag & SUNK) != 0 && return TotComplex(w.r, w.t, w.flag | NOB | AUNK)
    TotComplex(w.r, w.t, w.flag | GE | (z.t == 0 ? 0x00 : AUNK))        # |log t| ≥ |log r| ⟹ |w| grows; the direction moves unless θ = 0
end
function Base.:^(z::TotComplex, n::Integer)
    n == 0 && return one(TotComplex)                                    # z⁰ = 1, 0⁰ = 1: the empty product
    truezero(z) && return zero(TotComplex)                              # 0ⁿ = 0, 0⁻ⁿ = 1/0 = 0
    dangerous(z) && return unknown()
    s = _sat_nz(z.r^n)
    TotComplex(s.val, n * z.t, s.flag | _powflag(z.flag & NOB, Float64(n)) | (z.flag & AUNK))
end
Base.literal_pow(::typeof(^), z::TotComplex, ::Val{N}) where {N} = z^N
function Base.:^(z::TotComplex, y::Real)
    (isinteger(y) && abs(y) < 9e15) && return z^Int(y)
    truezero(z) && return zero(TotComplex)                              # 0^y = 0 (y ≠ 0): a definition, not exp(y·log 0)
    dangerous(z) && return unknown()
    s = _sat_nz(z.r^Float64(y))
    TotComplex(s.val, Float64(y) * z.t, s.flag | _powflag(z.flag & NOB, Float64(y)) | (z.flag & AUNK))   # principal branch
end
function Base.:^(z::TotComplex, w::TotComplex)
    truezero(w) && return one(TotComplex)
    truezero(z) && return zero(TotComplex)
    isreal(w) && !isflagged(w) && return z^(w.t == 0 ? w.r : -w.r)
    exp(w * log(z))                                                     # the flag algebra of the composition
end
Base.:^(z::TotComplex, w::Complex) = z^TotComplex(w)
Base.:^(z::TotComplex, w::TotNum) = z^TotComplex(w)
ScalarTot.isflagged(z::TotComplex) = z.flag != 0x00
ScalarTot.flag_of(z::TotComplex) = z.flag
# periodic in the real direction, exponential in the imaginary one: a flagged input says nothing.
# For |y| > 700 cosh / sinh overflow, but the direction does not: sin(x+iy) ≈ (e^|y|/2)(sin x + i·sgn(y) cos x),
# so the magnitude saturates to MAX⟦≥⟧ with the true direction (a ±Inf component would have put it at an odd
# multiple of π/4 — a lie the oracle caught).
function Base.sin(z::TotComplex)
    isflagged(z) && return unknown()
    x, y = cart(z)
    abs(y) ≤ 700 && return fromcart(sin(x) * cosh(y), cos(x) * sinh(y))
    m = _sat(exp(abs(y) - 0.6931471805599453))
    TotComplex(m.val, atan(sign(y) * cos(x), sin(x)) / pi, m.flag)
end
function Base.cos(z::TotComplex)
    isflagged(z) && return unknown()
    x, y = cart(z)
    abs(y) ≤ 700 && return fromcart(cos(x) * cosh(y), -sin(x) * sinh(y))
    m = _sat(exp(abs(y) - 0.6931471805599453))
    TotComplex(m.val, atan(-sign(y) * sin(x), cos(x)) / pi, m.flag)
end
Base.abs2(z::TotComplex) = abs(z) * abs(z)
# mixed arithmetic with the reals goes through promotion
for op in (:+, :-, :*, :/)
    @eval Base.$op(a::TotComplex, b::Real) = $op(a, TotComplex(b))
    @eval Base.$op(a::Real, b::TotComplex) = $op(TotComplex(a), b)
    @eval Base.$op(a::TotComplex, b::TotNum) = $op(a, TotComplex(b))
    @eval Base.$op(a::TotNum, b::TotComplex) = $op(TotComplex(a), b)
    @eval Base.$op(a::TotComplex, b::Complex) = $op(a, TotComplex(b))
    @eval Base.$op(a::Complex, b::TotComplex) = $op(TotComplex(a), b)
end

function Base.show(io::IO, z::TotComplex)
    print(io, z.r, "∠", z.t, "π")
    z.flag == 0 && return
    print(io, "⟦", (z.flag & GE) > 0 ? "≥" : "", (z.flag & LE) > 0 ? "≤" : "", (z.flag & AUNK) > 0 ? "∠?" : "", "⟧")
end

end # module
