# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
MultiF32 — float128 / float256 / float512 precision total arithmetic built from **Float32
operations only** (`+ - * fma`, compares, selects).  Reference implementation, CPU, stdlib.

  Why not float32 "expansions" (double-double style)?  A non-overlapping expansion of
  Float32 limbs spans 24 bits per limb *in magnitude*: with the leading limb ≈ 1 the 6th
  limb is ≈ 2⁻¹²⁰ and the 7th ≈ 2⁻¹⁴⁴ — below Float32's subnormal floor (2⁻¹⁴⁹).  So
  expansions stop at ≈ 128 bits; 256/512 are unreachable because Float32's *exponent
  range* is the wall, not its significand.  The way through is the block-floating-point
  one (精度と範囲の分離): each Float32 carries an 18-bit **digit** (an integer 0 … 2¹⁸−1)
  whose weight is implicit in its position, and one shared exponent carries the range.

  Representation  `MFloat{P,K,EMAX}`:  sign · (Σᵢ dᵢ·2^{18(i−1)}) · 2^{ex−18K+1},
    dᵢ ∈ [0, 2¹⁸) held in Float32,  ex = exponent of the leading bit,  leading bit at bit 17 of
    d_K (canonical form), low 18K−P bits zero (the value has P significant bits), zero =
    all digits 0.  Flags = ScalarTot's vocabulary (GE / LE / SUNK / CPLX), same rules.
      F128 = MFloat{113, 7, 16383}       (binary128's 113-bit significand, 15-bit exponent)
      F256 = MFloat{237, 14, 262143}     (binary256: 237 bits, 19-bit exponent)
      F512 = MFloat{489, 28, 4194303}    (IEEE generalisation: p = k − round(4log₂k) + 13)

  Digit engine (the only Float32 arithmetic there is):
    • product split  a·b = q·2¹⁸ + r  (0 ≤ a,b < 2¹⁸):  q = fma(a,b,C) − C with C = 1.5·2⁴¹
      (a·b + C lies in one binade whose ulp is 2¹⁸, so the single rounding of the fma lands
      on a multiple of 2¹⁸), r = fma(a,b,−q) exact; one select fixes round-to-nearest into
      floor.  2 fma + 1 sub + 1 select, everything exact.
    • the same constant trick splits an integer-valued Float32 at any bit s (constants
      1.5·2^{23+s}) — that is the carry ripple, the bit shifts and the rounding masks.
    • column sums of a K×K digit product stay below 2K·2¹⁸ < 2²⁴ for K ≤ 31, so every
      Float32 addition in the schoolbook product is exact — no error-free transformations
      needed anywhere.  (18 = 24 − 6: six bits of headroom buy exact accumulation.)

  Rounding: every operation first forms the **exact** result (product: all 2K digits; sum:
  a K+2-digit window plus a sticky for what fell below — the sticky premise counted
  explicitly, cancellation ≤ 1 bit once a sticky can be non-zero), then rounds to nearest
  even at bit P with the exact tail, then saturates: overflow → ±MAX·GE, underflow →
  ±MIN·LE, never NaN/Inf (ScalarTot's `_sat`).  Division and square root: Newton in
  18K-bit digit arithmetic gives a candidate; the **exact residual** (a − q·b, resp.
  a − s²) fixes it to the floor of the true quotient and decides the rounding — candidate
  + verify, the same discipline as `nsolve`/`verify_sqrt` in this directory.

  Self-test: `julia MultiF32.jl` — digit primitives against Int64/BigInt, every operation
  against MPFR (BigFloat at precision P, round-to-nearest) including constructed ties,
  1-ulp cancellations, sticky-at-a-binade-boundary, exact quotients/roots and the range
  edges; flag algebra cross-checked against ScalarTot.jl when it is present.
"""
module MultiF32

export MFloat, F128, F256, F512, GE, LE, SUNK, CPLX, isflagged, flag_of, self_test

using Base.Cartesian: @nexprs, @ntuple

const GE   = 0x01
const LE   = 0x02
const SUNK = 0x04
const CPLX = 0x08
const NOB  = GE | LE

# ---------------------------------------------------------------------------------------
# 1. digit primitives — Float32 only
# ---------------------------------------------------------------------------------------
const DB      = 18                                  # bits per digit
const RADIX   = Float32(2^DB)                       # 262144
const IRADIX  = Float32(2.0^-DB)
const CSPLIT  = Float32(3 * 2^40)                   # 1.5·2^41 : RN to a multiple of 2^18 for |v| ≤ 2^40
const POW2    = ntuple(i -> Float32(2.0^(i - 1)), DB + 1)          # 2^0 … 2^18
const IPOW2   = ntuple(i -> Float32(2.0^-(i - 1)), DB + 1)
const CBIT    = ntuple(i -> Float32(3 * 2.0^(22 + (i - 1))), DB + 1)  # 1.5·2^(23+s) : RN to a multiple of 2^s

"""1f0 if r < 0 else 0f0, from the sign bit only (copysign is bit logic): no compare, so LLVM
cannot turn it back into a branch — with random digits a branch here mispredicts every
other product, which doubled the cost of a K×K product before this was made arithmetic."""
@inline negf(r::Float32) = (1f0 - copysign(1f0, r)) * 0.5f0

"""integer-valued v (|v| ≤ 2^40, exactly representable) → (q, r) with v = q·2^18 + r, 0 ≤ r < 2^18."""
@inline function split18(v::Float32)
    h = (v + CSPLIT) - CSPLIT
    r = v - h                                     # ∈ [−2^17, 2^17]
    s = negf(r)                                   # 1 if r < 0 else 0, branch-free
    return fma(-s, 1f0, h * IRADIX), fma(s, RADIX, r)
end

"""digits a, b ∈ [0, 2^18) → (q, r) with a·b = q·2^18 + r, 0 ≤ r < 2^18 (a·b itself is never formed)."""
@inline function mulsplit18(a::Float32, b::Float32)
    h = fma(a, b, CSPLIT) - CSPLIT
    r = fma(a, b, -h)
    s = negf(r)
    return fma(-s, 1f0, h * IRADIX), fma(s, RADIX, r)
end

"""digit d ∈ [0, 2^18) → (q, r) with d = q·2^s + r, 0 ≤ r < 2^s   (0 ≤ s ≤ 18)."""
@inline function splitbit(d::Float32, s::Int)
    s == 0 && return d, 0f0
    c = CBIT[s + 1]; p = POW2[s + 1]
    h = (d + c) - c
    r = d - h
    t = negf(r)
    return fma(-t, 1f0, h * IPOW2[s + 1]), fma(t, p, r)
end

# tuple helpers with literal indices — Base's map / all / splat / tail leave the fast path
# for tuples longer than 32 elements (Base.Any32), which F512's 56-digit products exceed
@inline tmap(f, a::NTuple{N,Float32}, b::NTuple{N,Float32}) where {N} = ntuple(i -> f(a[i], b[i]), Val(N))
@inline tappend(d::NTuple{N,Float32}, x::Float32) where {N} = ntuple(i -> i ≤ N ? d[i] : x, Val(N + 1))
@inline tprepend2(d::NTuple{N,Float32}) where {N} = ntuple(i -> i ≤ 2 ? 0f0 : d[i - 2], Val(N + 2))
@inline tsetfirst(d::NTuple{N,Float32}, x::Float32) where {N} = ntuple(i -> i == 1 ? x : d[i], Val(N))
@inline function tallzero(d::NTuple{N,Float32}) where {N}
    @inbounds for i in 1:N
        d[i] != 0f0 && return false
    end
    return true
end

# ---------------------------------------------------------------------------------------
# 2. digit tuples (little-endian: v[1] least significant) — static lengths, no heap
#    Every kernel is straight-line Float32 code once specialised on the tuple length
#    (`@nexprs` unrolling), so an operation allocates nothing: CPU without GC, and the same
#    functions run unchanged inside a CUDA.jl kernel.
# ---------------------------------------------------------------------------------------
"""carry ripple with floor semantics (negative entries become borrows); returns (digits, carry out)."""
@generated function carry_norm(v::NTuple{N,Float32}) where {N}
    quote
        c = 0f0
        @nexprs $N i -> begin
            (q, r) = split18(v[i] + c)
            d_i = r; c = q
        end
        return (@ntuple($N, i -> d_i), c)
    end
end

"""column sums of one P×Q block of digit products (P+Q columns, each < 2·min(P,Q)·2^18: exact)."""
@generated function mac_block(a::NTuple{P,Float32}, b::NTuple{Q,Float32}) where {P,Q}
    NC = P + Q
    quote
        Base.@_noinline_meta
        @nexprs $NC k -> c_k = 0f0
        @nexprs $P i -> begin
            @nexprs $Q j -> begin
                (h, l) = mulsplit18(a[i], b[j])
                c_{i+j-1} += l
                c_{i+j}   += h
            end
        end
        return @ntuple($NC, k -> c_k)
    end
end

const MULBLK = 7      # block edge of the schoolbook product (see mul_digits)

"""exact schoolbook product (NA+NB digits); column sums < 2·min(NA,NB)·2^18 < 2^24 needs min(NA,NB) ≤ 31.

The product is assembled from MULBLK×MULBLK blocks (`mac_block`, compiled once per block shape and
kept out of line) that the caller adds at static offsets.  A single straight-line body of NA·NB
products is one giant basic block that LLVM's -O2 pipeline handles super-linearly (28×28: 173 s to
compile; the blocks: 0.4 s) — the blocked form runs within ~10 % of it and, since every partial sum is
an integer below 2^24, returns bit-identical digits."""
@generated function mul_digits(a::NTuple{NA,Float32}, b::NTuple{NB,Float32}) where {NA,NB}
    min(NA, NB) ≤ 31 || error("mul_digits: column sums would leave the exact range")
    NC = NA + NB
    body = Expr[]
    for ra in 1:MULBLK:NA, rb in 1:MULBLK:NB
        P = min(MULBLK, NA - ra + 1); Q = min(MULBLK, NB - rb + 1)
        blk = gensym(:blk)
        push!(body, :($blk = mac_block(@ntuple($P, i -> a[$(ra - 1) + i]), @ntuple($Q, j -> b[$(rb - 1) + j]))))
        for m in 1:P + Q
            push!(body, :($(Symbol(:c_, ra + rb - 2 + m)) += $blk[$m]))
        end
    end
    quote
        Base.@_noinline_meta
        @nexprs $NC k -> c_k = 0f0
        $(body...)
        (c, co) = carry_norm(@ntuple($NC, k -> c_k))
        co == 0f0 || error("mul_digits: carry out")
        return c
    end
end

"""left shift by s bits (0 ≤ s < 18); returns N+1 digits."""
@generated function shl_bits(d::NTuple{N,Float32}, s::Int) where {N}
    top = Symbol(:o_, N + 1)
    quote
        s == 0 && return tappend(d, 0f0)
        p = POW2[s + 1]; carry = 0f0
        @nexprs $N i -> begin
            (q, r) = splitbit(d[i], DB - s)        # d = q·2^(18−s) + r
            o_i = r * p + carry                    # r·2^s + q of the digit below  (< 2^18, exact)
            carry = q
        end
        $top = carry
        return @ntuple($(N + 1), i -> o_i)
    end
end

"""d·2^shift placed in an L-digit window whose LSB has weight 2^0 (relative); bits that fall
below the window are folded into the sticky; bits above the window must be zero."""
@inline function place(d::NTuple{N,Float32}, shift::Int, ::Val{L}) where {N,L}
    ds = fld(shift, DB); s = shift - DB * ds
    e = shl_bits(d, s)                                     # N+1 digits
    w = ntuple(Val(L)) do j
        i = j - ds
        (1 ≤ i ≤ N + 1) ? (@inbounds e[i]) : 0f0
    end
    st = false
    @inbounds for i in 1:N + 1
        j = i + ds
        if j < 1
            st |= (e[i] != 0f0)
        elseif j > L
            e[i] == 0f0 || error("place: non-zero digit above the window")
        end
    end
    return w, st
end

"""left shift by n ≥ 0 bits into LW digits, exact (errors if a non-zero digit would leave the window)."""
@inline shl_by(d::NTuple{N,Float32}, n::Int, v::Val) where {N} = place(d, n, v)[1]

@inline function cmp_digits(a::NTuple{N,Float32}, b::NTuple{N,Float32}) where {N}
    @inbounds for i in N:-1:1
        a[i] > b[i] && return 1
        a[i] < b[i] && return -1
    end
    return 0
end

"""position of the leading bit (LSB of digit 1 = position 0), −1 for zero."""
@inline function topbit(d::NTuple{N,Float32}) where {N}
    @inbounds for i in N:-1:1
        d[i] != 0f0 && return DB * (i - 1) + exponent(d[i])
    end
    return -1
end

"""add a small integer to a non-negative digit tuple (fixed width: a carry out is an error)."""
@inline function add_int(d::NTuple{N,Float32}, v::Int) where {N}
    dd, co = carry_norm(tsetfirst(d, d[1] + Float32(v)))
    co == 0f0 || error(co > 0f0 ? "add_int: carry out of the fixed width" : "add_int: negative result")
    return dd
end

@inline isodd_digits(d::NTuple{N,Float32}) where {N} = splitbit(d[1], 1)[2] == 1f0
@inline lowbits(d::NTuple{N,Float32}, s::Int) where {N} = splitbit(d[1], s)[2]
@inline iszero_digits(d::NTuple{N,Float32}) where {N} = tallzero(d)

"""exact (da·2^ea) − (db·2^eb) in an LW-digit window → (neg, digits, e0)."""
@inline function wide_sub(da::NTuple{NA,Float32}, ea::Int, db::NTuple{NB,Float32}, eb::Int, v::Val{LW}) where {NA,NB,LW}
    e0 = min(ea, eb)
    A = shl_by(da, ea - e0, v); B = shl_by(db, eb - e0, v)
    c = cmp_digits(A, B)
    c == 0 && return (false, ntuple(_ -> 0f0, v), e0)
    neg = c < 0
    if neg; A, B = B, A; end
    r, co = carry_norm(tmap(-, A, B))
    co == 0f0 || error("wide_sub: borrow out")
    return (neg, r, e0)
end

@inline function wide_cmp(da::NTuple{NA,Float32}, ea::Int, db::NTuple{NB,Float32}, eb::Int, v::Val) where {NA,NB}
    neg, r, _ = wide_sub(da, ea, db, eb, v)
    iszero_digits(r) && return 0
    return neg ? -1 : 1
end

# ---------------------------------------------------------------------------------------
# 3. the number type
# ---------------------------------------------------------------------------------------
struct MFloat{P,K,EMAX} <: Real
    neg::Bool
    ex::Int32                 # exponent of the leading bit; 0 for zero
    d::NTuple{K,Float32}      # little-endian 18-bit digits, canonical (d[K] ∈ [2^17, 2^18) unless zero)
    flag::UInt8
    function MFloat{P,K,EMAX}(neg::Bool, ex::Integer, d::NTuple{K,Float32}, flag::UInt8) where {P,K,EMAX}
        (DB * K ≥ P + 12 && K ≤ 31 && P ≥ 24) || error("MFloat{P,K,EMAX}: need 18K ≥ P+12, K ≤ 31")
        new{P,K,EMAX}(neg, Int32(ex), d, flag)
    end
end

const F128 = MFloat{113, 7, 16383}
const F256 = MFloat{237, 14, 262143}
const F512 = MFloat{489, 28, 4194303}

prec(::Type{MFloat{P,K,E}}) where {P,K,E} = P
ndig(::Type{MFloat{P,K,E}}) where {P,K,E} = K
emax(::Type{MFloat{P,K,E}}) where {P,K,E} = E
emin(::Type{MFloat{P,K,E}}) where {P,K,E} = 1 - E
Base.precision(::Type{T}) where {T<:MFloat} = prec(T)
Base.precision(::T) where {T<:MFloat} = prec(T)

zerod(::Type{T}) where {T<:MFloat} = ntuple(_ -> 0f0, Val(ndig(T)))
function maxd(::Type{T}) where {T<:MFloat}
    P, K = prec(T), ndig(T)
    ntuple(Val(K)) do i                     # digit i (from the bottom): bits kept = clamp(P − 18(K−i), 0, 18)
        nb = clamp(P - DB * (K - i), 0, DB)
        Float32((2^nb - 1) * 2^(DB - nb))
    end
end
mind(::Type{T}) where {T<:MFloat} = ntuple(i -> i == ndig(T) ? Float32(2^(DB - 1)) : 0f0, Val(ndig(T)))

Base.zero(::Type{T}) where {T<:MFloat} = T(false, 0, zerod(T), 0x00)
Base.one(::Type{T}) where {T<:MFloat} = T(false, 0, mind(T), 0x00)
Base.zero(::T) where {T<:MFloat} = zero(T)
Base.one(::T) where {T<:MFloat} = one(T)
Base.typemax(::Type{T}) where {T<:MFloat} = T(false, emax(T), maxd(T), GE)
Base.typemin(::Type{T}) where {T<:MFloat} = T(true, emax(T), maxd(T), GE)
Base.floatmax(::Type{T}) where {T<:MFloat} = T(false, emax(T), maxd(T), 0x00)
Base.floatmin(::Type{T}) where {T<:MFloat} = T(false, emin(T), mind(T), 0x00)
Base.eps(::Type{T}) where {T<:MFloat} = T(false, 1 - prec(T), mind(T), 0x00)
Base.iszero(x::T) where {T<:MFloat} = x.d[ndig(T)] == 0f0
flag_of(x::MFloat) = x.flag
isflagged(x::MFloat) = x.flag != 0x00
Base.isnan(::MFloat) = false
Base.isinf(::MFloat) = false
Base.isfinite(::MFloat) = true

e0_of(x::T) where {T<:MFloat} = Int(x.ex) - DB * ndig(T) + 1          # weight of the LSB of digit 1

"""sticky || any non-zero digit among d[1..i]."""
@inline function below(d::NTuple{N,Float32}, i::Int, sticky::Bool) where {N}
    sticky && return true
    @inbounds for k in 1:i
        d[k] != 0f0 && return true
    end
    return false
end

"""normalize + round-to-nearest-even at Pr bits (exact tail: digits + sticky) + saturate → T
(flag = the saturation flag only).  d is little-endian with LSB weight 2^e0."""
@noinline function round_pack(::Type{T}, neg::Bool, d::NTuple{N,Float32}, e0::Int, sticky::Bool, Pr::Int) where {T<:MFloat,N}
    K = ndig(T)
    L = N
    @inbounds while L ≥ 1 && d[L] == 0f0; L -= 1; end
    if L == 0
        sticky && error("round_pack: zero window with a non-zero sticky (window too narrow)")
        return zero(T)
    end
    t = e0 + DB * (L - 1) + exponent(@inbounds d[L])   # leading bit
    u = t - Pr + 1                                     # ulp position
    dd = tappend(d, 0f0)                               # one spare digit for the rounding carry
    if u > e0                                          # something to round inside the window
        iu = fld(u - e0, DB) + 1; bu = mod(u - e0, DB)
        q, r = splitbit(@inbounds(dd[iu]), bu)         # d[iu] = q·2^bu + r
        if bu ≥ 1
            half = POW2[bu]                            # 2^(bu−1)
            cmp = r > half ? 1 : (r < half ? -1 : (below(dd, iu - 1, sticky) ? 1 : 0))
        elseif iu ≥ 2
            x = @inbounds dd[iu - 1]; half = POW2[DB]
            cmp = x > half ? 1 : (x < half ? -1 : (below(dd, iu - 2, sticky) ? 1 : 0))
        else
            sticky && error("round_pack: round bit below the window")
            cmp = -1
        end
        up = cmp > 0 || (cmp == 0 && splitbit(q, 1)[2] == 1f0)
        kept = q * POW2[bu + 1] + (up ? POW2[bu + 1] : 0f0)
        dr = ntuple(Val(N + 1)) do i                    # digits below the ulp → 0, digit iu → kept
            i < iu ? 0f0 : (i == iu ? kept : (@inbounds dd[i]))
        end
        if up
            dr, co = carry_norm(dr)
            co == 0f0 || error("round_pack: carry out of the widened window")
            L = N + 1
            @inbounds while dr[L] == 0f0; L -= 1; end
            t = e0 + DB * (L - 1) + exponent(@inbounds dr[L])
        end
    else
        sticky && error("round_pack: round bit below the window")
        dr = dd
    end
    # pack: leading bit → bit 17 of digit K   (new LSB weight t − 18K + 1)
    w, st = place(dr, e0 - (t - DB * K + 1), Val(K))
    st && error("round_pack: non-zero bits dropped while packing")
    if t > emax(T)
        return T(neg, emax(T), maxd(T), GE)
    elseif t < emin(T)
        return T(neg, emin(T), mind(T), LE)
    end
    return T(neg, t, w, 0x00)
end

# ---------------------------------------------------------------------------------------
# 4. conversions
# ---------------------------------------------------------------------------------------
function digits_of(n::Integer)
    n = big(n); n ≥ 0 || error("digits_of: negative")
    d = Float32[]
    while n > 0
        push!(d, Float32(n & 0x3FFFF)); n >>= DB
    end
    return d
end

"""non-negative integer → L little-endian digits (test helper; errors if it does not fit)."""
function digits_tuple(n::Integer, ::Val{L}) where {L}
    d = digits_of(n)
    length(d) ≤ L || error("digits_tuple: does not fit")
    ntuple(i -> i ≤ length(d) ? d[i] : 0f0, Val(L))
end

function (::Type{T})(x::Float64) where {T<:MFloat}
    isnan(x) && return T(false, 0, zerod(T), GE | LE | SUNK)
    isinf(x) && return T(x < 0, emax(T), maxd(T), GE)
    x == 0 && return zero(T)
    num, pow, den = Base.decompose(x)                  # num ≥ 0 (53 bits), den = ±1
    d = (Float32(num & 0x3FFFF), Float32((num >> DB) & 0x3FFFF), Float32(num >> (2 * DB)))
    round_pack(T, den < 0, d, pow, false, prec(T))     # no BigInt: three digits from Int64
end
(::Type{T})(x::Float32) where {T<:MFloat} = T(Float64(x))
(::Type{T})(x::Float16) where {T<:MFloat} = T(Float64(x))
function (::Type{T})(x::BigFloat) where {T<:MFloat}
    isnan(x) && return T(false, 0, zerod(T), GE | LE | SUNK)
    isinf(x) && return T(x < 0, emax(T), maxd(T), GE)
    x == 0 && return zero(T)
    num, pow, den = Base.decompose(x)
    round_pack(T, den < 0, Tuple(digits_of(abs(num))), pow, false, prec(T))   # host only (BigInt)
end
function (::Type{T})(x::Integer) where {T<:MFloat}
    x == 0 && return zero(T)
    round_pack(T, x < 0, Tuple(digits_of(abs(big(x)))), 0, false, prec(T))
end
(::Type{T})(x::Rational) where {T<:MFloat} = T(numerator(x)) / T(denominator(x))
(::Type{T})(x::T) where {T<:MFloat} = x
(::Type{T})(x::Real) where {T<:MFloat} = T(BigFloat(x))

function Base.BigFloat(x::T) where {T<:MFloat}
    K = ndig(T)
    setprecision(BigFloat, DB * K + 16) do
        acc = BigFloat(0)
        for i in K:-1:1
            acc = acc * RADIX + x.d[i]
        end
        v = ldexp(acc, Int(x.ex) - DB * K + 1)
        x.neg ? -v : v
    end
end
Base.Float64(x::MFloat) = Float64(BigFloat(x))
Base.Float32(x::MFloat) = Float32(BigFloat(x))
Base.big(x::MFloat) = BigFloat(x)
Base.promote_rule(::Type{T}, ::Type{<:Real}) where {T<:MFloat} = T
Base.promote_rule(::Type{BigFloat}, ::Type{T}) where {T<:MFloat} = T   # both orders agree → no promotion loop

function Base.show(io::IO, x::T) where {T<:MFloat}
    v = BigFloat(BigFloat(x); precision = prec(T))
    print(io, string(v))
    x.flag == 0 && return
    print(io, "⟦", (x.flag & GE) > 0 ? "≥" : "", (x.flag & LE) > 0 ? "≤" : "",
              (x.flag & SUNK) > 0 ? "±" : "", (x.flag & CPLX) > 0 ? "ℂ" : "", "⟧")
end

# ---------------------------------------------------------------------------------------
# 5. comparisons (values only — flags are not ordered, as in ScalarTot)
# ---------------------------------------------------------------------------------------
function cmp_mag(a::T, b::T) where {T<:MFloat}
    za, zb = iszero(a), iszero(b)
    (za && zb) && return 0
    za && return -1
    zb && return 1
    a.ex != b.ex && return a.ex > b.ex ? 1 : -1
    @inbounds for i in ndig(T):-1:1
        a.d[i] > b.d[i] && return 1
        a.d[i] < b.d[i] && return -1
    end
    return 0
end
function Base.:<(a::T, b::T) where {T<:MFloat}
    a.neg != b.neg && return a.neg
    c = cmp_mag(a, b)
    a.neg ? c > 0 : c < 0
end
Base.:(==)(a::T, b::T) where {T<:MFloat} = a.neg == b.neg && a.ex == b.ex && a.d == b.d
Base.:<=(a::T, b::T) where {T<:MFloat} = a < b || a == b
Base.isless(a::T, b::T) where {T<:MFloat} = a < b
Base.abs(a::T) where {T<:MFloat} = T(false, a.ex, a.d, a.flag)
Base.:-(a::T) where {T<:MFloat} = T(!a.neg && !iszero(a), a.ex, a.d, a.flag)
Base.sign(a::T) where {T<:MFloat} = iszero(a) ? zero(T) : T(a.neg, 0, mind(T), a.flag & SUNK)
Base.signbit(a::MFloat) = a.neg

# ---------------------------------------------------------------------------------------
# 6. flag algebra — ported from ScalarTot.jl (GE/LE are |·|-bounds; cancellation-safe)
# ---------------------------------------------------------------------------------------
@inline function addflag(a::MFloat, b::MFloat, flipb::Bool)
    fin = a.flag | b.flag
    fin == 0x00 && return 0x00
    same = !iszero(a) && !iszero(b) && (a.neg == (b.neg ⊻ flipb))
    known = (fin & SUNK) == 0x00
    (known && same) ? fin : (GE | LE | SUNK)
end
@inline function mulflag(fa::UInt8, fb::UInt8)
    ga = fa & GE; la = (fa >> 1) & 0x01
    gb = fb & GE; lb = (fb >> 1) & 0x01
    ge = ((ga | gb) & ~(la | lb)) & 0x01
    le = ((la | lb) & ~(ga | gb)) & 0x01
    nb = ((ga | gb) & (la | lb)) & 0x01
    (ge * GE) | (le * LE) | (nb * (GE | LE)) | ((fa | fb) & SUNK)
end
@inline sign_untrusted(a::MFloat) = (a.flag & SUNK) != 0 || (iszero(a) && (a.flag & GE) != 0)

# ---------------------------------------------------------------------------------------
# 7. the operations (cores return a T whose flag is the saturation flag only)
# ---------------------------------------------------------------------------------------
"""a + b (flipb: a − b), exact window + sticky, rounded at Pr bits."""
@noinline function add_core(::Type{T}, a::T, b::T, flipb::Bool, Pr::Int) where {P,K,E,T<:MFloat{P,K,E}}
    negb = b.neg ⊻ flipb
    if iszero(a)
        iszero(b) && return zero(T)
        return T(negb, b.ex, b.d, 0x00)
    end
    iszero(b) && return T(a.neg, a.ex, a.d, 0x00)
    if a.ex ≥ b.ex
        A, negA, B, negB = a, a.neg, b, negb
    else
        A, negA, B, negB = b, negb, a, a.neg
    end
    Δ = Int(A.ex) - Int(B.ex)
    e0w = (Int(A.ex) - DB * K + 1) - 2 * DB                  # window LSB
    wA = tprepend2(A.d)                                      # L = K+2 digits
    wB, stB = place(B.d, 2 * DB - Δ, Val(K + 2))
    if negA == negB
        s, co = carry_norm(tmap(+, wA, wB))
        return round_pack(T, negA, tappend(s, co), e0w, stB, Pr)
    end
    if Δ == 0
        c = cmp_digits(wA, wB)
        c == 0 && return zero(T)
        if c < 0; wA, wB = wB, wA; negA = negB; end
    end
    s = tmap(-, wA, wB)
    stB && (s = tsetfirst(s, s[1] - 1f0))                    # the sticky is a borrow of one window-ulp …
    s, co = carry_norm(s)
    co == 0f0 || error("add_core: borrow out")
    return round_pack(T, negA, s, e0w, stB, Pr)              # … and stays sticky (remainder ∈ (0, ulp_w))
end

@noinline function mul_core(::Type{T}, a::T, b::T, Pr::Int) where {P,K,E,T<:MFloat{P,K,E}}
    (iszero(a) || iszero(b)) && return zero(T)
    p = mul_digits(a.d, b.d)                                 # 2K digits, exact
    e0 = (Int(a.ex) - DB * K + 1) + (Int(b.ex) - DB * K + 1)
    round_pack(T, a.neg ⊻ b.neg, p, e0, false, Pr)
end

half(x::T) where {T<:MFloat} = iszero(x) ? x : T(x.neg, Int(x.ex) - 1, x.d, x.flag)

"""Float64 of the top three digits (≈ 54 bits), exponent applied."""
function top_float64(m::T) where {T<:MFloat}
    K = ndig(T); d = m.d
    v = Float64(d[K]) * 2.0^-17 + Float64(d[K - 1]) * 2.0^-35 + Float64(d[K - 2]) * 2.0^-53
    ldexp(v, Int(m.ex))
end
newton_iters(::Type{T}) where {T<:MFloat} = 1 + ceil(Int, log2(DB * ndig(T) / 50))

"""≈ 1/m at 18K bits (m: significand with ex = 0, m ∈ [1,2))."""
function recip_sig(::Type{T}, m::T) where {T<:MFloat}
    Pr = DB * ndig(T)
    y = T(1.0 / top_float64(m))
    o = one(T)
    for _ in 1:newton_iters(T)
        e = add_core(T, o, mul_core(T, m, y, Pr), true, Pr)      # 1 − m·y
        y = add_core(T, y, mul_core(T, y, e, Pr), false, Pr)     # y + y·e
    end
    return y
end

"""≈ 1/√m at 18K bits (m ∈ [1,4), ex ∈ {0,1})."""
function rsqrt_sig(::Type{T}, m::T) where {T<:MFloat}
    Pr = DB * ndig(T)
    y = T(1.0 / sqrt(top_float64(m)))
    o = one(T)
    for _ in 1:newton_iters(T) + 1
        y2 = mul_core(T, y, y, Pr)
        e = add_core(T, o, mul_core(T, m, y2, Pr), true, Pr)     # 1 − m·y²
        y = add_core(T, y, half(mul_core(T, y, e, Pr)), false, Pr)
    end
    return y
end

"""R = asig − Q·m·2^-P as (neg, digits, e0) in a 2K+2-digit window."""
@noinline function div_resid(da::NTuple{K,Float32}, e0s::Int, Q::NTuple{K1,Float32}, db::NTuple{K,Float32}, P::Int) where {K,K1}
    pm = mul_digits(Q, db)                              # 2K+1 digits
    wide_sub(da, e0s, pm, e0s - P, Val(2 * K + 2))
end

"""|a| / |b| correctly rounded at P (b ≠ 0), with the sign; flag = saturation only.
Returns (value, number of residual fix-ups of the Newton candidate) — the count is diagnostic."""
@noinline function div_core(::Type{T}, a::T, b::T) where {P,K,E,T<:MFloat{P,K,E}}
    e0s = 1 - DB * K
    m    = T(false, 0, b.d, 0x00)
    asig = T(false, 0, a.d, 0x00)
    y  = recip_sig(T, m)
    qh = mul_core(T, asig, y, DB * K)                  # q̂ ∈ (1/2, 2)
    Q, _ = place(qh.d, e0_of(qh) + P, Val(K + 1))      # Q = ⌊q̂·2^P⌋   (weight 2^-P)
    da = a.d; db = b.d
    negR, dR, eR = div_resid(da, e0s, Q, db, P)        # R = asig − Q·m·2^-P
    fix = 0
    while negR                                          # Q too large
        Q = add_int(Q, -1); negR, dR, eR = div_resid(da, e0s, Q, db, P); fix += 1
        fix > 8 && error("div_core: candidate too far (−)")
    end
    while wide_cmp(dR, eR, db, e0s - P, Val(2 * K + 2)) ≥ 0    # R ≥ m·2^-P → Q too small
        Q = add_int(Q, 1); negR, dR, eR = div_resid(da, e0s, Q, db, P); fix += 1
        fix > 8 && error("div_core: candidate too far (+)")
    end
    # now Q·2^-P ≤ q_exact < (Q+1)·2^-P with remainder R ∈ [0, m·2^-P)
    if topbit(Q) < P                                    # q < 1 : ulp = 2^-P, round bit = 2R vs m·2^-P
        c = wide_cmp(dR, eR + 1, db, e0s - P, Val(2 * K + 3))
        (c > 0 || (c == 0 && isodd_digits(Q))) && (Q = add_int(Q, 1))
    else                                                # q ≥ 1 : ulp = 2^(1-P) = two grid steps
        if isodd_digits(Q)
            if iszero_digits(dR)                         # exact midpoint between Q−1 and Q+1 → the one ≡ 0 (mod 4)
                Q = add_int(Q, lowbits(Q, 2) == 1f0 ? -1 : 1)
            else
                Q = add_int(Q, 1)
            end
        end
    end
    round_pack(T, a.neg ⊻ b.neg, Q, -P + Int(a.ex) - Int(b.ex), false, P), fix
end

"""R = m′ − S²·2^{2g} as (neg, digits, e0) in a 2K+2-digit window."""
@noinline function sqrt_resid(dm::NTuple{K,Float32}, em::Int, S::NTuple{K1,Float32}, g::Int) where {K,K1}
    ss = mul_digits(S, S)                               # 2K+2 digits
    wide_sub(dm, em, ss, 2 * g, Val(2 * K + 2))
end

"""√|a| correctly rounded at P (a ≠ 0); flag = saturation only.  Returns (value, fix-up count)."""
@noinline function sqrt_core(::Type{T}, a::T) where {P,K,E,T<:MFloat{P,K,E}}
    odd = isodd(Int(a.ex))
    mp  = T(false, odd ? 1 : 0, a.d, 0x00)             # m' ∈ [1,4), √m' ∈ [1,2)
    exr = fld(Int(a.ex), 2)
    r  = rsqrt_sig(T, mp)
    sh = mul_core(T, mp, r, DB * K)                     # ŝ ≈ √m'
    g = 1 - P                                           # grid = ulp of [1,2) at P bits
    S, _ = place(sh.d, e0_of(sh) - g, Val(K + 1))       # S = ⌊ŝ / 2^g⌋
    dm = mp.d; em = e0_of(mp)
    negR, dR, eR = sqrt_resid(dm, em, S, g)             # R = m' − S²·2^{2g}
    fix = 0
    while negR
        S = add_int(S, -1); negR, dR, eR = sqrt_resid(dm, em, S, g); fix += 1
        fix > 8 && error("sqrt_core: candidate too far (−)")
    end
    while true                                          # (S+1)² ≤ m'/2^{2g}  ⟺  R ≥ 2S+1
        t = add_int(shl_bits(S, 1), 1)
        wide_cmp(dR, eR, t, 2 * g, Val(2 * K + 2)) ≥ 0 || break
        S = add_int(S, 1); negR, dR, eR = sqrt_resid(dm, em, S, g); fix += 1
        fix > 8 && error("sqrt_core: candidate too far (+)")
    end
    f = add_int(shl_bits(S, 2), 1)                      # round up ⟺ 4R > 4S+1  (never equal)
    wide_cmp(dR, eR + 2, f, 2 * g, Val(2 * K + 3)) > 0 && (S = add_int(S, 1))
    round_pack(T, false, S, g + exr, false, P), fix
end

# --- user-facing operators: value core + ScalarTot's flag rules --------------------------
function Base.:+(a::T, b::T) where {T<:MFloat}
    r = add_core(T, a, b, false, prec(T))
    T(r.neg, r.ex, r.d, r.flag | addflag(a, b, false))
end
function Base.:-(a::T, b::T) where {T<:MFloat}
    r = add_core(T, a, b, true, prec(T))
    T(r.neg, r.ex, r.d, r.flag | addflag(a, b, true))
end
function Base.:*(a::T, b::T) where {T<:MFloat}
    tz = (iszero(a) && (a.flag & GE) == 0) || (iszero(b) && (b.flag & GE) == 0)
    tz && return zero(T)
    r = mul_core(T, a, b, prec(T))
    T(r.neg, r.ex, r.d, r.flag | mulflag(a.flag, b.flag))
end
function Base.:/(a::T, b::T) where {T<:MFloat}
    bz = iszero(b)
    r = (bz || iszero(a)) ? zero(T) : div_core(T, a, b)[1]       # a/0 = 0
    fin = a.flag | b.flag
    nb = (fin & (GE | LE)) > 0
    dz = (iszero(a) && (a.flag & GE) > 0) || (bz && (b.flag & GE) > 0)
    f = r.flag | (nb ? (GE | LE) : 0x00) | (fin & SUNK) | (dz ? SUNK : 0x00)
    T(r.neg, r.ex, r.d, f)
end
function Base.sqrt(a::T) where {T<:MFloat}
    if a.neg && !iszero(a)
        return (a.flag & SUNK) != 0 ? T(false, 0, zerod(T), NOB | SUNK | CPLX) : T(false, 0, zerod(T), a.flag | CPLX)
    end
    iszero(a) && return sign_untrusted(a) ? T(false, 0, zerod(T), NOB | SUNK | CPLX) : zero(T)
    r = sqrt_core(T, a)[1]
    sign_untrusted(a) && return T(false, r.ex, r.d, r.flag | NOB | SUNK | CPLX)
    T(false, r.ex, r.d, r.flag | (a.flag & (GE | LE)))
end
Base.inv(a::T) where {T<:MFloat} = one(T) / a
Base.:^(a::T, n::Integer) where {T<:MFloat} = Base.power_by_squaring(a, n)
Base.literal_pow(::typeof(^), a::T, ::Val{N}) where {T<:MFloat,N} = a^N

# ---------------------------------------------------------------------------------------
# 8. self-test
# ---------------------------------------------------------------------------------------
using Random

"""MPFR oracle: op at precision P with round-to-nearest, unbounded exponent, then ScalarTot's saturation."""
function oracle(::Type{T}, f, args...) where {T<:MFloat}
    r = setprecision(BigFloat, prec(T)) do
        f(map(BigFloat, args)...)
    end
    iszero(r) && return zero(T)
    e = exponent(r)
    e > emax(T) && return T(r < 0, emax(T), maxd(T), GE)
    e < emin(T) && return T(r < 0, emin(T), mind(T), LE)
    T(r)
end

function rand_val(::Type{T}, rng, e::Int) where {T<:MFloat}
    P = prec(T)
    m = rand(rng, big(2)^(P - 1):big(2)^P - 1)
    setprecision(BigFloat, P + 8) do                 # ldexp allocates at the *default* precision
        T(ldexp(BigFloat(m), e - P + 1))
    end
end
rand_exp(::Type{T}, rng) where {T<:MFloat} = rand(rng, -emax(T) ÷ 2:emax(T) ÷ 2)

function check(name, got::T, want::T, log) where {T<:MFloat}
    ok = (got == want) && got.flag == want.flag
    ok || push!(log, (name, got, want))
    ok
end

function twin_check(S)
    bad = 0
    for fa in 0x00:0x07, fb in 0x00:0x07, sa in (-1.0, 1.0), sb in (-1.0, 1.0)
        a = S.TotNum(sa * 3.0, fa); b = S.TotNum(sb * 5.0, fb)
        A = F128(sa < 0, F128(3.0).ex, F128(3.0).d, fa)
        B = F128(sb < 0, F128(5.0).ex, F128(5.0).d, fb)
        (a + b).flag == (A + B).flag || (bad += 1)
        (a - b).flag == (A - B).flag || (bad += 1)
        (a * b).flag == (A * B).flag || (bad += 1)
        (a / b).flag == (A / B).flag || (bad += 1)
        sqrt(a).flag == sqrt(A).flag || (bad += 1)
    end
    return bad
end

function self_test(; n = 1000, seed = 20260903)
    rng = MersenneTwister(seed)
    println("MultiF32 self-test (NTuple kernels) — digits: $DB bits in Float32, radix 2^$DB, K ≤ 31")
    # 1. digit primitives vs Int64 / BigInt
    let bad = 0
        for _ in 1:200_000
            a = rand(rng, 0:2^DB - 1); b = rand(rng, 0:2^DB - 1)
            q, r = mulsplit18(Float32(a), Float32(b))
            (Int(q) == (a * b) >> DB && Int(r) == (a * b) & (2^DB - 1)) || (bad += 1)
            v = rand(rng, -2^23:2^24 - 1)
            q2, r2 = split18(Float32(v))
            (Int(q2) == fld(v, 2^DB) && Int(r2) == mod(v, 2^DB)) || (bad += 1)
            s = rand(rng, 0:DB); d = rand(rng, 0:2^DB - 1)
            q3, r3 = splitbit(Float32(d), s)
            (Int(q3) == d >> s && Int(r3) == d & (2^s - 1)) || (bad += 1)
        end
        for a in (0, 1, 2^DB - 1, 2^17, 2^17 - 1), b in (0, 1, 2^DB - 1, 2^17)
            q, r = mulsplit18(Float32(a), Float32(b))
            (Int(q) == (a * b) >> DB && Int(r) == (a * b) & (2^DB - 1)) || (bad += 1)
        end
        for (la, lb) in ((1, 1), (2, 3), (7, 7), (8, 7), (14, 14), (15, 14), (28, 28), (29, 28), (31, 31), (5, 31)), _ in 1:200
            A = rand(rng, big(0):big(2)^(DB * la) - 1); B = rand(rng, big(0):big(2)^(DB * lb) - 1)
            da = digits_tuple(A, Val(la)); dbv = digits_tuple(B, Val(lb))
            p = mul_digits(da, dbv)
            val = sum(big(Int(p[i])) << (DB * (i - 1)) for i in eachindex(p))
            val == A * B || (bad += 1)
            sh = rand(rng, 0:60)
            w, st = place(da, sh, Val(la + 4))
            wv = sum(big(Int(w[i])) << (DB * (i - 1)) for i in eachindex(w))
            (wv == (A << sh) && !st) || (bad += 1)
            w2, st2 = place(da, -sh, Val(la))
            wv2 = sum(big(Int(w2[i])) << (DB * (i - 1)) for i in eachindex(w2))
            (wv2 == (A >> sh) && st2 == ((A & (big(2)^sh - 1)) != 0)) || (bad += 1)
        end
        println("  primitives (mulsplit18 / split18 / splitbit / mul_digits / place): bad = $bad")
        bad == 0 || return false
    end
    allok = true
    for T in (F128, F256, F512)
        P, K = prec(T), ndig(T)
        log = Any[]
        println("  $(T): P=$P bits, K=$K digits ($(DB*K) bits), emax=$(emax(T))")
        # 2. conversion round trip
        for _ in 1:300
            x = rand_val(T, rng, rand_exp(T, rng))
            check("roundtrip", T(BigFloat(x)), x, log)
            BigFloat(T(BigFloat(x))) == BigFloat(x) || push!(log, ("roundtrip-bf", x, x))
        end
        # 3. random operands vs MPFR
        let w = rand_val(T, rng, 0), v = rand_val(T, rng, 1)                 # JIT warm-up (not timed)
            w + v; w - v; w * v; w / v; sqrt(w)
        end
        t_add = t_mul = t_div = t_sqrt = 0.0
        mf_div = mf_sqrt = 0                                                 # max residual fix-ups seen
        for i in 1:n
            ea = rand_exp(T, rng)
            eb = i % 2 == 0 ? ea + rand(rng, -(P + 40):(P + 40)) : rand_exp(T, rng)
            a = rand_val(T, rng, ea); b = rand_val(T, rng, eb)
            rand(rng) < 0.5 && (a = -a); rand(rng) < 0.5 && (b = -b)
            t0 = time(); s = a + b; d = a - b; t_add += time() - t0
            check("add", s, oracle(T, +, a, b), log); check("sub", d, oracle(T, -, a, b), log)
            t0 = time(); p = a * b; t_mul += time() - t0
            check("mul", p, oracle(T, *, a, b), log)
            t0 = time(); q = a / b; t_div += time() - t0
            check("div", q, oracle(T, /, a, b), log)
            t0 = time(); r = sqrt(abs(a)); t_sqrt += time() - t0
            check("sqrt", r, oracle(T, sqrt, abs(a)), log)
            mf_div = max(mf_div, div_core(T, a, b)[2]); mf_sqrt = max(mf_sqrt, sqrt_core(T, abs(a))[2])
        end
        # 4. adversarial constructions
        o = one(T); ulp1 = eps(T)
        for _ in 1:200
            e = rand(rng, -20:20)
            a = rand_val(T, rng, e)
            b = a - T(ldexp(BigFloat(1), e - P + 1))                    # a − 1 ulp
            check("cancel-1ulp", a - b, oracle(T, -, a, b), log)
            c = T(ldexp(BigFloat(1), e))                                 # 2^e
            tiny = rand_val(T, rng, e - P - rand(rng, 1:60))
            check("sticky-sub", c - tiny, oracle(T, -, c, tiny), log)    # just below a binade boundary
            check("sticky-add", c + tiny, oracle(T, +, c, tiny), log)
            check("sticky-sub2", (c - tiny) - tiny, oracle(T, -, oracle(T, -, c, tiny), tiny), log)
            hlf = T(ldexp(BigFloat(1), e - P))                           # exactly half an ulp of c
            check("tie-add-even", c + hlf, oracle(T, +, c, hlf), log)
            codd = c + T(ldexp(BigFloat(1), e - P + 1))                  # LSB odd
            check("tie-add-odd", codd + hlf, oracle(T, +, codd, hlf), log)
            check("tie-sub", codd - hlf, oracle(T, -, codd, hlf), log)
            # multiplication tie: odd x, y with bitlength(x·y) = P+1  → exact midpoint
            hi = (P + 1) ÷ 2
            while true
                x = rand(rng, big(2)^(hi - 1):big(2)^hi - 1) | 1
                y = rand(rng, big(2)^(P - hi):big(2)^(P + 1 - hi) - 1) | 1
                ndigits(x * y, base = 2) == P + 1 || continue
                X = T(x); Y = T(y)
                check("tie-mul", X * Y, oracle(T, *, X, Y), log)
                break
            end
            # exact quotient / exact root
            x = rand_val(T, rng, 0); y = T(rand(rng, 1:2^20))
            xy = x * y
            check("exact-div", xy / y, oracle(T, /, xy, y), log)
            sq = T(rand(rng, big(2)^(P ÷ 2 - 1):big(2)^(P ÷ 2) - 1))
            check("exact-sqrt", sqrt(sq * sq), oracle(T, sqrt, sq * sq), log)
            a1 = rand_val(T, rng, 0); b1 = a1 + ulp1 * T(rand(rng, 1:5))       # quotient just below 1
            check("near1-div", a1 / b1, oracle(T, /, a1, b1), log)
            near1 = o + ulp1 * T(rand(rng, -8:8))
            check("near1-sqrt", sqrt(near1), oracle(T, sqrt, near1), log)
            odd3 = T(ldexp(BigFloat(3), 2 * e + 1))
            check("odd-exp-sqrt", sqrt(odd3), oracle(T, sqrt, odd3), log)
            mf_div = max(mf_div, div_core(T, xy, y)[2], div_core(T, a1, b1)[2])
            mf_sqrt = max(mf_sqrt, sqrt_core(T, sq * sq)[2], sqrt_core(T, near1)[2], sqrt_core(T, odd3)[2])
        end
        # 5. range edges and total-arithmetic rules
        MX = floatmax(T); MN = floatmin(T); two = T(2.0); z = zero(T)
        check("MAX+MAX", MX + MX, T(false, emax(T), maxd(T), GE), log)
        check("MAX*2", MX * two, T(false, emax(T), maxd(T), GE), log)
        check("-MAX*MAX", (-MX) * MX, T(true, emax(T), maxd(T), GE), log)
        check("MIN/2", MN / two, T(false, emin(T), mind(T), LE), log)
        check("MIN*MIN", MN * MN, T(false, emin(T), mind(T), LE), log)
        check("MAX/MIN", MX / MN, T(false, emax(T), maxd(T), GE), log)
        check("MIN/MAX", MN / MX, T(false, emin(T), mind(T), LE), log)
        check("sqrt(MIN)", sqrt(MN), oracle(T, sqrt, MN), log)
        check("sqrt(MAX)", sqrt(MX), oracle(T, sqrt, MX), log)
        check("MIN*(1-eps)", MN * (o - ulp1), T(false, emin(T), mind(T), LE), log)
        check("MAX+1ulp", MX + T(ldexp(BigFloat(1), emax(T) - P + 1)), T(false, emax(T), maxd(T), GE), log)
        check("MAX+half-ulp", MX + T(ldexp(BigFloat(1), emax(T) - P)), T(false, emax(T), maxd(T), GE), log)   # ties to even → 2^(emax+1) → GE
        check("1/0", o / z, z, log)
        check("0/0", z / z, z, log)
        check("sqrt(-1)", sqrt(-o), T(false, 0, zerod(T), CPLX), log)
        a_ = rand_val(T, rng, 5)
        check("x-x", a_ - a_, z, log)
        check("x+(-x)", a_ + (-a_), z, log)
        # flag propagation (ScalarTot rules)
        g = T(false, emax(T), maxd(T), GE)
        check("GE+1", g + o, T(false, emax(T), maxd(T), GE), log)
        check("GE-GE", g - g, T(false, 0, zerod(T), GE | LE | SUNK), log)
        l = T(false, emin(T), mind(T), LE)
        check("LE*GE", l * g, T(false, l.ex + g.ex, (l * g).d, GE | LE), log)
        check("sqrt(GE)", sqrt(g), T(false, (sqrt(MX)).ex, (sqrt(MX)).d, GE), log)
        check("1/GE", o / g, T(false, (o / MX).ex, (o / MX).d, GE | LE), log)
        # 6. a few identities at full precision
        x = rand_val(T, rng, 0); y = rand_val(T, rng, 3)
        check("(x/y)*y≈x", (x / y) * y, oracle(T, *, x / y, y), log)
        check("sqrt(x)^2", sqrt(x) * sqrt(x), oracle(T, *, sqrt(x), sqrt(x)), log)
        e400 = T(ldexp(BigFloat(1), -(P - 20)))
        check("(1+2^-(P-20))-1", (o + e400) - o, e400, log)
        nbad = length(log)
        println("    vs MPFR: n=$n random ×(add,sub,mul,div,sqrt) + 200×15 adversarial + edges → mismatches = $nbad" *
                "   [max residual fix-ups: div $mf_div, sqrt $mf_sqrt]")
        println("    time/op: add $(round(1e6*t_add/(2n), digits=1)) µs, mul $(round(1e6*t_mul/n, digits=1)) µs, " *
                "div $(round(1e6*t_div/n, digits=1)) µs, sqrt $(round(1e6*t_sqrt/n, digits=1)) µs  (NTuple kernels; per-op time() included)")
        for (name, got, want) in log[1:min(5, nbad)]
            println("      ✗ $name: got $(got)  want $(want)")
        end
        nbad == 0 || (allok = false)
    end
    # 7. flag algebra twin check against ScalarTot (if present)
    st = joinpath(@__DIR__, "ScalarTot.jl")
    if isfile(st)
        m = Module(); Base.include(m, st)
        bad = Base.invokelatest(twin_check, m.ScalarTot)          # the module was just defined → world age
        println("  flag algebra vs ScalarTot.TotNum (8×8 flags × 4 sign pairs × 5 ops): mismatches = $bad")
        bad == 0 || (allok = false)
    end
    # 8. demo
    s2 = sqrt(F512(2))
    ref = setprecision(BigFloat, 489) do; sqrt(BigFloat(2)); end
    println("  √2 in F512 = ", s2)
    println("  == MPFR(489 bits): ", BigFloat(s2) == ref)
    println(allok ? "PASS" : "FAIL")
    return allok
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 1000
    MultiF32.self_test(n = n) || exit(1)
end
