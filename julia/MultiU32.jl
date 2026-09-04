# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
MultiU32 — the algorithms of MultiF32.jl on an **integer digit engine**: 29-bit digits held in
UInt32, digit products and column sums in UInt64 (`mul.wide.u32` / `mad.wide.u32` on an NVIDIA
GPU, one `mul r64` on x86).  This is MPFR's limb arithmetic at the width a GPU multiplies
natively: MPFR's limbs are 64-bit integers chained by the hardware carry flag (`adc`), which
compilers do not expose on a GPU; 29 = 32 − 3 bits per digit buys what 18 = 24 − 6 buys in
MultiF32 — every column sum of a schoolbook product (≤ 32 products of 29-bit digits, < 2^63)
stays exact in the accumulator, so there is no carry handling inside the product at all, one
carry ripple at the end.

  Representation `MFloat{P,K,EMAX}` and everything above the digit engine — canonical form (the
  leading bit at bit 28 of digit K, the low 29K−P bits zero), the flags GE / LE / SUNK / CPLX,
  rounding to nearest even from the exact result, Newton candidate + exact residual for ÷ and
  √, ScalarTot's flag rules — are those of MultiF32.jl (read that file first); only §1–2, the
  digit engine, differ.
      F128 = MFloat{113, 5, 16383}      5 digits = 145 bits   (4 digits = 116 bits would hold
      F256 = MFloat{237, 9, 262143}     9 digits = 261 bits    the value; the Newton candidate
      F512 = MFloat{489, 18, 4194303}  18 digits = 522 bits    wants ≥ 12 guard bits: 29K ≥ P+12)

  Digit engine (§1–2):
    • product: a·b < 2^58 as UInt64, accumulated into 64-bit columns — K×K digits = K² wide
      multiplies + one carry ripple (2K shifts and masks).
    • sums: 32-bit adds, carry = s >> 29;  differences: borrow = top bit of the wrapped difference.
    • shifts and rounding masks: `<<`, `>>`, `&` — no magic constants anywhere.

  Written for a warp, not a core (§2–3, §7): no digit tuple is ever indexed by a data-dependent
  value and no loop has a data-dependent exit — both send the digits through local memory and
  split the warp.  Shifts by a data-dependent number of digits are barrel shifters (⌈log₂ N⌉
  steps of full-width selects); rounding is FPU-style, **normalize first, then round at a static
  bit position** (`round_norm`: round bit, sticky mask, increment digit and the packed digits are
  compile-time constants of P and K); the residual windows of ÷ and √ shift by amounts that
  depend on P and K only; `wide_sub` and the cancellation path of `add_core` compute both A−B and
  B−A and select.  On the RTX 5090 this rewrite was 3.3–5.4× on F512 (add 1.94 → 0.36 ns,
  mul 1.31 → 0.40, div 15.8 → 3.3, sqrt 19.5 → 4.2) with every result bit-identical.  The
  arrays matter as much as the kernels: one array per field (SoA, `bench_multif32_cuda.jl`)
  instead of an array of these structs takes F512 add / mul to 0.21 / 0.29 ns — the add within
  1.3× of the measured memory bandwidth — and leaves div / sqrt where they are (register-bound).

  ÷ and √ (§7): the candidate comes from a Newton **ladder** written in fixed point.  The seed's
  50 bits double to 100 / 200 / 400 …, so a step is only as wide as the accuracy it produces
  (F512: 4, 8, 14, 18 digits, the correction y·e narrower still), and inside the ladder every
  scale is a constant of the type — where the 1 sits in the product, how far the correction is
  shifted in — so nothing is normalized and nothing is rounded until the candidate reaches the
  exact residual.  √ runs the ladder on 1/√m to *half* the width and one Karp–Markstein step
  (s = m·y, then s ← s + (y/2)(m − s²)) to √m itself, which is also the multiplication `sqrt_core`
  no longer does.  Against the plain full-width Newton loop (kept in MultiU32Ref.jl) that is
  **2.3–3.7× on ÷ and 3.6–5.7× on √ with every result bit-identical** — the residual and the
  rounding rule define the answer, the candidate only proposes it, so a cheaper candidate can
  cost a fix-up but never a wrong digit.  What the measurement said and the cost model did not:
  at these widths a rounded operation is mostly its normalize + round_pack, not its digit
  products.  The first version of the ladder narrowed only the multiplies — the model promised
  2.2 full-width products, it measured 10.4 — because 16 operations × ~30 ns of normalization is
  the bill; the fixed point is what removes it.

  exp and log (§7b, 2026-09-03): correctly rounded by Ziv's test — computed in a working type of
  K+2 digits (≥ 70 bits beyond P) and rounded to P only when no midpoint of the P-bit grid lies
  within the error bound; otherwise again at K+6 and 2K+6 digits (the 2P+ bits where the hard
  cases live: exp(2^−P) = 1 + 2^−P + 2^−(2P+1)), and past that the truncated value with ⟦≥⟧ —
  total and honest without an exception (counted; never reached).  exp: k = round(x/ln2),
  r = x − k·ln2 at +58 bits, Taylor after s halvings, s squarings.  log: x = m·2^e with
  m ∈ [1/√2, √2), square roots only until |m′−1| ≈ 2^−(s+2.5) (a root's rounding is absolute and
  m′−1 makes it relative: roots on an m already near 1 would only lose bits), then the odd atanh
  series and e·ln2.  Semantics = ScalarTot's, twin-checked (16×16 flags × 12 values × 7 ops):
  exp 0 = 1, exp(−MAX) = +MIN⟦≤⟧ (never 0), log 1 = 0, **log 0 = 0 by definition** (the reserved
  word: Σ (1/n)((x−1)/x)ⁿ is 0 term by term under a/0 = 0; the limit is ε's: log(MIN⟦≤⟧) =
  log MIN⟦≥⟧), log(−x) = 0⟦ℂ⟧, and ℂ sticks through every operation (z⁰ = 1 excepted).
  MPFR bit-identity on 3 × 22 000 random + adversarial inputs; CPU 1.7–3.3× MPFR C (bench_explog.jl).

  Self-test: `julia MultiU32.jl` — primitives against BigInt (now to 42 digits: the columns are
  folded every 32 rows, so K ≤ 64), the normalize-and-round step against an MPFR reference on
  3600 random windows (with a negative control: a flipped round bit must be reported), then the
  MPFR battery of MultiF32.jl unchanged (constructed ties, 1-ulp cancellations, stickies at binade
  boundaries, exact quotients and roots, the range edges, flag propagation) plus exp / log (random,
  the tiny-|x| and near-1 neighbourhoods, the midpoint cases, the saturation edges) and the
  ScalarTot twin check.
"""
module MultiU32

export MFloat, F128, F256, F512, GE, LE, SUNK, CPLX, isflagged, flag_of, self_test

using Base.Cartesian: @nexprs, @ntuple

const GE   = 0x01
const LE   = 0x02
const SUNK = 0x04
const CPLX = 0x08
const NOB  = GE | LE

# ---------------------------------------------------------------------------------------
# 1. digit primitives — UInt32 digits, UInt64 accumulators
# ---------------------------------------------------------------------------------------
const DB     = 29                                   # bits per digit
const MASK   = UInt32(2^DB - 1)                     # 0x1FFFFFFF
const MASK64 = UInt64(MASK)
const Z32    = zero(UInt32)

"""position of the leading bit of a non-zero digit (0 … 28)."""
@inline lead(x::UInt32) = 31 - leading_zeros(x)

# tuple helpers with literal indices — Base's map / all / splat / tail leave the fast path
# for tuples longer than 32 elements (Base.Any32), which F512's 36-digit products exceed
@inline tmap(f, a::NTuple{N,UInt32}, b::NTuple{N,UInt32}) where {N} = ntuple(i -> f(a[i], b[i]), Val(N))
@inline tappend(d::NTuple{N,UInt32}, x::UInt32) where {N} = ntuple(i -> i ≤ N ? d[i] : x, Val(N + 1))
# `c ? a : b` on whole tuples, as a named function: a @generated body may not contain a closure
@inline tsel(c::Bool, a::NTuple{N,UInt32}, b::NTuple{N,UInt32}) where {N} = ntuple(i -> ifelse(c, a[i], b[i]), Val(N))
@inline tprepend2(d::NTuple{N,UInt32}) where {N} = ntuple(i -> i ≤ 2 ? Z32 : d[i - 2], Val(N + 2))
@inline tsetfirst(d::NTuple{N,UInt32}, x::UInt32) where {N} = ntuple(i -> i == 1 ? x : d[i], Val(N))
@inline function tallzero(d::NTuple{N,UInt32}) where {N}
    @inbounds for i in 1:N
        d[i] != Z32 && return false
    end
    return true
end

# ---------------------------------------------------------------------------------------
# 2. digit tuples (little-endian: v[1] least significant) — static lengths, no heap
#    Every kernel is straight-line integer code once specialised on the tuple length
#    (`@nexprs` unrolling), so an operation allocates nothing: CPU without GC, and the same
#    functions run unchanged inside a CUDA.jl kernel.
# ---------------------------------------------------------------------------------------
"""carry ripple over 64-bit column sums (each < 2^63) → (29-bit digits, carry out)."""
@generated function carry_norm(v::NTuple{N,UInt64}) where {N}
    quote
        Base.@_inline_meta
        c = zero(UInt64)
        @nexprs $N i -> begin
            s = v[i] + c                              # < 2^63 + 2^35: no overflow
            d_i = (s & MASK64) % UInt32
            c = s >> DB
        end
        return (@ntuple($N, i -> d_i), c)
    end
end

"""carry ripple over 32-bit digit sums (each < 2^31) → (digits, carry out ∈ {0, 1})."""
@generated function carry_norm(v::NTuple{N,UInt32}) where {N}
    quote
        Base.@_inline_meta
        c = Z32
        @nexprs $N i -> begin
            s = v[i] + c
            d_i = s & MASK
            c = s >> DB
        end
        return (@ntuple($N, i -> d_i), c)
    end
end

"""a − b − borrow_in with a borrow ripple → (digits, borrow out ∈ {0, 1})."""
@generated function sub_digits(a::NTuple{N,UInt32}, b::NTuple{N,UInt32}, borrow_in::UInt32) where {N}
    quote
        Base.@_inline_meta
        bw = borrow_in
        @nexprs $N i -> begin
            t = a[i] - b[i] - bw                      # wraps iff negative: then bit 31 is set
            d_i = t & MASK                            # 2^32 ≡ 0 (mod 2^29): the wrapped digit is right
            bw = t >> 31
        end
        return (@ntuple($N, i -> d_i), bw)
    end
end

"""a carry pass over UInt64 columns c_1 … c_NC in place (each column left < 2^29): emitted between
row blocks of a long product so that a column never holds more than 32 products + one carry."""
fold_columns(col::Vector{Symbol}) = Expr[:($(col[k + 1]) += $(col[k]) >> DB; $(col[k]) &= MASK64) for k in 1:length(col) - 1]

"""exact schoolbook product (NA+NB digits): NA·NB wide multiplies into 64-bit columns, one carry
ripple.  A column holds one product < 2^58 per row of `a`, so after every 32 rows the columns are
folded (`fold_columns`) and the sums stay below 2^63 at any width."""
@generated function mul_digits(a::NTuple{NA,UInt32}, b::NTuple{NB,UInt32}) where {NA,NB}
    NC = NA + NB
    col = [Symbol(:c_, k) for k in 1:NC]
    body = Expr[:($(col[k]) = zero(UInt64)) for k in 1:NC]
    for i in 1:NA
        push!(body, :($(Symbol(:ai_, i)) = UInt64(a[$i])))
        for j in 1:NB
            push!(body, :($(col[i+j-1]) += $(Symbol(:ai_, i)) * UInt64(b[$j])))   # zext·zext → mul.wide.u32
        end
        (i % 32 == 0 && i < NA) && append!(body, fold_columns(col))
    end
    quote
        Base.@_noinline_meta
        $(body...)
        (c, co) = carry_norm($(Expr(:tuple, col...)))
        co == 0 || error("mul_digits: carry out")
        return c
    end
end

"""the low L digits of the exact product — the columns above L are never formed.  Where the caller
knows the high digits cancel (a Newton residual: m·y is 1 to the accuracy of y), the difference that
would read them is the difference of the low L digits in two's complement, so those columns are work
not done and, on a GPU, registers not spilled."""
@generated function mul_digits_low(a::NTuple{NA,UInt32}, b::NTuple{NB,UInt32}, ::Val{L}) where {NA,NB,L}
    LL = min(L, NA + NB)
    col = [Symbol(:c_, k) for k in 1:LL]
    body = Expr[:($(col[k]) = zero(UInt64)) for k in 1:LL]
    for i in 1:NA
        push!(body, :($(Symbol(:ai_, i)) = UInt64(a[$i])))
        for j in 1:NB
            i + j - 1 ≤ LL && push!(body, :($(col[i+j-1]) += $(Symbol(:ai_, i)) * UInt64(b[$j])))
        end
        (i % 32 == 0 && i < NA) && append!(body, fold_columns(col))
    end
    quote
        Base.@_noinline_meta
        $(body...)
        (c, _) = carry_norm($(Expr(:tuple, col...)))
        return c
    end
end

"""left shift by s bits (0 ≤ s < 29); returns N+1 digits."""
@generated function shl_bits(d::NTuple{N,UInt32}, s::Int) where {N}
    top = Symbol(:o_, N + 1)
    quote
        Base.@_inline_meta
        carry = Z32
        @nexprs $N i -> begin
            o_i = ((d[i] << s) | carry) & MASK
            carry = d[i] >> (DB - s)                  # s = 0: >> 29 of a 29-bit digit = 0
        end
        $top = carry
        return @ntuple($(N + 1), i -> o_i)
    end
end

"""expression for e[lo] | … | e[hi] as a balanced tree of binary ORs.  (An N-ary call `|(e[1], …, e[N])`
with more than 32 arguments leaves Base's unrolled `afoldl` and allocates its varargs — 176 bytes per
call on the 38-digit residual windows of F512.)"""
ortree(e::Symbol, lo::Int, hi::Int) =
    hi < lo ? :(Z32) : lo == hi ? :($e[$lo]) : (m = (lo + hi) ÷ 2; :($(ortree(e, lo, m)) | $(ortree(e, m + 1, hi))))

"""OR of digits lo..hi (static range) — one expression, no loop."""
@generated function or_digits(e::NTuple{N,UInt32}, ::Val{lo}, ::Val{hi}) where {N,lo,hi}
    :(Base.@_inline_meta; $(ortree(:e, lo, hi)))
end

"""first L digits of e; also the OR of the digits beyond L (which the callers require to be zero)."""
@generated function fit_digits(e::NTuple{N,UInt32}, ::Val{L}) where {N,L}
    tup = Expr(:tuple, [i ≤ N ? :(e[$i]) : :(Z32) for i in 1:L]...)
    :(Base.@_inline_meta; ($tup, or_digits(e, Val($(L + 1)), Val($N))))
end

# Digit shifts by a *data-dependent* number of digits are barrel shifters: ⌈log₂(N+1)⌉ steps of
# full-width selects with static indices.  A tuple indexed by a runtime value is spilled to the
# stack and read back (on a GPU: local memory), and a loop over the digits with a data-dependent
# exit diverges inside a warp — on the RTX 5090 the dynamic-index version of `place` and the
# scan loops of `round_pack` cost 3× the 18×18-digit product itself.  Selects stay in registers.

"""right shift by r ≥ 0 digits (r data-dependent); returns (digits, OR of the digits shifted out).
r ≥ N shifts everything out."""
@generated function shr_digits(e::NTuple{N,UInt32}, r::Int) where {N}
    nb = ndigits(N, base = 2)                                  # 2^nb − 1 ≥ N
    body = Expr[]
    for k in 0:nb - 1
        s = 1 << k
        dropped = ortree(:e, 1, min(s, N))
        tup = Expr(:tuple, [i + s ≤ N ? :((e[$i] & ~msk) | (e[$(i + s)] & msk)) : :(e[$i] & ~msk) for i in 1:N]...)
        push!(body, quote
            msk = Z32 - UInt32((r >> $k) & 1)                  # all ones iff this step is taken
            st |= ($dropped) & msk
            e = $tup
        end)
    end
    quote
        Base.@_inline_meta
        r = min(r, $N)
        st = Z32
        $(body...)
        return e, st
    end
end

"""left shift by r ≥ 0 digits (r data-dependent) inside the N digits; returns (digits, OR of the
digits shifted out at the top)."""
@generated function shl_digits(e::NTuple{N,UInt32}, r::Int) where {N}
    nb = ndigits(N, base = 2)
    body = Expr[]
    for k in 0:nb - 1
        s = 1 << k
        dropped = ortree(:e, max(N - s + 1, 1), N)
        tup = Expr(:tuple, [i - s ≥ 1 ? :((e[$i] & ~msk) | (e[$(i - s)] & msk)) : :(e[$i] & ~msk) for i in 1:N]...)
        push!(body, quote
            msk = Z32 - UInt32((r >> $k) & 1)
            ov |= ($dropped) & msk
            e = $tup
        end)
    end
    quote
        Base.@_inline_meta
        r = min(r, $N)
        ov = Z32
        $(body...)
        return e, ov
    end
end

"""d·2^shift placed in an L-digit window whose LSB has weight 2^0 (relative); bits that fall
below the window are folded into the sticky; bits above the window must be zero."""
@inline function place(d::NTuple{N,UInt32}, shift::Int, ::Val{L}) where {N,L}
    ds = fld(shift, DB); s = shift - DB * ds
    e = shl_bits(d, s)                                     # N+1 digits
    e, st = shr_digits(e, max(-ds, 0))                     # right by −ds digits: dropped digits → sticky
    f, ov0 = fit_digits(e, Val(L))                         # window width
    w, ov = shl_digits(f, max(ds, 0))                      # left by ds digits
    (ov0 | ov) == Z32 || error("place: non-zero digit above the window")
    return w, st != Z32
end

"""left shift by n ≥ 0 bits into LW digits, exact (errors if a non-zero digit would leave the window)."""
@inline shl_by(d::NTuple{N,UInt32}, n::Int, v::Val) where {N} = place(d, n, v)[1]

"""right shift by s bits (0 ≤ s < 29) inside the N digits; returns (digits, the s bits shifted out)."""
@generated function shr_bits(d::NTuple{N,UInt32}, s::Int) where {N}
    tup = Expr(:tuple, [i < N ? :((d[$i] >> s) | ((d[$(i + 1)] << (DB - s)) & MASK)) : :(d[$i] >> s) for i in 1:N]...)
    :(Base.@_inline_meta; ($tup, d[1] & ((UInt32(1) << s) - UInt32(1))))   # s = 0: << 29 is masked away, mask = 0
end

"""right shift by n ≥ 0 bits (any size, data-dependent) inside the N digits; returns (digits, sticky)."""
@inline function shr_total(d::NTuple{N,UInt32}, n::Int) where {N}
    ds, s = divrem(n, DB)
    e, lo = shr_bits(d, s)
    f, st = shr_digits(e, ds)
    return f, (lo | st) != Z32
end

"""digits L−W+1..L of e (zero-padded below when L < W) and the OR of the digits below them — static."""
@generated function window_at(e::NTuple{N,UInt32}, ::Val{L}, ::Val{W}) where {N,L,W}
    L ≤ N || error("window_at: L > N")
    tup = Expr(:tuple, [j ≥ 1 ? :(e[$j]) : :(Z32) for j in (L - W + 1):L]...)
    :(Base.@_inline_meta; ($tup, or_digits(e, Val(1), Val($(L - W)))))
end

# Normalization = the floating-point unit's order: bring the leading bit to the top of a fixed
# window first, then everything that follows (round bit, sticky mask, increment) is at a static
# position.  The window is W = K+1 digits: the K canonical digits plus one digit of tail, which
# together with the sticky is all a correctly rounded result needs.

"""d (LSB weight 2^e0; leading digit L holding top ≠ 0, nothing above it) normalized into a W-digit
window with the leading bit at bit 28 of digit W → (window, t = exponent of the leading bit, OR of
the digits that fell below the window).  One bit shift + one barrel digit shift + a static slice."""
@inline function normalize(d::NTuple{N,UInt32}, e0::Int, L::Int, top::UInt32, ::Val{W}) where {N,W}
    lb = lead(top)
    e = shl_bits(d, DB - 1 - lb)                        # N+1 digits: leading bit → bit 28 of digit L
    f, ov = shl_digits(e, N + 1 - L)                    # leading digit → digit N+1
    ov == Z32 || error("normalize: non-zero digit above the leading digit")
    w, dropped = window_at(f, Val(N + 1), Val(W))
    return w, e0 + DB * (L - 1) + lb, dropped
end

"""the same when the leading digit is known statically to be L (mul: the product of two canonical
significands has its leading bit in digit 2K, bit 27 or 28) — no barrel at all."""
@inline function normalize_at(d::NTuple{N,UInt32}, e0::Int, ::Val{L}, ::Val{W}) where {N,L,W}
    top = d[L]
    top != Z32 || error("normalize_at: the leading digit is zero")
    lb = lead(top)
    e = shl_bits(d, DB - 1 - lb)
    w, dropped = window_at(e, Val(L), Val(W))
    return w, e0 + DB * (L - 1) + lb, dropped
end

@inline function cmp_digits(a::NTuple{N,UInt32}, b::NTuple{N,UInt32}) where {N}
    @inbounds for i in N:-1:1
        a[i] > b[i] && return 1
        a[i] < b[i] && return -1
    end
    return 0
end

"""position of the leading bit (LSB of digit 1 = position 0), −1 for zero."""
@inline function topbit(d::NTuple{N,UInt32}) where {N}
    @inbounds for i in N:-1:1
        d[i] != Z32 && return DB * (i - 1) + lead(d[i])
    end
    return -1
end

"""add a small integer to a non-negative digit tuple (fixed width: a carry out is an error)."""
@inline function add_int(d::NTuple{N,UInt32}, v::Int) where {N}
    if v ≥ 0
        dd, co = carry_norm(tsetfirst(d, d[1] + UInt32(v)))
        co == Z32 || error("add_int: carry out of the fixed width")
        return dd
    end
    dd, bo = sub_digits(d, tsetfirst(ntuple(_ -> Z32, Val(N)), UInt32(-v)), Z32)
    bo == Z32 || error("add_int: negative result")
    return dd
end

@inline isodd_digits(d::NTuple{N,UInt32}) where {N} = (d[1] & 0x01) == 0x01
@inline lowbits(d::NTuple{N,UInt32}, s::Int) where {N} = d[1] & ((UInt32(1) << s) - UInt32(1))
@inline iszero_digits(d::NTuple{N,UInt32}) where {N} = tallzero(d)

"""exact (da·2^ea) − (db·2^eb) in an LW-digit window → (neg, |difference| digits, e0).  Both orders
are subtracted and the borrow out selects: no compare loop, no data-dependent branch."""
@inline function wide_sub(da::NTuple{NA,UInt32}, ea::Int, db::NTuple{NB,UInt32}, eb::Int, v::Val{LW}) where {NA,NB,LW}
    e0 = min(ea, eb)
    A = shl_by(da, ea - e0, v); B = shl_by(db, eb - e0, v)
    r1, bo = sub_digits(A, B, Z32)
    r2, _  = sub_digits(B, A, Z32)
    neg = bo == UInt32(1)
    return (neg, tmap((x, y) -> ifelse(neg, y, x), r1, r2), e0)
end

@inline function wide_cmp(da::NTuple{NA,UInt32}, ea::Int, db::NTuple{NB,UInt32}, eb::Int, v::Val) where {NA,NB}
    neg, r, _ = wide_sub(da, ea, db, eb, v)
    z = or_digits(r, Val(1), Val(length(r))) == Z32
    return z ? 0 : (neg ? -1 : 1)
end

# ---------------------------------------------------------------------------------------
# 3. the number type
# ---------------------------------------------------------------------------------------
struct MFloat{P,K,EMAX} <: Real
    neg::Bool
    ex::Int32                 # exponent of the leading bit; 0 for zero
    d::NTuple{K,UInt32}       # little-endian 29-bit digits, canonical (d[K] ∈ [2^28, 2^29) unless zero)
    flag::UInt8
    function MFloat{P,K,EMAX}(neg::Bool, ex::Integer, d::NTuple{K,UInt32}, flag::UInt8) where {P,K,EMAX}
        (DB * K ≥ P + 12 && K ≤ 64 && P ≥ 24) || error("MFloat{P,K,EMAX}: need 29K ≥ P+12, K ≤ 64")
        new{P,K,EMAX}(neg, Int32(ex), d, flag)
    end
end

const F128 = MFloat{113, 5, 16383}
const F256 = MFloat{237, 9, 262143}
const F512 = MFloat{489, 18, 4194303}

prec(::Type{MFloat{P,K,E}}) where {P,K,E} = P
ndig(::Type{MFloat{P,K,E}}) where {P,K,E} = K
emax(::Type{MFloat{P,K,E}}) where {P,K,E} = E
emin(::Type{MFloat{P,K,E}}) where {P,K,E} = 1 - E
Base.precision(::Type{T}) where {T<:MFloat} = prec(T)
Base.precision(::T) where {T<:MFloat} = prec(T)

zerod(::Type{T}) where {T<:MFloat} = ntuple(_ -> Z32, Val(ndig(T)))
function maxd(::Type{T}) where {T<:MFloat}
    P, K = prec(T), ndig(T)
    ntuple(Val(K)) do i                     # digit i (from the bottom): bits kept = clamp(P − 29(K−i), 0, 29)
        nb = clamp(P - DB * (K - i), 0, DB)
        UInt32((2^nb - 1) * 2^(DB - nb))
    end
end
mind(::Type{T}) where {T<:MFloat} = ntuple(i -> i == ndig(T) ? UInt32(2^(DB - 1)) : Z32, Val(ndig(T)))

Base.zero(::Type{T}) where {T<:MFloat} = T(false, 0, zerod(T), 0x00)
Base.one(::Type{T}) where {T<:MFloat} = T(false, 0, mind(T), 0x00)
Base.zero(::T) where {T<:MFloat} = zero(T)
Base.one(::T) where {T<:MFloat} = one(T)
Base.typemax(::Type{T}) where {T<:MFloat} = T(false, emax(T), maxd(T), GE)
Base.typemin(::Type{T}) where {T<:MFloat} = T(true, emax(T), maxd(T), GE)
Base.floatmax(::Type{T}) where {T<:MFloat} = T(false, emax(T), maxd(T), 0x00)
Base.floatmin(::Type{T}) where {T<:MFloat} = T(false, emin(T), mind(T), 0x00)
Base.eps(::Type{T}) where {T<:MFloat} = T(false, 1 - prec(T), mind(T), 0x00)
Base.iszero(x::T) where {T<:MFloat} = x.d[ndig(T)] == Z32
flag_of(x::MFloat) = x.flag
isflagged(x::MFloat) = x.flag != 0x00
Base.isnan(::MFloat) = false
Base.isinf(::MFloat) = false
Base.isfinite(::MFloat) = true

e0_of(x::T) where {T<:MFloat} = Int(x.ex) - DB * ndig(T) + 1          # weight of the LSB of digit 1

"""(index, value) of the most significant non-zero digit, (0, 0) for zero — a select scan, no loop."""
@generated function topdigit(d::NTuple{N,UInt32}) where {N}
    quote
        Base.@_inline_meta
        L = 0; top = Z32
        @nexprs $N i -> begin
            hit = (L == 0) & (d[$N + 1 - i] != Z32)
            L = ifelse(hit, $N + 1 - i, L); top = ifelse(hit, d[$N + 1 - i], top)
        end
        return L, top
    end
end

precval(::Type{MFloat{P,K,E}}) where {P,K,E} = Val(P)            # the working precision …
fullval(::Type{MFloat{P,K,E}}) where {P,K,E} = Val(DB * K)       # … and the digit-array precision (Newton)

"""round-to-nearest-even at Pr bits of a normalized window (W = K+1 digits, leading bit at bit 28 of
digit W, exponent t; sticky = the bits that fell below the window) + saturate → T (flag = the
saturation flag only).  Round bit, sticky mask, increment digit and the K packed digits are all
static positions — no data-dependent index, no loop with a data-dependent exit."""
@generated function round_norm(::Type{T}, neg::Bool, w::NTuple{W,UInt32}, t::Int, sticky::Bool, ::Val{Pr}) where {T<:MFloat,W,Pr}
    K = ndig(T)
    W == K + 1 || error("round_norm: the window is K+1 digits")
    rb = DB * W - Pr - 1;  ir = rb ÷ DB + 1;  br = rb % DB          # round bit = the bit below the kept Pr
    iu = (rb + 1) ÷ DB + 1;  bu = (rb + 1) % DB                     # lsb of the kept part
    (rb ≥ 0 && iu ≥ 2) || error("round_norm: precision does not fit the window")   # Pr ≤ 29K: digit 1 is tail only
    lowmask  = (UInt32(1) << br) - UInt32(1)                         # bits below the round bit in digit ir
    keepmask = ~((UInt32(1) << bu) - UInt32(1))                      # bits at/above the lsb in digit iu
    v0   = Expr(:tuple, [i < iu ? :(Z32) : (i == iu ? :((w[$i] & $keepmask) + inc) : :(w[$i])) for i in 1:W]...)
    v2   = Expr(:tuple, [i == W ? :(v1[$i] | (co << $(DB - 1))) : :(v1[$i]) for i in 1:W]...)
    pack = Expr(:tuple, [:(v2[$(i + 1)]) for i in 1:K]...)
    quote
        Base.@_inline_meta
        rbit = (w[$ir] >> $br) & 0x01
        lsb  = (w[$iu] >> $bu) & 0x01
        st   = sticky | (((w[$ir] & $lowmask) | or_digits(w, Val(1), Val($(ir - 1)))) != Z32)
        up   = (rbit == 0x01) & (st | (lsb == 0x01))
        inc  = UInt32(up) << $bu
        v1, co = carry_norm($v0)                        # the increment may ripple to the top …
        t += Int(co)                                    # … and out: the value is then exactly 2^t
        v2 = $v2                                        #     (all digits zero → set the leading bit)
        if t > emax(T)
            return T(neg, emax(T), maxd(T), GE)
        elseif t < emin(T)
            return T(neg, emin(T), mind(T), LE)
        end
        return T(neg, t, $pack, 0x00)
    end
end

"""normalize + round-to-nearest-even at Pr bits (exact tail: digits + sticky) + saturate → T
(flag = the saturation flag only).  d is little-endian with LSB weight 2^e0, any length."""
@inline function round_pack(::Type{T}, neg::Bool, d::NTuple{N,UInt32}, e0::Int, sticky::Bool, pr::Val) where {T<:MFloat,N}
    L, top = topdigit(d)
    if L == 0
        sticky && error("round_pack: zero window with a non-zero sticky (window too narrow)")
        return zero(T)
    end
    w, t, dropped = normalize(d, e0, L, top, Val(ndig(T) + 1))
    return round_norm(T, neg, w, t, sticky | (dropped != Z32), pr)
end

# ---------------------------------------------------------------------------------------
# 4. conversions
# ---------------------------------------------------------------------------------------
function digits_of(n::Integer)
    n = big(n); n ≥ 0 || error("digits_of: negative")
    d = UInt32[]
    while n > 0
        push!(d, UInt32(n & MASK)); n >>= DB
    end
    return d
end

"""non-negative integer → L little-endian digits (test helper; errors if it does not fit)."""
function digits_tuple(n::Integer, ::Val{L}) where {L}
    d = digits_of(n)
    length(d) ≤ L || error("digits_tuple: does not fit")
    ntuple(i -> i ≤ length(d) ? d[i] : Z32, Val(L))
end

function (::Type{T})(x::Float64) where {T<:MFloat}
    isnan(x) && return T(false, 0, zerod(T), GE | LE | SUNK)
    isinf(x) && return T(x < 0, emax(T), maxd(T), GE)
    x == 0 && return zero(T)
    num, pow, den = Base.decompose(x)                  # num ≥ 0 (53 bits), den = ±1
    d = ((num & Int64(MASK)) % UInt32, (num >> DB) % UInt32)   # two digits from Int64, no BigInt
    round_pack(T, den < 0, d, pow, false, precval(T))
end
(::Type{T})(x::Float32) where {T<:MFloat} = T(Float64(x))
(::Type{T})(x::Float16) where {T<:MFloat} = T(Float64(x))
function (::Type{T})(x::BigFloat) where {T<:MFloat}
    isnan(x) && return T(false, 0, zerod(T), GE | LE | SUNK)
    isinf(x) && return T(x < 0, emax(T), maxd(T), GE)
    x == 0 && return zero(T)
    num, pow, den = Base.decompose(x)
    round_pack(T, den < 0, Tuple(digits_of(abs(num))), pow, false, precval(T))   # host only (BigInt)
end
function (::Type{T})(x::Integer) where {T<:MFloat}
    x == 0 && return zero(T)
    round_pack(T, x < 0, Tuple(digits_of(abs(big(x)))), 0, false, precval(T))
end
function (::Type{T})(x::Int64) where {T<:MFloat}                     # no BigInt (a BigInt branch made the
    x == 0 && return zero(T)                                         # return type Any: 110 B per call)
    u = x < 0 ? ~reinterpret(UInt64, x) + one(UInt64) : reinterpret(UInt64, x)      # |x|, typemin included
    d = ((u & MASK64) % UInt32, ((u >> DB) & MASK64) % UInt32, (u >> (2 * DB)) % UInt32)    # 3 digits = 87 bits
    round_pack(T, x < 0, d, 0, false, precval(T))
end
(::Type{T})(x::Rational) where {T<:MFloat} = T(numerator(x)) / T(denominator(x))
(::Type{T})(x::T) where {T<:MFloat} = x
(::Type{T})(x::Real) where {T<:MFloat} = T(BigFloat(x))

function Base.BigFloat(x::T) where {T<:MFloat}
    K = ndig(T)
    setprecision(BigFloat, DB * K + 16) do
        acc = BigFloat(0)
        for i in K:-1:1
            acc = ldexp(acc, DB) + x.d[i]
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
"""ℂ の席: 値の欄は空(0 は placeholder)・境界なし・符号不明・ℂ.  ℂ は粘る — どの演算に入っても席のまま出る
(ScalarTot の ZCPLX と同じ; z^0 = 1 だけが `^` の側で先に抜ける)."""
zcplx(::Type{T}) where {T<:MFloat} = T(false, 0, zerod(T), NOB | SUNK | CPLX)
@inline cplx(a::MFloat) = (a.flag & CPLX) != 0
@inline cplx(a::MFloat, b::MFloat) = ((a.flag | b.flag) & CPLX) != 0
Base.abs(a::T) where {T<:MFloat} = cplx(a) ? zcplx(T) : T(false, a.ex, a.d, a.flag)
Base.:-(a::T) where {T<:MFloat} = T(!a.neg && !iszero(a), a.ex, a.d, a.flag)
Base.sign(a::T) where {T<:MFloat} = cplx(a) ? zcplx(T) : iszero(a) ? zero(T) : T(a.neg, 0, mind(T), a.flag & SUNK)
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
"""a + b (flipb: a − b), exact window + sticky, rounded at Pr bits.  The two sign cases differ only
in the digit sum / difference; the normalization and rounding after them is one shared call, so a
warp whose lanes hold both cases runs the expensive part once."""
@noinline function add_core(::Type{T}, a::T, b::T, flipb::Bool, pr::Val) where {P,K,E,T<:MFloat{P,K,E}}
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
    wA = tprepend2(A.d)                                      # K+2 digits, A at the top
    wB, stB = shr_total(tprepend2(B.d), Δ)                   # B aligned: right by Δ bits, the rest → sticky
    if negA == negB
        s, co = carry_norm(tmap(+, wA, wB))
        dd = tappend(s, co)
    else
        s1, bo = sub_digits(wA, wB, stB ? UInt32(1) : Z32)  # the sticky is a borrow of one window-ulp …
        s2, _  = sub_digits(wB, wA, Z32)                     # … and stays sticky (remainder ∈ (0, ulp_w));
        neg = bo == UInt32(1)                                # a borrow out (only when Δ = 0, so no sticky)
        s = tmap((x, y) -> ifelse(neg, y, x), s1, s2)        # means |B| > |A|: take B − A, B's sign
        negA = neg ? negB : negA
        dd = tappend(s, Z32)                                 # (A = B gives all zeros → round_pack → +0)
    end
    return round_pack(T, negA, dd, e0w, stB, pr)
end

"""a · b rounded at Pr bits.  The product of two canonical significands has its leading bit in
digit 2K (bit 27 or 28), so the normalization is a static slice and a 0/1-bit shift."""
@noinline function mul_core(::Type{T}, a::T, b::T, pr::Val) where {P,K,E,T<:MFloat{P,K,E}}
    (iszero(a) || iszero(b)) && return zero(T)
    p = mul_digits(a.d, b.d)                                 # 2K digits, exact
    e0 = (Int(a.ex) - DB * K + 1) + (Int(b.ex) - DB * K + 1)
    w, t, dropped = normalize_at(p, e0, Val(2 * K), Val(K + 1))
    round_norm(T, a.neg ⊻ b.neg, w, t, dropped != Z32, pr)
end

half(x::T) where {T<:MFloat} = iszero(x) ? x : T(x.neg, Int(x.ex) - 1, x.d, x.flag)

"""Float64 of the top three digits (87 bits, rounded to 53), exponent applied."""
function top_float64(m::T) where {T<:MFloat}
    K = ndig(T); d = m.d
    v = Float64(d[K]) * 2.0^-(DB - 1) + Float64(d[K - 1]) * 2.0^-(2 * DB - 1) + Float64(d[K - 2]) * 2.0^-(3 * DB - 1)
    ldexp(v, Int(m.ex))
end
# --- the Newton ladder --------------------------------------------------------------------
# Newton squares the error — the seed's 50 bits become 100, 200, 400 … — so a step only has to be
# as *wide* as the accuracy it produces: the first steps of a 522-bit reciprocal are 4-digit
# multiplies, not 18-digit ones.  Running every step at the full width (the plain loop this
# replaced, kept in MultiU32Ref.jl) pays for digits that are known to be wrong.  Two widths per
# step, both fixed by the type alone:
#   • the step itself (m·y, and y² for √) at `w` digits — this is what sets the new accuracy;
#   • the correction y·e at `w2` digits — e = 1 − m·y is already below 2^-acc, so the product only
#     has to supply the 29w − acc bits that are still missing (≈ half the step, a quarter the work).
# Operands are truncated into each stage (`retype`): m to w digits is an error of 2^-29w, which is
# the stage's own accuracy — the ladder converges to 1/m, not to 1/truncated-m, because the last
# step sees the full m.  Cost in full-width products, seed → 29K bits:
#     F128  6 → 2.2      F256  8 → 2.5      F512  10 → 2.2
# The residual + fix-up path of div_core / sqrt_core is untouched: the ladder only proposes the
# candidate, and the exact residual still decides the last bit, so a candidate that is a few ulps
# worse costs a fix-up, never a wrong answer (and a candidate far out is an error, not a lie).
const SEED = 50                                     # bits `top_float64`'s Float64 seed is trusted to

"""(step width, accuracy going in) in digits/bits for each Newton step, ending at the full K
digits.  Each step sizes its own residual slice from the accuracy it is handed."""
function ladder(K::Int, guard::Int = 4)
    acc = SEED; st = Tuple{Int,Int}[]
    while true
        w = min(K, max(2, cld(2 * acc + guard, DB)))               # the step produces min(2·acc, 29w) bits
        push!(st, (w, acc))
        acc = min(2 * acc, DB * w - guard)
        (w == K && acc ≥ DB * K - 2 * guard) && return Tuple(st)
    end
end

"""the type of a ladder stage: w digits, rounded at the full 29w bits (the stage's P only has to
satisfy the constructor, 29w ≥ P+12); the last stage is T itself, so div_core / sqrt_core are
unchanged."""
stage(::Type{MFloat{P,K,E}}, w::Int) where {P,K,E} = w == K ? MFloat{P,K,E} : MFloat{DB * w - 12, w, E}

# Inside the ladder nothing is normalized and nothing is rounded: every value has a scale the type
# fixes, so "where the 1 sits in the product" and "how far the correction is shifted in" are digit
# positions known at compile time.  Measured on the same ladder written with MFloat values, that is
# where the time was — a 4-digit multiply is ~10 ns of digit products behind ~30 ns of normalize +
# round_pack, and the ladder does 16 operations.  Fixed point:
#     y = Y·2^(1−29w)   with y ∈ [1/2, 1]  ⟹  Y ∈ [2^(29w−2), 2^(29w−1)] fits in w digits
#     m = Mw·2^(1−29w)  (Mw = the top w digits of the significand; the tail it drops is below the
#                        stage's own ulp, so the ladder still converges to 1/m — the last stage
#                        sees all of m)
#     m·y = P·2^(2−58w) with P = Mw·Y, so "1" is the constant bit 58w−2 = digit 2w, bit 27.
# The residual e = 1 − m·y is used through a *static* slice: its magnitude is bounded by the
# accuracy going into the step, so the digits above that bound are known to be zero and the w2
# digits below it carry everything that is still missing.  `ladder` sizes the slice.

"""digits hi−L+1 … hi of d (static, zero below index 1)."""
@generated function slice_at(d::NTuple{N,UInt32}, ::Val{hi}, ::Val{L}) where {N,hi,L}
    :(Base.@_inline_meta; $(Expr(:tuple, [(j = hi - L + i; 1 ≤ j ≤ N ? :(d[$j]) : :(Z32)) for i in 1:L]...)))
end

"""Y widened from wp to W digits (y unchanged: the scale moves with the width)."""
@generated function widen_fix(Y::NTuple{WP,UInt32}, ::Val{W}) where {WP,W}
    :(Base.@_inline_meta; $(Expr(:tuple, [i ≤ W - WP ? :(Z32) : :(Y[$(i - (W - WP))]) for i in 1:W]...)))
end

"""one reciprocal step at width W: y ← y + y·(1 − m·y).  m·y = P·2^(2−58W), so the "1" it is
compared with is the single bit 58W−2, which contributes nothing to the digits below it: the low L
digits of the product **are** (m·y − 1) in two's complement, sign in the top bit.  L follows from the
accuracy ACC the step is handed (|m·y − 1| ≤ 2^-ACC), and the correction reads the top W2 digits."""
@generated function recip_step(Mw::NTuple{W,UInt32}, Y::NTuple{W,UInt32}, ::Val{ACC}) where {W,ACC}
    L = fld(DB * 2W - 1 - ACC, DB) + 1
    L < 2W || error("recip_step: the residual does not fit below the leading 1")
    W2 = clamp(max(L + 1 - W, W - fld(ACC - 4, DB)), 2, W)
    zs = Expr(:tuple, [:(Z32) for _ in 1:L]...)
    quote
        Base.@_inline_meta
        T = mul_digits_low(Mw, Y, Val($L))                       # (m·y − 1)·2^(58W−2), two's complement
        ge = (T[$L] >> $(DB - 1)) == Z32                         # top bit clear ⟹ m·y ≥ 1 ⟹ y ← y − |y·e|
        n, _ = sub_digits($zs, T, Z32)                           # |m·y − 1| in the other case
        G = mul_digits(slice_at(Y, Val(W), Val($W2)), slice_at(tsel(ge, T, n), Val($L), Val($W2)))
        D, _ = place(G, $(DB * (L - 2 * W2) - DB * W + 2), Val(W))
        s, _ = carry_norm(tmap(+, Y, D))                         # y' ≤ 1/m ≤ 1: no carry out
        t, _ = sub_digits(Y, D, Z32)
        return tsel(ge, t, s)
    end
end

"""one inverse-square-root step at width W: y ← y + (y/2)(1 − m·y²), m = Mw·2^(2−29W) ∈ [1,4);
m·y² = P·2^(4−87W), the same two's-complement reading with the 1 at bit 87W−4."""
@generated function rsqrt_step(Mw::NTuple{W,UInt32}, Y::NTuple{W,UInt32}, ::Val{ACC}) where {W,ACC}
    L = fld(DB * 3W - 2 - ACC, DB) + 1
    L < 3W || error("rsqrt_step: the residual does not fit below the leading 1")
    W2 = clamp(max(L - 2W + 1, W - fld(ACC - 4, DB)), 2, W)
    zs = Expr(:tuple, [:(Z32) for _ in 1:L]...)
    quote
        Base.@_inline_meta
        T = mul_digits_low(mul_digits_low(Mw, Y, Val($(min(2W, L)))), Y, Val($L))
        ge = (T[$L] >> $(DB - 1)) == Z32
        n, _ = sub_digits($zs, T, Z32)
        G = mul_digits(slice_at(Y, Val(W), Val($W2)), slice_at(tsel(ge, T, n), Val($L), Val($W2)))
        D, _ = place(G, $(DB * (L - 2 * W2) - DB * 2W + 3), Val(W))
        s, _ = carry_norm(tmap(+, Y, D))
        t, _ = sub_digits(Y, D, Z32)
        return tsel(ge, t, s)
    end
end

"""the Float64 seed as the ladder's fixed point: y = D·2^(ex−29w+1) = Y·2^(1−29w) ⟹ Y = D·2^ex,
ex ∈ {−1, 0} (ex = 0 only for y = 1 exactly)."""
@inline function seed_fix(::Type{S}, y0::Float64, ::Val{W}) where {S<:MFloat,W}
    s = S(y0)
    Y, _ = shr_bits(s.d, -Int(s.ex))
    widen_fix(Y, Val(W))
end

"""≈ 1/m at 29K bits (m: significand with ex = 0, m ∈ [1,2)) — the ladder, unrolled by the type."""
@generated function recip_sig(::Type{T}, m::T) where {T<:MFloat}
    K = ndig(T); st = ladder(K)
    body = Expr[:(Y_0 = seed_fix($(stage(T, st[1][1])), 1.0 / top_float64(m), Val($(st[1][1]))))]
    for (i, (w, acc)) in enumerate(st)
        Yi, Yp, wp = Symbol(:Y_, i), Symbol(:Y_, i - 1), i == 1 ? w : st[i - 1][1]
        push!(body, :($Yi = recip_step(slice_at(m.d, Val($K), Val($w)),
                                       $(wp == w ? Yp : :(widen_fix($Yp, Val($w)))), Val($acc))))
    end
    quote
        $(body...)
        return round_pack(T, false, $(Symbol(:Y_, length(st))), $(1 - DB * K), false, fullval(T))
    end
end

"""≈ √m at 29K bits in the ladder's fixed point (s = S·2^(1−29K) ∈ [1,2)), m ∈ [1,4), ex ∈ {0,1}.
The ladder runs on 1/√m to *half* the width and one Karp–Markstein step turns it into √m itself:
s = m·y, then s ← s + (y/2)(m − s²).  Half the width costs a quarter of the products, and the step
that buys the other half is one squaring — where a further rsqrt step would cost m·y² (two
full-width products) *and* the m·r that sqrt_core no longer has to do.  y stays half-width: it is
only the multiplier of a second-order correction."""
@generated function sqrt_sig(::Type{T}, m::T) where {T<:MFloat}
    K = ndig(T); wh = min(K, cld(K, 2) + 1); st = ladder(wh)
    body = Expr[:(Y_0 = seed_fix($(stage(T, st[1][1])), 1.0 / sqrt(top_float64(m)), Val($(st[1][1])))),
                :(Md = shr_bits(m.d, 1 - Int(m.ex))[1])]                  # m = Md·2^(2−29K) ∈ [1,4)
    for (i, (w, acc)) in enumerate(st)
        Yi, Yp, wp = Symbol(:Y_, i), Symbol(:Y_, i - 1), i == 1 ? w : st[i - 1][1]
        push!(body, :($Yi = rsqrt_step(slice_at(Md, Val($K), Val($w)),
                                       $(wp == w ? Yp : :(widen_fix($Yp, Val($w)))), Val($acc))))
    end
    Y = Symbol(:Y_, length(st))
    acc = min(2 * st[end][2], DB * wh - 4)                                # accuracy of y going in
    L = fld(DB * 2K + 2 - acc, DB) + 1                                    # |s² − m| ≤ 2^(3−acc)
    L < 2K || error("sqrt_sig: the Karp–Markstein residual does not fit")
    w2 = clamp(max(L - K + 1, K - fld(acc - 6, DB)), 2, wh)
    Wm = Expr(:tuple, [i ≤ K ? :(Z32) : :(Md[$(i - K)]) for i in 1:L]...)  # m at the scale of s²
    zs = Expr(:tuple, [:(Z32) for _ in 1:L]...)
    quote
        $(body...)
        S0, _ = place(mul_digits(Md, $Y), $(2 - DB * wh), Val($K))        # s = m·y
        R, _ = sub_digits(mul_digits_low(S0, S0, Val($L)), $Wm, Z32)      # (s² − m)·2^(58K−2)
        ge = (R[$L] >> $(DB - 1)) == Z32                                  # s² ≥ m ⟹ s ← s − |·|
        n, _ = sub_digits($zs, R, Z32)
        G = mul_digits(slice_at($Y, Val($wh), Val($w2)), slice_at(tsel(ge, R, n), Val($L), Val($w2)))
        D, _ = place(G, $(DB * (L - 2 * w2) - DB * K + 1), Val($K))       # (y/2)·(m − s²)
        s, _ = carry_norm(tmap(+, S0, D))
        t, _ = sub_digits(S0, D, Z32)
        return tsel(ge, t, s)
    end
end

"""≈ 1/√m at 29K bits (m ∈ [1,4), ex ∈ {0,1}) — the ladder on its own (sqrt_core takes sqrt_sig)."""
@generated function rsqrt_sig(::Type{T}, m::T) where {T<:MFloat}
    K = ndig(T); st = ladder(K)
    body = Expr[:(Y_0 = seed_fix($(stage(T, st[1][1])), 1.0 / sqrt(top_float64(m)), Val($(st[1][1])))),
                :(Md = shr_bits(m.d, 1 - Int(m.ex))[1])]
    for (i, (w, acc)) in enumerate(st)
        Yi, Yp, wp = Symbol(:Y_, i), Symbol(:Y_, i - 1), i == 1 ? w : st[i - 1][1]
        push!(body, :($Yi = rsqrt_step(slice_at(Md, Val($K), Val($w)),
                                       $(wp == w ? Yp : :(widen_fix($Yp, Val($w)))), Val($acc))))
    end
    quote
        $(body...)
        return round_pack(T, false, $(Symbol(:Y_, length(st))), $(1 - DB * K), false, fullval(T))
    end
end

# The residual windows: every shift amount below is a function of P and K alone, so after inlining
# the barrel shifters fold to static digit moves (a runtime shift would run the full barrel).

"""R = asig − Q·m·2^-P as (neg, digits) in a 2K+2-digit window with LSB weight 2^(1−29K−P)."""
@noinline function div_resid(da::NTuple{K,UInt32}, Q::NTuple{K1,UInt32}, db::NTuple{K,UInt32}, ::Val{P}) where {K,K1,P}
    pm = mul_digits(Q, db)                              # 2K+1 digits
    e0s = 1 - DB * K
    negR, dR, _ = wide_sub(da, e0s, pm, e0s - P, Val(2 * K + 2))
    return negR, dR
end

"""|a| / |b| correctly rounded at P (b ≠ 0), with the sign; flag = saturation only.
Returns (value, number of residual fix-ups of the Newton candidate) — the count is diagnostic."""
@noinline function div_core(::Type{T}, a::T, b::T) where {P,K,E,T<:MFloat{P,K,E}}
    e0s = 1 - DB * K                                    # LSB weight of a significand with ex = 0 …
    eR  = e0s - P                                       # … and of the residual window
    m    = T(false, 0, b.d, 0x00)
    asig = T(false, 0, a.d, 0x00)
    y  = recip_sig(T, m)
    qh = mul_core(T, asig, y, fullval(T))               # q̂ ∈ (1/2, 2): ex ∈ {−1, 0}
    sq = Int(qh.ex) + 1
    0 ≤ sq ≤ 1 || error("div_core: Newton candidate outside (1/2, 2)")
    Q, _ = place(shl_bits(qh.d, sq), P - DB * K, Val(K + 1))   # Q = ⌊q̂·2^P⌋ (weight 2^-P): static shift
    da = a.d; db = b.d
    negR, dR = div_resid(da, Q, db, Val(P))             # R = asig − Q·m·2^-P
    fix = 0
    while negR                                          # Q too large
        Q = add_int(Q, -1); negR, dR = div_resid(da, Q, db, Val(P)); fix += 1
        fix > 8 && error("div_core: candidate too far (−)")
    end
    while wide_cmp(dR, eR, db, e0s - P, Val(2 * K + 2)) ≥ 0    # R ≥ m·2^-P → Q too small
        Q = add_int(Q, 1); negR, dR = div_resid(da, Q, db, Val(P)); fix += 1
        fix > 8 && error("div_core: candidate too far (+)")
    end
    # now Q·2^-P ≤ q_exact < (Q+1)·2^-P with remainder R ∈ [0, m·2^-P)
    if (Q[P ÷ DB + 1] >> (P % DB)) & 0x01 == 0x00      # Q < 2^P (Q < 2^(P+1) always: bit P decides)
        # q < 1 : ulp = 2^-P, round bit = 2R vs m·2^-P
        c = wide_cmp(dR, eR + 1, db, e0s - P, Val(2 * K + 3))
        (c > 0 || (c == 0 && isodd_digits(Q))) && (Q = add_int(Q, 1))
    else                                                # q ≥ 1 : ulp = 2^(1-P) = two grid steps
        if isodd_digits(Q)
            if iszero_digits(dR)                         # exact midpoint between Q−1 and Q+1 → the one ≡ 0 (mod 4)
                Q = add_int(Q, lowbits(Q, 2) == 0x01 ? -1 : 1)
            else
                Q = add_int(Q, 1)
            end
        end
    end
    round_pack(T, a.neg ⊻ b.neg, Q, -P + Int(a.ex) - Int(b.ex), false, precval(T)), fix
end

"""R = m′ − S²·2^{2g} (m′ = dm·2^(1−29K), g = 1−P) as (neg, digits) in a 2K+2-digit window with
LSB weight 2^min(1−29K, 2g)."""
@noinline function sqrt_resid(dm::NTuple{K1,UInt32}, S::NTuple{K1,UInt32}, ::Val{P}, ::Val{K}) where {K1,P,K}
    ss = mul_digits(S, S)                               # 2K+2 digits
    negR, dR, _ = wide_sub(dm, 1 - DB * K, ss, 2 * (1 - P), Val(2 * K + 2))
    return negR, dR
end

"""√|a| correctly rounded at P (a ≠ 0); flag = saturation only.  Returns (value, fix-up count)."""
@noinline function sqrt_core(::Type{T}, a::T) where {P,K,E,T<:MFloat{P,K,E}}
    odd = isodd(Int(a.ex))
    mp  = T(false, odd ? 1 : 0, a.d, 0x00)             # m' ∈ [1,4), √m' ∈ [1,2)
    exr = fld(Int(a.ex), 2)
    g = 1 - P                                           # grid = ulp of [1,2) at P bits
    Sf = sqrt_sig(T, mp)                                # ŝ = Sf·2^(1−29K) ∈ [1,2), the ladder's fixed point
    S, _ = place(Sf, P - DB * K, Val(K + 1))            # S = ⌊ŝ·2^(P−1)⌋: static shift
    dm = shl_bits(a.d, odd ? 1 : 0)                     # m' as K+1 digits at the static weight 2^(1−29K)
    eR = min(1 - DB * K, 2 * g)                         # LSB weight of the residual window
    negR, dR = sqrt_resid(dm, S, Val(P), Val(K))        # R = m' − S²·2^{2g}
    fix = 0
    while negR
        S = add_int(S, -1); negR, dR = sqrt_resid(dm, S, Val(P), Val(K)); fix += 1
        fix > 8 && error("sqrt_core: candidate too far (−)")
    end
    while true                                          # (S+1)² ≤ m'/2^{2g}  ⟺  R ≥ 2S+1
        t = add_int(shl_bits(S, 1), 1)
        wide_cmp(dR, eR, t, 2 * g, Val(2 * K + 2)) ≥ 0 || break
        S = add_int(S, 1); negR, dR = sqrt_resid(dm, S, Val(P), Val(K)); fix += 1
        fix > 8 && error("sqrt_core: candidate too far (+)")
    end
    f = add_int(shl_bits(S, 2), 1)                      # round up ⟺ 4R > 4S+1  (never equal)
    wide_cmp(dR, eR + 2, f, 2 * g, Val(2 * K + 3)) > 0 && (S = add_int(S, 1))
    round_pack(T, false, S, g + exr, false, precval(T)), fix
end

# --- user-facing operators: value core + ScalarTot's flag rules --------------------------
function Base.:+(a::T, b::T) where {T<:MFloat}
    cplx(a, b) && return zcplx(T)
    r = add_core(T, a, b, false, precval(T))
    T(r.neg, r.ex, r.d, r.flag | addflag(a, b, false))
end
function Base.:-(a::T, b::T) where {T<:MFloat}
    cplx(a, b) && return zcplx(T)
    r = add_core(T, a, b, true, precval(T))
    T(r.neg, r.ex, r.d, r.flag | addflag(a, b, true))
end
function Base.:*(a::T, b::T) where {T<:MFloat}
    cplx(a, b) && return zcplx(T)
    tz = (iszero(a) && (a.flag & GE) == 0) || (iszero(b) && (b.flag & GE) == 0)
    tz && return zero(T)
    r = mul_core(T, a, b, precval(T))
    T(r.neg, r.ex, r.d, r.flag | mulflag(a.flag, b.flag))
end
function Base.:/(a::T, b::T) where {T<:MFloat}
    cplx(a, b) && return zcplx(T)
    bz = iszero(b)
    r = (bz || iszero(a)) ? zero(T) : div_core(T, a, b)[1]       # a/0 = 0
    fin = a.flag | b.flag
    nb = (fin & (GE | LE)) > 0
    dz = (iszero(a) && (a.flag & GE) > 0) || (bz && (b.flag & GE) > 0)
    f = r.flag | (nb ? (GE | LE) : 0x00) | (fin & SUNK) | (dz ? SUNK : 0x00)
    T(r.neg, r.ex, r.d, f)
end
function Base.sqrt(a::T) where {T<:MFloat}
    cplx(a) && return zcplx(T)
    if a.neg && !iszero(a)
        return (a.flag & SUNK) != 0 ? T(false, 0, zerod(T), NOB | SUNK | CPLX) : T(false, 0, zerod(T), a.flag | CPLX)
    end
    iszero(a) && return sign_untrusted(a) ? T(false, 0, zerod(T), NOB | SUNK | CPLX) : zero(T)
    r = sqrt_core(T, a)[1]
    sign_untrusted(a) && return T(false, r.ex, r.d, r.flag | NOB | SUNK | CPLX)
    T(false, r.ex, r.d, r.flag | (a.flag & (GE | LE)))
end
Base.inv(a::T) where {T<:MFloat} = one(T) / a
Base.:^(a::T, n::Integer) where {T<:MFloat} = n == 0 ? one(T) : Base.power_by_squaring(a, n)   # z^0 = 1 even for ℂ
Base.literal_pow(::typeof(^), a::T, ::Val{N}) where {T<:MFloat,N} = a^N

# ---------------------------------------------------------------------------------------
# 7b. exp and log — correctly rounded (Ziv), total
# ---------------------------------------------------------------------------------------
# Both are computed in a *working type* S = MFloat{29w−12, w, E} with w = K+2 digits (58 bits
# beyond the 29K of T, ≥ 70 beyond P), then rounded to P bits only when the rounding is decided:
# round-to-nearest can change only at a midpoint of the P-bit grid, so if the working value is
# further than its error bound B (ulps of S) from every midpoint, rounding it *is* rounding the
# true value (Ziv's test).  Otherwise the same computation is repeated at K+6 and then 2K+6 digits
# — 2P+ bits, which is where the known hard cases live (exp(2^−P) = 1 + 2^−P + 2^−(2P+1): a
# midpoint plus a 2P-bit tail).  If even that is undecided the result is the truncated value with
# ⟦≥⟧: true ∈ (v, v + ulp) is then a *proven* statement, so the library stays total and honest
# without an exception (the cores return (value, stage) with stage 4 = undecided, as div_core
# returns its fix-up count — a host counter would not compile into a GPU kernel; never reached).
#
#   exp x = 2^k · (e^{r/2^s})^{2^s},  k = round(x / ln 2),  r = x − k·ln 2  (ln 2 at w+2 digits:
#           the cancellation of ≤ 25 bits is paid there), Taylor of N terms with 1/n! as
#           compile-time constants (Horner), then s squarings.  Error ≤ 2^s·(2N+2) ulps.
#   log x = e·ln 2 + 2^{s+1}·atanh z,  x = m·2^e with m ∈ [1/√2, √2),  m → m^{1/2^s} by s square
#           roots, z = (m′−1)/(m′+1) (|z| < 2^−(s+2.5)), odd series of N terms with 1/(2n+1) as
#           constants.  The s roots halve the relative error each; (m′−1) then loses s+2.5 bits to
#           cancellation, which is the 2^{s+5} in the bound B = 4·(2^{s+5} + 4N + 16) — the measured
#           worst error is 225 / 347 / 689 ulps (F128 / F256 / F512), the ×4 is the margin.
# The exact special cases never enter a series: exp 0 = 1, log 1 = 0, and log 0 = 0 by definition
# (the reserved word: Σ (1/n)((x−1)/x)ⁿ is 0 term by term under a/0 = 0 — a limit belongs to ε).

"""the working type of w digits (rounded at 29w−12 bits, T's exponent range)."""
wide(::Type{MFloat{P,K,E}}, w::Int) where {P,K,E} = MFloat{DB * w - 12, w, E}
"""the three Ziv stages of T, as literal types."""
@generated stages(::Type{T}) where {T<:MFloat} = (K = ndig(T); :(($(wide(T, K + 2)), $(wide(T, K + 6)), $(wide(T, 2K + 6)))))

"""x rounded into S at S's working precision (flags dropped: working values only)."""
@inline retype(::Type{S}, x::T) where {S<:MFloat,T<:MFloat} =
    iszero(x) ? zero(S) : round_pack(S, x.neg, x.d, e0_of(x), false, precval(S))
"""x·2^n (exact: an exponent shift)."""
@inline scale2(x::T, n::Int) where {T<:MFloat} = iszero(x) ? x : T(x.neg, Int(x.ex) + n, x.d, x.flag)

"""ln 2 rounded to S's working precision — a compile-time constant."""
@generated function ln2c(::Type{S}) where {S<:MFloat}
    c = setprecision(BigFloat, DB * ndig(S) + 64) do; S(log(BigFloat(2))); end
    :($c)
end

"""is the P-bit (T) rounding of v (canonical in S) decided, given |true − v| ≤ B ulps of S?
Round-to-nearest changes only at a midpoint 2^q of the tail (q = the round bit), so the test is
|tail − 2^q| > B — static digit positions, two subtractions and a select."""
@generated function ziv_ok(::Type{T}, v::S, ::Val{B}) where {T<:MFloat,S<:MFloat,B}
    P = prec(T); Ks = ndig(S)
    q = DB * Ks - P - 1                                     # round bit (LSB of digit 1 = 0)
    Ld = q ÷ DB + 1; bq = q % DB
    topmask = (UInt32(1) << (bq + 1)) - UInt32(1)           # bits 0 … bq of digit Ld
    tail = Expr(:tuple, [i < Ld ? :(v.d[$i]) : :(v.d[$i] & $topmask) for i in 1:Ld]...)
    mid  = Expr(:tuple, [i < Ld ? :(Z32) : :(UInt32(1) << $bq) for i in 1:Ld]...)
    (Ld ≥ 2 && B < 2^56) || error("ziv_ok: the bound must fit two digits")
    quote
        Base.@_inline_meta
        t = $tail; m = $mid
        r1, bo = sub_digits(t, m, Z32)
        r2, _  = sub_digits(m, t, Z32)
        r = tsel(bo == Z32, r1, r2)                          # |tail − mid|
        return (or_digits(r, Val(3), Val($Ld)) != Z32) | (((UInt64(r[2]) << DB) | UInt64(r[1])) > UInt64($B))
    end
end

"""v with the bits below T's P cleared (truncation toward zero onto the P-bit grid)."""
@generated function zero_tail(v::S, ::Type{T}) where {S<:MFloat,T<:MFloat}
    P = prec(T); Ks = ndig(S)
    q = DB * Ks - P                                         # first kept bit
    Ld = q ÷ DB + 1; bq = q % DB
    keep = ~((UInt32(1) << bq) - UInt32(1))
    Expr(:tuple, [i < Ld ? :(Z32) : i == Ld ? :(v.d[$i] & $keep) : :(v.d[$i]) for i in 1:Ks]...)
end

# --- exp -----------------------------------------------------------------------------------
"""(halvings s, Taylor terms N, error bound B in ulps) for a working precision of Wb bits."""
function exp_params(Wb::Int)
    s = round(Int, 0.7 * sqrt(Wb))                          # balances N ≈ Wb/(s+1.5) against s squarings
    N = 1
    while (s + 1.5) * N + log2(factorial(big(N))) < Wb + 8; N += 1; end
    B = 2^(s + ceil(Int, log2(2N + 2)) + 1)
    return s, N, B
end

"""e^x as (h ≈ e^{x − k ln2} ∈ (0.7, 1.42) in S, k, rounding decided?)."""
@generated function exp_stage(::Type{T}, x::T, ::Type{S}) where {T<:MFloat,S<:MFloat}
    Ks = ndig(S); s, N, B = exp_params(prec(S))
    S2 = wide(T, Ks + 2)                                    # the reduction: ln2 at +58 bits
    ln2w = setprecision(BigFloat, DB * (Ks + 2) + 64) do; S2(log(BigFloat(2))); end
    q = setprecision(BigFloat, DB * Ks + 64) do; [S(1 / BigFloat(factorial(big(n)))) for n in 0:N]; end
    body = Expr[:(h = $(q[N + 1]))]
    for n in N - 1:-1:0
        push!(body, :(h = add_core(S, mul_core(S, h, r, pv), $(q[n + 1]), false, pv)))
    end
    for _ in 1:s
        push!(body, :(h = mul_core(S, h, h, pv)))
    end
    quote
        pv = precval(S); pv2 = precval($S2)
        k = round(Int, top_float64(x) / 0.6931471805599453); x.neg && (k = -k)   # top_float64 is |x|
        kl = mul_core($S2, $S2(k), $ln2w, pv2)
        r = scale2(retype(S, add_core($S2, retype($S2, x), kl, true, pv2)), -$s)
        $(body...)
        return h, k, ziv_ok(T, h, Val($B))
    end
end

"""e^x correctly rounded at P; flag = saturation only.  exp 0 = 1 exactly.  Returns (value, stage):
stage 0 = no series (exact or saturated), 1–3 = the Ziv stage that decided, 4 = undecided (⟦≥⟧)."""
function exp_core(::Type{T}, x::T) where {P,K,E,T<:MFloat{P,K,E}}
    iszero(x) && return one(T), 0
    ex = Int(x.ex)
    ex ≥ 24 && return (x.neg ? T(false, emin(T), mind(T), LE) : T(false, emax(T), maxd(T), GE)), 0   # |x| ≥ 2^24 ≫ ln MAX
    ex < -(P + 3) && return one(T), 0                       # |x| < 2^−(P+3): e^x rounds to 1 (midpoints at 1 ± 2^−(P+1), 1 + 2^−P)
    S1, S2, S3 = stages(T)                                  # one name per stage: a Union-typed local would box
    h1, k, ok = exp_stage(T, x, S1)
    ok && return round_pack(T, false, h1.d, e0_of(h1) + k, false, precval(T)), 1
    h2, k, ok = exp_stage(T, x, S2)
    ok && return round_pack(T, false, h2.d, e0_of(h2) + k, false, precval(T)), 2
    h3, k, ok = exp_stage(T, x, S3)
    ok && return round_pack(T, false, h3.d, e0_of(h3) + k, false, precval(T)), 3
    r = round_pack(T, false, zero_tail(h3, T), e0_of(h3) + k, false, precval(T))
    return T(false, r.ex, r.d, r.flag | GE), 4
end

# --- log -----------------------------------------------------------------------------------
"""(square roots s, series terms N, error bound B) for a working precision of Wb bits."""
function log_params(Wb::Int)
    s = Wb < 250 ? 4 : (Wb < 400 ? 5 : 6)
    N = cld(Wb + 8, 2 * (s + 2.5)) |> Int                   # z^{2N+2} < 2^−(Wb+8) with |z| < 2^−(s+2.5)
    B = 4 * (2^(s + 5) + 4N + 16)                           # measured worst 225 / 347 / 689 ulps (F128/256/512) vs
    return s, N, B                                          # 2^{s+5}+4N+16 = 592 / 1124 / 2200: ×4 for a real margin
end                                                         # (bench_explog_libs.jl reports it)

"""log |x| as (value in S, rounding decided?); x ≠ 0, x ≠ 1."""
@generated function log_stage(::Type{T}, x::T, ::Type{S}) where {T<:MFloat,S<:MFloat}
    Ks = ndig(S); s, N, B = log_params(prec(S))
    c = setprecision(BigFloat, DB * Ks + 64) do; [S(1 / BigFloat(2n + 1)) for n in 0:N]; end
    body = Expr[:(h = $(c[N + 1]))]
    for n in N - 1:-1:0
        push!(body, :(h = add_core(S, mul_core(S, h, p, pv), $(c[n + 1]), false, pv)))
    end
    quote
        pv = precval(S)
        m = retype(S, T(false, 0, x.d, 0x00)); e = Int(x.ex)          # |x| = m·2^e, m ∈ [1,2)
        if top_float64(m) > 1.4142135623730951                          # → m ∈ [1/√2, √2): no cancellation in e·ln2 + log m
            m = half(m); e += 1
        end
        num = add_core(S, m, one(S), true, pv)                          # m − 1 (exact: Sterbenz)
        if iszero(num)
            L = zero(S)
        else
            # roots only until |m′ − 1| ≈ 2^−(s+2.5): each halves |m − 1| and adds an *absolute* rounding
            # error that the cancellation in m′ − 1 turns into a relative one — on an m already within
            # 2^−(s+2.5) of 1 they would only lose bits (m − 1 itself is exact)
            sp = clamp(Int(num.ex) + $s + 4, 0, $s)
            for _ in 1:sp
                m = sqrt_core(S, m)[1]
            end
            num = add_core(S, m, one(S), true, pv)
            z = div_core(S, num, add_core(S, m, one(S), false, pv))[1]
            p = mul_core(S, z, z, pv)
            $(body...)
            L = scale2(mul_core(S, z, h, pv), sp + 1)                   # 2^{s′+1}·z·h
        end
        res = e == 0 ? L : add_core(S, mul_core(S, S(e), ln2c(S), pv), L, false, pv)
        return res, ziv_ok(T, res, Val($B))
    end
end

"""log |x| correctly rounded at P (x ≠ 0); flag = saturation only.  log 1 = 0 exactly.  Returns
(value, stage) as exp_core."""
function log_core(::Type{T}, x::T) where {P,K,E,T<:MFloat{P,K,E}}
    (x.ex == 0 && x.d == mind(T)) && return zero(T), 0      # |x| = 1
    S1, S2, S3 = stages(T)
    v1, ok = log_stage(T, x, S1)
    ok && return round_pack(T, v1.neg, v1.d, e0_of(v1), false, precval(T)), 1
    v2, ok = log_stage(T, x, S2)
    ok && return round_pack(T, v2.neg, v2.d, e0_of(v2), false, precval(T)), 2
    v3, ok = log_stage(T, x, S3)
    ok && return round_pack(T, v3.neg, v3.d, e0_of(v3), false, precval(T)), 3
    r = round_pack(T, v3.neg, zero_tail(v3, T), e0_of(v3), false, precval(T))
    return T(r.neg, r.ex, r.d, r.flag | GE), 4
end

# --- user-facing exp / log: value core + ScalarTot's flag rules (twin-checked) --------------
function Base.exp(a::T) where {T<:MFloat}
    cplx(a) && return zcplx(T)
    r = exp_core(T, a)[1]                                    # e^x > 0: exp(−MAX) = +MIN⟦≤⟧, never 0
    f = a.flag
    f == 0x00 && return r
    (f & SUNK) != 0 && return T(false, r.ex, r.d, r.flag | NOB)   # the sign of e^x is certain: SUNK never leaves
    dir = f & NOB
    dir == NOB && return T(false, r.ex, r.d, r.flag | NOB)
    out = if iszero(a)
        dir == LE ? 0x00 : NOB                               # (0,≤) ⇒ true = 0 ⇒ exactly 1; (0,≥) ⇒ anything
    elseif !a.neg
        dir                                                  # x > 0: monotone, the direction survives
    else
        dir == GE ? LE : GE                                  # x < 0: |true| ≥ |x| means true ≤ x ⇒ e^true ≤ e^x
    end
    T(false, r.ex, r.d, r.flag | out)
end
function Base.log(a::T) where {T<:MFloat}
    cplx(a) && return zcplx(T)
    if a.neg && !iszero(a)                                   # out of ℝ: named as such
        return (a.flag & SUNK) != 0 ? zcplx(T) : T(false, 0, zerod(T), a.flag | CPLX)
    end
    if iszero(a)
        # the reserved word: log 0 = 0 by definition (the same shelf as a/0 = 0).  −∞ is not 0's job
        # but ε's: log(MIN⟦≤⟧) = log MIN⟦≥⟧ falls out of the rule below.  A dangerous zero (⟦≥⟧) has
        # an unknown true value, negative ones included → the ℂ seat.
        return (a.flag & GE) != 0 ? zcplx(T) : zero(T)
    end
    r = log_core(T, a)[1]
    f = a.flag
    f == 0x00 && return r
    (f & SUNK) != 0 && return T(r.neg, r.ex, r.d, r.flag | NOB | SUNK | CPLX)   # a negative true value is complex
    dir = f & NOB
    gt1 = cmp_mag(a, one(T)) > 0
    out = if dir == GE && gt1
        GE                                                   # true ≥ x > 1: log grows
    elseif dir == LE && !gt1
        GE                                                   # 0 < true ≤ x < 1: |log| grows toward 0⁺
    else
        NOB | SUNK                                           # the admissible set may cross 1
    end
    T(r.neg, r.ex, r.d, r.flag | out)
end

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

"""the flag algebra against ScalarTot.TotNum: all 16 × 16 flag pairs on values that reach every branch
(zeros, a value below 1 for log, both signs) × 7 operations.  The flag must agree, and so must
"the value seat is empty" (a 0 placeholder) — the ℂ seat, a/0 and log 0 are checked by that."""
function twin_check(S)
    bad = 0
    for fa in 0x00:0x0f, fb in 0x00:0x0f, va in (-3.0, 3.0, 0.7, 0.0), vb in (-5.0, 5.0, 0.0)
        a = S.TotNum(va, fa); b = S.TotNum(vb, fb)
        xa = F128(va); xb = F128(vb)
        A = F128(xa.neg, xa.ex, xa.d, fa); B = F128(xb.neg, xb.ex, xb.d, fb)
        for (r, R) in ((a + b, A + B), (a - b, A - B), (a * b, A * B), (a / b, A / B),
                       (sqrt(a), sqrt(A)), (exp(a), exp(A)), (log(a), log(A)))
            (r.flag == R.flag && (r.val == 0) == iszero(R)) || (bad += 1)
        end
    end
    return bad
end

bigval(p::NTuple{N,UInt32}) where {N} = sum(big(p[i]) << (DB * (i - 1)) for i in 1:N)

"""reference for `round_pack` by MPFR: the exact value (digits·2^e0, + a quarter LSB when sticky)
rounded to nearest even at Pr bits, then saturated.  Independent of the digit engine."""
function round_pack_ref(::Type{T}, neg::Bool, d::NTuple{N,UInt32}, e0::Int, sticky::Bool, Pr::Int) where {T<:MFloat,N}
    K = ndig(T)
    v = bigval(d)
    iszero(v) && return zero(T)
    u = (v << 2) | (sticky ? big(1) : big(0))                          # exact: v + ¼ (sticky) at weight 2^(e0−2)
    nb = ndigits(u, base = 2)
    r = setprecision(BigFloat, nb) do
        BigFloat(BigFloat(u; precision = nb); precision = Pr)          # → nearest even at Pr bits
    end
    t = Int(r.exp) - 1 + e0 - 2                                         # exponent of the leading bit
    t > emax(T) && return T(neg, emax(T), maxd(T), GE)
    t < emin(T) && return T(neg, emin(T), mind(T), LE)
    m = setprecision(BigFloat, Pr) do; BigInt(ldexp(r, Pr - Int(r.exp))); end     # the Pr-bit significand
    T(neg, t, digits_tuple(m << (DB * K - Pr), Val(K)), 0x00)
end

same_bits(a::T, b::T) where {T<:MFloat} = a.neg === b.neg && a.ex === b.ex && a.flag === b.flag && a.d === b.d

function self_test(; n = 1000, seed = 20260903)
    rng = MersenneTwister(seed)
    println("MultiU32 self-test (NTuple kernels) — digits: $DB bits in UInt32, radix 2^$DB, UInt64 columns, K ≤ 64")
    # 1. digit primitives vs BigInt
    let bad = 0
        for (la, lb) in ((1, 1), (2, 3), (5, 5), (6, 5), (9, 9), (10, 9), (18, 18), (19, 18), (32, 32), (5, 32)), _ in 1:200
            A = rand(rng, big(0):big(2)^(DB * la) - 1); B = rand(rng, big(0):big(2)^(DB * lb) - 1)
            da = digits_tuple(A, Val(la)); dbv = digits_tuple(B, Val(lb))
            bigval(mul_digits(da, dbv)) == A * B || (bad += 1)                       # product
            sh = rand(rng, 0:90)
            w, st = place(da, sh, Val(la + 4))                                       # shifts + sticky
            (bigval(w) == (A << sh) && !st) || (bad += 1)
            w2, st2 = place(da, -sh, Val(la))
            (bigval(w2) == (A >> sh) && st2 == ((A & (big(2)^sh - 1)) != 0)) || (bad += 1)
            s = rand(rng, 0:DB - 1)
            bigval(shl_bits(da, s)) == (A << s) || (bad += 1)
            f, stf = shr_total(da, sh)                                               # right by any number of bits
            (bigval(f) == (A >> sh) && stf == ((A & (big(2)^sh - 1)) != 0)) || (bad += 1)
            nb = ndigits(A, base = 2)
            L, top = topdigit(da)                                                    # leading digit, no loop
            (A == 0 ? (L == 0 && top == Z32) : (L == cld(nb, DB) && top == UInt32(A >> (DB * (L - 1))))) || (bad += 1)
            if A != 0                                                                # normalization: leading bit → top of a W-digit window
                for W in (max(la - 2, 1), la, la + 2)
                    e0 = rand(rng, -1000:1000)
                    w, t, dropped = normalize(da, e0, L, top, Val(W))
                    sh2 = DB * W - nb                                                # window = A·2^sh2 (or A >> −sh2)
                    want = sh2 ≥ 0 ? A << sh2 : A >> -sh2
                    wantdrop = sh2 ≥ 0 ? false : (A & (big(2)^-sh2 - 1)) != 0
                    (bigval(w) == want && t == e0 + nb - 1 && (dropped != Z32) == wantdrop) || (bad += 1)
                end
            end
            if la == lb                                                              # differences + borrow
                hi, lo = max(A, B), min(A, B)
                r, bo = sub_digits(digits_tuple(hi, Val(la)), digits_tuple(lo, Val(la)), Z32)
                (bigval(r) == hi - lo && bo == 0) || (bad += 1)
                if hi > lo
                    r1, bo1 = sub_digits(digits_tuple(hi, Val(la)), digits_tuple(lo, Val(la)), UInt32(1))
                    (bigval(r1) == hi - lo - 1 && bo1 == 0) || (bad += 1)
                    r2, bo2 = sub_digits(digits_tuple(lo, Val(la)), digits_tuple(hi, Val(la)), Z32)
                    (bo2 == 1 && bigval(r2) == big(2)^(DB * la) + lo - hi) || (bad += 1)
                end
                (topbit(da) == (A == 0 ? -1 : ndigits(A, base = 2) - 1)) || (bad += 1)
            end
        end
        # the saturated column: 32 products of the largest digit + the largest carry-in stay below 2^64
        full = ntuple(_ -> MASK, Val(32))
        bigval(mul_digits(full, full)) == (big(2)^(DB * 32) - 1)^2 || (bad += 1)
        full40 = ntuple(_ -> MASK, Val(40))                                          # crosses the 32-row fold
        bigval(mul_digits(full40, full40)) == (big(2)^(DB * 40) - 1)^2 || (bad += 1)
        for _ in 1:50
            A = rand(rng, big(0):big(2)^(DB * 42) - 1); B = rand(rng, big(0):big(2)^(DB * 42) - 1)
            bigval(mul_digits(digits_tuple(A, Val(42)), digits_tuple(B, Val(42)))) == A * B || (bad += 1)
            bigval(mul_digits_low(digits_tuple(A, Val(42)), digits_tuple(B, Val(42)), Val(50))) == (A * B) & (big(2)^(DB * 50) - 1) || (bad += 1)
        end
        println("  primitives (mul_digits / place / shr_total / shl_bits / sub_digits / topdigit / normalize): bad = $bad")
        bad == 0 || return false
    end
    # 1b. normalization + fixed-position rounding vs MPFR on random windows: every input width the
    #     operations use (2 digits … 2K), the working and the full precision, leading bit anywhere,
    #     ties, all-ones kept parts (the increment ripples out to a power of two), range edges, sticky.
    #     Negative control: the same comparison with the round bit flipped must be detected.
    let bad = 0, ctrl = 0, cases = 0
        for T in (F128, F256, F512), Pr in (prec(T), DB * ndig(T))
            K = ndig(T)
            for N in (2, K, K + 1, K + 3, 2 * K), _ in 1:120
                nb = rand(rng, 1:DB * N)                                            # bit length of the window value
                v = rand(rng, big(2)^(nb - 1):big(2)^nb - 1)
                if nb > Pr
                    kind = rand(rng, 1:4)
                    kind == 1 && (v |= (big(2)^Pr - 1) << (nb - Pr))                # kept part all ones
                    kind == 2 && (v = ((v >> (nb - Pr)) << (nb - Pr)) | big(2)^(nb - Pr - 1))   # exact tie
                    kind == 3 && (v = ((v >> (nb - Pr)) << (nb - Pr)))              # exact, nothing below
                end
                st = nb ≥ Pr + 1 && rand(rng, Bool)                                 # a sticky needs the round bit inside the window
                t = rand(rng, 1:6) == 1 ? rand(rng, (emax(T) - 2, emax(T) - 1, emax(T), emax(T) + 1, emin(T) - 1, emin(T), emin(T) + 1)) : rand(rng, -3000:3000)
                e0 = t - (nb - 1)
                d = digits_tuple(v, Val(N)); ng = rand(rng, Bool)
                got  = round_pack(T, ng, d, e0, st, Val(Pr))                        # (dynamic dispatch on Val(Pr): test only)
                want = round_pack_ref(T, ng, d, e0, st, Pr)
                same_bits(got, want) || (bad += 1)
                cases += 1
                if nb > Pr                                                          # negative control: flip the round bit
                    d2 = digits_tuple(v ⊻ big(2)^(nb - Pr - 1), Val(N))
                    same_bits(got, round_pack_ref(T, ng, d2, e0, st, Pr)) || (ctrl += 1)
                end
            end
        end
        println("  round_pack vs MPFR reference on $cases random windows: bad = $bad   (negative control: $ctrl flipped round bits detected)")
        (bad == 0 && ctrl > 0) || return false
    end
    allok = true
    for T in (F128, F256, F512)
        P, K = prec(T), ndig(T)
        log = Any[]; o_ = one(T)
        println("  $(T): P=$P bits, K=$K digits ($(DB*K) bits), emax=$(emax(T))")
        # 2. conversion round trip
        for _ in 1:300
            x = rand_val(T, rng, rand_exp(T, rng))
            check("roundtrip", T(BigFloat(x)), x, log)
            BigFloat(T(BigFloat(x))) == BigFloat(x) || push!(log, ("roundtrip-bf", x, x))
        end
        # 3. random operands vs MPFR
        let w = rand_val(T, rng, 0), v = rand_val(T, rng, 1)                 # JIT warm-up (not timed)
            w + v; w - v; w * v; w / v; sqrt(w); exp(w); Base.log(w); exp(T(ldexp(BigFloat(1), -P))); Base.log(o_ + eps(T))
        end
        t_add = t_mul = t_div = t_sqrt = t_exp = t_log = 0.0
        ziv = zeros(Int, 5)                                                  # Ziv stage histogram (index = stage + 1)
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
            x = T(a.neg, rand(rng, -P - 10:22), a.d, 0x00)                    # exp: from "rounds to 1" to past overflow
            t0 = time(); ex = exp(x); t_exp += time() - t0
            check("exp", ex, oracle(T, exp, x), log)
            t0 = time(); lg = Base.log(abs(a)); t_log += time() - t0        # log: any exponent
            check("log", lg, oracle(T, Base.log, abs(a)), log)
            ziv[exp_core(T, x)[2] + 1] += 1; ziv[log_core(T, abs(a))[2] + 1] += 1
        end
        # 4. adversarial constructions
        o = one(T); ulp1 = eps(T); two_ = T(2.0)
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
            # exp / log hard neighbourhoods: |x| tiny with P significant bits (e^x = 1 + x + x²/2: the
            # midpoint + a 2P-bit tail lives here), x near 1, and the saturation boundaries
            tx = rand_val(T, rng, -P - rand(rng, 0:P ÷ 2)); rand(rng, Bool) && (tx = -tx)
            check("exp-tiny", exp(tx), oracle(T, exp, tx), log)
            n1 = o + ulp1 * T(rand(rng, -2^16:2^16))
            check("log-near1", Base.log(n1), oracle(T, Base.log, n1), log)
            p2 = T(ldexp(BigFloat(1), rand(rng, -emax(T):emax(T))))
            check("log-pow2", Base.log(p2), oracle(T, Base.log, p2), log)
            bx = T(ldexp(BigFloat(1), e)) * T(0.6931471805599453) * T(rand(rng, 1:10))
            check("exp-kln2", exp(bx), oracle(T, exp, bx), log)             # x ≈ k·ln2: r ≈ 0
        end
        for x in (T(ldexp(BigFloat(1), -P)), -T(ldexp(BigFloat(1), -P - 1)), T(ldexp(BigFloat(1), -P + 1)),
                  T(ldexp(BigFloat(1), -P)) + T(ldexp(BigFloat(1), -2P)), -T(ldexp(BigFloat(1), -P - 1)) - T(ldexp(BigFloat(1), -2P)))
            check("exp-midpoint", exp(x), oracle(T, exp, x), log)          # decided only at 2P+ bits
            ziv[exp_core(T, x)[2] + 1] += 1
        end
        for x in (o + ulp1, o - ulp1 / two_, o - ulp1, o + ulp1 + ulp1)
            check("log-1±ulp", Base.log(x), oracle(T, Base.log, x), log)
            ziv[log_core(T, x)[2] + 1] += 1
        end
        # 5. range edges and total-arithmetic rules
        MX = floatmax(T); MN = floatmin(T); two = T(2.0); z = zero(T)
        lnmax = Base.log(MX); lnmin = Base.log(MN)
        check("log(MAX)", lnmax, oracle(T, Base.log, MX), log)
        check("log(MIN)", lnmin, oracle(T, Base.log, MN), log)
        check("exp(log MAX)", exp(lnmax), oracle(T, exp, lnmax), log)
        check("exp(log MIN)", exp(lnmin), oracle(T, exp, lnmin), log)
        check("exp(MAX)=MAX⟦≥⟧", exp(MX), T(false, emax(T), maxd(T), GE), log)
        check("exp(−MAX)=+MIN⟦≤⟧", exp(-MX), T(false, emin(T), mind(T), LE), log)
        check("exp(±MIN)=1", exp(MN), o, log); check("exp(−MIN)=1", exp(-MN), o, log)
        check("exp(0)=1", exp(z), o, log)
        check("log(1)=0", Base.log(o), z, log)
        check("log(0)=0 (reserved word)", Base.log(z), z, log)
        check("log(−1)=0⟦ℂ⟧", Base.log(-o), T(false, 0, zerod(T), CPLX), log)
        check("log(MIN⟦≤⟧)=log MIN⟦≥⟧", Base.log(T(false, emin(T), mind(T), LE)), T(lnmin.neg, lnmin.ex, lnmin.d, GE), log)
        check("log(MAX⟦≥⟧)=log MAX⟦≥⟧", Base.log(T(false, emax(T), maxd(T), GE)), T(false, lnmax.ex, lnmax.d, GE), log)
        check("exp(0⟦≥⟧)=1⟦≥≤⟧", exp(T(false, 0, zerod(T), GE)), T(false, 0, mind(T), NOB), log)
        check("log(0⟦≥⟧)=ℂ seat", Base.log(T(false, 0, zerod(T), GE)), zcplx(T), log)
        check("ℂ sticks: ℂ+1", zcplx(T) + o, zcplx(T), log)
        check("ℂ sticks: exp ℂ", exp(zcplx(T)), zcplx(T), log)
        check("ℂ^0 = 1", zcplx(T)^0, o, log)
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
        println("    vs MPFR: n=$n random ×(add,sub,mul,div,sqrt,exp,log) + 200×19 adversarial + edges → mismatches = $nbad" *
                "   [max residual fix-ups: div $mf_div, sqrt $mf_sqrt; Ziv stages 1/2/3 $(ziv[2:4]), undecided $(ziv[5])]")
        println("    time/op: add $(round(1e6*t_add/(2n), digits=1)) µs, mul $(round(1e6*t_mul/n, digits=1)) µs, " *
                "div $(round(1e6*t_div/n, digits=1)) µs, sqrt $(round(1e6*t_sqrt/n, digits=1)) µs, " *
                "exp $(round(1e6*t_exp/n, digits=1)) µs, log $(round(1e6*t_log/n, digits=1)) µs  (NTuple kernels; per-op time() included)")
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
        println("  flag algebra vs ScalarTot.TotNum (16×16 flags × 12 value pairs × 7 ops): mismatches = $bad")
        bad == 0 || (allok = false)
    end
    # 8. demo
    s2 = sqrt(F512(2))
    ref = setprecision(BigFloat, 489) do; sqrt(BigFloat(2)); end
    println("  √2 in F512 = ", s2)
    println("  == MPFR(489 bits): ", BigFloat(s2) == ref)
    e1 = exp(one(F512)); l2 = Base.log(F512(2))
    refe = setprecision(BigFloat, 489) do; exp(BigFloat(1)); end
    refl = setprecision(BigFloat, 489) do; log(BigFloat(2)); end
    println("  e in F512  = ", e1, "   == MPFR: ", BigFloat(e1) == refe)
    println("  ln2 in F512 = ", l2, "   == MPFR: ", BigFloat(l2) == refl)
    println(allok ? "PASS" : "FAIL")
    return allok
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    n = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 1000
    MultiU32.self_test(n = n) || exit(1)
end
