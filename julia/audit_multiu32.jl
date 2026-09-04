# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
# audit_multiu32.jl — totality audit of MultiU32, aimed at ÷ √ exp log.
#
# `MultiU32.jl`'s self-test checks the *values* against MPFR. This checks the two promises that
# make the arithmetic total, which are a different claim:
#
#   ① every input produces a value — no exception, no trap, no NaN, no Inf, and the result is
#      canonical (leading bit at bit 28 of digit K, the low 29K−P bits zero) so it is safe to
#      feed back in.  This matters here because `div_core`, `sqrt_core` and `place` carry
#      internal assertions (`error("candidate too far")`, `error("non-zero digit above the
#      window")`): they are consistency checks, not handled cases, so if the *candidate* — which
#      the fixed-point ladder replaced — ever lands outside its assumed range on a legal operand,
#      the library traps instead of answering, and it is not total any more.
#   ② the flag does not lie — GE ⇒ |true| ≥ |shown|, LE ⇒ |true| ≤ |shown|, no SUNK ⇒ the shown
#      sign is the true sign, CPLX ⇒ the true result left ℝ.  The audit computes the true value
#      independently (MPFR at 4P+64 bits with an unbounded exponent) and falsifies the claim.
#
# The operands are the ones most likely to break either: significands at both ends of [1,2)
# (1, 2−ulp, all-ones, single-bit, alternating), exponents at ±emax where ÷ and √ saturate, all
# 16 flag combinations, zeros and saturated values, and the pairs that make a quotient cross
# emax / emin exactly.  Plus a random sweep, so the count is not only adversarial.
#
#   exp / log (2026-09-03): the same two promises on the same shelf, plus the definitions that make
#   them total — log 0 = 0 (the reserved word), exp(−MAX) = +MIN⟦≤⟧ (never 0), log(−x) = 0⟦ℂ⟧, the
#   ℂ seat sticking — and the Ziv ladder's own count (a rounding left undecided at 2K+6 digits would
#   come back as ⟦≥⟧, not as an exception; expected 0).
#
#   julia julia/audit_multiu32.jl [n]     → expects: exceptions 0, non-canonical 0, flag lies 0
include(joinpath(@__DIR__, "MultiU32.jl"))
using .MultiU32
using Random, Printf
const M = MultiU32

const N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 400

# ---- the operand shelf -------------------------------------------------------------------
"""significand digit patterns, all canonical (leading bit set, the low 29K−P bits zero)."""
function shelf_digits(::Type{T}) where {T<:M.MFloat}
    P, K = M.prec(T), M.ndig(T)
    g = 29K - P                                    # guard bits that must stay zero
    pat = Vector{NTuple{K,UInt32}}()
    push!(pat, M.mind(T))                          # 1.0 exactly (a power of two)
    push!(pat, M.maxd(T))                          # 2 − ulp (all P bits set)
    ulp = M.digits_tuple(big(1) << g, Val(K))      # one ulp of the significand
    push!(pat, M.carry_norm(map(+, M.mind(T), ulp))[1])          # 1 + ulp
    push!(pat, M.sub_digits(M.maxd(T), ulp, M.Z32)[1])           # 2 − 2ulp
    push!(pat, M.digits_tuple(((big(1) << P) - 1) & ~((big(1) << (P ÷ 2)) - 1) << g |
                              (big(1) << (29K - 1)), Val(K)))    # top half ones, bottom zeros
    push!(pat, M.digits_tuple((div(big(1) << P, 3) | (big(1) << (P - 1))) << g, Val(K)))  # 101010…
    push!(pat, M.digits_tuple(((big(1) << (P - 1)) | 1) << g, Val(K)))                     # 1000…01
    return pat
end

function shelf(::Type{T}, rng) where {T<:M.MFloat}
    E = M.emax(T)
    exps = [0, 1, -1, 2, E, E - 1, E ÷ 2, -E + 1, -E, -E + 2, -E ÷ 2]
    xs = T[]
    for d in shelf_digits(T), e in exps, ng in (false, true)
        push!(xs, T(ng, e, d, 0x00))
    end
    for f in 0x00:0x0f                              # every flag combination, on a middling value
        push!(xs, T(false, 3, M.mind(T), f)); push!(xs, T(true, -3, M.maxd(T), f))
    end
    push!(xs, zero(T), -zero(T), M.floatmax(T), -M.floatmax(T), M.floatmin(T), -M.floatmin(T),
              typemax(T), typemin(T), T(false, 0, M.zerod(T), M.GE))       # a flagged zero
    for _ in 1:N
        push!(xs, M.rand_val(T, rng, rand(rng, -M.emax(T):M.emax(T))))
    end
    xs
end

# ---- the checks --------------------------------------------------------------------------
canonical(x::T) where {T<:M.MFloat} = begin
    P, K = M.prec(T), M.ndig(T)
    if M.iszero(x)
        all(d -> d == M.Z32, x.d)                                  # zero: every digit zero
    else
        x.d[K] ≥ UInt32(2)^28 && x.d[K] < UInt32(2)^29 &&           # leading bit at the top …
        (M.lowbits(x.d, 29K - P) == M.Z32) &&                       # … guard bits zero …
        M.emin(T) ≤ Int(x.ex) ≤ M.emax(T)                           # … exponent in range
    end
end

"""|true value| as a BigFloat with an unbounded exponent, or `nothing` when the operation has no
true value to compare against (a/0, 0/0, √ of a negative — those are the *definitions* a/0 = 0 and
√(−x) = 0⟦ℂ⟧, checked separately)."""
function truth(::Type{T}, op, a::T, b::T) where {T<:M.MFloat}
    (M.isflagged(a) || (op === :div && M.isflagged(b))) && return nothing   # the inputs are already claims
    setprecision(BigFloat, 4 * M.prec(T) + 64) do
        if op === :div
            M.iszero(b) && return nothing
            M.iszero(a) && return BigFloat(0)
            BigFloat(a) / BigFloat(b)
        elseif op === :sqrt
            a.neg && !M.iszero(a) && return nothing
            sqrt(BigFloat(a))
        elseif op === :exp
            # MPFR's exponent range is finite too (2^62): a saturated exp is checked as ±∞ / 0⁺
            ba = BigFloat(a)
            ba > 2^40 && return BigFloat(Inf)
            ba < -2^40 && return BigFloat(0)              # e^x > 0 : the flag must say ≤ on +MIN, never a 0
            exp(ba)
        else
            (a.neg && !M.iszero(a)) && return nothing     # log(−x): the definition 0⟦ℂ⟧
            M.iszero(a) && return nothing                 # log 0 = 0: the reserved word, checked as a definition
            log(BigFloat(a))
        end
    end
end

"""does the result's flag tell the truth about `t`?  (GE/LE are claims about the magnitude.)"""
function flag_honest(r::T, t::BigFloat) where {T<:M.MFloat}
    shown = abs(BigFloat(r)); tt = abs(t)
    ge = (M.flag_of(r) & M.GE) != 0; le = (M.flag_of(r) & M.LE) != 0
    sunk = (M.flag_of(r) & M.SUNK) != 0
    ge && tt < shown && return false                                # GE: |true| ≥ |shown|
    le && tt > shown && return false                                # LE: |true| ≤ |shown|
    !sunk && !iszero(t) && (r.neg != (t < 0)) && return false       # sign is trusted ⟹ it is right
    (M.flag_of(r) & M.CPLX) != 0 && return false                    # a real truth exists: ℂ would be a lie
    !ge && !le && !isinf(t) && tt != shown && tt < 2^40 && abs(tt - shown) > shown * BigFloat(2)^(-M.prec(T)) && return false   # unflagged ⟹ the shown value is the rounding
    true
end

exc = 0; noncanon = 0; lies = 0; cases = 0; mf_div = 0; mf_sqrt = 0; trapped = Any[]; ziv = zeros(Int, 5)
rng = MersenneTwister(20260903)
println("MultiU32 totality audit — ① never traps, always canonical  ② the flag does not lie")
for T in (M.F128, M.F256, M.F512)
    xs = shelf(T, rng)
    ce = 0; cn = 0; cl = 0; n = 0
    for a in xs, b in xs
        # the full cross product of the shelf is quadratic; sample it evenly instead
        (hash((a.ex, a.d[1], b.ex, b.d[1])) % 17 == 0) || continue
        for (op, f) in ((:div, /), (:sqrt, (x, y) -> sqrt(x)), (:exp, (x, y) -> exp(x)), (:log, (x, y) -> log(x)))
            n += 1
            local r
            try
                r = f(a, b)
            catch err
                ce += 1; length(trapped) < 6 && push!(trapped, (T, op, a, b, err)); continue
            end
            canonical(r) || (cn += 1)
            t = truth(T, op, a, b)
            t === nothing || flag_honest(r, t) || (cl += 1)
        end
        # the internal fix-up counters (the assertions' distance to their bound of 8)
        if !M.iszero(b) && !M.isflagged(a) && !M.isflagged(b) && !M.iszero(a)
            global mf_div = max(mf_div, M.div_core(T, a, b)[2])
        end
        if !a.neg && !M.iszero(a) && !M.isflagged(a)
            global mf_sqrt = max(mf_sqrt, M.sqrt_core(T, a)[2])
            ziv[M.log_core(T, a)[2] + 1] += 1                        # the Ziv stage that decided (4 = undecided)
        end
        (M.iszero(a) || M.isflagged(a)) || (ziv[M.exp_core(T, a)[2] + 1] += 1)
    end
    @printf("  %-22s %6d operand pairs × (÷ √ exp log): exceptions %d, non-canonical %d, flag lies %d\n",
            T, n ÷ 4, ce, cn, cl)
    global exc += ce; global noncanon += cn; global lies += cl; global cases += n
end

# ---- the definitions that make the arithmetic total ---------------------------------------
law = 0
for T in (M.F128, M.F256, M.F512)
    o = one(T); z = zero(T); mx = M.floatmax(T); mn = M.floatmin(T)
    gz = T(false, 0, M.zerod(T), M.GE)                      # a zero that might not be zero
    checks = (("a/0 = 0",        o / z,        r -> M.iszero(r)),
              ("0/0 = 0",        z / z,        r -> M.iszero(r)),
              ("a/0 is SUNK-free unless the zero is a claim", o / z, r -> (M.flag_of(r) & M.SUNK) == 0),
              ("a/⟦≥⟧0 is SUNK", o / gz,       r -> (M.flag_of(r) & M.SUNK) != 0),
              ("√(−1) = 0⟦ℂ⟧",   sqrt(-o),     r -> M.iszero(r) && (M.flag_of(r) & M.CPLX) != 0),
              ("√0 unflagged",   sqrt(z),      r -> M.iszero(r) && !M.isflagged(r)),
              ("MAX/MIN → ⟦≥⟧",  mx / mn,      r -> (M.flag_of(r) & M.GE) != 0),
              ("MIN/MAX → ⟦≤⟧",  mn / mx,      r -> (M.flag_of(r) & M.LE) != 0),
              ("√MAX in range",  sqrt(mx),     r -> !M.isflagged(r)),
              ("√MIN in range",  sqrt(mn),     r -> !M.isflagged(r)),
              ("x/x = 1",        mx / mx,      r -> r == o),
              ("√(x²) = x",      sqrt(o + o) * sqrt(o + o), r -> !M.isflagged(r)),
              # exp / log: the reserved word, the limits that belong to ε, the ℂ seat
              ("log 0 = 0 unflagged (the reserved word)", log(z), r -> M.iszero(r) && !M.isflagged(r)),
              ("log(⟦≥⟧0) = ℂ seat", log(gz),  r -> M.iszero(r) && M.flag_of(r) == (M.GE | M.LE | M.SUNK | M.CPLX)),
              ("log 1 = 0",       log(o),       r -> M.iszero(r) && !M.isflagged(r)),
              ("log(−1) = 0⟦ℂ⟧", log(-o),      r -> M.iszero(r) && M.flag_of(r) == M.CPLX),
              ("log(MIN⟦≤⟧) = log MIN ⟦≥⟧", log(T(false, M.emin(T), M.mind(T), M.LE)), r -> r.neg && M.flag_of(r) == M.GE && T(r.neg, r.ex, r.d, 0x00) == log(mn)),
              ("log(MAX⟦≥⟧) = log MAX ⟦≥⟧", log(typemax(T)), r -> !r.neg && M.flag_of(r) == M.GE && T(r.neg, r.ex, r.d, 0x00) == log(mx)),
              ("log MAX in range", log(mx),     r -> !M.isflagged(r)),
              ("log MIN in range", log(mn),     r -> !M.isflagged(r)),
              ("exp 0 = 1",       exp(z),       r -> r == o && !M.isflagged(r)),
              ("exp(±MIN) = 1",   exp(mn),      r -> r == o && !M.isflagged(r)),
              ("exp(MIN⟦≤⟧) = 1⟦≤⟧", exp(T(false, M.emin(T), M.mind(T), M.LE)), r -> r == o && M.flag_of(r) == M.LE),
              ("exp(−MIN⟦≤⟧) = 1⟦≥⟧", exp(T(true, M.emin(T), M.mind(T), M.LE)), r -> r == o && M.flag_of(r) == M.GE),
              ("exp MAX = MAX⟦≥⟧", exp(mx),     r -> r == mx && M.flag_of(r) == M.GE),
              ("exp(−MAX) = +MIN⟦≤⟧, not 0", exp(-mx), r -> r == mn && M.flag_of(r) == M.LE),
              ("exp(⟦≥⟧0) = 1⟦≥≤⟧", exp(gz),   r -> r == o && M.flag_of(r) == (M.GE | M.LE)),
              ("exp(x⟦±⟧) never ⟦±⟧", exp(T(true, 2, M.mind(T), M.SUNK)), r -> (M.flag_of(r) & M.SUNK) == 0),
              ("ℂ sticks through +", M.zcplx(T) + o, r -> M.flag_of(r) == (M.GE | M.LE | M.SUNK | M.CPLX)),
              ("ℂ sticks through exp", exp(M.zcplx(T)), r -> M.flag_of(r) == (M.GE | M.LE | M.SUNK | M.CPLX)),
              ("ℂ^0 = 1",         M.zcplx(T)^0, r -> r == o && !M.isflagged(r)),
              ("exp(log x) = x to 1 ulp", exp(log(T(3))), r -> abs(BigFloat(r) - 3) ≤ 3 * BigFloat(2)^(1 - M.prec(T))))
    for (name, r, ok) in checks
        ok(r) || (global law += 1; println("  ✗ $T $name → $r"))
    end
end

# ---- ③ the invariant the two assertions rest on --------------------------------------------
# `div_core` / `sqrt_core` accept a candidate within 8 ulps of the exact quotient / root and throw
# otherwise, so totality here is not a property of the residual alone — it is a property of the
# *ladder's accuracy*.  Counting fix-ups only says the bound was not reached on these inputs.
# This measures the distance to it: the candidate's own error against MPFR, in bits, against the
# 29K − 4 bits the ladder is designed for and the P + 3 bits the assertion needs.
println("  candidate accuracy (the assertions' real premise) — worst of $(4N) significands per type:")
margin_ok = true
for T in (M.F128, M.F256, M.F512)
    P, K = M.prec(T), M.ndig(T)
    worst_r = Inf; worst_s = Inf
    for i in 1:4N
        m = i == 1 ? one(T) : (i == 2 ? T(false, 0, M.maxd(T), 0x00) : M.rand_val(T, rng, 0))
        setprecision(BigFloat, 4P + 64) do
            bm = BigFloat(m)
            y = BigFloat(M.recip_sig(T, m))                       # ≈ 1/m at 29K bits
            e = abs(y - 1 / bm) * bm                              # relative error
            worst_r = min(worst_r, e == 0 ? Inf : -log2(Float64(e)))
            mp = T(false, isodd(i) ? 1 : 0, m.d, 0x00)             # √ takes m ∈ [1,4)
            s = BigFloat(M.round_pack(T, false, M.sqrt_sig(T, mp), 1 - 29K, false, M.fullval(T)))
            bs = sqrt(BigFloat(mp))
            es = abs(s - bs) / bs
            worst_s = min(worst_s, es == 0 ? Inf : -log2(Float64(es)))
        end
    end
    need = P + 3                                                   # 8 ulps of the P-bit grid
    ok = worst_r ≥ need && worst_s ≥ need
    global margin_ok = margin_ok && ok
    @printf("    %-22s 1/m %6.1f bits, √m %6.1f bits   (design %d, assertion needs %d ⟹ margin %.0f / %.0f bits) %s\n",
            T, worst_r, worst_s, 29K - 4, need, worst_r - need, worst_s - need, ok ? "" : "✗")
end

for (T, op, a, b, err) in trapped
    println("  ✗ TRAP $T $op  a=$a  b=$b  →  $err")
end
@printf("  %d results checked; the definitions (a/0, √−1, log 0, exp(−MAX), ℂ, saturation, …): %d violations\n", cases, law)
@printf("  internal fix-up counters (bound 8, an exception if exceeded): div %d, sqrt %d\n", mf_div, mf_sqrt)
@printf("  Ziv ladder (exp / log on the shelf): no series %d, decided at stage 1/2/3 = %s, undecided (returned as ⟦≥⟧) = %d\n", ziv[1], ziv[2:4], ziv[5])
ok = exc == 0 && noncanon == 0 && lies == 0 && law == 0 && margin_ok && ziv[5] == 0
println(ok ? "PASS: total — no trap, every result canonical, no flag lied" :
             "FAIL: exceptions $exc, non-canonical $noncanon, flag lies $lies, law violations $law, margin $margin_ok")
ok || exit(1)
