# ab_multiu32.jl — bitwise A/B between the two MultiU32 implementations
#
#   MultiU32Ref.jl : the first version — dynamic tuple indices, scan loops, sign branches
#                    (the plain way to write it; local memory + warp divergence on a GPU)
#   MultiU32.jl    : barrel shifters, normalize-then-round at a static bit position, selects
#                    (3.3–5.4× faster on the RTX 5090 for F512) — must reproduce every bit of the above
#
# Same seeded operands go through both; every result is compared field by field — sign, exponent,
# flag and every digit.  A negative control flips one bit of one digit and must be reported as a
# mismatch, otherwise the comparator itself is broken.
#
#   julia ab_multiu32.jl [n]          (n operand pairs per type and operation, default 2000)

include(joinpath(@__DIR__, "MultiU32Ref.jl"))
include(joinpath(@__DIR__, "MultiU32.jl"))
using Random, Printf

const N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 2000

fields(x) = (x.neg, Int(x.ex), UInt8(x.flag), collect(x.d))
same(x, y) = fields(x) == fields(y)

conv(::Type{T}, x) where {T} = T(x.neg, x.ex, x.d, x.flag)      # Ref value → new value, same bits

const PAIRS = ((MultiU32Ref.F128, MultiU32.F128), (MultiU32Ref.F256, MultiU32.F256), (MultiU32Ref.F512, MultiU32.F512))

function operands(TR, rng, n)
    # ordinary magnitudes with random signs (both add paths), plus the edges: zero, saturated, and
    # values near ±emax / −emax so that the results carry GE / LE / SUNK flags and the saturation
    # paths get exercised too
    E = TR.parameters[3]
    rnd(e) = (v = MultiU32Ref.rand_val(TR, rng, e); rand(rng, Bool) ? TR(true, v.ex, v.d, v.flag) : v)
    xs = [rnd(rand(rng, -8:8)) for _ in 1:n]
    edge = [rnd(e) for e in (E, E - 1, E - 3, -E + 1, -E + 3, E ÷ 2, -E ÷ 2)]
    push!(edge, zero(TR))
    push!(edge, typemax(TR)); push!(edge, typemin(TR)); push!(edge, floatmax(TR)); push!(edge, floatmin(TR))
    for _ in 1:max(8, n ÷ 8)
        push!(xs, rand(rng, edge))
    end
    # exponent gaps up to P+40 (the sticky-only path of add) and near-equal magnitudes (cancellation)
    P = TR.parameters[1]
    for _ in 1:max(8, n ÷ 8)
        push!(xs, rnd(rand(rng, -(P + 40):(P + 40))))
    end
    xs
end

total_bad = 0
neg_control_ok = false
rng = MersenneTwister(20260903)
println("MultiU32 bitwise A/B (first version with dynamic indices and loops vs barrel shifters and static rounding), n=$N pairs per op + edges")
for (TR, TN) in PAIRS
    global total_bad, neg_control_ok
    xs = operands(TR, rng, N); ys = operands(TR, rng, N)
    # near-cancellation pairs: y = −x with one of the two lowest kept bits flipped (1–3 ulp apart;
    # the operand stays canonical because only bits at or above the lsb of the precision change)
    for i in 1:max(8, N ÷ 8)
        j = rand(rng, 1:N)                                       # a random operand, never an edge
        x = xs[j]
        g = 29 * MultiU32Ref.ndig(TR) - TR.parameters[1] + rand(rng, 0:1)   # global bit index of the flip
        d = collect(x.d); d[g ÷ 29 + 1] ⊻= UInt32(1) << (g % 29)
        ys[j] = TR(!x.neg, x.ex, Tuple(d), x.flag)
    end
    xn = [conv(TN, x) for x in xs]; yn = [conv(TN, y) for y in ys]
    all(i -> same(xs[i], xn[i]) && same(ys[i], yn[i]), eachindex(xs)) || error("operand conversion is not bit-identical")
    @printf("  %-22s K=%2d  ", TN, MultiU32.ndig(TN))
    for (name, f) in (("add", +), ("sub", -), ("mul", *), ("div", /), ("sqrt", (a, b) -> sqrt(abs(a))), ("lt", (a, b) -> a < b))
        bad = 0
        for i in eachindex(xs)
            r = f(xs[i], ys[i]); s = f(xn[i], yn[i])
            ok = r isa Bool ? (r == s) : same(r, s)
            bad += !ok
        end
        total_bad += bad
        @printf("%s %d  ", name, bad)
    end
    # conversions: Float64 and BigFloat in, BigFloat out
    P = TR.parameters[1]
    bad = 0
    for i in 1:N
        x = ldexp(rand(rng) - 0.5, rand(rng, -60:60))
        bad += !same(TR(x), TN(x))
        b = setprecision(BigFloat, P + 40) do; BigFloat(x) * (1 + BigFloat(rand(rng)) * 1e-30); end
        bad += !same(TR(b), TN(b))
        bad += !(setprecision(BigFloat, P) do; BigFloat(xs[i]) == BigFloat(xn[i]); end)
    end
    total_bad += bad
    @printf("conv %d\n", bad)
    # negative control: flip the lowest bit of one digit of one result — the comparator must see it
    r = xs[1] * ys[1]
    d = collect(r.d); d[1] = d[1] ⊻ 0x00000001
    r2 = TR(r.neg, r.ex, Tuple(d), r.flag)
    neg_control_ok |= !same(r, conv(TN, r2))
end
println(neg_control_ok ? "  negative control (1 flipped digit bit): mismatch detected — comparator alive" :
                         "  negative control FAILED: a flipped bit was not detected")
println(total_bad == 0 && neg_control_ok ? "PASS: all results bit-identical" : "FAIL: $total_bad mismatches")
