# ab_multif32.jl — bitwise A/B between the two MultiF32 implementations
#
#   MultiF32Ref.jl : Vector{Float32} digits, heap per operation (the reference kept for this test)
#   MultiF32.jl    : NTuple{K,Float32} digits, straight-line kernels, no heap (the fast version)
#
# Same seeded operands go through both; every result is compared field by field — sign, exponent,
# flag and the bit pattern (reinterpret UInt32) of every digit.  A negative control flips one bit
# of one digit and must be reported as a mismatch, otherwise the comparator itself is broken.
#
#   julia ab_multif32.jl [n]          (n operand pairs per type and operation, default 2000)

include(joinpath(@__DIR__, "MultiF32Ref.jl"))
include(joinpath(@__DIR__, "MultiF32.jl"))
using Random, Printf

const N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 2000

# (neg, ex, flag, digit bit patterns): identical for the two representations when the numbers agree
fields(x) = (x.neg, Int(x.ex), UInt8(x.flag), [reinterpret(UInt32, Float32(v)) for v in x.d])
same(x, y) = fields(x) == fields(y)

conv(::Type{T}, x) where {T} = T(x.neg, x.ex, x.d, x.flag)      # Ref value → NTuple value, same bits

const PAIRS = ((MultiF32Ref.F128, MultiF32.F128), (MultiF32Ref.F256, MultiF32.F256), (MultiF32Ref.F512, MultiF32.F512))

function operands(TR, rng, n)
    # ordinary magnitudes, plus the edges: zero, saturated, and values near ±emax / −emax so that the
    # results carry GE / LE / SUNK flags and the saturation paths get exercised too
    E = TR.parameters[3]
    xs = [MultiF32Ref.rand_val(TR, rng, rand(rng, -8:8)) for _ in 1:n]
    edge = [MultiF32Ref.rand_val(TR, rng, e) for e in (E, E - 1, E - 3, -E + 1, -E + 3, E ÷ 2, -E ÷ 2)]
    push!(edge, zero(TR))
    push!(edge, typemax(TR)); push!(edge, typemin(TR)); push!(edge, floatmax(TR))
    for _ in 1:max(8, n ÷ 8)
        push!(xs, rand(rng, edge))
    end
    xs
end

total_bad = 0
neg_control_ok = false
rng = MersenneTwister(20260903)
println("MultiF32 bitwise A/B (Vector reference vs NTuple kernels), n=$N pairs per op + edges")
for (TR, TN) in PAIRS
    global total_bad, neg_control_ok
    xs = operands(TR, rng, N); ys = operands(TR, rng, N)
    xn = [conv(TN, x) for x in xs]; yn = [conv(TN, y) for y in ys]
    all(i -> same(xs[i], xn[i]) && same(ys[i], yn[i]), eachindex(xs)) || error("operand conversion is not bit-identical")
    @printf("  %-22s K=%2d  ", TN, MultiF32.ndig(TN))
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
    d = collect(r.d); d[1] = reinterpret(Float32, reinterpret(UInt32, d[1]) ⊻ 0x00000001)
    r2 = TR(r.neg, r.ex, Tuple(d), r.flag)
    neg_control_ok |= !same(r, conv(TN, r2))
end
println(neg_control_ok ? "  negative control (1 flipped digit bit): mismatch detected — comparator alive" :
                         "  negative control FAILED: a flipped bit was not detected")
println(total_bad == 0 && neg_control_ok ? "PASS: all results bit-identical" : "FAIL: $total_bad mismatches")
