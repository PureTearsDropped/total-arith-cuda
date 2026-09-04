# MultiF32 (F128/F256/F512) vs MPFR at the same precision — CPU, single thread.
# Reproduces the speed paragraph of README.md (§ MultiF32.jl):  julia julia/bench_multif32.jl
#
# Four columns: MultiF32Ref (Vector digits, the allocation-bound reference), MultiF32 (NTuple
# kernels, no heap), MPFR through Julia's BigFloat (allocates every result: 2 allocations per
# operation, GC included) and MPFR the C library itself — mpfr_add / mul / div / sqrt called into a
# preallocated result, which is what a C caller pays and the number "MPFR on one core" means.
# The BigFloat column was the only MPFR column at first and it overstated MPFR by 1.5–4×: the
# allocation and the collector are Julia's cost, not MPFR's.  Same operands for all four columns;
# every time is the minimum of 5 passes (a pass that hits a collection is not the cost of the operation).
include(joinpath(@__DIR__, "MultiF32Ref.jl"))
include(joinpath(@__DIR__, "MultiF32.jl"))
using Random, Printf

const N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 20_000

# every result is consumed (its exponent field is summed) so that no operation can be dead-code eliminated
sig(x::Float64) = Int(reinterpret(Int64, x) >> 52)
sig(x::BigFloat) = Int(x.exp)
sig(x) = Int(x.ex)

function bench_op(f, xs, ys)
    acc = sig(f(xs[1], ys[1]))                  # warm-up (compile)
    GC.gc()
    g0 = Base.gc_num()
    best = Inf
    for _ in 1:5
        t0 = time_ns()
        @inbounds for i in eachindex(xs)
            acc += sig(f(xs[i], ys[i]))
        end
        best = min(best, (time_ns() - t0) / length(xs))
    end
    d = Base.GC_Diff(Base.gc_num(), g0)
    return best, Base.gc_alloc_count(d) / (5length(xs)), d.allocd / (5length(xs)), acc
end

# MPFR, the C library: results into preallocated BigFloats (the current precision), no allocation in the loop
const libmpfr = Base.MPFR.libmpfr
const RM = Base.MPFR.MPFRRoundingMode
mpfr_add!(z, x, y) = ccall((:mpfr_add, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, RM), z, x, y, Base.MPFR.MPFRRoundNearest)
mpfr_mul!(z, x, y) = ccall((:mpfr_mul, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, RM), z, x, y, Base.MPFR.MPFRRoundNearest)
mpfr_div!(z, x, y) = ccall((:mpfr_div, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, RM), z, x, y, Base.MPFR.MPFRRoundNearest)
mpfr_sqrt!(z, x, y) = ccall((:mpfr_sqrt, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, RM), z, x, Base.MPFR.MPFRRoundNearest)   # √|x|: x is |x| below
# f! is an argument, not looked up inside: a Function-typed local would make every call a dynamic
# dispatch (+15–20 ns per operation, measured — the size of the whole MPFR add at 113 bits)
function bench_mpfr_c(f!::F, xs::Vector{BigFloat}, ys::Vector{BigFloat}) where {F}
    xa = f! === mpfr_sqrt! ? abs.(xs) : xs      # the C sqrt takes the operand as is
    z = [BigFloat() for _ in 1:length(xs)]      # allocated once, outside the timing
    best = Inf
    for _ in 1:5
        t0 = time_ns()
        @inbounds for i in eachindex(xs)
            f!(z[i], xa[i], ys[i])
        end
        best = min(best, (time_ns() - t0) / length(xs))
    end
    best
end

conv(::Type{T}, x) where {T} = T(x.neg, x.ex, x.d, x.flag)      # same representation, different module

rng = MersenneTwister(7)
println("Julia $(VERSION), threads=$(Threads.nthreads()), $(Sys.cpu_info()[1].model), N=$N ops per measurement, minimum of 5 passes; ns/op (allocations/op, bytes/op)")
println()
for (TR, TN, P) in ((MultiF32Ref.F128, MultiF32.F128, 113), (MultiF32Ref.F256, MultiF32.F256, 237), (MultiF32Ref.F512, MultiF32.F512, 489))
    # operands: random significands and signs, exponents within ±8 so add/sub stay non-trivial (no
    # early-out on exponent gap; half the pairs have opposite signs, so the cancellation path runs too)
    rnd() = (v = MultiF32Ref.rand_val(TR, rng, rand(rng, -8:8)); rand(rng, Bool) ? TR(true, v.ex, v.d, v.flag) : v)
    xs = [rnd() for _ in 1:N]
    ys = [rnd() for _ in 1:N]
    xn = [conv(TN, x) for x in xs]; yn = [conv(TN, y) for y in ys]
    # MPFR operands at exactly P bits, allocated in one clean pass.  Two lessons from the first version
    # of this table: (1) BigFloat(x::MFloat) builds its value at 18K+16 bits, and MPFR with operands
    # wider than the result (one limb more, a different precision) runs 1.5–4× slower, so the value is
    # copied to P bits (exact: it has P bits); (2) a BigFloat is a heap object plus a separate limb
    # buffer, and operands allocated with temporaries in between measure 1.5–1.9× slower than the same
    # values allocated back to back (113-bit add: 27 vs 16 ns) — so the temporaries are built first and
    # the P-bit copies in their own pass.  That is MPFR's best case, the conservative one for this table.
    tx = [BigFloat(x) for x in xs]; ty = [BigFloat(y) for y in ys]
    bx = setprecision(BigFloat, P) do; [BigFloat(t) for t in tx]; end
    by = setprecision(BigFloat, P) do; [BigFloat(t) for t in ty]; end
    all(precision(b) == P for b in bx) || error("MPFR operands are not at P bits")
    all(BigFloat(xn[i]) == bx[i] for i in 1:N) || error("conversion is not exact")
    @printf("%s  (P=%d bits, K=%d digits)\n", TN, P, MultiF32.ndig(TN))
    setprecision(BigFloat, P) do
        for (name, f, f!) in (("add", +, mpfr_add!), ("mul", *, mpfr_mul!), ("div", /, mpfr_div!), ("sqrt", (a, b) -> sqrt(abs(a)), mpfr_sqrt!))
            tr, ar, br, _ = bench_op(f, xs, ys)
            tn, an, bn, _ = bench_op(f, xn, yn)
            tb, ab, _, _ = bench_op(f, bx, by)
            tc = bench_mpfr_c(f!, bx, by)
            @printf("  %-4s  Ref(Vector) %7.0f ns (%5.1f allocs, %6.0f B)   NTuple %7.0f ns (%3.1f allocs, %2.0f B)   MPFR BigFloat %5.0f ns (%3.1f allocs)   MPFR C %5.0f ns   NTuple/Ref %4.2fx   NTuple/MPFR-C %5.1fx\n",
                    name, tr, ar, br, tn, an, bn, tb, ab, tc, tn / tr, tn / tc)
        end
    end
    println()
end

# Float64 baseline (same loop harness, so harness overhead is visible)
xs = rand(rng, N) .+ 0.5; ys = rand(rng, N) .+ 0.5
for (name, f) in (("add", +), ("mul", *), ("div", /), ("sqrt", (a, b) -> sqrt(a)))
    ns, _, _, _ = bench_op(f, xs, ys)
    @printf("Float64 %-5s %6.2f ns/op (harness incl.)\n", name, ns)
end
