# MultiU32 (29-bit integer digits) vs MultiF32 (18-bit Float32 digits) vs MPFR, same precision,
# same operands — CPU, single thread.   julia julia/bench_multiu32.jl [N]
#
# Operands: random significands and signs, exponents within ±8 (no early-out on the exponent gap;
# half the pairs have opposite signs so the cancellation path of add runs).  The MultiF32 values are
# converted through BigFloat (exact: both hold P bits) so every column sees the same numbers.
# MPFR twice: through Julia's BigFloat (allocates every result, 2 allocations per operation, GC
# included) and the C library itself (mpfr_add / mul / div / sqrt into a preallocated result — what
# a C caller pays; the number "MPFR on one core" means this one).  The BigFloat column was the only
# MPFR column at first and overstated MPFR's cost by 1.5–4×.  Every time is the minimum of 5 passes.
include(joinpath(@__DIR__, "MultiF32.jl"))
include(joinpath(@__DIR__, "MultiU32.jl"))
using Random, Printf

const N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 20_000

# every result is consumed (its exponent field is summed) so that no operation can be dead-code eliminated
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
    return best, Base.gc_alloc_count(d) / (5length(xs)), acc
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

rng = MersenneTwister(7)
println("Julia $(VERSION), threads=$(Threads.nthreads()), $(Sys.cpu_info()[1].model), N=$N ops per measurement, minimum of 5 passes; ns/op (allocations/op)")
println()
for (TF, TU, P) in ((MultiF32.F128, MultiU32.F128, 113), (MultiF32.F256, MultiU32.F256, 237), (MultiF32.F512, MultiU32.F512, 489))
    rnd() = (v = MultiF32.rand_val(TF, rng, rand(rng, -8:8)); rand(rng, Bool) ? TF(true, v.ex, v.d, v.flag) : v)
    xf = [rnd() for _ in 1:N]; yf = [rnd() for _ in 1:N]
    # MPFR operands at exactly P bits, allocated in one clean pass.  Two lessons from the first version
    # of this table: (1) BigFloat(x::MFloat) builds its value at 29K+16 bits, and MPFR with operands
    # wider than the result (one limb more, a different precision) runs 1.5–4× slower, so the value is
    # copied to P bits (exact: it has P bits); (2) a BigFloat is a heap object plus a separate limb
    # buffer, and operands allocated with temporaries in between measure 1.5–1.9× slower than the same
    # values allocated back to back (113-bit add: 27 vs 16 ns) — so the temporaries are built first and
    # the P-bit copies in their own pass.  That is MPFR's best case, the conservative one for this table.
    tx = [BigFloat(x) for x in xf]; ty = [BigFloat(y) for y in yf]
    bx = setprecision(BigFloat, P) do; [BigFloat(t) for t in tx]; end
    by = setprecision(BigFloat, P) do; [BigFloat(t) for t in ty]; end
    all(precision(b) == P for b in bx) || error("MPFR operands are not at P bits")
    xu = [TU(x) for x in bx]; yu = [TU(y) for y in by]
    all(BigFloat(xu[i]) == bx[i] for i in 1:N) || error("conversion is not exact")
    @printf("P=%d bits:  MultiF32 %s (K=%d × 18-bit Float32, %d B)   MultiU32 %s (K=%d × 29-bit UInt32, %d B)\n",
            P, TF, MultiF32.ndig(TF), sizeof(TF), TU, MultiU32.ndig(TU), sizeof(TU))
    setprecision(BigFloat, P) do
        for (name, f, f!) in (("add", +, mpfr_add!), ("mul", *, mpfr_mul!), ("div", /, mpfr_div!), ("sqrt", (a, b) -> sqrt(abs(a)), mpfr_sqrt!))
            tf, af, _ = bench_op(f, xf, yf)
            tu, au, _ = bench_op(f, xu, yu)
            tb, ab, _ = bench_op(f, bx, by)
            tc = bench_mpfr_c(f!, bx, by)
            # the two engines must agree on every value (both are MPFR-verified; this is the cheap cross-check)
            nd = count(i -> BigFloat(f(xf[i], yf[i])) != BigFloat(f(xu[i], yu[i])), 1:min(N, 2000))
            @printf("  %-4s  MultiF32 %7.0f ns (%3.1f allocs)   MultiU32 %7.0f ns (%3.1f allocs)   MPFR BigFloat %5.0f ns (%3.1f allocs)   MPFR C %5.0f ns   U32/F32 %5.2fx   U32/MPFR-C %5.2fx   F32≠U32: %d\n",
                    name, tf, af, tu, au, tb, ab, tc, tu / tf, tu / tc, nd)
        end
    end
    println()
end
