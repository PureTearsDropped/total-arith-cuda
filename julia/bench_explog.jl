# MultiU32 exp / log vs MPFR at the same precision — CPU, single thread.   julia julia/bench_explog.jl [N]
#
# Operands: exp on x with |x| ∈ [2^-8, 2^8) (no shortcut, no saturation), log on x with exponents
# within ±8; random significands, both signs for exp.  MPFR twice as in bench_multiu32.jl: through
# Julia's BigFloat (allocating) and the C library into a preallocated result (mpfr_exp / mpfr_log —
# "MPFR on one core").  The operands are copied to exactly P bits.  Minimum of 5 passes.
include(joinpath(@__DIR__, "MultiU32.jl"))
using Random, Printf

const N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 5_000
sig(x::BigFloat) = Int(x.exp)
sig(x) = Int(x.ex)
function bench_op(f, xs)
    acc = sig(f(xs[1]))
    GC.gc(); g0 = Base.gc_num(); best = Inf
    for _ in 1:5
        t0 = time_ns()
        @inbounds for i in eachindex(xs); acc += sig(f(xs[i])); end
        best = min(best, (time_ns() - t0) / length(xs))
    end
    d = Base.GC_Diff(Base.gc_num(), g0)
    return best, Base.gc_alloc_count(d) / (5length(xs)), acc
end
const libmpfr = Base.MPFR.libmpfr
const RM = Base.MPFR.MPFRRoundingMode
mpfr_exp!(z, x) = ccall((:mpfr_exp, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, RM), z, x, Base.MPFR.MPFRRoundNearest)
mpfr_log!(z, x) = ccall((:mpfr_log, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, RM), z, x, Base.MPFR.MPFRRoundNearest)
function bench_mpfr_c(f!::F, xs::Vector{BigFloat}) where {F}
    z = [BigFloat() for _ in 1:length(xs)]
    best = Inf
    for _ in 1:5
        t0 = time_ns()
        @inbounds for i in eachindex(xs); f!(z[i], xs[i]); end
        best = min(best, (time_ns() - t0) / length(xs))
    end
    best
end

rng = MersenneTwister(7)
println("Julia $(VERSION), $(Sys.cpu_info()[1].model), N=$N ops per measurement, minimum of 5 passes; ns/op (allocations/op)")
for (TU, P) in ((MultiU32.F128, 113), (MultiU32.F256, 237), (MultiU32.F512, 489))
    xe = [(v = MultiU32.rand_val(TU, rng, rand(rng, -8:8)); rand(rng, Bool) ? -v : v) for _ in 1:N]
    xl = [MultiU32.rand_val(TU, rng, rand(rng, -8:8)) for _ in 1:N]
    be = setprecision(BigFloat, P) do; [BigFloat(BigFloat(x)) for x in xe]; end
    bl = setprecision(BigFloat, P) do; [BigFloat(BigFloat(x)) for x in xl]; end
    all(precision(b) == P for b in be) || error("MPFR operands are not at P bits")
    @printf("P=%d bits  %s (K=%d × 29-bit UInt32)\n", P, TU, MultiU32.ndig(TU))
    setprecision(BigFloat, P) do
        for (name, f, f!, xu, xb) in (("exp", exp, mpfr_exp!, xe, be), ("log", log, mpfr_log!, xl, bl))
            tu, au, _ = bench_op(f, xu)
            tb, ab, _ = bench_op(f, xb)
            tc = bench_mpfr_c(f!, xb)
            nd = count(i -> BigFloat(f(xu[i])) != f(xb[i]), 1:min(N, 2000))       # bit-identity with MPFR on the way
            @printf("  %-4s  MultiU32 %7.0f ns (%3.1f allocs)   MPFR BigFloat %6.0f ns (%3.1f allocs)   MPFR C %6.0f ns   U32/MPFR-C %5.2fx   U32≠MPFR: %d\n",
                    name, tu, au, tb, ab, tc, tu / tc, nd)
        end
    end
end
