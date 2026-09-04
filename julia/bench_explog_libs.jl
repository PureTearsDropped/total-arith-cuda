# bench_explog_libs.jl — exp and log: MultiU32 against the existing multi-word libraries and MPFR,
# speed and accuracy on the same operands.   julia julia/bench_explog_libs.jl [N]
#   (needs MultiFloats, DoubleFloats, Quadmath — installed by you, not vendored)
#   Speed: ns per operation, CPU single thread, minimum of 5 passes; MPFR both as Julia's BigFloat
#          (allocating) and as the C library into a preallocated result (mpfr_exp / mpfr_log).
#   Accuracy: against MPFR at 600 bits, two domains — generic (|x| ∈ [2^-8, 2^8) for exp, exponents
#          ±8 for log) and the hard neighbourhoods (exp of |x| < 2^-4 … 2^-(P/2), log of 1 ± 2^-k) —
#          reported as the maximum and mean error in ulps of the library's own precision (correctly
#          rounded ⟹ max ≤ 0.5) and, for MultiU32, the error of the first Ziv stage against its bound.
using Random, Printf
include(joinpath(@__DIR__, "MultiU32.jl"))
using .MultiU32
const M = MultiU32
using MultiFloats, DoubleFloats, Quadmath

const NP = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 20_000
const NA = min(NP, 4000)                          # accuracy sample
rng = MersenneTwister(20260904)

function sources(n, elo, ehi; signs = true)
    setprecision(BigFloat, 600) do
        [((signs && rand(rng, Bool)) ? -1 : 1) * ldexp(BigFloat(1) + rand(rng, BigFloat), rand(rng, elo:ehi)) for _ in 1:n]
    end
end
srce = sources(NP, -8, 8)                         # exp: both signs
srcl = sources(NP, -8, 8; signs = false)          # log: positive
# hard neighbourhoods (per precision P): exp of tiny |x|, log of 1 ± tiny
function hard(P, n)
    setprecision(BigFloat, 600) do
        e = [((rand(rng, Bool)) ? -1 : 1) * ldexp(BigFloat(1) + rand(rng, BigFloat), -rand(rng, 4:P ÷ 2)) for _ in 1:n]
        l = [BigFloat(1) + ((rand(rng, Bool)) ? -1 : 1) * ldexp(BigFloat(1) + rand(rng, BigFloat), -rand(rng, 4:P ÷ 2)) for _ in 1:n]
        e, l
    end
end

conv(::Type{T}, v) where {T} = T(v)
conv(::Type{BigFloat}, v) = BigFloat(v)

function time_op(f, xs::Vector{T}) where {T}
    z = similar(xs); best = Inf
    for _ in 1:5
        t = @elapsed begin
            @inbounds for i in eachindex(xs); z[i] = f(xs[i]); end
        end
        best = min(best, t)
    end
    1e9 * best / length(xs)
end
const libmpfr = Base.MPFR.libmpfr
const RN = Base.MPFR.MPFRRoundNearest
mpfr_exp!(z, x) = ccall((:mpfr_exp, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, x, RN)
mpfr_log!(z, x) = ccall((:mpfr_log, libmpfr), Int32, (Ref{BigFloat}, Ref{BigFloat}, Base.MPFR.MPFRRoundingMode), z, x, RN)
function time_c(f!::F, xs::Vector{BigFloat}) where {F}
    z = [BigFloat() for _ in 1:length(xs)]; best = Inf
    for _ in 1:5
        t = @elapsed begin
            @inbounds for i in eachindex(xs); f!(z[i], xs[i]); end
        end
        best = min(best, t)
    end
    1e9 * best / length(xs)
end

"""error of r against the truth t in ulps of a P-bit format (ulp = 2^(e−P+1), e = exponent of t)."""
function ulps(r, t::BigFloat, P)
    iszero(t) && return iszero(r) ? 0.0 : Inf
    e = exponent(t)
    Float64(abs(BigFloat(r) - t) / ldexp(BigFloat(1), e - P + 1))
end

function accuracy(::Type{T}, P, xs_src, f, truth) where {T}
    xs = setprecision(BigFloat, P) do; [conv(T, v) for v in xs_src]; end
    errs = Float64[]
    setprecision(BigFloat, 600) do
        for i in eachindex(xs)
            t = truth(BigFloat(xs[i]))                     # the truth of the *converted* operand
            r = try; f(xs[i]); catch; NaN; end
            push!(errs, (r isa Number && isfinite(Float64(r))) ? ulps(r, t, P) : Inf)
        end
    end
    maximum(errs), sum(errs) / length(errs), count(e -> e > 0.5, errs)
end

println("Julia $(VERSION), $(Sys.cpu_info()[1].model), N=$NP ops per timing (minimum of 5 passes), $NA operands per accuracy sample")
println()
println("speed — ns per operation, CPU single thread:")
@printf("  %-30s %8s %8s\n", "", "exp", "log")
rows = Any[("MultiU32 F128 (113)", F128, 113), ("Quadmath Float128 (113)", Float128, 113),
           ("DoubleFloats Double64 (106)", Double64, 106), ("MultiFloats Float64x2 (106)", Float64x2, 106),
           ("MPFR BigFloat (113)", BigFloat, 113),
           ("MultiU32 F256 (237)", F256, 237), ("MultiFloats Float64x4 (212)", Float64x4, 212), ("MPFR BigFloat (237)", BigFloat, 237),
           ("MultiU32 F512 (489)", F512, 489), ("MPFR BigFloat (489)", BigFloat, 489)]
function speed_row(label, ::Type{T}, P) where {T}
    xe = setprecision(BigFloat, P) do; [conv(T, v) for v in srce]; end
    xl = setprecision(BigFloat, P) do; [conv(T, v) for v in srcl]; end
    te = try; setprecision(BigFloat, P) do; time_op(exp, xe); end; catch err; NaN; end
    tl = try; setprecision(BigFloat, P) do; time_op(log, xl); end; catch err; NaN; end
    @printf("  %-30s %8.0f %8.0f%s\n", label, te, tl, (isnan(te) || isnan(tl)) ? "   (not available in this library)" : "")
    if T === BigFloat
        be = setprecision(BigFloat, P) do; [BigFloat(v) for v in srce]; end
        bl = setprecision(BigFloat, P) do; [BigFloat(v) for v in srcl]; end
        tce, tcl = setprecision(BigFloat, P) do; time_c(mpfr_exp!, be), time_c(mpfr_log!, bl); end   # the results are allocated at P
        @printf("  %-30s %8.0f %8.0f\n", "MPFR C library ($P)", tce, tcl)
    end
end
for (label, T, P) in rows; speed_row(label, T, P); end

println()
println("accuracy — error against MPFR@600 in ulps of the library's own precision (max / mean / count > 0.5 ulp of $NA); correctly rounded ⟹ max ≤ 0.5:")
@printf("  %-30s %-26s %-26s %-26s %-26s\n", "", "exp generic", "exp tiny |x|", "log generic", "log near 1")
function accuracy_row(label, ::Type{T}, P) where {T}
    he, hl = hard(P, NA)
    cells = String[]
    for (src, f, truth) in ((srce[1:NA], exp, exp), (he, exp, exp), (srcl[1:NA], log, log), (hl, log, log))
        mx, mn, nb = try; accuracy(T, P, src, f, truth); catch err; (NaN, NaN, -1); end
        push!(cells, isnan(mx) ? "n/a" : @sprintf("%7.3f / %6.4f / %4d", mx, mn, nb))
    end
    @printf("  %-30s %-26s %-26s %-26s %-26s\n", label, cells...)
end
for (label, T, P) in rows; T === BigFloat || accuracy_row(label, T, P); end

println()
println("MultiU32: the first Ziv stage's actual error against its bound (ulps of the working type S; the stage decides when |tail − midpoint| > B):")
function ziv_row(label, ::Type{T}, P) where {T}
    S = M.stages(T)[1]; PS = M.prec(S)
    se, Ne, Be = M.exp_params(PS); sl, Nl, Bl = M.log_params(PS)
    he, hl = hard(P, NA ÷ 2)
    xe = setprecision(BigFloat, P) do; [T(v) for v in vcat(srce[1:NA ÷ 2], he)]; end
    xl = setprecision(BigFloat, P) do; [T(v) for v in vcat(srcl[1:NA ÷ 2], hl)]; end
    worst_e = 0.0; worst_l = 0.0
    setprecision(BigFloat, 4PS + 64) do
        for x in xe
            (M.iszero(x) || Int(x.ex) ≥ 24 || Int(x.ex) < -(P + 3)) && continue
            h, k, _ = M.exp_stage(T, x, S)
            t = exp(BigFloat(x))
            worst_e = max(worst_e, ulps(ldexp(BigFloat(h), k), t, PS))
        end
        for x in xl
            (x.ex == 0 && x.d == M.mind(T)) && continue
            v, _ = M.log_stage(T, x, S)
            worst_l = max(worst_l, ulps(v, log(BigFloat(x)), PS))
        end
    end
    @printf("  %-5s S = %d bits: exp worst %8.2f ulp vs bound %9d (s=%d, N=%d, margin 2^%.1f)   log worst %8.2f ulp vs bound %6d (s=%d, N=%d, margin 2^%.1f)\n",
            label, PS, worst_e, Be, se, Ne, log2(Be / max(worst_e, 1e-9)), worst_l, Bl, sl, Nl, log2(Bl / max(worst_l, 1e-9)))
end
for (label, T, P) in (("F128", F128, 113), ("F256", F256, 237), ("F512", F512, 489)); ziv_row(label, T, P); end
