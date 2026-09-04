# prof_divsqrt.jl — where the time of ÷ and √ actually goes.
#
# The Newton ladder cut the *candidate*; this splits what is left, so the next change is chosen by
# measurement and not by a cost model.  (The model said the ladder would be 2.2 full-width products
# ≈ 194 ns at F512 and it measured 904: at these widths a rounded operation is mostly its
# normalize + round_pack, not its digit products.  Hence this file.)
#
#   recip_sig / rsqrt_sig  the candidate (the fixed-point ladder)
#   mul_core               one full-width product, the unit everything else is quoted in
#   div_core / sqrt_core   the whole thing — the difference is the residual + fix-up + rounding path
#
# Every piece is timed inside a function with T as a *type parameter*: at global scope a closure
# over the loop variable T is a box, and the dynamic dispatch that follows adds ~90 ns to every
# call — more than the pieces being compared.
#
#   julia julia/prof_divsqrt.jl [N]
include(joinpath(@__DIR__, "MultiU32.jl"))
using .MultiU32
using Random, Printf
const M = MultiU32

const N = length(ARGS) ≥ 1 ? parse(Int, ARGS[1]) : 20_000

@inline sig(x::M.MFloat) = Int(x.ex)
@inline sig(x::Tuple{<:M.MFloat,Int}) = Int(x[1].ex)
@inline sig(x::NTuple{K,UInt32}) where {K} = Int(x[1])
@inline sig(x::Int) = x

function best(f::F, xs, ys) where {F}
    acc = sig(f(xs[1], ys[1]))
    b = Inf
    for _ in 1:5
        t0 = time_ns()
        @inbounds for i in eachindex(xs)
            acc += sig(f(xs[i], ys[i]))
        end
        b = min(b, (time_ns() - t0) / length(xs))
    end
    b, acc
end

function pieces(::Type{T}, rng, n) where {T<:M.MFloat}
    rnd() = (v = M.rand_val(T, rng, rand(rng, -8:8)); rand(rng, Bool) ? T(true, v.ex, v.d, v.flag) : v)
    xs = [rnd() for _ in 1:n]; ys = [rnd() for _ in 1:n]
    sa = [T(false, 0, x.d, 0x00) for x in xs]                 # significands, ex = 0
    sb = [T(false, 0, y.d, 0x00) for y in ys]
    # the residual path, with the candidate a real division produces
    K, P = M.ndig(T), M.prec(T)
    Q = Vector{NTuple{K + 1,UInt32}}(undef, n)
    for i in 1:n
        qh = M.mul_core(T, sa[i], M.recip_sig(T, sb[i]), M.fullval(T))
        Q[i], _ = M.place(M.shl_bits(qh.d, Int(qh.ex) + 1), P - 29K, Val(K + 1))
    end
    dR = [M.div_resid(xs[i].d, Q[i], ys[i].d, Val(P))[2] for i in 1:n]
    eR = 1 - 29K - P
    (("mul_core (full product)", best((a, b) -> M.mul_core(T, a, b, M.fullval(T)), sa, sb)[1]),
     ("add_core (full width)",   best((a, b) -> M.add_core(T, a, b, false, M.fullval(T)), sa, sb)[1]),
     ("recip_sig (ladder)",      best((a, b) -> M.recip_sig(T, b), sa, sb)[1]),
     ("div_core (all)",          best((a, b) -> M.div_core(T, a, b), sa, sb)[1]),
     ("rsqrt_sig (ladder)",      best((a, b) -> M.rsqrt_sig(T, b), sa, sb)[1]),
     ("sqrt_core (all)",         best((a, b) -> M.sqrt_core(T, a), sa, sb)[1]),
     ("  div_resid (prod + sub)", best((i, j) -> M.div_resid(xs[i].d, Q[i], ys[i].d, Val(P))[2], 1:n, 1:n)[1]),
     ("  wide_cmp (2K+2 window)", best((i, j) -> M.wide_cmp(dR[i], eR, ys[i].d, eR, Val(2K + 2)), 1:n, 1:n)[1]),
     ("  round_pack (K+1 dig)",   best((i, j) -> M.round_pack(T, false, Q[i], -P, false, M.precval(T)), 1:n, 1:n)[1]))
end

rng = MersenneTwister(11)
println("Julia $(VERSION), $(Sys.cpu_info()[1].model), N=$N, min of 5 passes; ns/op")
res = [pieces(T, rng, N) for T in (M.F128, M.F256, M.F512)]
println("piece                              F128            F256            F512      (× one full product)")
for k in eachindex(res[1])
    @printf("%-30s", res[1][k][1])
    for c in 1:3
        @printf(" %7.0f (%4.1f×)", res[c][k][2], res[c][k][2] / res[c][1][2])
    end
    println()
end
for (c, T) in enumerate((M.F128, M.F256, M.F512))
    d = res[c]
    @printf("%s: ladder %.0f%% of ÷ (residual+rounding %.0f ns), ladder %.0f%% of √ (residual+rounding %.0f ns)\n",
            T, 100 * d[3][2] / d[4][2], d[4][2] - d[3][2], 100 * d[5][2] / d[6][2], d[6][2] - d[5][2])
end
