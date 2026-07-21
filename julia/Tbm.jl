# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
Tbm — 総ビリニア機械 v0.1 アセンブラの Julia 双子 (tbm.py と 同じ 6 命令・2 ダイヤル)。

  薄い層の 規律 (tbm.py と 同一): 意味論は ここに 住まない —
  TOTALIZE/BILIN(evidence)/AXPY は 監査済み HyperAlgebra (cuda_total の 双子) へ、
  LINMAP は Transforms へ、CHECK は NestedSeries の 反証子へ 委譲する。
  新規の 意味論は coarse (粗誠実) と 貯め幅 width のみ — tbm.py §3 と 同じ 定義。

  バックエンド: Julia CPU のみ。NORM は 未対応 (適合表の 空欄は 空欄のまま —
  golden は total-arith-hardware の Python 実装が 唯一の 正)。
  言語間 適合: tbm_cross.jl が run_everywhere.py の 吐く ベクタで
  torch 実装と 値 bit一致・フラグ bit一致を 照合する。
"""
module Tbm

import Random
include(joinpath(@__DIR__, "HyperAlgebra.jl")); using .HyperAlgebra
include(joinpath(@__DIR__, "Transforms.jl"))
const HA = HyperAlgebra
const TR = Transforms
const NS = TR.NestedSeries

export Program, TOTALIZE!, BILIN!, LINMAP!, AXPY!, CHECK!, run_program,
       coarse_group_mul, bare_group_mul, LAWS

# ================================================================ coarse / bare (SPEC §3)
function _raw_bilin(T, a::HA.Tot, b::HA.Tot, width::Symbol)
    dt = width === :f64 ? Float64 : Float32
    M = size(T, 1); n = size(a.val, 1)
    Td = dt.(T); av = dt.(a.val); bv = dt.(b.val)
    raw = zeros(dt, n, M)
    @inbounds for k in 1:M, i in 1:M, j in 1:M
        t = Td[k, i, j]
        t == 0 && continue
        @simd for r in 1:n
            raw[r, k] += t * av[r, i] * bv[r, j]
        end
    end
    HA._sat(Float64.(raw))
end

"粗誠実 BILIN: 値経路は evidence と 同一 (width=:f64 で bit一致)。入力に 札 → その行の 全成分に GE|LE|SUNK。"
function coarse_group_mul(T, a::HA.Tot, b::HA.Tot; width::Symbol = :f64)
    val, sflag = _raw_bilin(T, a, b, width)
    dirty = vec(maximum(a.flag .| b.flag, dims = 2) .> 0)
    fin = UInt8.(dirty) .* UInt8(HA.GE | HA.LE | HA.SUNK)
    HA.Tot(val, sflag .| fin)
end

"値のみ (NaN 非生成は 維持)。両端を evidence が 守る 区間の 内側 専用。"
function bare_group_mul(T, a::HA.Tot, b::HA.Tot; width::Symbol = :f64)
    val, sflag = _raw_bilin(T, a, b, width)
    HA.Tot(val, zero.(sflag))
end

# ================================================================ LAWS (SPEC §6 — Julia 部分集合)
const LAWS = Dict{Symbol,Function}(
    :powerassoc => (alg; seed = 1) -> NS.powerassoc_defect(alg; rng = NS._LCG(UInt64(seed))),
    :assoc      => (alg; seed = 1) -> NS.assoc_defect(alg; rng = NS._LCG(UInt64(seed))),
    :homomorphy => (M, alg; seed = 0) -> TR.hom_defect(M, alg, Random.MersenneTwister(seed)),
)   # rank_exact は IMPLS 棚が Julia に 未移植のため 空欄 (偽装しない)

# ================================================================ プログラム (6 命令)
struct Program
    name::String
    ins::Vector{Tuple{Symbol,Dict{Symbol,Any}}}
end
Program(name::String = "tbm") = Program(name, Tuple{Symbol,Dict{Symbol,Any}}[])

TOTALIZE!(P, dst, src) = (push!(P.ins, (:TOTALIZE, Dict(:dst => dst, :src => src))); P)
function BILIN!(P, dst, a, b; alg = :sedenion, honesty = :evidence, width = :f64)
    @assert honesty in (:evidence, :coarse, :bare)
    @assert width === :f64 || honesty !== :evidence "evidence は f64 固定 (bit一致 契約)"
    push!(P.ins, (:BILIN, Dict(:dst => dst, :a => a, :b => b, :alg => alg,
                               :honesty => honesty, :width => width))); P
end
LINMAP!(P, dst, src; map = :wh8, honesty = :bare) =
    (push!(P.ins, (:LINMAP, Dict(:dst => dst, :src => src, :map => map,
                                 :honesty => honesty))); P)
AXPY!(P, dst, src; c = 1.0) =
    (push!(P.ins, (:AXPY, Dict(:dst => dst, :src => src, :c => Float64(c)))); P)
CHECK!(P, dst, law; args...) =
    (push!(P.ins, (:CHECK, Dict(:dst => dst, :law => law, :args => args))); P)

const _ALG_M = Dict(:sedenion => 16, :octonion => 8, :quaternion => 4, :complex => 2)
_wiring(alg::Symbol) = HA.wiring_tensor(:cd, _ALG_M[alg])
_maps(m::Symbol) = m === :wh8 ? TR.wh_matrix(8) : error("unknown map $m")

"実行 (Julia CPU バックエンド)。feed: Dict{Symbol,Matrix}。"
function run_program(P::Program, feed::Dict)
    env = Dict{Symbol,Any}()
    for (op, p) in P.ins
        if op === :TOTALIZE
            env[p[:dst]] = HA.Tot(Float64.(feed[p[:src]]))
        elseif op === :BILIN
            T = _wiring(p[:alg]); a = env[p[:a]]; b = env[p[:b]]
            env[p[:dst]] = p[:honesty] === :evidence ? HA.group_mul(T, a, b) :
                p[:honesty] === :coarse ? coarse_group_mul(T, a, b; width = p[:width]) :
                bare_group_mul(T, a, b; width = p[:width])
        elseif op === :LINMAP
            M = _maps(p[:map]); x = env[p[:src]]
            val, sflag = HA._sat(Float64.(x.val) * M')
            flag = p[:honesty] === :bare ? zero.(sflag) :
                sflag .| (UInt8.(vec(maximum(x.flag, dims = 2) .> 0)) .*
                          UInt8(HA.GE | HA.LE | HA.SUNK))
            env[p[:dst]] = HA.Tot(val, flag)
        elseif op === :AXPY
            x = env[p[:src]]
            if p[:c] != 1.0
                sc = HA.Tot(Float64.(x.val) .* p[:c])
                x = HA.Tot(sc.val, sc.flag .| x.flag)          # 札は 保守的に 通す
            end
            env[p[:dst]] = HA.tot_add(env[p[:dst]], x)
        elseif op === :CHECK
            env[p[:dst]] = Float64(LAWS[p[:law]](values(p[:args])...))
        end
    end
    env
end

# ================================================================ self-test (tbm.py の 鏡)
function self_test()
    println("Tbm.jl — アセンブラ self-test (意味論の 正: HyperAlgebra / NestedSeries / Transforms)")
    rng = Random.MersenneTwister(0)
    T = _wiring(:sedenion)

    println("① coarse の 契約: 値 ≡ evidence 値 / 汚れ1行 → 全札・他行 0")
    a = HA.Tot(Float64.(rand(rng, -9:9, 64, 16)))
    b = HA.Tot(Float64.(rand(rng, -9:9, 64, 16)))
    ev = HA.group_mul(T, a, b)
    co = coarse_group_mul(T, a, b)
    @assert ev.val == co.val
    @assert maximum(co.flag) == 0
    f = zeros(UInt8, 64, 16); f[4, 8] = HA.GE
    cod = coarse_group_mul(T, HA.Tot(a.val, f), b)
    @assert all(cod.flag[4, :] .== UInt8(HA.GE | HA.LE | HA.SUNK))
    @assert maximum(cod.flag[[1:3; 5:64], :]) == 0
    println("   値 bit一致 ✓ / 汚れ行の 全成分 GE|LE|SUNK・漏れなし ✓")

    println("①b 貯め幅ダイヤル: f32 蓄積 (evidence は f64 固定)")
    ar = HA.Tot(randn(rng, 4096, 16)); br = HA.Tot(randn(rng, 4096, 16))
    e64 = HA.group_mul(T, ar, br)
    c32 = coarse_group_mul(T, ar, br; width = :f32)
    rel = maximum(abs.(Float64.(c32.val) .- Float64.(e64.val))) / maximum(abs.(e64.val))
    @assert 0 < rel < 1e-5
    ok = try BILIN!(Program(), :s, :a, :b; honesty = :evidence, width = :f32); false
         catch; true end
    @assert ok
    println("   coarse(f32) vs f64: 相対差 $(round(rel, sigdigits=2)) ✓ / evidence×f32 拒否 ✓")

    println("② LAWS: 反証子 (Julia 部分集合 — rank_exact は IMPLS 未移植で 空欄)")
    r2 = LAWS[:powerassoc](NS.cd_alg(8))
    r3 = LAWS[:assoc](NS.cd_alg(8))
    r4 = LAWS[:homomorphy](TR.wh_matrix(8), TR.xor_alg(8))
    @assert r2 < 1e-12 && r4 < 1e-12 && r3 > 1e-3
    println("   powerassoc(oct)=$(round(r2, sigdigits=2)) homomorphy(wh8)=$(round(r4, sigdigits=2))" *
            " / assoc(oct)=$(round(r3, sigdigits=2)) (破れ 検出) ✓")

    println("③ プログラム: t=a·b; t+=c ≡ NestedSeries.rawmul 参照 (整数 厳密)")
    A16 = NS.cd_alg(16)
    fa = Float64.(rand(rng, -9:9, 8, 16)); fb = Float64.(rand(rng, -9:9, 8, 16))
    fc = Float64.(rand(rng, -9:9, 8, 16))
    P = Program("mac")
    TOTALIZE!(P, :a, :in_a); TOTALIZE!(P, :b, :in_b); TOTALIZE!(P, :c, :in_c)
    BILIN!(P, :s, :a, :b); AXPY!(P, :s, :c)
    env = run_program(P, Dict(:in_a => fa, :in_b => fb, :in_c => fc))
    ref = vcat([reshape(NS.rawmul(A16, fa[i, :], fb[i, :]) .+ fc[i, :], 1, 16) for i in 1:8]...)
    @assert Float64.(env[:s].val) == ref
    println("   一致 ✓")

    println("④ EXP マクロ (BILIN×12 + AXPY×12): 独立参照 (rawmul Float64 手回し) と 一致")
    x4 = 0.3 .* randn(rng, 4, 4)
    A4 = NS.cd_alg(4)
    Pe = Program("exp")
    TOTALIZE!(Pe, :x, :in_x); TOTALIZE!(Pe, :acc, :unit); TOTALIZE!(Pe, :term, :unit)
    for k in 1:12
        BILIN!(Pe, :term, :term, :x; alg = :quaternion, honesty = :coarse)
        AXPY!(Pe, :acc, :term; c = 1.0 / factorial(k))
    end
    unit = repeat([1.0 0 0 0], 4)
    enve = run_program(Pe, Dict(:in_x => x4, :unit => unit))
    ref4 = zeros(4, 4)
    for i in 1:4
        acc = [1.0, 0, 0, 0]; term = [1.0, 0, 0, 0]
        for k in 1:12
            term = NS.rawmul(A4, term, x4[i, :])
            acc = acc .+ term ./ factorial(k)
        end
        ref4[i, :] = acc
    end
    d = maximum(abs.(Float64.(enve[:acc].val) .- ref4))
    @assert d < 1e-6
    println("   最大差 $(round(d, sigdigits=2)) ✓")
    println("done — 薄い層は 薄いまま (Julia でも)")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Tbm.self_test()
end
