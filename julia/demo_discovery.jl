# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
# demo_discovery.jl — 法則発見パイプライン、全部この棚から:
#   微分     = Λ1⊗Λ1 (超双対数: ε₁ε₂ が f'' を運ぶ) — 有限差分なしの厳密微分(誤差 0.0)
#   全域化   = ScalarTot の監査済み入口 — 毒(Inf/NaN)をフラグで名指し(拡散なし)
#   発見     = 単項式+微分ライブラリの零空間 SVD (stdlib)
# 実測: 調和振動子 ψ=e^{-x²/2} + 毒10点 → 10行だけ除外 → σ_min 8e-16 (ギャップ10¹⁵) →
#        ψ'' + ψ − x²ψ = 0 を発見・E = 0.5 を15桁で読取。
# 教訓: フラグは「表現違反」の検出器(Float64 の ScalarTot は 1e39 を通す=正当な値。
#        捕まえたのは Inf/NaN)。範囲内の統計的外れ値はロバスト統計の管轄。
include(joinpath(@__DIR__, "NestedSeries.jl"))
using .NestedSeries
const NS = NestedSeries
import LinearAlgebra
const LA = LinearAlgebra
import Random

# ═══ 微分は棚から: Λ1⊗Λ1 = 超双対数 [1, ε₁, ε₂, ε₁ε₂] — ε₁ε₂ が f'' を運ぶ
HD = tensor(grassmann_alg(1), grassmann_alg(1))
hd(x) = nel(HD, [x, 1.0, 1.0, 0.0])                    # x + ε₁ + ε₂

"exp on hyperdual: スカラー分離 exp(s+n)=eˢ·(1+n+n²/2) — 冪零で厳密に切れる"
function hd_exp(z)
    s = NS.coeffs(z)[1]
    n = NS.Nel([0.0; NS.coeffs(z)[2:end]], NS.flagof(z))
    en = NS.tadd(nel(HD), NS.tadd(n, NS.tscale(NS.tmul(HD, n, n), 0.5)))
    NS.tscale(en, exp(s))
end

println("① 微分の厳密性検査: ψ=e^{-x²/2} の ψ'' を 棚の超双対数で")
x0 = 1.3
z = hd(x0)
psi = hd_exp(NS.tscale(NS.tmul(HD, z, z), -0.5))       # exp(−x²/2) を代数ごと評価
c = NS.coeffs(psi)
tru = exp(-x0^2/2)
println("  ψ=", c[1], " (真値 ", tru, ")")
println("  ψ'  誤差 ", abs(c[2] - (-x0*tru)), "   ψ'' 誤差 ", abs(c[4] - (x0^2-1)*tru),
        "  ← 有限差分なしの厳密微分")

println()
println("② 発見パイプライン(全部棚から): 調和振動子 + 毒10点")
rng = Random.MersenneTwister(0)
xs = collect(range(-4, 4, length=4000))
n = length(xs)
U = zeros(n); U1 = zeros(n); U2 = zeros(n)
for (i, x) in enumerate(xs)
    p = hd_exp(NS.tscale(NS.tmul(HD, hd(x), hd(x)), -0.5))
    cc = NS.coeffs(p)
    U[i], U1[i], U2[i] = cc[1], cc[2], cc[4]
end
pois = Random.shuffle(rng, 1:n)[1:10]
U[pois[1:5]] .= Inf                                     # センサ飽和(ADCレール)
U[pois[6:10]] .= NaN                                    # 欠測
# ★毒の後の微分…ではなく: 超双対evalは点ごと独立なので 毒は その行だけに留まる
#   (有限差分なら隣2行も死ぬ)。ライブラリを ScalarTot(監査済み入口)に通して フラグで名指し:
include(joinpath(@__DIR__, "ScalarTot.jl")); using .ScalarTot
lib = hcat(U2, U, xs .* U, xs.^2 .* U)
flagged = falses(n)
for i in 1:n, j in 1:4
    flagged[i] |= ScalarTot.isflagged(TotNum(lib[i, j]))
end
println("  フラグが名指しした汚染行: ", count(flagged), " / ", n,
        "  (注入10点 → 10行のまま・差分拡散なし)")
clean = lib[.!flagged, :]
norms = [LA.norm(c_) for c_ in eachcol(clean)]
Un_, S_, Vt = LA.svd(clean ./ norms')
cvec = Vt[:, end] ./ norms
cvec = cvec ./ cvec[1]
println("  σ_min = ", S_[end], " / σ_next = ", S_[end-1])
println("  発見: 0 = +", round(cvec[1], digits=10), "·ψ''  +", round(cvec[2], digits=10),
        "·ψ  ", round(cvec[3], digits=10), "·xψ  ", round(cvec[4], digits=10), "·x²ψ")
println("  ★E = ", cvec[2]/2, "   (Python有限差分版: σ_min 2.6e-13・毒拡散30行)")
