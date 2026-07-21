# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
Transforms — WH・複素WH・FFT の Julia 実装 (nested_registry.py の MAPS/変換群の 双子)。

  すべて「有限アーベル群の 指標表」という 同じレシピの 製品:
    (ℤ/2)³   → ウォルシュ・アダマール (成分 ±1・純クロネッカー・捻れなし)
    ℤ/4×ℤ/2 → 複素WH / Chrestenson-4 (成分 {±1,±i}・×i=スワップ=乗算器ゼロ)
    ℤ/8      → DFT (8乗根・Cooley-Tukey 分解に twiddle=捻れの請求書 が 挟まる)

  実ランク階段(位数8): ℝ⁸(t=8)→8 / ℝ⁴⊕ℂ²(t=6)→10 / ℝ²⊕ℂ³(t=5)→11。
  FFT = 変換行列の 疎因数分解(バタフライ×log n 段) — ランク R と 直交する 第2のダイヤル。
  検証は 全部 self_test: 準同型 M(a⊛b)=M(a)·M(b)・因子積≡行列・高速適用≡密適用・
  整数入力の 全段厳密性(WH/複素WH は 誤差 文字通り 0)。
"""
module Transforms

import LinearAlgebra
import Random
include(joinpath(@__DIR__, "NestedSeries.jl")); using .NestedSeries
const NS = NestedSeries

export xor_alg, diag_alg, wh_matrix, cwh_matrix, dft_matrix,
       wh_factors, dft_factors, fwht!, fft_rec, hom_defect

# ================================================================ 群環 (セル追加)
"XOR群 (ℤ/2)^k の群環: e_i·e_j = e_{i⊻j} — 捻れなしCD・WHが対角化"
xor_alg(n::Int) = NS._from_mul("xor$n", n, Float64.(1:n .== 1),
    (x, y) -> begin
        r = zeros(n)
        for i in 0:n-1, j in 0:n-1
            r[(i ⊻ j) + 1] += x[i+1] * y[j+1]
        end
        r
    end)

"周波数代数(各点積): e_f·e_f = e_f・単位元=全1・零因子だらけ(死んだ帯域の正体)"
diag_alg(n::Int) = NS._from_mul("diag$n", n, ones(n),
    (x, y) -> x .* y)

# ================================================================ 変換行列
"WH: H[f,i] = (−1)^{popcount(f&i)} — ±1のみ"
wh_matrix(n::Int) = [(-1.0)^count_ones((f-1) & (i-1)) for f in 1:n, i in 1:n]

"複素WH (Chrestenson-4) = F₄⊗H₂ — 成分 {±1,±i}"
function cwh_matrix()
    F4 = [1 1 1 1; 1 -im -1 im; 1 -1 1 -1; 1 im -1 -im]
    kron(F4, Complex{Float64}[1 1; 1 -1])
end

"DFT: F[f,j] = ω^{fj}"
dft_matrix(n::Int) = [exp(-2im * π * (f-1) * (j-1) / n) for f in 1:n, j in 1:n]

# ================================================================ 疎因数分解 (FFT=これ)
"WHのバタフライ: H_{2^k} = Π (I⊗H₂⊗I) — 全因子±1・行あたり非ゼロ2・積は厳密"
function wh_factors(n::Int)
    H2 = [1.0 1.0; 1.0 -1.0]
    k = trailing_zeros(n)
    [kron(kron(Matrix(1.0LinearAlgebra.I, 2^s, 2^s), H2),
          Matrix(1.0LinearAlgebra.I, 2^(k-1-s), 2^(k-1-s))) for s in 0:k-1]
end

"DFTのCooley-Tukey全段: バタフライ + twiddle対角(捻れの請求書) + 並べ替え"
function dft_factors(n::Int)
    n == 2 && return [Complex{Float64}[1 1; 1 -1]]
    w = exp(-2im * π / n)
    h = n ÷ 2
    P = zeros(n, n)
    for k in 1:h
        P[k, 2k-1] = 1; P[k+h, 2k] = 1
    end
    D = LinearAlgebra.diagm(vcat(ones(Complex{Float64}, h), [w^k for k in 0:h-1]))
    F2 = Complex{Float64}[1 1; 1 -1]
    Ih = Matrix(1.0LinearAlgebra.I, h, h)
    vcat([kron(F2, Ih), D],
         [kron(Matrix(1.0LinearAlgebra.I, 2, 2), S) for S in dft_factors(h)],
         [Complex{Float64}.(P)])
end

# ================================================================ 高速適用 (本物のループ実装)
"in-place 高速WH変換: O(n log n) の 加減算のみ・乗算ゼロ"
function fwht!(v::AbstractVector)
    n = length(v); h = 1
    while h < n
        for base in 0:2h:n-1
            for i in 1:h
                a = v[base+i]; b = v[base+i+h]
                v[base+i] = a + b; v[base+i+h] = a - b
            end
        end
        h *= 2
    end
    v
end

"再帰 radix-2 FFT (Cooley-Tukey そのもの)"
function fft_rec(x::AbstractVector{<:Number})
    n = length(x)
    n == 1 && return Complex{Float64}[x[1]]
    E = fft_rec(x[1:2:end]); O = fft_rec(x[2:2:end])
    h = n ÷ 2
    tw = [exp(-2im * π * k / n) for k in 0:h-1] .* O
    vcat(E .+ tw, E .- tw)
end

"準同型のズレ: max|M(a⊛b) − M(a)·M(b)| (変換が畳み込みを各点積に移すか)"
function hom_defect(M, A::NS.Alg, rng)
    a = randn(rng, A.dim); b = randn(rng, A.dim)
    maximum(abs.(M * NS.rawmul(A, a, b) .- (M * a) .* (M * b)))
end

# ================================================================ self-test
function self_test()
    rng = Random.MersenneTwister(0)
    println("Transforms — WH・複素WH・FFT (全部「群の指標表」・検証は測って主張)")

    # ① WH = XOR群のDFT
    n = 8
    H = wh_matrix(n)
    X8 = xor_alg(n)
    @assert hom_defect(H, X8, rng) < 1e-12
    Hk = reduce(kron, fill([1.0 1; 1 -1], 3))
    @assert maximum(abs.(H .- Hk)) == 0.0                    # 純クロネッカー(捻れなし)
    Pf = reduce(*, wh_factors(n))
    @assert maximum(abs.(Pf .- H)) == 0.0                    # バタフライ因子積 ≡ H 厳密
    x = randn(rng, n)
    @assert maximum(abs.(fwht!(copy(x)) .- H * x)) < 1e-12   # ループ実装 ≡ 行列
    ai = Float64.(rand(rng, -100:100, n)); bi = Float64.(rand(rng, -100:100, n))
    conv = fwht!(fwht!(copy(ai)) .* fwht!(copy(bi))) ./ n
    @assert maximum(abs.(conv .- NS.rawmul(X8, ai, bi))) == 0.0   # 整数入力: 全段厳密
    println("  ① WH: 準同型✓ H=H₂⊗H₂⊗H₂厳密✓ 因子積≡H厳密✓ fwht!≡H·x✓ 整数畳み込み誤差0 ✓")

    # ② 複素WH = ℤ/4×ℤ/2 のDFT
    C = cwh_matrix()
    Z = tensor(cyclic_alg(4), cyclic_alg(2))
    @assert all(u -> abs(abs(real(u)) + abs(imag(u)) - 1) < 1e-12, C)  # 成分は±1,±iのみ
    @assert hom_defect(C, Z, rng) < 1e-12
    T_rec = [sum(C[f,i] * C[f,j] * conj(C[f,k]) / 8 for f in 1:8) for i in 1:8, j in 1:8, k in 1:8]
    Tz = [Z.tab[i][j][k] for i in 1:8, j in 1:8, k in 1:8]
    @assert maximum(abs.(T_rec .- Tz)) == 0.0                # ΣUVW ≡ T 厳密
    ci = Float64.(rand(rng, -100:100, 8)); di = Float64.(rand(rng, -100:100, 8))
    conv2 = real.(C' * ((C * ci) .* (C * di)) ./ 8)
    @assert maximum(abs.(conv2 .- NS.rawmul(Z, ci, di))) == 0.0
    println("  ② 複素WH: 成分{±1,±i}✓ 準同型✓ ΣUVW≡T厳密0✓ 整数畳み込み誤差0 ✓")

    # ③ DFT/FFT = ℤ/8 (捻れの世界)
    F = dft_matrix(8)
    Cy = cyclic_alg(8)
    @assert hom_defect(F, Cy, rng) < 1e-12
    Pd = reduce(*, dft_factors(8))
    @assert maximum(abs.(Pd .- F)) < 1e-13                   # Cooley-Tukey因子積 ≡ F
    y = randn(rng, 8)
    @assert maximum(abs.(fft_rec(y) .- F * y)) < 1e-13       # 再帰FFT ≡ 行列
    a3 = randn(rng, 8); b3 = randn(rng, 8)
    conv3 = real.(F' * (fft_rec(a3) .* fft_rec(b3)) ./ 8)
    @assert maximum(abs.(conv3 .- NS.rawmul(Cy, a3, b3))) < 1e-12
    println("  ③ DFT/FFT: 準同型✓ CT因子積≡F✓ 再帰FFT≡F·x✓ FFT畳み込み≡巡回積 ✓")
    println("  実ランク階段(位数8): (ℤ/2)³:8 / ℤ/4×ℤ/2:10 / ℤ/8:11 — 捻れ(繰り上がり)の値段")
    println("done — 3つの変換は同じレシピ(指標表)・違いは群の捻れだけ")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    Transforms.self_test()
end
