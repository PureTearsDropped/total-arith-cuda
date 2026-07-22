# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
generic_mlp — 数の型に 総称な MLP: **同じコード**が Float64 でも ComplexF64 でも 走る。

  問い (ユーザ): 実数 MLP を そのまま 複素数に 変えたら 何が 起きるか。
  予想 (meta 法則): 型を 変えるだけでは 何も 買えない — 複素 MLP は 重み共有 制約つき
  実 MLP に すぎない。効くとしたら **複素の 掛け算を 課題と 噛み合う 場所に 置いたとき**
  (ℤ/16 の 指標は 複素 16 乗根 — 要素積 x⊙y が 群法則 そのもの)。

  課題: ℤ/16 合成 (a,b → a+b)。腕 = {実, 複素} × {MLP 合成器, ⊙ 合成器}。
  実装の 規律: 手書き backprop (Wirtinger: 勾配 g = 2·∂L/∂conj、実型では conj が
  恒等なので **同一コード**) + 有限差分 検査 (Re/Im 両方向・1e-6)。8 シード・最悪も 報告。
"""
module GenericMLP

using Random, LinearAlgebra

const M = 16

# ================================================================ パラメタ (型 T に 総称)
mutable struct P{T}
    E::Matrix{T}                    # d×16 埋め込み (列 = トークン)
    W1::Matrix{T}; b1::Vector{T}
    W2::Matrix{T}; b2::Vector{T}
end

function init(::Type{T}, d, w, rng) where {T}
    s = T <: Complex ? 0.3 / sqrt(2) : 0.3
    P{T}(s .* randn(rng, T, d, M),
         s .* randn(rng, T, w, 2d), zeros(T, w),
         s .* randn(rng, T, d, w), zeros(T, d))
end

params(p) = (p.E, p.W1, p.b1, p.W2, p.b2)

# ================================================================ 順 + 逆 (同一コードで 実/複素)
"forward: logits[c,n] = Re(E[:,c]' h_n)。mode=:mlp は tanh MLP・:mul は 要素積 x⊙y。"
function forward(p::P{T}, ia, ib, mode) where {T}
    X = p.E[:, ia]; Y = p.E[:, ib]
    if mode === :mlp
        Z1 = p.W1 * vcat(X, Y) .+ p.b1
        A1 = tanh.(Z1)
        H = p.W2 * A1 .+ p.b2
        cache = (X, Y, Z1, A1, H)
    else
        H = X .* Y
        cache = (X, Y, nothing, nothing, H)
    end
    logits = real.(p.E' * H)                     # E' = 共役転置 (実では ただの 転置)
    logits, H, cache
end

function backward(p::P{T}, ia, ib, ic, mode) where {T}
    n = length(ia)
    logits, H, (X, Y, Z1, A1, _) = forward(p, ia, ib, mode)
    mx = maximum(logits, dims = 1)
    pr = exp.(logits .- mx); pr ./= sum(pr, dims = 1)
    L = -sum(log.(max.(pr[CartesianIndex.(ic, 1:n)], 1e-300))) / n
    Δ = pr; for (j, c) in enumerate(ic); Δ[c, j] -= 1; end
    Δ ./= n                                      # ∂L/∂logits (実)
    gE = H * Δ'                                  # 読み出し側: g_E[:,c] += Σ_n Δ[c,n]·h_n
    gH = p.E * Δ
    if mode === :mlp
        gA1 = p.W2' * gH
        gW2 = gH * A1'; gb2 = vec(sum(gH, dims = 2))
        gZ1 = conj.(1 .- A1 .^ 2) .* gA1         # 正則 tanh の Wirtinger 鎖則
        gW1 = gZ1 * vcat(X, Y)'; gb1 = vec(sum(gZ1, dims = 2))
        gXY = p.W1' * gZ1
        gX = gXY[1:size(X, 1), :]; gY = gXY[size(X, 1)+1:end, :]
    else
        gW1 = zeros(T, size(p.W1)); gb1 = zeros(T, size(p.b1))
        gW2 = zeros(T, size(p.W2)); gb2 = zeros(T, size(p.b2))
        gX = conj.(Y) .* gH; gY = conj.(X) .* gH
    end
    for (j, a) in enumerate(ia); gE[:, a] .+= gX[:, j]; end
    for (j, b) in enumerate(ib); gE[:, b] .+= gY[:, j]; end
    L, (gE, gW1, gb1, gW2, gb2)
end

# ================================================================ 有限差分 検査 (掟)
function fd_check(::Type{T}, mode, rng) where {T}
    p = init(T, 4, 8, rng)
    ia = rand(rng, 1:M, 5); ib = rand(rng, 1:M, 5); ic = mod.(ia .+ ib .- 2, M) .+ 1
    _, gs = backward(p, ia, ib, ic, mode)
    worst = 0.0
    for (arr, g) in zip(params(p), gs)
        for k in eachindex(arr)[1:min(6, length(arr))]
            for dir in (T <: Complex ? (1.0, im) : (1.0,))
                h = 1e-6 * dir
                arr[k] += h; Lp = backward(p, ia, ib, ic, mode)[1]
                arr[k] -= 2h; Lm = backward(p, ia, ib, ic, mode)[1]
                arr[k] += h
                fd = (Lp - Lm) / 2e-6
                an = dir == 1.0 ? real(g[k]) : imag(g[k])
                worst = max(worst, abs(fd - an))
            end
        end
    end
    worst
end

# ================================================================ 学習 (手書き Adam・複素も 同一コード)
function train!(p::P{T}, ia, ib, ic, mode, steps; lr = 3e-3) where {T}
    ms = [zeros(T, size(a)) for a in params(p)]
    vs = [zeros(Float64, size(a)) for a in params(p)]
    for t in 1:steps
        _, gs = backward(p, ia, ib, ic, mode)
        for (a, g, m, v) in zip(params(p), gs, ms, vs)
            @. m = 0.9m + 0.1g
            @. v = 0.999v + 0.001 * abs2(g)
            @. a -= lr * (m / 0.9) / (sqrt(v / 0.999) + 1e-8)
        end
    end
end

function run_arm(::Type{T}, mode, frac, seed) where {T}
    rng = MersenneTwister(seed)
    pairs = [(a, b) for a in 1:M for b in 1:M]
    shuffle!(rng, pairs)
    ntr = round(Int, frac * length(pairs))
    tr = pairs[1:ntr]; te = pairs[ntr+1:end]
    ia = [x[1] for x in tr]; ib = [x[2] for x in tr]
    ic = mod.(ia .+ ib .- 2, M) .+ 1
    d, w = T <: Complex ? (8, 32) : (16, 64)     # 実自由度を ほぼ 揃える
    p = init(T, d, w, rng)
    train!(p, ia, ib, ic, mode, 4000)
    ja = [x[1] for x in te]; jb = [x[2] for x in te]
    jc = mod.(ja .+ jb .- 2, M) .+ 1
    lg, _, _ = forward(p, ja, jb, mode)
    mean([argmax(lg[:, j]) == jc[j] for j in eachindex(jc)])
end
mean(v) = sum(v) / length(v)

# ================================================================ self-test = 実験
function self_test()
    println("generic_mlp — 同じコードを Float64 ⇄ ComplexF64 で (手書き backprop + FD 検査)")
    rng = MersenneTwister(0)
    for T in (Float64, ComplexF64), mode in (:mlp, :mul)
        w = fd_check(T, mode, rng)
        println("  FD 検査 $(T) $(mode): 最大差 $(round(w, sigdigits=2))")
        @assert w < 1e-5
    end
    println("  腕: {実, 複素} × {MLP 合成器, ⊙ 合成器} / ℤ/16 合成・中央値/最悪・8 シード")
    println("  frac   実MLP           複素MLP         実⊙             複素⊙")
    for frac in (0.3, 0.5)
        cells = String[]
        for (T, mode) in ((Float64, :mlp), (ComplexF64, :mlp),
                          (Float64, :mul), (ComplexF64, :mul))
            accs = sort([run_arm(T, mode, frac, s) for s in 1:8])
            push!(cells, "$(round(accs[4], digits=3))/$(round(accs[1], digits=3))")
        end
        println("  $frac   " * join(rpad.(cells, 16)))
    end
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    GenericMLP.self_test()
end
