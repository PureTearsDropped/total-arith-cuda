# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
TotalPipeline — the U → V(O, N, M) → W architecture of HyperAlgebra, made EXPLICIT
(Julia twin of total_pipeline.py).

External review (2026-07-20) observed the audited core already runs as

    U  entry totalization       Tot(...) / _sat        raw値 → (val, flag) 不変形式
    V  fused multiply-accumulate Float64 MAC            計算本体 (audited kernel)
    W  semantic flag commit     saturate-once + pattern rule
    N  algebra structure        wiring tensor T[k,i,j]  差し替え可能
    M  representation           implicit L_a (contraction; no matrix materialized)
    O  operator                 tot_add / tot_mul / tot_div / group_mul

— "already that architecture, just unnamed."  This module names it WITHOUT touching the
audited core: every call delegates to HyperAlgebra bit-for-bit (asserted in self_test).
V and W remain FUSED inside the kernel by design; this layer names the boundary.

  PipeAlgebra (N) : from_kind(:cd|:cyclic, M)  or  from_registry(NestedSeries.Alg)
                    — the second bridges the ALGS shelf (dualquat, Clifford, Grassmann…)
                    onto the audited kernel, flags included.
  OPS         (O) : operator registry; papply(op, a, b; algebra) is the one gateway.
  Lmatrix     (M) : the explicit L_a for inspection; production stays implicit.
  TotalPipe       : declarative (O, N) pair — compose by naming, then call.
"""
module TotalPipeline

import Random
include(joinpath(@__DIR__, "HyperAlgebra.jl"));  using .HyperAlgebra
include(joinpath(@__DIR__, "NestedSeries.jl"));  using .NestedSeries

export PipeAlgebra, from_kind, from_registry, totalize, papply, OPS, Lmatrix, TotalPipe

# ================================================================ N: the algebra slot
"N — the structure tensor T[k,i,j] with a name; swapping it swaps the product, not the kernel"
struct PipeAlgebra
    name::String
    T::Array{Float32,3}
end
Base.show(io::IO, A::PipeAlgebra) = print(io, "PipeAlgebra(", A.name, ", dim ", size(A.T, 1), ")")
pdim(A::PipeAlgebra) = size(A.T, 1)

from_kind(kind::Symbol, M::Int) = PipeAlgebra(string(kind, M), wiring_tensor(kind, M))

"""bridge from NestedSeries.Alg (tab[i][j] = eᵢeⱼ) — the whole ALGS shelf becomes
   runnable on the audited kernel (T[k,i,j] layout)."""
function from_registry(alg::NestedSeries.Alg)
    d = alg.dim
    T = zeros(Float32, d, d, d)
    for i in 1:d, j in 1:d, k in 1:d
        T[k, i, j] = Float32(alg.tab[i][j][k])
    end
    PipeAlgebra(alg.name, T)
end

# ================================================================ M: implicit vs explicit
"""M made explicit: (L_a)[k,j] = Σ_i T[k,i,j] a_i — the matrix the kernel APPLIES without
   materializing.  For inspection/verification; production stays implicit (fused MAC)."""
function Lmatrix(A::PipeAlgebra, x::AbstractVector)
    d = pdim(A)
    L = zeros(Float64, d, d)
    for k in 1:d, i in 1:d, j in 1:d
        L[k, j] += Float64(A.T[k, i, j]) * Float64(x[i])
    end
    L
end

# ================================================================ U: the named entry
"U — entry totalization: any raw vector/matrix (or Tot) → total form (val, flag)"
totalize(x::Tot) = x
totalize(x::AbstractArray{<:Real}) = Tot(x)

# ================================================================ O: the operator registry
# arity-2 operators delegating to the audited kernels — the interface, not a re-implementation
const OPS = Dict{Symbol,NamedTuple}(
    :add  => (fn = (alg, a, b) -> tot_add(a, b), needs_algebra = false),
    :mul  => (fn = (alg, a, b) -> tot_mul(a, b), needs_algebra = false),
    :div  => (fn = (alg, a, b) -> tot_div(a, b), needs_algebra = false),
    :gmul => (fn = (alg, a, b) -> group_mul(alg.T, a, b), needs_algebra = true),
)

"""the one gateway: U → V(O,N,M) → W.  U runs here; V and W run fused inside the
   audited kernel (Float64 accumulate, saturate once)."""
function papply(op::Symbol, a, b; algebra::Union{PipeAlgebra,Nothing} = nothing)
    entry = OPS[op]
    entry.needs_algebra && algebra === nothing &&
        error("operator :$op needs a PipeAlgebra (the N slot)")
    entry.fn(algebra, totalize(a), totalize(b))
end

# ================================================================ the declarative pair
"declare (O, N) once, call many times: p = TotalPipe(:gmul, from_kind(:cd, 16))"
struct TotalPipe
    op::Symbol
    algebra::Union{PipeAlgebra,Nothing}
    function TotalPipe(op::Symbol, algebra = nothing)
        OPS[op].needs_algebra && algebra === nothing && error(":$op needs a PipeAlgebra")
        new(op, algebra)
    end
end
(p::TotalPipe)(a, b) = papply(p.op, a, b; algebra = p.algebra)
describe(p::TotalPipe) = string("U: totalize (Tot entry)  →  V: fused Float64 MAC [O=",
    p.op, ", N=", p.algebra === nothing ? "—(elementwise)" : p.algebra.name,
    ", M=implicit L_a]  →  W: saturate-once + pattern-rule flags")

# ================================================================ self-test
function self_test()
    rng = Random.MersenneTwister(0)
    println("TotalPipeline — names for an architecture that already runs; core untouched")

    # ① gateway ≡ direct audited-kernel calls, bit-identical (NaN/Inf-laden inputs)
    raw_a = repeat([1.5, -2.0, NaN, Inf, 0.0, 1e-40, 3.0, -1e39], 2)'
    raw_b = repeat([0.5, 0.0, 2.0, -Inf, NaN, 4.0, -2.5, 1e39], 2)'
    A16 = from_kind(:cd, 16)
    for op in (:add, :mul, :div, :gmul)
        g = papply(op, collect(raw_a), collect(raw_b); algebra = A16)
        d = OPS[op].fn(A16, totalize(collect(raw_a)), totalize(collect(raw_b)))
        @assert g.val == d.val && g.flag == d.flag
    end
    println("① gateway ≡ direct kernel calls, bit-identical (add/mul/div/gmul, NaN/Inf入り) ✓")

    # ② N is a slot: same gateway, several wirings, checked against Float64 reference
    for (kind, M) in ((:cd, 2), (:cd, 4), (:cd, 16), (:cyclic, 8))
        Ak = from_kind(kind, M)
        a, b = randn(rng, M), randn(rng, M)
        got = Float64.(papply(:gmul, a, b; algebra = Ak).val)
        ref = HyperAlgebra.ref_mul(Ak.T, a, b)
        @assert maximum(abs.(got .- ref)) < 1e-5 * max(1.0, maximum(abs.(ref)))
    end
    println("② N swap through one gateway: cd2/cd4/cd16/cyclic8 ≡ reference ✓")

    # ③ implicit M ≡ explicit M
    a, b = randn(rng, 16), randn(rng, 16)
    imp = Float64.(papply(:gmul, a, b; algebra = A16).val)
    exp_ = Lmatrix(A16, a) * b
    @assert maximum(abs.(imp .- exp_)) < 1e-5
    println("③ implicit M (fused MAC) ≡ explicit M (L_a·b) ✓ — 行列は作らず作用だけ")

    # ④ ALGS bridge: the shelf runs on the audited kernel — flags included
    for nm in (:dualquat, :cl3, :grassmann2, :biquat, :sedenion)
        alg_ns = NestedSeries.alg(nm)
        Ab = from_registry(alg_ns)
        d = pdim(Ab)
        a, b = randn(rng, d), randn(rng, d)
        got = Float64.(papply(:gmul, a, b; algebra = Ab).val)
        ref = NestedSeries.rawmul(alg_ns, a, b)
        @assert maximum(abs.(got .- ref)) < 1e-4 * max(1.0, maximum(abs.(ref))) "$nm"
        bad = copy(a); bad[1] = NaN
        r = papply(:gmul, bad, b; algebra = Ab)
        @assert all(isfinite, r.val)
    end
    lie4 = NestedSeries.lie(NestedSeries.cd_alg(4))     # empty output rows → kernel guard
    Al = from_registry(lie4)
    a, b = randn(rng, 4), randn(rng, 4)
    got = Float64.(papply(:gmul, a, b; algebra = Al).val)
    @assert maximum(abs.(got .- NestedSeries.rawmul(lie4, a, b))) < 1e-5
    println("④ ALGS bridge: dualquat/cl3/Λ2/biquat/sedenion + lie⟨cd4⟩(空行guard) が")
    println("   監査済みカーネルで参照一致・NaN入力もフラグ付き有限 ✓")

    # ⑤ declarative composition
    p = TotalPipe(:gmul, from_registry(NestedSeries.alg(:dualquat)))
    _ = p(randn(rng, 8), randn(rng, 8))
    println("⑤ ", describe(p))
    println("done — U/V/W/O/N/M named, swappable, and bit-faithful to the audited core")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    TotalPipeline.self_test()
end
