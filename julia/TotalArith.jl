# ⚠️ AI-assisted; verify before relying on it. / 生成AI使用・要検証
"""
TotalArith.jl — total arithmetic + swappable wiring (structure tensor), Julia port of cuda_total.py.

  * A number is (val::Float32 array, flag::UInt8 array). Flag bits: GE=1 (≥), LE=2 (≤), SUNK=4.
  * Total: overflow→±MAX+GE, underflow→±MIN(=ε, direction kept)+LE, a/0=0. NaN/Inf are never produced.
  * Wiring table = structure tensor T[k,i,j]. Swap T and the same code is complex /
    quaternion / sedenion (Cayley–Dickson, XOR routing) or cyclic convolution.
  * Accumulate in Float64, saturate (round) once at the end — the quire discipline.

  Written generically over AbstractArray: the same functions run on `Array` (CPU) and are
  CuArray-ready (broadcasts + matmul only). CPU path is what the self-test below verifies.

  Semantics mirror cuda_total.py exactly (same flag algebra, same honest caveat: flags mark
  totalization events only; Float32 nearest rounding is not flagged — it has no direction).
"""
module TotalArith

import Random

export Tot, tot_mul, tot_add, tot_div, wiring_tensor, group_mul, GE, LE, SUNK, MAXF, MINF

const GE   = 0x01
const LE   = 0x02
const SUNK = 0x04
const MAXF = floatmax(Float32)          # saturation ceiling
const MINF = floatmin(Float32)          # ε = smallest normal (directed infinitesimal)

struct Tot{V<:AbstractArray{Float32},F<:AbstractArray{UInt8}}
    val::V
    flag::F
end
Tot(v::AbstractArray{<:Real}) = (v32 = Float32.(v); Tot(v32, zero_flags(v32)))
zero_flags(v) = fill!(similar(v, UInt8), 0x00)

# ---- totalize: Float64 raw -> (Float32 val, UInt8 flag). Never emits NaN/Inf. ----
function _sat(raw::AbstractArray{Float64})
    s     = sign.(raw)
    a     = abs.(raw)
    over  = a .> MAXF
    under = (a .> 0) .& (a .< MINF)
    val   = Float32.(ifelse.(over, s .* Float64(MAXF),
                     ifelse.(under, s .* Float64(MINF), raw)))
    flag  = (UInt8.(over) .* GE) .| (UInt8.(under) .* LE)
    return val, flag
end

# ---- flag algebra for products (conservative rule E1): ≥·≥=≥, ≤·≤=≤, =·x=x, ≥·≤=no-bound ----
function _mul_flags(fa, fb)
    ga = fa .& GE;          la = (fa .>> 1) .& 0x01
    gb = fb .& GE;          lb = (fb .>> 1) .& 0x01
    ge_o = ((ga .| gb) .& .~(la .| lb)) .& 0x01
    le_o = ((la .| lb) .& .~(ga .| gb)) .& 0x01
    nb   = ((ga .| gb) .& (la .| lb)) .& 0x01
    return (ge_o .* GE) .| (le_o .* LE) .| (nb .* (GE | LE)) .| ((fa .| fb) .& SUNK)
end

function tot_mul(a::Tot, b::Tot)
    raw = Float64.(a.val) .* Float64.(b.val)
    val, sflag = _sat(raw)
    return Tot(val, sflag .| _mul_flags(a.flag, b.flag))
end

function tot_add(a::Tot, b::Tot)
    raw = Float64.(a.val) .+ Float64.(b.val)
    val, sflag = _sat(raw)
    # saturated cancellation (MAX−MAX): both GE, opposite signs → sign-unknown + no-bound
    clash = ((a.flag .& GE) .> 0) .& ((b.flag .& GE) .> 0) .&
            (sign.(a.val) .!= sign.(b.val)) .& (a.val .!= 0) .& (b.val .!= 0)
    f = sflag .| a.flag .| b.flag .| (UInt8.(clash) .* (SUNK | GE | LE))
    return Tot(val, f)
end

function tot_div(a::Tot, b::Tot)
    bz  = b.val .== 0
    raw = Float64.(a.val) ./ Float64.(ifelse.(bz, one(Float32), b.val))
    raw = ifelse.(bz, 0.0, raw)                       # a/0 = 0 (Moore–Penrose)
    val, sflag = _sat(raw)
    fin = a.flag .| b.flag
    nb  = (fin .& (GE | LE)) .> 0                     # bounded input ⟹ quotient: no-bound
    f   = sflag .| (UInt8.(nb) .* (GE | LE)) .| (fin .& SUNK)
    return Tot(val, f)
end

# ---------------------------------------------------------------- wiring (structure tensor)

# Cayley–Dickson product, same convention as sedenion_tensor_logic._cd:
#   (a,b)(c,d) = (ac − d̄b, da + bc̄)
_conj(x::Vector{Float64}) = length(x) == 1 ? copy(x) :
    vcat(_conj(x[1:end÷2]), -x[end÷2+1:end])
function _cd(x::Vector{Float64}, y::Vector{Float64})
    n = length(x)
    n == 1 && return x .* y
    h = n ÷ 2
    a, b, c, d = x[1:h], x[h+1:end], y[1:h], y[h+1:end]
    return vcat(_cd(a, c) .- _cd(_conj(d), b),
                _cd(d, a) .+ _cd(b, _conj(c)))
end

"""wiring_tensor(kind, M) -> T[k,i,j] (M×M×M Float32). kind: :cd (Cayley–Dickson,
XOR routing e_i·e_j = ±e_{i⊻j}) or :cyclic (cyclic convolution, k = (i+j) mod M)."""
function wiring_tensor(kind::Symbol, M::Int)
    T = zeros(Float32, M, M, M)
    if kind === :cd
        E = [Float64.(1:M .== i) for i in 1:M]
        for i in 1:M, j in 1:M
            v = _cd(E[i], E[j])
            k = argmax(abs.(v))
            @assert k == ((i-1) ⊻ (j-1)) + 1 "XOR routing broken at ($i,$j)"
            T[k, i, j] = Float32(sign(v[k]))
        end
    elseif kind === :cyclic
        for i in 1:M, j in 1:M
            T[mod(i + j - 2, M) + 1, i, j] = 1f0
        end
    else
        error("unknown wiring kind: $kind")
    end
    return T
end

"""group_mul(T, a, b): batched wiring product c[n,k] = Σ_ij T[k,i,j]·a[n,i]·b[n,j].
Batch is rows (N×M). Accumulates in Float64, saturates once (fused-MAC philosophy).
Only broadcasts + matmul ⟹ runs on Array and CuArray alike."""
function group_mul(T::AbstractArray{Float32,3}, a::Tot, b::Tot)
    A = Float64.(a.val); B = Float64.(b.val)
    M = size(T, 1)
    raw = similar(A)
    for k in 1:M                                   # M ≤ 16: small loop, big matmuls
        Tk = Float64.(@view T[k, :, :])
        raw[:, k] = sum((A * Tk) .* B, dims=2)
    end
    val, sflag = _sat(raw)
    fin  = a.flag .| b.flag
    anyf = any((fin .& (GE | LE)) .> 0, dims=2)     # components mix ⟹ conservative
    f    = sflag .| (UInt8.(anyf) .* (GE | LE))
    return Tot(val, f)
end

# ---------------------------------------------------------------- reference + self-test
"""ref_mul(T, x, y): direct Float64 structure-tensor product for verification."""
function ref_mul(T::AbstractArray{Float32,3}, x::Vector{Float64}, y::Vector{Float64})
    M = size(T, 1)
    r = zeros(Float64, M)
    for k in 1:M, i in 1:M, j in 1:M
        r[k] += Float64(T[k, i, j]) * x[i] * y[j]
    end
    return r
end

function self_test()
    println("Julia $(VERSION), threads=$(Threads.nthreads()) — TotalArith.jl self-test (CPU)")
    println("="^72)
    println("① totality: never NaN/Inf, flags never lie (adversarial)")
    println("="^72)
    rng = Random.MersenneTwister(20260810)
    special = Float32[0.0, MINF, -MINF, MINF/2, -MINF/2, MAXF, -MAXF, 1f30, -1f30,
                      1f-30, -1f-30, sqrt(MAXF), -sqrt(MAXF), 1.0, -1.0]
    N = 1_000_000
    pool = [rand(rng) < 0.3 ? special[rand(rng, 1:length(special))] :
            Float32(randn(rng)) * exp2(Float32(rand(rng, -120:120))) for _ in 1:N]
    av = Tot(pool); bv = Tot(circshift(pool, 7))
    bad_nan = 0; bad_lie = 0
    for (name, op) in (("mul", tot_mul), ("add", tot_add), ("div", tot_div))
        c = op(av, bv)
        bad_nan += count(x -> isnan(x) || isinf(x), c.val)
        # flag lies: |val|==MAX must carry GE; val==±MIN from a collapse must carry LE
        raw = name == "mul" ? Float64.(av.val) .* Float64.(bv.val) :
              name == "add" ? Float64.(av.val) .+ Float64.(bv.val) :
              ifelse.(bv.val .== 0, 0.0,
                      Float64.(av.val) ./ Float64.(ifelse.(bv.val .== 0, 1f0, bv.val)))
        over  = abs.(raw) .> MAXF
        under = (abs.(raw) .> 0) .& (abs.(raw) .< MINF)
        bad_lie += count(over  .& ((c.flag .& GE) .== 0))
        bad_lie += count(under .& ((c.flag .& LE) .== 0))
    end
    println("  $(N) × mul/add/div: NaN/Inf $(bad_nan), flag lies $(bad_lie)")
    @assert bad_nan == 0 && bad_lie == 0

    println("="^72)
    println("② wiring swap: same code, different algebra (T only)")
    println("="^72)
    for (label, kind, M) in (("complex", :cd, 2), ("quaternion", :cd, 4),
                             ("sedenion", :cd, 16), ("cyclic ℤ/8", :cyclic, 8))
        T = wiring_tensor(kind, M)
        viol = 0
        for _ in 1:200
            x = randn(rng, M); y = randn(rng, M)
            got = group_mul(T, Tot(reshape(Float32.(x), 1, M)),
                               Tot(reshape(Float32.(y), 1, M)))
            refv, _ = _sat(reshape(ref_mul(T, Float64.(Float32.(x)), Float64.(Float32.(y))), 1, M))
            viol += count(vec(got.val) .!= vec(refv))
        end
        println("  $(rpad(label, 12)) M=$(lpad(M, 2)): violations $(viol)/200 " *
                (viol == 0 ? "✓" : "✗"))
        @assert viol == 0
    end

    println("="^72)
    println("③ throughput (CPU reference; the same code is CuArray-ready)")
    println("="^72)
    T = wiring_tensor(:cd, 16)
    for NB in (10_000, 100_000)
        a = Tot(randn(Float32, NB, 16)); b = Tot(randn(Float32, NB, 16))
        group_mul(T, a, b)                             # warm-up
        t = @elapsed group_mul(T, a, b)
        println("  batch $(lpad(NB, 7)): $(round(t*1000, digits=2)) ms = " *
                "$(round(NB/t/1e6, digits=2)) M sed-products/s")
    end
    println()
    println("TotalArith.jl: totality (no NaN, honest flags) + wiring swap + " *
            "round-once accumulation, in generic Julia.")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    TotalArith.self_test()
end
