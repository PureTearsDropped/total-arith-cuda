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
# Entry totalization (added after external AI audit 2026-07-19): the public constructor
# itself totalizes — NaN→(0, no-bound+SUNK), ±Inf / out-of-range→±MAX+GE, subnormal→±MIN+LE.
# "Never NaN/Inf" only becomes an invariant when the entry door enforces it.
Tot(v::AbstractArray{<:Real}) = Tot(_sat(Float64.(v))...)

# ---- totalize: Float64 raw -> (Float32 val, UInt8 flag). Never emits NaN/Inf. ----
function _sat(raw::AbstractArray{Float64})
    nanm  = isnan.(raw)
    raw   = ifelse.(nanm, 0.0, raw)
    s     = sign.(raw)
    a     = abs.(raw)
    over  = a .> MAXF                                # Inf lands here too
    under = (a .> 0) .& (a .< MINF)
    val   = Float32.(ifelse.(over, s .* Float64(MAXF),
                     ifelse.(under, s .* Float64(MINF), raw)))
    flag  = (UInt8.(over) .* GE) .| (UInt8.(under) .* LE) .|
            (UInt8.(nanm) .* (GE | LE | SUNK))
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
    # Addition flag rule (revised 2026-07-19 after an external AI audit found the
    # counterexample (+MIN,LE)+(−MIN,=)→(0,LE), a lie under cancellation):
    # same-sign with known signs → simple OR is sound (|sum|=|a|+|b| is monotone);
    # cancellation possible (opposite signs / a zero / sign unknown) with any bound
    # on either input → one-sided bounds cannot survive → no-bound + SUNK.
    raw = Float64.(a.val) .+ Float64.(b.val)
    val, sflag = _sat(raw)
    fin  = a.flag .| b.flag
    anyb = (fin .& (GE | LE)) .> 0
    sign_known = (fin .& SUNK) .== 0x00
    same_sign  = (sign.(a.val) .* sign.(b.val)) .> 0     # strict: zero is not same-sign
    cancel = .!(sign_known .& same_sign)
    f = ifelse.(anyb .& cancel, GE | LE | SUNK, fin)
    return Tot(val, sflag .| f)
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
    if maximum(fin) == 0x00                          # fast path: no flags
        return Tot(val, sflag)
    end
    # Pattern rule (post-audit design: zero lies is absolute; keep the maximum within it).
    # An output component is a sum of products; the only danger is cancellation, so judge
    # per component: P0 all contributing terms exact → keep claims; P1 single live term →
    # cancellation impossible, scalar E1 survives (SUNK keeps the magnitude claim, only the
    # sign is unknown); P2 all live terms same known sign → the sum is monotone (all GE→GE,
    # all LE→LE, sign = the common sign); P3/4 mixed signs or SUNK among ≥2 terms →
    # no-bound + SUNK (6−10 vs 6−2: both magnitude and sign are lost).
    M = size(T, 1)
    outf = fill!(similar(sflag), 0x00)
    outsunk = fill!(similar(sflag, Bool), false)
    for k in 1:M
        nzs = findall(!=(0f0), @view T[k, :, :])
        ii = [c[1] for c in nzs]; jj = [c[2] for c in nzs]
        ss = reshape(Float64.(sign.(T[k, :, :][nzs])), 1, :)
        ai = a.val[:, ii]; bj = b.val[:, jj]
        fa = a.flag[:, ii]; fb = b.flag[:, jj]
        live = (ai .!= 0) .& (bj .!= 0)
        tf = ifelse.(live, _mul_flags(fa, fb), 0x00)            # per-term E1
        touched = vec(any((tf .| ifelse.(live, fa .| fb, 0x00)) .> 0, dims=2))
        sunk_any = vec(any((((fa .| fb) .& SUNK) .> 0) .& live, dims=2))
        n_live = vec(sum(live, dims=2))
        tsgn = ss .* sign.(Float64.(ai)) .* sign.(Float64.(bj))
        smax = vec(maximum(ifelse.(live, tsgn, -2.0), dims=2))
        smin = vec(minimum(ifelse.(live, tsgn, 2.0), dims=2))
        same_sign = smax .== smin
        ge_ok = vec(all((tf .& LE) .== 0, dims=2))
        le_ok = vec(all((tf .& GE) .== 0, dims=2))
        any_ge = vec(any((tf .& GE) .> 0, dims=2))
        any_le = vec(any((tf .& LE) .> 0, dims=2))
        f2 = (UInt8.(ge_ok .& any_ge) .* GE) .| (UInt8.(le_ok .& any_le) .* LE) .|
             (UInt8.(.!ge_ok .& .!le_ok) .* (GE | LE))
        keep = (.!sunk_any .& (same_sign .| (n_live .<= 1))) .| (sunk_any .& (n_live .== 1))
        p0 = .!touched
        outf[:, k] = ifelse.(p0, 0x00, ifelse.(keep, f2, GE | LE))
        outsunk[:, k] = .!p0 .& (.!keep .| (sunk_any .& (n_live .== 1)))
    end
    f = sflag .| outf .| (UInt8.(outsunk) .* SUNK)
    return Tot(val, f)
end

# ---------------------------------------------------------------- reference + self-test
"""Vector convenience (parity note: Python's einsum accepts arbitrary leading batch
dims; this Julia port accepts N×M matrices and, via this method, plain M-vectors)."""
function group_mul(T::AbstractArray{Float32,3},
                   a::Tot{<:AbstractVector{Float32}}, b::Tot{<:AbstractVector{Float32}})
    M = size(T, 1)
    c = group_mul(T, Tot(reshape(a.val, 1, M), reshape(a.flag, 1, M)),
                     Tot(reshape(b.val, 1, M), reshape(b.flag, 1, M)))
    return Tot(vec(c.val), vec(c.flag))
end

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
    println("="^72)
    println("④ entry totalization + regressions (external AI audit 2026-07-19)")
    println("="^72)
    t = Tot([NaN, Inf, -Inf, 1e300, -1e300])
    ok_entry = !any(isnan, t.val) && !any(isinf, t.val)
    println("  Tot([NaN,±Inf,±1e300]) → NaN/Inf leaked: $(ok_entry ? "none ✓" : "YES ✗") " *
            "(flags=$(Int.(t.flag)))")
    zero = tot_mul(Tot(Float32[0.0]), Tot([1e300]))
    zok = !any(isnan, zero.val)
    println("  0 × Tot(1e300): val=$(zero.val[1]) (old version: NaN) $(zok ? "✓" : "✗")")
    ra = Tot(Float32[MINF]); ra.flag[1] = LE
    rr = tot_add(ra, Tot(Float32[-MINF]))
    reg_ok = rr.flag[1] == (GE | LE | SUNK)
    println("  (+MIN,LE)+(−MIN,=): flag=$(Int(rr.flag[1])) = no-bound+SUNK " *
            (reg_ok ? "✓" : "✗ (old version: LE = a lie)"))
    @assert ok_entry && zok && reg_ok

    println("="^72)
    println("⑤ flag-algebra oracle: sample admissible true values, check the contract")
    println("="^72)
    K = 100_000
    function rand_flagged(K)
        mag = 10.0 .^ (rand(rng, K) .* 40 .- 20)
        sgn = rand(rng, (-1.0, 1.0), K)
        val = Float32.(mag .* sgn)
        fl  = rand(rng, UInt8[0x00, GE, LE, GE|SUNK, LE|SUNK, GE|LE], K)
        u   = rand(rng, K)
        ge_ = (fl .& GE) .> 0; le_ = (fl .& LE) .> 0
        m = ones(K)
        m = ifelse.(ge_ .& .!le_, 1 .+ 7 .* u, m)        # GE: |true| ∈ |val|·[1,8]
        m = ifelse.(le_ .& .!ge_, u, m)                   # LE: |true| ∈ |val|·[0,1]
        m = ifelse.(ge_ .& le_, 8 .* u, m)                # no-bound: anything
        ts = ifelse.((fl .& SUNK) .> 0, rand(rng, (-1.0, 1.0), K),
                     sign.(Float64.(val)))                # SUNK: sign is random too
        tv = abs.(Float64.(val)) .* m .* ts
        return Tot(val, fl), tv
    end
    A2, ta = rand_flagged(K); B2, tb = rand_flagged(K)
    lies5 = 0
    for (name, op) in (("mul", tot_mul), ("add", tot_add), ("div", tot_div))
        r = op(A2, B2)
        t = name == "mul" ? ta .* tb : name == "add" ? ta .+ tb :
            ifelse.(tb .== 0, 0.0, ta ./ ifelse.(tb .== 0, 1.0, tb))
        ge = (r.flag .& GE) .> 0; le = (r.flag .& LE) .> 0; sk = (r.flag .& SUNK) .> 0
        vo = Float64.(r.val); slack = 2.0^-20
        lies5 += count(ge .& .!le .& (abs.(t) .< abs.(vo) .* (1 - slack)))
        lies5 += count(le .& .!ge .& (abs.(t) .> abs.(vo) .* (1 + slack)))
        lies5 += count(.!sk .& (vo .!= 0) .& (t .!= 0) .& (sign.(t) .!= sign.(vo)))
        lies5 += count(x -> isnan(x) || isinf(x), r.val)
    end
    println("  $(3K) flagged cases (true values sampled from admissible sets): lies $(lies5)")
    @assert lies5 == 0

    println("="^72)
    println("⑥ group_mul oracle (audit round 2 + pattern-rule soundness/retention)")
    println("="^72)
    function check6(label, kind, M, KB, mode)
        T = wiring_tensor(kind, M)
        Af, taf = rand_flagged(KB * M); Bf, tbf = rand_flagged(KB * M)
        av = reshape(Af.val, KB, M); afl = reshape(Af.flag, KB, M); ta = reshape(taf, KB, M)
        bv = reshape(Bf.val, KB, M); bfl = reshape(Bf.flag, KB, M); tb = reshape(tbf, KB, M)
        if mode == :sparse
            keep = falses(KB, M)
            for r in 1:KB, _ in 1:2; keep[r, rand(rng, 1:M)] = true; end
            av = ifelse.(keep, av, 0f0); afl = ifelse.(keep, afl, 0x00); ta = ifelse.(keep, ta, 0.0)
            bv = ifelse.(keep, bv, 0f0); bfl = ifelse.(keep, bfl, 0x00); tb = ifelse.(keep, tb, 0.0)
        elseif mode == :positive
            av = abs.(av); bv = abs.(bv); ta = abs.(ta); tb = abs.(tb)
            afl = afl .& ~SUNK; bfl = bfl .& ~SUNK
        end
        A = Tot(av, afl); B = Tot(bv, bfl)
        r = group_mul(T, A, B)
        t = similar(ta)
        for k in 1:M
            Tk = Float64.(@view T[k, :, :])
            t[:, k] = sum((ta * Tk) .* tb, dims=2)
        end
        ge = (r.flag .& GE) .> 0; le = (r.flag .& LE) .> 0; sk = (r.flag .& SUNK) .> 0
        vo = Float64.(r.val); slack = 2.0^-20
        lies = count(ge .& .!le .& (abs.(t) .< abs.(vo) .* (1 - slack)))
        lies += count(le .& .!ge .& (abs.(t) .> abs.(vo) .* (1 + slack)))
        lies += count(.!sk .& (vo .!= 0) .& (t .!= 0) .& (sign.(t) .!= sign.(vo)))
        lies += count(x -> isnan(x) || isinf(x), r.val)
        frow = repeat(any((afl .| bfl) .> 0, dims=2), 1, M)
        claims = ((ge .⊻ le) .| .!sk) .& frow
        ret = count(claims) / max(count(frow), 1)
        println("  $(rpad(label, 24)) $(KB) rows: lies $(lies), claims retained $(round(100ret, digits=1))%")
        return lies
    end
    bad6 = check6("quaternion dense ±", :cd, 4, 10_000, :dense)
    bad6 += check6("sedenion sparse(2)", :cd, 16, 2_000, :sparse)
    bad6 += check6("cyclic ℤ/8 positive", :cyclic, 8, 5_000, :positive)
    T1 = wiring_tensor(:cd, 1)
    r1 = group_mul(T1, Tot(reshape(Float32[2.0], 1, 1), reshape(UInt8[SUNK], 1, 1)),
                       Tot(reshape(Float32[3.0], 1, 1)))
    reg6 = r1.flag[1] == SUNK && r1.val[1] == 6f0
    println("  audit counterexample (2,SUNK)×(3,=): val=$(r1.val[1]) flag=$(Int(r1.flag[1])) " *
            "= exact magnitude + unknown sign $(reg6 ? "✓" : "✗")")
    @assert bad6 == 0 && reg6

    println()
    println("TotalArith.jl: totality (no NaN, honest flags) + wiring swap + " *
            "round-once accumulation, in generic Julia.")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    TotalArith.self_test()
end
