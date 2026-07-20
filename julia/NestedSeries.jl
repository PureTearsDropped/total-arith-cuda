# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""
NestedSeries — the M/N/O layers as FREELY COMPOSABLE registries (Julia twin of
nested_series.py, generalized).

  Everything is an `Alg`: a bilinear algebra (dim, unit, structure table).  The three
  layers are three registries over that one interface:

    N layer (cells)      : cd_alg(M) (ℝ/ℂ/ℍ/𝕆/sedenion), cyclic_alg(M) (group ℤ/M),
                           matn_alg(n) (real n×n matrices)  — the wiring tables
    M layer (combinators): mat_over(alg, N) (N×N matrix of cells),
                           tensor(A, B)     (A ⊗ B — two wiring tables multiplied)
                           — each RETURNS a new Alg, so they nest recursively:
                           mat_over(tensor(cd_alg(4), cd_alg(2)), 2) just works.
    O layer (tapes)      : TAPES — series coefficients (exp, sin, cos, sinh, cosh, …)
                           + a declared bracket (:left / :right) for building powers.

  Nothing about a combination is assumed: `assoc_defect(alg)` MEASURES whether the
  composed algebra is associative, and `nlog` (inverse ⇒ candidate) verifies its answer
  with the safe forward exp — unverifiable ⇒ INEXACT flag, never a silent lie.
  Total: elements carry (coeffs, flag); NaN→0+SING, overflow→±MAX+OVER at every step.

  Measured laws this module lets you reproduce (self_test):
    · associativity survives composition iff every ingredient is associative
      (cd(≤4), cyclic, matn, and their tensors/matrices — but one octonion cell
      infects the whole tower)
    · exp∘log = id verifies (1e-15) exactly on the associative combinations and
      breaks structurally (≈1e-3) on the non-associative ones — same code, same tape
    · brackets :left / :right agree on scalars (power-associativity), split for
      matrices of non-associative cells (the "many exps")
"""
module NestedSeries

export Alg, cd_alg, cyclic_alg, matn_alg, mat_over, tensor, jordan, lie, commutator,
       Nel, nel, coeffs, flagof, tmul, tadd,
       TAPES, series, nexp, nsin, ncos, nsinh, ncosh, nexp_ss, nlog, ninv,
       assoc_defect, powerassoc_defect, commut_defect, SING, OVER, INEXACT

const SING    = 0x01
const OVER    = 0x04
const INEXACT = 0x08
const MAXF = floatmax(Float64)

# ================================================================ the one interface
"""A bilinear algebra: `dim`, `unit` (the 1), and the structure table
   `tab[i][j] :: Vector` = eᵢ·eⱼ expanded in the basis. The table IS the wiring."""
struct Alg
    name::String
    dim::Int
    unit::Vector{Float64}
    tab::Vector{Vector{Vector{Float64}}}      # tab[i][j] = basis product eᵢ eⱼ
end
Base.show(io::IO, A::Alg) = print(io, A.name, "(dim ", A.dim, ")")

function _from_mul(name, d, unit, mul)        # extract the wiring table once
    E = [Float64.(1:d .== i) for i in 1:d]
    Alg(name, d, unit, [[mul(E[i], E[j]) for j in 1:d] for i in 1:d])
end

"raw bilinear product through the wiring table (dense loops; clarity over speed)"
function rawmul(A::Alg, x::Vector{Float64}, y::Vector{Float64})
    r = zeros(A.dim)
    for i in 1:A.dim
        xi = x[i]; xi == 0.0 && continue
        ti = A.tab[i]
        for j in 1:A.dim
            yj = y[j]; yj == 0.0 && continue
            r .+= (xi * yj) .* ti[j]
        end
    end
    r
end

# ================================================================ N layer: cell registry
function _cdconj(x); n = length(x); n == 1 ? copy(x) : vcat(x[1], -x[2:end]); end
function _cdprod(x, y)
    n = length(x); n == 1 && return x .* y
    h = n ÷ 2
    a, b, c, d = x[1:h], x[h+1:end], y[1:h], y[h+1:end]
    vcat(_cdprod(a, c) .- _cdprod(_cdconj(d), b), _cdprod(d, a) .+ _cdprod(b, _cdconj(c)))
end
"Cayley–Dickson algebra of dim M: ℝ(1) ℂ(2) ℍ(4) 𝕆(8) sedenion(16) …"
cd_alg(M::Int) = _from_mul("cd$M", M, Float64.(1:M .== 1), _cdprod)

"group algebra of ℤ/M: eᵢ·eⱼ = e_{(i+j) mod M} — commutative AND associative"
cyclic_alg(M::Int) = _from_mul("cyc$M", M, Float64.(1:M .== 1),
    (x, y) -> begin
        r = zeros(M)
        for i in 0:M-1, j in 0:M-1
            r[mod(i + j, M) + 1] += x[i+1] * y[j+1]
        end
        r
    end)

"real n×n matrices as a dim-n² algebra (column-major vec) — associative, with zero divisors"
matn_alg(n::Int) = _from_mul("mat$n", n * n, vec(Matrix{Float64}(I0(n))),
    (x, y) -> vec(reshape(x, n, n) * reshape(y, n, n)))
I0(n) = [i == j ? 1.0 : 0.0 for i in 1:n, j in 1:n]

# ================================================================ M layer: combinators
"""N×N matrix over any Alg — a new Alg of dim N²·cell.dim (block index (r,c,k)).
   The matrix product's summation order is fixed; whether the RESULT is associative
   depends on the cell (measure with assoc_defect, don't assume)."""
function mat_over(cell::Alg, N::Int)
    d = cell.dim; D = N * N * d
    at(r, c, k) = ((c - 1) * N + (r - 1)) * d + k          # column-major blocks
    unit = zeros(D); for i in 1:N, k in 1:d; unit[at(i, i, k)] = cell.unit[k]; end
    mul = (x, y) -> begin
        r = zeros(D)
        xb = (i, j) -> x[at(i, j, 1):at(i, j, d)]
        yb = (i, j) -> y[at(i, j, 1):at(i, j, d)]
        for i in 1:N, j in 1:N
            acc = zeros(d)
            for m in 1:N
                acc .+= rawmul(cell, xb(i, m), yb(m, j))
            end
            r[at(i, j, 1):at(i, j, d)] = acc
        end
        r
    end
    _from_mul("mat$(N)⟨$(cell.name)⟩", D, unit, mul)
end

"tensor product A ⊗ B — two wiring tables multiplied: (eₐ⊗f_b)(e_c⊗f_d) = (eₐe_c)⊗(f_bf_d)"
function tensor(A::Alg, B::Alg)
    dA, dB = A.dim, B.dim; D = dA * dB
    at(a, b) = (a - 1) * dB + b
    unit = zeros(D)
    for a in 1:dA, b in 1:dB; unit[at(a, b)] = A.unit[a] * B.unit[b]; end
    mul = (x, y) -> begin
        r = zeros(D)
        for a in 1:dA, b in 1:dB
            xab = x[at(a, b)]; xab == 0.0 && continue
            for c in 1:dA, d in 1:dB
                ycd = y[at(c, d)]; ycd == 0.0 && continue
                sA = A.tab[a][c]; sB = B.tab[b][d]
                for p in 1:dA
                    sA[p] == 0.0 && continue
                    for q in 1:dB
                        r[at(p, q)] += xab * ycd * sA[p] * sB[q]
                    end
                end
            end
        end
        r
    end
    _from_mul("$(A.name)⊗$(B.name)", D, unit, mul)
end

# ================================================================ total elements + ops
"""symmetrized (Jordan) product a∘b = (ab+ba)/2 — commutative by construction, but
   associativity is generally LOST (measure it). This is the 'symmetrized exp' member
   of the exp family made into a combinator. Measured role: a commutative-but-non-
   associative tensor partner does NOT preserve power-associativity — commutativity
   alone is not enough, the partner must be commutative AND associative."""
function jordan(A::Alg)
    _from_mul("sym⟨$(A.name)⟩", A.dim, copy(A.unit),
              (x, y) -> (rawmul(A, x, y) .+ rawmul(A, y, x)) ./ 2)
end

"""antisymmetrized product ½[a,b] = (ab−ba)/2 — jordan's sibling: the ORDER-ONLY half.
   Every product splits EXACTLY as  ab = a∘b + ½[a,b]  (order-forgetting + order-carrying);
   commutativity is precisely "the lie half vanishes".  Measured ladder of what the
   commutator machinery can repair (see self_test):
     · 2-variable BCH  exp(a)exp(b) = exp(a+b+½[a,b]+1/12[a,[a,b]]+1/12[b,[b,a]]+…)
       repairs at s⁴-scaling up to the OCTONIONS (Artin: any 2-generated subalgebra is
       associative) and breaks to s³ at the sedenions (alternativity lost).
     · 3-variable Jacobi [[a,b],c]+[[b,c],a]+[[c,a],b]=0 breaks already at the octonions
       (the commutator algebra is Malcev, not Lie)."""
function lie(A::Alg)
    _from_mul("lie⟨$(A.name)⟩", A.dim, zeros(A.dim),        # no unit: [1,x]=0 kills it
              (x, y) -> (rawmul(A, x, y) .- rawmul(A, y, x)) ./ 2)
end
"the commutator [a,b] = ab − ba on Nel — the order information itself"
commutator(A::Alg, x, y) = tadd(tmul(A, x, y), tscale(tmul(A, y, x), -1.0))

"element of an Alg: coefficients + flag; totalized at every step (never NaN/Inf)"
struct Nel
    c::Vector{Float64}
    flag::UInt8
end
nel(A::Alg, c::AbstractVector) = _tot(Float64.(collect(c)), 0x00)
nel(A::Alg) = Nel(copy(A.unit), 0x00)                       # the 1
coeffs(x::Nel) = x.c
flagof(x::Nel) = x.flag
function _tot(c::Vector{Float64}, f::UInt8)
    for i in eachindex(c)
        v = c[i]
        if isnan(v); c[i] = 0.0; f |= SING
        elseif !isfinite(v) || abs(v) > MAXF; c[i] = sign(v) * MAXF; f |= OVER
        end
    end
    Nel(c, f)
end
tmul(A::Alg, x::Nel, y::Nel) = _tot(rawmul(A, x.c, y.c), x.flag | y.flag)
tadd(x::Nel, y::Nel) = _tot(x.c .+ y.c, x.flag | y.flag)
tscale(x::Nel, s::Float64) = _tot(x.c .* s, x.flag)

# ================================================================ O layer: tape registry
const TAPES = Dict{Symbol,Function}(
    :exp  => k -> 1.0 / factorial(big(k)),
    :sin  => k -> iseven(k) ? 0.0 : Float64((-1)^((k - 1) ÷ 2) / factorial(big(k))),
    :cos  => k -> isodd(k)  ? 0.0 : Float64((-1)^(k ÷ 2) / factorial(big(k))),
    :sinh => k -> iseven(k) ? 0.0 : 1.0 / factorial(big(k)),
    :cosh => k -> isodd(k)  ? 0.0 : 1.0 / factorial(big(k)),
)

"""Σ c_k x^k on ANY Alg, powers built by the DECLARED bracket
   (:left → x^k = x^{k-1}·x, :right → x·x^{k-1}). One skeleton, many tapes."""
function series(A::Alg, x::Nel, tape; order::Int = 20, bracket::Symbol = :left)
    c = tape isa Symbol ? TAPES[tape] : tape
    acc = tscale(nel(A), Float64(c(0)))
    P = nel(A)
    for k in 1:order
        P = bracket === :left ? tmul(A, P, x) : tmul(A, x, P)
        ck = Float64(c(k))
        ck != 0.0 && (acc = tadd(acc, tscale(P, ck)))
    end
    acc
end
nexp(A, x; kw...)  = series(A, x, :exp;  kw...)
nsin(A, x; kw...)  = series(A, x, :sin;  order = 21, kw...)
ncos(A, x; kw...)  = series(A, x, :cos;  kw...)
nsinh(A, x; kw...) = series(A, x, :sinh; order = 21, kw...)
ncosh(A, x; kw...) = series(A, x, :cosh; kw...)

"exp by scaling-and-squaring — a DIFFERENT cell connection; agreement with nexp is measured"
function nexp_ss(A::Alg, x::Nel; order::Int = 12, s::Int = 3, bracket::Symbol = :left)
    acc = series(A, tscale(x, 1.0 / 2^s), :exp; order, bracket)
    for _ in 1:s; acc = tmul(A, acc, acc); end
    acc
end

"""log = inverse ⇒ CANDIDATE: series log(1+X) (X = x − 1, needs ‖X‖ small), then verified
   by the safe forward exp; unverified ⇒ INEXACT — a candidate, never a silent lie."""
function nlog(A::Alg, x::Nel; order::Int = 30, verify_order::Int = 20)
    X = tadd(x, tscale(nel(A), -1.0))
    y = series(A, X, k -> k == 0 ? 0.0 : (-1.0)^(k + 1) / k; order)
    resid = maximum(abs.(coeffs(nexp(A, y; order = verify_order)).- x.c))
    resid < 1e-6 ? (y, resid) : (Nel(y.c, y.flag | INEXACT), resid)
end

"""1/x WITHOUT a divider: the all-ones tape Σ u^k = (1−u)⁻¹ with u = 1 − x (converges for
   ‖u‖ < 1), verified TWO-SIDED (x·y ≈ 1 AND y·x ≈ 1 — left and right inverse can differ
   in a non-commutative algebra, so both are checked).  Inverse ⇒ candidate: a zero divisor
   (or any x outside the basin) fails verification and is flagged INEXACT — the series
   diverges honestly instead of returning a lie.  This is division rebuilt from the same
   cells as everything else: one more coefficient tape on the one skeleton."""
function ninv(A::Alg, x::Nel; order::Int = 60)
    u = tadd(nel(A), tscale(x, -1.0))
    y = series(A, u, k -> 1.0; order)
    resid = max(maximum(abs.(coeffs(tmul(A, x, y)) .- A.unit)),
                maximum(abs.(coeffs(tmul(A, y, x)) .- A.unit)))
    resid < 1e-6 ? (y, resid) : (Nel(y.c, y.flag | INEXACT), resid)
end

# ================================================================ measure, don't assume
"max |(xy)z − x(yz)| over random triples — the associativity of the COMPOSED algebra"
function assoc_defect(A::Alg; trials::Int = 4, rng = nothing)
    rnd = rng === nothing ? _lcg() : rng
    worst = 0.0
    for _ in 1:trials
        x, y, z = (Nel(0.3 .* rand_vec(rnd, A.dim), 0x00) for _ in 1:3)
        l = tmul(A, tmul(A, x, y), z); r = tmul(A, x, tmul(A, y, z))
        worst = max(worst, maximum(abs.(l.c .- r.c)))
    end
    worst
end

"""max |(xx)x − x(xx)| — POWER-associativity, the true gate for single-element series:
   Cayley–Dickson scalars keep it even when non-associative (octonion, sedenion), so
   exp∘log verifies there; matrix/tensor composites can LOSE it — measure, don't assume."""
function powerassoc_defect(A::Alg; trials::Int = 4, rng = nothing)
    rnd = rng === nothing ? _lcg() : rng
    worst = 0.0
    for _ in 1:trials
        x = Nel(0.3 .* rand_vec(rnd, A.dim), 0x00)
        x2 = tmul(A, x, x)
        worst = max(worst, maximum(abs.(tmul(A, x2, x).c .- tmul(A, x, x2).c)))
    end
    worst
end
"max |xy − yx| — commutativity of the composed algebra (the third probe)"
function commut_defect(A::Alg; trials::Int = 4, rng = nothing)
    rnd = rng === nothing ? _lcg() : rng
    worst = 0.0
    for _ in 1:trials
        x, y = (Nel(0.3 .* rand_vec(rnd, A.dim), 0x00) for _ in 1:2)
        worst = max(worst, maximum(abs.(tmul(A, x, y).c .- tmul(A, y, x).c)))
    end
    worst
end

mutable struct _LCG; s::UInt64; end
_lcg() = _LCG(0x9E3779B97F4A7C15)
function rand_vec(g::_LCG, n)
    v = zeros(n)
    for i in 1:n
        g.s = g.s * 6364136223846793005 + 1442695040888963407
        v[i] = (Float64(g.s >> 11) / 2.0^53) * 2 - 1
    end
    v
end

# ================================================================ self-test
function self_test()
    println("NestedSeries — every combination measured, none assumed")
    combos = [
        cd_alg(2), cd_alg(4), cd_alg(8), cd_alg(16), cyclic_alg(6), matn_alg(2),
        mat_over(cd_alg(4), 2), mat_over(cd_alg(16), 2),
        tensor(cd_alg(4), cd_alg(4)), tensor(cd_alg(8), cd_alg(2)),
        mat_over(tensor(cd_alg(4), cd_alg(2)), 2),          # free recursion: mat(H⊗C)
    ]
    println(rpad("algebra", 26), rpad("dim", 6), rpad("assoc", 9), rpad("pow-assoc", 11),
            rpad("exp(0)=1", 10), rpad("exp∘log", 12), "verdict")
    for A in combos
        g = _lcg()
        ad = assoc_defect(A; rng = g)
        pa = powerassoc_defect(A; rng = g)
        e0ok = maximum(abs.(coeffs(nexp(A, Nel(zeros(A.dim), 0x00))) .- A.unit)) < 1e-12
        x = Nel(0.25 .* rand_vec(g, A.dim), 0x00)
        xnear = tadd(nel(A), tscale(x, 0.5))
        _, resid = nlog(A, xnear)
        verdict = resid < 1e-6 ? "✓ inverse pair" : "✗ INEXACT (structural)"
        println(rpad(A.name, 26), rpad(string(A.dim), 6),
                rpad(ad < 1e-9 ? "✓" : "✗", 9),
                rpad(pa < 1e-9 ? "✓" : "✗ $(round(pa, sigdigits=2))", 11),
                rpad(e0ok ? "✓" : "✗", 10),
                rpad(string(round(resid, sigdigits = 2)), 12), verdict)
        @assert e0ok
        # measured law: exp∘log verifies iff POWER-associativity holds (not full
        # associativity — octonion/sedenion scalars are the counterexample that
        # falsified the naive "assoc ⟺ verify" version of this assertion)
        @assert (pa < 1e-9) == (resid < 1e-6) "pow-assoc/verify mismatch on $(A.name)"
    end
    # brackets: agree on scalar cells, split for matrices of non-associative cells
    g = _lcg()
    x16 = Nel(0.3 .* rand_vec(g, 16), 0x00)
    dscalar = maximum(abs.(coeffs(nexp(cd_alg(16), x16)) .-
                           coeffs(nexp(cd_alg(16), x16; bracket = :right))))
    Am = mat_over(cd_alg(16), 2)
    xm = Nel(0.15 .* rand_vec(g, Am.dim), 0x00)
    dmat = maximum(abs.(coeffs(nexp(Am, xm)) .- coeffs(nexp(Am, xm; bracket = :right))))
    dss  = maximum(abs.(coeffs(nexp(Am, xm)) .- coeffs(nexp_ss(Am, xm))))
    println("brackets — scalar cd16 left vs right: ", round(dscalar, sigdigits = 2),
            " (agree)   mat2⟨cd16⟩ left vs right: ", round(dmat, sigdigits = 2),
            "  vs sqring: ", round(dss, sigdigits = 2), " (distinct exps)")
    @assert dscalar < 1e-9 && dmat > 1e-6
    # totality: NaN/huge input crashes nothing, names everything
    bad = nel(cd_alg(16), [NaN; fill(1e308, 15)])
    r = nexp(cd_alg(16), bad)
    @assert flagof(bad) & SING != 0 && all(isfinite, coeffs(r))
    println("totality: NaN/1e308 input → flags ", string(flagof(bad), base = 2),
            ", exp stays finite ✓")
    # ninv: division rebuilt as a tape — verified two-sided, INEXACT on zero divisors
    A16 = cd_alg(16); g2 = _lcg()
    xr = tadd(nel(A16), tscale(Nel(0.3 .* rand_vec(g2, 16), 0x00), 1.0))
    yinv, r1 = ninv(A16, xr)
    @assert r1 < 1e-6 && (flagof(yinv) & INEXACT) == 0
    zd = zeros(16); zd[4] = 1.0; zd[11] = 1.0                 # 1−x = e3+e10 zero divisor
    ybad, r2 = ninv(A16, tadd(nel(A16), tscale(nel(A16, zd), -1.0)))
    @assert (flagof(ybad) & INEXACT) != 0
    println("ninv: (1/x)·x = x·(1/x) = 1 at ", round(r1, sigdigits = 2),
            " ✓ ; zero-divisor → INEXACT ✓ (division as a tape, no divider)")
    # measured tensor law: a non-associative base keeps power-associativity under ⊗
    # ONLY when the partner is commutative AND associative — either alone fails.
    # (jordan(cd8) is the pincer: commutative ✓, associative ✗ → still loses it.)
    for (partner, keeps) in ((cyclic_alg(3), true), (cd_alg(4), false), (jordan(cd_alg(8)), false))
        T = tensor(cd_alg(8), partner); gt = _lcg()
        pa = powerassoc_defect(T; rng = gt)
        @assert (pa < 1e-9) == keeps "tensor law violated on $(T.name)"
        xn = tadd(nel(T), tscale(Nel(0.25 .* rand_vec(gt, T.dim), 0x00), 0.5))
        _, res = nlog(T, xn)
        @assert (pa < 1e-9) == (res < 1e-6) "pow-assoc/verify mismatch on $(T.name)"
    end
    println("tensor law: ⊗-partner must be commutative AND associative to preserve",
            " power-associativity (jordan pincer: commutative alone fails) ✓")
    # order machinery: exact split ab = a∘b + ½[a,b]; Jacobi and BCH gates measured
    for M in (4, 16)
        Ao = cd_alg(M); go = _lcg()
        a = Nel(0.4 .* rand_vec(go, M), 0x00); b = Nel(0.4 .* rand_vec(go, M), 0x00)
        Aj = jordan(Ao); Al = lie(Ao)
        recon = tadd(_tot(rawmul(Aj, a.c, b.c), 0x00), _tot(rawmul(Al, a.c, b.c), 0x00))
        @assert maximum(abs.(recon.c .- tmul(Ao, a, b).c)) < 1e-12
    end
    jac(A, x, y, z) = tadd(tadd(commutator(A, commutator(A, x, y), z),
                                commutator(A, commutator(A, y, z), x)),
                           commutator(A, commutator(A, z, x), y))
    jd = Dict{Int,Float64}()
    for M in (4, 8, 16)
        Ao = cd_alg(M); go = _lcg()
        x, y, z = (Nel(0.4 .* rand_vec(go, M), 0x00) for _ in 1:3)
        jd[M] = maximum(abs.(jac(Ao, x, y, z).c))
    end
    @assert jd[4] < 1e-12 && jd[8] > 1e-3 && jd[16] > 1e-3
    println("order split ab = a∘b + ½[a,b] exact ✓ ; Jacobi: cd4 ✓ Lie, cd8/cd16 ✗ (Malcev)")
    # BCH repair gate by scaling exponent: s⁴ (repaired) through octonions — Artin's
    # theorem measured — s³ (structural) at sedenions
    ratios = Dict{Int,Float64}()
    for M in (4, 8, 16)
        Ao = cd_alg(M); go = _lcg()
        ba = rand_vec(go, M); bb = rand_vec(go, M)
        r = Float64[]
        for s in (0.2, 0.1)
            a = Nel(s .* ba, 0x00); b = Nel(s .* bb, 0x00)
            lhs = tmul(Ao, nexp(Ao, a), nexp(Ao, b))
            zc = tadd(tadd(a, b), tscale(commutator(Ao, a, b), 0.5))
            zc = tadd(zc, tadd(tscale(commutator(Ao, a, commutator(Ao, a, b)), 1 / 12),
                               tscale(commutator(Ao, b, commutator(Ao, b, a)), 1 / 12)))
            push!(r, maximum(abs.(coeffs(lhs) .- coeffs(nexp(Ao, zc)))))
        end
        ratios[M] = r[1] / r[2]
    end
    @assert ratios[4] > 12 && ratios[8] > 12 && ratios[16] < 10
    println("BCH gate: cd4 ", round(ratios[4], sigdigits = 3), " / cd8 ",
            round(ratios[8], sigdigits = 3), " ≈ s⁴ repaired (Artin measured) ; cd16 ",
            round(ratios[16], sigdigits = 3), " ≈ s³ structural break ✓")
    # tape user-extensibility: a custom tape (Bessel-ish) runs on any Alg unchanged
    j0 = series(cd_alg(4), Nel(0.3 .* rand_vec(g, 4), 0x00),
                k -> iseven(k) ? Float64((-1)^(k ÷ 2) / (factorial(big(k ÷ 2))^2 * big(2)^k)) : 0.0)
    @assert all(isfinite, coeffs(j0))
    println("custom tape (user-defined coefficients) on cd4 ✓ — O layer is open, not an enum")
    println("done: cells × combinators × tapes compose freely; laws measured per combination")
end

end # module

if abspath(PROGRAM_FILE) == @__FILE__
    NestedSeries.self_test()
end
