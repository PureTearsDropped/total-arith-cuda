# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
# audit_cplx.jl — the semantic oracle of audit_flags.jl lifted to ℂ, for ScalarTotComplex.jl.
#
# A flagged complex value names a SET of admissible true values: |true| ≥ |shown| (GE), ≤ (LE),
# any (both); arg true = arg shown, or any direction (AUNK); a dangerous zero (0⟦≥⟧) is anything.
# For every operation, every input value and every flag combination the audit enumerates admissible
# true inputs, computes the true result in Complex{BigFloat} and falsifies the output's claim:
#   unflagged ⟹ the shown value IS the true value (to 1e-9 relative);  GE / LE ⟹ the magnitude bound
#   holds;  no AUNK ⟹ the shown direction IS the true direction (mod 2π, to 1e-9 in units of π).
# The reserved-word definitions (a/0 = 0, log 0 = 0, arg 0 = 0, 0^−n = 0, 0^0 = 1, …) have no true
# value to compare against and are checked as definitions: exact and unflagged.  The real axis is
# twin-checked against TotNum: same values, and never a contradiction between the two flag claims.
#
#   julia julia/audit_cplx.jl     → expects violations 0, definitions 0, contradictions 0
include(joinpath(@__DIR__, "ScalarTotComplex.jl"))
using .ScalarTot, .ScalarTotComplex
using Printf
const C = TotComplex
const GEb, LEb, AUb, CPb = 0x01, 0x02, 0x04, 0x08
const TOL = 1e-9

"""e^{iπt} in Complex{BigFloat}, exact at the quarter turns."""
function ecis(t::Float64)
    q = wrap_t(t)
    q == 0 && return Complex{BigFloat}(1, 0)
    q == 0.5 && return Complex{BigFloat}(0, 1)
    q == 1 && return Complex{BigFloat}(-1, 0)
    q == -0.5 && return Complex{BigFloat}(0, -1)
    a = BigFloat(pi) * BigFloat(q)
    Complex{BigFloat}(cos(a), sin(a))
end
wrap_t(t) = (t = rem(t, 2.0); t > 1 ? t - 2 : (t <= -1 ? t + 2 : t))
tobig(z::C) = BigFloat(z.r) * ecis(z.t)
const HUGE = BigFloat(10)^400            # "∞": beyond every finite claim
const TINY = BigFloat(MINF) / 2          # "0⁺": below every finite nonzero claim

"""admissible true values of z (a sample of the set its flags name)."""
function truths(z::C; full::Bool = true)
    f = z.flag; ge = (f & GEb) != 0; le = (f & LEb) != 0; au = (f & AUb) != 0
    r = BigFloat(z.r)
    if z.r == 0
        ge || return [Complex{BigFloat}(0, 0)]                           # a true zero
        mags = full ? [BigFloat(0), TINY, BigFloat(1e-30), BigFloat(1), BigFloat(1e30), HUGE] : [BigFloat(0), BigFloat(1), HUGE]
    elseif ge && le
        mags = full ? [BigFloat(0), TINY, r / 10, r / 2, r, 2r, 10r, r * BigFloat(1e30), HUGE] : [BigFloat(0), r / 3, r, 3r, HUGE]
    elseif ge
        mags = full ? [r, 2r, 10r, r * BigFloat(1e30), HUGE] : [r, 3r, HUGE]
    elseif le
        mags = full ? [r, r / 2, r / 10, r * BigFloat(1e-30), TINY, BigFloat(0)] : [r, r / 3, BigFloat(0)]
    else
        mags = [r]
    end
    args = au ? (full ? [z.t, z.t + 0.5, z.t + 1, z.t - 0.5, z.t + 0.123, z.t - 0.777] : [z.t, z.t + 0.37, z.t + 1]) : [z.t]
    out = Complex{BigFloat}[]
    for m in mags, a in args
        u = m * ecis(a)
        (m == 0 && !isempty(out) && any(iszero, out)) && continue
        push!(out, u)
    end
    out
end

"""does the claim z2 admit the true value u?"""
function admits(z2::C, u::Complex{BigFloat})
    f = z2.flag; ge = (f & GEb) != 0; le = (f & LEb) != 0; au = (f & AUb) != 0
    r2 = BigFloat(z2.r); mu = abs(u)
    if !ge && !le
        abs(mu - r2) ≤ TOL * mu + BigFloat(1e-300) || return false        # the magnitude is claimed exact
    elseif ge != le                                                        # (both: no bound at all)
        ge && !(mu ≥ r2 * (1 - TOL)) && return false
        le && !(mu ≤ r2 * (1 + TOL) + BigFloat(1e-300)) && return false
    end
    if !au
        if z2.r == 0
            mu ≤ BigFloat(1e-300) || return false                          # a true zero claims 0 (no direction)
        elseif mu > 0
            du = Float64(angle(u) / BigFloat(pi))                           # arg u / π
            d = abs(wrap_t(du - z2.t)); d = min(d, 2 - d)
            d ≤ TOL || return false                                        # the direction is claimed exact
        end
    end
    true
end
"""the real-valued outputs (abs, arg, real, imag): audit_flags' rule."""
function admits_real(v::TotNum, u::BigFloat)
    f = v.flag; ge = (f & GEb) != 0; le = (f & LEb) != 0; sunk = (f & AUb) != 0
    (f & CPb) != 0 && return false                                        # a real truth exists
    au = abs(u); av = BigFloat(abs(v.val))
    if !ge && !le
        abs(au - av) ≤ TOL * au + BigFloat(1e-300) || return false
    elseif ge != le
        ge && !(au ≥ av * (1 - TOL)) && return false
        le && !(au ≤ av * (1 + TOL) + BigFloat(1e-300)) && return false
    end
    if !sunk && v.val != 0 && u != 0
        (v.val < 0) == (u < 0) || return false
    end
    if !sunk && v.val == 0 && !ge
        au ≤ BigFloat(1e-300) || return false
    end
    true
end

total = 0; viol = 0; shown = 0
function report(name, z, got, u)
    global shown
    shown < 25 && (shown += 1; println("  ✗ $name($z) = $got   but a true input ", u[1], " gives ", u[2]))
end
"""unary: op on every admissible input; `truth(u)` → Complex{BigFloat}, BigFloat, or nothing (a definition)."""
function chk1(name, op, truth, z::C)
    global total, viol
    got = op(z)
    for u in truths(z)
        t = truth(u)
        t === nothing && continue
        total += 1
        ok = t isa BigFloat ? admits_real(got, t) : admits(got, t)
        ok || (viol += 1; report(name, z, got, (u, t)))
    end
end
function chk2(name, op, truth, a::C, b::C)
    global total, viol
    got = op(a, b)
    for u in truths(a; full = false), v in truths(b; full = false)
        t = truth(u, v)
        t === nothing && continue
        total += 1
        admits(got, t) || (viol += 1; report(name, (a, b), got, ((u, v), t)))
    end
end

# ---- the shelf ----------------------------------------------------------------------------------
vals = [C(1.0), C(-1.0), IM, -IM, C(1.0, 1.0), C(-1.0, 1.0), C(0.5), C(2.0), C(3.0, 4.0), C(-2.0, -0.5), C(0.3, -0.9),
        C(0.0), C(MAXF), C(-MAXF), C(MINF), polar(MINF, 0.3), C(1e-200), C(1e200), C(700.0), C(-700.0), C(0.0, 700.0),
        C(0.0, 20.0), polar(0.7, 0.999), polar(1.0, -0.25), C(1e6, 1e6)]
fset = UInt8[0x00, GEb, LEb, GEb | LEb, AUb, GEb | AUb, LEb | AUb, GEb | LEb | AUb]
withflag(z::C, f::UInt8) = C(z.r, z.t, f)
bigpow(u, y) = u^BigFloat(y)                                              # principal branch: exp(y log u)

setprecision(BigFloat, 1400) do                                            # exp / sin / cos of |Im| up to MAX need the reduction bits
    for v in vals, f in fset
        z = withflag(v, f)
        chk1("sqrt", sqrt, u -> sqrt(u), z)
        chk1("log", log, u -> u == 0 ? nothing : log(u), z)
        chk1("exp", exp, u -> abs(u) > HUGE / 2 ? nothing : exp(u), z)     # the "∞" sample has no exp
        chk1("conj", conj, u -> conj(u), z)
        chk1("neg", -, u -> -u, z)
        chk1("inv", inv, u -> u == 0 ? nothing : 1 / u, z)
        chk1("abs", abs, u -> abs(u), z)
        chk1("arg", arg, u -> u == 0 ? nothing : angle(u), z)
        chk1("real", real, u -> real(u), z)
        chk1("imag", imag, u -> imag(u), z)
        chk1("sin", sin, u -> abs(u) > HUGE / 2 ? nothing : sin(u), z)
        chk1("cos", cos, u -> abs(u) > HUGE / 2 ? nothing : cos(u), z)
        for n in (2, 3, -1, -2)
            chk1("^$n", w -> w^n, u -> (u == 0 && n < 0) ? nothing : u^n, z)
        end
        for y in (0.5, 2.5, -0.5)
            chk1("^$y", w -> w^y, u -> u == 0 ? nothing : bigpow(u, y), z)
        end
    end
end
n1 = total
bvals = [C(1.0), IM, C(1.0, 1.0), C(-2.0), C(0.5, -0.5), C(0.0), C(MAXF), C(MINF), C(1e-200), C(1e200), polar(2.0, 0.4)]
setprecision(BigFloat, 400) do
    for a0 in bvals, b0 in bvals, fa in fset, fb in fset
        a = withflag(a0, fa); b = withflag(b0, fb)
        chk2("+", +, (u, v) -> u + v, a, b)
        chk2("-", -, (u, v) -> u - v, a, b)
        chk2("*", *, (u, v) -> u * v, a, b)
        chk2("/", /, (u, v) -> v == 0 ? nothing : u / v, a, b)
    end
end
println("oracle: unary $(n1) + binary $(total - n1) = $total checks, violations $viol")

# ---- the definitions: exact and unflagged ------------------------------------------------------
z0 = C(0.0); o = C(1.0)
ex(z, r, t) = z.r == r && z.t == t && z.flag == 0x00
defs = [("a/0 = 0", o / z0, z -> ex(z, 0.0, 0.0)), ("0/0 = 0", z0 / z0, z -> ex(z, 0.0, 0.0)),
        ("(2∠¼π⟦≥⟧)/0 = 0 (every admissible a)", polar(2.0, 0.25, GEb) / z0, z -> ex(z, 0.0, 0.0)),
        ("log 0 = 0", log(z0), z -> ex(z, 0.0, 0.0)), ("arg 0 = 0", arg(z0), v -> v.val == 0 && v.flag == 0x00),
        ("√0 = 0", sqrt(z0), z -> ex(z, 0.0, 0.0)), ("exp 0 = 1", exp(z0), z -> ex(z, 1.0, 0.0)),
        ("0^0 = 1", z0^0, z -> ex(z, 1.0, 0.0)), ("0^−2 = 0", z0^-2, z -> ex(z, 0.0, 0.0)), ("0^2.5 = 0", z0^2.5, z -> ex(z, 0.0, 0.0)),
        ("0^i = 0 (not exp(i·log 0) = 1)", z0^IM, z -> ex(z, 0.0, 0.0)),
        ("i·i = −1 exactly", IM * IM, z -> ex(z, 1.0, 1.0)), ("e^{iπ} + 1 = 0 exactly", exp(IM * pi) + o, z -> ex(z, 0.0, 0.0)),
        ("log(−1) = iπ exactly", log(C(-1.0)), z -> ex(z, Float64(pi), 0.5)), ("√(−1) = i", sqrt(C(-1.0)), z -> ex(z, 1.0, 0.5)),
        ("(−1)^½ = i", C(-1.0)^0.5, z -> ex(z, 1.0, 0.5)), ("(−8)^⅓ = 2∠⅓π", C(-8.0)^(1/3), z -> z.r == 2.0 && abs(z.t - 1/3) < 1e-15 && z.flag == 0),
        ("0·MAX⟦≥⟧ = 0", z0 * C(MAXF, 0.0, GEb), z -> ex(z, 0.0, 0.0)), ("0·(unknown seat) = 0", z0 * ScalarTotComplex.unknown(), z -> ex(z, 0.0, 0.0)),
        ("0 + z = z with z's flags", z0 + polar(2.0, 0.3, GEb | AUb), z -> z.r == 2.0 && z.t == 0.3 && z.flag == (GEb | AUb)),
        ("ε keeps its direction: (1e-200∠0.3π)·1e-200 = MIN∠0.3π⟦≤⟧", polar(1e-200, 0.3) * C(1e-200), z -> z.r == MINF && z.t == 0.3 && z.flag == LEb),
        ("exp(−MAX) = MIN⟦≤⟧", exp(C(-MAXF)), z -> z.r == MINF && z.t == 0 && z.flag == LEb),
        ("exp(MAX) = MAX⟦≥⟧", exp(C(MAXF)), z -> z.r == MAXF && z.flag == GEb),
        ("log ε (real) = log MIN ⟦≥⟧, direction certain", log(C(MINF, 0.0, LEb)), z -> z.r == -log(MINF) && z.t == 1.0 && z.flag == GEb),
        ("log ε∠0.3π: ⟦≥ ∠?⟧ (the direction moves with the unknown log)", log(polar(MINF, 0.3, LEb)), z -> z.flag == (GEb | AUb)),
        ("ℂ seat of TotNum → the unknown seat", C(sqrt(TotNum(-1.0))), z -> z.r == 0 && z.flag == (GEb | LEb | AUb)),
        ("√ of the unknown seat stays unknown", sqrt(ScalarTotComplex.unknown()), z -> z.flag == (GEb | LEb | AUb)),
        ("z^0 = 1 for the unknown seat", ScalarTotComplex.unknown()^0, z -> ex(z, 1.0, 0.0)),
        ("exp(i·MAX): the direction is MAX mod 2π, certain", exp(IM * MAXF), z -> z.r == 1.0 && z.flag == 0x00 && abs(z.t - Float64(rem2pi(MAXF, RoundNearest) / pi)) < 1e-12),
        ("|3+4i| = 5, arg = atan(4/3) (radians: 1 ulp from the stored arg/π)", (abs(C(3.0, 4.0)), arg(C(3.0, 4.0))), p -> p[1].val == 5.0 && abs(p[2].val - atan(4, 3)) ≤ 2e-16 && p[1].flag == 0 && p[2].flag == 0)]
nd = 0
for (name, r, ok) in defs
    ok(r) || (global nd += 1; println("  ✗ definition: $name → $r"))
end
println("definitions: $(length(defs)) checked, violations $nd")

# ---- the real axis against TotNum ----------------------------------------------------------------
# value: the same number (a TotNum in the ℂ seat has no value to compare — the complex side must then
# hold a value: "cashed in").  flags: identical / complex weaker (its set ⊇ TotNum's: the bits ⊇) /
# complex stronger (⊆) / contradiction (neither) — a contradiction would mean one of the two lies.
rvals = [0.3, 0.7, 1.0, 2.0, -0.5, -2.0, 0.0, MAXF, -MAXF, MINF, -MINF, 1e-200, -1e-200, 1e200, 700.0, -700.0]
ident = Dict{String,Int}(); weaker = Dict{String,Int}(); stronger = Dict{String,Int}(); contra = 0; vmis = 0; cashed = 0; cashed0 = 0; certain_ident = 0; certain_n = 0
examples = Dict{Tuple{String,String},Vector{String}}()
function note(cat, name, rr, zz, a)
    v = get!(examples, (cat, name), String[])
    length(v) < 2 && push!(v, "$(a) → TotNum $rr / ℂ $zz")
end
function twin(name, rr::TotNum, zz::C, signcertain::Bool, a)
    global contra, vmis, cashed, cashed0, certain_ident, certain_n
    if (rr.flag & CPb) != 0                                                             # TotNum could only name the seat
        zz.r > 0 && (cashed += 1; return)                                               # ℂ delivers the value
        (zz.flag & (GEb | LEb | AUb)) == (GEb | LEb | AUb) && return                    # unknown on both sides
        zz.flag == 0x00 && (cashed0 += 1; return)                                       # an exact 0 (0·z = 0 for every z ∈ ℂ, where ℝ's ℂ sticks)
        contra += 1; println("  ✗ twin $name: TotNum $rr (ℂ seat) but complex $zz"); return
    end
    rc = C(rr)
    nob = (rr.flag & (GEb | LEb)) == (GEb | LEb) && (zz.flag & (GEb | LEb)) == (GEb | LEb)   # both: the number is a placeholder
    vok = nob || (zz.r == 0 && rc.r == 0) || (rc.r > 0 && abs(zz.r - rc.r) ≤ 1e-12 * rc.r && ((zz.flag & AUb) != 0 || zz.t == rc.t))
    vok || (vmis += 1; vmis ≤ 10 && println("  ✗ twin value $name: TotNum $rr vs complex $zz"))
    fr = rc.flag; fz = zz.flag
    if fr == fz
        ident[name] = get(ident, name, 0) + 1; signcertain && (certain_ident += 1)
    elseif (fz & fr) == fr
        weaker[name] = get(weaker, name, 0) + 1; note("weaker", name, rr, zz, a)
    elseif (fz & fr) == fz
        stronger[name] = get(stronger, name, 0) + 1; note("stronger", name, rr, zz, a)
    else
        contra += 1; println("  ✗ twin contradiction $name: TotNum $rr vs complex $zz")
    end
    signcertain && (certain_n += 1)
end
uops = [("sqrt", sqrt), ("log", log), ("exp", exp), ("abs", abs), ("neg", -), ("inv", inv), ("sin", sin), ("cos", cos),
        ("^2", x -> x^2), ("^3", x -> x^3), ("^-1", x -> x^-1), ("^-2", x -> x^-2), ("^0.5", x -> x^0.5), ("^2.5", x -> x^2.5), ("^-0.5", x -> x^-0.5)]
for v in rvals, f in 0x00:0x0f
    a = TotNum(v, f); A = C(a)
    for (name, op) in uops
        rr = op(a); zz = op(A)
        rr isa TotNum && (zz isa TotNum ? twin(name, rr, C(zz), f ∈ (0x00, GEb, LEb, GEb | LEb), a) : twin(name, rr, zz, f ∈ (0x00, GEb, LEb, GEb | LEb), a))
    end
end
bops = [("+", +), ("-", -), ("*", *), ("/", /)]
for v in rvals[1:12], w in rvals[1:12], fa in 0x00:0x0f, fb in 0x00:0x0f
    a = TotNum(v, fa); b = TotNum(w, fb); A = C(a); B = C(b)
    sc = fa ∈ (0x00, GEb, LEb, GEb | LEb) && fb ∈ (0x00, GEb, LEb, GEb | LEb)
    for (name, op) in bops
        twin(name, op(a, b), op(A, B), sc, (a, b))
    end
end
println("real axis vs TotNum: value mismatches $vmis, contradictions $contra, ℂ seats cashed in as a value $cashed, as an exact 0 $cashed0")
println("  identical on sign-certain inputs: $certain_ident of $certain_n")
for (name, _) in vcat(uops, bops)
    @printf("  %-6s identical %6d   complex weaker %6d   complex stronger %6d\n", name, get(ident, name, 0), get(weaker, name, 0), get(stronger, name, 0))
    for cat in ("weaker", "stronger"), e in get(examples, (cat, name), String[])
        println("           $cat: $e")
    end
end
ok = viol == 0 && nd == 0 && contra == 0 && vmis == 0
println(ok ? "PASS: ℂ の旗はすべて真値に対して健全・定義則成立・実軸は TotNum と矛盾なし" : "FAIL: violations $viol, definitions $nd, contradictions $contra, value mismatches $vmis")
exit(ok ? 0 : 1)
