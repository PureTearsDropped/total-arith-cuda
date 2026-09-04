# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
# audit_flags.jl — SEMANTIC flag oracle for ScalarTot's operations.
#
# The flag vocabulary makes CLAIMS: GE ⇒ |true| ≥ |shown|, LE ⇒ |true| ≤ |shown|,
# no SUNK ⇒ the shown sign is the true sign, CPLX ⇒ the true result left ℝ, and NO flag ⇒
# the shown value IS the true value (up to rounding).  This audit falsifies them: for each
# input it enumerates admissible true values t, computes the exact f(t) (BigFloat, so the
# extremes of Float64 are not the oracle's own overflow), and checks the OUTPUT's claims.
# (The five 2026-07-20 external-audit lies — sqrt(-1)→0, log(-1) silent, exp direction,
# sin/cos passthrough, negative-power direction — are all caught by exactly this check; it
# is a permanent regression.)
#
# 2026-09-03: three more classes, found while defining exp/log for MultiU32:
#   • underflow to an unflagged 0 — Float64 exhausts its subnormals and returns an exact 0,
#     which `_sat` could not tell from a true zero: exp(−MAX), 1e-200·1e-200, 1e-200/1e200,
#     0.5^2000 all said "0" (the true values are positive).  The oracle now checks unflagged
#     outputs too (an unflagged 0 claims "= 0") and feeds the extremes.
#   • ℂ was a dead end — the placeholder 0 of a ⟦ℂ⟧ value was consumed as a true zero by the
#     next operation (exp(log −1) → 1, log(−1)·2 → 0, √−4·√−4 → 0).  ℂ inputs are fed now.
#   • the oracle never fed binary operations at all.
# Definitions (the reserved word 0): a/0 = 0, 0·x = 0, log 0 = 0, 0^−n = 0 — these are
# checked as definitions, not as limits (the limit case belongs to ε = ±MIN⟦≤⟧).
#
#   julia audit_flags.jl    → expects: 違反 0

include(joinpath(@__DIR__, "ScalarTot.jl")); using .ScalarTot

const GEb, LEb, SUNKb, CPLXb = 0x01, 0x02, 0x04, 0x08
const MAXF = floatmax(Float64); const MINF = floatmin(Float64)
setprecision(BigFloat, 320)

viol = 0; total = 0

"admissible true values for a displayed (v, f) — the set the input flag CLAIMS (BigFloat)"
function truths(v::Float64, f::UInt8)
    (f & CPLXb) != 0 && return Any[:cplx]                 # the true value is not a real number
    ge, le, sunk = (f & GEb) != 0, (f & LEb) != 0, (f & SUNKb) != 0
    a = abs(big(v))
    mags = if ge && le
        [big(0), big(0.3), a, 2a + 1, big(1e5) * (a + 1)]
    elseif ge
        [a, big(1.5) * a + big(1e-9), 3a + 1, big(1e5) * (a + 1)]
    elseif le
        a == 0 ? [big(0)] : [a, a / 2, a / 100, a / big(1e100)]
    else
        [a]
    end
    signs = (sunk || v == 0) ? [big(1), big(-1)] : [big(sign(v))]
    unique(Any[s * m for s in signs for m in mags])
end

"""does the OUTPUT (v2, f2) admit the exact result u?
u ∈ BigFloat (finite or ±Inf) | :cplx / nothing (f(t) ∉ ℝ) | :tiny (a positive real below MIN)"""
function admits(v2::Float64, f2::UInt8, u)
    (u === nothing || u === :cplx) && return (f2 & CPLXb) != 0     # left ℝ: must be named CPLX
    ge, le, sunk = (f2 & GEb) != 0, (f2 & LEb) != 0, (f2 & SUNKb) != 0
    if u === :tiny                                         # 0 < true < MIN: the claim set must reach it
        (!ge && !le) && return false                       #   "= v2" — no Float64 v2 is there
        (ge && !le) && return v2 == 0                      #   "|true| ≥ |v2|" — only the dangerous zero
        return sunk || v2 >= 0                             #   LE / no bound: sign must not say negative
    end
    isnan(u) && return false
    a2 = abs(big(v2)); au = abs(u)
    if !ge && !le                                          # no flag: "the shown value is the value"
        au == 0 && return v2 == 0
        isinf(u) && return false
        abs(u - big(v2)) <= au * 1e-9 || return false
    end
    # convention: GE|LE together = "no bound either way" — no magnitude claim at all
    ge && !le && !(au >= a2 * (1 - 1e-9) - big(1e-300)) && return false
    le && !ge && !(au <= a2 * (1 + 1e-9) + big(1e-300)) && return false
    if !sunk && v2 != 0 && u != 0 && sign(u) != sign(v2)
        return false                                       # sign claimed trusted but wrong
    end
    true
end

function report(name, r, t, u)
    global viol += 1
    println("  違反: $name → $r  だが真値 $(t) → $(u)")
end

function chk(name, fn, exact, v, f)
    global total
    a = TotNum(v, f)
    r = fn(a)
    for t in truths(v, f)
        total += 1
        u = t === :cplx ? :cplx : exact(t)                 # exact real result, :tiny, ±Inf, or nothing
        admits(Float64(r), ScalarTot.flag_of(r), u) || report("$name($(v)⟦$(string(f, base=2))⟧)", r, t, u)
    end
end

function chk2(name, fn, exact, v, f, w, g)
    global total
    r = fn(TotNum(v, f), TotNum(w, g))
    for t in truths(v, f), s in truths(w, g)
        total += 1
        u = (t === :cplx || s === :cplx) ? :cplx : exact(t, s)
        admits(Float64(r), ScalarTot.flag_of(r), u) || report("$name($(v)⟦$(string(f, base=2))⟧, $(w)⟦$(string(g, base=2))⟧)", r, (t, s), u)
    end
end

# exact functions on BigFloat truths; the reserved-word definitions are written as definitions
ex_exp(t)  = t < -1e6 ? :tiny : exp(t)                    # e^t below MIN — BigFloat underflows to 0 itself
ex_log(t)  = t < 0 ? nothing : (t == 0 ? big(0) : log(t)) # log 0 = 0 by definition (the reserved word)
ex_sqrt(t) = t < 0 ? nothing : sqrt(t)
ex_pow(t, y) = t == 0 ? (y < 0 ? big(0) : (y == 0 ? big(1) : big(0))) :   # 0^−n = 0, 0^0 = 1 (definitions)
               (t < 0 && !isinteger(y)) ? nothing : (t < 0 && isodd(Int(y)) ? -abs(t)^y : abs(t)^y)
ex_div(t, s) = s == 0 ? big(0) : t / s                    # a/0 = 0 by definition
ex_mul(t, s) = t * s
ex_add(t, s) = t + s
ex_sub(t, s) = t - s

vals  = [0.3, 0.7, 1.0, 2.0, pi / 2, -0.5, -2.0, 0.0,
         MAXF, -MAXF, MINF, -MINF, 1e-200, -1e-200, 1e200, 700.0, -700.0, 750.0, -750.0]
fset  = UInt8[0x00, GEb, LEb, GEb | SUNKb, LEb | SUNKb, GEb | LEb | SUNKb, CPLXb, GEb | CPLXb, GEb | LEb | SUNKb | CPLXb]

for v in vals, f in fset
    chk("sqrt", sqrt, ex_sqrt, v, f)
    chk("log",  log,  ex_log,  v, f)
    chk("exp",  exp,  ex_exp,  v, f)
    chk("sin",  sin,  t -> sin(t), v, f)
    chk("cos",  cos,  t -> cos(t), v, f)
    chk("abs",  abs,  t -> abs(t), v, f)
    chk("neg",  -,    t -> -t, v, f)
    for n in (2, 3, -1, -2, 2000)
        chk("^$n", a -> a^n, t -> ex_pow(t, n), v, f)
    end
    for y in (0.5, 2.5, -0.5)
        chk("^$y", a -> a^TotNum(y), t -> ex_pow(t, y), v, f)
    end
end
# binary operations: a smaller value grid, every flag pair
bvals = [0.3, 2.0, -2.0, 0.0, MAXF, -MAXF, MINF, 1e-200, -1e-200, 1e200]
for v in bvals, f in fset, w in bvals, g in fset
    chk2("+", +, ex_add, v, f, w, g)
    chk2("-", -, ex_sub, v, f, w, g)
    chk2("*", *, ex_mul, v, f, w, g)
    chk2("/", /, ex_div, v, f, w, g)
end
# flagged EXPONENT: output must not claim a direction it cannot prove
for fy in UInt8[GEb, LEb, GEb | SUNKb, CPLXb]
    global total
    a = TotNum(2.0, 0x00); b = TotNum(1.5, fy)
    r = a^b
    for ty in truths(1.5, fy)
        total += 1
        u = ty === :cplx ? :cplx : big(2)^ty
        admits(Float64(r), ScalarTot.flag_of(r), u) || report("2.0^(1.5⟦$(string(fy, base=2))⟧)", r, ty, u)
    end
end
# the definitions of the reserved word 0 (values, not limits)
defs = [("a/0 = 0", TotNum(3.0) / TotNum(0.0), 0.0), ("0/0 = 0", TotNum(0.0) / TotNum(0.0), 0.0),
        ("log 0 = 0", log(TotNum(0.0)), 0.0), ("exp 0 = 1", exp(TotNum(0.0)), 1.0),
        ("0·MAX⟦≥⟧ = 0", TotNum(0.0) * typemax(TotNum), 0.0), ("0^−2 = 0", TotNum(0.0)^-2, 0.0),
        ("0^0 = 1", TotNum(0.0)^0, 1.0), ("√0 = 0", sqrt(TotNum(0.0)), 0.0)]
for (nm, r, want) in defs
    global total += 1
    (Float64(r) == want && ScalarTot.flag_of(r) == 0x00) || (global viol += 1; println("  違反(定義): $nm → $r"))
end
# ε = ±MIN⟦≤⟧ carries the limits: exp(−MAX) is a positive number below MIN, never 0
lims = [("exp(−MAX) = +MIN⟦≤⟧", exp(TotNum(-MAXF)), TotNum(MINF, LEb)),
        ("exp(−MAX⟦≥⟧) = +MIN⟦≤⟧", exp(typemin(TotNum)), TotNum(MINF, LEb)),
        ("exp(MAX) = MAX⟦≥⟧", exp(TotNum(MAXF)), TotNum(MAXF, GEb)),
        ("1e-200·1e-200 = +MIN⟦≤⟧", TotNum(1e-200) * TotNum(1e-200), TotNum(MINF, LEb)),
        ("−1e-200·1e-200 = −MIN⟦≤⟧", TotNum(-1e-200) * TotNum(1e-200), TotNum(-MINF, LEb)),
        ("1e-200/1e200 = +MIN⟦≤⟧", TotNum(1e-200) / TotNum(1e200), TotNum(MINF, LEb)),
        ("0.5^2000 = +MIN⟦≤⟧", TotNum(0.5)^2000, TotNum(MINF, LEb)),
        ("(−0.5)^2001 = −MIN⟦≤⟧", TotNum(-0.5)^2001, TotNum(-MINF, LEb)),
        ("0.5^2000.0 = +MIN⟦≤⟧", TotNum(0.5)^TotNum(2000.0), TotNum(MINF, LEb)),
        ("log(MIN⟦≤⟧) = log MIN⟦≥⟧", log(TotNum(MINF, LEb)), TotNum(log(MINF), GEb)),
        ("log(MAX⟦≥⟧) = log MAX⟦≥⟧", log(typemax(TotNum)), TotNum(log(MAXF), GEb))]
for (nm, r, want) in lims
    global total += 1
    (Float64(r) == Float64(want) && ScalarTot.flag_of(r) == ScalarTot.flag_of(want)) ||
        (global viol += 1; println("  違反(ε): $nm  だが → $r"))
end

println("=" ^ 52)
println("意味論オラクル: 総チェック $total 回 / 違反 $viol")
println(viol == 0 ? "★ フラグの主張はすべて真値に対して健全 ✓" : "!! フラグが嘘をついている(上記)")
exit(viol == 0 ? 0 : 1)
