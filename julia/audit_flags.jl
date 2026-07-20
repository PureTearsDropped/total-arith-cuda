# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
# audit_flags.jl — SEMANTIC flag oracle for ScalarTot's transcendentals.
#
# The flag vocabulary makes CLAIMS: GE ⇒ |true| ≥ |shown|, LE ⇒ |true| ≤ |shown|,
# no SUNK ⇒ the shown sign is the true sign, CPLX ⇒ the true result left ℝ.
# This audit falsifies them: for each flagged input it enumerates admissible true
# values t, computes the exact f(t), and checks the OUTPUT's claims against it.
# (The five 2026-07-20 external-audit lies — sqrt(-1)→0, log(-1)/log(0) silent,
# exp direction, sin/cos passthrough, negative-power direction — are all caught
# by exactly this check; it is now a permanent regression.)
#
#   julia audit_flags.jl    → expects: 違反 0

include(joinpath(@__DIR__, "ScalarTot.jl")); using .ScalarTot

const GEb, LEb, SUNKb, CPLXb = 0x01, 0x02, 0x04, 0x08

viol = 0; total = 0

"admissible true values for a displayed (v, f) — the set the input flag CLAIMS"
function truths(v::Float64, f::UInt8)
    ge, le, sunk = (f & GEb) != 0, (f & LEb) != 0, (f & SUNKb) != 0
    mags = if ge && le
        [0.0, 0.3, abs(v), 2 * abs(v) + 1, 1e5 * (abs(v) + 1)]
    elseif ge
        [abs(v), 1.5 * abs(v) + 1e-9, 3 * abs(v) + 1, 1e5 * (abs(v) + 1)]
    elseif le
        abs(v) == 0 ? [0.0] : [abs(v), 0.5 * abs(v), 0.01 * abs(v)]
    else
        [abs(v)]
    end
    signs = (sunk || v == 0) ? [1.0, -1.0] : [sign(v)]
    unique([s * m for s in signs for m in mags])
end

"does the OUTPUT (v2, f2) admit the exact result u?  (u === nothing ⇒ f(t) ∉ ℝ)"
function admits(v2::Float64, f2::UInt8, u)
    u === nothing && return (f2 & CPLXb) != 0            # left ℝ: must be named CPLX
    ge, le, sunk = (f2 & GEb) != 0, (f2 & LEb) != 0, (f2 & SUNKb) != 0
    # convention: GE|LE together = "no bound either way" — no magnitude claim at all
    ge && !le && !(abs(u) >= abs(v2) * (1 - 1e-9) - 1e-300) && return false
    le && !ge && !(abs(u) <= abs(v2) * (1 + 1e-9) + 1e-300) && return false
    if !sunk && v2 != 0 && u != 0 && sign(u) != sign(v2)
        return false                                      # sign claimed trusted but wrong
    end
    true
end

function chk(name, fn, exact, v, f)
    global total, viol
    a = TotNum(v, f)
    r = fn(a)
    for t in truths(v, f)
        total += 1
        u = exact(t)                                      # exact real result or nothing
        if !admits(Float64(r), ScalarTot.flag_of(r), u)
            viol += 1
            println("  違反: $name($(v)⟦$(string(f, base=2))⟧) → $r  だが真値 t=$t → f(t)=$u")
        end
    end
end

vals  = [0.3, 0.7, 1.0, 2.0, pi / 2, -0.5, -2.0, 0.0]
fset  = UInt8[0x00, GEb, LEb, GEb | SUNKb, LEb | SUNKb, GEb | LEb | SUNKb]

for v in vals, f in fset
    chk("sqrt", sqrt, t -> t < 0 ? nothing : sqrt(t), v, f)
    chk("log",  log,  t -> t < 0 ? nothing : (t == 0 ? -Inf : log(t)), v, f)
    chk("exp",  exp,  t -> exp(t), v, f)
    chk("sin",  sin,  t -> sin(t), v, f)
    chk("cos",  cos,  t -> cos(t), v, f)
    for n in (2, 3, -1, -2)
        chk("^$n", a -> a^n, t -> (t == 0 && n < 0) ? 0.0 : Float64(t)^n, v, f)
    end
    for y in (0.5, 2.5, -0.5)
        chk("^$y", a -> a^TotNum(y),
            t -> t < 0 ? nothing : (t == 0 && y < 0) ? 0.0 : Float64(t)^y, v, f)
    end
end
# flagged EXPONENT: output must not claim a direction it cannot prove
for fy in UInt8[GEb, LEb, GEb | SUNKb]
    global total
    a = TotNum(2.0, 0x00); b = TotNum(1.5, fy)
    r = a^b
    for ty in truths(1.5, fy)
        total += 1
        u = 2.0^ty
        admits(Float64(r), ScalarTot.flag_of(r), u) || (global viol += 1;
            println("  違反: 2.0^(1.5⟦$(string(fy, base=2))⟧) → $r だが 2^$ty = $u"))
    end
end

println("=" ^ 52)
println("意味論オラクル: 総チェック $total 回 / 違反 $viol")
println(viol == 0 ? "★ フラグの主張はすべて真値に対して健全 ✓" : "!! フラグが嘘をついている(上記)")
exit(viol == 0 ? 0 : 1)
