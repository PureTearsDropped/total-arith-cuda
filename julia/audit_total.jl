# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
# Totality audit: does the total-arithmetic library itself ever emit NaN/Inf, throw,
# or FALSE-flag a well-defined result (the sqrt(0) class)? 720 adversarial cases.
#   julia audit_total.jl   → expects: NaN/Inf 0, exceptions 0, false-flags 0

include(joinpath(@__DIR__, "ScalarTot.jl")); using .ScalarTot
include(joinpath(@__DIR__, "HyperTranscend.jl")); using .HyperTranscend
import Random; rng = Random.MersenneTwister(0)

bad_nan = 0; bad_exc = 0; false_flag = 0; total = 0
vals_of(r) = r isa TotNum ? [Float64(r)] : (r isa Hyper ? r.c : Float64.([r]))

# ①② NaN/Inf を出さないか・例外を投げないか
function chk(name, f, arg)
    global total += 1
    try
        r = f(arg)
        if any(x -> isnan(x) || isinf(x), vals_of(r))
            global bad_nan += 1; println("  NaN/Inf: $name  arg=$arg")
        end
    catch e
        global bad_exc += 1; println("  例外: $name  arg=$arg  → ", split(sprint(showerror, e), "\n")[1])
    end
end

# ③ 誤検出(√0 型): 数学的に定義できるのに フラグを立ててないか
function chk_falseflag(name, got_flag::Bool, should_be_clean::Bool)
    global total += 1
    if should_be_clean && got_flag
        global false_flag += 1; println("  誤検出(定義できるのに旗): $name")
    end
end

println("=== ScalarTot 敵対的スカラー ===")
sp = [0.0, -0.0, 1e308, -1e308, 1e-308, floatmax(Float64), floatmin(Float64), 1.0, -1.0, 2.0]
for a in sp, b in sp
    ta, tb = TotNum(a), TotNum(b)
    for (nm, op) in (("+",+),("-",-),("*",*),("/",/)); chk(nm, x->op(x,tb), ta); end
    chk("^", x->x^tb, ta)
end
for a in sp
    ta = TotNum(a)
    for (nm,f) in (("sqrt",sqrt),("exp",exp),("log",log),("abs",abs),
                   ("^0.5",x->x^0.5),("^-1",x->x^(-1.0)),("^2.5",x->x^2.5),("sin",sin),("cos",cos))
        chk(nm, f, ta)
    end
end
huge = floatmax(Float64) * 2
for v in (NaN, Inf, -Inf, huge, 0.0, -0.0); chk("Tot(構築)", TotNum, v); end
# ScalarTot 誤検出: √0, 0^0.5, 0^2.5, sqrt(正数) は 旗なし が正しい
chk_falseflag("ScalarTot √0",   ScalarTot.isflagged(sqrt(TotNum(0.0))),   true)
chk_falseflag("ScalarTot 4^0.5", ScalarTot.isflagged(TotNum(4.0)^0.5),    true)
chk_falseflag("ScalarTot exp0",  ScalarTot.isflagged(exp(TotNum(0.0))),   true)

println("=== HyperTranscend 敵対的セデニオン ===")
cases = [zeros(16), [i∈(4,11) ? 1.0 : 0.0 for i in 1:16], 1e200 .* randn(rng,16),
         1e-200 .* randn(rng,16), [NaN; zeros(15)], [Inf; zeros(15)], randn(rng,16),
         [i==1 ? -1.0 : 0.0 for i in 1:16], [i==1 ? 2.0 : 0.0 for i in 1:16]]
for c in cases
    x = Hyper(c)
    for (nm,f) in (("hexp",hexp),("hlog",hlog),("hsqrt",hsqrt),("hsin",hsin),("hcos",hcos),
                   ("hsinh",hsinh),("hcosh",hcosh),("^2",x->x^2),("^0.5",x->x^0.5),
                   ("^2.5",x->x^2.5),("^-1",x->x^(-1.0)),("lp3",x->left_power(x,3)))
        chk(nm, f, x)
    end
    chk("left_action", y->left_action(Hyper(randn(rng,16)), y, 0.5), x)
end
# HyperTranscend 誤検出: 定義できるものに 旗/INEXACT を 立ててないか
chk_falseflag("Hyper √0",       flags(hsqrt(Hyper(zeros(16)))) != 0, true)
chk_falseflag("Hyper 0^2.5",    flags(Hyper(zeros(16))^2.5)   != 0, true)
chk_falseflag("Hyper exp(零因子)", flags(hexp(Hyper([i∈(4,11) ? 1.0 : 0.0 for i in 1:16]))) != 0, true)
# √(正の実数e0倍) は 厳密に定義できる → 旗なしのはず
chk_falseflag("Hyper √(4·e0)",  flags(hsqrt(Hyper([i==1 ? 4.0 : 0.0 for i in 1:16]))) != 0, true)

println("="^52)
println("総チェック $total 回")
println("  NaN/Inf を出した   : $bad_nan")
println("  例外を投げた       : $bad_exc")
println("  誤検出(√0型)       : $false_flag")
println((bad_nan==0 && bad_exc==0 && false_flag==0) ?
    "★ 全域 かつ 誤検出なし ✓" : "!! 問題あり(上記)")
