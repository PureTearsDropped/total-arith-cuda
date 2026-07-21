# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
# tbm_cross — 言語間 適合試験の Julia 脚。run_everywhere.py が 吐く ベクタ
# (f64 入力 = UInt64 ビット / 期待 = f32 UInt32 ビット + フラグ) を 読み、
# 同じ TBM プログラム (TOTALIZE×3 → BILIN evidence → AXPY) を Tbm.jl で 実行して
# 値 bit一致・フラグ bit一致を 検査する。exit 0 = 合格。
include(joinpath(@__DIR__, "Tbm.jl"))
using .Tbm
const HA = Tbm.HyperAlgebra

function read_mat_u64(lines, i0, B, M)
    m = zeros(Float64, B, M)
    for r in 1:B
        m[r, :] = reinterpret.(Float64, parse.(UInt64, split(lines[i0 + r - 1])))
    end
    m
end

lines = readlines(ARGS[1])
B, M = parse.(Int, split(lines[1]))
a = read_mat_u64(lines, 2, B, M)
b = read_mat_u64(lines, 2 + B, B, M)
c = read_mat_u64(lines, 2 + 2B, B, M)
i0 = 2 + 3B
@assert lines[i0] == "EXPECT"
ev = [parse.(UInt32, split(lines[i0 + r])) for r in 1:B]
ef = [parse.(UInt8, split(lines[i0 + B + r])) for r in 1:B]

P = Tbm.Program("cross")
Tbm.TOTALIZE!(P, :a, :in_a); Tbm.TOTALIZE!(P, :b, :in_b); Tbm.TOTALIZE!(P, :c, :in_c)
Tbm.BILIN!(P, :s, :a, :b; alg = :sedenion, honesty = :evidence)
Tbm.AXPY!(P, :s, :c)
env = Tbm.run_program(P, Dict(:in_a => a, :in_b => b, :in_c => c))

nv = 0; nf = 0
for r in 1:B, k in 1:M
    global nv, nf
    nv += reinterpret(UInt32, env[:s].val[r, k]) != ev[r][k]
    nf += env[:s].flag[r, k] != ef[r][k]
end
println("tbm_cross(julia): 値の不一致 $nv/$(B*M)  フラグの不一致 $nf/$(B*M)")
exit(nv == 0 && nf == 0 ? 0 : 1)
