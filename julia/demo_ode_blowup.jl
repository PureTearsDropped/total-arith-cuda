# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
#
# demo: a THIRD-PARTY ODE solver, unchanged, run on the total-arithmetic Number —
#       when the solution blows up it names where, instead of dying "unstable".
#
# Needs OrdinaryDiffEq (install it yourself; it is NOT vendored here):
#     julia> import Pkg; Pkg.add("OrdinaryDiffEq")
#     julia  demo_ode_blowup.jl
#
# The pain (real, from the Julia forum): a solver hits NaN / "dt <= dtmin" and
# aborts, but you cannot tell *where* or *which variable* went bad. Switching the
# initial value to TotNum — the model and the solver are untouched — makes the
# solver flow through this total-arithmetic type, and the flag names the step and
# direction at which the run left the machine's representable range.

include(joinpath(@__DIR__, "ScalarTot.jl")); using .ScalarTot
using OrdinaryDiffEq

blowup(u, p, t) = u * u                    # du/dt = u²  →  true solution 1/(1−t), ∞ at t=1

println("du/dt = u²  on [0, 2]   (blows up at t = 1)")
println("="^60)

sf = solve(ODEProblem(blowup, 1.0, (0.0, 2.0)), Tsit5())
println("Float64 : retcode = $(sf.retcode)   (aborts 'unstable' — where? which? unknown)")

st = solve(ODEProblem(blowup, TotNum(1.0), (TotNum(0.0), TotNum(2.0))), Tsit5())
hit = findfirst(ScalarTot.isflagged, st.u)
println("TotNum  : retcode = $(st.retcode)   (same model, same solver, u0::TotNum only)")
if hit !== nothing
    println("          ★ first flag at t = $(round(Float64(st.t[hit]), digits=5)) → $(st.u[hit])")
    println("            (named just before the true blow-up t = 1.0, no NaN)")
end
println("          flagged steps: $(count(ScalarTot.isflagged, st.u))/$(length(st.u)); " *
        "final $(st.u[end]) — finished, holding the out-of-range value")
