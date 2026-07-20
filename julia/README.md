# julia/ — three modules: basic algebra / transcendental / scalar

All three are standalone (stdlib only) and carry the ⚠️ AI-assisted banner. Run each with
`julia <file>` for its self-test.

| file | module | what it is |
|---|---|---|
| **`HyperAlgebra.jl`** | `HyperAlgebra` | Array/batch total arithmetic **+ swappable wiring tensor** (`group_mul`) — the Julia mirror of `cuda_total.py`, written generically so it is `CuArray`-ready (GPU). Use this for the same thing the Python library does. |
| **`ScalarTot.jl`** | `ScalarTot` | Total arithmetic **as a Julia `Number`**: `TotNum <: Real` overloads `+ - * / ^ exp log ...`, so **existing generic code runs on it unchanged** — an ODE solver from OrdinaryDiffEq.jl solves with `TotNum` and the flag names *where/which-direction* the computation left the machine's representable range. This is the "used, not demo" bridge. |
| **`HyperTranscend.jl`** | `HyperTranscend` | Analytic functions (`exp`, `log`, `sqrt`, `^`) for a hypercomplex number of **any** dimension M = 2^k (real → complex → quaternion → octonion → sedenion), all through one recipe: `f(x) = f(L_x)·e₀`, the matrix function of the regular representation. **Forward ops (`*`, `exp`) are total for every input, zero divisors included; only inverse-type ops (`/ log √ x^neg`) can break — at exactly one place, `L_x` singular (a zero divisor) — and there it names the value with a `⟦零因子⟧` flag instead of emitting NaN.** |

## `demo_ode_blowup.jl` — a third-party solver, unchanged, naming the blow-up

Run `julia demo_ode_blowup.jl` (needs `OrdinaryDiffEq`, which you install yourself —
**it is not vendored here**; the demo only *calls* it). Solving `du/dt = u²` on `[0,2]`
(true solution `1/(1−t)`, ∞ at `t=1`):

```
Float64 : retcode = Unstable      ← aborts "dt below eps / unstable"; where? which? unknown
TotNum  : retcode = Success       ← same model, same solver, only u0::TotNum
          ★ first flag at t = 0.99998  →  ...e292 ⟦≥≤±⟧
          (named just before the true blow-up t = 1.0, no NaN; finishes holding ±MAX)
```

This is the forum pain answered: a NaN/instability that Float64 reports as an opaque abort
becomes, by *switching the number type* (not editing the solver or the model), a named
event — the step and direction at which the run left representable range — flowing straight
through the external library's internals. This is what `ScalarTot`'s `Number` interface buys
that Python cannot.

**The through-line.** `TotNum` (scalar) is the M = 1 case of `Hyper` (any M): a hypercomplex
number is its left-multiplication matrix `L_x`, and every operation — product, power, exp,
log, sqrt — is one recipe with M and the wiring table as the only knobs. Forward computation
is uniformly total across all M; the single place anything breaks is *inversion*, and it
breaks the same way everywhere (`L_x` singular). The flag is not "the algebra broke" — it is
"you asked for an inverse that has no unique answer." (Scalars/complex/quaternion/octonion
are division algebras and never hit it; zero divisors first appear at M = 16, the sedenions.)
