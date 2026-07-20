# julia/ — three modules, three jobs

All three are standalone (stdlib only) and carry the ⚠️ AI-assisted banner. Run each with
`julia <file>` for its self-test.

| file | module | what it is |
|---|---|---|
| **`TotalArith.jl`** | `TotalArith` | Array/batch total arithmetic **+ swappable wiring tensor** (`group_mul`) — the Julia mirror of `cuda_total.py`, written generically so it is `CuArray`-ready (GPU). Use this for the same thing the Python library does. |
| **`TotArith.jl`** | `TotArith` | Total arithmetic **as a Julia `Number`**: `TotNum <: Real` overloads `+ - * / ^ exp log ...`, so **existing generic code runs on it unchanged** — an ODE solver from OrdinaryDiffEq.jl solves with `TotNum` and the flag names *where/which-direction* the computation left the machine's representable range. This is the "used, not demo" bridge. |
| **`HyperTot.jl`** | `HyperTot` | Analytic functions (`exp`, `log`, `sqrt`, `^`) for a hypercomplex number of **any** dimension M = 2^k (real → complex → quaternion → octonion → sedenion), all through one recipe: `f(x) = f(L_x)·e₀`, the matrix function of the regular representation. **Forward ops (`*`, `exp`) are total for every input, zero divisors included; only inverse-type ops (`/ log √ x^neg`) can break — at exactly one place, `L_x` singular (a zero divisor) — and there it names the value with a `⟦零因子⟧` flag instead of emitting NaN.** |

**The through-line.** `TotNum` (scalar) is the M = 1 case of `Hyper` (any M): a hypercomplex
number is its left-multiplication matrix `L_x`, and every operation — product, power, exp,
log, sqrt — is one recipe with M and the wiring table as the only knobs. Forward computation
is uniformly total across all M; the single place anything breaks is *inversion*, and it
breaks the same way everywhere (`L_x` singular). The flag is not "the algebra broke" — it is
"you asked for an inverse that has no unique answer." (Scalars/complex/quaternion/octonion
are division algebras and never hit it; zero divisors first appear at M = 16, the sedenions.)
