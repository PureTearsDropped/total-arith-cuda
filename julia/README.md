# julia/ — three modules: basic algebra / transcendental / scalar

All three are standalone (stdlib only) and carry the ⚠️ AI-assisted banner. Run each with
`julia <file>` for its self-test.

| file | module | what it is |
|---|---|---|
| **`HyperAlgebra.jl`** | `HyperAlgebra` | Array/batch total arithmetic **+ swappable wiring tensor** (`group_mul`) — the Julia mirror of `cuda_total.py`, written generically so it is `CuArray`-ready (GPU). Use this for the same thing the Python library does. |
| **`ScalarTot.jl`** | `ScalarTot` | Total arithmetic **as a Julia `Number`**: `TotNum <: Real` overloads `+ - * / ^ exp log ...`, so **existing generic code runs on it unchanged** — an ODE solver from OrdinaryDiffEq.jl solves with `TotNum` and the flag names *where/which-direction* the computation left the machine's representable range. This is the "used, not demo" bridge. |
| **`NestedSeries.jl`** | `NestedSeries` | The M/N/O layers as **freely composable registries** over one `Alg` interface: cells (`cd_alg`/`cyclic_alg`/`matn_alg`) × combinators (`mat_over`, `tensor`, `jordan` (symmetrized a∘b=(ab+ba)/2) — recursive: `mat_over(tensor(cd_alg(4),cd_alg(2)),2)` just works) × coefficient tapes (`:exp :sin :cos :sinh :cosh` + user-defined). Nothing assumed per combination: `assoc_defect`/`powerassoc_defect`/`commut_defect` measure the composed algebra, `nlog` verifies with forward exp or flags INEXACT. Measured law: **exp∘log verifies iff POWER-associativity holds** — octonion/sedenion scalars (non-associative, power-associative) verify at 1e-16; `mat2⟨cd16⟩` loses power-associativity (0.97) and breaks structurally (7e-3). Second measured law: a non-associative ⊗-base keeps power-associativity **only when the partner is commutative AND associative** — the `jordan` pincer (commutative, non-associative) shows commutativity alone fails. `ninv` = division rebuilt as the all-ones tape Σu^k, verified two-sided, INEXACT on zero divisors — no divider circuit. **`nsolve_left`/`nsolve_right`** = equation-solving division `L_a⁺x`/`R_a⁺x` (the full-rank completion of the founding axiom: a/0=0 IS Moore–Penrose at 1×1): multiplication-only Ben-Israel iteration `X←X(2I−AX)` — no pivots, no branches, and **zero divisions** (the initial scale rounds up to a power of two = an exponent shift, the same trick as `2^{-s}` in the exp unit), so solve is cells end-to-end. Two-tier verification: forward residual → exact solution (clean), normal-equation residual → least-squares (flagged SING — never pretends an inconsistent system was solved), unconverged → INEXACT. Measured: exact solves at 1e-16 (stdlib-pinv two-witness), left≠right (1.3 apart), zero divisor e3+e10 range-in exact / range-out honestly SING, and the **conj-div boundary**: `ā·x/|a|²` IS the solution of ay=x through the octonions and fails at sedenions (residual 0.84) while nsolve stays 1e-16 — mechanism measured as **Hurwitz's theorem** (conjugate=transpose holds at every CD dim; the composition property `L_aᵀL_a=|a|²I` holds exactly for dims 1,2,4,8 and breaks at 16). The review's full **division family of five** is on the shelf: `nconj_div_left/right` (the always-computable algebraic formula — VERIFIED against the equation and flagged INEXACT when merely formal, i.e. beyond the Hurwitz four), `nsolve_left/right` (equation-solving `L⁺/R⁺`), `nnorm_div` (‖x‖/‖a‖, with norm-multiplicativity holding in ℍ and measurably failing in sedenions — the zero-divisor consequence), `nnormalize` (`a/‖a‖`, real-scalar division only, 0→0), plus `ninv` (the geometric tape). Two preset shelves: `OPS`/`nop(A,:sqrt,x)` (operators as data: forward vs candidate+verify) and `ALGS`/`alg(:dualquat)` (famous algebras as compositions — dual numbers Λ1 give forward-mode AD `f(a+ε)=f(a)+f′(a)ε`; `:dualquat`=Λ1⊗ℍ is the rigid-body pose algebra; Grassmann/Clifford cells included), with `list_ops`/`list_algs` printing measured id-cards. |
| **`HyperTranscend.jl`** | `HyperTranscend` | **Experimental** unified computation of `exp`/`log`/`sqrt`/`^` for any M = 2^k via `f(x) = f(L_x)·e₀` (function values through the left regular representation — *not* a proof that every hypercomplex analytic function is captured; identities are **checked per dimension** in `self_test()`). Forward ops (`*`, `exp`, `x^{p≥0}`, `√`) stay total for every input incl. zero divisors — `√0 = 0` even though `L_0` is singular.  Only genuine **inversion** (`log`, `x^{neg}`) needs `L_x` nonsingular; a zero divisor there is named `⟦零因子⟧`. **Safe forward group** (`exp sin cos sinh cosh`, `x^{p≥0}` via `left_power` with explicit bracketing, `left_action(a,x0,t)=exp(t·L_a)·x0` for sedenion-valued linear ODEs) is total for every input. **Candidate group** (`sqrt log x^{frac}`) computes then **verifies the defining identity by a non-recursive residual** — trusted only if it holds, else flagged `⟦INEXACT⟧` (never a silent lie); `verify_sqrt`/`verify_log` are exposed for the caller. |

## `TotalPipeline.jl` — U → V(O,N,M) → W, named (Julia twin of `total_pipeline.py`)

The audited `HyperAlgebra` core already runs as U (entry totalization) → V (fused Float64
MAC) → W (saturate-once + pattern-rule flags), with N (the wiring tensor) swappable and M
implicit. This module names that architecture without touching the core: `papply(op, a, b;
algebra=…)` is the one gateway (asserted bit-identical to direct kernel calls),
`from_kind`/`from_registry` fill the N slot — the latter bridges the **NestedSeries ALGS
shelf** (dualquat, Clifford, Grassmann, …) onto the audited kernel with flags — `Lmatrix`
exposes the explicit M (implicit ≡ explicit asserted), `TotalPipe(:gmul, algebra)` is the
declarative pair. `julia TotalPipeline.jl` runs the five-way self-test.

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

**The through-line.** `TotNum` (scalar) is the M = 1 case of `Hyper` (any M): function
values are computed through the **left regular action** — `f(x) := f(L_x)·e₀`, where `L_x`
is the left-multiplication matrix. Precision matters here: for non-associative M,
`L_{xy} ≠ L_x·L_y` in general (measured: the defect is O(1) for sedenions), so this is NOT
"everything is the same matrix algebra" — it is one *declared* recipe (left action), whose
defining identities are then **checked per dimension** in the self-tests, and which reduces
to the ordinary matrix-function calculus in the associative cases. Forward computation is
uniformly total across all M; the single place anything breaks is *inversion*, and it breaks
the same way everywhere (`L_x` singular). The flag is not "the algebra broke" — it is "you
asked for an inverse that has no unique answer." (Scalars/complex/quaternion/octonion are
division algebras and never hit it; zero divisors first appear at M = 16, the sedenions.)

**Flag soundness** (`audit_flags.jl`): GE/LE are absolute-value bounds and do NOT commute
with a function unless monotonicity on the admissible set is proven — a 2026-07-20 external
audit found five transcendental flag lies in `ScalarTot` (sqrt(-1)→clean 0, log(-1)/log(0)
silent finite, exp direction not flipped for negative inputs, sin/cos passing bounds through
a period, negative powers not swapping GE↔LE). All fixed under one principle — *if monotone
+ sign-consistent cannot be proven on the admissible set, drop to GE|LE|SUNK (and CPLX when
the true value may leave ℝ)* — and a permanent **semantic oracle** now enumerates admissible
true values per flagged input and falsifies every output claim: `julia audit_flags.jl` →
3039 checks, 0 violations.
