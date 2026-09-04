# julia/ — three modules: basic algebra / transcendental / scalar

All three are standalone (stdlib only) and carry the ⚠️ AI-assisted banner. Run each with
`julia <file>` for its self-test.

| file | module | what it is |
|---|---|---|
| **`HyperAlgebra.jl`** | `HyperAlgebra` | Array/batch total arithmetic **+ swappable wiring tensor** (`group_mul`) — the Julia mirror of `cuda_total.py`, written generically so it is `CuArray`-ready (GPU). Use this for the same thing the Python library does. |
| **`ScalarTot.jl`** | `ScalarTot` | Total arithmetic **as a Julia `Number`**: `TotNum <: Real` overloads `+ - * / ^ exp log ...`, so **existing generic code runs on it unchanged** — an ODE solver from OrdinaryDiffEq.jl solves with `TotNum` and the flag names *where/which-direction* the computation left the machine's representable range. This is the "used, not demo" bridge. |
| **`NestedSeries.jl`** | `NestedSeries` | The M/N/O layers as **freely composable registries** over one `Alg` interface: cells (`cd_alg`/`cyclic_alg`/`matn_alg`) × combinators (`mat_over`, `tensor`, `jordan` (symmetrized a∘b=(ab+ba)/2) — recursive: `mat_over(tensor(cd_alg(4),cd_alg(2)),2)` just works) × coefficient tapes (`:exp :sin :cos :sinh :cosh` + user-defined). Nothing assumed per combination: `assoc_defect`/`powerassoc_defect`/`commut_defect` measure the composed algebra, `nlog` verifies with forward exp or flags INEXACT. Measured law: **exp∘log verifies iff POWER-associativity holds** — octonion/sedenion scalars (non-associative, power-associative) verify at 1e-16; `mat2⟨cd16⟩` loses power-associativity (0.97) and breaks structurally (7e-3). Second measured law: a non-associative ⊗-base keeps power-associativity **only when the partner is commutative AND associative** — the `jordan` pincer (commutative, non-associative) shows commutativity alone fails. `ninv` = division rebuilt as the all-ones tape Σu^k, verified two-sided, INEXACT on zero divisors — no divider circuit. **`nsolve_left`/`nsolve_right`** = equation-solving division `L_a⁺x`/`R_a⁺x` (the full-rank completion of the founding axiom: a/0=0 IS Moore–Penrose at 1×1): multiplication-only Ben-Israel iteration `X←X(2I−AX)` — no pivots, no branches, and **zero divisions** (the initial scale rounds up to a power of two = an exponent shift, the same trick as `2^{-s}` in the exp unit), so solve is cells end-to-end. Two-tier verification: forward residual → exact solution (clean), normal-equation residual → least-squares (flagged SING — never pretends an inconsistent system was solved), unconverged → INEXACT. Measured: exact solves at 1e-16 (stdlib-pinv two-witness), left≠right (1.3 apart), zero divisor e3+e10 range-in exact / range-out honestly SING, and the **conj-div boundary**: `ā·x/|a|²` IS the solution of ay=x through the octonions and fails at sedenions (residual 0.84) while nsolve stays 1e-16 — mechanism measured as **Hurwitz's theorem** (conjugate=transpose holds at every CD dim; the composition property `L_aᵀL_a=|a|²I` holds exactly for dims 1,2,4,8 and breaks at 16). The review's full **division family of five** is on the shelf: `nconj_div_left/right` (the always-computable algebraic formula — VERIFIED against the equation and flagged INEXACT when merely formal, i.e. beyond the Hurwitz four), `nsolve_left/right` (equation-solving `L⁺/R⁺`), `nnorm_div` (‖x‖/‖a‖, with norm-multiplicativity holding in ℍ and measurably failing in sedenions — the zero-divisor consequence), `nnormalize` (`a/‖a‖`, real-scalar division only, 0→0), plus `ninv` (the geometric tape). Two preset shelves: `OPS`/`nop(A,:sqrt,x)` (operators as data: forward vs candidate+verify) and `ALGS`/`alg(:dualquat)` (famous algebras as compositions — dual numbers Λ1 give forward-mode AD `f(a+ε)=f(a)+f′(a)ε`; `:dualquat`=Λ1⊗ℍ is the rigid-body pose algebra; Grassmann/Clifford cells included), with `list_ops`/`list_algs` printing measured id-cards. |
| **`HyperTranscend.jl`** | `HyperTranscend` | **Experimental** unified computation of `exp`/`log`/`sqrt`/`^` for any M = 2^k via `f(x) = f(L_x)·e₀` (function values through the left regular representation — *not* a proof that every hypercomplex analytic function is captured; identities are **checked per dimension** in `self_test()`). Forward ops (`*`, `exp`, `x^{p≥0}`, `√`) stay total for every input incl. zero divisors — `√0 = 0` even though `L_0` is singular.  Only genuine **inversion** (`log`, `x^{neg}`) needs `L_x` nonsingular; a zero divisor there is named `⟦零因子⟧`. **Safe forward group** (`exp sin cos sinh cosh`, `x^{p≥0}` via `left_power` with explicit bracketing, `left_action(a,x0,t)=exp(t·L_a)·x0` for sedenion-valued linear ODEs) is total for every input. **Candidate group** (`sqrt log x^{frac}`) computes then **verifies the defining identity by a non-recursive residual** — trusted only if it holds, else flagged `⟦INEXACT⟧` (never a silent lie); `verify_sqrt`/`verify_log` are exposed for the caller. |

## `Transforms.jl` — WH, complex WH, and FFT (Julia twin of the Python MAPS additions)

All three transforms as one recipe — the character table of a finite abelian group — with
the difference being only the group's twist (= carry between layers): `(ℤ/2)³` → Walsh–
Hadamard (±1, pure Kronecker `H₂⊗H₂⊗H₂`, no twiddles), `ℤ/4×ℤ/2` → complex WH /
Chrestenson-4 ({±1,±i}, ×i = swap = still multiplier-free), `ℤ/8` → DFT (8th roots; the
Cooley–Tukey factorization carries twiddle diagonals — the frequency-side bill for the
carries). Ships real fast implementations, not just matrices: `fwht!` (in-place butterfly,
adds only) and `fft_rec` (recursive radix-2). Everything measured in `self_test`:
homomorphisms `M(a⊛b)=M(a)·M(b)`, factor products ≡ matrices (WH exact 0.0), fast ≡ dense,
`ΣUVW≡T` exact for complex WH, integer-input convolutions END-TO-END error 0.0 for WH and
complex WH, FFT convolution ≡ cyclic product. Real-rank ladder of order 8: 8 / 10 / 11.

## `TotalPipeline.jl` — U → V(O,N,M) → W, named (Julia twin of `total_pipeline.py`)

The audited `HyperAlgebra` core already runs as U (entry totalization) → V (fused Float64
MAC) → W (saturate-once + pattern-rule flags), with N (the wiring tensor) swappable and M
implicit. This module names that architecture without touching the core: `papply(op, a, b;
algebra=…)` is the one gateway (asserted bit-identical to direct kernel calls),
`from_kind`/`from_registry` fill the N slot — the latter bridges the **NestedSeries ALGS
shelf** (dualquat, Clifford, Grassmann, …) onto the audited kernel with flags — `Lmatrix`
exposes the explicit M (implicit ≡ explicit asserted), `TotalPipe(:gmul, algebra)` is the
declarative pair. `julia TotalPipeline.jl` runs the five-way self-test.

## `Discovery.jl` — implicit law discovery, everything from this shelf

Formalizes `demo_discovery.jl`: laws as `Σc·(monomials/derivative-columns) = 0`, solved as
the null space of the library matrix (one SVD — no gradients, no seeds; trivial solutions
are excluded structurally: c=0 by the unit sphere, `x^w−x^w` tautologies by the discrete
grid). **The differentiation library is the shelf itself**: `HDNum` wraps Λ1⊗Λ1 (hyperdual
numbers — the same mathematics inside ForwardDiff.jl) with ordinary Julia operators, and the
nilpotent lift `f(s+n)=f(s)+f′(s)n+f″(s)n²/2` (exact — the series terminates) turns any
elementary formula into exact ψ′, ψ″. Poison (Inf rails, NaN dropouts) is named by
ScalarTot's audited entry and stays LOCAL (hyperdual evaluation is pointwise; finite
differences would smear it). Measured in `self_test`: harmonic oscillator E=0.5 and hydrogen
E=−0.5 to ~1e-15 with σ_min ≈ 4e-16, 10 poisoned points → exactly 10 rows excluded, and
honest refusal (σ_min large) on lawless data. Python twin (complex support, e.g. discovering
the imaginary unit in free Schrödinger): `varpro-powersum-nn/implicit_discovery.py`.

## `MultiF32.jl` — float128 / float256 / float512 from Float32 operations only

`MFloat{P,K,EMAX}` with `F128` (113 bits), `F256` (237 bits), `F512` (489 bits): every
operation is Float32 `+ - * fma` plus compares/selects, so the same kernels run under CUDA.jl
unchanged (measured below; Float64 appears once, in the Newton seed). **Why not float32 "expansions"
(double-double style)?** Non-overlapping limbs stack 24 bits *of magnitude* each, and the
7th limb of a number ≈ 1 sits at 2⁻¹⁴⁴ — below Float32's subnormal floor. Expansions stop at
≈ 128 bits; the wall is Float32's *exponent range*, not its significand. The way through is
block floating point (精度と範囲の分離): each Float32 carries an 18-bit **digit** (an
integer 0 … 2¹⁸−1) whose weight is its position, one shared exponent carries the range.
Digit engine: `q = fma(a,b,C) − C`, `r = fma(a,b,−q)` with `C = 1.5·2⁴¹` splits a product
into high/low digits exactly (the fma's single rounding lands on a multiple of 2¹⁸); the
same constant trick at other bit positions is the carry ripple, the shifts and the rounding
masks; 18 = 24 − 6 bits of headroom make every column sum of a K×K schoolbook product exact
(K ≤ 31) — no error-free transformations anywhere. Rounding is from the **exact** result
(sum: K+2-digit window + sticky; product: all 2K digits), nearest-even; overflow → ±MAX·GE,
underflow → ±MIN·LE, never NaN/Inf; the flag algebra is `ScalarTot`'s, cross-checked
against `TotNum` (8×8 flags × 4 sign pairs × 5 ops = 0 mismatches). Division and square root:
Newton candidate in 18K-bit digit arithmetic → **exact residual** fixes it to the floor of
the true value and decides the rounding (candidate + verify, as `nsolve`/`verify_sqrt`
above); ties provably cannot occur for either. `julia MultiF32.jl` checks the primitives
against Int64/BigInt (200 000 cases), then each type against **MPFR** (BigFloat at
precision P, nearest): 1000 random operand pairs × (+ − × ÷ √) with exponent gaps up to
P+40, 200 × 15 constructed cases (1-ulp cancellation, sticky just below a binade boundary,
exact ties for + − ×, exact quotients/roots, quotients/roots near 1, odd-exponent √) and the
range edges (MAX+MAX, MIN/2, MIN·(1−ε), MAX+½ulp → GE, 1/0 = 0, √−1 → ℂ). Measured:
**0 mismatches for F128 / F256 / F512**, residual fix-ups ≤ 1, √2 in F512 equals MPFR's 489-bit
value.

**Two implementations, one bit pattern.** `MultiF32Ref.jl` is the first version: digit kernels
on `Vector{Float32}`, allocation-bound, kept as the reference. `MultiF32.jl` is the same
algorithm on `NTuple{K,Float32}`: every kernel is `@generated` straight-line Float32 code
specialised on the digit count (`Base.Cartesian`), static window widths are `Val`s, and no
operation touches the heap — the shape a CUDA.jl kernel needs. `julia julia/ab_multif32.jl`
sends the same seeded operands (2 000 pairs per type and operation plus the range edges: zero,
±MAX, values at ±emax) through both and compares sign, exponent, flag and the bit pattern of
every digit: **0 differences in 27 000 results** (add, sub, mul, div, sqrt, `<`, and the
Float64/BigFloat conversions), with a negative control (one flipped digit bit) that the
comparator does report. Three things had to be learned on the way, all measured: (1) Julia's
tuple helpers (`map`, `all`, splatting, `Base.tail`) fall back to a generic path beyond 32
elements (`Base.Any32`) and allocate — F512 has up to 59-digit windows — so the kernels use
literal-index `ntuple` helpers; (2) the sign fix-up of the digit split was written as `ifelse`,
which LLVM turned into a branch that mispredicts on random digits — a branch-free
`(1 − copysign(1, r))/2` halved the F512 product time; (3) the resulting branch-free 28×28
product is one basic block of ~11 000 instructions that LLVM's −O2 pipeline compiles
super-linearly (F512: 173 s for the product alone, 10 minutes for the type) — the product is now
assembled from 7×7 blocks compiled once per shape (`mac_block`), which compiles in 0.4 s, runs
within 10 % of the monolith and returns identical digits (every partial sum is an integer below
2²⁴). **Speed (CPU, single thread, loop of 20 000 random operand pairs with random signs,
minimum of 5 passes, Julia 1.11.5, Core Ultra 9 285K; `julia julia/bench_multif32.jl`; MPFR =
the C library, `mpfr_add / mul / div / sqrt` into a preallocated result, at 113 / 237 / 489
bits on the same operands):**

| | add | mul | div | sqrt |
|---|---|---|---|---|
| F128 Vector ref → **NTuple** | 242 → **108 ns** | 442 → **160 ns** | 6.0 → **1.9 µs** | 9.0 → **3.1 µs** |
| F128 MPFR (113 bits) | 16 ns | 13 ns | 21 ns | 31 ns |
| F256 Vector ref → **NTuple** | 382 → **145 ns** | 1.3 µs → **320 ns** | 14.1 → **4.4 µs** | 21.7 → **7.1 µs** |
| F256 MPFR (237 bits) | 21 ns | 25 ns | 58 ns | 88 ns |
| F512 Vector ref → **NTuple** | 572 → **234 ns** | 4.4 µs → **941 ns** | 43.4 → **12.4 µs** | 67.4 → **20.3 µs** |
| F512 MPFR (489 bits) | 22 ns | 41 ns | 85 ns | 165 ns |

The register version is 2.2–4.6× faster than the Vector one with 0 allocations, and against
MPFR at the same precision it is **7–11× slower for add, 13–23× for mul and 76–146× for
div / sqrt** (the Newton candidate + exact-residual verification costs several full-width
products and wide compares; MPFR's ÷ √ are single-pass 64-bit-limb algorithms). The first
version of this table put MPFR at 73–304 ns and the kernels "within 1.1–1.3× of MPFR for
add": that column was Julia's `BigFloat` wrapper — every result allocated (2 allocations per
operation, the collector's time included in a single pass), on operands one limb wider than
the result (the conversion builds 18K+16 bits), allocated interleaved with temporaries — and
each of those costs 1.5–4× on an operation of 15–90 ns, together 1.4–9×. The C library called
into a preallocated result on P-bit operands is what a C caller pays, and every "MPFR" number
in this file now means that (the header of `bench_multif32.jl` records the protocol; the
`BigFloat` column is still printed beside it). On a CPU with 64-bit limbs MPFR is the right tool
by an order of magnitude. What the digit design buys is a different machine: every
operation is Float32 `fma` on 18-bit digits with no carries between lanes, so the same kernels
run on a GPU, where 64-bit integer and FP64 throughput are the bottleneck (RTX 5090: FP64 =
1/64 of FP32).

**GPU (`julia julia/bench_multif32_cuda.jl`; RTX 5090, sm_120, CUDA 13.3 runtime / 13.1
driver 591.86, CUDA.jl 6.3.1, Julia 1.11.5).** `MultiF32.jl` is included unchanged (the type is
`isbits`, so `CuArray{F512}` just works); the kernel is one thread per operation, 2²⁰ operations
per launch, kernel time by CUDA events, minimum of 10 launches after ≥ 1 s of warm-up launches.
Operands: random significands and signs, exponents −8…8, plus 1/64 of range edges (0, ±MAX,
MIN, ±MAX·GE). **Every GPU result is compared bit for bit with the CPU result of the same
operands: 0 differences in 12 × 2²⁰.**

| ns per operation | add | mul | div | sqrt | registers / local memory per thread |
|---|---|---|---|---|---|
| F128 **GPU** | **1.06** | **0.62** | **4.7** | **6.9** | 90–243 / 0.7–3.1 KB |
| F128 CPU NTuple · MPFR C | 108 · 16 | 160 · 13 | 1854 · 21 | 3090 · 31 | |
| F256 **GPU** | **2.00** | **1.55** | **18.0** | **25.6** | 153–255 / 1.3–6.1 KB |
| F256 CPU NTuple · MPFR C | 145 · 21 | 320 · 25 | 4398 · 58 | 7125 · 88 | |
| F512 **GPU** | **4.34** | **4.69** | **83.7** | **119** | 255 / 2.9–12.9 KB |
| F512 CPU NTuple · MPFR C | 234 · 22 | 941 · 41 | 12424 · 85 | 20299 · 165 | |

Per operation the GPU is 54–102× one CPU core for add, 200–260× for mul and 150–450× for
div / sqrt (the same code), and against MPFR on one core it is **5–15× faster for add, 9–21×
for mul, 1.0–4.5× for div and 1.4–4.5× for sqrt** (F512 ÷ on the GPU = MPFR ÷ on one core):
F512 multiplies at 213 M/s, F128 at 1.6 G/s. That is the number the design was for, and it is
modest: ≈ 1–2 % of the FP32 peak (a
F512 product is ≈ 9 000 Float32 operations, so 213 M/s of them is ≈ 1.9 TFLOP/s of the 105
available). Three measured reasons, in order: (1) **local memory** — every kernel spills its
digit windows to the per-thread stack (0.7–13 KB: the `@noinline` kernels pass tuples by
pointer and `place` indexes tuples dynamically), and the div / sqrt kernels sit at the
255-register ceiling, so occupancy is 0.17–0.42; forcing everything inline was tried and is
worse (40 KB of local memory for F512 div, div / sqrt 1.5–2× slower); (2) **warp divergence** —
with same-sign operands and no edges the adds are 0.65 / 1.16 / 2.47 ns, random signs cost
1.6–1.8× because every warp then runs both the addition and the cancellation path, and one zero
lane in 32 costs 20–40 % on F512; (3) **the clock** — after the CPU pass idles the GPU for
seconds, 10 warm-up launches of a kernel shorter than 10 ms measure at a reduced clock (F512
add 4.05 instead of 2.46 ns/op), so the warm-up is by time. The layout — arrays of structs, 124 B per
F512, every field access strided across the warp — was measured afterwards with `MultiU32.jl`
(below): a structure-of-arrays layout is worth 1.4–1.7× on + ×, nothing on ÷ √. The honest
statement is now "correct at 128/256/512 bits from Float32 only; on the CPU 7–11× slower than
MPFR for +, 13–23× for ×, 76–146× for ÷ √; on one GPU 5–21× MPFR-on-one-core for + ×,
1.0–4.5× for ÷ √, at 1–2 % of the GPU's FP32 peak".

## `MultiU32.jl` — the same numbers on 29-bit integer digits (UInt32 limbs, UInt64 columns)

The Float32 constraint of `MultiF32.jl` was the question ("float128 from float32 only"); this
file asks the same design of the lanes a GPU multiplies natively. A digit is 29 bits in a
`UInt32`, a digit product is one `mul.wide.u32` into a `UInt64`, and column sums of the K×K
schoolbook product stay exact in 64 bits (≤ 32 products below 2⁵⁸): 29 = 32 − 3 buys what
18 = 24 − 6 buys in MultiF32 — no carry handling inside the product, one carry ripple at the
end. This is MPFR's limb arithmetic at the width the GPU has: MPFR's 64-bit limbs are chained
by the hardware carry flag (`adc`), which compilers do not expose on a GPU. Everything above
the digit engine — `MFloat{P,K,EMAX}`, the canonical form, the flags, rounding to nearest-even
from the exact result, candidate + exact residual for ÷ √, ScalarTot's flag rules — is
MultiF32's, and so is the verification: `julia julia/MultiU32.jl` runs the primitives against
`BigInt`, the MPFR battery (**0 mismatches for F128 / F256 / F512**, residual fix-ups ≤ 1,
√2 in F512 = MPFR's 489-bit value) and the ScalarTot twin check; `bench_multiu32.jl` also
sends the same operands through both engines (**MultiF32 ≠ MultiU32: 0**). (The candidate
itself is no longer MultiF32's full-width Newton loop but the fixed-point ladder below — a
change of the *proposal*, not of the answer: `ab_multiu32.jl` holds it to the old file's bits.)
The candidate wants ≥ 12 guard bits (29K ≥ P + 12), so K = 5 / 9 / 18 digits = 145 / 261 / 522
bits and 32 / 48 / 84 bytes per element (MultiF32: 40 / 68 / 124).

**Written for a warp, not a core.** The first version (`MultiU32Ref.jl`, kept as the reference)
was the plain way to write it: `place` indexed digit tuples by the data-dependent shift,
`round_pack` found the leading digit and rounded with loops, the subtraction branched on the
sign. Correct, MPFR-verified, and on the RTX 5090 a decomposition of the F512 multiply gave
copy 0.25 ns, the 18×18 product 0.35 ns, the full multiply 1.22 ns — **the normalize-and-round
step took 3/4 of the multiply, 2.5× the product itself**, because a tuple indexed by a runtime value goes through local
memory and a loop with a data-dependent exit splits the warp. `MultiU32.jl` removes both:
shifts by a data-dependent number of digits are barrel shifters (⌈log₂ N⌉ steps of full-width
selects with static indices), rounding is FPU-style — **normalize first, then round at a static
bit position** (`round_norm`: round bit, sticky mask, increment digit and the packed digits are
compile-time constants of P and K), the residual windows of ÷ and √ shift by amounts that
depend on P and K only (so the barrels fold to static digit moves), and `wide_sub` and the
cancellation path of the add compute both A−B and B−A and select. Two Julia facts had to be
found on the way: a shift amount that is a compile-time constant folds the barrel only if the
generated helpers are inlined (`Base.@_inline_meta` inside `@generated` bodies); and an N-ary
`|(e[1], …, e[38])` with more than 32 arguments leaves Base's unrolled `afoldl` and heap-allocates
its varargs — 176 bytes per call on the 38-digit residual windows of F512, which cost F512
÷ √ +45 % until the OR became a balanced binary tree. **Bit identity was required, not
assumed**: `julia julia/ab_multiu32.jl` sends the same seeded operands (2 500 per type — random
signs, exponent gaps to P+40, 1–3-ulp cancellations, the range edges) through both files and
compares sign, exponent, flag and every digit: **0 differences in 63 000 results** (4 000 per
type after the ladder went in — 100 000 results, still 0), the
normalize-and-round step alone is checked against an MPFR reference on 3 600 random windows,
and both tests carry a negative control (a flipped bit that must be reported).

**Totality is a separate claim from correctness, so it has its own audit** (`julia
julia/audit_multiu32.jl`; the older `audit_total.jl` / `audit_flags.jl` cover ScalarTot and
HyperTranscend, not this file). Total arithmetic promises two things: **① every input produces a
value** — no trap, no NaN, no Inf, and the result is canonical, so it is safe to feed back in —
and **② the flag does not lie**: GE ⇒ |true| ≥ |shown|, LE ⇒ |true| ≤ |shown|, no SUNK ⇒ the shown
sign is the true sign. ① is not free here: `div_core`, `sqrt_core` and `place` carry internal
assertions (`error("candidate too far")`, `error("non-zero digit above the window")`) which are
consistency checks, not handled cases — a candidate outside its assumed range would make the
library *throw* instead of answer. The audit runs the two operations over significands at both
ends of [1,2) (1 exactly, 2−ulp, 1+ulp, all-ones, 1000…01, alternating), exponents at ±emax where
÷ and √ saturate, all 16 flag combinations, the zeros (including a GE-flagged zero) and the
saturated values, plus a random sweep; the true value comes from MPFR at 4P+64 bits with an
unbounded exponent. Measured: **exceptions 0, non-canonical 0, flag lies 0 in 218 632 results**,
and the defining laws (a/0 = 0, 0/0 = 0, a/⟦≥⟧0 is SUNK, √(−1) = 0⟦ℂ⟧, √0 unflagged,
MAX/MIN → ⟦≥⟧, MIN/MAX → ⟦≤⟧) hold in all three types. With exp and log on the same shelf
(2026-09-03, below): **242 140 results, exceptions 0, non-canonical 0, flag lies 0**, and the
laws now include log 0 = 0 unflagged, log(⟦≥⟧0) = the ℂ seat, log(−1) = 0⟦ℂ⟧, log(MIN⟦≤⟧) =
log MIN⟦≥⟧, exp(−MAX) = +MIN⟦≤⟧ (never 0), exp MAX = MAX⟦≥⟧, exp(⟦≥⟧0) = 1⟦≥≤⟧, exp(x⟦±⟧)
never ⟦±⟧, ℂ sticking through + and exp, ℂ⁰ = 1 — 0 violations.

The third check is the one the ladder made necessary. **Totality of ÷ √ rests on the candidate
landing within 8 ulps of the exact result** — the fix-up loop throws past that — so it is a
property of the *candidate*, which the ladder replaced, and counting fix-ups (÷ 1, √ 0 at their
worst here, against a bound of 8) only says the bound was not reached on these inputs. The audit
therefore measures the candidate's own accuracy against MPFR: **143 / 259 / 520 bits for 1/m and
144 / 260 / 521 for √m** (worst of 2 400 significands per type), against the 141 / 257 / 518 the
ladder is designed for and the 116 / 240 / 492 the assertion needs — a margin of **19–29 bits**,
i.e. the candidate is 10⁵–10⁸ times better than the assertion requires, and the tightest of the
three is F256 because 29K − P leaves it only 24 guard bits. That margin is what keeps the
assertions unreachable in practice; it is measured, not proven, and the honest statement is that
they are assertions, not handled cases.

**CPU, single thread (`julia julia/bench_multiu32.jl`, 20 000 random operand pairs with random
signs, minimum of 5 passes; MPFR = the C library into a preallocated result, at the same
precision on the same operands):**

| ns per op | add | mul | div | sqrt |
|---|---|---|---|---|
| F128 MultiF32 · **MultiU32** · MPFR C | 107 · **31** · 17 | 154 · **25** · 13 | 1775 · **125** · 21 | 2975 · **156** · 29 |
| F256 | 152 · **53** · 21 | 319 · **40** · 24 | 4280 · **261** · 57 | 6888 · **244** · 84 |
| F512 | 233 · **82** · 23 | 867 · **89** · 41 | 12312 · **565** · 85 | 19768 · **516** · 159 |

Integer digits are 3.4–35× the Float32 digits on the CPU, 0 allocations, and against MPFR at the
same precision **1.6–3.6× slower for add and mul**, **4.6–6.6× for ÷** and **2.9–5.3× for √**. The
first version of this table said "1.9–2.4× *faster* than MPFR for + ×, 5–8× slower for ÷ √" —
retracted: its MPFR column (70 / 61 / 69 / 92 ns for F128) was Julia's `BigFloat` wrapper on
operands one limb wider than the result, allocated interleaved with temporaries, and the
allocation, the collector, the extra limb and the memory layout each cost 1.5–4× on an operation of
15–90 ns; the C library called into a preallocated result on P-bit operands takes 13–159 ns (the
`bench_multiu32.jl` header records the four findings, and the `BigFloat` column is still printed
beside the C one).

**The ÷ √ column is the second retraction.** It read 344 / 837 / 2055 and 543 / 1260 / 2907 ns —
14–24× MPFR — with the diagnosis "`newton_iters` = 5 iterations, each 2 full-width products + 2
full-width adds at 522 bits ⟹ ≈ 13 products + 10 adds ≈ 1 960 ns against 2 055 measured", and the
proposal to fix it with a precision-doubling ladder. The diagnosis was arithmetic and the
proposal followed it, so the first implementation narrowed the multiplies: four steps at 4 / 8 /
14 / 18 digits with the correction y·e narrower still, **2.2 full-width products of digit work by
the model — and it measured 10.4** (F512 ÷ 2055 → 1181, not the predicted ≈ 640). Profiling the
pieces (`prof_divsqrt.jl`) said why: at these widths a *rounded operation is mostly its normalize
+ round_pack*, not its digit products — a 4-digit multiply is ~10 ns of products behind ~30 ns of
normalization, an F512 add (83 ns) costs as much as an F512 product (87), and the ladder does 16
operations. The ladder is now written in **fixed point**: y = Y·2^(1−29w) with m at the same
scale, so "where the 1 sits in the product" and "how far the correction is shifted in" are
compile-time digit positions and *nothing between the seed and the candidate is normalized or
rounded*. The residual of a step is read as the low L digits of the product in two's complement —
the leading 1 it is compared against contributes nothing below itself, so the digits that would
cancel are neither multiplied nor subtracted. √ takes the ladder to half the width on 1/√m and one
Karp–Markstein step (s = m·y, s ← s + (y/2)(m − s²)) to √m itself, which is also the multiplication
`sqrt_core` no longer does. Measured: **÷ 2.8 / 3.2 / 3.6× and √ 3.5 / 5.2 / 5.6×**, every result
bit-identical to the full-width Newton loop (`MultiU32Ref.jl`, `ab_multiu32.jl`, 4 000 pairs per
type and operation plus edges, with the flipped-bit control) — bit identity is by construction
here, since the exact residual and the rounding rule define the answer and the candidate only
proposes it (the fix-up counts stayed at ÷ 0, √ 1). What is left at F512 is 3 full-width products
(candidate 267 ns, the a·y and the (K+1)×K residual) and the final normalize (46) against 85 ns
for MPFR's single-pass limb recurrence; a digit recurrence written branchless for the warp is the
remaining idea and is not implemented. The first version measured 44 / 52 / 594 / 938 ns (F128)
and 91 / 121 / 2122 / 3244 ns (F512): the barrel-shifter rewrite was worth 4–50 % on the CPU too.

**GPU (`julia julia/bench_multif32_cuda.jl 4194304 u32 both`, same machine and protocol as above;
every result bit-identical to the CPU: 0 differences in 24 × 2²² — 12 operations × 2 layouts; the
first-version column is the earlier run at 2²⁰):**

| ns per operation | add | mul | div | sqrt | registers / local memory per thread |
|---|---|---|---|---|---|
| F128 first version → rewrite, AoS → SoA → **+ ladder** | 0.416 → 0.106 → **0.066** | 0.290 → 0.112 → **0.086** | 1.92 → 0.74 → 0.71 → **0.501** | 2.14 → 1.05 → 1.03 → **0.464** | 48–119 / 0.2–1.0 KB |
| F256 | 0.717 → 0.174 → **0.126** | 0.734 → 0.212 → **0.183** | 3.69 → 1.64 → 1.36 → **1.135** | 4.70 → 1.88 → 1.85 → **0.952** | 55–167 / 0.2–1.6 KB |
| F512 | 1.937 → 0.353 → **0.204** | 1.310 → 0.406 → **0.286** | 15.8 → 3.25 → 3.18 → **2.324** | 19.5 → 4.16 → 4.09 → **1.640** | 85–255 / 0.4–2.9 KB |
| F512 MultiF32 GPU (above) | 4.34 | 4.69 | 83.7 | 119 | 255 / 2.9–12.9 KB |

The rewrite is 3.3–5.4× on F512 (local memory 1.3 → 0.4 KB for add, 6.5 → 2.8 KB for div). The
decomposition after it (F512, AoS): copy 0.253, product 0.347, full multiply 0.365 ns — the
normalize-and-round step is now ≈ 0.02 ns, and the add moved 3 × 84 B at ≈ 700 GB/s, within
1.4× of the copy floor *of that layout*. Forcing the cores inline was measured again and is
again mixed (F512 add 0.31, mul 0.44 ns), so the digit kernels stay `@noinline`.

**The layout.** The copy floor itself was the suspect: with `CuArray{F512}` (an array of
structs, AoS) the 32 lanes of a warp that load one field touch addresses 84 bytes apart — 21
cache lines per load instruction — and the copy kernel moved its 2 × 84 B at 627 GB/s, **41 % of
what a device-to-device `copyto!` reaches on this machine (1518 GB/s, 85 % of the nominal
1.8 TB/s; that measured number is the floor used below)**. The `soa` layout of
`bench_multif32_cuda.jl` keeps one array per field, the digits as an N × K matrix with the element
index first, so a warp's load of digit j is one 128-byte line; the arithmetic is the same
functions, only the load / store glue differs. The first SoA run was *slower* than AoS in every
row (F512 add 0.596 ns, and 160–264 B of local memory in a copy kernel that should have none): the
two glue helpers were `@generated` without `Base.@_inline_meta`, so the 18-digit tuple crossed a
call boundary through the stack — the barrel lesson a third time. Inlined: **F512 copy
0.249 → 0.107 ns (1451 GB/s, 96 % of the floor), add 0.353 → 0.205, mul 0.406 → 0.289; F128 add
0.106 → 0.068, mul 0.112 → 0.088** — 1.4–1.7× for add, 1.15–1.4× for mul, and ÷ √ unmoved
within 2 % (F256 div 1.64 → 1.36 is the one exception), as expected for kernels bound by the
255-register ceiling and the Newton iterations rather than by memory. (Measure the layout
comparison at N ≥ 2²²: at 2²⁰ two F128 / F256 arrays fit in the 96 MB L2 and the SoA copy
measures above DRAM bandwidth, 2169 GB/s.)

**The ladder on the GPU is not the CPU's win.** The fixed-point ladder is 2.8–3.6× on ÷ and
3.5–5.6× on √ on a core; here it is **1.2–1.4× on ÷ and 1.9–2.5× on √**, and the reason is in the
last column: ÷ still compiles to 255 registers and 2.9 KB of per-thread local memory, so what the
ladder removed in instructions the spill traffic keeps charging. √ escaped it — the Karp–Markstein
step dropped it to 188 registers and 2.7 KB, and that is where its 2.5× comes from. The
two's-complement residual (the low L digits of the product instead of the full 2w) was written for
exactly this and moved ÷ from 2.59 to 2.32 ns without touching the register count: 255 is a
ceiling, not a measurement. Against MPFR (the C library) on one core: **F512 add 113×, mul 143×,
div 37×, sqrt 97×** (F128: 258× / 151× / 42× / 63×; F256: 167× / 131× / 50× / 88×; an earlier
version of this paragraph said 956× / 751× / 87× / 84× — the `BigFloat`-wrapper MPFR column,
retracted above); F512 multiplies at 3.5 G/s, F128 at 11 G/s. Where it stands: the
add moves 3 × 78 B in 0.204 ns = 1149 GB/s, **1.33× the bandwidth floor** (F128 1.32×, F256
1.55×); the multiply is at 1.9× the floor, i.e. ≈ 0.18 ns of arithmetic per F512 product that the
memory system no longer hides — the product's 108 registers and 528 B of local memory (the
`@noinline` call) cap the occupancy that would hide it; and ÷ keeps its register ceiling. The
honest statement: "correct at 128 / 256 / 512 bits from 32-bit integer operations; on the CPU
1.6–3.6× slower than MPFR for + ×, 4.6–6.6× for ÷, 2.9–5.3× for √; on one GPU 113–258×
MPFR-on-one-core for + ×, 37–50× for ÷, 63–97× for √, the add within 1.3–1.6× of the memory
bandwidth, the multiply within 1.9×".

**Against the existing libraries (`julia julia/bench_libs.jl`; needs MultiFloats, DoubleFloats,
Quadmath and CUDA, which you install yourself — not vendored).** Same operands for every row
(500-bit random significands, random signs, exponents −8…8), the same loop `z[i] = f(x[i], y[i])`
(so LLVM may vectorize the pure-Float64 libraries — that is their real advantage, counted), minimum
of 5 passes over 20 000 pairs; ns per operation on one core of the Core Ultra 9 285K, and on the
RTX 5090 in the one-thread-one-operation kernel over 2²² elements, array of structs for every type
(MultiU32's SoA numbers above are 1.4–1.7× better on + ×):

| ns per op (bits of precision) | CPU add | mul | div | sqrt | GPU add · mul · div · sqrt |
|---|---|---|---|---|---|
| **MultiU32 F128** (113) | 33.1 | 30.4 | 349.5 | 537.3 | 0.105 · 0.108 · 0.720 · 1.033 |
| Quadmath `Float128` (113, libquadmath) | 18.8 | 17.4 | 21.4 | 134.2 | — |
| DoubleFloats `Double64` (106) | 4.8 | 4.6 | 4.8 | 3.0 | — |
| MultiFloats `Float64x2` (106) | 0.8 | 0.4 | 0.6 | 1.8 | 0.030 · 0.029 · 0.029 · 0.030 |
| MPFR, C library (113) | 18.8 | 15.0 | 22.5 | 29.2 | — |
| **MultiU32 F256** (237) | 53.6 | 43.4 | 765.6 | 1171.7 | 0.173 · 0.200 · 1.515 · 1.847 |
| MultiFloats `Float64x4` (212) | 3.5 | 3.2 | 11.3 | 88.9 | 0.151 · 0.151 · 0.431 · 0.839 |
| MPFR, C library (237) | 21.8 | 26.8 | 53.1 | 83.5 | — |
| **MultiU32 F512** (489) | 88.5 | 95.1 | 2044.2 | 2938.2 | 0.376 · 0.372 · 3.152 · 4.072 |
| MPFR, C library (489) | 25.6 | 44.8 | 87.7 | 157.9 | — |

(MultiFloats 3.3.1 ships kernels for ≤ 4 words, so there is no ~512-bit expansion row; every GPU
result equals the CPU result of the same library, 0 differences in 5 × 4 × 20 000.) Accuracy —
the largest relative error against MPFR at 600 bits over 2 000 operands, in units of 2^−(bits),
add / mul / div / sqrt: MultiU32 0.96–1.00 at every size (correctly rounded ⟹ ≤ 1); Quadmath
0.998 / 0.984 / 0.983 / **1.46** (its sqrt is not correctly rounded); Double64 1.8 / 3.0 / 4.1 /
2.2; Float64x2 1.4 / 3.0 / 6.6 / 2.2; Float64x4 0.15 / 2.1 / **12.5 / 33** — the Float64
expansions are fast because they are not correctly rounded and carry 106 / 212 rather than 113 /
237 bits (a 212-bit `Float64x4` sqrt is good to 207). Where MultiU32 stands: on the CPU it loses
to everything — 1.6–3.5× MPFR and 7–76× the Float64 expansions on + ×, 14–23× MPFR on ÷ √; on the
GPU `Float64x2` sits at the memory floor for all four operations (3 × 16 B per operation at
1.6 TB/s = 0.03 ns — the RTX 5090 runs FP64 at 1/64 rate and that is still enough for 106 bits)
and `Float64x4` is 1.15–3.5× faster than F256 (it is FP64-bound: 0.151 ns for the add). What
MultiU32 has that these do not: correct rounding at the full 113 / 237 / 489 bits, the wide
exponent and the flags, and the 489-bit size on a GPU at all, with + × within 1.3–1.9× of the
bandwidth floor.

**exp and log (2026-09-03) — correctly rounded, total, the semantics of the reserved word.**
Both are computed in a *working type* of K+2 digits (≥ 70 bits beyond P) and rounded to P only
when the rounding is *decided*: round-to-nearest changes only at a midpoint of the P-bit grid,
so if the working value is further than its error bound B from every midpoint, rounding it is
rounding the true value (Ziv's test — `ziv_ok`, two static-position subtractions and a select).
Otherwise the same computation runs again at K+6 and then 2K+6 digits, which is 2P+ bits — where
the known hard cases live: exp(2^−P) = 1 + 2^−P + 2^−(2P+1) is a midpoint plus a tail that starts
2P bits down, and log(1 + ulp) is the mirror of it. Should even that be undecided, the result
is the truncated value with ⟦≥⟧: true ∈ (v, v + ulp) is then a *proven* statement, so the
library stays total and honest without an exception (`ZIV_UNDECIDED`, expected and measured 0).
The 2K+6 stage needed 42-digit products for F512, so `mul_digits` now folds its 64-bit columns
every 32 rows (a carry pass emitted between row blocks) and the type admits K ≤ 64; the
primitives test covers 42 × 42 digits and the 40-digit all-ones square. exp: k = round(x/ln 2),
r = x − k·ln 2 with ln 2 at K+4 digits (the ≤ 25 bits of cancellation are paid there), Taylor of
N terms after s halvings with 1/n! as compile-time constants (Horner), then s squarings; the
error bound is 2^s·(2N+2) ulps and s ≈ 0.7√bits balances the two (F128 / F256 / F512: s = 10 /
12 / 17, N = 15 / 20 / 27). log: x = m·2^e with m brought into [1/√2, √2) so that e·ln 2 + log m
never cancels, then square roots of m *only until |m′ − 1| ≈ 2^−(s+2.5)* — the first version
took s roots unconditionally and lost the last bits of log(1 ± 2^−100): a root's rounding error
is absolute, and m′ − 1 turns it into a relative one, so on an m already within 2^−(s+2.5) of 1
the roots only destroy (m − 1 itself is exact by Sterbenz) — then z = (m′−1)/(m′+1), the odd
atanh series with 1/(2n+1) as constants, ×2^{s′+1}, + e·ln 2 (s = 4 / 5 / 6, N = 16 / 21 / 34).
The exact cases never enter a series: exp 0 = 1, log 1 = 0, and **log 0 = 0 by definition** —
the reserved word (the ScalarTot paragraph below): Σ (1/n)((x−1)/x)ⁿ is 0 term by term under
a/0 = 0, and the limit −∞ belongs to ε, log(MIN⟦≤⟧) = log MIN⟦≥⟧, which falls out of the flag
rule. Negative → 0⟦ℂ⟧, and ℂ now sticks through every `MultiU32` operation as in `TotNum` (the
ℂ seat 0⟦≥≤±ℂ⟧; z⁰ = 1 excepted). exp(−MAX) = +MIN⟦≤⟧, never 0 (e^x > 0). Bugs found on the
way: k took the sign of |x| (`top_float64` is a magnitude), the unconditional roots above, and
a Ziv bound that did not fit one digit at the 2K+6 stage.

Measured: **0 mismatches against MPFR at P bits** in the self-test (1 000 random × 2 + 200 × 4
adversarial + the midpoint cases + edges per type), in a separate sweep of 3 × 22 000
(random, the tiny-|x| and near-1 neighbourhoods), in the totality audit and in the benchmark's
own cross-check; the twin-check against `TotNum` at 16 × 16 flags × 12 values × 7 ops: 0. Ziv
stages in the audit: **56 221 decided at K+2, 28 at K+6, 56 at 2K+6, 0 undecided** — the
deep stage is reached only by the constructed cases, as designed. CPU, single thread
(`julia julia/bench_explog.jl`, 5 000 operands with |x| ∈ [2^−8, 2^8) for exp and exponents
within ±8 for log, minimum of 5 passes, MPFR = the C library into a preallocated result at the
same precision):

| ns per op | exp MultiU32 · MPFR C | log MultiU32 · MPFR C |
|---|---|---|
| F128 | **1 490** · 799 (1.9×) | **2 374** · 1 361 (1.7×) |
| F256 | **2 730** · 1 235 (2.2×) | **4 452** · 2 004 (2.2×) |
| F512 | **7 414** · 2 452 (3.0×) | **11 643** · 3 581 (3.3×) |

1.7–3.3× MPFR, 0 allocations (a BigInt branch in the Int64 constructor had made its return
type `Any` and cost 110 B per call until it was written in three digits) — closer to MPFR than
÷ √ are (4.6–6.6×), because here the cost is ~45 (exp) and ~60 (log) rounded operations of the
working type rather than a single residual, and MPFR's exp / log at these sizes are the same
kind of loop. Not done: a fixed-point Taylor (the ladder's lesson — at these widths a rounded
operation is mostly its normalize + round) and a table-driven reduction.

**Against the existing libraries, speed and accuracy on the same operands** (`julia
julia/bench_explog_libs.jl`; 20 000 operands per timing, 4 000 per accuracy sample; the error is
measured against MPFR at 600 bits in ulps of each library's *own* precision, so a correctly
rounded result has max ≤ 0.5; "tiny |x|" = exp of |x| ∈ [2^−(P/2), 2^−4], "near 1" = log of
1 ± 2^−4 … 2^−(P/2), the neighbourhoods where the expansions lose their bits):

| exp · log | ns per op | exp generic max / mean ulp | exp tiny | log generic | log near 1 |
|---|---|---|---|---|---|
| **MultiU32 F128** (113) | 1 385 · 2 393 | **0.500** / 0.253 | **0.500** / 0.247 | **0.500** / 0.249 | **0.500** / 0.250 |
| Quadmath Float128 (113) | 603 · 512 | 0.500 / 0.253 | 0.500 / 0.251 | 0.691 / 0.253 | 0.775 / 0.251 |
| DoubleFloats Double64 (106) | 29 · 26 | 9.78 / 1.44 | 7.42 / 1.28 | 2.48 / 0.297 | 4.70 / 0.645 |
| MultiFloats Float64x2 (106) | 62 · 37 | 494.8 / 10.1 | 8.09 / 1.29 | 2.48 / 0.298 | 5.15 / 0.631 |
| MPFR C (113) | 789 · 1 361 | (correctly rounded) | | | |
| **MultiU32 F256** (237) | 2 758 · 4 445 | **0.500** / 0.252 | **0.500** / 0.251 | **0.500** / 0.251 | **0.500** / 0.249 |
| MultiFloats Float64x4 (212) | 517 · 457 | 321.6 / 3.44 | 8.49 / 0.303 | 2.00 / 0.097 | 15.2 / 0.308 |
| MPFR C (237) | 1 235 · 2 024 | | | | |
| **MultiU32 F512** (489) | 7 406 · 11 510 | **0.500** / 0.256 | **0.500** / 0.252 | **0.500** / 0.251 | **0.500** / 0.247 |
| MPFR C (489) | 2 289 · 3 520 | | | | |

MultiU32 is correctly rounded in every cell (max exactly 0.500, mean ≈ 0.25 = the mean of a
uniform rounding error, 0 of 4 000 above half an ulp); libquadmath's exp is too, its log is not
(0.69–0.78 ulp); the Float64 expansions are 5–50× faster and 2–10 ulps off on their own scale
in the ordinary domain, and `Float64x2` / `Float64x4` exp reach **495 and 322 ulps** on |x| ∈
[2^−8, 2^8) — a lost reduction, not a rounding — so their speed buys a different thing. The
same file measures the premise of the Ziv test: the *actual* error of the first stage against
its bound, on the generic and the hard operands (worst of 4 000): exp 1 091 / 4 340 / 142 017
ulps of the working type against bounds 65 536 / 524 288 / 16 777 216 (margin 2^5.9–2^6.9);
log 225 / 347 / 689 against a bound that was 592 / 1 124 / 2 200 — a margin of only 2.6–3.2×,
so the log bound was multiplied by 4 (2 368 / 4 496 / 8 800, margin 2^3.4–2^3.7). A larger
bound costs only fallback probability, ≈ 2^−(tail − log₂ B) ≈ 2^−66 per operation, and the
self-test / audit counts after the change are unchanged (stage 2: 5–28, stage 3: 5–56, all of
them the constructed cases).

**GPU** (`julia julia/bench_multif32_cuda.jl 1048576 u32 both`, the same one-thread-one-operation
kernel as + × ÷ √, operands with exponents −8…8, 2²⁰ per array, min of 10 launches after a 1 s
warm-up; every result bit-identical to the CPU — 0 of 2²⁰ differ, per operation and layout):

| ns per op (SoA) | exp | log | registers / local memory | vs MPFR C, one core |
|---|---|---|---|---|
| F128 | **2.96** | **4.59** | 152 / 6.4 KB · 255 / 9.1 KB | 267× · 296× |
| F256 | **4.99** | **8.11** | 188 / 10.6 KB · 255 / 15.6 KB | 247× · 250× |
| F512 | **10.7** | **23.6** | 255 / 22.9 KB · 255 / 38.0 KB | 214× · 149× |

The Ziv ladder compiles as it is — all three stages, the 42-digit one included, are in the
kernel, and the stage-2 / stage-3 paths ran for the operands that need them (the CPU–GPU
comparison would have caught a divergence). The cost is in the last column: exp at 150–255
registers and log at the 255-register ceiling with 9–38 KB of per-thread local memory (the
working-type values of K+2 … 2K+6 digits and the unrolled Horner), so the layout no longer
matters (AoS and SoA within 1 %) and F512 log runs at 10 GB/s of operand traffic — 149× one
MPFR core, against 214–296× for the smaller sizes. The spill traffic is the next thing to
remove (a fixed-point Taylor with static digit positions, as the ÷ √ ladder did); it is not done.

## `ScalarTotComplex.jl` — total arithmetic on ℂ: arg 0 = 0, and the flags become polar

The question was whether `arg 0 = 0` can be a reserved-word definition like a/0 = 0 and log 0 = 0,
with the rest of complex arithmetic defined on top of it. It can, and the flags come out
simpler in ℂ than they were in ℝ. `TotComplex` stores a value in **polar form** — |z| and arg z
in units of π, so i = 1∠½π, i·i = 1∠1π = −1 *exactly*, e^{iπ} + 1 is an exact 0 and log(−1) = π∠½π
= iπ exactly (`sinpi` / `cospi` are exact at the quarter turns; the price is one rounding on the
Cartesian parts, real(3+4i) = 3.0000000000000004). Two flag systems in `TotNum`'s bits: GE / LE
bound the magnitude, exactly as they bound |x| on ℝ, and **AUNK — direction unknown — is
`TotNum`'s SUNK bit read in ℂ**: "sign unknown" is "arg ∈ {0, π} unknown", the two-point case of
a direction. There is no ℂ flag in ℂ (nothing leaves an algebraically closed field): a `TotNum`
in the ℂ seat converts to the *unknown seat* 0⟦≥≤ ∠?⟧ — a dangerous zero with unknown direction,
canonical for every zero carrying GE — and √(−1), log(−1), (−1)^½ deliver i, iπ, i where the real
type could only name the seat (34 336 such cash-ins in the twin check, plus 1 544 where ℂ gives
an exact 0 — 0·z = 0 for every z ∈ ℂ, where ℝ's ℂ had to stick).

**The reserved word in ℂ.** 0 is exactly 0 and has no direction: arg 0 = 0. Then z = |z|·e^{i arg z}
holds at 0 (0 = 0·1), u = z/|z| = z/0 = 0 is the one direction vector off the circle, and
log 0 = log|0| + i·arg 0 = 0 + 0i agrees with `TotNum` on the real axis — the complex extension does
not reopen the real decision. The limits are ε's, and **ε carries its direction**: an underflow is
MIN⟦≤⟧∠θ with θ kept ((1e-200∠0.3π)·1e-200 = MIN∠0.3π⟦≤⟧), so log ε = log MIN⟦≥⟧ + iθ; "0 is a
value without direction, ε is a limit with one". IEEE's ±0 with atan2(±0, −1) = ±π is the limit
reading. The price is the same one, stated: exp(log 0) = 1 ≠ 0, so powers are *not* derived from
exp(w·log z) at 0 — 0^w = 0 for w ≠ 0 (0^i = 0, not 1) and 0^0 = 1 stay definitions — and
u = e^{i·arg z} fails at 0 only, where x·(1/x) = 1 already fails.

**The rules, each one proven or dropped** (the audit principle of 2026-07-20). × ÷ are exact in
the direction and take `TotNum`'s magnitude algebra, with a bound on the divisor flipped (GE ↔ LE:
a reciprocal), a true zero absorbing and a/0 = 0 for *every* admissible a — ℝ's `/` is more
conservative there (any flag → ⟦≥≤⟧), so ℂ is stronger on 3 225 of the 9 216 real pairs and
identical on the rest. + − go through Cartesian: |t₁e^{iθ₁} + t₂e^{iθ₂}|² = t₁² + t₂² + 2t₁t₂cos Δ
is monotone in both magnitudes iff cos Δ ≥ 0, so a magnitude bound survives an addition only
within 90° and the direction survives only when the two are collinear (Δ = 0); past 90° the sum
can cancel — no bound, no direction. On ℝ, Δ ∈ {0, π} reproduces `TotNum`'s same-sign rule
exactly. √: |·|^½ monotone, the half-angle exact. exp: x = |z|cos θ grows with the magnitude
iff cos θ > 0 (the direction bit flips otherwise, `TotNum`'s negative-input rule), the result's
direction y = |z|sin θ moves unless sin θ = 0; the arg of e^{iy} is y mod 2π *exactly* (`rem2pi`
— y/π alone loses the direction of e^{i·MAX}, which is definite: MAX is an exact number). log:
`TotNum`'s two provable cells for the real part; |log z|² = (log|z|)² + θ² ≥ (log|z|)², so an
unknown direction still leaves a floor ⟦≥ ∠?⟧. Integer and real powers on the principal branch
with `TotNum`'s direction swap; AUNK stays through every power (2θ is as unknown as θ — on ℝ an
even power fixes the sign, in ℂ it fixes nothing). sin / cos: a flagged input says nothing; for
|Im z| > 700 the hyperbolic parts overflow but the direction does not — sin(x+iy) ≈
(e^{|y|}/2)(sin x + i·sgn(y)cos x) — so the magnitude saturates with the true direction (an
Inf component would have put it at an odd multiple of π/4: the oracle caught that lie).

**The oracle, lifted to ℂ** (`julia julia/audit_cplx.jl`): every flagged input names a set —
magnitudes ≥ / ≤ / any, direction fixed / any, a dangerous zero = anything — and the audit
enumerates it (magnitudes from 0⁺ to an "∞" of 10⁴⁰⁰, six directions), computes the true result
in `Complex{BigFloat}` (1 400 bits for exp / sin / cos, whose reduction of Im z = MAX needs them)
and falsifies the claim: unflagged ⟹ the shown value is the value, GE / LE ⟹ the bound holds,
no AUNK ⟹ the shown direction is the direction (to 10⁻⁹ in units of π). 25 values × 8 flag
sets × 20 unary operations and 11 × 11 values × 64 flag pairs × 4 binary ones: **931 472 checks,
0 violations**; 30 reserved-word definitions (a/0, log 0, arg 0, 0^−2, 0^i, i·i = −1, e^{iπ}+1 = 0,
log(−1) = iπ, ε∠θ keeping θ, exp(−MAX) = MIN⟦≤⟧, exp(i·MAX) with its exact direction, …) **0
violations**; the real axis against `TotNum` (16 values × 16 flags × 15 unary + 12² × 256 × 4
binary): **0 value mismatches, 0 contradictions**, identical flags on 8 906 of 10 046 sign-certain
cases and the rest *stronger* in ℂ (a zero with only ± is exactly 0 — no direction to be unsure
of; abs drops the ± `TotNum` keeps; the reciprocal swap; 0⟦≤⟧ + b = b), or *weaker* only where
ℂ's "direction unknown" is genuinely larger than ℝ's "sign unknown" (exp of x⟦±⟧: e^{iy} rotates;
even powers). Found on the way: the first oracle read ⟦≥≤⟧ as two bounds instead of none
(658 194 false violations — the negative control nobody asked for), a − a was 7e-16 because the
angle was rotated instead of the parts negated, and the sin/cos overflow direction above.

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
3039 checks, 0 violations (2026-07-20).

**Three more lie classes (2026-09-03)**, found while defining exp/log for `MultiU32`, and one
reversal. The oracle was blind in three places: it never checked an *unflagged* output (an
unflagged value claims "this is the value"), it never fed the extremes of Float64, and it never
fed a ⟦ℂ⟧ value back in. Widening it (BigFloat truths, ±MAX/±MIN/1e±200/±750 inputs, ℂ inputs,
the four binary operations over a 10 × 9-flag grid, the reserved-word definitions as
definitions) raised the count from 3 039 to **561 295 checks and exposed 82 825 violations in
the then-current code** — the negative control — of exactly three kinds. ① *Underflow to an
unflagged zero*: Float64 exhausts its subnormals and hands `_sat` an exact 0 it cannot tell
from a true zero, so `1e-200·1e-200`, `1e-200/1e200`, `0.5^2000`, `exp(−750)` and `exp(−MAX)`
all answered "0" while their true values are positive (43 cases; `+ − √` cannot do this — a
sum of Float64s is 0 only when it is exactly 0). Now the operation, which knows its operands
are nonzero, totalizes that 0 to ε = ±MIN⟦≤⟧ with the direction kept (`_uflow`). ② *ℂ was a
dead end*: the placeholder 0 of a ⟦ℂ⟧ value was consumed as a true zero by the next operation
— `exp(log(−1)) = 1`, `log(−1)·2 = 0`, `√−4·√−4 = 0` (82 000+ cases). ℂ is now sticky: every
operation with a ⟦ℂ⟧ operand returns the ℂ seat `0⟦≥≤±ℂ⟧` (the one exception is z⁰ = 1, the
empty product). ③ *`log 0`* — the reversal. The 2026-07-20 fix had made `log(0) = −MAX⟦≥⟧`
("−∞, saturated, flagged"). That treats the input 0 as a *limit* — which is exactly IEEE's
reading, the reason IEEE needs a signed ±0 to say which limit — and it is not this
arithmetic's reading: here underflow never produces 0 (it produces ε = ±MIN⟦≤⟧, direction
kept), so an input 0 is exactly 0, the reserved word, where log has no value and the
reserved-word definitions apply: **log 0 = 0**, in the same shelf as a/0 = 0 and 0·MAX⟦≥⟧ = 0.
The limits are ε's job and fall out of the flag rules — `log(MIN⟦≤⟧) = log MIN⟦≥⟧` ("−708.4 or
anything more negative"). The series say the same thing: no expansion of log converges at 0
(it is the singularity), and the *direction* of the divergence is decided by the form, not
the point — Σ(−1)ⁿ⁺¹(x−1)ⁿ/n runs to −∞ at 0, its mirror Σ(1−1/x)ⁿ/n (with 1/0 = 0) to +∞,
and the inversion-symmetric forms (2ᵏ(x^{1/2ᵏ} − x^{−1/2ᵏ})/2, or log(1/x) = −log x with 1/0 = 0)
give exactly 0; 0 is the fixed point of inversion, and v = −v has one solution. The cleanest
statement is the series **log x = Σₙ₌₁ (1/n)·((x−1)/x)ⁿ**: evaluated with the reserved-word
rules every term is ((0−1)/0)ⁿ/n = 0, so log 0 = 0 term by term — from a/0 = 0 alone (the
same series written with 1 − 1/x instead of (x−1)/x runs to +∞, because x/x = 1 is exactly
the identity total arithmetic denies at 0). Fed ε = MIN⟦≤⟧ instead, the same series diverges
and the flags say so — 1.03e307⟦≥≤±⟧ — which is the whole distinction in one formula: 0 is a
value, ε is a limit. The price is
stated, not hidden: log is not injective on the total line (log 0 = log 1 = 0, so
exp(log 0) = 1), and the homomorphism log(0·y) = log 0 + log y fails at 0 — the same place
x·(1/x) = 1 already fails. After the three fixes: `julia audit_flags.jl` → **561 295 checks,
0 violations**; `audit_total.jl` 720 / 0; `MultiU32`'s twin-check against `TotNum`, widened
to 16 × 16 flags × 12 value pairs × 7 ops (+ − × ÷ √ exp log, zeros and a value below 1
included): 0 mismatches — after one alignment in `TotNum`, √(0⟦≤⟧) = 0 exactly (⟦≤⟧ on a 0
means the true value *is* 0, as its own exp rule already read it), where `MultiU32` had the
stronger claim.
