# total-arith-cuda — the Total Bilinear Machine (TBM)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21489922.svg)](https://doi.org/10.5281/zenodo.21489922)

**A small computer for honest arithmetic.** Its core is one *totalized bilinear* multiply
`c = Wᵀ((U·a)⊙(V·b))` over numbers that are `(value, quality-flag)` pairs — it **never
emits `NaN`/`Inf`**, every algebra (complex … sedenion, matrices, convolution, *your own*)
is a swappable table, exp/log/sqrt/solve are *programs* built on that one instruction, and
the same program runs **bit-identically on CPU (torch), GPU (fused Triton), and
auto-generated SystemVerilog gates**.

> ⚠️ Written with AI assistance. Every claim ships with a command that reproduces it. Verify before relying on it.

日本語は下段に。

## Three doors — pick the one that matches you / 三つの入口

| you are… | what you get here | start at |
|---|---|---|
| **"I want GPU math that never silently fails"** | `(val, flag)` numbers: overflow→`±MAX`+flag, `a/0=0`, `NaN` never exists; **solve that returns a least-squares answer + `SING` flag instead of crashing on singular/zero-divisor systems** — 67.4 M solves/s, 13–35× vs batched `pinv` | `cuda_total.py`, `cuda_fused_solve.py` |
| **"I work with quaternions / octonions / my own algebra"** | register an algebra from **two small tables** → exp/log/sqrt/sin/…, five association modes, division in three senses, element-wise maps, all **derived automatically and verified** — matrices over your algebra included; quaternion-rotation ops at parity speed with roma/kornia, plus flags they don't have | `nested_registry.py` (`table_alg`, `nop`, `bop`, `emap`, `nsolve`) |
| **"I care about compilation / hardware"** | a 6-instruction machine ([TBM_SPEC.md](TBM_SPEC.md)); programs compile to CPU / fused Triton / SystemVerilog RTL and pass a **bit-exact conformance test with adversarial NaN/Inf injection**; ternary wiring contract → straight-line kernel codegen | `TBM_SPEC.md`, `run_everywhere.py`, `cuda_fused.py` |

---

## Research Use and Reproducibility / 研究利用と再現性

This software implements totalization, status propagation, singularity handling, and verification rules that differ from conventional floating-point arithmetic.

When these arithmetic semantics affect research results, it is advisable, for reproducibility, to identify this repository by URL and to record the commit ID used, together with any arithmetic settings that affect the results.

本ソフトウェアは、通常の浮動小数点演算とは異なる全域化・状態伝播・特異点処理・検証規則を実装しています。これらの算術意味論が研究結果に影響する場合は、再現性のため、本リポジトリを URL で特定し、使用したコミット ID と、結果に影響する算術設定（例: honesty = evidence/coarse/bare、width = f64/f32、フラグ除外の有無）を記録することを推奨します。

## What this is (EN)

The GPU "height" of a wider project on **total arithmetic** and **"wiring = computation"** (see *Related repositories*). A single, small torch library:

- **A number is `(val: float32, flag: uint8)`.** Flag bits — defined precisely, in the
  **absolute-value** sense (the sign is carried by `val` itself and is trustworthy unless `SUNK`):
  - `GE` (1): `|true| ≥ |val|` — the magnitude saturated upward.
  - `LE` (2): `|true| ≤ |val|` — the magnitude collapsed (to ε).
  - `GE|LE`: *no bound* — the value makes no magnitude claim.
  - `SUNK` (4): the sign of `val` is not trustworthy.
- **Total:** overflow → `±MAX` + `GE`; underflow → `±MIN = ε` (direction preserved) + `LE`; `a/0 = 0`; **`NaN`/`Inf` are never produced** — enforced at the public constructor too (`Tot(x)` totalizes NaN → `(0, no-bound+SUNK)`, ±Inf and out-of-range → `±MAX+GE`, subnormal → `±MIN+LE`).
- **Wiring table = structure tensor `T[k,i,j]`.** Swap `T` and the same kernel becomes a different algebra. Complex (M=2), quaternion (M=4), sedenion (M=16), cyclic convolution ℤ/8 (M=8) all verified with zero violations against reference products.
- **Accumulate wide, round once.** Group products / MAC accumulate in float64 and saturate (round) exactly once at the end — the same discipline as a posit *quire*.

**Honest caveat.** The flags mark *totalization events only* (saturation to `±MAX`/`ε`, division by zero). Ordinary float32 round-to-nearest is **not** flagged, because nearest rounding has no direction and cannot be turned into a one-sided bound. (Measured: the 320,019 cases that differ from the float64 truth with no flag are all explained by float32 nearest rounding; zero saturation-flag lies.)

## The Total Bilinear Machine (TBM) — one multiply instruction, three silicons

The pieces of this project assemble into **one machine** ([TBM_SPEC.md](TBM_SPEC.md)): a
computer whose only multiplying instruction is the totalized bilinear contraction
`c = Wᵀ((U·a)⊙(V·b))` over `(val, flag)` operands. Six instructions total
(TOTALIZE / BILIN / LINMAP / AXPY / NORM / CHECK); everything else — exp, solve, FFT
convolution, law discovery — is a *program* (macro), and that smallness is the point.
("One multiply instruction" does **not** mean the machine can only multiply: BILIN
realizes *any* bilinear map, LINMAP/AXPY are its degenerate linear/additive forms, and
higher operations are program compositions — the claim is that one multiplying semantics
suffices, the way one ALU suffices in a CPU.)

Every instruction carries an **honesty dial**, because the cost of honesty is graded, not
flat (measured, RTX 5090):

| dial | what it keeps | measured cost |
|---|---|---|
| `evidence` | full forensic flags (pattern rule P0–P4, dangerous zeros, sign tracking) | **free** at control-loop batch sizes; 37× at bulk |
| `coarse` | never-lying coarse flags (whole-row GE\|LE\|SUNK on any contamination) | **1.13× ≈ free** at bulk |
| `bare` | values only | 1× |

so *forensic flags at the boundary, coarse flags inside* is measured-optimal, not a
compromise. `tbm.py` is the assembler (a thin layer: all semantics live in the audited
modules it calls); `run_everywhere.py` is the conformance test — **the same program runs
on CPU (torch), GPU (fused Triton), and auto-generated SystemVerilog gates (iverilog RTL
simulation, 48,222-gate sedenion components), with bit-identical values and bit-identical
flags, adversarial Inf/NaN injection included** (passed 2026-07-21).

*Compile once, run on three silicons, never lie.*

### The algebra shelf: two tables in, a verified function library out (2026-07-23)

The **wiring normal form is ternary** (TBM_SPEC §1.5): coefficients live in {−1, 0, +1}
(× power-of-2 diagonals), so the linear stages are exact in every backend, `R` honestly
counts the true multiplies, and wirings **compile to straight-line kernels**
(`cuda_fused.compile_wiring`: quaternion product 6.4× vs dense einsum, values/flags
bit-identical). Register your own algebra from two tables — `table_alg(name, mul, sig)`
(path + sign, ternary-gatekept) — and the whole operation family is *derived from the
table*, nothing hand-written per algebra:

- `nop(A, op, x, bracket=…)` — exp/sin/cos/sinh/cosh/log/sqrt/cbrt/inv in **five
  association modes** (left / right / symmetric-Jordan / left-action / right-action —
  kept as separate implementations; identities are measured, not assumed). Candidates
  verify their defining identity or flag `INEXACT`. Measured law: the five modes agree
  ⟺ power-associativity (sedenions agree at 1e-15; `mat2⟨𝕆⟩` is the first break, 0.31 —
  declared, not silenced). Action modes reach beyond the series radius (PSD matrix sqrt
  without rescaling).
- `bop(A, (α,β), x, y, path)` — **binary ops generated from pencil coordinates**
  `αL+βR`: mul (1,0), Jordan (½,½), commutator (1,−1) = ad = L−R, anticommutator (1,1),
  *any* (α,β); `path` ∈ cell / operator-matrix / solve (pseudo-inverse, two-tier
  verified) / solve-inv (true inverse — splits observably from pseudo-inverse at zero
  divisors). `nsolve` = division in three senses (left/right/symmetric).
- `emap(cell, f, X)` — element-wise map over matrices/containers (the *activation
  function* shape; f may be a shelf op in any mode **or an arbitrary function**, entry/exit
  totalized); `nnormalize` = a/‖a‖ with `0→0` (unitization; the unit sphere is
  multiplicative exactly through the Hurwitz four — measured, sedenions put zero divisors
  *on* the unit sphere).
- `ekernel` / `ekernel_gpu` / `fused_ekernel` — the scalar (1×1) case kernelized:
  bit-identical to the loop at 5,700× (numpy) / +92× (GPU) / +16× (Triton fusion), with a
  **pre-proof mode** (`tot="auto"`): one range check proves "no overflow/NaN can occur
  inside", the per-step totalization is then provably an identity and is skipped —
  honesty tax 3.19× → **1.41×**, and on subnormal-dense CPU data this *reverses* the race
  (total arithmetic 3.03× **faster** than raw IEEE at 100% density, flat at any density).

Everything above runs unchanged over `mat_over(cell, N)` (matrices over your algebra),
tensor compositions, and all ALGS shelf citizens.

### `total_core.py` — one convention for two halves (2026-08-24)

This repository has two halves that speak **different flag languages on purpose**, and a
second external review asked the fair question: *what does a flag mean here versus there?*
The answer is not to merge them — they assert different things — but to write the map down
once and machine-check it. `total_core.py` (numpy only, no torch) is that one place.

| bit | order layer (`cuda_total.Tot`, GPU) | verify layer (`nested_registry.Nel` / `Hyper`) |
|-----|------------------------------------|-----------------------------------------------|
|0x01 | `GE` — true value is **≥** this    | `SING` — no unique inverse/solution (zero divisor) |
|0x02 | `LE` — true value is **≤** this    | `CPLX` — the result left the reals             |
|0x04 | `SUNK` — sign unknown              | `OVER` — saturated at ±MAX                     |
|0x08 | *(unused — free bit)*              | `INEXACT` — the defining identity was **not** verified |

The order layer states **bounds on a value**; the verify layer states **what happened to a
computation**. Same bits, different meanings — so the two words must never be OR'ed. The one
legal exception is bit 3: the order layer never uses it, which is why `cuda_fused_solve` can
ride `INEXACT` alongside `GE|LE|SUNK` in a single `uint8` (documented, no longer a bare
magic number).

`to_verify` / `to_order` bridge the two, and the correspondence is **not** a bijection:
`GE|LE|SUNK` ⇔ `SING` is the only point where the meanings coincide; `GE` and `LE` both
collapse to `OVER` (direction is lost); `INEXACT` and `CPLX` have **no order-layer image at
all** — they are claims about a verification, not bounds on a value. So `to_order` returns
`(flag, residue)` and never drops those bits silently. A test asserts the bridge only ever
*weakens* a claim.

Two more shared conventions moved here, and both were quietly split before:

- **The structure tensor's index order.** The numpy half carries `Alg.T[i,j,k]` ("the
  coefficient of e_k in e_i·e_j"); the GPU half carries `wiring_tensor` as `T[k,i,j]` (so the
  pattern rule in `group_mul` can take output row `T[k]` in one slice). Both orders are right
  for their side, so they stay — but they are now *named* (`T_ijk` / `T_kij`), converted in
  exactly one place (`to_kij` / `to_ijk`), and checked against each other for every algebra.
  An einsum does not care if you transpose a cube, which is why this needed a test rather
  than a convention.
- **Cayley–Dickson.** There were two implementations; the second one (in `nested_registry`)
  turned out to be dead code. One survives, and a test builds the table *both* ways — XOR
  routing (`cd_omega`) and the exhaustive basis product (`cd_prod`, the path the Julia twin
  takes) — and asserts they agree.

**Consequence, not just tidying:** `wiring_tensor` now accepts *any* `Alg` (or a preset name),
so Clifford, Grassmann/dual numbers, `mat_n`, and tensor compositions run on the GPU kernel —
before this, the "swap T, get a different algebra" claim held only for `'cd'` and `'cyclic'`
on the CUDA side. The ternary gate (TBM_SPEC §1.5) still guards the door: a table with ½
coefficients (`jordan(mat2)`) is refused unless you pass `ternary=False` and say so out loud.

日本語: 二つの旗の語彙は**統合しない**（別のことを主張しているため）。統合の代わりに、ビット
地図・層間の橋・構造テンソルの添字順・Cayley–Dickson の符号表を `total_core.py` の一箇所に
置き、`test_total_arith.py` が両半身の一致を恒久検査する。副産物として numpy 側は torch を
要求しなくなり、任意の代数が GPU カーネルに載るようになった。

### Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python cuda_total.py         # self-test: adversarial totality, algebra swap, throughput
python test_total_arith.py   # boundary tests: index order, flag bridge, algebra swap
                             #   TOTAL_ARITH_SLOW=1 also runs every module's self_test
python tools/ab_check.py     # "nothing changed": bitwise A/B of this tree vs main
                             #   --negative-control checks that the check has teeth
```

`self_test` draws a **fixed sample** by default so a number that moves means the code moved,
not the dice; `TOTAL_ARITH_SEED=1` (2, 3, …) runs the same contracts on a different sample and
prints the seed, so a failure is reproducible. See [`tools/README.md`](tools/README.md).

Measured on an **RTX 5090**:

```
① totality: 1,000,000 × mul/add/div  →  NaN/Inf 0,  flag lies 0
② wiring swap (same kernel, different T):
     complex   M= 2   violations 0/200
     quaternion M= 4  violations 0/200
     sedenion  M=16   violations 0/200
     cyclic ℤ/8 M= 8  violations 0/200
③ throughput: 68.0 M sedenion products/s (batch 1e6, flags + no-NaN included)
④ entry totalization + audit regressions: NaN/Inf leak none; (+MIN,LE)+(−MIN,=) → no-bound+SUNK
⑤ flag-algebra oracle: 600,000 flagged-input cases × two witnesses (incl. flagged
   zeros, lone SUNK, 10^6 multipliers) → lies 0 (six contracts: one-sided bound,
   exact-magnitude, sign, zero-display, witness-agreement, no-NaN)
⑥ group_mul oracle (pattern rule): dense ± / sparse / positive-cyclic → lies 0 each,
   retention 0.5% / 89% / 39%; eight audit counterexamples kept as regressions
```

### External adversarial review (2026-07-19)

An independent AI (ChatGPT) was asked to *refute* this code. Outcome, in the open:

- **Real bug found & fixed** — the addition flag rule lied under cancellation
  (`(+MIN,LE)+(−MIN,=)` returned `(0, LE)`). New sign-aware rule: same-sign inputs OR their
  flags (sound: magnitudes add monotonically); when cancellation is possible and any bound
  is present, the result drops to *no-bound + SUNK*. Kept as regression ④.
- **Real gap found & fixed** — the public constructor admitted `NaN`/`Inf` (so `0 ×
  Tot(1e300)` produced NaN). Entry totalization added in both implementations.
- **Doc bug (ours)** — the README glossed `GE` as a signed bound ("true ≥ val"), while the
  design (and tests) use absolute-value bounds with the sign carried by `val`. The reviewer
  correctly showed the signed reading is untenable; the definition above is now exact.
- **Test blind spot (fair)** — the self-test never exercised flagged inputs. The oracle
  test ⑤ (admissible-true-value sampling) now covers the flag algebra directly.
- **Independently confirmed** — the Cayley–Dickson sign convention was hand-checked by the
  reviewer against the standard quaternion table (i·j=k, j·k=i, k·i=j): consistent.

**Round 2** (same day): the reviewer re-read the fixed version and found `group_mul`
dropped `SUNK` (an unknown-sign input component yielded a confidently-signed output).
Confirmed by execution, fixed — and while writing the oracle test ⑥ for it, the test
found a *deeper* soundness gap the reviewer hadn't named: in a multiply–accumulate,
**mere magnitude bounds (GE/LE) on an input also invalidate the output's sign claim**
(uncertainty shifts the cancellation balance: 6−10 vs 6−2). This led to a **pattern rule** (zero
lies is absolute; keep the maximum within it), judged per output component: **P0** all
contributing terms exact → keep claims; **P1** a single live term → cancellation is
impossible, the scalar rule survives (with SUNK the magnitude claim is kept, only the
sign is unknown); **P2** all live terms share one known sign → the sum is monotone
(all-GE→GE, all-LE→LE, sign = the common sign); **P3/4** mixed signs or SUNK among ≥2
terms → no-bound + SUNK. Measured retention on flagged rows: dense random ±: 0.5% (the
blanket rule was near-optimal there), sparse 2-component products: 89%, all-positive
cyclic convolution: 100% — with **0 lies** in every scenario (oracle ⑥).

**Round 3** (same day): the reviewer found the pattern rule's premise itself was broken —
`live = (val ≠ 0)` conflated *displayed* zero with *true* zero, so a term like
`(0, no-bound+SUNK)` (true value arbitrary — exactly the shape `_sat(NaN)` produces) was
silently dropped as a dead term, and `group_mul` then claimed exact results. Confirmed by
execution, fixed: a displayed zero with no `GE` bit is *definitely zero* (droppable); a
displayed zero **with** a `GE` bit is a *dangerous zero* — every component it touches
falls to no-bound + SUNK. While strengthening the oracle per the reviewer's blind-spot
list (flagged zeros, lone `SUNK`, unbounded multipliers), **our own oracle then found a
fourth bug the reviewer hadn't**: SUNK-only *addition* claimed exact magnitude, but
`(2,SUNK)+(3,SUNK)` has true value ±2±3, i.e. |true| ∈ {1,5} — cancellation breaks the
magnitude too. The addition rule now drops to no-bound+SUNK whenever cancellation is
possible and *any* flag (not just a bound) is present. All four counterexamples are
permanent regressions; the oracle now checks four contracts (one-sided bound,
exact-magnitude, sign, no-NaN).

*Design note:* this bug family — "unknown" encoded as awkward corners of `(val, flag)` —
is exactly what a **4-valued digit** `{0, 1, −1, unknown}` representation eliminates by
construction (the hardware repo's `quadsign.py` / `sed/trit_status.py` explored this;
there, unknown is a first-class value and its algebra is the digit product itself).

**Round 4** (same day): the reviewer found the dangerous-zero semantics existed **only in
`group_mul`** — the same encoding `(0, GE)` meant "sign unknown" there but "sign known" in
scalar `tot_mul`/`tot_div` (two semantics for one representation): `(0,GE)×(3,=)` and
`(1,=)/(0,GE)` lacked `SUNK`. Fixed: the dangerous-zero rule now applies to scalars too.
The oracle gained the two contracts the reviewer showed it was blind to: *no `SUNK` ⟹ a
displayed zero is truly zero*, and *no `SUNK` ⟹ two independently sampled witnesses never
disagree in sign* (one witness cannot detect sign-indeterminacy). Alongside, the zero
doctrine was made explicit and implemented: **a true zero is signless** (direction lives
in `±MIN`, never in `0`) **and absorbs**: `(0, exact) × (x, any flags) = (0, no flags)` —
the other factor's uncertainty vanishes, per `x×0=0` exact.

Requires a CUDA GPU. Falls back to CPU (correctness holds; throughput numbers won't).

### Demos → [robust-attitude-control](../../robust-attitude-control)

The drone / 6-DoF flight demos that exercise this library live in their own showcase repo,
**[robust-attitude-control](../../robust-attitude-control)**: fault-tolerant attitude and
position control that survives sensor spikes (IEEE crashes; total arithmetic + flag
rejection completes the mission), quaternion product via `group_mul`, dual-quaternion pose,
motor mixing as a wiring, and a differential-flatness speed envelope. Those demos also
surfaced a real library bug here — `group_mul` crashed on wiring tables with empty output
rows (matrix-vector wirings produce them; hypercomplex ones never do), now fixed.

### `cuda_fused.py` — the fused kernel (Triton): buys latency, sells no semantics

One Triton kernel fuses the float64 MAC (saturate once) with the full pattern-rule flag
algebra (P0–P4, definite/dangerous zeros, E1 retention, sign-consistency). Contract:
**flags bit-identical, values float32-identical** to `cuda_total.group_mul` — asserted over
adversarial batteries (560k components, 3 wirings, zero mismatches). Measured (RTX 5090,
cd16, flagged): the legacy flagged path has a batch-independent ~36ms floor (per-component
`.nonzero()` forces GPU↔CPU syncs); the fused kernel removes it — **B=1: 36.5ms → 63µs
(582×)**, B=1024: 387×, B=16k: 32×, B=1M: parity. Large-batch clean inputs stay faster on
the existing einsum path (memory-optimal there). The win is exactly the control-loop /
per-sample regime — a 1 kHz attitude loop fits in 63µs, not in 36ms.

### `cuda_fused_pipeline.py` — the M/N/O × (U,V,W) fused-kernel compiler

The generalization of the hand-written fused kernels: **one Triton kernel source**,
specialized per shelf configuration via `constexpr` (dead branches eliminated at JIT).
`compile_pipeline(impl, program, tape...)` takes any IMPLS citizen (N + U/V/W; R zero-padded
to a power of two), applies it implicitly (M — no L materialized), and runs the O-layer
program (product, or series = tape × chained bilinears + scaling + squarings — the
gate_series skeleton on GPU) entirely in registers, one HBM round trip. Verified: the same
source specializes to R=3 (Gauss) … R=256 (sedenion naive) products and to exp/sin tapes
(vs float64 references, ~1e-7). Measured honestly: the fusion dividend depends on rank —
sedenion R=256 series is compute-bound (fusion 1.0×, stated), while **xor8×WH (R=8) series
gets 5.0× (277M exp/s): low-rank IMPLS × fused O-program is a compounding product of the
two dials**. Flag-semantics fusion remains with `cuda_fused`/`cuda_fused_solve`; this module
is the generality demonstration (values, v1).

### `cuda_fused_solve.py` — the solve family as one fused kernel

`nsolve` (equation-solving division `L_a⁺x`) fused into a single Triton kernel: per element,
building `L_a` from the wiring table, K Ben-Israel iterations `X←X(2I−LX)` (multiplication
only), `y=X·x`, AND the two-tier verification (forward residual → exact / normal equations →
least-squares SING / neither → SING|INEXACT) all stay in registers — one launch, one HBM
round trip, `tl.dot(..., input_precision="ieee")` (no silent tf32). Semantics two-witnessed
against `torch.linalg.pinv(float64)`: regulars 3.1e-7, zero-divisor least-squares 5.5e-8,
flags honest, NaN poison propagated. Measured rigorously (RTX 5090, cd16, CUDA-event
timing, 5-run warmup, median[p10,p90], solve-body only — entry totalization outside the
timed region): **13–35× vs batched `pinv` at every batch size** (B=64: 0.07ms = 13×;
B=1M: 14.8ms = 34× = 67.4M solves/s), 15–24× vs unfused Ben-Israel. An earlier wall-clock
benchmark undersold small batches (0.5×) — the artifact was Tot conversion inside the timed
loop; the review-driven rigor upgrade REVERSED that finding. `quality_report()` backs the
flags to the distribution tails (B=200k, 25% zero divisors): clean elements' forward
residual max 2.4e-6 (p99.9 6.6e-7); SING elements' normal-equation residual max 3.6e-7
while their forward residual is honestly O(1) — the two-tier verification works as quality
assurance, not decoration.
Julia CPU twin: `nsolve_batch` in `julia/NestedSeries.jl` (allocation-free single pass,
2.3× vs the naive loop, 0.08M solves/s single-threaded ≈ single-core BLAS floor; values and
flags match the sequential `nsolve_left` to 2.2e-15).

### `hyper_transcend.py` — transcendental functions for any-M hypercomplex (Python twin of Julia)

`exp / log / sqrt / ^ / sin / cos / sinh / cosh` and a linear-ODE mover `left_action(a,x0,t)`
for a hypercomplex number of **any** M = 2^k, via `f(x) = f(L_x)·e₀` (matrix function of the
regular representation; `cuda_total.py` supplies the wiring table). Same three tiers as the
[Julia twin](julia/README.md): **safe forward** (exp/trig/`x^{p≥0}`, total for every input incl.
zero divisors), **candidate** (sqrt/log/fractional power — computed, then the defining identity
verified by residual; else flagged `~`/INEXACT), **break** (log / `x^{neg}` of a zero divisor →
`⟦零因子⟧`). NaN/Inf named at construction so no matrix routine can crash.

```bash
python hyper_transcend.py            # per-dimension identity self-test (M = 1..16)
python hyper_transcend.py --audit    # adversarial totality audit → NaN/Inf 0, exceptions 0, false-flags 0
```

### `hyper_quad.py` — the quadratic-algebra reduction: scalar functions + bilinear, no matrix function (2026-09-02, research)

⚠️ 生成AI使用・要検証（research ブランチ・未監査）

Every Cayley–Dickson element `x = a·e₀ + v` (v ⊥ e₀) satisfies **v² = −|v|²·e₀** — checked
by `_mul` for M = 1..32, not assumed — so for a real-analytic f

    f(a + v) = Re f(a + i|v|) + (v/|v|) · Im f(a + i|v|)

and the M×M `expm/logm/funm` of `hyper_transcend.py` collapses to **1 rsqrt + 3–4 scalar
functions + (3M+2) multiplies**: `n2 = |v|²` (bilinear) → `r = rsqrt(n2)` (n2 = 0 → 0, the
a/0 = 0 rule, which makes the real case fall out with u = 0) → `u = v·r`, `s = n2·r` → complex
scalar `f(a + i s)` assembled from exp/expm1/log/sqrt/rsqrt/sin/cos → `Re·e₀ + u·Im`. The
scalar backend is pluggable: `np` (float64) or **`spec`** — the f64 coefficient-tape
specification of [total-arith-hardware/remez](../total-arith-hardware/remez) (Remez tapes,
≤ 0.5 ulp near mode), whose order-layer flags (GE/LE/SUNK) ride along per component
(`QHyper.oflag`); the bilinear steps run on exact rationals and truncate to 53 bits at each
tape entry (truncation = |shown| ≤ |true| → GE, composed by the E1 product rule). Verify-layer
flags come from **events** only: saturation → OVER, an imaginary part dropped because u = 0
(log/sqrt of a negative real) → CPLX, a failed defining identity → INEXACT.

What was checked (`python hyper_quad.py`, `--spec` for the tape backend; both PASS):
- **agrees with the matrix path** (same principal branch, Re > 0, |v| < π) for exp / sin / cos /
  sinh / cosh / log / sqrt / x^½ / x^2.5, M = 1,2,4,8,16 × 12 points: worst relative
  difference 2.6e-15 … 7.2e-15, SING/CPLX verdicts identical;
- identities exp(log x) = x, log(exp x) = x, sqrt(x)² = x, sin²+cos² = 1, cosh²−sinh² = 1,
  x² = x·x, (x^½)² = x at ≤ 1e-15 (5.7e-14 for sin²+cos² where cosh²|v| ≈ 25); the only
  exemptions are the 7 cases M = 1 ∧ a < 0 where log/sqrt drop the imaginary part and say CPLX;
- |v| → 0 continuity: f(a + v·10^-k), k = 0..20, converges monotonically to f(a)·e₀, is exact at
  v = 0, and the flag never jumps;
- totality: NaN / Inf / 1e±200 / 0 / MAX × 9 functions → 0 NaN, 0 exceptions;
- `x^p` for p = m/n (n ≤ 8) is **verified** by (x^p)^n = x^m in ℝ[x] ≅ ℂ — the matrix path
  cannot verify p = 2.5 and flags INEXACT 60/60; the reduction verifies it 60/60.

Two findings the matrix path could not show:
1. **The log of a zero divisor exists.** For the sedenion zero divisor z = e₃ + e₁₀,
   `hlog` says SING (L_z is singular, `logm` fails) but the reduction returns
   w = ½ln 2 + (π/4)(e₃ + e₁₀), and **exp(w) = z holds under both exps** (reduction 1e-16,
   matrix `expm` 4e-16). The singularity of L_z is a property of the representation, not of
   the algebra ℝ[z] ≅ ℂ that contains z. Uniqueness is not claimed; `hyper_transcend.py` keeps
   its SING verdict until the dispatch question is decided (see below).
2. **Order flags cannot pass through a non-monotone function of an uncertain argument.**
   With the `spec` backend, log / sqrt / sin come out `11` (no bound, sign known) but exp and
   cosh come out `111` (sign unknown) on every component — because s = n2·rsqrt(n2) carries
   rsqrt's near-mode `11`, and sin(s)/cos(s) of an argument with no bound has no bound *and*
   no sign. That is the honest order-layer answer, not a bug: the direction vocabulary has no
   word for "within 2^-57"; carrying error width is the four-value / interval layer's job.

Cost (reduction measured by the backend counters; matrix path analytic, `expm` Padé-13 with
s = 0 squarings ≈ (6 + s + ⅓)·M³ multiply-adds — a lower bound, `funm`/`logm` cost more):

| M | reduction exp (mul / add / scalar fn) | matrix expm (MACs) | gates f64 est.: reduction / matrix |
|---|---|---|---|
| 2 | 6 / 0 / 4 | 51 | 3.8 M / 2.9 M (matrix ×0.8) |
| 4 | 12 / 2 / 4 | 405 | 4.0 M / 23 M (×5.7) |
| 8 | 24 / 6 / 4 | 3,243 | 4.5 M / 185 M (×41) |
| 16 | 48 / 14 / 4 | 25,941 | 5.5 M / 1,480 M (×270) |
| 32 | 96 / 30 / 4 | 207,531 | 7.5 M / 11,800 M (×1,600) |

Gate estimate: rsqrt = 1,536 k (measured f64, `remez/gate_logroot.py`), exp / sin / cos ≈ 687 k
(the measured f64 exp of `remez/gate_funcs.py`; sin / cos are specified but not yet gate-built —
assumed the same order, their heavier reduction makes the true figure higher), multiply ≈ 33 k,
add ≈ 24 k, MAC ≈ 57 k. The crossover lies between M = 2 and M = 4 (complex numbers: a 2×2
matrix function costs about the same as four scalar functions and, with the measured rsqrt, is
20 % cheaper); from quaternions on the reduction wins by M³. (The first draft assumed rsqrt ≈ exp
and put the crossover at M = 2; the measured rsqrt gate count replaced that.)

Not done / open: **atan2 is outside the six tape functions** — the imaginary part of log (and
of pow) uses float64 `atan2` with flag `11`, stated as such in the code; an atan tape (kernel
atan(√y)/√y on y ≤ tan²(π/8) after three unit-vector half-angle steps, each one rsqrt) is the
next piece. Whether `hyper_transcend.py` should *dispatch* CD wirings to this path (which would
change its documented "log of a zero divisor → SING" verdict) is left to the maintainer.

```bash
python hyper_quad.py           # np backend: identity / matrix-path / continuity / zero-divisor / totality
python hyper_quad.py --spec    # same with the (A) f64 tape spec as the scalar backend (order flags ride along)
python hyper_quad.py --cost    # op counts per M
```

### `total_pipeline.py` — U → V(O,N,M) → W, named

External review observed that `cuda_total.py` already runs as U (entry totalization) →
V (fused float64 MAC) → W (saturate-once + pattern-rule flags), with N (the wiring tensor)
swappable and M implicit (the einsum applies `L_a` without materializing it) — "already that
architecture, just unnamed." This module gives it the names and the common interface
**without touching the five-round-audited core**: `apply(op, a, b, algebra=…)` is the one
gateway (asserted bit-identical to direct kernel calls), `Algebra.from_kind/.from_registry`
fills the N slot — the second **bridges the entire ALGS shelf** (dual quaternions, Clifford,
Grassmann, …) onto the audited kernel, flags included — `Lmatrix` exposes the explicit M for
verification (implicit ≡ explicit asserted), and `Pipeline('gmul', algebra)` is the
declarative composition. `python total_pipeline.py` runs the five-way self-test.

### `nested_series.py` / `nested_registry.py` — nesting and the twin shelves

`nested_series.py`: the original three-layer experiment (matrix ⊃ sedenion ⊃ digit) —
operation = coefficient tape on one skeleton; non-associativity infects upward; left-exp
and left-log are NOT an inverse pair for non-associative cells (cell-swap decisive test).

`nested_registry.py` (Python twin of `julia/NestedSeries.jl`): the generalization — every
algebra is a structure tensor `T[i,j,k]` (product = one einsum; `jordan`/`lie` = literally
symmetrize/antisymmetrize T). Twin preset shelves: **`ALGS`** (13 named algebras incl.
Grassmann Λn — Λ1 = dual numbers ⇒ forward-mode AD `f(a+ε)=f(a)+f′(a)ε`; Clifford;
`dualquat` = Λ1⊗ℍ rigid-body pose) × **`OPS`** (`nop(A,'sqrt',x)` — forward vs
candidate+verify). Fourth shelf **`MAPS`** — maps BETWEEN algebras, homomorphy measured not claimed:
`AlgMap` + `map_verify` (falsifies `M(a·b)=M(a)·M(b)` and unit preservation; a random matrix
is caught as a non-homomorphism — negative control asserted). First citizens: **DFT** (the
time-domain convolution algebra `cyc8` → the frequency algebra `diag8` = its Wedderburn
normal form) and IDFT, both verified homomorphic, unit-preserving (`DFT(δ)=all-ones`) and
mutually inverse. The frequency algebra joins ALGS (`diag_alg`: idempotents = ideal bandpass
filters, zero divisors everywhere = dead bands — deconvolution's ill-posedness as algebra),
and the **convolution theorem joins IMPLS** (`cyclic8_fft`: (U,V,W)=(DFT,DFT,IDFT/n), rank
R=64→8, `ΣUVW≡T` exact — FFT proper is the butterfly factorization for applying these
matrices fast). DSP dictionary: filter=multiply, bandpass=idempotent, dead band=zero
divisor, deconvolution=solve with SING naming the lost frequencies. **Walsh–Hadamard joins
as the XOR-group's DFT** (user's recall): `xor_alg(n)` (the untwisted-CD group algebra —
this fabric's native wiring, all characters real ±1) is diagonalized by WH with `ΣUVW≡T`
**exactly 0.0** and rank R=n — beating cyclic ℤ/8's real rank 11 at n=8 with R=8, transforms
that are pure add/sub wiring (zero multipliers), measured precision amplification 0.94×
(better than naive), and **bit-exact end-to-end on integer inputs** (±1 transforms + /8
exponent shift). **Complex Walsh–Hadamard** (Chrestenson-4, `z4z2` = `tensor(cyc4,cyc2)`) extends the free
zone: entries {±1,±i} (×i = swap+negate = wiring), exact ΣUVW≡T, and the review-corrected
real-rank ladder of the three abelian groups of order 8 — `(ℤ/2)³`: ℝ⁸→8, `ℤ/4×ℤ/2`:
ℝ⁴⊕ℂ²→10, `ℤ/8`: ℝ²⊕ℂ³→**11** (12 only if constants are restricted to rationals — the
price of a machine that cannot hold √2 exactly). The rank-10 claim is IMPLEMENTED, not just
derived: `z4z2_rank10` compresses the spectrum by conjugate symmetry (4 real + 2 complex
channels, Gauss 3-mult each), a multiplication counter proves exactly 10 real multiplies per
product, `ΣUVW≡T` verified, and integer inputs stay end-to-end bit-exact. Honest scope:
XOR/quaternary convolutions are dyadic-shift-invariant filtering, not time-shift FIR — same
phenomenon, different groups; and the {±1,±i} boundary marks "exact and free", not
"impossible beyond" (√2 constants can be made cheap via constant multipliers/shift-add,
just not free-and-exact).

Third shelf **`IMPLS`**: bilinear ALGORITHMS in (U,V,W) normal form — same structure tensor T, different implementations (complex: naive R=4 vs Gauss R=3; 2×2 matmul: naive R=8 vs Strassen R=7; sedenion naive R=256); correctness is the tensor equation `Σ_r U⊗V⊗W ≡ T`, checked to 0.0, and algorithms COMPOSE by Kronecker product mirroring `tensor()` (gauss⊗gauss computes cd2⊗cd2 with R=9 < naive 16). Probes measure every combination; measured laws asserted in
`python nested_registry.py`: exp∘log ⟺ power-associativity; ⊗-partner must be
commutative AND associative; BCH repairs at s⁴ through octonions (Artin) and s³ at
sedenions; Jacobi breaks at octonions.

### Julia — `julia/` (three modules, see `julia/README.md`)

- **`HyperAlgebra.jl`** (basic algebra) — array/batch total arithmetic + swappable wiring
  tensor (`group_mul`), `CuArray`-ready (below).
- **`ScalarTot.jl`** (scalar) — total arithmetic as a Julia `Number` (`TotNum <: Real`): overloads the
  operators so **existing generic code runs on it unchanged** — an ODE solver from
  OrdinaryDiffEq.jl integrates with `TotNum` and the flag names *where/which-way* the run
  left the representable range (the "used, not demo" bridge that Julia's multiple dispatch
  makes possible and Python cannot). Its semantic oracle (`julia/audit_flags.jl`, 561 295
  checks) is what forced the 2026-09-03 readings recorded in `julia/README.md`: an underflow
  that Float64 has already collapsed to 0 is still ε = ±MIN⟦≤⟧, never 0; ℂ is sticky; and
  **log 0 = 0** — the reserved word 0 is a value, not a limit (log x = Σ(1/n)((x−1)/x)ⁿ is 0
  term by term under a/0 = 0), while the limit −∞ belongs to ε: log(MIN⟦≤⟧) = log MIN⟦≥⟧.
  (The CUDA side's `log` in `cuda_total.py` is a Mercator candidate + exp-verify and marks
  its log 0 as `INEXACT` rather than returning 0 — flagged, not silent, but not yet aligned.)
- **`HyperTranscend.jl`** (transcendental) — `exp`/`log`/`√`/`^` for a hypercomplex number of **any** M = 2^k, all as
  `f(x) = f(Lₓ)·e₀` (matrix function of the regular representation). Forward ops are total for
  every input incl. zero divisors; only inversion breaks — where `Lₓ` is singular — and there
  it names the value (`⟦zero-divisor⟧`) instead of `NaN`. The scalar `TotNum` is the M = 1 case.
- **`MultiF32.jl`** — **float128 / float256 / float512 (`F128` `F256` `F512`) built from Float32
  `+ - * fma` only**, with `ScalarTot`'s flags. Not an expansion (double-double style limbs
  hit Float32's exponent floor at ≈ 128 bits) but block floating point: 18-bit integer digits
  in Float32 lanes + one shared exponent; fma with the constant `1.5·2⁴¹` splits products
  exactly, 6 bits of headroom keep every column sum exact. Correctly rounded `+ − × ÷ √`
  (exact result → nearest-even; ÷ √ = Newton candidate + exact-residual verify), 0 mismatches
  against MPFR for all three types in `julia julia/MultiF32.jl`. Register (`NTuple`) kernels,
  0 allocations, bit-identical to the Vector reference (`ab_multif32.jl`); the same file runs
  unchanged as a CUDA.jl kernel, bit-identical to the CPU (`bench_multif32_cuda.jl`): on an
  RTX 5090, F512 × at 4.7 ns and F128 × at 0.62 ns per operation — 5–21× MPFR (the C library,
  one core) for + ×, 1.0–4.5× for ÷ √, and only 1–2 % of the GPU's FP32 peak; on the CPU it is
  7–23× slower than MPFR for + × and 76–146× for ÷ √ (numbers and the three measured reasons
  in `julia/README.md`).
- **`ScalarTotComplex.jl`** — total arithmetic on ℂ (2026-09-03): a polar `TotComplex` (|z|, arg/π —
  i·i = −1, e^{iπ}+1 = 0 and log(−1) = iπ exact) whose flags are `TotNum`'s read in ℂ — GE / LE on
  the magnitude, the sign bit as "direction unknown" — and **arg 0 = 0 as a reserved-word
  definition** (0 has no direction; ε = MIN⟦≤⟧∠θ carries one, so log ε = log MIN⟦≥⟧ + iθ), the
  ℂ seat of the real type cashed in as a value (√−1 = i). Every rule is falsified against
  `Complex{BigFloat}` truths (`audit_cplx.jl`: 931 472 checks, 0 violations; 30 definitions; the
  real axis against `TotNum` with 0 contradictions). Details in `julia/README.md`.
- **`MultiU32.jl`** — the same `F128` `F256` `F512` on **29-bit integer digits** (`UInt32` limbs,
  `UInt64` columns — MPFR's limb arithmetic at the width a GPU multiplies natively, no carry
  flag needed): same interface, flags and rounding, same MPFR battery (0 mismatches), and
  **written for a warp**: no data-dependent tuple index, no data-dependent loop exit — barrel
  shifters, normalize-then-round at a static bit position, selects instead of sign branches.
  Bit-identical to its first (plain, loop-and-index) version `MultiU32Ref.jl` on 63 000 results
  (`ab_multiu32.jl`, negative control included). ÷ and √ take their candidate from a **Newton
  ladder in fixed point** — a step only as wide as the accuracy it produces, nothing normalized
  or rounded between the seed and the candidate, the residual read as the low digits of the
  product in two's complement, and √ finished by one Karp–Markstein step — which is 2.8–3.6× and
  3.5–5.6× over the full-width Newton loop with **every result still bit-identical** (the exact
  residual decides the last bit, the candidate only proposes it). **exp and log** (2026-09-03):
  correctly rounded by Ziv's test in a K+2-digit working type with fallbacks at K+6 and 2K+6
  digits (the 2P+ bits of the hard cases) and, past that, the truncated value with ⟦≥⟧ instead
  of an exception — total by construction; MPFR bit-identity on 3 × 22 000 inputs plus the
  constructed midpoint cases, 1.7–3.3× MPFR C on the CPU (correctly rounded in every
  measured cell, max 0.500 ulp, where libquadmath's log reaches 0.78 and the Float64
  expansions 5–495 ulps on their own scale), RTX 5090 F128 exp / log at 3.0 / 4.6 ns and F512
  at 10.7 / 23.6 ns per operation — 149–296× one MPFR core, bit-identical to the CPU; the
  semantics are `TotNum`'s
  (log 0 = 0 the reserved word, exp(−MAX) = +MIN⟦≤⟧, log(−x) = 0⟦ℂ⟧, ℂ sticky), twin-checked at
  16 × 16 flags × 12 values × 7 operations. CPU: 1.6–3.6× slower than MPFR
  (the C library, same precision, same operands) for + ×, 4.6–6.6× for ÷, 2.9–5.3× for √ — and
  slower than the Float64
  expansions (`Double64`, `Float64x2/x4`), which are not correctly rounded and carry fewer bits
  (`julia/bench_libs.jl`, with an accuracy check). RTX 5090, one array per field (SoA — the
  array-of-structs layout costs 1.4–1.7× on + ×, measured): F512 + × at 0.20 / 0.29 ns per
  operation, ÷ √ at 2.3 / 1.6 ns — 113× / 143× / 37× / 97× MPFR on one core, 21× / 16× / 36× the
  Float32-digit version; the add within 1.3–1.6× of the measured memory bandwidth, the
  multiply within 1.9×; `Float64x2` at 106 bits sits at the memory floor on the same GPU
  (0.03 ns for all four operations) and `Float64x4` at 212 bits beats F256 by 1.15–3.5×
  (`julia/README.md`; an earlier version of this line said "1.9–2.4× faster than MPFR" and
  "950× / 750× / 85×" — that MPFR column was Julia's allocating `BigFloat` wrapper on wider
  operands, retracted).

The port below is `julia/HyperAlgebra.jl`: written against `AbstractArray` with only broadcasts
+ matmul, so the same functions run on `Array` (CPU) and are **CuArray-ready** (CUDA.jl) —
Julia's multiple dispatch gives the CPU/GPU "backend swap" for free.

```bash
julia julia/HyperAlgebra.jl     # self-test, no packages needed (stdlib only)
```

Measured here (Julia 1.11.5, CPU):

```
① totality: 1,000,000 × mul/add/div → NaN/Inf 0, flag lies 0
② wiring swap: complex / quaternion / sedenion / cyclic ℤ/8 — violations 0/200 each
③ CPU throughput: ~1 M sedenion products/s (reference; GPU path untested here)
④ entry totalization + audit regressions: all green (same cases as the Python ④)
⑤ flag-algebra oracle: 300,000 flagged-input cases → lies 0 (four contracts)
⑥ group_mul oracle (pattern rule): same three scenarios — lies 0, retention 0.5%/88%/40%
⑦ cross-validation vs cuda_total.py: 49 cases (adversarial mul/add/div, flagged
   additions, entry totalization, quaternion group_mul incl. SUNK/GE inputs) —
   values AND flags bit-identical between the two implementations, after all fixes
```

Shape parity note: Python's `group_mul` accepts arbitrary leading batch dims (einsum `...`);
the Julia port accepts `N×M` matrices and plain `M`-vectors (a reviewer caught the
divergence; N-d batches remain Python-only).

The CuArray path is written but not exercised in this environment; if you run it on GPU,
an issue reporting the result (either way) is welcome.

---

## これは何か（JP）

**誠実な算術のための小さな計算機（総ビリニア機械 TBM）。** 核は全域化された双線形積
`c = Wᵀ((U·a)⊙(V·b))` ただ1つ。数 = `(val: float32, flag: uint8)`、フラグ = `GE`(≥・上飽和) / `LE`(≤・ε潰れ) / `SUNK`(符号不明)。

**三つの入口**（冒頭の表と同じ）: ①「黙って壊れないGPU数値計算が欲しい」→ `cuda_total.py`・`cuda_fused_solve.py`（特異/零因子でも落ちず最小二乗解+`SING`旗、67.4M solves/s）。②「四元数・八元数・自作代数を使う」→ `nested_registry.py`（表2枚 `table_alg` で登録すると exp/log/sqrt/solve 一式が自動導出・検算つき）。③「コンパイル・ハードウェアに興味」→ `TBM_SPEC.md`・`run_everywhere.py`（CPU/GPU/RTL でビット一致）。

- **全域**: 溢れ→`±MAX`+`GE`、潰れ→`±MIN=ε`（向き保持）+`LE`、`a/0=0`、**`NaN`/`Inf` は決して出さない**。
- **配線表 = 構造テンソル `T[k,i,j]`**。`T` を差し替えると同じカーネルが別の代数に（複素・四元数・セデニオン・巡回畳み込み、いずれも違反0）。**配線正規形は三値 {−1,0,+1}**（TBM_SPEC §1.5）——だから配線は直線コードにコンパイルできる（`compile_wiring`: 四元数積 6.4×・ビット一致のまま）。
- **広く貯めて最後に1回丸め**。群積/MACはfloat64で貯めて最後に一度だけ飽和（＝positの*quire*と同じ規律）。
- **表2枚→演算族の自動導出**: `nop`（五モード: 左結合/右結合/対称/左作用/右作用——別実装のまま分け、一致は測って主張）・`bop`（二項演算＝ペンシル (α,β) 座標から生成: 積・Jordan・交換子=L−R・反交換子=L+R・任意係数、逆演算も自動）・`emap`（行列の要素写像＝活性化の形・任意関数可）・`nnormalize`（単位化・0→0）・`nsolve`（解く除算three面）。測った法則: 五モード一致⟺べき結合性（破れの初出は mat2⟨𝕆⟩）。
- **事前証明モード** (`tot="auto"`): 演算前の範囲検査1回で「級数中に事故は起き得ない」を証明したら内部検査を省略（ビット一致のまま税 3.19×→1.41×）。サブノーマル濃度が数%を超える CPU データでは素の IEEE より**速くなる**（100%で3.03×・全域側は濃度非依存フラット＝定時性）。

**正直な但し書き**: フラグは*全域化イベントのみ*（`±MAX`/`ε`飽和・0除算）。float32の最近接丸めはフラグしない（方向を持たず片側境界にできないため）。

**総ビリニア機械（TBM）**: 全6命令（掛け算はBILIN 1つ——「掛け算しかできない」ではなく「掛け算の意味論は1つで足りる」の意）。exp/solve/FFT/法則発見は全部「プログラム（マクロ）」。全命令が誠実さのダイヤル（証拠級/粗/裸）を持つ——誠実さの税金は一枚岩でなく傾斜だから（実測: 証拠級は制御ループ帯でタダ・大バッチ37×、粗は1.13×≒タダ）。`tbm.py` がアセンブラ、`run_everywhere.py` が適合試験で、**同じプログラムが CPU / GPU（融合Triton）/ 自動生成SystemVerilogゲート（RTLシミュ）で値・フラグともbit一致**（敵対的Inf/NaN注入込み・2026-07-21合格）。*Compile once, run on three silicons, never lie.*

- **規約は一箇所** (`total_core.py`, numpy のみ): 旗の二語彙（順序層 `GE/LE/SUNK` = 値の境界の主張 ／ 検算層 `SING/CPLX/OVER/INEXACT` = 計算に何が起きたかの主張）は**統合せず**、ビット地図と層間の橋 `to_verify`/`to_order` を書いて機械検査する。橋は全単射でなく、像を持たない旗（`INEXACT`/`CPLX`）は `residue` で返して黙って捨てない。構造テンソルの添字順（numpy `T[i,j,k]` ／ torch `T[k,i,j]`）も同様に、統一せず**名前で区別して変換を一点に**置き `test_total_arith.py` が両半身の一致を検査する。副産物: numpy 側が torch を要求しなくなり、任意の `Alg`（Clifford・Grassmann・行列代数・テンソル積）が `wiring_tensor` 経由でそのまま GPU カーネルに載る。

### 再現方法

上記コマンド。RTX 5090 実測値は上段の通り。CUDA GPU 必須（CPUフォールバックは正しさは保つがスループット値は出ない）。境界検査は `python test_total_arith.py`（GPU 不要・`TOTAL_ARITH_SLOW=1` で各モジュールの self_test も続けて走る）。

---

**Basis convention / 基底規約 (2026-09-06).** `cd_alg(16).T` is the same Cayley–Dickson table as total-arith-hardware's `OMEGA`
(checked: products identical to 0.0 with the index order `T[i,j,k] x_i y_j`), i.e. the XOR labelling `e_i e_j = ±e_{i XOR j}`.
It agrees element-wise with numpy-quaternion / Quaternions.jl for quaternions, but **not** with Octonions.jl for octonions, which
labels the imaginary units differently (same algebra, related by a signed permutation such as `p = (0,1,2,3,4,7,6,5)`,
`s = (+,−,+,−,−,−,−,−)`). See `sed/crosscheck_external.py` in total-arith-hardware.
`cd_alg(16).T` は total-arith-hardware の `OMEGA` と同じ表（添字順 `T[i,j,k]` で積が完全一致）。四元数は numpy-quaternion /
Quaternions.jl と要素ごとに一致するが、八元数は Octonions.jl と基底の番号付けが違うので数値はそのままでは互換でない（同型）。

## Related repositories

- **[total-arith-hardware](../../total-arith-hardware)** — the same total arithmetic + wiring, from primitive gates up to synthesizable SystemVerilog / FPGA.
- **[varpro-powersum-nn](../../varpro-powersum-nn)** — where total arithmetic pays off in learning: totalized gradients keep poisoned data from killing a fit.

These three are, in effect, **three backends of one total-arithmetic contract** (CPU / GPU / hardware). A planned next step is a pluggable backend so a model can run its total arithmetic on any of them.

## 興味を持ったら / If this interests you

これは利用条件ではありません。ただの声かけです — もしこの方向性に興味を持って、議論したい・一緒に発展させたい・仕事として相談したい等があれば、この repo の Issue で気軽にどうぞ。（連絡は GitHub 経由で OK、本名は不要です。）

*Not a term of use — just an open door. If this direction interests you and you'd like to discuss it, develop it together, or talk about it as work, feel free to open an Issue. Reach me via GitHub; no real name needed.*

## License

Zero-Clause BSD (0BSD). See `LICENSE`.
