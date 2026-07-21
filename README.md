# total-arith-cuda

**Total arithmetic on the GPU (torch/CUDA) — a kernel that never emits `NaN`/`Inf`, plus a swappable structure tensor: change one table and the same kernel computes complex, quaternion, sedenion, matrix product, or convolution.**

> ⚠️ Written with AI assistance. Every claim ships with a command that reproduces it. Verify before relying on it.

日本語は下段に。

---

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

### Reproduce

```bash
python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python cuda_total.py         # self-test: adversarial totality, algebra swap, throughput
```

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

### `cuda_fused_solve.py` — the solve family as one fused kernel

`nsolve` (equation-solving division `L_a⁺x`) fused into a single Triton kernel: per element,
building `L_a` from the wiring table, K Ben-Israel iterations `X←X(2I−LX)` (multiplication
only), `y=X·x`, AND the two-tier verification (forward residual → exact / normal equations →
least-squares SING / neither → SING|INEXACT) all stay in registers — one launch, one HBM
round trip, `tl.dot(..., input_precision="ieee")` (no silent tf32). Semantics two-witnessed
against `torch.linalg.pinv(float64)`: regulars 3.1e-7, zero-divisor least-squares 5.5e-8,
flags honest, NaN poison propagated. Measured (RTX 5090, cd16): **B=1M: 27× vs batched
`pinv`, 19× vs unfused Ben-Israel — 53.5M solves/s**; small batches are launch-bound (1.7×).
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
candidate+verify). Third shelf **`IMPLS`**: bilinear ALGORITHMS in (U,V,W) normal form — same structure tensor T, different implementations (complex: naive R=4 vs Gauss R=3; 2×2 matmul: naive R=8 vs Strassen R=7; sedenion naive R=256); correctness is the tensor equation `Σ_r U⊗V⊗W ≡ T`, checked to 0.0, and algorithms COMPOSE by Kronecker product mirroring `tensor()` (gauss⊗gauss computes cd2⊗cd2 with R=9 < naive 16). Probes measure every combination; measured laws asserted in
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
  makes possible and Python cannot).
- **`HyperTranscend.jl`** (transcendental) — `exp`/`log`/`√`/`^` for a hypercomplex number of **any** M = 2^k, all as
  `f(x) = f(Lₓ)·e₀` (matrix function of the regular representation). Forward ops are total for
  every input incl. zero divisors; only inversion breaks — where `Lₓ` is singular — and there
  it names the value (`⟦zero-divisor⟧`) instead of `NaN`. The scalar `TotNum` is the M = 1 case.

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

**全域算術**と**「配線＝計算」**のGPU（torch/CUDA）実装。数 = `(val: float32, flag: uint8)`。フラグ = `GE`(≥・上飽和) / `LE`(≤・ε潰れ) / `SUNK`(符号不明)。

- **全域**: 溢れ→`±MAX`+`GE`、潰れ→`±MIN=ε`（向き保持）+`LE`、`a/0=0`、**`NaN`/`Inf` は決して出さない**。
- **配線表 = 構造テンソル `T[k,i,j]`**。`T` を差し替えると同じカーネルが別の代数に（複素・四元数・セデニオン・巡回畳み込み、いずれも違反0）。
- **広く貯めて最後に1回丸め**。群積/MACはfloat64で貯めて最後に一度だけ飽和（＝positの*quire*と同じ規律）。

**正直な但し書き**: フラグは*全域化イベントのみ*（`±MAX`/`ε`飽和・0除算）。float32の最近接丸めはフラグしない（方向を持たず片側境界にできないため）。

### 再現方法

上記コマンド。RTX 5090 実測値は上段の通り。CUDA GPU 必須（CPUフォールバックは正しさは保つがスループット値は出ない）。

---

## Related repositories

- **[total-arith-hardware](../../total-arith-hardware)** — the same total arithmetic + wiring, from primitive gates up to synthesizable SystemVerilog / FPGA.
- **[varpro-powersum-nn](../../varpro-powersum-nn)** — where total arithmetic pays off in learning: totalized gradients keep poisoned data from killing a fit.

These three are, in effect, **three backends of one total-arithmetic contract** (CPU / GPU / hardware). A planned next step is a pluggable backend so a model can run its total arithmetic on any of them.

## 興味を持ったら / If this interests you

これは利用条件ではありません。ただの声かけです — もしこの方向性に興味を持って、議論したい・一緒に発展させたい・仕事として相談したい等があれば、この repo の Issue で気軽にどうぞ。（連絡は GitHub 経由で OK、本名は不要です。）

*Not a term of use — just an open door. If this direction interests you and you'd like to discuss it, develop it together, or talk about it as work, feel free to open an Issue. Reach me via GitHub; no real name needed.*

## License

Zero-Clause BSD (0BSD). See `LICENSE`.
