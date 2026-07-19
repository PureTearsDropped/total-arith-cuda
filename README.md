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
⑤ flag-algebra oracle: 600,000 flagged-input cases, true values sampled from the
   admissible sets → lies 0 (one-sided bound, sign, and no-NaN contracts)
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

Requires a CUDA GPU. Falls back to CPU (correctness holds; throughput numbers won't).

### Julia port — `julia/TotalArith.jl`

The same semantics in **generic Julia**: written against `AbstractArray` with only
broadcasts + matmul, so the same functions run on `Array` (CPU) and are **CuArray-ready**
(CUDA.jl) without modification — Julia's multiple dispatch gives the CPU/GPU "backend swap"
for free.

```bash
julia julia/TotalArith.jl     # self-test, no packages needed (stdlib only)
```

Measured here (Julia 1.11.5, CPU):

```
① totality: 1,000,000 × mul/add/div → NaN/Inf 0, flag lies 0
② wiring swap: complex / quaternion / sedenion / cyclic ℤ/8 — violations 0/200 each
③ CPU throughput: ~1 M sedenion products/s (reference; GPU path untested here)
④ entry totalization + audit regressions: all green (same cases as the Python ④)
⑤ flag-algebra oracle: 300,000 flagged-input cases → lies 0
⑥ cross-validation vs cuda_total.py: 41 cases (adversarial mul/add/div, flagged
   additions, entry totalization, quaternion group_mul) — values AND flags
   bit-identical between the two implementations, after the audit fixes
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
