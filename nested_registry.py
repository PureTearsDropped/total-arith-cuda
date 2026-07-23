#!/usr/bin/env python3
# ⚠️ AI-assisted; verify. / 生成AI使用・要検証
"""nested_registry — Python twin of julia/NestedSeries.jl: the M/N/O layers as freely
composable registries.  (nested_series.py is the original fixed three-layer experiment;
this is its generalization.)

  Everything is an `Alg` carrying its structure tensor T[i,j,k] (coefficient of e_k in
  e_i·e_j) as an ndarray — so a product is one einsum, and the combinators become tensor
  operations on T itself:

    N layer (cells)      : cd_alg (ℝ→sedenion), cyclic_alg (ℤ/M), matn_alg (n×n reals),
                           grassmann_alg (exterior; Λ1 = dual numbers = forward AD),
                           clifford_alg (geometric product)
    M layer (combinators): mat_over(alg,N), tensor(A,B) — built directly on T;
                           jordan(A) = symmetrize T, lie(A) = antisymmetrize T
    O layer (tapes)      : OPS — operators as data (kind, tape, shift, verify);
                           nop(A, 'sqrt', x) runs any preset on any Alg

  ALGS is the algebra preset shelf (a preset IS a composition: 'dualquat' is literally
  tensor(grassmann_alg(1), cd_alg(4))).  Probes (assoc/powerassoc/commut) MEASURE each
  combination — nothing is assumed.  Elements are total: NaN→0+SING, overflow→±MAX+OVER
  at every step; candidates verify their defining identity or carry INEXACT.

  Measured laws replicated from the Julia twin (self_test asserts them):
    · exp∘log verifies  ⟺  power-associativity holds (associativity NOT required —
      octonion/sedenion scalars pass at 1e-16)
    · a non-associative ⊗-base keeps power-associativity only for a commutative AND
      associative partner (the jordan pincer: commutative alone fails)
    · BCH repair scales s⁴ through the octonions (Artin) and degrades to s³ at the
      sedenions; Jacobi breaks already at the octonions (Malcev)
    · Λn: nilpotent generators ⇒ series terminate; Λ1 gives f(a+ε)=f(a)+f′(a)ε (AD)
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from cuda_total import cd_omega

SING, OVER, INEXACT = 0x01, 0x04, 0x08
MAXF = float(np.finfo(np.float64).max)


# ================================================================ the one interface
class Alg:
    """bilinear algebra: name, dim, unit, and the structure tensor T[i,j,k]"""
    __slots__ = ("name", "dim", "unit", "T")
    def __init__(self, name, unit, T):
        self.name, self.unit, self.T = name, np.asarray(unit, float), np.asarray(T, float)
        self.dim = len(self.unit)
    def __repr__(self): return f"{self.name}(dim {self.dim})"

def _from_mul(name, d, unit, mul):
    T = np.zeros((d, d, d))
    E = np.eye(d)
    for i in range(d):
        for j in range(d):
            T[i, j] = mul(E[i], E[j])
    return Alg(name, unit, T)

def rawmul(A, x, y):
    with np.errstate(over="ignore", invalid="ignore"):
        return np.einsum("i,j,ijk->k", x, y, A.T)

# ================================================================ N layer: cell registry
def _cdconj(x): return np.concatenate([x[:1], -x[1:]]) if len(x) > 1 else x.copy()
def _cdprod(x, y):
    n = len(x)
    if n == 1: return x * y
    h = n // 2
    a, b, c, d = x[:h], x[h:], y[:h], y[h:]
    return np.concatenate([_cdprod(a, c) - _cdprod(_cdconj(d), b),
                           _cdprod(d, a) + _cdprod(b, _cdconj(c))])

def cd_alg(M):
    "Cayley–Dickson: ℝ(1) ℂ(2) ℍ(4) 𝕆(8) sedenion(16) — wiring table via cd_omega"
    OM = cd_omega(M)
    T = np.zeros((M, M, M))
    for i in range(M):
        for j in range(M):
            T[i, j, i ^ j] = OM[i, j]
    return Alg(f"cd{M}", np.eye(M)[0], T)

def cyclic_alg(M):
    "group algebra of ℤ/M: eᵢeⱼ = e_{(i+j)%M} — commutative AND associative"
    T = np.zeros((M, M, M))
    for i in range(M):
        for j in range(M):
            T[i, j, (i + j) % M] = 1.0
    return Alg(f"cyc{M}", np.eye(M)[0], T)

def diag_alg(M):
    """周波数代数(各点積の対角代数): e_f·e_f = e_f, それ以外 0。単位元 = 全成分 1。
       巡回代数の Wedderburn 標準形 — DFT が 同型写像(MAPS 参照)。
       冪等元 e_f = 理想バンドパスフィルタ・f≠g の e_f·e_g=0 = 零因子だらけ
       (帯域の死 = デコンボリューション不良設定の 代数的正体)。"""
    T = np.zeros((M, M, M))
    for f in range(M):
        T[f, f, f] = 1.0
    return Alg(f"diag{M}", np.ones(M), T)

def xor_alg(n):
    """XOR群 (ℤ/2)^k の群環 (符号なし=捻れなしCD・可換結合): e_i·e_j = e_{i⊻j}。
       指標が全部±1の実数値 ⟹ DFT = ウォルシュ・アダマール・ランク = n ちょうど
       (2n−t の t=n の 最良ケース)。cd 族の ω≡+1 極限 = このファブリックの母語。"""
    T = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            T[i, j, i ^ j] = 1.0
    return Alg(f"xor{n}", np.eye(n)[0], T)

def _wh_matrix(n):
    "ウォルシュ・アダマール H[f,i] = (−1)^{popcount(f&i)} — ±1のみ(乗算器ゼロの変換)"
    return np.array([[(-1.0) ** bin(f & i).count("1") for i in range(n)] for f in range(n)])

def _cwh_matrix():
    """複素WH (Chrestenson-4) = F₄⊗H₂: 成分 {±1,±i} のみ。×i = 実虚スワップ+符号 = 配線。
       ℤ/4×ℤ/2 の指標変換 — 「厳密かつ乗算器ゼロの直接実装」の上限
       (指標がガウス整数の単数{±1,±i}に収まる限界。√2係数も定数乗算器/シフト加算で
       安くはできる — 絶対の壁でなく「厳密・無料」の壁)。
       位数8のアーベル群の実ランク階段(外部監査 2026-07-21 で ℤ/8 を訂正):
         (ℤ/2)³: ℝ⁸ (t=8) → 8 / ℤ/4×ℤ/2: ℝ⁴⊕ℂ² (t=6) → 10 / ℤ/8: ℝ²⊕ℂ³ (t=5) → 11
       (ℤ/8 は 実指標が k=0,4 の 2個+複素対3組。係数を有理数に制限すると x⁸−1 の
        ℚ分解 t=4 → 12 = √2 を厳密に持てない機械の値段)。"""
    F4 = np.array([[1,1,1,1],[1,-1j,-1,1j],[1,-1,1,-1],[1,1j,-1,-1j]])
    H2 = np.array([[1,1],[1,-1]], dtype=complex)
    return np.kron(F4, H2)

def matn_alg(n):
    "real n×n matrices as a dim-n² algebra — associative, with zero divisors"
    return _from_mul(f"mat{n}", n * n, np.eye(n).ravel(),
                     lambda x, y: (x.reshape(n, n) @ y.reshape(n, n)).ravel())

def _reorder_sign(a, b):
    cnt = 0
    for i in range(63):
        if (a >> i) & 1:
            cnt += bin(b & ((1 << i) - 1)).count("1")   # pairs (i∈A, j∈B, j<i)
    return 1.0 if cnt % 2 == 0 else -1.0

def _blade_alg(name, n, keep_overlap):
    D = 1 << n
    T = np.zeros((D, D, D))
    for a in range(D):
        for b in range(D):
            if a & b and not keep_overlap: continue      # Grassmann: eᵢ² = 0
            T[a, b, a ^ b] = _reorder_sign(a, b)
    return Alg(name, np.eye(D)[0], T)

def grassmann_alg(n):
    """exterior algebra Λℝⁿ (dim 2ⁿ): generators nilpotent ⇒ series TERMINATE.
       Λ1 = dual numbers a+bε: f(a+ε)=f(a)+f′(a)ε — forward-mode AD off the shelf."""
    return _blade_alg(f"Λ{n}", n, keep_overlap=False)

def clifford_alg(n):
    "Clifford Cl(n,0) (dim 2ⁿ): geometric product — Grassmann wiring, overlap → +1"
    return _blade_alg(f"Cl{n}", n, keep_overlap=True)

# ================================================================ M layer: combinators
def mat_over(cell, N):
    "N×N matrix over any Alg (row-major blocks); result properties: measure, don't assume"
    d = cell.dim; D = N * N * d
    T = np.zeros((D, D, D))
    at = lambda i, j: (i * N + j) * d
    for i in range(N):
        for j in range(N):
            for m in range(N):
                T[at(i, m):at(i, m) + d, at(m, j):at(m, j) + d, at(i, j):at(i, j) + d] += cell.T
    unit = np.zeros(D)
    for i in range(N): unit[at(i, i):at(i, i) + d] = cell.unit
    return Alg(f"mat{N}⟨{cell.name}⟩", unit, T)

def tensor(A, B):
    "A ⊗ B — two structure tensors multiplied: one einsum builds the composed wiring"
    dA, dB = A.dim, B.dim; D = dA * dB
    T = np.einsum("acp,bdq->abcdpq", A.T, B.T).reshape(D, D, D)
    return Alg(f"{A.name}⊗{B.name}", np.outer(A.unit, B.unit).ravel(), T)

def jordan(A):
    "symmetrized a∘b=(ab+ba)/2 = SYMMETRIZE T — commutative, associativity usually lost"
    return Alg(f"sym⟨{A.name}⟩", A.unit.copy(), (A.T + A.T.transpose(1, 0, 2)) / 2)

def lie(A):
    """antisymmetrized ½[a,b] = ANTISYMMETRIZE T — the order-only half.  Every product
       splits exactly: ab = a∘b + ½[a,b].  Ladder measured in self_test: BCH repairs at
       s⁴ through octonions (Artin), s³ at sedenions; Jacobi breaks at octonions."""
    return Alg(f"lie⟨{A.name}⟩", np.zeros(A.dim), (A.T - A.T.transpose(1, 0, 2)) / 2)

# ================================================================ total elements + ops
class Nel:
    __slots__ = ("c", "flag")
    def __init__(self, c, flag=0):
        self.c, self.flag = np.asarray(c, float), int(flag)
    def __repr__(self): return f"Nel({np.round(self.c, 4).tolist()}, flag={self.flag:#x})"

def _tot(c, f):
    c = np.asarray(c, float).copy()
    nan = np.isnan(c)
    if nan.any(): c[nan] = 0.0; f |= SING
    ovf = ~np.isfinite(c) | (np.abs(c) > MAXF)
    if ovf.any(): c[ovf] = np.sign(c[ovf]) * MAXF; f |= OVER
    return Nel(c, f)

def nel(A, c=None): return _tot(A.unit if c is None else np.asarray(c, float), 0)
def tmul(A, x, y): return _tot(rawmul(A, x.c, y.c), x.flag | y.flag)
def tadd(x, y): return _tot(x.c + y.c, x.flag | y.flag)
def tscale(x, s): return _tot(x.c * s, x.flag)
def commutator(A, x, y): return tadd(tmul(A, x, y), tscale(tmul(A, y, x), -1.0))

# ================================================================ O layer: operator shelf
def binom_tape(p):
    "coefficients of (1+u)^p"
    def c(k):
        v = 1.0
        for i in range(1, k + 1): v *= (p - i + 1) / i
        return v
    return c

TAPES = {
    "exp":  lambda k: 1.0 / math.factorial(k),
    "sin":  lambda k: 0.0 if k % 2 == 0 else (-1.0) ** ((k - 1) // 2) / math.factorial(k),
    "cos":  lambda k: 0.0 if k % 2 == 1 else (-1.0) ** (k // 2) / math.factorial(k),
    "sinh": lambda k: 0.0 if k % 2 == 0 else 1.0 / math.factorial(k),
    "cosh": lambda k: 0.0 if k % 2 == 1 else 1.0 / math.factorial(k),
}

def series(A, x, tape, order=20, bracket="left"):
    "Σ c_k x^k with powers built by the DECLARED bracket — one skeleton, many tapes"
    c = TAPES[tape] if isinstance(tape, str) else tape
    acc = tscale(nel(A), float(c(0)))
    P = nel(A)
    for k in range(1, order + 1):
        P = tmul(A, P, x) if bracket == "left" else tmul(A, x, P)
        ck = float(c(k))
        if ck != 0.0: acc = tadd(acc, tscale(P, ck))
    return acc

def nexp(A, x, **kw):  return series(A, x, "exp", **kw)
def nsin(A, x, **kw):  return series(A, x, "sin", **{"order": 21, **kw})
def ncos(A, x, **kw):  return series(A, x, "cos", **kw)
def nsinh(A, x, **kw): return series(A, x, "sinh", **{"order": 21, **kw})
def ncosh(A, x, **kw): return series(A, x, "cosh", **kw)

def nexp_ss(A, x, order=12, s=3, bracket="left"):
    "scaling-and-squaring exp — a DIFFERENT cell connection; agreement is measured"
    acc = series(A, tscale(x, 1.0 / 2 ** s), "exp", order, bracket)
    for _ in range(s): acc = tmul(A, acc, acc)
    return acc

def nlog(A, x, order=30, verify_order=20):
    "inverse ⇒ candidate: series log(1+X), verified by the safe forward exp, else INEXACT"
    X = tadd(x, tscale(nel(A), -1.0))
    y = series(A, X, lambda k: 0.0 if k == 0 else (-1.0) ** (k + 1) / k, order)
    resid = float(np.max(np.abs(nexp(A, y, order=verify_order).c - x.c)))
    return (y, resid) if resid < 1e-6 else (Nel(y.c, y.flag | INEXACT), resid)

def ninv(A, x, order=60):
    "1/x as the geometric tape Σ(1−x)^k — verified TWO-SIDED, INEXACT on zero divisors"
    u = tadd(nel(A), tscale(x, -1.0))
    y = series(A, u, lambda k: 1.0, order)
    resid = max(float(np.max(np.abs(rawmul(A, x.c, y.c) - A.unit))),
                float(np.max(np.abs(rawmul(A, y.c, x.c) - A.unit))))
    return (y, resid) if resid < 1e-6 else (Nel(y.c, y.flag | INEXACT), resid)

OPS = {
    "exp":  dict(kind="forward", tape=TAPES["exp"], shift=False, order=20, verify=None),
    "sin":  dict(kind="forward", tape=TAPES["sin"], shift=False, order=21, verify=None),
    "cos":  dict(kind="forward", tape=TAPES["cos"], shift=False, order=20, verify=None),
    "sinh": dict(kind="forward", tape=TAPES["sinh"], shift=False, order=21, verify=None),
    "cosh": dict(kind="forward", tape=TAPES["cosh"], shift=False, order=20, verify=None),
    "atan": dict(kind="forward", tape=lambda k: (-1.0) ** ((k - 1) // 2) / k if k % 2 else 0.0,
                 shift=False, order=41, verify=None),
    "log":  dict(kind="candidate", tape=lambda k: 0.0 if k == 0 else (-1.0) ** (k + 1) / k,
                 shift=True, order=30,
                 verify=lambda A, x, y: float(np.max(np.abs(nexp(A, y).c - x.c)))),
    "inv":  dict(kind="candidate", tape=lambda k: (-1.0) ** k, shift=True, order=60,
                 verify=lambda A, x, y: max(
                     float(np.max(np.abs(rawmul(A, x.c, y.c) - A.unit))),
                     float(np.max(np.abs(rawmul(A, y.c, x.c) - A.unit))))),
    "sqrt": dict(kind="candidate", tape=binom_tape(0.5), shift=True, order=40,
                 verify=lambda A, x, y: float(np.max(np.abs(rawmul(A, y.c, y.c) - x.c)))),
    "cbrt": dict(kind="candidate", tape=binom_tape(1 / 3), shift=True, order=40,
                 verify=lambda A, x, y: float(np.max(np.abs(
                     rawmul(A, rawmul(A, y.c, y.c), y.c) - x.c)))),   # (y·y)·y declared
}

def nop(A, name, x, order=None, bracket="left"):
    "run any preset on any Alg — forward: total; candidate: verified or INEXACT"
    op = OPS[name]
    arg = tadd(x, tscale(nel(A), -1.0)) if op["shift"] else x
    y = series(A, arg, op["tape"], order or op["order"], bracket)
    if op["kind"] == "forward": return y
    resid = op["verify"](A, x, y)
    return y if resid < 1e-6 else Nel(y.c, y.flag | INEXACT)

def list_ops():
    for nm in sorted(OPS):
        k = OPS[nm]["kind"]
        print(f"{nm:<7}{'forward (total)' if k == 'forward' else 'candidate (verified or INEXACT)'}")

# ================================================================ algebra preset shelf
ALGS = {
    "real":       lambda: cd_alg(1),
    "complex":    lambda: cd_alg(2),
    "quaternion": lambda: cd_alg(4),
    "octonion":   lambda: cd_alg(8),
    "sedenion":   lambda: cd_alg(16),
    "split":      lambda: cyclic_alg(2),
    "dual":       lambda: grassmann_alg(1),
    "grassmann2": lambda: grassmann_alg(2),
    "cl2":        lambda: clifford_alg(2),
    "cl3":        lambda: clifford_alg(3),
    "dualquat":   lambda: tensor(grassmann_alg(1), cd_alg(4)),
    "biquat":     lambda: tensor(cd_alg(2), cd_alg(4)),
    "m4real":     lambda: tensor(cd_alg(4), cd_alg(4)),
}
def alg(name): return ALGS[name]()

# ================================================================ UVW: the algorithm shelf
class Impl:
    """ONE bilinear algorithm in (U,V,W) normal form: c = Wᵀ((U·a) ⊙ (V·b)).
       R = U.shape[0] = number of scalar multiplications — the hardware cost.
       Correctness is a TENSOR EQUATION: Σ_r U[r,i]V[r,j]W[r,k] must equal the target
       algebra's T[i,j,k] — verified numerically, never assumed."""
    __slots__ = ("name", "U", "V", "W")
    def __init__(self, name, U, V, W):
        self.name = name
        self.U, self.V, self.W = (np.asarray(m) if np.iscomplexobj(np.asarray(m))
                                  else np.asarray(m, float) for m in (U, V, W))
    @property
    def R(self): return self.U.shape[0]
    def __repr__(self): return f"{self.name}(R={self.R})"

def impl_tensor(im): return np.einsum("ri,rj,rk->ijk", im.U, im.V, im.W)
def impl_verify(im, A):
    "0.0 ⟺ this algorithm computes exactly this algebra's product"
    return float(np.max(np.abs(impl_tensor(im) - A.T)))
def impl_mul(im, x, y):
    "run a product THROUGH the algorithm: R scalar multiplications, then wiring"
    return im.W.T @ ((im.U @ x) * (im.V @ y))

def naive_impl(A):
    "the trivial decomposition read off T: one multiplication per nonzero (i,j) pair"
    rows = [(i, j) for i in range(A.dim) for j in range(A.dim) if np.any(A.T[i, j])]
    U = np.zeros((len(rows), A.dim)); V = np.zeros((len(rows), A.dim)); W = np.zeros((len(rows), A.dim))
    for r, (i, j) in enumerate(rows):
        U[r, i] = 1.0; V[r, j] = 1.0; W[r] = A.T[i, j]
    return Impl(f"naive⟨{A.name}⟩", U, V, W)

def impl_kron(a, b):
    """algorithms COMPOSE like algebras do: the Kronecker product of two (U,V,W)s computes
       the tensor-product algebra, with R = R_a·R_b — the UVW mirror of tensor(A,B)."""
    return Impl(f"{a.name}⊗{b.name}", np.kron(a.U, b.U), np.kron(a.V, b.V), np.kron(a.W, b.W))

def prune_impl(im):
    """死に積の刈り込み (出力不変): U/V/W の行が全0の積 r を落とす。naive_impl は生成時に
       非零 (i,j) しか拾わない=生まれつき刈り込み済なので、これは合成・手書き (U,V,W) 用。
       方針: 併合はしない — 順序無視の併合は反対称部(=非可換の住処)を消し、可換代数でも
       a_i·b_j ≠ a_j·b_i で壊れる (2026-07-23 実測・total-arith-hardware の prune_uvw と同方針)。"""
    keep = [r for r in range(im.R)
            if np.any(im.U[r]) and np.any(im.V[r]) and np.any(im.W[r])]
    if len(keep) == im.R:
        return im
    return Impl(f"{im.name}∖dead", im.U[keep], im.V[keep], im.W[keep])

def _gauss_cd2():
    "complex multiply in 3 real multiplications (Gauss/Karatsuba) instead of 4"
    return Impl("gauss⟨cd2⟩", U=[[1, 0], [0, 1], [1, 1]], V=[[1, 0], [0, 1], [1, 1]],
                W=[[1, -1], [-1, -1], [0, 1]])

def _strassen_mat2():
    "Strassen: 2×2 matrix multiply in 7 multiplications instead of 8 (row-major A11..A22)"
    U = [[1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
         [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]]
    V = [[1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
         [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]]
    W = [[1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1], [1, 0, 1, 0],
         [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]]
    return Impl("strassen⟨mat2⟩", U, V, W)

def _dft_impl(n):
    """畳み込み定理を IMPLS の言葉で: 巡回畳み込みの (U,V,W) = (DFT, DFT, IDFT/n)。
       R = n (素朴 n² からの ランク削減 — Strassen と 同じ現象・FFT は この U,V,W を
       速く適用する butterfly 分解)。Σ_f U V W ≡ T_cyc は 虚部相殺込みで 厳密。"""
    F = np.exp(-2j * np.pi / n) ** np.outer(np.arange(n), np.arange(n))
    return Impl(f"dft⟨cyc{n}⟩", F, F, F.conj() / n)

IMPLS = {
    "complex_naive":    lambda: naive_impl(cd_alg(2)),          # R=4
    "complex_gauss":    _gauss_cd2,                             # R=3
    "quaternion_naive": lambda: naive_impl(cd_alg(4)),          # R=16
    "sedenion_naive":   lambda: naive_impl(cd_alg(16)),         # R=256
    "mat2_naive":       lambda: naive_impl(matn_alg(2)),        # R=8
    "mat2_strassen":    _strassen_mat2,                         # R=7
    "gauss2":           lambda: impl_kron(_gauss_cd2(), _gauss_cd2()),  # cd2⊗cd2, R=9<16
    "cyclic8_naive":    lambda: naive_impl(cyclic_alg(8)),      # R=64
    "cyclic8_fft":      lambda: _dft_impl(8),                   # R=8 ← 畳み込み定理
    "xor8_naive":       lambda: naive_impl(xor_alg(8)),         # R=64
    "xor8_wh":          lambda: Impl("wh⟨xor8⟩", _wh_matrix(8), _wh_matrix(8),
                                     _wh_matrix(8) / 8),        # R=8・U,V,Wは±1のみ・厳密
    "z4z2_cwh":         lambda: Impl("cwh⟨z4z2⟩", _cwh_matrix(), _cwh_matrix(),
                                     _cwh_matrix().conj() / 8), # R=8複素・{±1,±i}=乗算器ゼロ
    "z4z2_rank10":      lambda: _z4z2_rank10_impl(),            # ★実R=10(共役圧縮+Gauss)
}

def _z4z2_rank10_impl():
    """実ランク10の 明示的 (U,V,W) — 外部監査の指摘「8複素積のままでは 10 実乗算の
       実装ではない」への 回答。共役対称で スペクトルを 実4チャネル+複素2チャネルに 圧縮し、
       複素積は Gauss 3実乗算: 4 + 2·3 = 10 = 2n−t (ℝ⁴⊕ℂ², t=6)。U,V は 実(±1と和差のみ)、
       W は 1/8 の 有理係数 — 整数入力で 全段厳密。"""
    C = _cwh_matrix()
    RE, CX = [0, 1, 4, 5], [2, 3]
    rows = [C[r].real for r in RE]
    for r in CX:
        xr, xi = C[r].real, C[r].imag
        rows += [xr, xi, xr + xi]                 # Gauss の 3 つの 線形結合
    U = np.array(rows)
    Tt = tensor(cyclic_alg(4), cyclic_alg(2)).T
    Mm = np.einsum('ri,rj->ijr', U, U).reshape(64, 10)
    Wm = np.linalg.lstsq(Mm, Tt.reshape(64, 8), rcond=None)[0]
    Wm = np.round(Wm * 16) / 16                   # 有理係数に スナップ(厳密性は verify が担保)
    return Impl("rank10⟨z4z2⟩", U, U, Wm)
def impl(name): return IMPLS[name]()

def list_impls():
    "the shelf with each algorithm's measured badge: target algebra, R, and exactness"
    targets = {"complex_naive": cd_alg(2), "complex_gauss": cd_alg(2),
               "quaternion_naive": cd_alg(4), "sedenion_naive": cd_alg(16),
               "mat2_naive": matn_alg(2), "mat2_strassen": matn_alg(2),
               "gauss2": tensor(cd_alg(2), cd_alg(2)),
               "cyclic8_naive": cyclic_alg(8), "cyclic8_fft": cyclic_alg(8),
               "xor8_naive": xor_alg(8), "xor8_wh": xor_alg(8),
               "z4z2_cwh": tensor(cyclic_alg(4), cyclic_alg(2)),
               "z4z2_rank10": tensor(cyclic_alg(4), cyclic_alg(2))}
    print(f"{'preset':<18}{'computes':<12}{'R(乗算)':<10}exact?")
    for nm in sorted(IMPLS):
        im, tg = IMPLS[nm](), targets[nm]
        print(f"{nm:<18}{tg.name:<12}{im.R:<10}{'✓ 0.0' if impl_verify(im, tg) < 1e-12 else '✗'}")

# ================================================================ MAPS: 4枚目の棚(代数間の写像)
class AlgMap:
    """代数間の 線形写像 M: src → dst。準同型性 M(a·b) = M(a)·M(b) は 主張でなく
       map_verify が 測る。ALGS(何を)×OPS(どの関数)×IMPLS(どう)に 続く 4 枚目:
       MAPS(どの代数の 言葉に 翻訳するか)。第 1 号 = DFT(時間の畳み込み代数 → 周波数の
       各点積代数 = Wedderburn 標準形への 基底変換)。"""
    __slots__ = ("name", "src", "dst", "M", "factors")
    def __init__(self, name, src, dst, M, factors=None):
        self.name, self.src, self.dst = name, src, dst
        self.M = np.asarray(M)
        # factors: M の 疎因数分解(バタフライ等)。「FFT = 変換行列の疎因数分解」を
        # データとして持つ — ランク R(双線形の中身)と 直交する 第2のダイヤル(適用コスト)。
        self.factors = [np.asarray(f) for f in factors] if factors is not None else None
    def __call__(self, a):
        return self.M @ np.asarray(a)
    def apply_fast(self, a):
        "因子列を 右から 順に 適用 (n² → 因子の非ゼロ総数 ≈ n log n)"
        if self.factors is None:
            return self.M @ np.asarray(a)
        y = np.asarray(a)
        for f in reversed(self.factors):
            y = f @ y
        return y
    def __repr__(self): return f"AlgMap({self.name}: {self.src.name}→{self.dst.name})"

def map_verify(mp, rng=None, trials=6):
    """準同型性の 反証: max|M(a·b) − M(a)·M(b)| と 単位元の 保存 |M(1_src) − 1_dst|。
       ≈0 ⟺ 本物の 代数準同型(ランダム行列は ここで 落ちる — 陰性対照)。"""
    rng = rng or np.random.default_rng(0)
    worst = 0.0
    for _ in range(trials):
        a = rng.standard_normal(mp.src.dim)
        b = rng.standard_normal(mp.src.dim)
        lhs = mp(rawmul(mp.src, a, b))
        rhs = rawmul(mp.dst, mp(a), mp(b))
        worst = max(worst, float(np.abs(lhs - rhs).max()))
    unit = float(np.abs(mp(mp.src.unit) - mp.dst.unit).max())
    if mp.factors is not None:
        P = mp.factors[0]
        for f in mp.factors[1:]:
            P = P @ f
        worst = max(worst, float(np.abs(P - mp.M).max()))    # 因子の積 ≡ M (FFT=正しい分解か)
    return worst, unit

def _wh_factors(n):
    "WH のバタフライ: H_{2^k} = Π (I⊗H₂⊗I) — 全因子±1・行あたり非ゼロ2・積は厳密"
    H2 = np.array([[1., 1], [1, -1]])
    k = n.bit_length() - 1
    return [np.kron(np.kron(np.eye(2**s_), H2), np.eye(2**(k-1-s_))) for s_ in range(k)]

def _dft_factors(n):
    "DFT の Cooley-Tukey 全段: バタフライ + twiddle対角(=捻れの請求書) + 並べ替え"
    if n == 2:
        return [np.array([[1, 1], [1, -1]], dtype=complex)]
    w = np.exp(-2j * np.pi / n)
    h = n // 2
    P = np.zeros((n, n))
    for k in range(h):
        P[k, 2*k] = 1; P[k+h, 2*k+1] = 1
    D = np.diag(np.concatenate([np.ones(h), w ** np.arange(h)]))
    F2 = np.array([[1, 1], [1, -1]], dtype=complex)
    return ([np.kron(F2, np.eye(h)), D]
            + [np.kron(np.eye(2), S) for S in _dft_factors(h)] + [P])

def _dft_map(n):
    F = np.exp(-2j * np.pi / n) ** np.outer(np.arange(n), np.arange(n))
    return AlgMap(f"dft{n}", cyclic_alg(n), diag_alg(n), F, factors=_dft_factors(n))

def _idft_map(n):
    F = np.exp(-2j * np.pi / n) ** np.outer(np.arange(n), np.arange(n))
    return AlgMap(f"idft{n}", diag_alg(n), cyclic_alg(n), F.conj().T / n)

MAPS = {
    "dft8":  lambda: _dft_map(8),      # 時間→周波数 (畳み込み → 各点積)
    "idft8": lambda: _idft_map(8),     # 周波数→時間 (逆向きも 準同型)
    "wh8":   lambda: AlgMap("wh8", xor_alg(8), diag_alg(8), _wh_matrix(8),
                            factors=_wh_factors(8)),        # XOR群のDFT(実!)+バタフライ因子
    "cwh8":  lambda: AlgMap("cwh8", tensor(cyclic_alg(4), cyclic_alg(2)),
                            diag_alg(8), _cwh_matrix()),        # ℤ/4×ℤ/2 のDFT({±1,±i})
}
def amap(name): return MAPS[name]()

# ================================================================ probes: measure, don't assume
def _rand(rng, n, s=0.3): return Nel(s * rng.standard_normal(n))

def assoc_defect(A, rng, trials=4):
    w = 0.0
    for _ in range(trials):
        x, y, z = (_rand(rng, A.dim) for _ in range(3))
        w = max(w, float(np.max(np.abs(tmul(A, tmul(A, x, y), z).c - tmul(A, x, tmul(A, y, z)).c))))
    return w

def powerassoc_defect(A, rng, trials=4):
    w = 0.0
    for _ in range(trials):
        x = _rand(rng, A.dim); x2 = tmul(A, x, x)
        w = max(w, float(np.max(np.abs(tmul(A, x2, x).c - tmul(A, x, x2).c))))
    return w

def commut_defect(A, rng, trials=4):
    w = 0.0
    for _ in range(trials):
        x, y = (_rand(rng, A.dim) for _ in range(2))
        w = max(w, float(np.max(np.abs(tmul(A, x, y).c - tmul(A, y, x).c))))
    return w

def list_algs():
    print(f"{'preset':<12}{'realizes':<22}{'dim':<5}{'assoc':<7}{'pow-assoc':<11}commut")
    for nm in sorted(ALGS):
        A = ALGS[nm](); rng = np.random.default_rng(0)
        ad, pa, cm = assoc_defect(A, rng), powerassoc_defect(A, rng), commut_defect(A, rng)
        t = lambda v: "✓" if v < 1e-9 else "✗"
        print(f"{nm:<12}{A.name:<22}{A.dim:<5}{t(ad):<7}{t(pa):<11}{t(cm)}")


# ================================================================ self-test
def self_test():
    print("nested_registry — Python twin of NestedSeries.jl; laws measured, never assumed")
    rng = np.random.default_rng(7)

    # law: exp∘log verifies ⟺ power-associativity (across the combo zoo)
    combos = [cd_alg(4), cd_alg(8), cd_alg(16), cyclic_alg(6), matn_alg(2),
              mat_over(cd_alg(4), 2), mat_over(cd_alg(16), 2),
              tensor(cd_alg(4), cd_alg(4)), tensor(cd_alg(8), cd_alg(2)),
              mat_over(tensor(cd_alg(4), cd_alg(2)), 2)]
    print(f"{'algebra':<22}{'dim':<5}{'assoc':<7}{'pow-assoc':<11}{'exp∘log':<11}verdict")
    for A in combos:
        ad, pa = assoc_defect(A, rng), powerassoc_defect(A, rng)
        xnear = tadd(nel(A), tscale(_rand(rng, A.dim, 0.25), 0.5))
        _, resid = nlog(A, xnear)
        t = lambda v: "✓" if v < 1e-9 else "✗"
        print(f"{A.name:<22}{A.dim:<5}{t(ad):<7}{t(pa):<11}{resid:<11.1e}"
              f"{'✓ inverse pair' if resid < 1e-6 else '✗ INEXACT (structural)'}")
        assert (pa < 1e-9) == (resid < 1e-6), f"pow-assoc/verify mismatch on {A.name}"

    # tensor law: partner must be commutative AND associative (jordan pincer)
    for partner, keeps in ((cyclic_alg(3), True), (cd_alg(4), False), (jordan(cd_alg(8)), False)):
        Tn = tensor(cd_alg(8), partner)
        pa = powerassoc_defect(Tn, rng)
        assert (pa < 1e-9) == keeps, f"tensor law violated on {Tn.name}"
    print("tensor law: ⊗-partner must be commutative AND associative ✓ (jordan pincer)")

    # order machinery: exact split + Jacobi + BCH scaling gate (Artin)
    for M in (4, 16):
        A = cd_alg(M); a, b = _rand(rng, M, 0.4), _rand(rng, M, 0.4)
        recon = tadd(_tot(rawmul(jordan(A), a.c, b.c), 0), _tot(rawmul(lie(A), a.c, b.c), 0))
        assert float(np.max(np.abs(recon.c - tmul(A, a, b).c))) < 1e-12
    jac = lambda A, x, y, z: tadd(tadd(commutator(A, commutator(A, x, y), z),
                                       commutator(A, commutator(A, y, z), x)),
                                  commutator(A, commutator(A, z, x), y))
    jd = {M: float(np.max(np.abs(jac(cd_alg(M), *(_rand(rng, M, 0.4) for _ in range(3))).c)))
          for M in (4, 8, 16)}
    assert jd[4] < 1e-12 and jd[8] > 1e-3 and jd[16] > 1e-3
    ratios = {}
    for M in (4, 8, 16):
        A = cd_alg(M); ba, bb = rng.standard_normal(M), rng.standard_normal(M)
        r = []
        for s in (0.2, 0.1):
            a, b = Nel(s * ba), Nel(s * bb)
            lhs = tmul(A, nexp(A, a), nexp(A, b))
            z = tadd(tadd(a, b), tscale(commutator(A, a, b), 0.5))
            z = tadd(z, tadd(tscale(commutator(A, a, commutator(A, a, b)), 1 / 12),
                             tscale(commutator(A, b, commutator(A, b, a)), 1 / 12)))
            r.append(float(np.max(np.abs(lhs.c - nexp(A, z).c))))
        ratios[M] = r[0] / r[1]
    assert ratios[4] > 12 and ratios[8] > 12 and ratios[16] < 10
    print(f"order split exact ✓ ; Jacobi cd4 ✓ / cd8,cd16 ✗ (Malcev) ; BCH gate "
          f"cd4 {ratios[4]:.1f} / cd8 {ratios[8]:.1f} ≈ s⁴ (Artin) ; cd16 {ratios[16]:.1f} ≈ s³ ✓")

    # operator shelf
    A16 = cd_alg(16); xp = _rand(rng, 16, 0.3)
    for nm, f in (("exp", nexp), ("sin", nsin), ("cos", ncos), ("sinh", nsinh), ("cosh", ncosh)):
        assert float(np.max(np.abs(nop(A16, nm, xp).c - f(A16, xp).c))) < 1e-12
    xn = tadd(nel(A16), tscale(xp, 0.5))
    ys = nop(A16, "sqrt", xn)
    assert not (ys.flag & INEXACT) and float(np.max(np.abs(rawmul(A16, ys.c, ys.c) - xn.c))) < 1e-6
    assert not (nop(A16, "cbrt", xn).flag & INEXACT)
    A1 = cd_alg(1)
    assert abs(nop(A1, "atan", nel(A1, [0.5])).c[0] - math.atan(0.5)) < 1e-9
    Am = mat_over(cd_alg(16), 2)
    zsq = nop(Am, "sqrt", tadd(nel(Am), tscale(_rand(rng, Am.dim, 0.2), 0.5)))
    print(f"OPS shelf: presets ≡ named fns, √/∛ verified, atan ≡ ℝ ✓ ; "
          f"√ on mat2⟨cd16⟩: {'INEXACT (measured)' if zsq.flag & INEXACT else 'verifies'}")

    # algebra shelf + gems
    print("--- ALGS shelf (id-cards measured) ---")
    list_algs()
    D = alg("dual"); a0 = 1.2
    xd = nel(D, [a0, 1.0])
    for nm, f, fp in (("exp", math.exp, math.exp), ("sin", math.sin, math.cos),
                      ("sqrt", math.sqrt, lambda t: 1 / (2 * math.sqrt(t))),
                      ("inv", lambda t: 1 / t, lambda t: -1 / t ** 2)):
        y = nop(D, nm, xd)
        assert not (y.flag & INEXACT)
        assert abs(y.c[0] - f(a0)) < 1e-9 and abs(y.c[1] - fp(a0)) < 1e-9
    print("dual (=Λ1): f(a+ε)=f(a)+f′(a)ε for exp/sin/√/inv ✓ — forward AD off the shelf")
    DQ = alg("dualquat")
    assert assoc_defect(DQ, rng) < 1e-9
    _, rq = nlog(DQ, tadd(nel(DQ), tscale(_rand(rng, DQ.dim, 0.3), 0.5)))
    assert rq < 1e-6
    G2 = alg("grassmann2")
    xg = nel(G2, [0.0, 0.7, 0.4, 0.0])
    assert float(np.max(np.abs(nexp(G2, xg, order=3).c - nexp(G2, xg, order=30).c))) < 1e-15
    print(f"dualquat: associative, exp∘log {rq:.1e} ✓ ; Λ2: exp terminates (order 3 ≡ 30) ✓")
    # UVW algorithm shelf: same T, different (U,V,W) — HOW is swappable, WHAT is verified
    print("--- IMPLS shelf (algorithms; exactness = tensor equation, measured) ---")
    list_impls()
    for nm, A in (("complex_gauss", cd_alg(2)), ("mat2_strassen", matn_alg(2)),
                  ("sedenion_naive", cd_alg(16))):
        im = impl(nm)
        assert impl_verify(im, A) < 1e-12                     # ΣUVW ≡ T exactly
        xa, xb = rng.standard_normal(A.dim), rng.standard_normal(A.dim)
        assert float(np.max(np.abs(impl_mul(im, xa, xb) - rawmul(A, xa, xb)))) < 1e-12
    g2 = impl("gauss2")                                       # algorithms compose like algebras
    assert impl_verify(g2, tensor(cd_alg(2), cd_alg(2))) < 1e-12 and g2.R == 9
    print(f"kron composition: gauss⊗gauss computes cd2⊗cd2 exactly with R=9 (naive 16) ✓")
    print(f"cost of exp on cd2 (order 20 = 20 products): naive {20 * impl('complex_naive').R}"
          f" mults vs gauss {20 * impl('complex_gauss').R} mults — same answer, verified same T")
    # 死に積の門番: 棚は全員 刈り込み済 (prune が no-op)・人工死に積は落ちて T 厳密のまま
    for nm in IMPLS:
        assert prune_impl(impl(nm)).R == impl(nm).R, f"棚に死に積: {nm}"
    imc = impl("complex_gauss")
    dead = Impl("gauss+dead", np.vstack([imc.U, [1, 1]]), np.vstack([imc.V, [1, -1]]),
                np.vstack([imc.W, [0, 0]]))                    # W行=0 → どの出力too不使用
    imp = prune_impl(dead)
    assert imp.R == 3 and impl_verify(imp, cd_alg(2)) < 1e-12
    print(f"死に積: 棚 {len(IMPLS)} 種 prune no-op ✓ ; 人工死に積 R 4→3・ΣUVW≡T のまま ✓ (併合はしない)")
    # MAPS: 4枚目の棚 — DFT準同型・畳み込み定理・周波数代数
    print("--- MAPS shelf (代数間の写像・準同型性は測って主張) ---")
    fq = diag_alg(8)
    rngm = np.random.default_rng(3)
    assert commut_defect(fq, rngm) < 1e-9 and assoc_defect(fq, rngm) < 1e-9
    e2 = np.eye(8)[2]
    assert np.abs(rawmul(fq, e2, e2) - e2).max() < 1e-12          # 冪等 e_f²=e_f
    assert np.abs(rawmul(fq, e2, np.eye(8)[5])).max() < 1e-12     # 零因子 e_f·e_g=0
    dft = amap("dft8"); idft = amap("idft8")
    hom, unit = map_verify(dft, rngm)
    assert hom < 1e-12 and unit < 1e-12                            # DFT(e0)=全1=freqの単位元
    hom2, unit2 = map_verify(idft, rngm)
    assert hom2 < 1e-12 and unit2 < 1e-12
    a8 = rngm.standard_normal(8)
    assert np.abs(np.real(idft(dft(a8))) - a8).max() < 1e-12       # 逆写像
    rnd = AlgMap("random", cyclic_alg(8), diag_alg(8), rngm.standard_normal((8, 8)))
    homr, _ = map_verify(rnd, rngm)
    assert homr > 1e-2                                             # 陰性対照: 見抜ける
    imf = impl("cyclic8_fft")
    assert impl_verify(imf, cyclic_alg(8)) < 1e-12 and imf.R == 8  # 畳み込み定理=R=n
    xa, xb = rngm.standard_normal(8), rngm.standard_normal(8)
    assert np.abs(np.real(impl_mul(imf, xa, xb)) - rawmul(cyclic_alg(8), xa, xb)).max() < 1e-10
    print("  freq代数: 冪等✓ 零因子✓ / DFT・IDFT: 準同型+単位元+可逆 ✓ /")
    print("  ランダム行列は準同型でないと検出 ✓ / 畳み込み定理: ΣUVW≡T_cyc・R=64→8 ✓")
    # WH = XOR群のDFT (実±1のみ) — ユーザの想起: 巡回のWinograd(R=12)より良いR=8・厳密
    wh = amap("wh8")
    hw, uw = map_verify(wh, rngm)
    assert hw < 1e-12 and uw < 1e-12
    imw = impl("xor8_wh")
    assert impl_verify(imw, xor_alg(8)) == 0.0 and imw.R == 8   # ±1と/8だけ ⟹ 厳密に0
    Ai = rngm.integers(-100, 101, (200, 8)).astype(np.float64)
    Bi = rngm.integers(-100, 101, (200, 8)).astype(np.float64)
    for i in range(200):
        got = impl_mul(imw, Ai[i], Bi[i])
        assert np.abs(got - rawmul(xor_alg(8), Ai[i], Bi[i])).max() == 0.0   # 整数入力=全段厳密
    print("  WH=XOR群のDFT: 準同型✓ ΣUVW≡T 厳密0 ✓ R=8(±1変換=乗算器0) 整数入力で全段厳密 ✓")
    # 複素WH (Chrestenson-4): {±1,±i} = ガウス単数まで = タダの変換の上限
    cwh = amap("cwh8")
    hc, uc = map_verify(cwh, rngm)
    assert hc < 1e-12 and uc < 1e-12
    imc = impl("z4z2_cwh")
    z42 = tensor(cyclic_alg(4), cyclic_alg(2))
    assert impl_verify(imc, z42) == 0.0
    Ai2 = rngm.integers(-100, 101, (100, 8)).astype(np.float64)
    Bi2 = rngm.integers(-100, 101, (100, 8)).astype(np.float64)
    for i in range(100):
        got = np.real(impl_mul(imc, Ai2[i], Bi2[i]))
        assert np.abs(got - rawmul(z42, Ai2[i], Bi2[i])).max() == 0.0
    print("  複素WH=ℤ/4×ℤ/2のDFT({±1,±i}=×iはスワップ=タダ): 準同型✓ ΣUVW厳密0✓ 整数厳密✓")
    # ★実ランク10の明示的(U,V,W) — 「8複素積では10実乗算の実装でない」(外部監査)への回答
    im10 = impl("z4z2_rank10")
    assert im10.R == 10 and impl_verify(im10, z42) < 1e-13
    for i in range(100):
        got = impl_mul(im10, Ai2[i], Bi2[i])
        assert np.abs(got - rawmul(z42, Ai2[i], Bi2[i])).max() == 0.0
    print("  実ランク10実装: R=10ちょうど・ΣUVW≡T✓・整数入力厳密✓ (=2n−t, ℝ⁴⊕ℂ²)")
    print("  実ランク階段(位数8, ℤ/8はℝ²⊕ℂ³に訂正): 8 / 10 / 11 (有理係数制限なら ℤ/8=12)")
    # FFT = 変換行列の疎因数分解 — MAPSの住人が factors として保持・検証・高速適用
    for nm in ("wh8", "dft8"):
        mpf = amap(nm)
        h2, _ = map_verify(mpf, rngm)               # 因子積≡M も込みで検証
        assert h2 < 1e-12
        xa = rngm.standard_normal(8)
        assert np.abs(mpf.apply_fast(xa) - mpf(xa)).max() < 1e-12
    nz_dense = 64
    nz_wh = sum(int((np.abs(f) > 1e-12).sum()) for f in amap("wh8").factors)
    print(f"  FFT=疎因数分解: wh8/dft8 の因子積≡M ✓ apply_fast≡密適用 ✓"
          f" (非ゼロ {nz_dense}→{nz_wh}; n=1024では 100万→2万)")
    # totality
    bad = nel(cd_alg(16), [np.nan] + [1e308] * 15)
    assert (bad.flag & SING) and np.isfinite(nexp(cd_alg(16), bad).c).all()
    print("totality: NaN/1e308 input → flagged, exp stays finite ✓")
    print("done — the twin shelves (ALGS × OPS) mirror julia/NestedSeries.jl")


if __name__ == "__main__":
    self_test()
