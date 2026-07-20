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
    # totality
    bad = nel(cd_alg(16), [np.nan] + [1e308] * 15)
    assert (bad.flag & SING) and np.isfinite(nexp(cd_alg(16), bad).c).all()
    print("totality: NaN/1e308 input → flagged, exp stays finite ✓")
    print("done — the twin shelves (ALGS × OPS) mirror julia/NestedSeries.jl")


if __name__ == "__main__":
    self_test()
